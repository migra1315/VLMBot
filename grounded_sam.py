import argparse
import os
import sys

import numpy as np
import json
import torch
import cv2
from PIL import Image
import warnings

# 屏蔽 FutureWarning 和 UserWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt

class groundedSAM():
    def __init__(self,params={
            "config_file": "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "grounded_checkpoint": "./groundingdino_swint_ogc.pth",
            "sam_version": "vit_h",
            "sam_checkpoint": "sam_vit_h_4b8939.pth",
            "sam_hq_checkpoint": None,
            "use_sam_hq": False,
            "device": "cuda",
            "output_dir": "outputs/SAM",
            "box_threshold": 0.3,
            "text_threshold": 0.25
            }):
        self.params = params
        self.box_threshold = params["box_threshold"]
        self.text_threshold = params["text_threshold"]
        self.device = params["device"]
        self.sam_version = params["sam_version"]
        self.sam_hq_checkpoint = params["sam_hq_checkpoint"]
        self.sam_checkpoint = params["sam_checkpoint"]
        self.output_dir = params["output_dir"]
        # make dir
        os.makedirs(self.output_dir, exist_ok=True)
        # load model
        self.model = self.load_model(params["config_file"], params["grounded_checkpoint"], device=self.device)

        # initialize SAM
        if params["use_sam_hq"]:
            self.predictor = SamPredictor(sam_hq_model_registry[self.sam_version](checkpoint=self.sam_hq_checkpoint).to(self.device))
        else:
            self.predictor = SamPredictor(sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(self.device))

    def forward(self, text_prompt = "red tabe", image_path = "outputs/color.png"):
        print(f"try to locate {text_prompt} ...")
        # load image
        image_pil, image = self.load_image(image_path)

        # run grounding dino model
        boxes_filt, pred_phrases_tmp = self.get_grounding_output(
            self.model, image, text_prompt, self.box_threshold, self.text_threshold, device=self.device)


        boxes_filt = boxes_filt[0].reshape(1, -1)
        pred_phrases = []
        pred_phrases.append(pred_phrases_tmp[0])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        masks, _, _ = self.predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )

        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            self.show_box(box.numpy(), plt.gca(), label)

        plt.axis('off')
        plt.savefig(
            os.path.join(self.output_dir, f"seg_{text_prompt}.png"),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )

        self.save_mask_data(self.output_dir, masks, boxes_filt, pred_phrases,text_prompt)
        


    def load_image(self, image_path):
        # load image
        image_pil = Image.open(image_path).convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image


    def load_model(self,model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model


    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
        return boxes_filt, pred_phrases
    

    def show_mask(self,mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    def show_box(self,box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
        ax.text(x0, y0, label)


    def save_mask_data(self,output_dir, mask_list, box_list, label_list,text_prompt):
        value = 0  # 0 for background

        mask_img = torch.zeros(mask_list.shape[-2:])
        for idx, mask in enumerate(mask_list):
            mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        
        graspnet_workspace_mask = (mask_img / mask_img.max() * 255).type(torch.uint8)
        cv2.imwrite('outputs/workspace_mask.png', graspnet_workspace_mask.numpy())

        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'mask_{text_prompt}.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

        json_data = [{
            'value': value,
            'label': 'background'
        }]
        for label, box in zip(label_list, box_list):
            value += 1
            name, logit = label.split('(')
            logit = logit[:-1] # the last is ')'
            json_data.append({
                'value': value,
                'label': name,
                'logit': float(logit),
                'box': box.numpy().tolist(),
            })
        with open(os.path.join(output_dir, f'mask_{text_prompt}.json'), 'w') as f:
            json.dump(json_data, f)



if __name__ == "__main__":
    # 构建参数词典
    params = {
        "config_file": "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "grounded_checkpoint": "./groundingdino_swint_ogc.pth",
        "sam_version": "vit_h",
        "sam_checkpoint": "sam_vit_h_4b8939.pth",
        "sam_hq_checkpoint": None,
        "use_sam_hq": False,
        "device": "cuda",
        "output_dir": "outputs/SAM",
        "box_threshold": 0.3,
        "text_threshold": 0.25
    }
    foo = groundedSAM(params)
    foo.forward(text_prompt="tabe")