from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import cv2
import warnings
from skimage import io

# 屏蔽 FutureWarning 和 UserWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class groudingDINO():
    def __init__(self):
        CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"
        self.DEVICE = "cuda"
        self.TEXT_PROMPT = "mouse"
        # TEXT_PROMPT = "Horse. Clouds. Grasses. Sky. Hill."
        self.BOX_TRESHOLD = 0.35
        self.TEXT_TRESHOLD = 0.5
        self.FP16_INFERENCE = True
        self.model = load_model(CONFIG_PATH, CHECKPOINT_PATH)

    def load_image(self, IMAGE_PATH="outputs/color.png"):
        self.image_source, self.image = load_image(IMAGE_PATH)
        if self.FP16_INFERENCE:
            self.image = self.image.half()
            self.model = self.model.half()

    def forward(self,TEXT_PROMPT="mouse"):
        boxes, logits, phrases = predict(
            model= self.model,
            image= self.image,
            caption=TEXT_PROMPT,
            box_threshold=self.BOX_TRESHOLD,
            text_threshold=self.TEXT_TRESHOLD,
            device=self.DEVICE,
        )
        annotated_frame = annotate(image_source=self.image_source, boxes=boxes, logits=logits, phrases=phrases)
        if boxes.size()[0]<1:
            return 0, 0
        cv2.imwrite(f"outputs/DINO/annotated_image_{TEXT_PROMPT}.png", annotated_frame)
        return int((boxes[0][0]*self.image_source.shape[1]).item()), int((boxes[0][1]*self.image_source.shape[0]).item())

if __name__ == '__main__':
    handler = groudingDINO()
    handler.load_image()
    x,y = handler.forward()
    print(x,y)


