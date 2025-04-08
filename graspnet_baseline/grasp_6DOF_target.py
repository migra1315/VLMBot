
import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
import torch
from graspnetAPI import GraspGroup
import matplotlib.pyplot as plt
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image


class predicter_6DOF_target():
    def __init__(self,data_dir = '/home/ju/Workspace/VLMBot'):
        self.checkpoint_path = '/home/ju/Workspace/VLMBot/graspnet_baseline/logs/logs_rs/checkpoint-rs.tar'
        self.num_point = 2000
        self.num_view = 300
        self.collision_thresh = 0.01
        self.voxel_size = 0.01
        self.net = self.get_net()
        self.data_dir = data_dir

    def forward(self,vis = False):
        end_points, cloud = self.get_and_process_data(self.data_dir)
        gg = self.get_grasps(self.net, end_points)
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
        if vis:
            self.vis_grasps(gg, cloud)
        gg.nms()
        gg.sort_by_score()
        gg = gg[:1]
        print(gg.rotation_matrices.shape)
        return gg.rotation_matrices.squeeze(0), gg.translations

    def get_net(self):
        # Init the model
        net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)"%(self.checkpoint_path, start_epoch))
        # set model to eval mode
        net.eval()
        return net

    def get_and_process_data(self, data_dir):
        # load data
        color = np.array(Image.open(os.path.join(data_dir, 'outputs/color.png')), dtype=np.float32) / 255.0
        depth = np.array(Image.open(os.path.join(data_dir, 'outputs/depth.png')))
        workspace_mask_uint8 = np.array(Image.open(os.path.join(data_dir, 'outputs/workspace_mask.png')))

        workspace_mask = (workspace_mask_uint8 == 255)
        # meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
        intrinsic = [[912.5797119140625, 0, 638.57470703125], [0, 912.6864624023438, 349.6040344238281], [0, 0, 1]]#meta['intrinsic_matrix']
        factor_depth = 0.0010000000474974513*1000*1000 #meta['factor_depth']

        # generate cloud
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        mask = (workspace_mask & (depth > 0))
        # plt.imshow(depth)
        # plt.show()
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        # sample points
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud

    def get_grasps(self, net, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]
        return gg

    def vis_grasps(self, gg, cloud, num_top_grasps=50):
        gg.nms()
        gg.sort_by_score()
        gg = gg[:1]
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])


    
if __name__=='__main__':
    foo = predicter_6DOF_target()
    print(foo.forward(True))


