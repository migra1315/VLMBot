
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
        # if vis:
        #     self.vis_grasps(gg, cloud)
        gg.nms()
        gg.sort_by_score()

        # ===== 筛选部分:对抓取预测的接近方向进行垂直角度限制=====
        #将 gg 转换为普通列表 
        all_grasps = list(gg)
        vertical = np.array([0,0,1])# 期望抓取接近方向(垂直桌面) 
        angle_threshold = np.deg2rad(30) # 30度的弧度值 
        filtered = []
        for grasp in all_grasps:
            # 抓取的接近方向取grasp.rotation_matrix的第一列 
            approach_dir = grasp.rotation_matrix[:,0]
            # 计算夹角:cos(angle)=dot(approach_dir, vertical) 
            cos_angle = np.dot(approach_dir, vertical) 
            cos_angle = np.clip(cos_angle,-1.0,1.0) 
            angle = np.arccos(cos_angle) 
            if angle < angle_threshold:
                filtered.append(grasp)
        if len(filtered) ==0:
            print("\n[Warning] No grasp predictions within vertical angle threshold. Using all predictions.") 
            filtered = all_grasps 
        else:
            print(f"\n[DEBUG] Filtered {len(filtered)} grasps within ±30° of vertical out of {len(all_grasps)} total predictions.")
        

        #对过滤后的抓取根据score排序(降序)
        filtered.sort(key=lambda g: g.score, reverse=True)
        #取前50个抓取(如果少于50个,则全部使用);此处示例中取前1 
        top_grasps = filtered[:10]

        if vis:
            #可视化过滤后的抓取,手动转换为0pen3D物体
            grippers = [g.to_open3d_geometry() for g in top_grasps]
            print(f"\nVisualizing top {len(top_grasps)} graspsafter filtering...") 
            o3d.visualization.draw_geometries([cloud, *grippers])

        #选择得分最高的抓取(filtered列表已按得分降序排序排序) 
        best_grasp = top_grasps[0]
        best_translation = best_grasp.translation
        best_rotation = best_grasp.rotation_matrix 
        best_width = best_grasp.width *1000
        if best_width >= 100:
            best_width = 100
        # print(best_translation, best_rotation, best_[width)
        # return best_translation, best_rotation, best_width

        # gg = filtered[:1]
        # # 提取旋转矩阵部分
        # R = gg.rotation_matrices.squeeze(0)
        R = best_rotation
    
        # 坐标系转换
        R_adjust = np.array([
            [0,0,1],
            [1,0,0],
            [0,1,0]
        ],dtype=np.float32)

        R = R @ R_adjust

        #计算欧拉角
        sy = np.sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2][1], R[2][2])
            y = np.arctan2(-R[2][0], sy)
            z = np.arctan2(R[1][0], R[0][0])
        else:
            x = np.arctan2(-R[1][2], R[1][1])
            y = np.arctan2(-R[2][0], sy)
            z = 0

        euler_angles = np.asarray([x, y, z])

        return best_translation, euler_angles

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
        intrinsic = [[910.3829956054688, 0, 628.5571899414062], [0, 910.4645385742188, 355.1884460449219], [0, 0, 1]]#meta['intrinsic_matrix']
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
    print(foo.forward(False))


