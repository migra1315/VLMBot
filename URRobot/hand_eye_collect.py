import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import os
from utils.UR_Robot import UR_Robot,rv2rpy 
import yaml
from scipy.spatial.transform import Rotation as R
font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)

# 配置摄像头与开启pipeline
pipeline = rs.pipeline()
config = rs.config()
# config.enable_device('040322073416')
config.enable_device('040322071066')

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)  #配置depth流
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)   #配置color流
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

# 获取对齐的rgb和深度图
def get_aligned_images():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    # 获取intelrealsense参数
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    # 内参矩阵，转ndarray方便后续opencv直接使用
    intr_matrix = np.array([
        [intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]
    ])
    # 深度图-16位
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    # 深度图-8位
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
    pos = np.where(depth_image_8bit == 0)
    depth_image_8bit[pos] = 255
    # rgb图
    color_image = np.asanyarray(color_frame.get_data())
    # return: rgb图，深度图，相机内参，相机畸变系数(intr.coeffs)
    return color_image, depth_image, intr_matrix, np.array(intr.coeffs)

# 假设 collect_arm_data 是一个包含 numpy 数据的字典
# 将 numpy 数据转换为 Python 原生数据类型
def convert_to_native(data):
    if isinstance(data, np.ndarray):
        return data.tolist()  # 将 numpy 数组转换为 Python 列表
    elif isinstance(data, np.float64):
        return float(data)  # 将 numpy.float64 转换为 Python float
    elif isinstance(data, dict):
        return {key: convert_to_native(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_native(item) for item in data]
    else:
        return data

def detect_aruco(index = 0):
    rgb, depth, intr_matrix, intr_coeffs = get_aligned_images()
    # 获取dictionary, 4x4的码，指示位50个
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_100)
    # 创建detector parameters
    parameters = aruco.DetectorParameters_create()
    # 输入rgb图, aruco的dictionary, 相机内参, 相机的畸变参数
    corners, ids, rejected_img_points = aruco.detectMarkers(rgb, aruco_dict, parameters=parameters,cameraMatrix=intr_matrix, distCoeff=intr_coeffs)
    if ids is None:
        return
    # 使用 enumerate 获取每个元素的索引和值，然后根据值进行排序
    sorted_ids = [index for value, index in sorted((value, index) for index, value in enumerate(ids))]
    # 估计出aruco码的位姿，0.045对应markerLength参数，单位是meter
    # rvec是旋转向量， tvec是平移向量
    rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.10, intr_matrix, intr_coeffs)
    rotation = R.from_rotvec(rvec[sorted_ids[index], 0, :])  # 将 rvec 的形状转换为 (3,)
    euler_angles = rotation.as_euler('xyz', degrees=False)
    target_positions = np.concatenate((tvec[sorted_ids[0], :, :].squeeze().squeeze(), euler_angles))
    return target_positions


if __name__ == "__main__":
    ur_robot = UR_Robot()
    n = 0
    collect_aruco_data = {}
    collect_arm_data = {}
    collect_index = 0
    while 1:
        rgb, depth, intr_matrix, intr_coeffs = get_aligned_images()
        # 获取dictionary, 4x4的码，指示位50个
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_100)
        # 创建detector parameters
        parameters = aruco.DetectorParameters_create()
        # 输入rgb图, aruco的dictionary, 相机内参, 相机的畸变参数
        corners, ids, rejected_img_points = aruco.detectMarkers(rgb, aruco_dict, parameters=parameters,cameraMatrix=intr_matrix, distCoeff=intr_coeffs)
        # if ids is None:
        #     continue
        # 使用 enumerate 获取每个元素的索引和值，然后根据值进行排序
        # 估计出aruco码的位姿，0.045对应markerLength参数，单位是meter
        # rvec是旋转向量， tvec是平移向量
        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.05, intr_matrix, intr_coeffs)


        try:
            sorted_ids = [index for value, index in sorted((value, index) for index, value in enumerate(ids))]
            for index in range(ids.shape[0]):
                # 将 rvec 转换为旋转矩阵
                rotation = R.from_rotvec(rvec[sorted_ids[index], 0, :])  # 将 rvec 的形状转换为 (3,)
                euler_angles = rotation.as_euler('xyz', degrees=False)
                # 在图片上标出aruco码的位置
                aruco.drawDetectedMarkers(rgb, corners)
                # 根据aruco码的位姿标注出对应的xyz轴, 0.05对应length参数，代表xyz轴画出来的长度 
                aruco.drawAxis(rgb, intr_matrix, intr_coeffs, rvec[sorted_ids[index]], tvec[sorted_ids[index]], 0.0125)

                # 显示ID，rvec, tvec, 欧拉角 (弧度制)
                cv2.putText(rgb, "Id:" + str(ids[sorted_ids[index]]), (0, 40 + 0 * 20 + index*80), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(rgb, "Euler: " + str(euler_angles), (0, 60 + 0 * 20 + index*80), font, 0.5, (0, 255, 0), 2,
                            cv2.LINE_AA)
                cv2.putText(rgb, "transform: " + str(tvec[sorted_ids[index], :, :]), (0, 80 + 0 * 20 + index*80), font, 0.5, (0, 0, 255), 2,
                            cv2.LINE_AA)
            cv2.imshow('RGB image', rgb)
        except:
            cv2.imshow('RGB image', rgb)
        key = cv2.waitKey(1)
        # 按ESC保存采集结果
        if key == 27:
            # 文件路径
            aruco_file_path = './outputs/calibration/aruco_data.yaml'
            arm_file_path = './outputs/calibration/arm_data.yaml'

            # 获取用户的主目录并拼接到文件路径
            aruco_file_path = os.path.expanduser(aruco_file_path)
            arm_file_path = os.path.expanduser(arm_file_path)

            # 确保目录存在
            dir_path = os.path.dirname(aruco_file_path)
            os.makedirs(dir_path, exist_ok=True)

            dir_path = os.path.dirname(arm_file_path)
            os.makedirs(dir_path, exist_ok=True)

            with open(aruco_file_path, 'a') as outfile:
                yaml.dump(collect_aruco_data, outfile)
            
            with open(arm_file_path, 'a') as outfile:
                yaml.dump(convert_to_native(collect_arm_data), outfile)

            pipeline.stop()
            print("save aruco in camera file ...")
            break
        # 按空格打印当前位置，保存图片
        elif key == 32:
            detect_index=0

            index_dict = {value[0]: i for i, value in enumerate(ids)}

            print(index_dict)
            print(index_dict[detect_index])
            n = n + 1
            # 保存rgb图
            # print(tvec[0,:,:],euler_angles)
            rotation = R.from_rotvec(rvec[index_dict[detect_index], 0, :])  # 将 rvec 的形状转换为 (3,)
            euler_angles = rotation.as_euler('xyz', degrees=False)
            target_positions = np.concatenate((tvec[index_dict[detect_index], :, :].squeeze().squeeze(), euler_angles))

            end_positions = ur_robot.get_state()
            collect_aruco_data[collect_index] = {"target_in_camera_array": target_positions.tolist()}
            collect_arm_data[collect_index] = {"end_in_base_array": end_positions}

            print("collect_index:", collect_index)
            print("target_in_camera:", target_positions)
            print("end_in_base:", end_positions)
            cv2.imwrite(f"outputs/calibration/{collect_index}.png",rgb)
            collect_index+=1

    cv2.destroyAllWindows()

