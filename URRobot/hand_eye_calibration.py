#!/usr/bin/env python
# coding: utf-8
import cv2
import transforms3d as tfs
import numpy as np
import math
import os
import yaml
from sympy.codegen import Print

import utils.UR_Robot as rv2rm
import numpy as np

np.set_printoptions(suppress=True, precision=10)

def get_matrix_eular_radu_hand(x, y, z, rx, ry, rz):
    # rmat = tfs.euler.euler2mat(math.radians(rx),math.radians(ry),math.radians(rz))
    rmat = tfs.euler.euler2mat(rx, ry, rz)
    # rmat = rmat.T
    p = np.asarray((x, y, z))
    # p=-np.dot(rmat, p)
    rmat = tfs.affines.compose(np.squeeze(p), rmat, [1, 1, 1])
    return rmat


def get_matrix_eular_radu(x, y, z, rx, ry, rz):
    # rmat = tfs.euler.euler2mat(math.radians(rx),math.radians(ry),math.radians(rz))
    rmat = tfs.euler.euler2mat(rx, -ry, rz)
    # rmat = rmat.T
    p = np.asarray((x, y, z))
    # p=-np.dot(rmat, p)
    rmat = tfs.affines.compose(np.squeeze(p), rmat, [1, 1, 1])
    return rmat


def get_rv_R(rx, ry, rz):
    # rmat = tfs.euler.euler2mat(rx, ry, rz)
    rmat = rv2rm(rx, ry, rz)
    # rmat = rmat.T
    return rmat


def get_r(rx, ry, rz):
    rmat = tfs.euler.euler2mat(rx, ry, rz)
    # rmat = util.rv2rm(rx, ry, rz)
    # rmat = rmat.T
    return rmat


def get_t(x, y, z):
    t = np.asarray((x, y, z))
    # t= -np.dot(R,t)
    return t

def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def rot2quat_minimal(m):
    quat = tfs.quaternions.mat2quat(m[0:3, 0:3])
    return quat[1:]

def quatMinimal2rot(q):
    p = np.dot(q.T, q)
    w = np.sqrt(np.subtract(1, p[0][0]))
    return tfs.quaternions.quat2mat([w, q[0], q[1], q[2]])

def R_T_to_RT(R,T):
    T = T.reshape((3, 1))
    RT = np.hstack([R, T])

    # 创建一个1x4的数组，最后一个元素是1，其他元素是0
    bottom_row = np.array([[0, 0, 0, 1]])

    # 将RT和bottom_row垂直堆叠起来
    RT = np.vstack([RT, bottom_row])
    return RT

def RT_to_R_T(RT):
    R = RT[0:3, 0:3]
    T = RT[0:3, 3]
    return R,T

def e_h_H(hand, camera):
    Hgs, Hcs = [], []
    for i in range(0, len(hand), 6):
        # for i in range(0, 10*6, 6):
        Hgs.append(get_matrix_eular_radu_hand(hand[i], hand[i + 1], hand[i + 2], hand[i + 3], hand[i + 4], hand[i + 5]))
        Hcs.append(
            get_matrix_eular_radu(camera[i], camera[i + 1], camera[i + 2], camera[i + 3], camera[i + 4], camera[i + 5]))

    Hgijs = []
    Hcijs = []
    A = []
    B = []
    size = 0
    for i in range(len(Hgs)):
        for j in range(i + 1, len(Hgs)):
            size += 1
            Hgij = np.dot(np.linalg.inv(Hgs[j]), Hgs[i])
            Hgijs.append(Hgij)
            Pgij = np.dot(2, rot2quat_minimal(Hgij))

            Hcij = np.dot(Hcs[j], np.linalg.inv(Hcs[i]))
            Hcijs.append(Hcij)
            Pcij = np.dot(2, rot2quat_minimal(Hcij))

            A.append(skew(np.add(Pgij, Pcij)))
            B.append(np.subtract(Pcij, Pgij))
    MA = np.asarray(A).reshape(size * 3, 3)
    MB = np.asarray(B).reshape(size * 3, 1)
    Pcg_ = np.dot(np.linalg.pinv(MA), MB)
    pcg_norm = np.dot(np.conjugate(Pcg_).T, Pcg_)
    Pcg = np.sqrt(np.add(1, np.dot(Pcg_.T, Pcg_)))
    Pcg = np.dot(np.dot(2, Pcg_), np.linalg.inv(Pcg))
    Rcg = quatMinimal2rot(np.divide(Pcg, 2)).reshape(3, 3)

    A = []
    B = []
    id = 0
    for i in range(len(Hgs)):
        for j in range(i + 1, len(Hgs)):
            Hgij = Hgijs[id]
            Hcij = Hcijs[id]
            A.append(np.subtract(Hgij[0:3, 0:3], np.eye(3, 3)))
            B.append(np.subtract(np.dot(Rcg, Hcij[0:3, 3:4]), Hgij[0:3, 3:4]))
            id += 1

    MA = np.asarray(A).reshape(size * 3, 3)
    MB = np.asarray(B).reshape(size * 3, 1)
    Tcg = np.dot(np.linalg.pinv(MA), MB).reshape(3, )
    print(tfs.affines.compose(Tcg, np.squeeze(Rcg), [1, 1, 1]))
    return tfs.affines.compose(Tcg, np.squeeze(Rcg), [1, 1, 1])


def eye_in_hand(end_to_base_arrays, target_to_camera_arrays):
    '''
    眼在手上标定算法
    Args:
        end_to_base_arrays:
        target_to_camera_arrays:

    Returns:

    '''
    R_all_end_to_base, T_all_end_to_base = [], []
    R_all_target_to_cam, T_all_target_to_cam = [], []

    for i in range(0, len(end_to_base_arrays), 6):
        r = get_r(end_to_base_arrays[i + 3], end_to_base_arrays[i + 4], end_to_base_arrays[i + 5])
        t = get_t(end_to_base_arrays[i], end_to_base_arrays[i + 1], end_to_base_arrays[i + 2])
        T_all_end_to_base.append(t)
        R_all_end_to_base.append(r)

        r = get_r(target_to_camera_arrays[i + 3], target_to_camera_arrays[i + 4], target_to_camera_arrays[i + 5])
        t = get_t(target_to_camera_arrays[i], target_to_camera_arrays[i + 1], target_to_camera_arrays[i + 2])
        T_all_target_to_cam.append(t)
        R_all_target_to_cam.append(r)
    """
        * @brief 眼在手上标定，通过
        * T^t_c * T^c_e * T^e_b = T^t_b, 已知 T^e_b 和 T^t_c, 求解T^c_e
        * 输入 end_to_base 与 target_to_camera, 输出 camera_in_end

    """
    R, T = cv2.calibrateHandEye(R_all_end_to_base, T_all_end_to_base, R_all_target_to_cam, T_all_target_to_cam)

    RT_camera_to_end = R_T_to_RT(R,T)
    print("camera_to_end:\n",RT_camera_to_end)
    np.savetxt('./outputs/calibration/camera_pose.txt', RT_camera_to_end, delimiter=' ')

    if 1:
        print('aruco码到机座的相对距离(mm):')
        for index in range(len(R_all_end_to_base)):
            RT_end_to_base = R_T_to_RT(R_all_end_to_base[index],T_all_end_to_base[index])
            RT_target_to_camera = R_T_to_RT(R_all_target_to_cam[index],T_all_target_to_cam[index])

            _, res = RT_to_R_T(RT_target_to_camera@RT_camera_to_end@RT_end_to_base)
            print(res*100)
    
    else:
        t_end_in_base = get_t(-0.17950118630599343, -0.3849716414164546, 0.13166553731077218)
        r_end_in_base = get_r(-2.9196729525138343, -0.11910969251505635, 2.959249079785294)    
        p_end_in_base = R_T_to_RT(r_end_in_base,t_end_in_base)

        RT_camera_in_base = p_end_in_base@RT_camera_to_end

        t_target_in_camera = get_t(-0.00200860109180212, -0.030193030834197998, 0.3240000009536743)
        r_target_in_camera = get_r(1,0,2)
        p_target_in_camera = R_T_to_RT(r_target_in_camera,t_target_in_camera)

        print('p_target_in_camera:\n',p_target_in_camera,'\n')
        print('计算出的 target in base:\n',RT_camera_in_base@p_target_in_camera)



def eye_to_hand(end_to_base_arrays, target_to_camera_arrays):
    '''
    眼在手外标定算法
    Args:
        end_to_base_arrays:
        target_to_camera_arrays:

    Returns:

    '''
    R_all_base_to_end, T_all_base_to_end = [], []
    R_all_target_to_cam, T_all_target_to_cam = [], []

    for i in range(0, len(end_to_base_arrays), 6):
        r = get_r(end_to_base_arrays[i + 3],
                  end_to_base_arrays[i + 4],
                  end_to_base_arrays[i + 5])

        # r = cv2.Rodrigues(np.array([end_to_base_arrays[i + 3], end_to_base_arrays[i + 4], end_to_base_arrays[i + 5]]))[0]
        t = get_t(end_to_base_arrays[i],
                  end_to_base_arrays[i + 1],
                  end_to_base_arrays[i + 2])

        rt_base_to_end = np.linalg.inv(R_T_to_RT(r,t))
        r,t = RT_to_R_T(rt_base_to_end)

        T_all_base_to_end.append(t)
        R_all_base_to_end.append(r)

        r = get_r(target_to_camera_arrays[i + 3], target_to_camera_arrays[i + 4], target_to_camera_arrays[i + 5])
        # r = cv2.Rodrigues(np.array([target_to_camera_arrays[i + 3],
        #                            target_to_camera_arrays[i + 4],
        #                            target_to_camera_arrays[i + 5]]))[0]
        t = get_t(target_to_camera_arrays[i], target_to_camera_arrays[i + 1], target_to_camera_arrays[i + 2])

        T_all_target_to_cam.append(t)
        R_all_target_to_cam.append(r)

    """
        * @brief 用于眼在手外标定场景，通过
        * T^t_c * T^c_b * T^b_e = T^t_e, 已知 T^b_e 和 T^t_c, 求解T^b_c
        * 输入 base_to_end 与 target_to_camera, 输出 camera_in_base
    """
    R, T = cv2.calibrateHandEye(R_all_base_to_end, T_all_base_to_end,
                                R_all_target_to_cam, T_all_target_to_cam,
                                method=cv2.CALIB_HAND_EYE_TSAI)

    print(f"R: \n{R}")
    print(f"T: \n{T}")

    RT_camera_to_base = R_T_to_RT(R,T)
    print("camera_to_base:\n",RT_camera_to_base)
    np.savetxt('./outputs/calibration/camera_pose.txt', RT_camera_to_base, delimiter=' ')

    if 1:
        t_target_in_camera = get_t(-0.06890208274126053, 0.13955539464950562, 0.718000054359436)
        r_target_in_camera = get_r(1.0686568,-0.3243425,1.0970761)
        p_target_in_camera = R_T_to_RT(r_target_in_camera,t_target_in_camera)
        print('p_target_in_camera:\n',p_target_in_camera,'\n')
        print('res:\n',RT_camera_to_base@p_target_in_camera)
    else:
        print('aruco码到夹爪的相对距离: \n')
        for index in range(len(R_all_base_to_end)):
            RT_base_to_end = R_T_to_RT(R_all_base_to_end[index],T_all_base_to_end[index])
            RT_target_to_camera = R_T_to_RT(R_all_target_to_cam[index],T_all_target_to_cam[index])
            """
            此处应为左乘, 即p_target_to_end = p_base_to_end*p_camera_to_base*p_target_to_camera
            所以 相机中一点映射到机械臂基座坐标系应为: p_target_to_base = T_camera_to_base *p_target_to_camera
            """
            res_R, res = RT_to_R_T(RT_base_to_end@RT_camera_to_base@RT_target_to_camera)
            print(res*100, '\n')

    # return RT_camera_to_base

    # RT_end_to_base_2 = np.linalg.inv(R_T_to_RT(R_all_base_to_end[1],T_all_base_to_end[1]))
    # RT_camera_to_target_2 = np.linalg.inv(R_T_to_RT(R_all_target_to_cam[1],T_all_target_to_cam[1]))
    # print(np.dot(np.dot(RT_end_to_base_2, RT_base_to_camera),RT_camera_to_target_2))

    # print('R=', R)
    # print('T=', T)


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=10)

    # eye_out_hand(ur, ndi)

    # 初始化两个空的数组
    gripper_in_base_arrays = np.array([])
    target_in_camera_arrays = np.array([])

    aruco_file_path = './outputs/calibration/aruco_data.yaml'
    aruco_file_path = os.path.expanduser(aruco_file_path)
    with open(aruco_file_path) as f:
        print("open file: ", aruco_file_path)
        loaded_aruco_data = yaml.load(f, Loader=yaml.FullLoader)

    for id, arrays in loaded_aruco_data.items():
        target_in_camera_array = np.array(arrays['target_in_camera_array'])
        target_in_camera_arrays = np.concatenate((target_in_camera_arrays, target_in_camera_array)) if target_in_camera_arrays.size else target_in_camera_array

    
    gripper_file_path = './outputs/calibration/arm_data.yaml'
    gripper_file_path = os.path.expanduser(gripper_file_path)
    with open(gripper_file_path) as f:
        loaded_aruco_data = yaml.load(f, Loader=yaml.FullLoader)

    for id, arrays in loaded_aruco_data.items():
        gripper_in_base_array = np.array(arrays['end_in_base_array'])
        gripper_in_base_arrays = np.concatenate((gripper_in_base_arrays, gripper_in_base_array)) if gripper_in_base_arrays.size else gripper_in_base_array

    # eye_out_hand(gripper_in_base_arrays, target_in_camera_arrays)
    eye_in_hand(gripper_in_base_arrays, target_in_camera_arrays)

