import numpy as np
import transforms3d as tfs


class hand_eye_map():
    def __init__(self):
        # 重新导入 RT_camera_to_base 矩阵
        self.RT_camera_to_base = np.loadtxt('camera_pose_0415.txt')

        # print("Re-loaded camera_to_base:\n", self.RT_camera_to_base)

    def get_r(self, rx, ry, rz):
        rmat = tfs.euler.euler2mat(rx, ry, rz)
        # rmat = util.rv2rm(rx, ry, rz)
        # rmat = rmat.T
        return rmat

    def get_t(self, x, y, z):
        t = np.asarray((x, y, z))
        # t= -np.dot(R,t)
        return t
        
    def R_T_to_RT(self,R,T):
        T = T.reshape((3, 1))
        RT = np.hstack([R, T])

        # 创建一个1x4的数组，最后一个元素是1，其他元素是0
        bottom_row = np.array([[0, 0, 0, 1]])

        # 将RT和bottom_row垂直堆叠起来
        RT = np.vstack([RT, bottom_row])
        return RT

    def RT_to_pose(self, RT):
        """
        从齐次变换矩阵RT中提取位置和欧拉角
        :param RT: 4x4的齐次变换矩阵
        :return: 位置向量和欧拉角
        """
        # 提取位置向量
        position = RT[:3, 3]

        # 提取旋转矩阵部分
        R = RT[:3, :3]
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

        return position, euler_angles

    def map(self,x,y,z):
        t_target_in_camera = self.get_t(x,y,z)
        r_target_in_camera = self.get_r(1,2,0)
        p_target_in_camera = self.R_T_to_RT(r_target_in_camera,t_target_in_camera)
        res = self.RT_camera_to_base@p_target_in_camera
        return res[0][3],res[1][3],res[2][3]
    
    def map_rt(self,rotation, translation):
        p_target_in_camera = self.R_T_to_RT(rotation, translation)
        res = self.RT_camera_to_base@p_target_in_camera
        print(res)
        return self.RT_to_pose(res)


if __name__ == '__main__':
    handler = hand_eye_map()
    res = handler.map(0.033707812428474426, 0.09388741105794907, 0.5550000071525574)
    print(res)