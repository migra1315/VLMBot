import numpy as np
import transforms3d as tfs


class hand_eye_map():
    def __init__(self):
        # 重新导入 RT_camera_to_base 矩阵
        self.RT_camera_to_base = np.loadtxt('camera_pose.txt')

        print("Re-loaded camera_to_base:\n", self.RT_camera_to_base)

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


    def map(self,x,y,z):
        t_target_in_camera = self.get_t(x,y,z)
        r_target_in_camera = self.get_r(1,2,0)
        p_target_in_camera = self.R_T_to_RT(r_target_in_camera,t_target_in_camera)
        res = self.RT_camera_to_base@p_target_in_camera
        return res[0][3],res[1][3],res[2][3]


if __name__ == '__main__':
    handler = hand_eye_map()
    res = handler.map(0.033707812428474426, 0.09388741105794907, 0.5550000071525574)
    print(res)