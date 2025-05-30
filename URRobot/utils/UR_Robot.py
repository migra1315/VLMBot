#coding=utf8
import time
import copy
import socket
import struct
import numpy as np

import math
import transforms3d as tfs


PI = 3.1415926

# 旋转矢量转旋转矩阵
def rv2rm(rx, ry, rz):
    theta = np.linalg.norm([rx, ry, rz])
    kx = rx / theta
    ky = ry / theta
    kz = rz / theta

    c = np.cos(theta)
    s = np.sin(theta)
    v = 1 - c

    R = np.zeros((3, 3))
    R[0][0] = kx * kx * v + c
    R[0][1] = kx * ky * v - kz * s
    R[0][2] = kx * kz * v + ky * s

    R[1][0] = ky * kx * v + kz * s
    R[1][1] = ky * ky * v + c
    R[1][2] = ky * kz * v - kx * s

    R[2][0] = kz * kx * v - ky * s
    R[2][1] = kz * ky * v + kx * s
    R[2][2] = kz * kz * v + c

    return R


# 旋转矩阵转rpy
def rm2rpy(R):
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

    return np.asarray([x, y, z])


# rpy转旋转矩阵
def rpy2rm(rpy):
    # Rx = np.zeros((3, 3), dtype=rpy.dtype)
    # Ry = np.zeros((3, 3), dtype=rpy.dtype)
    # Rz = np.zeros((3, 3), dtype=rpy.dtype)

    R0 = np.zeros((3, 3), dtype=rpy.dtype)

    thetaX = rpy[0]
    thetaY = rpy[1]
    thetaZ = rpy[2]    #1.19806981 1.21851354 1.1783059

#    thetaX = 1.19806981
#    thetaY = 1.21851354
#    thetaZ = 1.1783059



    cx = np.cos(thetaX)
    sx = np.sin(thetaX)

    cy = np.cos(thetaY)
    sy = np.sin(thetaY)

    cz = np.cos(thetaZ)
    sz = np.sin(thetaZ)

    R0[0][0] = cz * cy
    R0[0][1] = cz * sy * sx - sz * cx
    R0[0][2] = cz * sy * cx + sz * sx
    R0[1][0] = sz * cy
    R0[1][1] = sz * sy * sx + cz * cx
    R0[1][2] = sz * sy * cx - cz * sx
    R0[2][0] = -sy
    R0[2][1] = cy * sx
    R0[2][2] = cy * cx
    #print(R0)
    return R0


# 旋转矩阵转旋转矢量
def rm2rv(R):
    theta = np.arccos((R[0][0] + R[1][1] + R[2][2] - 1) / 2)
    K = (1 / (2 * np.sin(theta))) * np.asarray([R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]])
    r = theta * K
    return r


def rv2rpy(rx, ry, rz):
    R = rv2rm(rx, ry, rz)
    rpy = rm2rpy(R)
    return rpy


def rpy2rv(rpy):
    R = rpy2rm(rpy)
    rv = rm2rv(R)
    return rv

class UR_Robot:
    def __init__(self, tcp_host_ip="192.168.123.100", tcp_port=30003, workspace_limits=None,
                 is_use_robotiq85=False, is_use_camera=False):
        # Init varibles
        if workspace_limits is None:
            workspace_limits = [[-0.7, 0.7], [-0.7, 0.7], [0.00, 0.6]]
        self.workspace_limits = workspace_limits
        self.tcp_host_ip = tcp_host_ip
        self.tcp_port = tcp_port

        # UR5 robot configuration
        # Default joint/tool speed configuration
        self.joint_acc = 1.4  # Safe: 1.4   8
        self.joint_vel = 1.05  # Safe: 1.05  3

        # Joint tolerance for blocking calls
        self.joint_tolerance = 0.01

        # Default tool speed configuration
        self.tool_acc = 0.5  # Safe: 0.5
        self.tool_vel = 0.2  # Safe: 0.2

        # Tool pose tolerance for blocking calls
        self.tool_pose_tolerance = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]

        # Default robot home joint configuration (the robot is up to air)
        self.home_joint_config = [(65/ 360.0) * 2 * np.pi, -(83 / 360.0) * 2 * np.pi,
                             (71 / 360.0) * 2 * np.pi, -(74 / 360.0) * 2 * np.pi,
                             -(90/ 360.0) * 2 * np.pi, (341/ 360.0) * 2 * np.pi,]
        
        self.place_joint_config = [(158. / 360.0) * 2 * np.pi, -(70 / 360.0) * 2 * np.pi,
                        (103 / 360.0) * 2 * np.pi, -(118 / 360.0) * 2 * np.pi,
                        -(88/ 360.0) * 2 * np.pi, (344/ 360.0) * 2 * np.pi,]
        
        self.horizen_joint_config = [(92. / 360.0) * 2 * np.pi, -(160 / 360.0) * 2 * np.pi,
                        (102 / 360.0) * 2 * np.pi, -(87 / 360.0) * 2 * np.pi,
                        -(97/ 360.0) * 2 * np.pi, (14/ 360.0) * 2 * np.pi,]

    # Test for robot controlmove_and_wait_for_pos
    def testRobot(self):
        try:
            print("Test for robot...")
            self.move_j([-(0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
                             (0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
                             -(0 / 360.0) * 2 * np.pi, 0.0])
            self.move_j([(57.04 / 360.0) * 2 * np.pi, (-65.26/ 360.0) * 2 * np.pi,
                             (73.52/ 360.0) * 2 * np.pi, (-100.89/ 360.0) * 2 * np.pi,
                             (-86.93/ 360.0) * 2 * np.pi, (-0.29/360)*2*np.pi])
            # self.open_gripper()
            self.move_j([(57.03 / 360.0) * 2 * np.pi, (-56.67 / 360.0) * 2 * np.pi,
                              (88.72 / 360.0) * 2 * np.pi, (-124.68 / 360.0) * 2 * np.pi,
                              (-86.96/ 360.0) * 2 * np.pi, (-0.3/ 360) * 2 * np.pi])
            # self.close_gripper()
            self.move_j([(57.04 / 360.0) * 2 * np.pi, (-65.26 / 360.0) * 2 * np.pi,
                              (73.52 / 360.0) * 2 * np.pi, (-100.89 / 360.0) * 2 * np.pi,
                              (-86.93 / 360.0) * 2 * np.pi, (-0.29 / 360) * 2 * np.pi])
            self.move_j([-(0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
                             (0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
                             -(0 / 360.0) * 2 * np.pi, 0.0])
            self.move_j_p([0.3,0,0.3,np.pi/2,0,0],0.5,0.5)
            # for i in range(10):
            #     self.move_j_p([0.3, 0, 0.3, np.pi, 0, i*0.1], 0.5, 0.5)
            #     time.sleep(1)
            self.move_j_p([0.3, 0, 0.3, -np.pi, 0, 0],0.5,0.5)
            self.move_p([0.3, 0.3, 0.3, -np.pi, 0, 0],0.5,0.5)
            self.move_l([0.2, 0.2, 0.3, -np.pi, 0, 0],0.5,0.5)
            # self.plane_grasp([0.3, 0.3, 0.1])
            # self.plane_push([0.3, 0.3, 0.1])
        except:
            print("Test fail! ")
    
    # joint control
    '''
    input:joint_configuration = joint angle
    '''
    def move_j(self, joint_configuration,k_acc=1,k_vel=1,t=0,r=0):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "movej([%f" % joint_configuration[0]  #"movej([]),a=,v=,\n"
        for joint_idx in range(1,6):
            tcp_command = tcp_command + (",%f" % joint_configuration[joint_idx])
        tcp_command = tcp_command + "],a=%f,v=%f,t=%f,r=%f)\n" % (k_acc*self.joint_acc, k_vel*self.joint_vel,t,r)
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.tcp_socket.recv(1500)
        actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
        while not all([np.abs(actual_joint_positions[j] - joint_configuration[j]) < self.joint_tolerance for j in range(6)]):
            state_data = self.tcp_socket.recv(1500)
            actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
            time.sleep(0.01)
        self.tcp_socket.close()
    # joint control
    '''
    move_j_p(self, tool_configuration,k_acc=1,k_vel=1,t=0,r=0)
    input:tool_configuration=[x y z r p y]
    其中x y z为三个轴的目标位置坐标, 单位为米
    r p y 为欧拉角 单位为弧度 采用ZYX顺序
    '''
    def move_j_p(self, tool_configuration,k_acc=0.5,k_vel=0.5,t=0,r=0):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        print(f"movej_p([{tool_configuration}])")
        # command: movej([joint_configuration],a,v,t,r)\n
        tcp_command = "def process():\n"
        tcp_command +=" array = rpy2rotvec([%f,%f,%f])\n" %(tool_configuration[3],tool_configuration[4],tool_configuration[5])
        tcp_command += "movej(get_inverse_kin(p[%f,%f,%f,array[0],array[1],array[2]]),a=%f,v=%f,t=%f,r=%f)\n" % (tool_configuration[0],
            tool_configuration[1],tool_configuration[2],k_acc * self.joint_acc, k_vel * self.joint_vel,t,r ) # "movej([]),a=,v=,\n"
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.tcp_socket.recv(1500)
        actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
        while not all([np.abs(actual_tool_positions[j] - tool_configuration[j]) < self.tool_pose_tolerance[j] for j in
                       range(3)]):
            state_data = self.tcp_socket.recv(1500)
            # print(f"tool_position_error{actual_tool_positions - tool_configuration}")
            actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
            time.sleep(0.01)
        time.sleep(1.5)
        self.tcp_socket.close()

    # move_l is mean that the robot keep running in a straight line
    def move_l(self, tool_configuration,k_acc=1,k_vel=1,t=0,r=0):
        print(f"movel([{tool_configuration}])")
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        # command: movel([tool_configuration],a,v,t,r)\n
        tcp_command = "def process():\n"
        tcp_command += " array = rpy2rotvec([%f,%f,%f])\n" % (
            tool_configuration[3], tool_configuration[4], tool_configuration[5])
        tcp_command += "movel(p[%f,%f,%f,array[0],array[1],array[2]],a=%f,v=%f,t=%f,r=%f)\n" % (
            tool_configuration[0], tool_configuration[1], tool_configuration[2],
            k_acc * self.joint_acc, k_vel * self.joint_vel,t,r)  # "movel([]),a=,v=,\n"
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.tcp_socket.recv(1500)
        actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
        while not all([np.abs(actual_tool_positions[j] - tool_configuration[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            state_data = self.tcp_socket.recv(1500)
            actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
            time.sleep(0.01)
        time.sleep(1.5)
        self.tcp_socket.close()

    # Usually, We don't use move_c
    # move_c is mean that the robot move circle
    # mode 0: Unconstrained mode. Interpolate orientation from current pose to target pose (pose_to)
    #      1: Fixed mode. Keep orientation constant relative to the tangent of the circular arc (starting from current pose)
    def move_c(self,pose_via,tool_configuration,k_acc=1,k_vel=1,r=0,mode=0):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        print(f"movec([{pose_via},{tool_configuration}])")
        # command: movec([pose_via,tool_configuration],a,v,t,r)\n
        tcp_command = "def process():\n"
        tcp_command += " via_pose = rpy2rotvec([%f,%f,%f])\n" % (
        pose_via[3],pose_via[4] ,pose_via[5] )
        tcp_command += " tool_pose = rpy2rotvec([%f,%f,%f])\n" % (
        tool_configuration[3], tool_configuration[4], tool_configuration[5])
        tcp_command = f" movec([{pose_via[0]},{pose_via[1]},{pose_via[2]},via_pose[0],via_pose[1],via_pose[2]], \
                [{tool_configuration[0]},{tool_configuration[1]},{tool_configuration[2]},tool_pose[0],tool_pose[1],tool_pose[2]], \
                a={k_acc * self.tool_acc},v={k_vel * self.tool_vel},r={r})\n"
        tcp_command += "end\n"

        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.tcp_socket.recv(1500)
        actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
        while not all([np.abs(actual_tool_positions[j] - tool_configuration[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            state_data = self.tcp_socket.recv(1500)
            actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
            time.sleep(0.01)
        self.tcp_socket.close()
        time.sleep(1.5)

    def go_home(self):
        # self.move_j(self.home_joint_config)
        self.move_j(self.horizen_joint_config)

    def go_place(self):
        self.move_j(self.place_joint_config)

    def restartReal(self):
        self.go_home()

    '''
    get robot current state and information
    tool_configuration=[x y z r p y]
    其中x y z为三个轴的目标位置坐标, 单位为米
    r p y 为欧拉角 单位为弧度 采用ZYX顺序
    '''
    def get_state(self):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        state_data = self.tcp_socket.recv(1500)
        self.tcp_socket.close()
        actual_tool_positions = self.parse_tcp_state_data(state_data, 'cartesian_info')
        euler_angle = rv2rpy(actual_tool_positions[3],actual_tool_positions[4],actual_tool_positions[5])
        return [actual_tool_positions[0],actual_tool_positions[1],actual_tool_positions[2],euler_angle[0],euler_angle[1],euler_angle[2]]
    
    # get robot current joint angles and cartesian pose
    def parse_tcp_state_data(self, data, subpasckage):
        dic = {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d',
               'I target': '6d',
               'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
               'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d', 'Tool vector target': '6d',
               'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d', 'Controller Timer': 'd',
               'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d',
               'Tool Accelerometer values': '3d',
               'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd', 'SoftwareOnly': 'd',
               'softwareOnly2': 'd',
               'V main': 'd',
               'V robot': 'd', 'I robot': 'd', 'V actual': '6d', 'Digital outputs': 'd', 'Program state': 'd',
               'Elbow position': 'd', 'Elbow velocity': '3d'}
        ii = range(len(dic))
        for key, i in zip(dic, ii):
            fmtsize = struct.calcsize(dic[key])
            data1, data = data[0:fmtsize], data[fmtsize:]
            fmt = "!" + dic[key]
            dic[key] = dic[key], struct.unpack(fmt, data1)

        if subpasckage == 'joint_data':  # get joint data
            q_actual_tuple = dic["q actual"]
            joint_data= np.array(q_actual_tuple[1])
            return joint_data
        elif subpasckage == 'cartesian_info':
            Tool_vector_actual = dic["Tool vector actual"]  # get x y z rx ry rz
            cartesian_info = np.array(Tool_vector_actual[1])
            return cartesian_info


    def rpy2rotating_vector(self,rpy):
        # rpy to R
        R = self.rpy2R(rpy)
        # R to rotating_vector
        return self.R2rotating_vector(R)

    def rpy2R(self,rpy): # [r,p,y] 单位rad
        rot_x = np.array([[1, 0, 0],
                          [0, math.cos(rpy[0]), -math.sin(rpy[0])],
                          [0, math.sin(rpy[0]), math.cos(rpy[0])]])
        rot_y = np.array([[math.cos(rpy[1]), 0, math.sin(rpy[1])],
                          [0, 1, 0],
                          [-math.sin(rpy[1]), 0, math.cos(rpy[1])]])
        rot_z = np.array([[math.cos(rpy[2]), -math.sin(rpy[2]), 0],
                          [math.sin(rpy[2]), math.cos(rpy[2]), 0],
                          [0, 0, 1]])
        R = np.dot(rot_z, np.dot(rot_y, rot_x))
        return R

    def R2rotating_vector(self,R):
        theta = math.acos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2)
        print(f"theta:{theta}")
        rx = (R[2, 1] - R[1, 2]) / (2 * math.sin(theta))
        ry = (R[0, 2] - R[2, 0]) / (2 * math.sin(theta))
        rz = (R[1, 0] - R[0, 1]) / (2 * math.sin(theta))
        return np.array([rx, ry, rz]) * theta

    def R2rpy(self,R):
    # assert (isRotationMatrix(R))
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])

class eye_in_hand_map():
    def __init__(self,camera_pose_path = 'camera_pose.txt'):
        # 重新导入 RT_camera_to_base 矩阵
        self.RT_camera_to_end = np.loadtxt(camera_pose_path)
        print("Re-loaded RT_camera_to_end ...", )

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
    
    def rotation_matrix_to_quaternion(self, R):
        # 计算四元数的实部 w
        w = 0.5 * np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
        
        # 计算四元数的虚部 x, y, z
        x = 0.5 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * np.sign(R[2, 1] - R[1, 2])
        y = 0.5 * np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2]) * np.sign(R[0, 2] - R[2, 0])
        z = 0.5 * np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2]) * np.sign(R[1, 0] - R[0, 1])
        
        return np.array([x, y, z,w])
    
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
        # print(self.rotation_matrix_to_quaternion(R))

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
    
    def map(self, 
            target_in_camera=[1.1984103e-01, 1.9107612e-04, 3.8799998e-01, 1.5707964,  -1.0159854,   0.60788405],
            end_in_base=[0.20767144621234807, -0.1611459217545891, 0.15221604958602167, 3.112154485720518, 0.03449244537342576, -0.9142481156439901],
            target_in_label = [0,0,0,0,0,0]
            ):
        t_end_in_base = self.get_t(end_in_base[0],end_in_base[1],end_in_base[2])
        r_end_in_base = self.get_r(end_in_base[3],end_in_base[4],end_in_base[5])    
        p_end_in_base = self.R_T_to_RT(r_end_in_base,t_end_in_base)

        RT_camera_in_base = p_end_in_base@self.RT_camera_to_end

        t_target_in_camera = self.get_t(target_in_camera[0],target_in_camera[1],target_in_camera[2])
        r_target_in_camera = self.get_r(target_in_camera[3],target_in_camera[4],target_in_camera[5])
        p_target_in_camera = self.R_T_to_RT(r_target_in_camera,t_target_in_camera)

        p_target_in_base = RT_camera_in_base@p_target_in_camera


        t_target_in_lable = self.get_t(target_in_label[0],target_in_label[1],target_in_label[2])
        r_target_in_lable = self.get_r(target_in_label[3],target_in_label[4],target_in_label[5])    
        p_target_in_lable = self.R_T_to_RT(r_target_in_lable,t_target_in_lable)

        p_target_in_base = p_target_in_base@p_target_in_lable
        print(p_target_in_base)
        return self.RT_to_pose(p_target_in_base)
    
    def multi(self,origin,to_multi):
        t_to_multi = self.get_t(to_multi[0],to_multi[1],to_multi[2])
        r_to_multi = self.get_r(to_multi[3],to_multi[4],to_multi[5])    
        p_to_multi = self.R_T_to_RT(r_to_multi,t_to_multi)

        t_origin = self.get_t(origin[0],origin[1],origin[2])
        r_origin = self.get_r(origin[3],origin[4],origin[5])    
        p_origin = self.R_T_to_RT(r_origin,t_origin)

        res = p_origin@p_to_multi
        print(res)
        return self.RT_to_pose(res)
    

if __name__ =="__main__":
    from chingtek import colorGripper
    ur_robot = UR_Robot() 
    ur_robot.go_home()
    # current_state = ur_robot.get_state()
    print("current pose: ",ur_robot.get_state())

    mapper = eye_in_hand_map(camera_pose_path="outputs/calibration/camera_pose.txt")

    
    # gripper = colorGripper()
    current_state = [0.08249215533854616, -0.1599912283659034, 0.3250502440050066, -2.277094816227065, -0.5004497860142124, -2.8715953442830644]
    label0_in_camera = [0.20387058,  0.1121708,   0.52550805, -2.11700235,  0.0260909,   2.93255117]
    # label0_in_camera = [0.21938735, 0.10891085, 0.52860357, 3.12184253, 0.86459451, 1.38741574]
    
    label3_in_camera = [-0.00332169, -0.10815987,  0.57675711, -2.80350585, -0.0908012,  -0.1230968]

    tool = False
    place = True
    pick = True
    oprate = True

    if (tool and pick): # pick
        print("pick ...")
        # gripper.open()
        position, euler_angles  = mapper.map(
            end_in_base=current_state,
            target_in_camera=label3_in_camera,
            target_in_label=[0.145,0.0,0.015,PI,0,0])
        print(position, euler_angles)
        ur_robot.move_j_p([position[0], position[1],position[2], euler_angles[0], euler_angles[1], euler_angles[2]])

        position, euler_angles  = mapper.map(
            end_in_base=current_state,
            target_in_camera=label3_in_camera,
            target_in_label=[0.145,0.0,-0.02,PI,0,0])
        print(position, euler_angles)
        ur_robot.move_j_p([position[0], position[1],position[2], euler_angles[0], euler_angles[1], euler_angles[2]])
        # gripper.close()

        position, euler_angles  = mapper.map(
            end_in_base=current_state,
            target_in_camera=label3_in_camera,
            target_in_label=[0.145,0.150,0.05,PI,0,0])
        print(position, euler_angles)
        ur_robot.move_j_p([position[0], position[1],position[2], euler_angles[0], euler_angles[1], euler_angles[2]])
        ur_robot.move_j_p([position[0], position[1],position[2], -1.57, euler_angles[1], euler_angles[2]])


    if(oprate):
        position, euler_angles  = mapper.map(
            end_in_base=current_state,
            target_in_camera=label0_in_camera,
            target_in_label=[0,0,0.0,PI,0,0])
        print(position, euler_angles)

        # position, euler_angles  = mapper.map(
        #     end_in_base=current_state,
        #     target_in_camera=label0_in_camera,
        #     target_in_label=[0.149,-0.038,0.0,PI,0,0])

        position, euler_angles = mapper.multi(
            [position[0], position[1],position[2], euler_angles[0], euler_angles[1], euler_angles[2]],
            [0.0,0.0,0.0,0,0,PI])
        
        ur_robot.move_j_p([position[0], position[1],position[2], euler_angles[0], euler_angles[1], euler_angles[2]])
   
        position, euler_angles = mapper.multi(
            [0.086,-0.009,0.1,0,0,0],
            [position[0], position[1],position[2], euler_angles[0], euler_angles[1], euler_angles[2]]
            )
        
        print(position, euler_angles)

        ur_robot.move_j_p([position[0], position[1],position[2], euler_angles[0], euler_angles[1], euler_angles[2]])
        ur_robot.move_j_p([position[0], position[1],position[2]-0.1, euler_angles[0], euler_angles[1], euler_angles[2]])

       
        
    if(tool and place): # place
        # gripper.close()
        position, euler_angles  = mapper.map(
            end_in_base=current_state,
            target_in_camera=label3_in_camera,
            target_in_label=[0.145,0.150,0.010,PI,0,0])
        print(position, euler_angles)
        ur_robot.move_j_p([position[0], position[1],position[2], euler_angles[0], euler_angles[1], euler_angles[2]])

        position, euler_angles  = mapper.map(
            end_in_base=current_state,
            target_in_camera=label3_in_camera,
            target_in_label=[0.145,0.0,-0.02,PI,0,0])
        print(position, euler_angles)
        ur_robot.move_j_p([position[0], position[1],position[2], euler_angles[0], euler_angles[1], euler_angles[2]])
        
        # gripper.open()

        position, euler_angles  = mapper.map(
            end_in_base=current_state,
            target_in_camera=label3_in_camera,
            target_in_label=[0.145,0.0,0.015,PI,0,0])
        print(position, euler_angles)
        ur_robot.move_j_p([position[0], position[1],position[2], euler_angles[0], euler_angles[1], euler_angles[2]])
        # gripper.close()
        
        ur_robot.go_home()