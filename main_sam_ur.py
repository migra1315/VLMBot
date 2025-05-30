from realsense import realsense_camera
from grounded_sam import groundedSAM
import time
# from openai import OpenAI
from graspnet_baseline.grasp_6DOF_target import predicter_6DOF_target
from URRobot.utils.chingtek import colorGripper
from URRobot.utils.UR_Robot import UR_Robot, eye_in_hand_map

SLEEP_TIME = 2

# client = OpenAI(
#     api_key = "sk-BYrhQry5cLABaIUl92DYnSgKzksGvE8PRzSFnXgWCyhsvaJN",
#     base_url = "https://api.moonshot.cn/v1",
# )
# api_description = """
# 你是一个机器人,你拥有的技能API如下:
# 1. x, y, z = foo.forward(TEXT_PROMPT): 输入类别文本，返回检测候选抓取的坐标
# 2. foo.move_and_pick(pick_robot_x, pick_robot_y, pick_robot_z): 输入候选抓取的坐标，然后执行抓取
# 3. foo.move_and_place(place_robot_x, place_robot_y, place_robot_z): 输入放置抓取的坐标，然后执行放置
# 现在需要你根据你所拥有的技能API,编写python代码完成给你的任务,只输出plan函数,不要输出其他代码以外的内容。注意foo.forward(TEXT_PROMPT)的输入应为英文
# 举例：
# 输入：你的任务是把笔放到盒子里
# 输出：
# def plan():
#     # 检测抓取物品 笔 的坐标
#     TEXT_PROMPT="pen"
#     pick_robot_x, pick_robot_y, pick_robot_z = foo.forward(TEXT_PROMPT)
#     print(f"{TEXT_PROMPT}: ",pick_robot_x, pick_robot_y, pick_robot_z )
#     if pick_robot_x==0 and pick_robot_y==0 and pick_robot_z == 0:
#         print("unfind pick object")
#         return

#     # 检测放置位置 盒子 的坐标
#     TEXT_PROMPT="box"
#     place_robot_x, place_robot_y, place_robot_z = foo.forward(TEXT_PROMPT)
#     print(f"{TEXT_PROMPT}: ",place_robot_x, place_robot_y, place_robot_z )
#     if place_robot_x==0 and place_robot_y==0 and place_robot_z == 0:
#         print("unfind place object")
#         return
    
#     # 抓取物品
#     foo.move_and_pick(pick_robot_x, pick_robot_y, pick_robot_z)
    
#     # 放置物品
#     foo.move_and_place(place_robot_x, place_robot_y, place_robot_z)
# """
# task_description = "把黑色胶带放到红色胶带上"


class VLMBot_sam():
    def __init__(self):
        self.ur_robot = UR_Robot()
        self.mapper = eye_in_hand_map(camera_pose_path = 'camera_pose_0507.txt')
        self.gripper = colorGripper()
        # self.gripper.connect(PORT='/dev/ttyUSB0')
        # self.gripper.Activate()

    def make_model(self):
        self.detect_handler = groundedSAM()
        self.camera_handler = realsense_camera()
        self.graspnet = predicter_6DOF_target()

    def forward(self,TEXT_PROMPT="mouse"):
        self.reset()
        # self.gripper_control(open=False)
        # 采集图像
        self.camera_handler.get_image()

        # 生成mask
        self.detect_handler.forward(TEXT_PROMPT)
        
        #生成抓取姿态
        translation, rotation = self.graspnet.forward(vis=True)

        # res_position,res_euler = self.hand_eye_mapper.map_rt(rotation,translation)
        return translation, rotation 

    
    def reset(self):
        self.ur_robot.go_home()
    

    def execute(self, res_position, res_euler):
        self.ur_robot.go_home()
        position, euler_angles  = self.mapper.map(
            target_in_camera=[res_position[0],res_position[1],res_position[2],res_euler[0],res_euler[1],res_euler[2]])
        
        self.gripper_control(open=True)

        print("目标点位：", position ,euler_angles)
        z_payoff = 0.03
        max_z = -0.085
        self.ur_robot.move_j_p([position[0], position[1],position[2]+0.1 ,3.14, -0.0, euler_angles[2]])

        self.ur_robot.move_j_p([position[0], position[1],position[2]-z_payoff if position[2]>(max_z+z_payoff) else max_z ,3.14, -0.0, euler_angles[2]])
        self.gripper_control(open=False)
        self.ur_robot.go_place()
        # self.ur_robot.move_j_p([position[0], position[1],position[2]+0.1 ,3.14, -0.0, euler_angles[2]])

        # self.ur_robot.go_place()
        self.gripper_control(open=True)
    
    def gripper_control(self,open):
        if open:
           self.gripper.open()
        else:
           self.gripper.close()
        time.sleep(2)
    
    def pick_and_place(self,pick,place):
        TEXT_PROMPT = pick
        res_position,res_euler  = foo.forward(TEXT_PROMPT)
        foo.execute(res_position, res_euler)
    




if __name__ == '__main__':


    foo = VLMBot_sam()
    foo.make_model()

    TEXT_PROMPT="orange bottle"
    res_position,res_euler  = foo.forward(TEXT_PROMPT)
    foo.execute(res_position, res_euler)

    TEXT_PROMPT="blue bottle"
    res_position,res_euler  = foo.forward(TEXT_PROMPT)
    print(TEXT_PROMPT,res_position, res_euler )
    foo.execute(res_position,res_euler)

 

