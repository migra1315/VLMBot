from realsense import realsense_camera
from grounding_dino import groudingDINO
from hand_eye_map import hand_eye_map
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Empty, Float32MultiArray
import time
from openai import OpenAI

SLEEP_TIME = 3

client = OpenAI(
    api_key = "sk-BYrhQry5cLABaIUl92DYnSgKzksGvE8PRzSFnXgWCyhsvaJN",
    base_url = "https://api.moonshot.cn/v1",
)
api_description = """
你是一个机器人,你拥有的技能API如下:
1. x, y, z = foo.forward(TEXT_PROMPT): 输入类别文本，返回检测候选抓取的坐标
2. foo.move_and_pick(pick_robot_x, pick_robot_y, pick_robot_z): 输入候选抓取的坐标，然后执行抓取
3. foo.move_and_place(place_robot_x, place_robot_y, place_robot_z): 输入放置抓取的坐标，然后执行放置
现在需要你根据你所拥有的技能API,编写python代码完成给你的任务,只输出plan函数,不要输出其他代码以外的内容。注意foo.forward(TEXT_PROMPT)的输入应为英文
举例：
输入：你的任务是把笔放到盒子里
输出：
def plan():
    # 检测抓取物品 笔 的坐标
    TEXT_PROMPT="pen"
    pick_robot_x, pick_robot_y, pick_robot_z = foo.forward(TEXT_PROMPT)
    print(f"{TEXT_PROMPT}: ",pick_robot_x, pick_robot_y, pick_robot_z )
    if pick_robot_x==0 and pick_robot_y==0 and pick_robot_z == 0:
        print("unfind pick object")
        return

    # 检测放置位置 盒子 的坐标
    TEXT_PROMPT="box"
    place_robot_x, place_robot_y, place_robot_z = foo.forward(TEXT_PROMPT)
    print(f"{TEXT_PROMPT}: ",place_robot_x, place_robot_y, place_robot_z )
    if place_robot_x==0 and place_robot_y==0 and place_robot_z == 0:
        print("unfind place object")
        return
    
    # 抓取物品
    foo.move_and_pick(pick_robot_x, pick_robot_y, pick_robot_z)
    
    # 放置物品
    foo.move_and_place(place_robot_x, place_robot_y, place_robot_z)
"""
task_description = "把黑色胶带放到蓝色盒子里"


class VLMBot(Node):
    def __init__(self):
        Node.__init__(self, "VLMBot")
        self.gripper_controller = self.create_publisher(Float32, 'chingtek_position_controller', 10)
        self.reset_trigger = self.create_publisher(Empty, '/robot_reset_trigger', 10)
        self.cartesian_move_controller = self.create_publisher(Float32MultiArray, '/robot_cartesian_move', 10)

        self.detect_handler = groudingDINO()
        self.camera_handler = realsense_camera()
        self.hand_eye_mapper = hand_eye_map()


    def forward(self,TEXT_PROMPT="mouse"):
        self.camera_handler.get_image()
        self.detect_handler.load_image("detect_img.png")
        pixel_x, pixel_y = self.detect_handler.forward(TEXT_PROMPT)
        if pixel_x==0 and pixel_y == 0 :
            return 0,0,0 
        print(f"{TEXT_PROMPT} locate pixel index x {pixel_x} y {pixel_y}")

        camera_x,camera_y,camera_z = self.camera_handler.location(pixel_x,pixel_y)
        if camera_x==0 and camera_y == 0 and camera_z == 0 :
            return 0,0,0 
        print(f"{TEXT_PROMPT} locate camera index x {camera_x} y {camera_y} z {camera_z}")

        robot_x, robot_y, robot_z = self.hand_eye_mapper.map(camera_x,camera_y,camera_z)
        if robot_x==0 and robot_y == 0 and robot_z == 0 :
            return 0,0,0 
        print(f"{TEXT_PROMPT} transformed robot index x {robot_x} y {robot_y} z {robot_z}")
        return robot_x, robot_y, robot_z
    
    def reset(self):
        msg = Empty()
        self.reset_trigger.publish(msg)
        time.sleep(SLEEP_TIME)
        self.get_logger().info('Published to /robot_reset_trigger')

    def cartesian_move(self, x, y, z):
        if x==0 and y==0 and z == 0:
            self.get_logger().info(f'invalid input ...')
            return
        z = 0.005 if z < 0.005 else z
        msg = Float32MultiArray()
        msg.data = [x, y, z]
        self.cartesian_move_controller.publish(msg)
        time.sleep(SLEEP_TIME)
        self.get_logger().info(f'Published to /robot_cartesian_move: {msg.data}')
    
    def gripper_control(self,open):
        msg = Float32()
        if open :
            msg.data = -1.0
        else:
            msg.data = 0.0
        for i in range(4):
            self.gripper_controller.publish(msg)
            time.sleep(0.5)
        self.get_logger().info(f'Published to chingtek_position_controller: {msg.data}')
    
    def move_and_pick(self,pick_robot_x, pick_robot_y, pick_robot_z):
        pick_prepare_z = 0.2 if pick_robot_z<0.1 else pick_robot_z + 0.05
        self.cartesian_move(pick_robot_x, pick_robot_y, pick_prepare_z)

        self.cartesian_move(pick_robot_x, pick_robot_y, pick_robot_z)

        self.gripper_control(open=False)

        self.cartesian_move(pick_robot_x, pick_robot_y, 0.1)
    
    def move_and_place(self,place_robot_x, place_robot_y,place_robot_z):

        self.cartesian_move(place_robot_x, place_robot_y, 0.1)

        self.gripper_control(open=True)

        self.reset()


def main_user_define():
    rclpy.init()
    foo = VLMBot()

    try:
        for i in range(3):
            time.sleep(SLEEP_TIME)
            foo.reset()
            foo.gripper_control(open=True)
            foo.gripper_control(open=False)
            foo.gripper_control(open=True)

            TEXT_PROMPT="pen"
            pick_robot_x, pick_robot_y, pick_robot_z = foo.forward(TEXT_PROMPT)
            print(TEXT_PROMPT,": ",pick_robot_x, pick_robot_y, pick_robot_z )
            if pick_robot_x==0 and pick_robot_y==0 and pick_robot_z == 0:
                pass

            TEXT_PROMPT="box"
            place_robot_x, place_robot_y, place_robot_z = foo.forward(TEXT_PROMPT)
            print(TEXT_PROMPT,": ",place_robot_x, place_robot_y, place_robot_z )
            if place_robot_x==0 and place_robot_y==0 and place_robot_z == 0:
                pass
            
            foo.move_and_pick(pick_robot_x, pick_robot_y, pick_robot_z)

            foo.move_and_place(place_robot_x, place_robot_y, place_robot_z)



            # foo.reset()

    finally:
        # 确保资源正确释放
        foo.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':

    rclpy.init()
    foo = VLMBot()
    try:
        time.sleep(SLEEP_TIME)
        foo.reset()
        foo.gripper_control(open=True)
        foo.gripper_control(open=False)
        foo.gripper_control(open=True)

        completion = client.chat.completions.create(
        model = "moonshot-v1-8k",
        messages = [
            {"role": "system", "content": api_description},
            
            {"role": "user", "content": f"你的任务是{task_description}"}
        ],
        temperature = 0.3,)

        plan_code = completion.choices[0].message.content
        print(plan_code)
        exec(plan_code)
        plan()

    finally:
        # 确保资源正确释放
        foo.destroy_node()
        rclpy.shutdown()

    # rclpy.init()
    # foo = VLMBot()
    # TEXT_PROMPT="adhesive tape"
    # pick_robot_x, pick_robot_y, pick_robot_z = foo.forward(TEXT_PROMPT)
    # print(TEXT_PROMPT,": ",pick_robot_x, pick_robot_y, pick_robot_z )












