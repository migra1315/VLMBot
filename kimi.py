import os
import base64
import json
from openai import OpenAI

class RobotWrapper():
    def pick_and_place(self,pick_object,place_object):

        pick_robot_x, pick_robot_y, pick_robot_z = self.locate(pick_object)
        if pick_robot_x==0 and pick_robot_y==0 and pick_robot_z == 0:
            print(f"unfind pick object {pick_object}")
            return

        place_robot_x, place_robot_y, place_robot_z = self.locate(place_object)
        if place_robot_x==0 and place_robot_y==0 and place_robot_z == 0:
            print(f"unfind place object {place_object}")
            return
        self.move_and_pick(pick_robot_x, pick_robot_y, pick_robot_z)
        self.move_and_place(place_robot_x, place_robot_y, place_robot_z)

    def describe(self,content):
        print(content)

    def locate(self,text_prompt):
        print(f"locate {text_prompt} success ...")
        return 1,2,3
    
    def move_and_pick(self,x,y,z):
        print(f"pick success ...")

    def move_and_place(self,x,y,z):
        print(f"place success ...")

    

class ChatBotHandler():
    def __init__(self):
        self.client = OpenAI(
            api_key="sk-3kaeuj4oP58oODxlW6Fv0QQYA51220V9BBglUCcSfgk0rvUS",
            base_url="https://api.moonshot.cn/v1",
        )
    
        # 创建字典作为JSON对象
        example_json_data_1 = {
            "can_execute": True,
            "input":"把红色打火机放到黑色盒子里",
            "response": "好的，我先瞅瞅有没有红色打火机，要是找到了，再去看看黑色盒子在哪儿。要是都能找到，我先抓起红色打火机，再把它放进黑色盒子里。",
            "code": """
def plan():
   foo.pick_and_place("red lighter","black box")
            """
        }
        example_json_string_1 = json.dumps(example_json_data_1, ensure_ascii=False, indent=4)

        example_json_data_2 = {
            "can_execute": False,
            "input":"把苹果放到篮子里",
            "response": "好的，我先得琢磨琢磨这事儿。先把苹果和篮子的位置找出来，才能进行下一步。结果呢，我左找找右找找，发现找不到苹果，也找不到篮子。这下完了，要不这样，我再仔细检查一下，要是还是找不到，那看来这个任务我真干不了了",
            "code": """
def plan():
    print("can't execute")
            """
        }
        example_json_string_2 = json.dumps(example_json_data_2, ensure_ascii=False, indent=4)
        
        example_json_data_3 = {
            "can_execute": False,
            "input":"如果太阳西升东落的话，把苹果放到篮子里",
            "response": "好的，我先得琢磨琢磨这事儿。太阳是东升西落的，所以我不会执行把苹果放到篮子里",
            "code": """
def plan():
    print("can't execute")
            """
        }
        example_json_string_3 = json.dumps(example_json_data_3, ensure_ascii=False, indent=4)

        example_json_data_4 = {
            "can_execute": True,
            "input":"告诉我你看到了什么",
            "response": "好的，我看到了一支黑色的笔和一个灰色的移动电源",
            "code": """
def plan():
    foo.describe("我看到了一支黑色的笔和一个灰色的移动电源")
            """
        }
        example_json_string_4 = json.dumps(example_json_data_4, ensure_ascii=False, indent=4)
        
        self.system_promotion = f""" 
【任务指令】
你是我的机械臂助手，机械臂内置了一些函数，请你根据我的指令，以json形式输出要运行的对应函数 plan()和你给我的回复
【技能API】
    foo.pick_and_place(pick_object,place_object):输入英文类别文本，执行抓取和放置动作。
    foo.describe(describe_content): 描述看到的场景。
【回复要求】
上传图片作为机器人此刻看到的场景，先判断任务能否完成。以 json 格式回复：
    能完成：
    input: 输入的指令
    response：以机器人的角色用大白话描述场景，说说准备咋执行动作。
    can_execute: True
    code ：输出执行代码。
    不能完成：
    response ：以机器人的角色用大白话描述场景，解释为啥搞不定。
    can_execute: False
    code ：输出执行代码。

我的指令中可能有部分内容是和你对话的, 对应这部分内容, 它们没有相应的函数可以去执行, 此时你不仅需要输出必要的函数, 也要在response中加入相应的聊天回复, 请注意, 此时你的聊天回复内容可以自由发挥，但不需要说明你不必执行动作.

【格式要求】
    确保 json 回复格式准确无误。
【示例】
    输入：红色打火机放到黑色盒子里 输出：{example_json_string_1}
    输入：为把苹果放到篮子里 输出：{example_json_string_2}
    输入：如果太阳西升东落的话，把苹果放到篮子里 输出：{example_json_string_3}
    输入：告诉我你看到了什么 输出：{example_json_string_4}
                            """
        
        self.messages =[
                {
                    "role": "system", 
                    "content": self.system_promotion
                }
            ]
    
    def encode_image(self,image_path):
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # 我们使用标准库 base64.b64encode 函数将图片编码成 base64 格式的 image_url
        image_url = f"data:image/{os.path.splitext(image_path)[1]};base64,{base64.b64encode(image_data).decode('utf-8')}"
        return image_url
    
    def forward_image(self, image_path, content= "把苹果放到盘子里"):
        image_url = self.encode_image(image_path)
        messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url", # <-- 使用 image_url 类型来上传图片，内容为使用 base64 编码过的图片内容
                            "image_url": {
                                "url": image_url,
                            },
                        },
                        {
                            "type": "text",
                            "text":content, # <-- 使用 text 类型来提供文字指令，例如“描述图片内容”
                        },
                    ],
                }
            ]
        
        self.messages.append(messages[0])

        completion = self.client.chat.completions.create(
            model="moonshot-v1-8k-vision-preview",
            messages=self.messages
        )
        
        chatBot_response = completion.choices[0].message.content
        return chatBot_response
    
    def extract_content(self, json_data, object = "response"):
        try:
            content = ""
            data = json.loads(json_data)
            for i in range(len(data[object])):
                content = content + data[object][i]
            return content
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Error extracting content: {e}")
            return None

if __name__ == '__main__':
    chatBot = ChatBotHandler()
    foo = RobotWrapper()
    chatBot_response = chatBot.forward_image(
        image_path= "outputs/color.png",
        content="你好，介绍下你自己")
        # content="告诉我你看到了什么")

    response = chatBot.extract_content(chatBot_response,"response")
    code = chatBot.extract_content(chatBot_response,"code")
    print(response)
    print(code)
    exec(code)
    plan()