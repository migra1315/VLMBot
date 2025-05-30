import time

import minimalmodbus
import serial
import threading
from pynput import keyboard

class changingtekGripper():
    def __init__(self):
        self.lock = threading.Lock()
        # 寄存器地址
        self.POSITION_HIGH_8 = 0x0102  # 位置寄存器高八位
        self.POSITION_LOW_8 = 0x0103  # 位置寄存器低八位
        self.SPEED = 0x0104
        self.FORCE = 0x0105
        self.MOTION_TRIGGER = 0x0108
        self.BAUD = 115200
        self.isOpen = False

    # 写入位置
    def write_position(self, value):
        with self.lock:
            self.instrument.write_long(self.POSITION_HIGH_8, value)

    # 写入速度
    def write_speed(self, speed):
        with self.lock:
            self.instrument.write_register(self.SPEED, speed, functioncode=6)

    # 写入力
    def write_force(self, force):
        with self.lock:
            self.instrument.write_register(self.FORCE, force, functioncode=6)

    # 触发运动
    def trigger_motion(self):
        with self.lock:
            self.instrument.write_register(self.MOTION_TRIGGER, 1, functioncode=6)

    def read_position(self):
        with self.lock:
            # 读取实时反馈位置信息 high（0x0609）和 low（0x060A）
            high_part = self.instrument.read_register(0x0609, functioncode=3)
            low_part = self.instrument.read_register(0x060A, functioncode=3)
            # 计算执行器实时位置
            return (high_part << 16) + low_part
    
    def connect(self, PORT = 'COM5'):
        self.PORT = PORT
        self.instrument = minimalmodbus.Instrument(self.PORT, 1)
        self.instrument.serial.baudrate = self.BAUD
        self.instrument.serial.timeout = 1     

    def close(self):
        actual_position = self.joint_states_to_actual_position(0)
        self.write_position(actual_position)
        self.trigger_motion()
        self.isOpen = False


    def open(self):
        actual_position = self.joint_states_to_actual_position(-0.5)
        self.write_position(actual_position)
        self.trigger_motion()
        self.isOpen = True


    def Activate(self):
        #写入位置

        self.write_position(0)
        # 写入速度
        self.write_speed(100)

        # 写输入
        self.write_force(100)

        # 触发运动
        # self.trigger_motion()
        # time.sleep(0.5)

        # self.write_position(0)
        # self.trigger_motion()

    # 将关节位置转为实际位置
    def joint_states_to_actual_position(self,joint_state):
        joint_state = 0 if joint_state > 0 else joint_state
        joint_state = -1 if joint_state < -1 else joint_state
        return int((joint_state+1)*9000)
    
    # 将关节位置转为实际位置
    def actual_position_to_joint_states(self, actual_position):
        actual_position = 9000 if actual_position > 9000 else actual_position
        actual_position = 0 if actual_position < 0 else actual_position
        return actual_position/9000-1
    
    # 将读取电机位置转为关节位置
    def motor_position_to_joint_states(self, motor_position):
        motor_position = 1360 if motor_position > 1360 else motor_position
        motor_position = 360 if motor_position < 360 else motor_position
        return (motor_position-1360)/1000
    
    def print_position_continuously(self):
        """以10Hz的频率打印位置信息"""
        while True:
            current_position = self.motor_position_to_joint_states(self.read_position())
            print(f"Current position (10Hz): {current_position}")
            time.sleep(0.1)  # 10Hz

    def on_keyboard_pressed(self, key):    
        try:
            if(key.name == 'space'):
                if(self.isOpen):
                    self.close()
                    print('set position as 0')

                else:
                    self.open()     
                    print('set position as -1')

            # 读取位置信息并打印
            elif(key.name=='alt'):
                current_position = self.read_position()
                print(f"Current position: {current_position}")
        except:
    
               
            return

class colorGripper():
    def __init__(self):
        # 串口设备和波特率
        self.port = "/dev/ttyACM0"
        self.baudrate = 115200
        data = self.make_control_frame(close=True, position=10)

        # data = bytes.fromhex(hex_data)  # 将16进制字符串转换为字节数据
        self.send_serial_data(data)
        self.isopen = False
        # time.sleep(2)

    def send_serial_data(self, data):
        try:
            # 打开串口
            ser = serial.Serial(
                port=self.port,  # 串口设备
                baudrate=self.baudrate,  # 波特率
                bytesize=serial.EIGHTBITS,  # 数据位（8位）
                parity=serial.PARITY_NONE,  # 校验位（无）
                stopbits=serial.STOPBITS_ONE,  # 停止位（1位）
                timeout=1  # 超时时间（秒）
            )
            print(f"串口 {self.port} 已成功打开，波特率：{self.baudrate}")

            # 发送数据
            ser.write(data)
            print(f"已发送数据：{data.hex()}")

            # 等待设备响应（可选）
            time.sleep(0.1)
            if ser.in_waiting > 0:
                response = ser.read(ser.in_waiting)
                print(f"收到响应：{response.hex()}")

        except serial.SerialException as e:
            print(f"串口通信错误：{e}")
        finally:
            # 关闭串口
            if ser.is_open:
                ser.close()
                print(f"串口 {self.port} 已关闭")
    
    def open(self):
        # 要发送的数据（16进制字符串）
        # hex_data = "7b01020020492000c8f97d"
        data = self.make_control_frame(close=False, position=5)

        # data = bytes.fromhex(hex_data)  # 将16进制字符串转换为字节数据
        self.send_serial_data(data)
        self.isopen = True
        # time.sleep(2)

    def close(self):
         # 要发送的数据（16进制字符串）
        # hex_data = "7b01020120492000c8f87d"
        data = self.make_control_frame(close=True, position=10)
        
        # data = bytes.fromhex(hex_data)  # 将16进制字符串转换为字节数据
        self.send_serial_data(data)
        self.isopen = False
        # time.sleep(3)

    def make_control_frame(self, close = True, position=10, velocity=25):
        # 第一个字节：帧头
        frame_head = 0x7B
        # 第二个字节：控制 ID
        control_id = 0x01
        # 第三个字节：控制模式
        control_mode = 0x02
        # 第四个字节：步进电机的转向
        
        direction = 1 if close else 0
        # 第五个字节：步进电机细分值
        subdivision = 32
        angle = position/10*1872
        # 第六个字节和第七个字节：位置控制模式下的角度数据，单位为角度，放大十倍传输
        angle_data = int(angle * 10)
        angle_high = (angle_data >> 8) & 0xFF  # 高八位
        angle_low = angle_data & 0xFF  # 低八位
        # 第八个字节和第九个字节：转速数据，单位为弧度/s，放大十倍传输
        speed_data = int(velocity * 10)
        speed_high = (speed_data >> 8) & 0xFF  # 高八位
        speed_low = speed_data & 0xFF  # 低八位
        # 第十个字节：BCC 校验位，为前面九个字节的异或和
        bcc = (frame_head ^ control_id ^ control_mode ^ direction ^ subdivision ^
            angle_high ^ angle_low ^ speed_high ^ speed_low)
        # 第十一个字节：帧尾
        frame_tail = 0x7D

        # 构造完整的控制指令帧
        control_frame = bytearray([
            frame_head, control_id, control_mode, direction, subdivision,
            angle_high, angle_low, speed_high, speed_low, bcc, frame_tail
        ])

        return control_frame

    def on_keyboard_pressed(self, key):    
        try:
            print(key.name)   
            if(key.name == 'space'):
                if(self.isopen):
                    self.close()
                    print('set position as 0')
              
                else:
                    self.open()     
                    print('set position as -1')

        except:
    
               
            return


if __name__ == '__main__':

    # global open
    # gripper = changingtekGripper()
    # gripper.connect(PORT='/dev/ttyUSB0')
    # gripper.Activate()
    # gripper.open()
    # # gripper.close()
    # """启动打印位置信息的线程"""
    # print_position_thread = threading.Thread(target=gripper.print_position_continuously)
    # print_position_thread.daemon = True  # 设置为后台线程
    # print_position_thread.start()


    foo = colorGripper()
    # foo.close()
    foo.close()
    with keyboard.Listener(on_press=foo.on_keyboard_pressed) as listener:
        listener.join()
    # foo.open()

