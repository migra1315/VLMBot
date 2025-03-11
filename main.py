from realsense import realsense_camera
from grounding_dino import groudingDINO
from hand_eye_map import hand_eye_map
class VLMBot():
    def __init__(self):
        self.detect_handler = groudingDINO()
        self.camera_handler = realsense_camera()
        self.hand_eye_mapper = hand_eye_map()

    def forward(self):
        self.camera_handler.get_image()
        self.detect_handler.load_image("detect_img.png")
        pixel_x,pixel_y = self.detect_handler.forward("mouse")
        print(f"locate pixel index x {pixel_x} y {pixel_y}")

        camera_x,camera_y,camera_z = self.camera_handler.location(pixel_x,pixel_y)
        print(f"locate camera index x {camera_x} y {camera_y} z {camera_z}")

        robot_x, robot_y, robot_z = self.hand_eye_mapper.map(camera_x,camera_y,camera_z)
        print(f"transformed robot index x {robot_x} y {robot_y} z {robot_z}")





if __name__ == '__main__':
    foo = VLMBot()
    foo.forward()




