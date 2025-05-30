import cv2
import numpy as np

def make_rgb():
    # 创建一个空白图像，尺寸为 720x1280，3通道（RGB），初始值为 0
    height, width = 720, 1280
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 设置指定区域的像素值为 255, 255, 255（白色）
    x_start, x_end = 0, 710
    y_start, y_end = 285,1279

    image[y_start:y_end, x_start:x_end] = [255, 255, 255]

    # 将图像保存为 PNG 文件
    cv2.imwrite('doc/example_data/workspace_mask.png', image)

    print("RGB图像生成完成，已保存为 output_rgb.png")

def make_gray():

    # 创建一个空白图像，尺寸为 720x1280，初始值为 0
    height, width = 720, 1280
    image = np.zeros((height, width), dtype=np.uint8)

    # 设置指定区域的像素值为 1
    x_start, x_end = 500, 1279 
    y_start, y_end =0, 520

    image[y_start:y_end, x_start:x_end] =  255

    # 将图像保存为 PNG 文件
    cv2.imwrite('./outputs/workspace_mask.png', image)

    print("图像生成完成，已保存为 output.png")

def show():

    # 读取图像
    image = cv2.imread('workspace_mask.png')  # 替换为你的图像路径
    print(image.shape)

    # 检查图像是否成功加载
    if image is None:
        print("图像加载失败，请检查路径是否正确")
    else:
        # 创建一个窗口并设置其尺寸
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image', 720, 480)  # 设置窗口尺寸为 720x480
        # 显示图像
        cv2.imshow('Image', image)

        # 等待用户按键，0 表示无限期等待
        cv2.waitKey(0)

        # 销毁所有窗口
        cv2.destroyAllWindows()

# show()
make_gray()