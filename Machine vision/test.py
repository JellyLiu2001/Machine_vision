import cv2
import os

# 文件路径
image_file = r"C:\Users\ASUS\Desktop\detection\train\images\20150921_131234_image6.png"# 原始图像文件
label_file = r"C:\Users\ASUS\Desktop\detection\train\label\20150921_131234_image6.txt"# 对应的标签文件

# 读取图像
image = cv2.imread(image_file)

# 获取图像尺寸
height, width = image.shape[:2]

# 打开标签文件并读取每行
with open(label_file, 'r') as file:
    for line in file:
        # 解析标签信息
        class_id, x_center, y_center, box_width, box_height = map(float, line.split())

        # 将归一化坐标转换回图像像素坐标
        x_center = int(x_center * width)
        y_center = int(y_center * height)
        box_width = int(box_width * width)
        box_height = int(box_height * height)

        # 计算左上角和右下角坐标
        top_left = (int(x_center - box_width / 2), int(y_center - box_height / 2))
        bottom_right = (int(x_center + box_width / 2), int(y_center + box_height / 2))

        # 绘制边界框
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# 显示图像
cv2.imshow("Detected Apples", image)  # 显示图像
cv2.waitKey(0)  # 等待按键
cv2.destroyAllWindows()  # 关闭窗口
