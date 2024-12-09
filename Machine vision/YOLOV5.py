import cv2
import numpy as np
import os

# 文件夹路径
image_dir = r"C:\Users\ASUS\Desktop\detection\train\images"  # 图片文件夹路径
mask_dir = r"C:\Users\ASUS\Desktop\detection\train\masks"   # 掩码文件夹路径
label_dir = r"C:\Users\ASUS\Desktop\detection\train\label"  # 标签文件夹路径

# 自动创建 labels 文件夹
os.makedirs(label_dir, exist_ok=True)

# 遍历图片文件夹中的所有文件
for image_file_name in os.listdir(image_dir):
    # 构造完整的图片和掩码路径
    image_file = os.path.join(image_dir, image_file_name)
    mask_file = os.path.join(mask_dir, image_file_name)  # 假设掩码文件名与图片文件名相同

    # 检查掩码文件是否存在
    if not os.path.exists(mask_file):
        print(f"Mask file for {image_file_name} not found. Skipping...")
        continue

    # 读取掩码图像
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    # 创建与图像同名的标签文件
    label_file_path = os.path.join(label_dir, image_file_name.replace('.png', '.txt'))
    with open(label_file_path, 'w') as label_file:
        # 遍历每个像素值1-255来检测苹果区域
        for pixel_value in range(1, 256):
            # 找到所有值为当前像素值的点
            y_indices, x_indices = np.where(mask == pixel_value)

            # 检查是否检测到任何苹果区域
            if len(x_indices) > 0 and len(y_indices) > 0:
                # 找到边界框的最小和最大坐标
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()

                # 计算边界框中心和尺寸，归一化到 YOLO 格式
                x_center = (x_min + x_max) / 2 / mask.shape[1]
                y_center = (y_min + y_max) / 2 / mask.shape[0]
                width = (x_max - x_min) / mask.shape[1]
                height = (y_max - y_min) / mask.shape[0]

                # 写入标签文件（类别ID假设为0）
                label_file.write(f"0 {x_center} {y_center} {width} {height}\n")

    print(f"Labels saved for {image_file_name} to {label_file_path}")

print("All labels have been generated.")
