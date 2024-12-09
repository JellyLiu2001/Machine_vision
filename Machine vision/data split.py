import os
import shutil
import random

# 数据路径
image_dir = r"C:\Users\ASUS\Desktop\detection\train\images"
label_dir = r"C:\Users\ASUS\Desktop\detection\train\label"
train_image_dir = r"C:\Users\ASUS\Desktop\detection\train-split\images"
train_label_dir = r"C:\Users\ASUS\Desktop\detection\train-split\labels"
val_image_dir = r"C:\Users\ASUS\Desktop\detection\val\images"
val_label_dir = r"C:\Users\ASUS\Desktop\detection\val\labels"

# 创建训练和验证文件夹
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 获取所有图片文件
images = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]

# 随机划分 80% 为训练集，20% 为验证集
random.shuffle(images)
split_idx = int(0.8 * len(images))
train_images = images[:split_idx]
val_images = images[split_idx:]

# 移动文件
for image in train_images:
    shutil.move(os.path.join(image_dir, image), os.path.join(train_image_dir, image))
    shutil.move(os.path.join(label_dir, image.replace('.png', '.txt').replace('.jpg', '.txt')),
                os.path.join(train_label_dir, image.replace('.png', '.txt').replace('.jpg', '.txt')))

for image in val_images:
    shutil.move(os.path.join(image_dir, image), os.path.join(val_image_dir, image))
    shutil.move(os.path.join(label_dir, image.replace('.png', '.txt').replace('.jpg', '.txt')),
                os.path.join(val_label_dir, image.replace('.png', '.txt').replace('.jpg', '.txt')))

print("Dataset split complete.")
