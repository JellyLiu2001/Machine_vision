import os
import shutil
import random

# Data paths
image_dir = r"C:\Users\ASUS\Desktop\detection\train\images"
label_dir = r"C:\Users\ASUS\Desktop\detection\train\label"
train_image_dir = r"C:\Users\ASUS\Desktop\detection\train-split\images"
train_label_dir = r"C:\Users\ASUS\Desktop\detection\train-split\labels"
val_image_dir = r"C:\Users\ASUS\Desktop\detection\val\images"
val_label_dir = r"C:\Users\ASUS\Desktop\detection\val\labels"

# Create train and validation folders
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Get all image files
images = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]

# Randomly split: 80% for training, 20% for validation
random.shuffle(images)
split_idx = int(0.8 * len(images))
train_images = images[:split_idx]
val_images = images[split_idx:]

# Move files
for image in train_images:
    shutil.move(os.path.join(image_dir, image), os.path.join(train_image_dir, image))
    shutil.move(os.path.join(label_dir, image.replace('.png', '.txt').replace('.jpg', '.txt')),
                os.path.join(train_label_dir, image.replace('.png', '.txt').replace('.jpg', '.txt')))

for image in val_images:
    shutil.move(os.path.join(image_dir, image), os.path.join(val_image_dir, image))
    shutil.move(os.path.join(label_dir, image.replace('.png', '.txt').replace('.jpg', '.txt')),
                os.path.join(val_label_dir, image.replace('.png', '.txt').replace('.jpg', '.txt')))

print("Dataset split complete.")

