import cv2
import numpy as np
import os


image_dir = r"C:\Users\ASUS\Desktop\detection\train\images" # Path to the training dataset
mask_dir = r"C:\Users\ASUS\Desktop\detection\train\masks"   # Path to the mask folder
label_dir = r"C:\Users\ASUS\Desktop\detection\train\label"  # Path to the label folder

# Create the labels folder
os.makedirs(label_dir, exist_ok=True)

# Iterate through all files in the image folder
for image_file_name in os.listdir(image_dir):
    image_file = os.path.join(image_dir, image_file_name)
    mask_file = os.path.join(mask_dir, image_file_name)

    # Check if the mask file exists
    if not os.path.exists(mask_file):
        print(f"Mask file for {image_file_name} not found. Skipping...")
        continue

    # Read the mask image
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    # Create the label file
    label_file_path = os.path.join(label_dir, image_file_name.replace('.png', '.txt'))
    with open(label_file_path, 'w') as label_file:
        # Iterate through each pixel value from 1 to 255 to detect apple regions
        for pixel_value in range(1, 256):
            # Find all points with the current pixel value
            y_indices, x_indices = np.where(mask == pixel_value)

            # Check if any apple region is detected
            if len(x_indices) > 0 and len(y_indices) > 0:
                # Find the minimum and maximum coordinates of the bounding box
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()

                # Calculate the center and size of the bounding box, normalized to YOLO format
                x_center = (x_min + x_max) / 2 / mask.shape[1]
                y_center = (y_min + y_max) / 2 / mask.shape[0]
                width = (x_max - x_min) / mask.shape[1]
                height = (y_max - y_min) / mask.shape[0]

                # Write the label to the file
                label_file.write(f"0 {x_center} {y_center} {width} {height}\n")

    print(f"Labels saved for {image_file_name} to {label_file_path}")

print("All labels have been generated.")

