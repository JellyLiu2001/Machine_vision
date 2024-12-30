import cv2
import json
import os
import matplotlib.pyplot as plt

# Define paths
ground_truth_path = r"C:\Users\ASUS\Desktop\test_data\detection\ground_truth.json"
detection_results_path = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\runs\detect\test_conf05_iou032\detections_coco_format.json"
test_images_dir = r"C:\Users\ASUS\Desktop\test_data\detection\images"

# Open ground truth and detection result files
with open(ground_truth_path, 'r') as gt_file, open(detection_results_path, 'r') as det_file:
    ground_truth = json.load(gt_file)
    detections = json.load(det_file)

# Load image information
images = {img["id"]: img["filename"] for img in ground_truth["images"]}

# Visualize results for the test set
image_id = 3
if image_id not in images:
    raise ValueError(f"Image ID {image_id} not found in ground truth data.")

image_file = images[image_id]
image_path = os.path.join(test_images_dir, image_file)

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

img = cv2.imread(image_path)# Open the image
if img is None:
    raise ValueError(f"Failed to load image at path: {image_path}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Draw ground truth boxes
for ann in ground_truth["annotations"]:
    if ann["image_id"] == image_id:
        x, y, w, h = ann["bbox"]
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)  # Green box for ground truth

# Draw detection results
for det in detections:
    if det["image_id"] == image_id:
        x, y, w, h = det["bbox"]
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)  # Red box for detections

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis("off")
plt.title(f"Visualization for Image ID: {image_id}")
plt.show()


