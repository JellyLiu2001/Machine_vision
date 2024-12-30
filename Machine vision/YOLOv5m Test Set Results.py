import os
import json
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# path
weights_path = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\apple_project\yolov5m_training3\weights\best.pt"
test_image_dir = r"C:\Users\ASUS\Desktop\test_data\detection\images"
ground_truth_path = r"C:\Users\ASUS\Desktop\test_data\detection\ground_truth.json"
labels_dir = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\runs\detect\test_conf05_iou032\labels"
output_dir = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\runs\detect\test_conf05_iou032"
coco_results_path = os.path.join(output_dir, "detections_coco_format.json")
mapping_file = r"C:\Users\ASUS\Desktop\test_data\detection\mapping.json"

# Set the image width and height
img_width = 720
img_height = 1280

# Output directory
os.makedirs(output_dir, exist_ok=True)

# Run YOLOv5 detection
command = f"""
python C:/Users/ASUS/PycharmProjects/PythonProject/yolov5/detect.py \
--weights {weights_path} \
--source {test_image_dir} \
--img 640 \
--conf 0.65 \
--iou 0.5 \
--save-txt \
--save-conf \
--project {output_dir} \
--name yolov5m_test2
"""
print(f"Executing: {command}")
os.system(command)

# Check if labels directory exists
if not os.path.exists(labels_dir):
    raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

# Ground truth file for the test set
print("Loading ground truth data...")
coco_gt = COCO(ground_truth_path)

# Mapping file for the test set
print("Loading mapping.json...")
with open(mapping_file, "r") as f:
    mapping = json.load(f)

# Convert YOLOv5 detection results to COCO format
print("Parsing YOLOv5 detection results...")
detections = []

for label_file in os.listdir(labels_dir):
    if label_file.endswith(".txt"):
        image_name = label_file.replace(".txt", ".png")
        if image_name not in mapping:
            print(f"Warning: {image_name} not found in mapping.json. Skipping.")
            continue
        image_id = mapping[image_name]
        label_path = os.path.join(labels_dir, label_file)

        with open(label_path, "r") as f:
            for line in f.readlines():
                data = line.strip().split()
                category_id = 0  # Category ID for apples
                bbox = [
                    (float(data[1]) - float(data[3]) / 2) * img_width,  # x_min
                    (float(data[2]) - float(data[4]) / 2) * img_height,  # y_min
                    float(data[3]) * img_width,                        # width
                    float(data[4]) * img_height                        # height
                ]
                score = float(data[5])

                detections.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": score
                })

# Save results to COCO format JSON
print("Saving results to COCO format JSON...")
with open(coco_results_path, "w") as f:
    json.dump(detections, f)
print(f"Saved YOLOv5 detection results to {coco_results_path}")

# Load detection results into COCO
print("Evaluating model performance...")
coco_dt = coco_gt.loadRes(coco_results_path)
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

# Set IoU range
coco_eval.params.iouThrs = [x / 100.0 for x in range(30, 96, 5)]
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print("Evaluation complete.")

