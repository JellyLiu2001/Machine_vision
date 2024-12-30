import json
from pycocotools.coco import COCO

# Path
ground_truth_path = r"C:\Users\ASUS\Desktop\test_data\detection\ground_truth.json"
detections_path = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\runs\yolov5m_detect\detections_coco_format.json"

# Load ground truth data
print("Loading ground truth data...")
coco_gt = COCO(ground_truth_path)

# Load detection result data
print("Loading detection results...")
with open(detections_path, 'r') as f:
    detection_results = json.load(f)

# Count the number of apples in ground truth for each image
gt_counts = {}
for ann in coco_gt.anns.values():
    image_id = ann['image_id']
    if image_id not in gt_counts:
        gt_counts[image_id] = 0
    gt_counts[image_id] += 1

# Count the number of predicted apples for each image
pred_counts = {}
for det in detection_results:
    image_id = det['image_id']
    if image_id not in pred_counts:
        pred_counts[image_id] = 0
    pred_counts[image_id] += 1

# Calculate counting error and average precision
total_images = len(gt_counts)
total_absolute_error = 0
relative_errors = []

print("\nPer Image Count Comparison:")
for image_id in gt_counts:
    gt_count = gt_counts[image_id]
    pred_count = pred_counts.get(image_id, 0)
    absolute_error = abs(gt_count - pred_count)
    relative_error = abs(gt_count - pred_count) / gt_count if gt_count != 0 else 0

    total_absolute_error += absolute_error
    relative_errors.append(relative_error)

    print(f"Image ID: {image_id} | Ground Truth: {gt_count} | Prediction: {pred_count} | Error: {absolute_error}")

# Calculate average precision
average_precision = 1 - (sum(relative_errors) / total_images)

print("\nCount Evaluation Metrics:")
print(f"Total Images: {total_images}")
print(f"Mean Absolute Error (MAE): {total_absolute_error / total_images:.2f}")
print(f"Average Precision (Counting): {average_precision:.4f}")

