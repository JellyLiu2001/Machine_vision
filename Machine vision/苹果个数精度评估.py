import json
from pycocotools.coco import COCO

# 文件路径
ground_truth_path = r"C:\Users\ASUS\Desktop\test_data\detection\ground_truth.json"  # ground truth 文件路径
detections_path = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\runs\yolov5m_detect\detections_coco_format.json"  # 检测结果路径

# 加载 ground truth 数据
print("Loading ground truth data...")
coco_gt = COCO(ground_truth_path)

# 加载检测结果数据
print("Loading detection results...")
with open(detections_path, 'r') as f:
    detection_results = json.load(f)

# 统计每张图像的真实苹果数量 (Ground Truth)
gt_counts = {}
for ann in coco_gt.anns.values():
    image_id = ann['image_id']
    if image_id not in gt_counts:
        gt_counts[image_id] = 0
    gt_counts[image_id] += 1

# 统计每张图像的预测苹果数量 (Detections)
pred_counts = {}
for det in detection_results:
    image_id = det['image_id']
    if image_id not in pred_counts:
        pred_counts[image_id] = 0
    pred_counts[image_id] += 1

# 计算计数误差和平均精度
total_images = len(gt_counts)
total_absolute_error = 0
relative_errors = []

print("\nPer Image Count Comparison:")
for image_id in gt_counts:
    gt_count = gt_counts[image_id]
    pred_count = pred_counts.get(image_id, 0)  # 如果没有检测结果，预测数量为0
    absolute_error = abs(gt_count - pred_count)
    relative_error = abs(gt_count - pred_count) / gt_count if gt_count != 0 else 0

    total_absolute_error += absolute_error
    relative_errors.append(relative_error)

    print(f"Image ID: {image_id} | Ground Truth: {gt_count} | Prediction: {pred_count} | Error: {absolute_error}")

# 计算平均精度（基于数量差异）
average_precision = 1 - (sum(relative_errors) / total_images)

print("\nCount Evaluation Metrics:")
print(f"Total Images: {total_images}")
print(f"Mean Absolute Error (MAE): {total_absolute_error / total_images:.2f}")
print(f"Average Precision (Counting): {average_precision:.4f}")
