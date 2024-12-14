import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

# 文件路径
ground_truth_path = r"C:\Users\ASUS\Desktop\test_data\detection\ground_truth.json"  # ground truth 文件路径
detections_path = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\runs\yolov5m_detect\detections_coco_format.json" # 转换后的检测结果路径

# 加载 ground truth 数据
print("Loading ground truth data...")
coco_gt = COCO(ground_truth_path)

# 加载检测结果数据
print("Loading detection results...")
with open(detections_path, 'r') as f:
    detection_results = json.load(f)

# 确保检测结果格式符合 COCO 格式
print("Preparing detection results...")
detections_coco = []
for det in detection_results:
    bbox = det['bbox']
    # 检查 bbox 格式是否正确
    detections_coco.append({
        "image_id": det['image_id'],
        "category_id": det['category_id'],
        "bbox": [
            bbox[0],  # x_min
            bbox[1],  # y_min
            bbox[2],  # width
            bbox[3]   # height
        ],
        "score": det['score']
    })

# 保存检测结果为临时 COCO JSON 文件
prepared_detections_path = detections_path.replace("detections_coco_format.json", "prepared_detections.json")
with open(prepared_detections_path, 'w') as f:
    json.dump(detections_coco, f)
print(f"Saved prepared detection results to {prepared_detections_path}")

# 加载检测结果到 COCO API
coco_dt = coco_gt.loadRes(prepared_detections_path)

# 进行评估
print("Evaluating detection results...")
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

# 修改 IoU 范围，将最低 IoU 阈值设为 0.3
coco_eval.params.iouThrs = np.linspace(0.5, 0.95, 10)

# 执行评估
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print("Evaluation complete.")

