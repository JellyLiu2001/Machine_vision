import cv2
import json
import os
import matplotlib.pyplot as plt

# 定义路径
ground_truth_path = r"C:\Users\ASUS\Desktop\test_data\detection\ground_truth.json"
detection_results_path = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\runs\yolov5m_detect\detections_coco_format.json"
test_images_dir = r"C:\Users\ASUS\Desktop\test_data\detection\images"

# 打开 ground truth 和 detection 文件
with open(ground_truth_path, 'r') as gt_file, open(detection_results_path, 'r') as det_file:
    ground_truth = json.load(gt_file)
    detections = json.load(det_file)

# 加载图片信息
images = {img["id"]: img["filename"] for img in ground_truth["images"]}

# 可视化第一个 image_id
image_id = 323# 你可以修改这个 ID 来选择其他图片  2 好一点 323 可以分析效果差的原因（绿苹果和叶子颜色相近，阴影遮挡）
if image_id not in images:
    raise ValueError(f"Image ID {image_id} not found in ground truth data.")

image_file = images[image_id]
image_path = os.path.join(test_images_dir, image_file)

# 检查图片文件是否存在
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

# 打开图片
img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"Failed to load image at path: {image_path}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 绘制 ground truth
for ann in ground_truth["annotations"]:
    if ann["image_id"] == image_id:
        x, y, w, h = ann["bbox"]
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)  # 绿色框表示 ground truth

# 绘制检测结果
for det in detections:
    if det["image_id"] == image_id:
        x, y, w, h = det["bbox"]
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)  # 红色框表示检测结果

# 显示图片
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis("off")
plt.title(f"Visualization for Image ID: {image_id}")
plt.show()

