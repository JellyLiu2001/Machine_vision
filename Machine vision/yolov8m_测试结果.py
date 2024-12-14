import os
import json

# 文件路径
labels_dir = r"C:\Users\ASUS\PycharmProjects\PythonProject\runs\detect\predict\labels"  # 更新为 labels 文件路径
mapping_file = r"C:\Users\ASUS\Desktop\test_data\detection\mapping.json"  # 替换为 mapping.json 文件路径
output_coco_file = r"C:\Users\ASUS\PycharmProjects\PythonProject\detections_coco_format.json"  # 替换为保存 JSON 文件路径

# 假设图片宽度和高度
img_width = 720  # 替换为实际图片宽度
img_height = 1280  # 替换为实际图片高度

# 加载 mapping.json
print("Loading mapping.json...")
with open(mapping_file, "r") as f:
    mapping = json.load(f)

# 转换 YOLOv8 的 labels 文件到 COCO 格式
print("Parsing labels...")
detections = []

for label_file in os.listdir(labels_dir):
    if label_file.endswith(".txt"):
        image_name = label_file.replace(".txt", ".png")  # 假设图片后缀为 .png
        if image_name not in mapping:
            print(f"Warning: {image_name} not found in mapping.json. Skipping.")
            continue
        image_id = mapping[image_name]
        label_path = os.path.join(labels_dir, label_file)

        with open(label_path, "r") as f:
            for line_num, line in enumerate(f.readlines(), start=1):
                data = line.strip().split()
                if len(data) < 5:  # 确保字段数至少为 5
                    print(f"Error: Invalid format in {label_file} at line {line_num}: {line.strip()}")
                    continue
                try:
                    category_id = 0  # 类别固定为 0
                    bbox = [
                        (float(data[1]) - float(data[3]) / 2) * img_width,  # x_min
                        (float(data[2]) - float(data[4]) / 2) * img_height,  # y_min
                        float(data[3]) * img_width,                        # width
                        float(data[4]) * img_height                        # height
                    ]
                    score = float(data[5]) if len(data) == 6 else 1.0  # 如果没有置信度，设置为 1.0

                    detections.append({
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": score
                    })
                except ValueError as e:
                    print(f"Error: Invalid data in {label_file} at line {line_num}: {line.strip()} ({e})")

# 保存为 COCO 格式 JSON
print("Saving results to COCO format JSON...")
with open(output_coco_file, "w") as f:
    json.dump(detections, f)

print(f"Saved YOLOv8 detection results to {output_coco_file}")
