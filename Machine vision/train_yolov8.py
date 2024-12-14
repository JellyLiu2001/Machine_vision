import os
from ultralytics import YOLO

def train_yolov8():
    data_yaml = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\apple.yaml"  # 数据集配置文件路径
    weights_path = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\yolov8m.pt"  # 使用 YOLOv8m 的预训练权重
    output_dir = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\runs\yolov8_training"

    # 创建保存路径
    os.makedirs(output_dir, exist_ok=True)

    # 初始化 YOLOv8 模型
    model = YOLO(weights_path)

    # 开始训练
    model.train(
        data=data_yaml,         # 数据集配置文件
        epochs=50,              # 训练轮数
        batch=16,               # 批量大小
        imgsz=640,              # 输入图像大小
        project=output_dir,     # 训练结果保存路径
        name="yolov8m_test",    # 运行名称
        device=0                # GPU 设备编号，0 表示第一块 GPU
    )

    # 验证模型性能
    model.val(
        data=data_yaml,         # 数据集配置文件
        imgsz=640,              # 输入图像大小
        conf=0.65               # 置信度阈值
    )

    print("YOLOv8 Training and Validation Complete!")

# 确保在主模块中运行
if __name__ == "__main__":
    train_yolov8()
