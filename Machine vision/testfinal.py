import subprocess
import os

# 定义路径
yolov5_dir = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5"  # YOLOv5 文件夹路径
test_image_dir = r"C:\Users\ASUS\Desktop\detection\test\images"  # 测试图片路径
weights_path = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\runs\train\exp2\weights\best.pt"  # 模型路径
output_dir = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\runs\detect\test_conf065_iou03"  # 输出结果存储路径

# 切换到 YOLOv5 文件夹
os.chdir(yolov5_dir)

# 构造命令参数列表
command = [
    "python", "detect.py",
    "--weights", weights_path,
    "--source", test_image_dir,
    "--img", "640",
    "--conf", "0.65",  # 调整置信度阈值
    "--iou", "0.3",    # 调整IOU阈值
    "--save-txt",
    "--save-conf",
    "--project", "runs/detect",
    "--name", "test_conf05_iou03"
]

# 打印命令（用于调试）
print(f"Executing command: {' '.join(command)}")

# 执行命令并捕获输出
try:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print("检测完成，结果保存在:", output_dir)
    print("命令输出:", result.stdout)
except subprocess.CalledProcessError as e:
    print("检测命令执行失败!")
    print("错误输出:", e.stderr)


# 下面是终端中进行最终测试的代码
#python detect.py --weights runs\train\exp2\weights\best.pt --source C:\Users\ASUS\Desktop\detection\test\images --img 640 --conf 0.65 --iou 0.3 --save-txt --save-conf --project runs/detect --name test_conf05_iou03

