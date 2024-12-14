import os

# 检查 apple.yaml 文件是否存在
file_path = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\apple.yaml"
if os.path.exists(file_path):
    print("apple.yaml file exists!")
else:
    print("apple.yaml file not found!")
