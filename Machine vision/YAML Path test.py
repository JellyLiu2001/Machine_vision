import os
# Check if the apple.yaml file exists
file_path = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\apple.yaml"
if os.path.exists(file_path):
    print("apple.yaml file exists!")
else:
    print("apple.yaml file not found!")
