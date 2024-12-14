

# 下面是终端中进行最终测试的代码
#python detect.py --weights runs\train\exp2\weights\best.pt --source C:\Users\ASUS\Desktop\detection\test\images --img 640 --conf 0.65 --iou 0.3 --save-txt --save-conf --project runs/detect --name test_conf05_iou03
# 下面是在终端运行yolo的代码
python train.py --img 640 --batch 16 --epochs 50 --data apple.yaml --weights yolov5m.pt --project apple_project --name yolov5m_training
# 导航到文件环境来运行
cd C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\apple.yaml



