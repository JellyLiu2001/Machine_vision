import pandas as pd
import matplotlib.pyplot as plt

# 加载 results.csv
results_path = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\apple_project\yolov5m_training\results.csv"  # 替换为你的路径
results = pd.read_csv(results_path)

# 去除列名中的多余空格
results.columns = results.columns.str.strip()

# 绘制 mAP 曲线
plt.figure(figsize=(10, 6))
plt.plot(results['epoch'], results['metrics/mAP_0.5'], label='mAP@0.5', color='blue', linestyle='-')
plt.plot(results['epoch'], results['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95', color='orange', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.title('mAP Curve')
plt.legend()
plt.grid()
plt.tight_layout()

# 绘制训练和验证的 Box Loss 对比曲线
plt.figure(figsize=(12, 8))
plt.plot(results['epoch'], results['train/box_loss'], label='Train Box Loss', color='red', linestyle='-', linewidth=2)
plt.plot(results['epoch'], results['val/box_loss'], label='Val Box Loss', color='orange', linestyle='-', linewidth=2)

# 设置图表标题和标签
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Box Loss', fontsize=14)
plt.title('Training and Validation Box Loss Curve', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 优化刻度显示
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


plt.figure(figsize=(10, 6))
plt.plot(results['epoch'], results['metrics/precision'], label='Precision', color='green')
plt.plot(results['epoch'], results['metrics/recall'], label='Recall', color='red')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 显示图表
plt.show()