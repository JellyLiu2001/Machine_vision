import pandas as pd
import matplotlib.pyplot as plt

# Load the results.csv file
results_path = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\apple_project\yolov5m_training\results.csv"  # Replace with your path
results = pd.read_csv(results_path)

# Remove any trailing spaces in column names
results.columns = results.columns.str.strip()

# Plot the mAP curve
plt.figure(figsize=(10, 6))
plt.plot(results['epoch'], results['metrics/mAP_0.5'], label='mAP@0.5', color='blue', linestyle='-')
plt.plot(results['epoch'], results['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95', color='orange', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('mAP')
plt.title('mAP Curve')
plt.legend()
plt.grid()
plt.tight_layout()

# Plot the Training and Validation Box Loss Curve
plt.figure(figsize=(12, 8))
plt.plot(results['epoch'], results['train/box_loss'], label='Train Box Loss', color='red', linestyle='-', linewidth=2)
plt.plot(results['epoch'], results['val/box_loss'], label='Val Box Loss', color='orange', linestyle='-', linewidth=2)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Box Loss', fontsize=14)
plt.title('Training and Validation Box Loss Curve', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Plot the Precision-Recall Curve
plt.figure(figsize=(10, 6))
plt.plot(results['epoch'], results['metrics/precision'], label='Precision', color='green')
plt.plot(results['epoch'], results['metrics/recall'], label='Recall', color='red')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.tight_layout()

# Display all the plots
plt.show()
