from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd

# 文件路径
ground_truth_path = r"C:\Users\ASUS\Desktop\test_data\detection\ground_truth.json"  # 替换为你的标准COCO JSON文件路径
predictions_path = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\runs\yolov5m_detect\detections_coco_format.json"  # 替换为你的预测结果JSON路径


# 加载 Ground Truth 和预测结果
print("Loading ground truth and predictions...")
coco_gt = COCO(ground_truth_path)
coco_pred = coco_gt.loadRes(predictions_path)

# 初始化 COCOeval
coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')

# 提取每张图片的评估结果
print("Evaluating per-image metrics...")
image_metrics = []

for img_id in coco_gt.getImgIds():
    # 设置为当前图片
    coco_eval.params.imgIds = [img_id]
    coco_eval.evaluate()
    coco_eval.accumulate()

    # 提取评估数据
    eval_img = coco_eval.evalImgs[0]  # 当前图片的评估结果
    if eval_img:  # 确保评估结果存在
        # 从评估结果中提取指标
        gt_annots = coco_gt.getAnnIds(imgIds=[img_id])  # Ground Truth 框数
        pred_annots = coco_pred.getAnnIds(imgIds=[img_id])  # 预测框数
        tp = sum(eval_img['dtMatches'][0])  # True Positive
        fp = len(pred_annots) - tp  # False Positive
        fn = len(gt_annots) - tp  # False Negative
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall

        image_metrics.append({
            'image_id': img_id,
            'Precision': precision,
            'Recall': recall,
            'True Positive': tp,
            'False Positive': fp,
            'False Negative': fn,
            'Ground Truth Boxes': len(gt_annots),
            'Predicted Boxes': len(pred_annots)
        })

# 转换为 DataFrame 并保存
image_metrics_df = pd.DataFrame(image_metrics)
image_metrics_df.sort_values(by='Precision', ascending=True, inplace=True)  # 按 Precision 排序
output_path = r"C:\Users\ASUS\PycharmProjects\PythonProject\image_metrics.csv"
image_metrics_df.to_csv(output_path, index=False)
print(f"Per-image metrics saved to: {output_path}")

# 输出最低 Precision 和最高 Precision 的图片
print("Images with lowest Precision:")
print(image_metrics_df.head())
print("\nImages with highest Precision:")
print(image_metrics_df.tail())

