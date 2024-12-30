from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd

# paths
ground_truth_path = r"C:\Users\ASUS\Desktop\test_data\detection\ground_truth.json"
predictions_path = r"C:\Users\ASUS\PycharmProjects\PythonProject\yolov5\runs\yolov5m_detect\detections_coco_format.json"

# Load Ground Truth and predictions
print("Loading ground truth and predictions...")
coco_gt = COCO(ground_truth_path)
coco_pred = coco_gt.loadRes(predictions_path)

# Initialize COCO
coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')

# Extract evaluation results for each image
print("Evaluating per-image metrics...")
image_metrics = []

for img_id in coco_gt.getImgIds():
    # Set the current image
    coco_eval.params.imgIds = [img_id]
    coco_eval.evaluate()
    coco_eval.accumulate()

    # Extract evaluation data
    eval_img = coco_eval.evalImgs[0]  # Evaluation results for the current image
    if eval_img:  # Ensure evaluation results exist

        gt_annots = coco_gt.getAnnIds(imgIds=[img_id])  # Number of Ground Truth boxes
        pred_annots = coco_pred.getAnnIds(imgIds=[img_id])  # Number of predicted boxes
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

# Convert to DataFrame and save
image_metrics_df = pd.DataFrame(image_metrics)
image_metrics_df.sort_values(by='Precision', ascending=True, inplace=True)
output_path = r"C:\Users\ASUS\PycharmProjects\PythonProject\image_metrics.csv"
image_metrics_df.to_csv(output_path, index=False)
print(f"Per-image metrics saved to: {output_path}")

# Output images with lowest and highest Precision
print("Images with lowest Precision:")
print(image_metrics_df.head())
print("\nImages with highest Precision:")
print(image_metrics_df.tail())


