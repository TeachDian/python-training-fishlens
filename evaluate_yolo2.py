# evaluate_yolo.py
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the trained YOLOv8 model
model = YOLO('runs/detect/train/weights/best.pt')

# Validate the model on the validation set and store metrics
metrics = model.val(data='data.yaml')

# Extract precision, recall, F1-score, and loss metrics
precision = metrics.box['precision']
recall = metrics.box['recall']
f1_score = metrics.box['f1']
box_loss = metrics.loss['box']
cls_loss = metrics.loss['cls']
val_box_loss = metrics.val_loss['box']
val_cls_loss = metrics.val_loss['cls']

# 1. Precision-Recall Curve
plt.figure()
plt.plot(recall, precision, label='Precision-Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('precision_recall_curve.png')
plt.show()

# 2. F1-Score Curve
plt.figure()
plt.plot(f1_score, label='F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('F1 Score Curve')
plt.legend()
plt.savefig('f1_score_curve.png')
plt.show()

# 3. Loss Curves (Box and Class Loss)
plt.figure()
plt.plot(box_loss, label='Train Box Loss')
plt.plot(val_box_loss, label='Validation Box Loss')
plt.plot(cls_loss, label='Train Class Loss')
plt.plot(val_cls_loss, label='Validation Class Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('loss_curves.png')
plt.show()

# 4. Confusion Matrix (Example for one class)
y_true = ...  # True labels (e.g., ground truth bounding boxes/classes)
y_pred = ...  # Predicted labels (from model predictions)
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# 5. mAP (Mean Average Precision)
mAP50 = metrics.box['map50']  # mAP at IoU 0.50
mAP50_95 = metrics.box['map']  # mAP at IoU 0.50:0.95

print(f'mAP@0.50: {mAP50}')
print(f'mAP@0.50:0.95: {mAP50_95}')

# 6. Overfitting/Underfitting Analysis (Training vs Validation Loss)
plt.figure()
plt.plot(box_loss, label='Training Box Loss')
plt.plot(val_box_loss, label='Validation Box Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Overfitting/Underfitting Analysis - Box Loss')
plt.legend()
plt.savefig('overfitting_analysis_box_loss.png')
plt.show()

plt.figure()
plt.plot(cls_loss, label='Training Class Loss')
plt.plot(val_cls_loss, label='Validation Class Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Overfitting/Underfitting Analysis - Class Loss')
plt.legend()
plt.savefig('overfitting_analysis_cls_loss.png')
plt.show()

# Optional: Saving precision, recall, and other stats to a text file
with open('evaluation_metrics.txt', 'w') as f:
    f.write(f'Precision: {precision}\n')
    f.write(f'Recall: {recall}\n')
    f.write(f'F1 Score: {f1_score}\n')
    f.write(f'mAP@0.50: {mAP50}\n')
    f.write(f'mAP@0.50:0.95: {mAP50_95}\n')
