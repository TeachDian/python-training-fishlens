# evaluate_yolo.py

from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/train8/weights/best.pt')

# Validate the model on the validation set
metrics = model.val(data='data.yaml')

# Print the evaluation metrics
print(metrics)
