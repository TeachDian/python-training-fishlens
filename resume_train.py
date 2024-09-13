import torch
from ultralytics import YOLO

model = YOLOv8('data.yaml')  # Load model configuration
model.load('runs/detect/train/weights/last.pt')  # Load from checkpoint

model.train(
    data='path/to/dataset.yaml',
    epochs=100,  # Continue training for additional epochs
    weights='runs/detect/train/weights/last.pt'
)

