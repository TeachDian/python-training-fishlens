import torch
from ultralytics import YOLO

# Load a pretrained YOLOv8 model (e.g., 'yolov8s.pt' for YOLOv8 small)
model = YOLO('yolovSabinet.pt')

# Train the model on your custom dataset
results = model.train(data='data.yaml', epochs=3, imgsz=640, device='cuda' if torch.cuda.is_available() else 'cpu')

# Print the results
print(results)
