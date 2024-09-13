import torch
from ultralytics import YOLO

print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should print NVIDIA RTX 4060 or similar

if __name__ == "__main__":
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a pretrained YOLOv8 model
    model_path = "yolov8n.pt"
    model = YOLO(model_path)

    # Train the model on your custom dataset
    results = model.train(data='data.yaml', epochs=250, imgsz=640, device=device)

    # Print the results
    print(results)
