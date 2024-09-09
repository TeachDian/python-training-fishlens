import torch
import os
from ultralytics import YOLO

def main():
    print(torch.cuda.is_available())  # Should return True if CUDA is available
    print(torch.cuda.get_device_name(0))  # Prints the name of your GPU (e.g., NVIDIA RTX 4060)

    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Check if model path exists
    model_path = "models/150epoch.pt"
    if not os.path.exists(model_path):
        print(f"Model path '{model_path}' not found, starting training from scratch...")
        model = YOLO('yolov8n.pt')  # Start with a base pretrained model
    else:
        print(f"Loading model from {model_path}...")
        model = YOLO(model_path)

    # Set up the training parameters
    data_yaml = 'data.yaml'  # Path to your dataset YAML configuration file

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset configuration file '{data_yaml}' not found.")

    # Train the model on your custom dataset
    results = model.train(data=data_yaml, epochs=1, imgsz=640, device=device)

    # Print results (optional)
    print(results)

    # Optionally save model after training
    final_model_path = 'models/final_model.pt'
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")

if __name__ == "__main__":
    main()
