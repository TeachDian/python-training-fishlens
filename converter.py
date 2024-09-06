from ultralytics import YOLO

# Load the YOLOv8 model from the .pt file
model = YOLO("models/3epoch.pt")  # Replace with your .pt model path

# Export the model to TFLite format
model.export(format="tflite")
