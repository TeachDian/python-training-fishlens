import torch
import ultralytics
import tensorflow as tf
from pathlib import Path

# Step 1: Train the YOLOv8 Model
def train_yolov8(data_yaml, epochs=1, img_size=640):
    model = ultralytics.YOLO("yolov8n.pt")  # Load pre-trained YOLOv8 model
    model.train(data=data_yaml, epochs=epochs, imgsz=img_size)  # Train the model
    return model

# Step 2: Export to TensorFlow format
def export_to_tensorflow(model, export_path="yolov8_tf_model"):
    model.export(format="tf")  # Export model to TensorFlow SavedModel format
    return Path(export_path)

# Step 3: Convert TensorFlow model to TFLite
def convert_to_tflite(tf_model_path, tflite_output="model.tflite"):
    # Load the TensorFlow model
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    
    # Convert the model to TensorFlow Lite format
    tflite_model = converter.convert()

    # Save the TFLite model to disk
    with open(tflite_output, "wb") as f:
        f.write(tflite_model)
    print(f"Model converted to TFLite and saved as {tflite_output}")

# Main script to run everything
if __name__ == "__main__":
    data_yaml = "data.yaml"  # Path to your dataset configuration file
    trained_model = train_yolov8(data_yaml)  # Train YOLOv8 model
    
    export_path = export_to_tensorflow(trained_model)  # Export model to TensorFlow format
    convert_to_tflite(export_path)  # Convert to TFLite
