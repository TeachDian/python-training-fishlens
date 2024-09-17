import onnx
from onnx_tf.backend import prepare
import os

# Load your ONNX model
onnx_model = onnx.load("modified_model.onnx")

# Convert to TensorFlow model
tf_rep = prepare(onnx_model)

# Define the directory to save the TensorFlow model
saved_model_dir = "saved_model_dir"  # Specify the path where you want to save the model

# Ensure the directory exists
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)

# Export the TensorFlow model to the specified directory
tf_rep.export_graph(saved_model_dir)
