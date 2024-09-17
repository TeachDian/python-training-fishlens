import onnx
from onnx_tf.backend import prepare

# Load the ONNX model
onnx_model_path = "model.onnx"  # Replace with the path to your ONNX model
onnx_model = onnx.load(onnx_model_path)

# Convert ONNX model to TensorFlow
tf_rep = prepare(onnx_model)

# Export the TensorFlow model as a .pb file
tf_model_path = ""
tf_rep.export_graph(tf_model_path)
