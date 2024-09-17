import onnx
from onnx_tf.backend import prepare

# Load the modified ONNX model
onnx_model = onnx.load("model.onnx")

# Convert the ONNX model to a TensorFlow representation
tf_rep = prepare(onnx_model)

# Export the TensorFlow model to a SavedModel directory
tf_rep.export_graph("saved_model_dir")
