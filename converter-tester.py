import onnx

# Load the ONNX model
onnx_model = onnx.load("model.onnx")

# Print a list of all operations (nodes) in the ONNX model
for node in onnx_model.graph.node:
    print(node.op_type)
