import onnx
from onnx import helper, numpy_helper
from onnx import TensorProto
from tqdm import tqdm
import time
import gc
import psutil  # To check memory usage

# Load the ONNX model
onnx_model = onnx.load("model.onnx")

# Function to clean node names
def sanitize_name(name):
    new_name = name.replace('/', '_').replace('-', '_')
    if not new_name[0].isalpha():
        new_name = 'n' + new_name
    return new_name

# Get the total number of nodes to set up the progress bar
total_nodes = len(onnx_model.graph.node)

# Cache initializer names to avoid repeated searches
initializer_names = {initializer.name for initializer in onnx_model.graph.initializer}

# Function to check and collect garbage if memory usage is high
def check_memory_and_collect(threshold=80):
    memory_usage = psutil.virtual_memory().percent
    if memory_usage > threshold:
        print(f"Memory usage is {memory_usage}%. Collecting garbage...")
        gc.collect()

# Iterate through the nodes with a progress bar
for node in tqdm(onnx_model.graph.node, desc="Processing Nodes", total=total_nodes):
    # Sanitize node names and input/output names
    node.name = sanitize_name(node.name)
    node.input[:] = [sanitize_name(i) for i in node.input]
    node.output[:] = [sanitize_name(o) for o in node.output]

    # Check for Mul operations to fix the data type of initializers and inputs
    if node.op_type == "Mul":
        for i, input_name in enumerate(node.input):
            if input_name in initializer_names:
                for initializer in onnx_model.graph.initializer:
                    if initializer.name == input_name and initializer.data_type == TensorProto.INT64:
                        # Convert int64 data to float64
                        float_data = numpy_helper.to_array(initializer).astype('float64')
                        new_initializer = numpy_helper.from_array(float_data, name=initializer.name)
                        onnx_model.graph.initializer.remove(initializer)
                        onnx_model.graph.initializer.append(new_initializer)

        # Add cast to float64 for non-initializer inputs
        for i, input_name in enumerate(node.input):
            if input_name not in initializer_names:
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[input_name],
                    outputs=[input_name + "_cast"],
                    to=TensorProto.FLOAT
                )
                onnx_model.graph.node.insert(0, cast_node)
                node.input[i] = input_name + "_cast"

    # Introduce a short delay to avoid overwhelming the system
    time.sleep(0.001)  # Slow down the loop slightly

    # Check memory usage and trigger garbage collection if necessary
    check_memory_and_collect()

# Sanitize initializer names
for initializer in onnx_model.graph.initializer:
    initializer.name = sanitize_name(initializer.name)

# Sanitize input/output names in the graph
for input_tensor in onnx_model.graph.input:
    input_tensor.name = sanitize_name(input_tensor.name)
for output_tensor in onnx_model.graph.output:
    output_tensor.name = sanitize_name(output_tensor.name)

# Save the modified ONNX model
onnx.save(onnx_model, "modified_model.onnx")
