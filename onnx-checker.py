import onnx
from onnx import helper, numpy_helper
from onnx import TensorProto
from tqdm import tqdm  # Import tqdm for the progress bar

# Load the ONNX model
onnx_model = onnx.load("model.onnx")

# Function to clean node names
def sanitize_name(name):
    # Replace invalid characters with an underscore
    new_name = name.replace('/', '_').replace('-', '_')
    # Ensure the name does not start with an underscore or non-alphabetical character
    if not new_name[0].isalpha():
        new_name = 'n' + new_name  # Prepend with 'n' to ensure valid name
    return new_name

# Get the total number of nodes to set up the progress bar
total_nodes = len(onnx_model.graph.node)

# Iterate through the nodes with a progress bar
for node in tqdm(onnx_model.graph.node, desc="Processing Nodes", total=total_nodes):
    # Sanitize the node names
    node.name = sanitize_name(node.name)
    # Sanitize input and output names of each node
    node.input[:] = [sanitize_name(i) for i in node.input]
    node.output[:] = [sanitize_name(o) for o in node.output]

    # Check for Mul operations to fix the data type of initializers and inputs
    if node.op_type == "Mul":
        for i, input_name in enumerate(node.input):
            # Check if the input is an initializer (constant)
            for initializer in onnx_model.graph.initializer:
                if initializer.name == input_name and initializer.data_type == TensorProto.INT64:
                    # Convert int64 data to float64
                    float_data = numpy_helper.to_array(initializer).astype('float64')
                    # Create a new initializer with float64 data
                    new_initializer = numpy_helper.from_array(float_data, name=initializer.name)
                    # Replace the old int64 initializer with the new float64 initializer
                    onnx_model.graph.initializer.remove(initializer)
                    onnx_model.graph.initializer.append(new_initializer)

        # Add a cast to float64 for inputs that are not initializers
        for i, input_name in enumerate(node.input):
            # Check if the input is a tensor and not an initializer
            if input_name not in [init.name for init in onnx_model.graph.initializer]:
                # Create a new Cast node to cast int64 inputs to float64
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[input_name],
                    outputs=[input_name + "_cast"],
                    to=TensorProto.FLOAT  # Cast to float64
                )
                # Insert the new Cast node before the Mul operation
                onnx_model.graph.node.insert(0, cast_node)
                # Update the input of the Mul node to the casted tensor
                node.input[i] = input_name + "_cast"

# Sanitize the initializer names
for initializer in onnx_model.graph.initializer:
    initializer.name = sanitize_name(initializer.name)

# Sanitize input and output names in the graph
for input_tensor in onnx_model.graph.input:
    input_tensor.name = sanitize_name(input_tensor.name)
for output_tensor in onnx_model.graph.output:
    output_tensor.name = sanitize_name(output_tensor.name)

# Save the modified ONNX model
onnx.save(onnx_model, "modified_model.onnx")
