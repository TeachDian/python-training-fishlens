import torch

# Load your trained PyTorch model from the .pt file
model = torch.load('models\150epoch.pt')  # Load the model
model.eval()  # Set the model to evaluation mode

# Create a dummy input with the correct shape for your model
# Adjust the shape (e.g., 1, 3, 640, 640) to match your model's input
# Replace (1, 3, 640, 640) with the appropriate dimensions for your input
dummy_input = torch.randn(1, 3, 640, 640)  # Adjust input dimensions as needed

# Export the model to ONNX format
torch.onnx.export(
    model, 
    dummy_input, 
    "models/onnx/model.onnx",  # Save path for the ONNX model
    opset_version=11,  # Set the ONNX opset version
    input_names=["input"],  # Define input layer name
    output_names=["output"]  # Define output layer name
)

print("Model successfully converted to ONNX format!")
