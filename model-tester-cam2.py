import cv2
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# Load the model
model = EfficientNet.from_name('efficientnet-b0')
model.load_state_dict(torch.load('runs/detect/efficientnet/model_250.pt'))
model.eval()

# Define preprocessing function
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resizing to EfficientNet's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Preprocess the frame
    input_tensor = preprocess(frame).unsqueeze(0)  # Add batch dimension
    
    # Make predictions using the model
    with torch.no_grad():
        output = model(input_tensor)
    
    # Process output (for classification or detection)
    # Assuming classification, get the predicted class
    predicted_class = torch.argmax(output, dim=1).item()
    
    # Display the frame with prediction (you can customize it)
    cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Webcam Feed', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
