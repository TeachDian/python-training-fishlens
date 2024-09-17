import torch
import cv2
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import numpy as np

# Define the class names based on your dataset
class_names = [
    "Aeromonas Septicemia",
    "Columnaris Disease",
    "Edwardsiella Ictaluri -Bacterial Red Disease-",
    "Epizootic Ulcerative Syndrome -EUS-",
    "Flavobacterium -Bacterial Gill Disease",
    "Fungal Disease -Saprolegniasis-",
    "Healthy Fish",
    "Ichthyophthirius -White Spots-",
    "Parasitic Disease",
    "Streptococcus",
    "Tilapia Lake Virus -TiLV-"
]

# Load the pre-trained EfficientNet model and adjust the last layer for 11 classes
model = EfficientNet.from_name('efficientnet-b0')
num_classes = 11  # Set this to match the number of classes in your trained model
model._fc = torch.nn.Linear(model._fc.in_features, num_classes)

# Load your trained model weights
model.load_state_dict(torch.load('runs/detect/efficientnet/model_250.pt'))

# Set the model to evaluation mode
model.eval()

# Define preprocessing function for webcam frames
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize image to EfficientNet's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    # Check if frame was captured successfully
    if not ret:
        print("Failed to grab frame")
        break

    # Simulate detection or no detection
    # Assume if there's significant movement or objects in the screen, we treat it as detection
    # For now, we'll treat any non-black frame as a detected object
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray_frame)

    # Preprocess the frame
    input_tensor = preprocess(frame).unsqueeze(0)  # Add batch dimension

    # Perform inference on the frame
    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted class index
    predicted_class_idx = torch.argmax(output, dim=1).item()

    # Get the predicted class name and confidence level
    predicted_class_name = class_names[predicted_class_idx]
    confidence = torch.softmax(output, dim=1)[0, predicted_class_idx].item()

    # Check if there is no significant movement (mean intensity is low) => "No Disease Detected"
    if mean_intensity < 10:  # Threshold for detection (treats as "no object")
        cv2.putText(frame, "No Disease Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Smaller font
    else:
        # Display the frame with prediction and confidence
        if predicted_class_name == "Healthy Fish":
            cv2.putText(frame, f"Prediction: {predicted_class_name} (No Disease Detected)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Smaller font
        else:
            cv2.putText(frame, f"Prediction: {predicted_class_name} (Confidence: {confidence:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Smaller font

        # Draw a bounding box (assuming detection, for simplicity, bounding the entire frame)
        cv2.rectangle(frame, (50, 50), (frame.shape[1] - 50, frame.shape[0] - 50), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Webcam Feed', frame)

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
