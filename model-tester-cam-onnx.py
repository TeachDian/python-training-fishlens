import cv2
import numpy as np
import onnxruntime as ort

# Load the ONNX model
model_path = "model.onnx"
session = ort.InferenceSession(model_path)

# Define class names
class_names = [
    "Aeromonas Septicemia",
    "Columnaris Disease",
    "Edwardsiella Ictaluri -Bacterial Red Disease-",
    "Epizootic Ulcerative Syndrome -EUS-",
    "Flavobacterium -Bacterial Gill Disease-",
    "Fungal Disease -Saprolegniasis-",
    "Healthy Fish",
    "Ichthyophthirius -White Spots",
    "Parasitic Disease",
    "Streptococcus",
    "Tilapia Lake Virus -TiLV-"
]

# Initialize webcam
cap = cv2.VideoCapture(0)

def preprocess_frame(frame):
    # Resize the frame and normalize it
    resized_frame = cv2.resize(frame, (640, 640))  # Adjust according to model input size
    normalized_frame = resized_frame / 255.0
    input_tensor = np.transpose(normalized_frame, (2, 0, 1))  # Change to CHW format
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
    return input_tensor

def postprocess_output(output):
    # Process the model output (print for debugging)
    print("Model output:", output)  # Debugging line to inspect raw output

    # Handle different output structures
    if isinstance(output, list):
        output = output[0]
    
    # Use argmax to find the predicted class index
    pred_class = np.argmax(output)
    confidence = np.max(output)

    return int(pred_class), float(confidence)  # Ensure proper types

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    input_tensor = preprocess_frame(frame)
    
    # Run inference
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    
    # Postprocess output
    pred_class, confidence = postprocess_output(outputs[0])
    
    # Check if class is valid and above confidence threshold (e.g., 0.5)
    if 0 <= pred_class < len(class_names) and confidence > 0.5:
        label = f"{class_names[pred_class]}: {confidence:.2f}"
    else:
        label = "Unknown Class"
    
    # Display the result
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("ONNX Model Tester", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
