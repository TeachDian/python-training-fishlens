import cv2
from ultralytics import YOLO

# Load the YOLOv8 model from .pt file
model_path = "models/yolov8/best.pt"  # Replace with the path to your .pt model
model = YOLO(model_path)

# Open the webcam
cap = cv2.VideoCapture(0)  # Use '0' for default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the desired width and height (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Confidence threshold
confidence_threshold = 0.5

# Webcam video loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run YOLOv8 detection on the frame
    results = model(frame)

    # Check if any detection has confidence above the threshold
    detection_confident = False
    for r in results[0].boxes:
        if r.conf > confidence_threshold:
            detection_confident = True
            break

    # Display "No disease detected" if no confident detection is found
    if not detection_confident:
        cv2.putText(frame, "No disease detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Convert results to OpenCV format for display
    annotated_frame = results[0].plot()

    # Display the frame with detections
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    # Press 'q' to quit the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
