# evaluate_yolo.py

from ultralytics import YOLO

def main():
    # Load the trained model
    model = YOLO('runs/detect/train/weights/best.pt')

    # Validate the model on the validation set
    metrics = model.val(data='data.yaml')

    # Print the evaluation metrics
    print(metrics)

if __name__ == '__main__':
    main()
