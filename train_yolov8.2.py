import torch
import os
from ultralytics import YOLO

def main():
    # Check if CUDA is available and print GPU name
    print(torch.cuda.is_available())  # Should return True if CUDA is available
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))  # Prints the name of your GPU (e.g., NVIDIA RTX 4060)
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Model path (ensure the path to your custom model is correct)
    model_path = "models/150epoch.pt"
    
    if not os.path.exists(model_path):
        print(f"Model path '{model_path}' not found, starting training from scratch...")
        model = YOLO('yolov8n.pt')  # Use base YOLOv8 model if custom model is not found
    else:
        print(f"Loading model from {model_path}...")
        model = YOLO(model_path)  # Load your pre-trained model
    
    # Path to your dataset YAML configuration file
    data_yaml = 'data.yaml'
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset configuration file '{data_yaml}' not found.")

    # Early stopping parameters
    patience = 10  # Number of epochs to wait before stopping if no improvement
    best_loss = float('inf')  # Best validation loss initialized to infinity
    no_improve_count = 0  # Counter for how many epochs validation loss has not improved

    # Training loop with early stopping
    for epoch in range(150):  # Limit to 150 epochs
        print(f"\nStarting epoch {epoch+1}...")

        # Train the model for one epoch
        results = model.train(data=data_yaml, epochs=1, imgsz=640, device=device)

        # Extract validation loss (YOLOv8's train method doesn't return metrics directly, so use results)
        # Extract the validation loss from the results' attribute `results.box.loss`
        val_loss = results.results[0].metrics.box_loss

        print(f"Validation loss for epoch {epoch+1}: {val_loss}")

        # Check if validation loss improved
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve_count = 0  # Reset the counter if we have a new best loss
            print(f"New best validation loss: {val_loss}, model improved!")
            # Save the best model
            model.save(f'models/best_model_epoch_{epoch+1}.pt')
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count} consecutive epochs.")

        # Early stopping condition
        if no_improve_count >= patience:
            print(f"Stopping early at epoch {epoch+1} due to no improvement in validation loss.")
            break

    # Save the final model after training (whether early stopped or not)
    final_model_path = 'models/final_model.pt'
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()
