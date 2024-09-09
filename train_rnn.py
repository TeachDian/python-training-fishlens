import torch
import torchvision
from torch.utils.data import DataLoader

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)

# Modify the number of classes (1 class + background)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Train loop with GPU
for epoch in range(10):  # Define number of epochs
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {losses.item()}")

# Save the model after training
torch.save(model.state_dict(), "faster_rcnn.pth")

# Export to ONNX format for Android deployment
dummy_input = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(model, dummy_input, "faster_rcnn.onnx")
