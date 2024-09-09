import torch
import torchvision
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchvision import transforms

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained SSD model
model = torchvision.models.detection.ssd300_vgg16(pretrained=True).to(device)

# Modify SSD head for your dataset
num_classes = 2  # Update based on your dataset
in_channels = model.head.classification_head.conv[0].in_channels
model.head.classification_head = torch.nn.Conv2d(in_channels, num_classes * 4, kernel_size=3, padding=1)

# Define a dataset (replace CustomDataset with your actual dataset class)
dataset = CustomDataset(root='path_to_dataset', transform=transforms.ToTensor())
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# Define an optimizer
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
for epoch in range(10):  # Adjust the number of epochs
    model.train()
    running_loss = 0.0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(data_loader)}")

# Save and export the model to ONNX format
dummy_input = torch.randn(1, 3, 300, 300).to(device)
torch.onnx.export(model, dummy_input, "ssd.onnx", opset_version=11)
