import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import os 

data_dir = '/kaggle/input/dogs-vs-cats'
print(os.listdir(data_dir))


!unzip -qq /kaggle/input/dogs-vs-cats/train.zip


import matplotlib.pyplot as plt
from PIL import Image
plt.figure(figsize = (15,15))

image =  os.listdir('/kaggle/working/train')

for i in range(36):
    plt.subplot(6,6,i+1)
    img = Image.open(os.path.join('/kaggle/working/train',image[i]))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Image {i+1}")
plt.tight_layout()
plt.show()


import os
import shutil
import pathlib
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define directories
original_dir = pathlib.Path('train')
new_base_dir = pathlib.Path('cats_vs_dogs_small')

# Function to create subsets
def make_subset(subset_name, start_index, end_index):
    for category in ('cat', 'dog'):
        dir = new_base_dir / subset_name / category
        os.makedirs(dir, exist_ok=True)
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copyfile(src=original_dir / fname, dst=dir / fname)

# Create training, validation, and test sets
make_subset("train", start_index=0, end_index=1000)
make_subset("validation", start_index=1000, end_index=1500)
make_subset("test", start_index=1500, end_index=2500)

# Define transformations for image processing
transform = transforms.Compose([
    transforms.Resize((150, 150)),  # Resize images to 150x150
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize([0.5], [0.5])  # Normalize images to [-1,1]
])

# Load datasets
train_dataset = datasets.ImageFolder(root=new_base_dir / "train", transform=transform)
val_dataset = datasets.ImageFolder(root=new_base_dir / "validation", transform=transform)
test_dataset = datasets.ImageFolder(root=new_base_dir / "test", transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Check the dataset
for images, labels in train_loader:
    print(f"Batch shape: {images.shape}, Labels: {labels}")
    break  # Only show first batch



# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.models as models

# # Check for GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define CNN Model
# class CNNModel(nn.Module):
#     def __init__(self):
#         super(CNNModel, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 18 * 18, 512),  # Adjusted for 150x150 input
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 2)  # Binary classification: Cat vs Dog
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = self.fc_layers(x)
#         return x

# # Initialize model
# model = CNNModel().to(device)



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained ResNet18 model
class ResNetModel(nn.Module):
    def __init__(self, num_classes=2):  # Binary classification: Cat vs Dog
        super(ResNetModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        
        # Modify the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Initialize model
model = ResNetModel().to(device)




# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # You can increase this for better results

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("Training Finished!")


# Function to calculate accuracy
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Validate the model
train_acc = evaluate(train_loader)
val_acc = evaluate(val_loader)
test_acc = evaluate(test_loader)

print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Validation Accuracy: {val_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")

