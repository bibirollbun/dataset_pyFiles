# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install py7zr



import py7zr
dataset_archive_dir = '/kaggle/input/cifar-10/'
dataset_train = pd.read_csv(dataset_archive_dir + "trainLabels.csv")
dataset_test = pd.read_csv(dataset_archive_dir + "sampleSubmission.csv")

train_archive_dir = dataset_archive_dir + 'train.7z'
test_archive_dir = dataset_archive_dir + 'test.7z'

dataset_output_dir = '/kaggle/working/cifar/'
train_dataset_dir = dataset_output_dir + 'train/'
test_dataset_dir = dataset_output_dir + 'test/'

# os.makedirs(train_dataset_dir, exist_ok=True)
# os.makedirs(test_dataset_dir, exist_ok=True)
os.makedirs(dataset_output_dir, exist_ok=True)

with py7zr.SevenZipFile(train_archive_dir, mode='r') as archive:
    archive.extractall(path=dataset_output_dir)

print(f"Successfully extracted '{train_archive_dir}' to '{dataset_output_dir}'")

with py7zr.SevenZipFile(test_archive_dir, mode='r') as archive:
    archive.extractall(path=dataset_output_dir)

print(f"Successfully extracted '{test_archive_dir}' to '{dataset_output_dir}'")


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 1: Convert images to tensors and scale pixel values to 0–1
transform = transforms.ToTensor()  # Only this is needed

# Step 2: Download and load CIFAR-10 training and test data
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Step 3: Create data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Step 4: Check one batch
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Step 5: Print info about the batch
print("Image batch shape:", images.shape)  # Should be [64, 3, 32, 32]
print("Max pixel value:", images.max().item())  # Should be <= 1.0
print("Min pixel value:", images.min().item())  # Should be >= 0.0



import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 1. Convolutional layer: input=3 channels (RGB), output=32 channels, 3x3 kernel
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        
        # 2. Max pooling layer: 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After conv + pooling, image size goes from 32x32 → 16x16
        # So, flatten size = 32 filters * 16 * 16 = 8192
        self.flatten_size = 32 * 16 * 16

        # 3. Fully connected (dense) layer with 128 units
        self.fc1 = nn.Linear(self.flatten_size, 128)

        # 4. Output layer: 10 units (for 10 classes)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Convolution + ReLU
        x = F.relu(self.conv(x))

        # Max Pooling
        x = self.pool(x)

        # Flatten
        x = x.view(-1, self.flatten_size)

        # Fully connected + ReLU
        x = F.relu(self.fc1(x))

        # Output layer (no softmax here — explained below)
        x = self.fc2(x)

        return x

# Instantiate model
model = SimpleCNN()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Print the model structure
print(model)



import torch
import torch.nn as nn
import torch.optim as optim

# ============================================
# 1. Compile the model
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============================================
# 2. Train the model
# ============================================

num_epochs = 10

# For storing metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(test_loader)
    val_acc = 100 * val_correct / val_total

    # Store metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    # Print progress
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    print(f"Val   Loss: {val_loss:.4f}, Val   Accuracy: {val_acc:.2f}%\n")



import matplotlib.pyplot as plt

# ============================================
# 3. Plot Loss and Accuracy
# ============================================

# Plot Loss
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1,2,2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()



# Set model to evaluation mode
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")



import torch

# ===== Save the trained model =====
torch.save(model.state_dict(), 'cnn_cifar10.pth')
print("✅ Model saved!")

# ===== Load the model later =====
model = SimpleCNN()  # Recreate the model structure
model.load_state_dict(torch.load('cnn_cifar10.pth'))
model.to(device)
model.eval()
print("✅ Model loaded!")

# Get a batch of test images
dataiter = iter(test_loader)  # Correct variable name
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# Run the model
outputs = model(images)

# Get predicted classes
_, predicted = torch.max(outputs, 1)

# Show results for first 10 images
print("Predicted: ", predicted[:10].cpu().numpy())
print("Actual:    ", labels[:10].cpu().numpy())



import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Define transform to convert PIL images to tensors and normalize
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load CIFAR-10 training dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)

# Create a data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=20,
                                           shuffle=False, num_workers=2)

# Get a batch of 20 images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# CIFAR-10 class names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Function to show images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize if needed (for normalized images)
    npimg = img.numpy()
    plt.figure(figsize=(12,6))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

# Show the images
imshow(torchvision.utils.make_grid(images))

# Print class labels
print(' '.join(f'{classes[labels[j]]:10s}' for j in range(20)))




model.eval()
predictions = []

with torch.no_grad():
    for images, _ in test_loader:  # Assuming test_loader yields (images, _) without labels
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())

# Create a DataFrame with two columns: Id and Category (predicted class)
submission_df = pd.DataFrame({
    'Id': list(range(len(predictions))),
    'Category': predictions
})

# Save to CSV
submission_df.to_csv('submission22.csv', index=False)
print("✅ Submission file saved as 'submission.csv'")



import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ===========================================
# 1. Custom Dataset for Unlabeled Images
# ===========================================
class TestImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_name

# ===========================================
# 2. Transform and DataLoader
# ===========================================
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dir = '/kaggle/working/cifar/test'  # ⬅️ Change this to your folder path

test_dataset = TestImageDataset(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ===========================================
# 3. Model Inference
# ===========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)  # ⬅️ Make sure SimpleCNN is defined and loaded with weights
model.eval()

predictions = []
image_ids = [x for x in range(300000)]

with torch.no_grad():
    
    for images, filenames in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        predictions.extend(preds.cpu().numpy())
        

# ===========================================
# 4. Create Submission CSV
# ===========================================
submission = pd.DataFrame({
    'Id': image_ids,
    'Label': predictions
})
submission.to_csv("submission.csv", index=False)
print("✅ Submission file saved as 'submission.csv'")


