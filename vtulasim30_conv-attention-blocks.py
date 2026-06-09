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


from PIL import Image
import matplotlib.pyplot as plt
train = pd.read_csv("/kaggle/input/ai-vs-human-generated-dataset/train.csv")
train['file_name'] = "/kaggle/input/ai-vs-human-generated-dataset/" + train['file_name']
img = Image.open(train['file_name'][0])
plt.imshow(img)
if train['label'][0]==1:
    print('AI')
else:
    print('Human')


test = pd.read_csv("/kaggle/input/ai-vs-human-generated-dataset/test.csv")


test


import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        attention = torch.bmm(query, key)
        attention = torch.softmax(attention, dim=-1)
        value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out

# Residual Block with GroupNorm
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(4, out_channels)  # GroupNorm with 4 groups
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(4, out_channels)  # GroupNorm with 4 groups
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(4, out_channels)  # GroupNorm with 4 groups
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class AIvsHumanDetector(nn.Module):
    def __init__(self):
        super(AIvsHumanDetector, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(4, 32)
        self.relu = nn.ReLU(inplace=True)

        # Residual Blocks
        self.res_block1 = ResidualBlock(32, 64, stride=2)
        self.attn1 = SelfAttention(64)  # Add Attention after this block
        self.res_block2 = ResidualBlock(64, 128, stride=2)
        self.attn2 = SelfAttention(128)  # Add another Attention block
        self.res_block3 = ResidualBlock(128, 256, stride=2)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Layer
        self.fc1 = nn.Linear(256, 128) 
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)# 256 features → 2 classes (AI vs Human)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.res_block1(x)
        x = self.attn1(x)  # Apply Self-Attention here
        x = self.res_block2(x)
        x = self.attn2(x)  # Apply Self-Attention here
        x = self.res_block3(x)

        x = self.global_pool(x)  # (Batch, 256, 1, 1)
        x = torch.flatten(x, start_dim=1)  # Convert to 1D
        x = self.fc1(x)  # Fully connected classification
        x = self.fc2(x)
        x = self.fc3(x)
        return x



from sklearn.model_selection import train_test_split
X = train["file_name"].tolist()
y = train["label"].tolist()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.17, random_state=21)



import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch.nn as nn
import torch

# Define Image Dataset
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")  # Load image
        if self.transform:
            image = self.transform(image)  # Apply transformations

        return image, torch.tensor(label, dtype=torch.float32)  # Convert label to tensor

# Define Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),          # Convert to Tensor
    transforms.Normalize([0.5], [0.5])  # Normalize
])

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Use all available CPU cores for data loading
num_workers = os.cpu_count()  
train_dataset = ImageDataset(X_train, y_train, transform=transform)
val_dataset = ImageDataset(X_val, y_val, transform=transform)  # Validation dataset

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers, pin_memory=True)

# Model Initialization with Multi-GPU
model = AIvsHumanDetector()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)  # Wrap model for multiple GPUs

model = model.to(device)

# Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()  # For binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Early Stopping
patience = 2
best_val_loss = float('inf')
epochs_no_improve = 0

# Training Loop with Early Stopping & Learning Rate Scheduler
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation Phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Check for Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epochs.")

    # Reduce learning rate if validation loss plateaus
    scheduler.step(avg_val_loss)

    # Stop training if no improvement for 'patience' epochs
    if epochs_no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}.")
        break



# Create DataLoader for Test Dataset
test_dataset = ImageDataset(X_val, y_val, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=num_workers, pin_memory=True)

# Function to Calculate Accuracy
def evaluate_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True).unsqueeze(1)

            outputs = model(images)
            predictions = torch.sigmoid(outputs)  # Convert logits to probabilities
            predicted_labels = (predictions > 0.5).float()  # Convert to binary labels (0 or 1)
            
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy

# Calculate and Print Accuracy
test_accuracy = evaluate_model(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.2f}%")



import pandas as pd
from tqdm import tqdm
import os
import torch
from PIL import Image

# Ensure model is in evaluation mode
model.eval()

# Define the correct base directory
BASE_DIR = "/kaggle/input/ai-vs-human-generated-dataset/test_data_v2/"

# Fix paths by adding the correct base directory
test['id'] = test['id'].apply(lambda x: os.path.join(BASE_DIR, os.path.basename(x)))

# Check for missing files before processing
missing_files = test[~test['id'].apply(os.path.exists)]
if not missing_files.empty:
    print(f"⚠️ Warning: {len(missing_files)} files are missing!")
    print(missing_files.head())  # Display some missing file paths

# Create a list to store predictions
predictions_list = []

with torch.no_grad():
    for file in tqdm(test['id'], desc="Predicting Labels"):
        if not os.path.exists(file):
            print(f"❌ File not found: {file}")
            predictions_list.append(None)  # Handle missing files gracefully
            continue

        image = Image.open(file).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        output = model(image)
        prediction = torch.sigmoid(output).item()  # Convert logits to probability
        predicted_label = 1 if prediction > 0.5 else 0  # Convert to binary label
        
        predictions_list.append(predicted_label)

# Create Submission DataFrame
submission = pd.DataFrame({
    "id": test['id'],  # Keep ID address unchanged
    "label": predictions_list
})

# Save to CSV
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("✅ Submission file saved as submission.csv")





