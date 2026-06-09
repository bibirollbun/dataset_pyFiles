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


import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import cv2
from PIL import Image

# Define a custom CLAHE transform to enhance contrast (domain-specific)
class CLAHETransform(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        # Convert the PIL image to a numpy array
        img_np = np.array(img)
        # If image is grayscale (2D array), return original image
        if len(img_np.shape) == 2:
            return img
        # Convert from RGB to LAB color space
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        # Apply CLAHE to the L-channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l)
        # Merge the CLAHE enhanced L-channel back with A and B channels
        lab = cv2.merge((cl, a, b))
        # Convert back to RGB color space
        img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_clahe)

# Define training transformations with enhanced data augmentation
train_transform = transforms.Compose([
    CLAHETransform(clip_limit=2.0, tile_grid_size=(8,8)),  # Enhance contrast via CLAHE
    transforms.RandomRotation(degrees=10),                 # Small rotations
    transforms.RandomResizedCrop((224,224), scale=(0.8, 1.0)), # Slight zoom variation
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)), # Random translations & rotations
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),  # Increased brightness/contrast jitter
    transforms.RandomHorizontalFlip(p=0.5),                # Horizontal flipping if acceptable
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Define validation transformations without augmentation
val_transform = transforms.Compose([
    transforms.Resize((224,224)),         # Resize directly to final size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Path to the dataset
dataset_path = '/kaggle/input/congenital-heart-disease'

# Create a full dataset to obtain class names and indices (without transform)
full_dataset = datasets.ImageFolder(dataset_path)
print("Classes:", full_dataset.classes)

# Get the total number of samples and create a list of indices
num_samples = len(full_dataset)
indices = list(range(num_samples))
np.random.shuffle(indices)

# Define the split size (80% training, 20% validation)
split = int(0.8 * num_samples)
train_indices = indices[:split]
val_indices = indices[split:]

# Create training and validation datasets with separate transforms using Subset
train_dataset = Subset(datasets.ImageFolder(dataset_path, transform=train_transform), train_indices)
val_dataset = Subset(datasets.ImageFolder(dataset_path, transform=val_transform), val_indices)

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# Check one batch from the training loader
images, labels = next(iter(train_loader))
print("Batch image shape:", images.shape)
print("Batch labels:", labels)



import timm
import torch
import torch.nn as nn

# Define the number of classes based on your dataset
num_classes = 4

# Load the pre-trained ViT model from timm
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# Replace the final classifier head with one that has the correct number of output classes
model.head = nn.Linear(model.head.in_features, num_classes)

print(model)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


import math
from tqdm.notebook import tqdm  # for progress bars in notebooks
import torch.optim as optim

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 100       # Total number of epochs
warmup_epochs = 5     # Number of epochs for warm-up

# Define a lambda function for the learning rate schedule:
# For the first `warmup_epochs`, linearly increase the LR.
# Then, apply cosine annealing from epoch `warmup_epochs` to `num_epochs`.
def lr_lambda(current_epoch):
    if current_epoch < warmup_epochs:
        # Linear warm-up
        return float(current_epoch + 1) / warmup_epochs
    else:
        # Cosine annealing
        return 0.5 * (1 + math.cos(math.pi * (current_epoch - warmup_epochs) / (num_epochs - warmup_epochs)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    # Create a tqdm progress bar for the training loop
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()          # Zero the parameter gradients
        outputs = model(images)        # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()                # Backpropagation
        optimizer.step()               # Update weights
        
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({'loss': loss.item()})
    
    # Step the scheduler at the end of the epoch
    scheduler.step()
    
    epoch_loss = running_loss / len(train_dataset)
    
    # Validation step
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")



from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Switch to evaluation mode
model.eval()

all_preds = []
all_labels = []

# Loop over the validation set
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert lists to arrays for metric computation
y_true = all_labels
y_pred = all_preds

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


