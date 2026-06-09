# This installs a library to easily extract the .7z files
!pip install py7zr

# Import all necessary libraries
import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


import py7zr

# Extract training images
with py7zr.SevenZipFile('/kaggle/input/cifar-10/train.7z', mode='r') as z:
    z.extractall(path='/kaggle/working/')
    
# Extract test images
with py7zr.SevenZipFile('/kaggle/input/cifar-10/test.7z', mode='r') as z:
    z.extractall(path='/kaggle/working/')

print("Extraction complete!")
print("Train images:", len(os.listdir('/kaggle/working/train/')))
print("Test images:", len(os.listdir('/kaggle/working/test/')))


# Load the training labels
train_labels_df = pd.read_csv('/kaggle/input/cifar-10/trainLabels.csv')
# Map the image id to its label (e.g., '1' -> 'frog')
label_dict = dict(zip(train_labels_df['id'], train_labels_df['label']))
le = LabelEncoder()
train_labels_df['label_encoded'] = le.fit_transform(train_labels_df['label'])
label_encoded_dict = dict(zip(train_labels_df['id'], train_labels_df['label_encoded']))

# Define image transformations (preprocessing)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])

class CIFAR10Dataset(Dataset):
    def __init__(self, image_dir, label_dict=None, transform=None):
        self.image_dir = image_dir
        self.image_filenames = os.listdir(image_dir)
        self.label_dict = label_dict
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        # Extract the ID from the filename (e.g., '1.png' -> 1)
        img_id = int(os.path.splitext(img_name)[0])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.label_dict is not None:
            # If it's the training set, get the label
            label = self.label_dict[img_id]
            return image, label
        else:
            # If it's the test set, just return the image and ID
            return image, img_id

# Create the full training dataset
full_train_dataset = CIFAR10Dataset(
    image_dir='/kaggle/working/train/',
    label_dict=label_encoded_dict,
    transform=transform
)

# Let's split the training data into train and validation sets (90%/10%)
train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create DataLoaders to feed data in batches
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")


# Load a pre-trained ResNet18 model and modify it for CIFAR-10 (32x32 images)
model = models.resnet18(weights='IMAGENET1K_V1')

# Modify the first convolutional layer to accept 32x32 images better (smaller kernel & stride)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# Modify the final fully connected layer for our 10 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

model = model.to(device) # Move model to GPU

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# Learning rate scheduler to help convergence
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


num_epochs = 15
train_losses, val_accuracies = [], []

for epoch in range(num_epochs):
    # Training Phase
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation Phase
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    val_accuracy = accuracy_score(all_targets, all_preds)
    val_accuracies.append(val_accuracy)
    
    scheduler.step()
    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Acc: {val_accuracy:.4f}')

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.subplot(1, 2, 2)
plt.plot(val_accuracies)
plt.title('Validation Accuracy')
plt.show()


# Create Dataset and DataLoader for the test set
test_dataset = CIFAR10Dataset(
    image_dir='/kaggle/working/test/',
    label_dict=None, # No labels for test set
    transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Predict on the test set
model.eval()
test_preds = {}
with torch.no_grad():
    for data, img_ids in test_loader:
        data = data.to(device)
        output = model(data)
        _, preds = torch.max(output, 1)
        
        # Map the numerical predictions back to class names (e.g., 0 -> 'airplane')
        pred_classes = le.inverse_transform(preds.cpu().numpy())
        
        for img_id, pred_class in zip(img_ids, pred_classes):
            test_preds[img_id] = pred_class

# Create the submission DataFrame
submission_df = pd.DataFrame.from_dict(test_preds, orient='index').reset_index()
submission_df.columns = ['id', 'label']
submission_df = submission_df.sort_values(by='id') # Ensure IDs are in order
submission_df.to_csv('/kaggle/working/submission.csv', index=False)

print("Submission file created!")
print(submission_df.head())




