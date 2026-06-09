# Forams Classification 2025 - Final Solution
# This notebook implements a multi-view 2D CNN approach for the semi-supervised classification of foraminifera

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Define file paths
BASE_PATH = '/kaggle/input/forams-classification-2025'
labeled_vols_path = f'{BASE_PATH}/volumes/volumes/labelled'
unlabeled_vols_path = f'{BASE_PATH}/volumes/volumes/unlabelled'
labeled_viz_path = f'{BASE_PATH}/visualizations/visualizations/labelled'
unlabeled_viz_path = f'{BASE_PATH}/visualizations/visualizations/unlabelled'

# Read CSV files
labeled_df = pd.read_csv(f'{BASE_PATH}/labelled.csv')
unlabeled_df = pd.read_csv(f'{BASE_PATH}/unlabelled.csv')
sample_submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')

print("Labeled data shape:", labeled_df.shape)
print("Unlabeled data shape:", unlabeled_df.shape)
print("Sample submission shape:", sample_submission.shape)

# Check label distribution
print("\nLabel distribution:")
print(labeled_df['label'].value_counts().sort_index())

# Create output directory
os.makedirs('/kaggle/working/models', exist_ok=True)

# Function to visualize a volume and its slices
def visualize_volume(volume_path, viz_path=None, title=None):
    # Load the volume
    volume = imread(volume_path)
    
    # Get middle slices in each dimension
    slice_x = volume[64, :, :]
    slice_y = volume[:, 64, :]
    slice_z = volume[:, :, 64]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Plot slices
    axes[0].imshow(slice_x, cmap='gray')
    axes[0].set_title('X-Slice (64)')
    
    axes[1].imshow(slice_y, cmap='gray')
    axes[1].set_title('Y-Slice (64)')
    
    axes[2].imshow(slice_z, cmap='gray')
    axes[2].set_title('Z-Slice (64)')
    
    # Plot the visualization if provided
    if viz_path and os.path.exists(viz_path):
        viz_img = plt.imread(viz_path)
        axes[3].imshow(viz_img)
        axes[3].set_title('Visualization')
    else:
        axes[3].axis('off')
    
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Multi-view Dataset class that extracts key 2D slices from the 3D volumes
class ForamMultiViewDataset(Dataset):
    def __init__(self, file_paths, labels=None, transform=None, n_slices=3):
        self.file_paths = file_paths
        self.labels = labels  # None for unlabeled data
        self.transform = transform
        self.n_slices = n_slices  # Number of slices per dimension
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load the 3D volume
        volume_path = self.file_paths[idx]
        volume = imread(volume_path).astype(np.float32)
        
        # Select key slices from each dimension
        slices = []
        
        # Get center slices and slices around 1/4 and 3/4 of each dimension
        dim_size = volume.shape[0]  # Assuming cubic volume
        indices = [dim_size // 4, dim_size // 2, 3 * dim_size // 4]
        
        # Extract slices from different planes (axial, coronal, sagittal)
        for i in indices:
            slices.append(volume[i, :, :])  # xy plane (axial)
            slices.append(volume[:, i, :])  # xz plane (coronal)
            slices.append(volume[:, :, i])  # yz plane (sagittal)
        
        # Convert to tensor and normalize
        slices = [torch.from_numpy(slice).float() / 255.0 for slice in slices]
        
        # Apply transforms if any
        if self.transform:
            slices = [self.transform(slice.unsqueeze(0)).squeeze(0) for slice in slices]
        
        # Stack slices as channels
        multi_view = torch.stack(slices)
        
        # Return volume and label (if available)
        if self.labels is not None:
            return multi_view, self.labels[idx]
        else:
            return multi_view

# Data augmentation classes
class RandomBrightness:
    def __init__(self, factor=0.2):
        self.factor = factor
        
    def __call__(self, x):
        factor = np.random.uniform(1.0 - self.factor, 1.0 + self.factor)
        x = x * factor
        return torch.clamp(x, 0, 1)

class RandomContrast:
    def __init__(self, factor=0.2):
        self.factor = factor
        
    def __call__(self, x):
        factor = np.random.uniform(1.0 - self.factor, 1.0 + self.factor)
        mean = x.mean()
        x = (x - mean) * factor + mean
        return torch.clamp(x, 0, 1)

class RandomGamma:
    def __init__(self, range=(0.7, 1.5)):
        self.range = range
        
    def __call__(self, x):
        gamma = np.random.uniform(self.range[0], self.range[1])
        return torch.pow(x, gamma)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

# Create transform compositions
train_transform = Compose([
    RandomBrightness(factor=0.3),
    RandomContrast(factor=0.3),
    RandomGamma(range=(0.7, 1.3))
])

# No transforms for validation and test
val_transform = None

# 2D CNN model for multi-view processing
class MultiViewForamCNN(nn.Module):
    def __init__(self, num_classes=15, num_views=9):
        super(MultiViewForamCNN, self).__init__()
        
        # Input: num_views x 128 x 128
        self.conv1 = nn.Conv2d(num_views, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # After pool1: 32 x 64 x 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # After pool2: 64 x 32 x 32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # After pool3: 128 x 16 x 16
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        # After pool4: 256 x 8 x 8
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # After global_avg_pool: 256 x 1 x 1
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        features = F.relu(self.fc1(x))
        x = self.dropout(features)
        logits = self.fc2(x)
        
        return logits, features

# Prepare file paths and labels for datasets
def prepare_datasets(labeled_vols_path, unlabeled_vols_path, labeled_df, val_size=0.2):
    # Get file paths for labeled data
    labeled_files = []
    labeled_labels = []
    
    for _, row in labeled_df.iterrows():
        id_num = row['id'].split('_')[-1]
        matching_files = [f for f in os.listdir(labeled_vols_path) if f.startswith(f"labelled_foram_{id_num}_")]
        if matching_files:
            labeled_files.append(os.path.join(labeled_vols_path, matching_files[0]))
            labeled_labels.append(row['label'])
    
    # Split labeled data into train and validation
    train_files, val_files, train_labels, val_labels = train_test_split(
        labeled_files, labeled_labels, test_size=val_size, stratify=labeled_labels, random_state=42
    )
    
    # Get file paths for unlabeled data
    unlabeled_files = [os.path.join(unlabeled_vols_path, f) for f in os.listdir(unlabeled_vols_path) 
                       if f.endswith('.tif')]
    
    return train_files, train_labels, val_files, val_labels, unlabeled_files

# Prepare datasets
train_files, train_labels, val_files, val_labels, unlabeled_files = prepare_datasets(
    labeled_vols_path, unlabeled_vols_path, labeled_df
)

print(f"Training samples: {len(train_files)}")
print(f"Validation samples: {len(val_files)}")
print(f"Unlabeled samples: {len(unlabeled_files)}")

# Create datasets
train_dataset = ForamMultiViewDataset(train_files, train_labels, transform=train_transform)
val_dataset = ForamMultiViewDataset(val_files, val_labels, transform=val_transform)

# Create dataloaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Initialize model
model = MultiViewForamCNN(num_classes=15, num_views=9)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# Training function
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, f1



# Train the model
num_epochs = 150  # Train for more epochs for final submission
best_f1 = 0.0
best_model_path = '/kaggle/working/models/best_model.pt'

print("Starting training with multi-view 2D CNN...")
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
    
    # Print progress
    print(f"Epoch {epoch+1}/{num_epochs}: "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
    
    # Update learning rate
    scheduler.step(val_f1)
    
    # Save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved new best model with F1: {best_f1:.4f}")

# Create dataset for unlabeled data
print("Creating dataset for unlabeled data...")
unlabeled_dataset = ForamMultiViewDataset(unlabeled_files, transform=val_transform)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=16, shuffle=False, num_workers=2)

# Load the best model
print("Loading best model for prediction...")
model.load_state_dict(torch.load(best_model_path))
model.eval()

# Generate predictions for unlabeled data
print("Generating predictions for unlabeled data...")
unlabeled_preds = []
unlabeled_probs = []
unlabeled_ids = []

with torch.no_grad():
    for i, batch in enumerate(unlabeled_loader):
        if i % 100 == 0:
            print(f"Processing batch {i}/{len(unlabeled_loader)}")
        
        # Get file paths for the current batch
        batch_files = unlabeled_files[i*unlabeled_loader.batch_size:min((i+1)*unlabeled_loader.batch_size, len(unlabeled_files))]
        batch_ids = [int(os.path.basename(f).split('_')[1]) for f in batch_files]
        
        # Forward pass
        inputs = batch.to(device)
        outputs, _ = model(inputs)
        
        # Get predictions and probabilities
        probs = F.softmax(outputs, dim=1)
        max_probs, preds = probs.max(1)
        
        # Store predictions, probabilities, and IDs
        unlabeled_preds.extend(preds.cpu().numpy())
        unlabeled_probs.extend(max_probs.cpu().numpy())
        unlabeled_ids.extend(batch_ids)

# Label any low-confidence predictions as 'unknown' (class 14)
confidence_threshold = 0.8
for i in range(len(unlabeled_probs)):
    if unlabeled_probs[i] < confidence_threshold:
        unlabeled_preds[i] = 14  # Assign to unknown class

# Create submission dataframe
submission_df = pd.DataFrame({'id': unlabeled_ids, 'label': unlabeled_preds})

# Sort by ID to match the order in sample_submission
submission_df = submission_df.sort_values('id').reset_index(drop=True)

# Save submission file
submission_path = '/kaggle/working/submission.csv'
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

# Display submission statistics
print("Prediction distribution:")
print(submission_df['label'].value_counts().sort_index())

