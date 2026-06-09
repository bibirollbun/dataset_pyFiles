import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from io import BytesIO

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

import cv2
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Current Device:", device)


# Load metadata
train_metadata = pd.read_csv('/kaggle/input/isic-2024-challenge/train-metadata.csv')
test_metadata = pd.read_csv('/kaggle/input/isic-2024-challenge/test-metadata.csv')

# Display basic information
print("Train metadata shape:", train_metadata.shape)
print("Test metadata shape:", test_metadata.shape)

# Check class distribution
print("\nClass distribution:")
print(train_metadata['target'].value_counts())
print("Percentage of malignant samples: {:.4f}%".format(
    100 * train_metadata['target'].mean()))

# Create stratified train/validation split
train_df, val_df = train_test_split(
    train_metadata,
    test_size=0.2,
    random_state=42,
    stratify=train_metadata['target']  # Ensure balanced class distribution
)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")


# Define transformations using the winner's normalization values
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(
        mean=[0.4815, 0.4578, 0.4082], 
        std=[0.2686, 0.2613, 0.2758], 
        max_pixel_value=255.0,
        p=1.0
    ),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.4815, 0.4578, 0.4082], 
        std=[0.2686, 0.2613, 0.2758], 
        max_pixel_value=255.0,
        p=1.0
    ),
    ToTensorV2(),
])


# Create a custom dataset class similar to the winner's approach
class ISICDataset(Dataset):
    def __init__(self, df, h5_file_path, transform=None):
        self.df = df
        self.h5_file = h5py.File(h5_file_path, mode="r")
        self.isic_ids = df['isic_id'].values
        self.targets = df['target'].values if 'target' in df.columns else np.zeros(len(df))
        self.transform = transform
        
    def __len__(self):
        return len(self.isic_ids)
    
    def __getitem__(self, idx):
        isic_id = self.isic_ids[idx]
        
        # Read image from HDF5 file using BytesIO
        try:
            img = np.array(Image.open(BytesIO(self.h5_file[isic_id][()])))
        except Exception as e:
            print(f"Error loading image {isic_id}: {e}")
            # Return a placeholder black image if there's an error
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        
        # Get label
        target = self.targets[idx]
        
        return {
            'image': img,
            'target': target
        }
    
    def __del__(self):
        # Close the HDF5 file when the dataset is deleted
        self.h5_file.close()

# Create datasets
train_dataset = ISICDataset(
    train_df,
    h5_file_path='/kaggle/input/isic-2024-challenge/train-image.hdf5',
    transform=train_transform
)

val_dataset = ISICDataset(
    val_df,
    h5_file_path='/kaggle/input/isic-2024-challenge/test-image.hdf5',
    transform=val_transform
)


# Create weighted sampler to handle class imbalance
def create_weighted_sampler(metadata_df):
    # Calculate class weights
    class_counts = metadata_df['target'].value_counts()
    weights = 1.0 / class_counts
    sample_weights = metadata_df['target'].map(weights).values
    
    # Create sampler
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

# Create weighted sampler for training
train_sampler = create_weighted_sampler(train_df)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=train_sampler,  # Use weighted sampler
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# Test the dataloader
print("\nTesting dataloader:")
for i, data in enumerate(train_loader):
    if i >= 1:  # Just test one batch
        break
    images = data['image']
    targets = data['target']
    
    print(f"Batch {i+1}:")
    print(f"  Image batch shape: {images.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Targets: {targets.numpy()}")
    
    # Display the image
    plt.figure(figsize=(6, 6))
    img = images[0].permute(1, 2, 0).cpu().numpy()
    img = img * np.array([0.2686, 0.2613, 0.2758]) + np.array([0.4815, 0.4578, 0.4082])
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(f"Target: {targets[0].item()}")
    plt.axis('off')
    plt.show()

print("Dataloader test completed!")



# Import the model architecture
import sys
sys.path.append('/kaggle/input/mobilenet/pytorch/default/1')
from models.build_mobilenet_v4 import mobilenetv4_conv_small as mobilenetv4_small

# Initialize the model for binary classification
try:
    model = mobilenetv4_small(pretrained=False, num_classes=1)
    model = model.to(device)
    print("Model successfully loaded and moved to device:", device)
except Exception as e:
    print(f"Error initializing model: {e}")
    # Fallback to a different model if needed
    model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=1)
    model = model.to(device)



# Calculate class weights for loss function
pos_weight = torch.tensor(
    [train_df['target'].value_counts()[0] / train_df['target'].value_counts()[1]]
).to(device)

# Use weighted BCE loss
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,  # Lower initial learning rate
    weight_decay=1e-4,  # Add weight decay for regularization
    eps=1e-8  # Increase epsilon for numerical stability
)

# Add learning rate scheduler for better convergence
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max', 
    factor=0.1, 
    patience=3, 
    verbose=True
)



def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    progress = tqdm(loader, desc="Training")
    
    for batch in progress:
        images = batch['image'].to(device)
        targets = batch['target'].float().to(device)
        
        # Forward pass
        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item() * images.size(0)
        progress.set_postfix(loss=loss.item())
    
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        progress = tqdm(loader, desc="Validation")
        for batch in progress:
            images = batch['image'].to(device)
            targets = batch['target'].float().to(device)
            
            # Forward pass
            outputs = model(images).squeeze()
            loss = criterion(outputs, targets)
            
            # Store predictions
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            
            running_loss += loss.item() * images.size(0)
            progress.set_postfix(loss=loss.item())
    
    # Calculate AUC
    auc = roc_auc_score(all_targets, all_preds)
    return running_loss / len(loader.dataset), auc


# num_epochs = 30
# best_auc = 0.0
# patience = 5
# patience_counter = 0

# for epoch in range(num_epochs):
#     print(f"\nEpoch {epoch+1}/{num_epochs}")
    
#     # Training
#     train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    
#     # Validation
#     val_loss, val_auc = validate(model, val_loader, criterion, device)
    
#     print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
    
#     # Save best model
#     if val_auc > best_auc:
#         best_auc = val_auc
#         torch.save(model.state_dict(), "best_mobilenetv4.pth")
#         print(f"New best model saved with AUC: {best_auc:.4f}")
#         patience_counter = 0
#     else:
#         patience_counter += 1
#         print(f"No improvement. Patience: {patience_counter}/{patience}")
#         if patience_counter >= patience:
#             print("Early stopping triggered.")
#             break



# Load best model
model.load_state_dict(torch.load('/kaggle/input/mobilenetv4/pytorch/default/1/best_mobilenetv4.pth'))


# Prepare test dataset (no labels)
class ISICTestDataset(Dataset):
    def __init__(self, df, h5_file_path, transform=None):
        self.df = df
        self.h5_file_path = h5_file_path
        self.isic_ids = df['isic_id'].values
        self.transform = transform
        self.h5_file = None

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_file_path, mode="r")
        isic_id = self.isic_ids[idx]
        img = np.array(Image.open(BytesIO(self.h5_file[isic_id][()])))
        if self.transform:
            img = self.transform(image=img)["image"]
        return {'image': img, 'isic_id': isic_id}

# Load test metadata
test_metadata = pd.read_csv('/kaggle/input/isic-2024-challenge/test-metadata.csv')

# Use the same transforms as validation
test_dataset = ISICTestDataset(
    test_metadata,
    h5_file_path='/kaggle/input/isic-2024-challenge/test-image.hdf5',
    transform=val_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)



model.eval()
all_isic_ids = []
all_probs = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Inference"):
        images = batch['image'].to(device)
        isic_ids = batch['isic_id']
        outputs = model(images).squeeze()
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_isic_ids.extend(isic_ids)
        all_probs.extend(probs)



submission = pd.DataFrame({
    'isic_id': all_isic_ids,
    'malignancy': all_probs
})

submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file saved as submission.csv")


