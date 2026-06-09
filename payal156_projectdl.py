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



import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
from PIL import Image

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
class Config:
    # Paths (adjust these to your Kaggle paths)
    DATA_DIR = '/kaggle/input/recursion-cellular-image-classification'
    TRAIN_CSV = f'{DATA_DIR}/train.csv'
    TEST_CSV = f'{DATA_DIR}/test.csv'
    
    # Model settings
    MODEL_NAME = 'efficientnet_b3'  # Fast and accurate
    IMG_SIZE = 320  # Reduced from 512 for speed
    BATCH_SIZE = 32  # Adjust based on GPU memory
    EPOCHS = 40  # Reduced for time
    LR = 3e-4
    
    # Training settings
    NUM_WORKERS = 2
    SEED = 42
    NUM_CLASSES = 1108
    
    # Use only HUVEC cell type for speed (you can add more if time permits)
    CELL_TYPES = ['HUVEC']  # Add 'RPE', 'HEPG2', 'U2OS' if you have time
    
    # sirna needs to be converted to numeric labels
    CONVERT_SIRNA = True


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(Config.SEED)

# Dataset class
class CellularDataset(Dataset):
    def __init__(self, df, data_dir, mode='train', transform=None):
        self.df = df
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def load_image(self, row):
        """Load 6-channel image"""
        if self.mode == 'train':
            exp = row['experiment']
            plate = row['plate']
            well = row['well']
            site = row['site']
            path_template = f'{self.data_dir}/train/{exp}/Plate{plate}/{well}_s{site}_w'
        else:
            img_id = row['id_code']
            exp = row['experiment']
            plate = row['plate']
            well = row['well']
            site = 1  # Test images are site 1
            path_template = f'{self.data_dir}/test/{exp}/Plate{plate}/{well}_s{site}_w'
        
        # Load all 6 channels
        channels = []
        for i in range(1, 7):
            img_path = f'{path_template}{i}.png'
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                channels.append(img)
            else:
                # Fallback if file doesn't exist
                channels.append(np.zeros((512, 512), dtype=np.uint8))
        
        # Stack channels and resize
        img = np.stack(channels, axis=-1)
        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        
        return img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.load_image(row)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        if self.transform:
            # Convert to PIL for transforms (handle 6 channels)
            img = torch.from_numpy(img).permute(2, 0, 1)  # C, H, W
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)
        
        if self.mode == 'train':
            label = row['label']
            return img, label
        else:
            return img, row['id_code']



class CellularModel(nn.Module):
    def __init__(self, model_name, num_classes, in_channels=6):
        super().__init__()
        # Load pretrained model
        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=3)
        
        # Modify first conv layer to accept 6 channels
        if hasattr(self.backbone, 'conv_stem'):
            old_conv = self.backbone.conv_stem
            self.backbone.conv_stem = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            # Initialize with average of pretrained weights
            with torch.no_grad():
                self.backbone.conv_stem.weight[:, :3] = old_conv.weight
                self.backbone.conv_stem.weight[:, 3:] = old_conv.weight
        
        # Get number of features
        n_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)  # Remove classifier
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_features, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/len(loader), 'acc': 100.*correct/total})
    
    return running_loss/len(loader), 100.*correct/total

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Validation'):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss/len(loader), 100.*correct/total



def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)
    
    # Extract cell type from experiment column
    train_df['cell_type'] = train_df['experiment'].str.split('-').str[0]
    test_df['cell_type'] = test_df['experiment'].str.split('-').str[0]
    
    # Filter by cell type for speed
    train_df = train_df[train_df['cell_type'].isin(Config.CELL_TYPES)].reset_index(drop=True)
    test_df = test_df[test_df['cell_type'].isin(Config.CELL_TYPES)].reset_index(drop=True)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Cell types in train: {train_df['cell_type'].unique()}")
    
    # Convert sirna labels to numeric (sirna_1 -> 1, sirna_10 -> 10, etc.)
    train_df['sirna_id'] = train_df['sirna'].str.replace('sirna_', '').astype(int)
    
    # Create label mapping (need to map to 0-indexed consecutive integers)
    unique_sirnas = sorted(train_df['sirna_id'].unique())
    sirna_to_label = {sirna: idx for idx, sirna in enumerate(unique_sirnas)}
    train_df['label'] = train_df['sirna_id'].map(sirna_to_label)
    
    print(f"Number of unique sirnas: {len(unique_sirnas)}")
    print(f"Label range: 0 to {train_df['label'].max()}")
    
    # Update NUM_CLASSES based on actual data
    Config.NUM_CLASSES = len(unique_sirnas)
    
    # Split train/val
    train_data, val_data = train_test_split(train_df, test_size=0.15, random_state=Config.SEED, stratify=train_df['label'])
    
    # Create datasets
    train_dataset = CellularDataset(train_data, Config.DATA_DIR, mode='train')
    val_dataset = CellularDataset(val_data, Config.DATA_DIR, mode='train')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                             shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # Create model
    print("Creating model...")
    model = CellularModel(Config.MODEL_NAME, Config.NUM_CLASSES).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    # Training loop
    best_acc = 0
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with accuracy: {best_acc:.2f}%")
    
    # Load best model for inference
    print("\nLoading best model for inference...")
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Inference on test set
    print("Generating predictions...")
    test_dataset = CellularDataset(test_df, Config.DATA_DIR, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, num_workers=Config.NUM_WORKERS)
    
    model.eval()
    predictions = []
    ids = []
    
    with torch.no_grad():
        for imgs, img_ids in tqdm(test_loader, desc='Inference'):
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            
            predictions.extend(preds.cpu().numpy())
            ids.extend(img_ids)
    
    # Convert predictions back to sirna format
    label_to_sirna = {idx: sirna for sirna, idx in sirna_to_label.items()}
    predictions_sirna = [label_to_sirna[pred] for pred in predictions]
    
    # Create submission
    submission = pd.DataFrame({
        'id_code': ids,
        'sirna': predictions_sirna
    })
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission saved to submission.csv")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Sample predictions:")
    print(submission.head(10))

if __name__ == '__main__':
    main()


import pandas as pd

# Load the CSVs
DATA_DIR = '/kaggle/input/recursion-cellular-image-classification'
train_df = pd.read_csv(f'{DATA_DIR}/train.csv')
test_df = pd.read_csv(f'{DATA_DIR}/test.csv')

# Print info about the dataframes
print("TRAIN CSV INFO:")
print(train_df.head())
print("\nTrain columns:", train_df.columns.tolist())
print("Train shape:", train_df.shape)

print("\n" + "="*50)
print("\nTEST CSV INFO:")
print(test_df.head())
print("\nTest columns:", test_df.columns.tolist())
print("Test shape:", test_df.shape)

# Check unique values for some columns
if 'experiment' in train_df.columns:
    print("\nUnique experiments:", train_df['experiment'].unique())
if 'plate' in train_df.columns:
    print("Unique plates:", train_df['plate'].nunique())


import os
import glob

# Check one example to see the file structure
sample = train_df.iloc[0]
exp = sample['experiment']
plate = sample['plate']
well = sample['well']

path = f'/kaggle/input/recursion-cellular-image-classification/train/{exp}/Plate{plate}/'
files = sorted(glob.glob(f'{path}{well}*.png'))
print(f"Files for {well}:")
for f in files[:10]:  # Show first 10
    print(f)


# Recursion Cellular Image Classification - Quick Solution
# Optimized for time constraints (< 10 hours)

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
from PIL import Image

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
class Config:
    # Paths (adjust these to your Kaggle paths)
    DATA_DIR = '/kaggle/input/recursion-cellular-image-classification'
    TRAIN_CSV = f'{DATA_DIR}/train.csv'
    TEST_CSV = f'{DATA_DIR}/test.csv'
    
    # Model settings
    MODEL_NAME = 'efficientnet_b3'  # Fast and accurate
    IMG_SIZE = 320  # Reduced from 512 for speed
    BATCH_SIZE = 32  # Adjust based on GPU memory
    EPOCHS = 40  # Reduced for time
    LR = 3e-4
    
    # Training settings
    NUM_WORKERS = 2
    SEED = 42
    NUM_CLASSES = 1108
    
    # Use only HUVEC cell type for speed (you can add more if time permits)
    CELL_TYPES = ['HUVEC']  # Add 'RPE', 'HEPG2', 'U2OS' if you have time
    
    # sirna needs to be converted to numeric labels
    CONVERT_SIRNA = True

# Set seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(Config.SEED)

# Dataset class
class CellularDataset(Dataset):
    def __init__(self, df, data_dir, mode='train', transform=None):
        self.df = df
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def load_image(self, row):
        """Load 6-channel image"""
        exp = row['experiment']
        plate = row['plate']
        well = row['well']
        
        # Extract site from id_code (format: CELLTYPE-XX_PLATE_WELL_siteN)
        # For train: HEPG2-01_1_B03 -> need to find the image files
        # The site information is in the actual filename, not the CSV
        # We need to try site 1 and site 2
        
        if self.mode == 'train':
            path_template = f'{self.data_dir}/train/{exp}/Plate{plate}/{well}_s1_w'
        else:
            path_template = f'{self.data_dir}/test/{exp}/Plate{plate}/{well}_s1_w'
        
        # Load all 6 channels
        channels = []
        for i in range(1, 7):
            img_path = f'{path_template}{i}.png'
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                channels.append(img)
            else:
                # Fallback if file doesn't exist
                channels.append(np.zeros((512, 512), dtype=np.uint8))
        
        # Stack channels and resize
        img = np.stack(channels, axis=-1)
        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        
        return img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.load_image(row)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        if self.transform:
            # Convert to PIL for transforms (handle 6 channels)
            img = torch.from_numpy(img).permute(2, 0, 1)  # C, H, W
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)
        
        if self.mode == 'train':
            label = row['label']
            return img, label
        else:
            return img, row['id_code']

# Model
class CellularModel(nn.Module):
    def __init__(self, model_name, num_classes, in_channels=6):
        super().__init__()
        # Load pretrained model
        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=3)
        
        # Modify first conv layer to accept 6 channels
        if hasattr(self.backbone, 'conv_stem'):
            old_conv = self.backbone.conv_stem
            self.backbone.conv_stem = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            # Initialize with average of pretrained weights
            with torch.no_grad():
                self.backbone.conv_stem.weight[:, :3] = old_conv.weight
                self.backbone.conv_stem.weight[:, 3:] = old_conv.weight
        
        # Get number of features
        n_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)  # Remove classifier
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_features, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/len(loader), 'acc': 100.*correct/total})
    
    return running_loss/len(loader), 100.*correct/total

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Validation'):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss/len(loader), 100.*correct/total

# Main training pipeline
def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)
    
    # Extract cell type from experiment column
    train_df['cell_type'] = train_df['experiment'].str.split('-').str[0]
    test_df['cell_type'] = test_df['experiment'].str.split('-').str[0]
    
    # Filter by cell type for speed
    train_df = train_df[train_df['cell_type'].isin(Config.CELL_TYPES)].reset_index(drop=True)
    test_df = test_df[test_df['cell_type'].isin(Config.CELL_TYPES)].reset_index(drop=True)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Cell types in train: {train_df['cell_type'].unique()}")
    
    # Convert sirna labels to numeric (sirna_1 -> 1, sirna_10 -> 10, etc.)
    train_df['sirna_id'] = train_df['sirna'].str.replace('sirna_', '').astype(int)
    
    # Create label mapping (need to map to 0-indexed consecutive integers)
    unique_sirnas = sorted(train_df['sirna_id'].unique())
    sirna_to_label = {sirna: idx for idx, sirna in enumerate(unique_sirnas)}
    train_df['label'] = train_df['sirna_id'].map(sirna_to_label)
    
    print(f"Number of unique sirnas: {len(unique_sirnas)}")
    print(f"Label range: 0 to {train_df['label'].max()}")
    
    # Update NUM_CLASSES based on actual data
    Config.NUM_CLASSES = len(unique_sirnas)
    
    # Split train/val
    train_data, val_data = train_test_split(train_df, test_size=0.15, random_state=Config.SEED, stratify=train_df['label'])
    
    # Create datasets
    train_dataset = CellularDataset(train_data, Config.DATA_DIR, mode='train')
    val_dataset = CellularDataset(val_data, Config.DATA_DIR, mode='train')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                             shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # Create model
    print("Creating model...")
    model = CellularModel(Config.MODEL_NAME, Config.NUM_CLASSES).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    # Training loop
    best_acc = 0
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with accuracy: {best_acc:.2f}%")
    
    # Load best model for inference
    print("\nLoading best model for inference...")
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Inference on test set
    print("Generating predictions...")
    test_dataset = CellularDataset(test_df, Config.DATA_DIR, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, num_workers=Config.NUM_WORKERS)
    
    model.eval()
    predictions = []
    ids = []
    
    with torch.no_grad():
        for imgs, img_ids in tqdm(test_loader, desc='Inference'):
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            
            predictions.extend(preds.cpu().numpy())
            ids.extend(img_ids)
    
    # Convert predictions back to sirna format
    label_to_sirna = {idx: sirna for sirna, idx in sirna_to_label.items()}
    predictions_sirna = [label_to_sirna[pred] for pred in predictions]
    
    # Create submission
    submission = pd.DataFrame({
        'id_code': ids,
        'sirna': predictions_sirna
    })
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission saved to submission.csv")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Sample predictions:")
    print(submission.head(10))

if __name__ == '__main__':
    main()




