# Cell 1: Setup and Data Exploration
import os
import pandas as pd
import numpy as np
import pydicom
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

# Load training data
train_df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')

print("=== Dataset Overview ===")
print(f"Total training samples: {len(train_df)}")
print(f"\nColumns: {train_df.columns.tolist()}")
print(f"\nAneurysm prevalence: {train_df['Aneurysm Present'].mean():.3f}")

# Check class distribution
vessel_cols = [col for col in train_df.columns if col not in ['SeriesInstanceUID', 'Modality', 'PatientAge', 'PatientSex', 'Aneurysm Present']]
print(f"\n=== Vessel Location Distribution ===")
for col in vessel_cols:
    pos_count = train_df[col].sum()
    print(f"{col}: {pos_count} ({pos_count/len(train_df)*100:.2f}%)")

print(f"\n=== Modality Distribution ===")
print(train_df['Modality'].value_counts())

print(f"\n=== First 3 rows ===")
print(train_df.head(3))


# Cell 2: Understand DICOM data structure
import random

# Get a random series to examine
sample_series = train_df.sample(1).iloc[0]['SeriesInstanceUID']
series_path = f'/kaggle/input/rsna-intracranial-aneurysm-detection/series/{sample_series}'

# Count slices
dcm_files = list(Path(series_path).glob('*.dcm'))
print(f"Sample Series: {sample_series}")
print(f"Number of DICOM slices: {len(dcm_files)}")

# Read one DICOM
sample_dcm = pydicom.dcmread(dcm_files[0], force=True)
print(f"\nImage shape: {sample_dcm.pixel_array.shape}")
print(f"Modality: {sample_dcm.Modality}")

# Check slice distribution across all series
print("\n=== Analyzing slice counts per series (sample of 100) ===")
sample_series_ids = train_df.sample(min(100, len(train_df)))['SeriesInstanceUID'].values
slice_counts = []

for sid in sample_series_ids:
    path = f'/kaggle/input/rsna-intracranial-aneurysm-detection/series/{sid}'
    if os.path.exists(path):
        n_slices = len(list(Path(path).glob('*.dcm')))
        slice_counts.append(n_slices)

print(f"Slice count stats:")
print(f"  Min: {min(slice_counts)}, Max: {max(slice_counts)}")
print(f"  Mean: {np.mean(slice_counts):.1f}, Median: {np.median(slice_counts):.1f}")


# Cell 3: Setup for windowing and preprocessing
import cv2
from typing import List, Tuple

# Define windowing (from 2019 winner approach)
WINDOWS = {
    'brain': (40, 80),
    'subdural': (80, 200), 
    'bone': (600, 2800)
}

def apply_window(image: np.ndarray, window: Tuple[int, int]) -> np.ndarray:
    """Apply CT windowing to image"""
    window_center, window_width = window
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    
    windowed = np.clip(image, img_min, img_max)
    windowed = ((windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    return windowed

def load_dicom_image(dcm_path: str, target_size: int = 256) -> np.ndarray:
    """Load and preprocess single DICOM slice with 3 windows"""
    dcm = pydicom.dcmread(dcm_path, force=True)
    img = dcm.pixel_array.astype(np.float32)
    
    # Apply RescaleSlope and RescaleIntercept if present
    if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
        img = img * dcm.RescaleSlope + dcm.RescaleIntercept
    
    # Create 3-channel image with different windows
    brain_window = apply_window(img, WINDOWS['brain'])
    subdural_window = apply_window(img, WINDOWS['subdural'])
    bone_window = apply_window(img, WINDOWS['bone'])
    
    # Stack as RGB
    img_3ch = np.stack([brain_window, subdural_window, bone_window], axis=-1)
    
    # Resize
    img_resized = cv2.resize(img_3ch, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    return img_resized

# Test the function
test_dcm_path = str(dcm_files[0])
test_img = load_dicom_image(test_dcm_path, target_size=256)
print(f"Preprocessed image shape: {test_img.shape}")
print(f"Preprocessed image dtype: {test_img.dtype}")
print(f"Preprocessed image range: [{test_img.min()}, {test_img.max()}]")

# Visualize
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(test_img[:,:,0], cmap='gray')
axes[0].set_title('Brain Window')
axes[1].imshow(test_img[:,:,1], cmap='gray')
axes[1].set_title('Subdural Window')
axes[2].imshow(test_img[:,:,2], cmap='gray')
axes[2].set_title('Bone Window')
axes[3].imshow(test_img)
axes[3].set_title('3-Channel Combined')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()

print("\n✓ Preprocessing pipeline ready")


# Cell 4: Create stratified train/validation split
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Define label columns
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present'
]

# Create fold column for 5-fold CV (we'll train on fold 0 for speed)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_df['fold'] = -1

# Use 'Aneurysm Present' for stratification
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['Aneurysm Present'])):
    train_df.loc[val_idx, 'fold'] = fold

# For this competition, we'll use fold 0 as validation
train_fold = train_df[train_df['fold'] != 0].reset_index(drop=True)
valid_fold = train_df[train_df['fold'] == 0].reset_index(drop=True)

print(f"Train samples: {len(train_fold)}")
print(f"Valid samples: {len(valid_fold)}")
print(f"\nTrain aneurysm prevalence: {train_fold['Aneurysm Present'].mean():.3f}")
print(f"Valid aneurysm prevalence: {valid_fold['Aneurysm Present'].mean():.3f}")

# Check series paths exist
train_exists = 0
for sid in train_fold['SeriesInstanceUID'].head(10):
    path = f'/kaggle/input/rsna-intracranial-aneurysm-detection/series/{sid}'
    if os.path.exists(path):
        train_exists += 1

print(f"\n✓ Verified {train_exists}/10 sample series paths exist")


# Cell 5: Create efficient PyTorch dataset
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RSNAAneurysmDataset(Dataset):
    def __init__(self, df, label_cols, transform=None, max_slices=64, sample_mode='uniform'):
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.transform = transform
        self.max_slices = max_slices
        self.sample_mode = sample_mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series_id = row['SeriesInstanceUID']
        series_path = f'/kaggle/input/rsna-intracranial-aneurysm-detection/series/{series_id}'
        
        # Get all DICOM files and sort
        dcm_files = sorted(list(Path(series_path).glob('*.dcm')))
        
        # Sample slices if too many
        if len(dcm_files) > self.max_slices:
            if self.sample_mode == 'uniform':
                indices = np.linspace(0, len(dcm_files)-1, self.max_slices, dtype=int)
                dcm_files = [dcm_files[i] for i in indices]
            elif self.sample_mode == 'random':
                indices = sorted(np.random.choice(len(dcm_files), self.max_slices, replace=False))
                dcm_files = [dcm_files[i] for i in indices]
        
        # Load middle slice (2D approach for now)
        middle_idx = len(dcm_files) // 2
        image = load_dicom_image(str(dcm_files[middle_idx]), target_size=256)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Get labels
        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        
        return image, labels

# Define augmentations
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

valid_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Create datasets
train_dataset = RSNAAneurysmDataset(train_fold, LABEL_COLS, transform=train_transform)
valid_dataset = RSNAAneurysmDataset(valid_fold, LABEL_COLS, transform=valid_transform)

# Test dataset
sample_img, sample_labels = train_dataset[0]
print(f"Sample image shape: {sample_img.shape}")
print(f"Sample labels shape: {sample_labels.shape}")
print(f"Sample labels: {sample_labels}")
print(f"\n✓ Dataset creation successful")


# Cell 6: Define model architecture
import torch
import torch.nn as nn
import timm

class AneurysmClassifier(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0_ns', num_classes=14, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AneurysmClassifier(model_name='tf_efficientnet_b0_ns', num_classes=14, pretrained=False)
model = model.to(device)

# Rest of Cell 6...
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Device: {device}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"\n✓ Model architecture ready")

test_input = torch.randn(2, 3, 256, 256).to(device)
test_output = model(test_input)
print(f"Test output shape: {test_output.shape}")


# Cell 7: Training configuration and loss function
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

# Custom weighted loss (Aneurysm Present gets 13x weight as per competition metric)
class WeightedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Last label is "Aneurysm Present" - weight it 13x
        self.weights = torch.ones(14)
        self.weights[-1] = 13.0
        
    def forward(self, logits, targets):
        self.weights = self.weights.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weighted_bce = bce * self.weights.unsqueeze(0)
        return weighted_bce.mean()

# Training configuration
CONFIG = {
    'epochs': 15,
    'batch_size': 32,
    'learning_rate': 3e-4,
    'weight_decay': 1e-5,
    'num_workers': 2,
    'gradient_clip': 1.0,
}

# Setup optimizer and scheduler
criterion = WeightedBCELoss()
optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=CONFIG['num_workers'],
    pin_memory=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=CONFIG['num_workers'],
    pin_memory=True
)

print(f"✓ Training configuration:")
print(f"  Epochs: {CONFIG['epochs']}")
print(f"  Batch size: {CONFIG['batch_size']}")
print(f"  Learning rate: {CONFIG['learning_rate']}")
print(f"  Train batches: {len(train_loader)}")
print(f"  Valid batches: {len(valid_loader)}")
print(f"  Estimated time per epoch: ~{len(train_loader) * 0.5:.1f} seconds")
print(f"  Total training time: ~{CONFIG['epochs'] * len(train_loader) * 0.5 / 60:.1f} minutes")


# Cell 8: Training and validation functions
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [TRAIN]')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / len(loader)

def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Epoch {epoch+1} [VALID]')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate AUC for each label
    aucs = []
    for i in range(14):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            aucs.append(auc)
        except:
            aucs.append(0.5)
    
    # Competition metric: Aneurysm Present weighted 13x
    competition_score = 0.5 * (aucs[-1] + np.mean(aucs[:-1]))
    
    return running_loss / len(loader), aucs, competition_score

print("✓ Training functions ready")
print("\nStarting training in next cell...")
print("\n⚠️ IMPORTANT: This will take ~15 minutes")
print("Monitor GPU usage: watch -n 1 nvidia-smi")


# Cell 10: Fixed data loading with error handling
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_dicom_image_safe(dcm_path: str, target_size: int = 256) -> np.ndarray:
    """Load and preprocess single DICOM slice with error handling"""
    try:
        dcm = pydicom.dcmread(dcm_path, force=True)
        img = dcm.pixel_array.astype(np.float32)
        
        # Check if image is valid
        if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            return None
        
        # Apply RescaleSlope and RescaleIntercept if present
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            img = img * dcm.RescaleSlope + dcm.RescaleIntercept
        
        # Create 3-channel image with different windows
        brain_window = apply_window(img, WINDOWS['brain'])
        subdural_window = apply_window(img, WINDOWS['subdural'])
        bone_window = apply_window(img, WINDOWS['bone'])
        
        # Stack as RGB
        img_3ch = np.stack([brain_window, subdural_window, bone_window], axis=-1)
        
        # Resize
        img_resized = cv2.resize(img_3ch, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        return img_resized
    except Exception as e:
        return None

class RSNAAneurysmDataset(Dataset):
    def __init__(self, df, label_cols, transform=None, max_slices=64, sample_mode='uniform'):
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.transform = transform
        self.max_slices = max_slices
        self.sample_mode = sample_mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series_id = row['SeriesInstanceUID']
        series_path = f'/kaggle/input/rsna-intracranial-aneurysm-detection/series/{series_id}'
        
        # Get all DICOM files and sort
        dcm_files = sorted(list(Path(series_path).glob('*.dcm')))
        
        # Sample slices if too many
        if len(dcm_files) > self.max_slices:
            if self.sample_mode == 'uniform':
                indices = np.linspace(0, len(dcm_files)-1, self.max_slices, dtype=int)
                dcm_files = [dcm_files[i] for i in indices]
        
        # Try to load middle slice, fallback to others if fails
        middle_idx = len(dcm_files) // 2
        image = None
        
        # Try middle first, then nearby slices
        for offset in [0, -5, 5, -10, 10, -20, 20]:
            try_idx = middle_idx + offset
            if 0 <= try_idx < len(dcm_files):
                image = load_dicom_image_safe(str(dcm_files[try_idx]), target_size=256)
                if image is not None:
                    break
        
        # If all failed, create blank image
        if image is None:
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Get labels
        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        
        return image, labels

# Recreate datasets with fixed loader (set num_workers=0 to avoid multiprocessing issues)
train_dataset = RSNAAneurysmDataset(train_fold, LABEL_COLS, transform=train_transform)
valid_dataset = RSNAAneurysmDataset(valid_fold, LABEL_COLS, transform=valid_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=0,  # Changed to 0 to avoid multiprocessing errors
    pin_memory=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=0,  # Changed to 0
    pin_memory=True
)

print("✓ Fixed dataset and dataloaders created")
print(f"  Train batches: {len(train_loader)}")
print(f"  Valid batches: {len(valid_loader)}")


# Cell 9: Run training loop
import time

best_score = 0.0
best_epoch = 0
history = {'train_loss': [], 'valid_loss': [], 'valid_score': []}

print("=" * 60)
print("TRAINING START")
print("=" * 60)

for epoch in range(CONFIG['epochs']):
    start_time = time.time()
    
    # Train
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
    
    # Validate
    valid_loss, aucs, competition_score = validate(model, valid_loader, criterion, device, epoch)
    
    # Update scheduler
    scheduler.step()
    
    # Track history
    history['train_loss'].append(train_loss)
    history['valid_loss'].append(valid_loss)
    history['valid_score'].append(competition_score)
    
    # Print epoch summary
    epoch_time = time.time() - start_time
    print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} - {epoch_time:.1f}s")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Valid Loss: {valid_loss:.4f}")
    print(f"  Competition Score: {competition_score:.4f}")
    print(f"  Aneurysm Present AUC: {aucs[-1]:.4f}")
    print(f"  Mean Vessel AUC: {np.mean(aucs[:-1]):.4f}")
    
    # Save best model
    if competition_score > best_score:
        best_score = competition_score
        best_epoch = epoch + 1
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  ✓ Best model saved!")
    
    print("-" * 60)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Best Score: {best_score:.4f} at Epoch {best_epoch}")
print(f"GPU Time Used: ~{CONFIG['epochs'] * 0.9:.1f} minutes")


# Cell 11: Submission with lazy model loading
import os
import torch
import torch.nn as nn
import timm
import polars as pl
import numpy as np
import cv2
import pydicom
from pathlib import Path
import kaggle_evaluation.rsna_inference_server
import shutil
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define label columns
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present'
]

# Model class definition (lightweight)
class AneurysmClassifier(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0_ns', num_classes=14, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def predict(series_path: str):
    """Predict function with LAZY model loading"""
    global model, device, transform
    
    # Load model ONLY on first prediction
    if 'model' not in globals():
        print("Loading model on first prediction...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AneurysmClassifier(model_name='tf_efficientnet_b0_ns', num_classes=14, pretrained=False)
        model.load_state_dict(torch.load('best_model.pth'))
        model = model.to(device)
        model.eval()
        
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        print("Model loaded successfully")
    
    # Helper functions
    def apply_window(image, window):
        window_center, window_width = window
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        windowed = np.clip(image, img_min, img_max)
        windowed = ((windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        return windowed
    
    def load_dicom_safe(dcm_path, target_size=256):
        try:
            dcm = pydicom.dcmread(dcm_path, force=True)
            img = dcm.pixel_array.astype(np.float32)
            
            if img.size == 0:
                return None
            
            if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                img = img * dcm.RescaleSlope + dcm.RescaleIntercept
            
            brain = apply_window(img, (40, 80))
            subdural = apply_window(img, (80, 200))
            bone = apply_window(img, (600, 2800))
            
            img_3ch = np.stack([brain, subdural, bone], axis=-1)
            img_resized = cv2.resize(img_3ch, (target_size, target_size))
            return img_resized
        except:
            return None
    
    # Get series files
    series_id = os.path.basename(series_path)
    dcm_files = sorted(list(Path(series_path).glob('*.dcm')))
    
    if len(dcm_files) == 0:
        predictions = pl.DataFrame(
            data=[[0.5] * 14],
            schema=LABEL_COLS,
            orient='row',
        )
        shutil.rmtree('/kaggle/shared', ignore_errors=True)
        return predictions
    
    # Load middle slice
    middle_idx = len(dcm_files) // 2
    image = load_dicom_safe(str(dcm_files[middle_idx]), 256)
    
    if image is None:
        image = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Transform and predict
    transformed = transform(image=image)
    img_tensor = transformed['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    predictions = pl.DataFrame(
        data=[probs.tolist()],
        schema=LABEL_COLS,
        orient='row',
    )
    
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    return predictions

# Create server and start IMMEDIATELY
print("Starting inference server...")
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print("Competition mode - serving")
    inference_server.serve()  # Called within seconds
else:
    print("Local mode - testing")
    inference_server.run_local_gateway()
    
    if os.path.exists('/kaggle/working/submission.parquet'):
        import polars as pl
        sub = pl.read_parquet('/kaggle/working/submission.parquet')
        print(f"\nSubmission shape: {sub.shape}")
        print(sub.head())

print("Submission complete")

