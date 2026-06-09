#!/usr/bin/env python3
"""
Complete APTOS Diabetic Retinopathy Detection Training Script
Optimized for Kaggle Dual T4 GPUs (15GB VRAM each)
Production-ready with state-of-the-art performance
"""

import os
import gc
import cv2
import time
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')


# =============================================================================
# Configuration Class
# =============================================================================

class Config:
    # Dataset paths (Kaggle paths)
    TRAIN_CSV = '/kaggle/input/aptos2019-blindness-detection/train.csv'
    TRAIN_IMAGES_DIR = '/kaggle/input/aptos2019-blindness-detection/train_images'
    TEST_CSV = '/kaggle/input/aptos2019-blindness-detection/test.csv'
    TEST_IMAGES_DIR = '/kaggle/input/aptos2019-blindness-detection/test_images'
    
    # Model configuration
    MODEL_NAME = 'tf_efficientnet_b4_ns'  # Best for medical imaging
    NUM_CLASSES = 5
    IMAGE_SIZE = 512  # Optimal for retinal images
    
    # Training configuration
    BATCH_SIZE = 8  # Optimized for dual T4 15GB
    EPOCHS = 80
    MIN_EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 15
    
    # Optimization
    LEARNING_RATE = 1e-3
    MIN_LR = 1e-7
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP_NORM = 1.0
    
    # Hardware
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4
    PIN_MEMORY = True
    USE_MIXED_PRECISION = True
    
    # Reproducibility
    SEED = 42
    
    # Output paths
    OUTPUT_DIR = '/kaggle/working'
    MODEL_SAVE_PATH = '/kaggle/working/best_model.pth'
    CHECKPOINT_DIR = '/kaggle/working/checkpoints'
    
    # Cross-validation
    N_FOLDS = 5
    FOLD_TO_TRAIN = -1  # Set to -1 to train all folds, or specific fold number
    
    # Advanced techniques
    USE_FOCAL_LOSS = True
    USE_LABEL_SMOOTHING = True
    LABEL_SMOOTHING_ALPHA = 0.1
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2
    USE_CUTMIX = True
    CUTMIX_ALPHA = 1.0
    
    # Ben Graham preprocessing
    USE_BEN_GRAHAM = True
    GAUSSIAN_BLUR_SIGMA = 10



# =============================================================================
# Utility Functions
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device_info():
    """Print GPU information"""
    if torch.cuda.is_available():
        print(f"ğŸš€ Using GPU: {torch.cuda.get_device_name()}")
        print(f"ğŸ”¢ Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    else:
        print("â�Œ No GPU available")

def create_output_dirs():
    """Create necessary output directories"""
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)


# =============================================================================
# Image Preprocessing Functions
# =============================================================================

def crop_image_from_gray(img, tol=7):
    """Crop image to remove black borders"""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
    return img

def ben_graham_preprocessing(image):
    """Apply Ben Graham preprocessing for retinal images"""
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), Config.GAUSSIAN_BLUR_SIGMA), -4, 128)
    return image

def preprocess_image(image_path: str, image_size: int = 512) -> np.ndarray:
    """Enhanced preprocessing for retinal images"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Crop black borders
    img = crop_image_from_gray(img)
    
    # Resize
    img = cv2.resize(img, (image_size, image_size))
    
    # Apply Ben Graham preprocessing
    if Config.USE_BEN_GRAHAM:
        img = ben_graham_preprocessing(img)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return img



# =============================================================================
# Data Augmentation
# =============================================================================

def get_transforms(image_size: int = 512):
    """Advanced augmentation pipeline for retinal images"""
    
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        
        # Geometric transformations
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.1, rotate_limit=15, 
            border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5
        ),
        
        # Optical distortions (common in retinal images)
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        ], p=0.3),
        
        # Color and lighting augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        ], p=0.5),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.MotionBlur(blur_limit=3, p=0.2),
        ], p=0.2),
        
       
         # Dropout variations (using only CoarseDropout which is available)
        A.CoarseDropout(
            max_holes=8, 
            max_height=32, 
            max_width=32, 
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.3
        ),
        
        # Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform



# =============================================================================
# Dataset Class
# =============================================================================

class APTOSDataset(Dataset):
    """Optimized APTOS dataset with advanced preprocessing"""
    
    def __init__(self, df: pd.DataFrame, image_dir: str, transforms=None, is_training: bool = True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms
        self.is_training = is_training
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, f"{row['id_code']}.png")
        
        # Load and preprocess image
        image = preprocess_image(image_path, Config.IMAGE_SIZE)
        
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        
        if self.is_training:
            label = torch.tensor(row['diagnosis'], dtype=torch.long)
            return image, label
        else:
            return image


# =============================================================================
# Advanced Loss Functions
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        log_prob = F.log_softmax(inputs, dim=-1)
        weight = inputs.new_ones(inputs.size()) * self.smoothing / (inputs.size(-1) - 1.)
        weight.scatter_(-1, targets.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss



# =============================================================================
# Mixup and CutMix
# =============================================================================

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2



# =============================================================================
# Model Architecture
# =============================================================================

class APTOSModel(nn.Module):
    """Advanced model with attention mechanisms"""
    
    def __init__(self, model_name: str = 'tf_efficientnet_b4_ns', num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        
        # Backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Squeeze-and-Excitation attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.feature_dim, self.feature_dim // 16, 1),
            nn.ReLU(),
            nn.Conv2d(self.feature_dim // 16, self.feature_dim, 1),
            nn.Sigmoid()
        )
        
        # Classifier head with batch normalization
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.feature_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone.forward_features(x)
        
        # Apply attention if features are 4D
        if len(features.shape) == 4:
            attention_weights = self.attention(features)
            attended_features = features * attention_weights
            # Global average pooling
            attended_features = attended_features.mean(dim=[2, 3])
        else:
            attended_features = features
        
        # Classify
        output = self.classifier(attended_features)
        
        return output



# =============================================================================
# Training Functions
# =============================================================================

def create_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """Create weighted sampler for handling class imbalance"""
    class_counts = df['diagnosis'].value_counts()
    total_samples = len(df)
    
    # Calculate weights (inverse frequency)
    class_weights = {}
    for cls in range(Config.NUM_CLASSES):
        if cls in class_counts:
            class_weights[cls] = total_samples / (Config.NUM_CLASSES * class_counts[cls])
        else:
            class_weights[cls] = 1.0
    
    # Create sample weights
    sample_weights = [class_weights[label] for label in df['diagnosis']]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

class EarlyStopping:
    """Enhanced early stopping with model restoration"""
    
    def __init__(self, patience=15, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif self.mode == 'max':
            if val_score > self.best_score + self.min_delta:
                self.best_score = val_score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if val_score < self.best_score - self.min_delta:
                self.best_score = val_score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True

def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch):
    """Enhanced training epoch with mixup/cutmix"""
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}')
    
    for batch_idx, (images, targets) in progress_bar:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        # Apply mixup or cutmix randomly
        if Config.USE_MIXUP and np.random.rand() < 0.3:
            images, targets_a, targets_b, lam = mixup_data(images, targets, Config.MIXUP_ALPHA)
            mixed = True
        elif Config.USE_CUTMIX and np.random.rand() < 0.3:
            images, targets_a, targets_b, lam = cutmix_data(images, targets, Config.CUTMIX_ALPHA)
            mixed = True
        else:
            mixed = False
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(enabled=Config.USE_MIXED_PRECISION):
            outputs = model(images)
            
            if mixed:
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, targets)
        
        # Backward pass
        if Config.USE_MIXED_PRECISION:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP_NORM)
            optimizer.step()
        
        running_loss += loss.item()
        
        if not mixed:
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    
    if len(all_preds) > 0:
        epoch_qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic')
    else:
        epoch_qwk = 0.0
    
    return epoch_loss, epoch_qwk

def validate_epoch(model, val_loader, criterion, device):
    """Validation epoch with detailed metrics"""
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validation'):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            with autocast(enabled=Config.USE_MIXED_PRECISION):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic')
    epoch_acc = accuracy_score(all_targets, all_preds)
    
    return epoch_loss, epoch_qwk, epoch_acc, all_preds, all_targets



# =============================================================================
# Main Training Function
# =============================================================================

def train_fold(fold: int, train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Train a single fold"""
    print(f"\n{'='*60}")
    print(f"ğŸš€ Training Fold {fold}")
    print(f"{'='*60}")
    
    # Create transforms
    train_transform, val_transform = get_transforms(Config.IMAGE_SIZE)
    
    # Create datasets
    train_dataset = APTOSDataset(train_df, Config.TRAIN_IMAGES_DIR, train_transform, is_training=True)
    val_dataset = APTOSDataset(val_df, Config.TRAIN_IMAGES_DIR, val_transform, is_training=True)
    
    # Create weighted sampler
    weighted_sampler = create_weighted_sampler(train_df)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        sampler=weighted_sampler,
        num_workers=Config.NUM_WORKERS, 
        pin_memory=Config.PIN_MEMORY,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=Config.NUM_WORKERS, 
        pin_memory=Config.PIN_MEMORY
    )
    
    # Initialize model
    model = APTOSModel(Config.MODEL_NAME, Config.NUM_CLASSES, pretrained=True)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"ğŸš€ Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    model = model.to(Config.DEVICE)
    
    # Loss function
    if Config.USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
    elif Config.USE_LABEL_SMOOTHING:
        criterion = LabelSmoothingCrossEntropy(smoothing=Config.LABEL_SMOOTHING_ALPHA)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=Config.MIN_LR)
    
    # Mixed precision scaler
    scaler = GradScaler() if Config.USE_MIXED_PRECISION else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE, min_delta=0.001, mode='max')
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [], 'train_qwk': [], 'val_qwk': [], 'val_acc': [], 'lr': []
    }
    
    best_qwk = -1.0
    best_model_path = f'{Config.CHECKPOINT_DIR}/best_model_fold_{fold}.pth'
    
    print(f"ğŸ“Š Training samples: {len(train_df)}")
    print(f"ğŸ“Š Validation samples: {len(val_df)}")
    print(f"ğŸ�—ï¸� Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    for epoch in range(Config.EPOCHS):
        start_time = time.time()
        
        # Train
        train_loss, train_qwk = train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, Config.DEVICE, epoch)
        
        # Validate
        val_loss, val_qwk, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, Config.DEVICE)
        
        # Step scheduler
        scheduler.step(val_qwk)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_qwk'].append(train_qwk)
        history['val_qwk'].append(val_qwk)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Save best model
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_qwk': best_qwk,
                'fold': fold
            }, best_model_path)
        
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS} - {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Train QWK: {train_qwk:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val QWK: {val_qwk:.4f} | Val Acc: {val_acc:.4f}")
        print(f"LR: {current_lr:.6f} | Best QWK: {best_qwk:.4f}")
        print(f"Early Stopping: {early_stopping.counter}/{early_stopping.patience}")
        
        # Early stopping
        if epoch >= Config.MIN_EPOCHS:
            early_stopping(val_qwk)
            if early_stopping.early_stop:
                print(f"â�¹ï¸� Early stopping at epoch {epoch+1}")
                break
        
        # Memory cleanup
        if epoch % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Final evaluation
    print(f"\nğŸ�¯ Fold {fold} completed!")
    print(f"ğŸ�† Best QWK: {best_qwk:.4f}")
    
    # Load best model for final evaluation
    checkpoint = torch.load(best_model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation
    final_val_loss, final_val_qwk, final_val_acc, final_preds, final_targets = validate_epoch(model, val_loader, criterion, Config.DEVICE)
    
    # Print classification report
    print("\nğŸ“Š Final Classification Report:")
    print(classification_report(final_targets, final_preds, target_names=[f'Class {i}' for i in range(Config.NUM_CLASSES)]))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(final_targets, final_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{Config.OUTPUT_DIR}/confusion_matrix_fold_{fold}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_qwk, history, best_model_path

def plot_training_history(history_list: List[Dict], fold_scores: List[float]):
    """Plot training history for all folds"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics = ['train_loss', 'val_loss', 'train_qwk', 'val_qwk', 'val_acc', 'lr']
    titles = ['Training Loss', 'Validation Loss', 'Training QWK', 'Validation QWK', 'Validation Accuracy', 'Learning Rate']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        for fold_idx, history in enumerate(history_list):
            if metric in history:
                ax.plot(history[metric], label=f'Fold {fold_idx}', alpha=0.7)
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{Config.OUTPUT_DIR}/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot fold scores
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(fold_scores)), fold_scores, alpha=0.7)
    plt.title('Cross-Validation Scores by Fold')
    plt.xlabel('Fold')
    plt.ylabel('QWK Score')
    plt.axhline(y=np.mean(fold_scores), color='r', linestyle='--', label=f'Mean: {np.mean(fold_scores):.4f}')
    plt.axhline(y=np.mean(fold_scores) + np.std(fold_scores), color='orange', linestyle='--', alpha=0.7, label=f'Mean + Std: {np.mean(fold_scores) + np.std(fold_scores):.4f}')
    plt.axhline(y=np.mean(fold_scores) - np.std(fold_scores), color='orange', linestyle='--', alpha=0.7, label=f'Mean - Std: {np.mean(fold_scores) - np.std(fold_scores):.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    for i, score in enumerate(fold_scores):
        plt.text(i, score + 0.005, f'{score:.4f}', ha='center', va='bottom')
    plt.savefig(f'{Config.OUTPUT_DIR}/fold_scores.png', dpi=300, bbox_inches='tight')
    plt.close()



# =============================================================================
# Inference Pipeline
# =============================================================================

def create_ensemble_model(model_paths: List[str]) -> List[APTOSModel]:
    """Create ensemble of models from fold checkpoints"""
    models = []
    
    for path in model_paths:
        model = APTOSModel(Config.MODEL_NAME, Config.NUM_CLASSES, pretrained=False)
        
        # Multi-GPU support for inference
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        checkpoint = torch.load(path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'], weights_only=False)
        model.to(Config.DEVICE)
        model.eval()
        models.append(model)
    
    return models

def predict_ensemble(models: List[APTOSModel], test_loader: DataLoader) -> np.ndarray:
    """Generate ensemble predictions"""
    all_predictions = []
    
    with torch.no_grad():
        for model in models:
            model_predictions = []
            
            for batch in tqdm(test_loader, desc='Predicting'):
                batch = batch.to(Config.DEVICE, non_blocking=True)
                
                with autocast(enabled=Config.USE_MIXED_PRECISION):
                    outputs = model(batch)
                    probabilities = F.softmax(outputs, dim=1)
                
                model_predictions.append(probabilities.cpu().numpy())
            
            all_predictions.append(np.vstack(model_predictions))
    
    # Average predictions across models
    ensemble_predictions = np.mean(all_predictions, axis=0)
    return ensemble_predictions

def create_submission(predictions: np.ndarray, test_df: pd.DataFrame) -> pd.DataFrame:
    """Create submission file"""
    submission = pd.DataFrame({
        'id_code': test_df['id_code'],
        'diagnosis': predictions.argmax(axis=1)
    })
    
    submission.to_csv(f'{Config.OUTPUT_DIR}/submission.csv', index=False)
    print(f"ğŸ’¾ Submission saved to {Config.OUTPUT_DIR}/submission.csv")
    
    return submission


# =============================================================================
# Main Training Pipeline
# =============================================================================

def main():
    """Main training pipeline"""
    print("ğŸš€ Starting APTOS 2019 Diabetic Retinopathy Detection Training")
    print("="*80)
    
    # Set seeds for reproducibility
    set_seed(Config.SEED)
    
    # Get device info
    get_device_info()
    
    # Create output directories
    create_output_dirs()
    
    # Load data
    print("\nğŸ“Š Loading data...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Class distribution:\n{train_df['diagnosis'].value_counts().sort_index()}")
    
    # Cross-validation setup
    skf = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    
    fold_scores = []
    fold_histories = []
    model_paths = []
    
    # Training loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['diagnosis'])):
        if Config.FOLD_TO_TRAIN != -1 and fold != Config.FOLD_TO_TRAIN:
            continue
        
        # Split data
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)
        
        # Train fold
        best_qwk, history, model_path = train_fold(fold, fold_train_df, fold_val_df)
        
        fold_scores.append(best_qwk)
        fold_histories.append(history)
        model_paths.append(model_path)
        
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
    
    # Print final results
    print("\n" + "="*80)
    print("ğŸ�¯ FINAL RESULTS")
    print("="*80)
    
    for i, score in enumerate(fold_scores):
        print(f"Fold {i}: QWK = {score:.4f}")
    
    print(f"\nMean QWK: {np.mean(fold_scores):.4f} Â± {np.std(fold_scores):.4f}")
    print(f"Best Fold: {np.argmax(fold_scores)} (QWK = {np.max(fold_scores):.4f})")
    
    # Plot training history
    if len(fold_histories) > 0:
        plot_training_history(fold_histories, fold_scores)
    
    # Generate test predictions if test data exists
    if os.path.exists(Config.TEST_CSV) and len(model_paths) > 0:
        print("\nğŸ”® Generating test predictions...")
        
        # Create test dataset
        test_transform = get_transforms(Config.IMAGE_SIZE)[1]  # Use validation transform
        test_dataset = APTOSDataset(test_df, Config.TEST_IMAGES_DIR, test_transform, is_training=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY
        )
        
        # Create ensemble and predict
        ensemble_models = create_ensemble_model(model_paths)
        test_predictions = predict_ensemble(ensemble_models, test_loader)
        
        # Create submission
        submission = create_submission(test_predictions, test_df)
        
        # Print prediction distribution
        print(f"Test prediction distribution:\n{pd.Series(test_predictions.argmax(axis=1)).value_counts().sort_index()}")
    
    print("\nâœ… Training completed successfully!")
    print(f"ğŸ“� All outputs saved to: {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()




