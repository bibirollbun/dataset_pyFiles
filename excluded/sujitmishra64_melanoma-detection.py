!pip install torch torchvision albumentations opencv-python scikit-learn seaborn



# ==========================================
# 1. IMPORTS AND SETUP
# ==========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import zipfile
import shutil
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

# Augmentation - First Prize Winner Strategy
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==========================================
# 2. ENHANCED CONFIGURATION
# ==========================================

# Model configuration
IMG_SIZE = 224  # Using winner's optimal size
BATCH_SIZE = 32  # Increased for better gradient estimates
NUM_EPOCHS = 50  # More epochs for better convergence
LEARNING_RATE = 3e-4  # Optimized learning rate
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 2
ACCUMULATION_STEPS = 2  # Gradient accumulation for larger effective batch size

# Dataset paths
ISIC_2024_TRAIN_DIR = '/kaggle/input/isic-2024-challenge/train-image/image'
ISIC_2024_METADATA = '/kaggle/input/isic-2024-challenge/train-metadata.csv'
SIIM_TRAIN_DIR = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train'
SIIM_METADATA = '/kaggle/input/siim-isic-melanoma-classification/train.csv'
SKIN_CANCER_FOLDER = '/kaggle/input/isic-melanoma-v4/skin cancer.v4i.folder/train'  # Updated path

# Output directories
OUTPUT_DIR = 'enhanced_skin_cancer_dataset'
TRAIN_DIR = f'{OUTPUT_DIR}/train'
VALID_DIR = f'{OUTPUT_DIR}/valid'
TEST_DIR = f'{OUTPUT_DIR}/test'

# Class names
CLASS_NAMES = ['Nevus', 'Melanoma']

# ==========================================
# 3. FIRST PRIZE WINNER AUGMENTATION STRATEGY
# ==========================================

# Training augmentations - First Prize Winner Strategy
train_transforms = A.Compose([
    A.Transpose(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=0.7),
    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.0),
        A.ElasticTransform(alpha=3),
    ], p=0.7),
    A.CLAHE(clip_limit=4.0, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, 
                       border_mode=cv2.BORDER_REFLECT_101, p=0.85),
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.CoarseDropout(
        max_holes=1,
        max_height=int(IMG_SIZE * 0.3),
        max_width=int(IMG_SIZE * 0.3),
        num_holes_range=(1, 1),
        p=0.5
    ),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
        p=1.0
    ),
    ToTensorV2()
], p=1.0)

# Validation transforms
valid_transforms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
        p=1.0
    ),
    ToTensorV2()
], p=1.0)

# Test Time Augmentation transforms
tta_transforms = [
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.VerticalFlip(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Transpose(p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
    A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]),
]

# ==========================================
# 4. ENHANCED DATA AGGREGATION
# ==========================================

def extract_and_process_datasets():
    """Extract and process all three datasets into a unified structure"""
    
    print("Creating output directories...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split in ['train', 'valid', 'test']:
        for class_id in ['0', '1']:  # 0: Nevus, 1: Melanoma
            os.makedirs(f'{OUTPUT_DIR}/{split}/{class_id}', exist_ok=True)
    
    all_samples = []
    
    # 1. Process ISIC 2024 Dataset
    print("Processing ISIC 2024 dataset...")
    try:
        metadata_2024 = pd.read_csv(ISIC_2024_METADATA)
        filtered_2024 = metadata_2024[metadata_2024['target'].isin([0, 1])].copy()
        
        # Get more balanced samples - prioritize melanoma
        melanoma_2024 = filtered_2024[filtered_2024['target'] == 1].head(800)
        nevus_2024 = filtered_2024[filtered_2024['target'] == 0].head(800)
        selected_2024 = pd.concat([melanoma_2024, nevus_2024], ignore_index=True)
        
        for idx, row in selected_2024.iterrows():
            img_name = row['isic_id'] + '.jpg'
            img_path = os.path.join(ISIC_2024_TRAIN_DIR, img_name)
            if os.path.exists(img_path):
                all_samples.append({
                    'image_path': img_path,
                    'label': row['target'],
                    'source': 'isic_2024',
                    'image_id': row['isic_id']
                })
        
        print(f"ISIC 2024: {len(selected_2024)} samples processed")
    except Exception as e:
        print(f"Error processing ISIC 2024: {e}")
    
    # 2. Process SIIM-ISIC Dataset
    print("Processing SIIM-ISIC dataset...")
    try:
        metadata_siim = pd.read_csv(SIIM_METADATA)
        filtered_siim = metadata_siim[metadata_siim['target'].isin([0, 1])].copy()
        
        # Get more melanoma samples from SIIM
        melanoma_siim = filtered_siim[filtered_siim['target'] == 1].head(1200)
        nevus_siim = filtered_siim[filtered_siim['target'] == 0].head(400)
        selected_siim = pd.concat([melanoma_siim, nevus_siim], ignore_index=True)
        
        for idx, row in selected_siim.iterrows():
            img_name = row['image_name'] + '.jpg'
            img_path = os.path.join(SIIM_TRAIN_DIR, img_name)
            if os.path.exists(img_path):
                all_samples.append({
                    'image_path': img_path,
                    'label': row['target'],
                    'source': 'siim_isic',
                    'image_id': row['image_name']
                })
        
        print(f"SIIM-ISIC: {len(selected_siim)} samples processed")
    except Exception as e:
        print(f"Error processing SIIM-ISIC: {e}")
    
    # 3. Process Third Dataset (from folder structure)
    print("Processing third dataset from folder...")
    try:
        # Check if the folder exists
        if not os.path.exists(SKIN_CANCER_FOLDER):
            print(f"Warning: Folder {SKIN_CANCER_FOLDER} does not exist")
        else:
            print(f"Found folder: {SKIN_CANCER_FOLDER}")
            
            # Process class 1 (nevus) and class 2 (melanoma)
            class_mapping = {'1': 0, '2': 1}  # 1->nevus(0), 2->melanoma(1)
            
            for original_class, new_class in class_mapping.items():
                class_dir = os.path.join(SKIN_CANCER_FOLDER, original_class)
                print(f"Looking for class folder: {class_dir}")
                
                if os.path.exists(class_dir):
                    image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    print(f"Found {len(image_files)} images in class {original_class}")
                    
                    # Get more samples, especially melanoma
                    if new_class == 1:  # melanoma
                        selected_files = image_files[:1500]  # More melanoma
                    else:  # nevus
                        selected_files = image_files[:800]   # Moderate nevus
                    
                    for img_file in selected_files:
                        img_path = os.path.join(class_dir, img_file)
                        all_samples.append({
                            'image_path': img_path,
                            'label': new_class,
                            'source': 'skin_cancer_v4',
                            'image_id': img_file.split('.')[0]
                        })
                    
                    print(f"Processed {len(selected_files)} images from class {original_class}")
                else:
                    print(f"Class folder {class_dir} does not exist")
                    # Let's explore the actual folder structure
                    if os.path.exists(SKIN_CANCER_FOLDER):
                        print(f"Contents of {SKIN_CANCER_FOLDER}:")
                        for item in os.listdir(SKIN_CANCER_FOLDER):
                            item_path = os.path.join(SKIN_CANCER_FOLDER, item)
                            if os.path.isdir(item_path):
                                print(f"  Directory: {item}")
                            else:
                                print(f"  File: {item}")
        
        print(f"Third dataset: processed successfully")
    except Exception as e:
        print(f"Error processing third dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # Convert to DataFrame and analyze
    df_all = pd.DataFrame(all_samples)
    print(f"\nTotal samples collected: {len(df_all)}")
    if len(df_all) > 0:
        print("Class distribution:")
        print(df_all['label'].value_counts())
        print("Source distribution:")
        print(df_all['source'].value_counts())
    else:
        print("No samples were collected. Please check the dataset paths.")
    
    return df_all


def copy_and_split_data(df_all):
    """Copy images and split into train/val/test with stratification"""
    
    # Stratified split - 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(df_all, test_size=0.3, random_state=42, stratify=df_all['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    print(f"\nDataset splits:")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Copy images to respective directories
    def copy_images_to_split(df, split_name):
        split_dir = os.path.join(OUTPUT_DIR, split_name)
        copied = 0
        
        for idx, row in df.iterrows():
            src_path = row['image_path']
            label = str(row['label'])
            img_id = row['image_id']
            
            # Create unique filename to avoid conflicts
            ext = os.path.splitext(src_path)[1]
            dest_filename = f"{row['source']}_{img_id}{ext}"
            dest_path = os.path.join(split_dir, label, dest_filename)
            
            try:
                shutil.copy2(src_path, dest_path)
                copied += 1
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
        
        print(f"{split_name}: {copied} images copied")
    
    copy_images_to_split(train_df, 'train')
    copy_images_to_split(val_df, 'valid')
    copy_images_to_split(test_df, 'test')
    
    return train_df, val_df, test_df

# ==========================================
# 5. ENHANCED DATASET CLASS
# ==========================================

class EnhancedSkinCancerDataset(Dataset):
    def __init__(self, data_dir, transform=None, use_mixup=False, mixup_alpha=0.2):
        self.data_dir = data_dir
        self.transform = transform
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.samples = []
        
        # Load all image paths and labels
        for class_idx, class_name in enumerate(['0', '1']):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx))
        
        print(f"Dataset {data_dir}: {len(self.samples)} samples")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = cv2.imread(img_path)
            if image is None:
                # Fallback to PIL if cv2 fails
                from PIL import Image
                image = Image.open(img_path).convert('RGB')
                image = np.array(image)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            try:
                augmented = self.transform(image=image)
                image = augmented['image']
            except Exception as e:
                print(f"Error in transform for {img_path}: {e}")
                # Create a simple transform as fallback
                image = cv2.resize(image, (224, 224))
                image = image.astype(np.float32) / 255.0
                image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # Always return consistent format - no mixup in __getitem__
        # Mixup will be handled in a custom collate function
        return image, label


# ==========================================
# 6. ADVANCED MODEL ARCHITECTURE
# ==========================================

class AdvancedSkinCancerClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_v2_l', num_classes=2, pretrained=True, dropout_rate=0.3):
        super(AdvancedSkinCancerClassifier, self).__init__()
        
        # Load backbone
        if model_name == 'efficientnet_v2_l':
            self.backbone = models.efficientnet_v2_l(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'convnext_large':  
            self.backbone = models.convnext_large(pretrained=pretrained)
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'swin_v2_b':
            self.backbone = models.swin_v2_b(pretrained=pretrained)
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        
        # Advanced multi-scale feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Advanced classifier with attention
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features * 2, 1024),  # *2 for avg+max pooling
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        if len(features.shape) == 4:  # If backbone returns feature maps
            # Apply attention mechanism
            attention_weights = self.attention(features.mean(dim=[2, 3]))
            features = features * attention_weights.unsqueeze(-1).unsqueeze(-1)
            
            # Global pooling
            avg_pool = self.global_pool(features).flatten(1)
            max_pool = self.global_max_pool(features).flatten(1)
            features = torch.cat([avg_pool, max_pool], dim=1)
        else:  # If backbone returns flattened features
            features = features
            features = torch.cat([features, features], dim=1)  # Duplicate for consistency
        
        # Classification
        output = self.classifier(features)
        return output


def mixup_collate_fn(batch, mixup_alpha=0.2, use_mixup=True):
    """Custom collate function that applies mixup at batch level"""
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    
    if use_mixup and np.random.random() < 0.5:
        # Apply mixup
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)
        
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        mixed_images = lam * images + (1 - lam) * images[indices]
        labels_a = labels
        labels_b = labels[indices]
        
        return mixed_images, labels_a, labels_b, torch.tensor(lam)
    else:
        return images, labels
# ==========================================
# 7. ADVANCED LOSS FUNCTIONS
# ==========================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean() if self.reduce else focal_loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        targets_smooth = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_smooth * (1 - self.smoothing) + self.smoothing / self.num_classes
        loss = (-targets_smooth * log_probs).sum(dim=1).mean()
        return loss

def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==========================================
# 8. ENHANCED TRAINING FUNCTION
# ==========================================

def train_advanced_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    scaler = GradScaler()
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        optimizer.zero_grad()
        
        for batch_idx, batch_data in enumerate(train_loader):
            try:
                if len(batch_data) == 4:  # Mixup batch
                    images, labels_a, labels_b, lam = batch_data
                    images = images.to(device, non_blocking=True)
                    labels_a, labels_b = labels_a.to(device, non_blocking=True), labels_b.to(device, non_blocking=True)
                    lam = lam.item() if isinstance(lam, torch.Tensor) else lam
                    
                    with autocast():
                        outputs = model(images)
                        loss = mixup_loss(criterion, outputs, labels_a, labels_b, lam)
                    
                    # For accuracy calculation with mixup
                    _, predicted = torch.max(outputs.data, 1)
                    total_predictions += labels_a.size(0)
                    correct_predictions += (lam * (predicted == labels_a).float() + 
                                          (1 - lam) * (predicted == labels_b).float()).sum().item()
                else:  # Regular batch
                    images, labels = batch_data
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total_predictions += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()
                
                # Gradient accumulation
                loss = loss / ACCUMULATION_STEPS
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                running_loss += loss.item() * ACCUMULATION_STEPS
                
                if batch_idx % 20 == 0:
                    print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item() * ACCUMULATION_STEPS:.4f}')
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                try:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    val_running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        val_loss = val_running_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, 'best_enhanced_model.pth')
            print(f'New best model saved! Val Acc: {best_val_acc:.4f}')
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping check
        if epoch > 20 and val_acc < 0.7:  # If not improving after 20 epochs
            print("Performance not improving, adjusting learning rate...")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        
        print()
    
    return train_losses, val_losses, train_accs, val_accs


# ==========================================
# 9. ENHANCED EVALUATION WITH TTA
# ==========================================

def evaluate_with_advanced_tta(model, test_loader, tta_transforms):
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.to(device)
            batch_probs = []
            
            # Original prediction
            images_device = images.to(device)
            with autocast():
                outputs = model(images_device)
                probs = F.softmax(outputs, dim=1)
                batch_probs.append(probs.cpu())
            
            # TTA predictions
            for tta_transform in tta_transforms:
                tta_images = []
                for img in images:
                    # Convert tensor back to numpy for albumentations
                    img_np = img.permute(1, 2, 0).numpy()
                    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                    
                    augmented = tta_transform(image=img_np)
                    tta_images.append(augmented['image'])
                
                tta_batch = torch.stack(tta_images).to(device)
                with autocast():
                    outputs = model(tta_batch)
                    probs = F.softmax(outputs, dim=1)
                    batch_probs.append(probs.cpu())
            
            # Average TTA predictions
            avg_probs = torch.stack(batch_probs).mean(dim=0)
            predictions = torch.argmax(avg_probs, dim=1)
            
            all_predictions.extend(predictions.numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(avg_probs[:, 1].numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)




# ==========================================
# 10. MAIN EXECUTION
# ==========================================

def main():
    print("="*80)
    print("ENHANCED SKIN CANCER CLASSIFICATION PIPELINE - TARGET: 92%+ ACCURACY")
    print("="*80)
    
    # Step 1: Extract and process all datasets
    print("Step 1: Processing all three datasets...")
    df_all = extract_and_process_datasets()
    
    # Step 2: Copy and split data
    print("\nStep 2: Copying and splitting data...")
    train_df, val_df, test_df = copy_and_split_data(df_all)
    
    # Step 3: Create enhanced datasets
    print("\nStep 3: Creating enhanced datasets...")
    train_dataset = EnhancedSkinCancerDataset(TRAIN_DIR, transform=train_transforms, use_mixup=False)  # Mixup handled in collate_fn
    val_dataset = EnhancedSkinCancerDataset(VALID_DIR, transform=valid_transforms, use_mixup=False)
    test_dataset = EnhancedSkinCancerDataset(TEST_DIR, transform=valid_transforms, use_mixup=False)
    
    # Calculate class weights for balanced training
    train_labels = [sample[1] for sample in train_dataset.samples]
    if len(train_labels) == 0:
        print("No training samples found! Please check dataset paths.")
        return 0, 0
        
    class_counts = np.bincount(train_labels)
    class_weights = len(train_labels) / (len(class_counts) * class_counts)
    
    # Create weighted sampler for balanced training
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create custom collate function for training with mixup
    def train_collate_fn(batch):
        return mixup_collate_fn(batch, mixup_alpha=0.2, use_mixup=True)
    
    def val_collate_fn(batch):
        return mixup_collate_fn(batch, use_mixup=False)
    
    # Create data loaders with custom collate functions
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        num_workers=2,  # Reduced num_workers to avoid issues
        pin_memory=True, 
        collate_fn=train_collate_fn,
        drop_last=True  # Drop last incomplete batch
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True, 
        collate_fn=val_collate_fn,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True, 
        collate_fn=val_collate_fn,
        drop_last=False
    )
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches") 
    print(f"Test loader: {len(test_loader)} batches")
    
    # Step 4: Initialize advanced model
    print("\nStep 4: Initializing advanced model...")
    model = AdvancedSkinCancerClassifier(
        model_name='efficientnet_v2_l', 
        num_classes=NUM_CLASSES, 
        pretrained=True,
        dropout_rate=0.3
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Step 5: Setup advanced loss and optimization
    print("\nStep 5: Setting up loss functions and optimizers...")
    
    # Combine multiple loss functions
    focal_loss = FocalLoss(alpha=1, gamma=2)
    label_smooth_loss = LabelSmoothingLoss(num_classes=NUM_CLASSES, smoothing=0.1)
    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    
    def combined_criterion(outputs, targets):
        loss1 = focal_loss(outputs, targets)
        loss2 = label_smooth_loss(outputs, targets)
        loss3 = ce_loss(outputs, targets)
        return 0.4 * loss1 + 0.3 * loss2 + 0.3 * loss3
    
    # Advanced optimizer setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Advanced scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Step 6: Train the model
    print("\nStep 6: Training the advanced model...")
    train_losses, val_losses, train_accs, val_accs = train_advanced_model(
        model, train_loader, val_loader, combined_criterion, optimizer, scheduler, NUM_EPOCHS
    )
    
    # Step 7: Load best model and evaluate
    print("\nStep 7: Loading best model and evaluating...")
    checkpoint = torch.load('best_enhanced_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best validation accuracy: {checkpoint['val_acc']:.4f}")
    
    # Step 8: Advanced evaluation with TTA
    print("\nStep 8: Performing advanced evaluation with TTA...")
    test_predictions, test_labels, test_probabilities = evaluate_with_advanced_tta(
        model, test_loader, tta_transforms
    )
    
    # Calculate comprehensive metrics
    test_accuracy = np.mean(test_predictions == test_labels)
    test_auc = roc_auc_score(test_labels, test_probabilities)
    
    print(f"\nFINAL RESULTS:")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, test_predictions, target_names=CLASS_NAMES))
    
    # Step 9: Visualization and analysis
    print("\nStep 9: Creating visualizations...")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Accuracy', color='blue')
    plt.plot(val_accs, label='Val Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # ROC Curve
    plt.subplot(1, 3, 3)
    fpr, tpr, _ = roc_curve(test_labels, test_probabilities)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(test_labels, test_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Step 10: Model ensemble for maximum performance
    print("\nStep 10: Creating model ensemble for maximum performance...")
    
    # Train additional models with different architectures
    ensemble_models = []
    ensemble_names = ['efficientnet_v2_l', 'convnext_large']
    
    for i, model_name in enumerate(ensemble_names):
        if i == 0:  # First model already trained
            ensemble_models.append(model)
            continue
            
        print(f"Training ensemble model {i+1}: {model_name}")
        ensemble_model = AdvancedSkinCancerClassifier(
            model_name=model_name,
            num_classes=NUM_CLASSES,
            pretrained=True,
            dropout_rate=0.3
        ).to(device)
        
        ensemble_optimizer = optim.AdamW(
            ensemble_model.parameters(),
            lr=LEARNING_RATE * 0.8,  # Slightly lower LR
            weight_decay=WEIGHT_DECAY
        )
        
        ensemble_scheduler = CosineAnnealingWarmRestarts(
            ensemble_optimizer, T_0=8, T_mult=2, eta_min=1e-6
        )
        
        # Train for fewer epochs
        _, _, _, _ = train_advanced_model(
            ensemble_model, train_loader, val_loader, 
            combined_criterion, ensemble_optimizer, ensemble_scheduler, 
            NUM_EPOCHS // 2
        )
        
        ensemble_models.append(ensemble_model)
    
    # Ensemble prediction
    print("\nPerforming ensemble prediction...")
    ensemble_predictions = []
    ensemble_probabilities = []
    
    for test_images, test_labels_batch in test_loader:
        batch_ensemble_probs = []
        
        for model_idx, ens_model in enumerate(ensemble_models):
            ens_model.eval()
            with torch.no_grad():
                test_images_device = test_images.to(device)
                with autocast():
                    outputs = ens_model(test_images_device)
                    probs = F.softmax(outputs, dim=1)
                    batch_ensemble_probs.append(probs.cpu())
        
        # Average ensemble predictions
        avg_ensemble_probs = torch.stack(batch_ensemble_probs).mean(dim=0)
        ensemble_predictions.extend(torch.argmax(avg_ensemble_probs, dim=1).numpy())
        ensemble_probabilities.extend(avg_ensemble_probs[:, 1].numpy())
    
    # Final ensemble metrics
    ensemble_accuracy = np.mean(np.array(ensemble_predictions) == test_labels)
    ensemble_auc = roc_auc_score(test_labels, ensemble_probabilities)
    
    print(f"\nENSEMBLE RESULTS:")
    print(f"Ensemble Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
    print(f"Ensemble AUC: {ensemble_auc:.4f}")
    
    # Step 11: Save final results and model
    print("\nStep 11: Saving final results...")
    
    # Save ensemble models
    torch.save({
        'models': [model.state_dict() for model in ensemble_models],
        'model_names': ensemble_names,
        'test_accuracy': ensemble_accuracy,
        'test_auc': ensemble_auc,
        'class_names': CLASS_NAMES
    }, 'final_ensemble_model.pth')
    
    # Save results summary
    results_summary = {
        'single_model_accuracy': test_accuracy,
        'single_model_auc': test_auc,
        'ensemble_accuracy': ensemble_accuracy,
        'ensemble_auc': ensemble_auc,
        'target_achieved': ensemble_accuracy >= 0.92,
        'training_epochs': NUM_EPOCHS,
        'total_samples': len(df_all),
        'class_distribution': df_all['label'].value_counts().to_dict()
    }
    
    # Save to JSON
    import json
    with open('results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "="*80)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Single Model Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Ensemble Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
    print(f"Target 92% Achieved: {'✓ YES' if ensemble_accuracy >= 0.92 else '✗ NO'}")
    print(f"AUC Score: {ensemble_auc:.4f}")
    print("="*80)
    
    
    return ensemble_accuracy, ensemble_auc

# ==========================================
# 11. ADDITIONAL UTILITY FUNCTIONS
# ==========================================

def predict_single_image(image_path, ensemble_models, transforms):
    """Predict a single image using the ensemble"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    ensemble_probs = []
    
    for model in ensemble_models:
        model.eval()
        with torch.no_grad():
            # Apply transforms
            augmented = transforms(image=image)
            image_tensor = augmented['image'].unsqueeze(0).to(device)
            
            with autocast():
                output = model(image_tensor)
                prob = F.softmax(output, dim=1)
                ensemble_probs.append(prob.cpu())
    
    # Average predictions
    avg_prob = torch.stack(ensemble_probs).mean(dim=0)
    prediction = torch.argmax(avg_prob, dim=1).item()
    confidence = avg_prob[0][prediction].item()
    
    return CLASS_NAMES[prediction], confidence

def generate_submission(test_loader, ensemble_models, output_file='submission.csv'):
    """Generate submission file for Kaggle"""
    predictions = []
    
    for images, _ in test_loader:
        batch_ensemble_probs = []
        
        for model in ensemble_models:
            model.eval()
            with torch.no_grad():
                images_device = images.to(device)
                with autocast():
                    outputs = model(images_device)
                    probs = F.softmax(outputs, dim=1)
                    batch_ensemble_probs.append(probs.cpu())
        
        avg_probs = torch.stack(batch_ensemble_probs).mean(dim=0)
        batch_predictions = avg_probs[:, 1].numpy()  # Probability of melanoma
        predictions.extend(batch_predictions)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': range(len(predictions)),
        'target': predictions
    })
    
    submission_df.to_csv(output_file, index=False)
    print(f"Submission file saved as {output_file}")

# ==========================================
# 12. EXECUTION
# ==========================================

if __name__ == "__main__":
    try:
        accuracy, auc = main()
        print(f"\nTraining completed successfully!")
        print(f"Final Accuracy: {accuracy:.4f}")
        print(f"Final AUC: {auc:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("GPU memory cleared.")

