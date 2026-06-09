!pip install --upgrade --force-reinstall --quiet numpy==1.26.4 pandas==2.2.2 scikit-learn==1.5.2


# Import all required libraries
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import pydicom
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# Load the competition data
train_df = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/train.csv')
print(f"Training data shape: {train_df.shape}")
print("\nFirst few rows:")
print(train_df.head())

# Check cancer distribution
print(f"\nCancer distribution:")
print(train_df['cancer'].value_counts())
print(f"Cancer rate: {train_df['cancer'].mean():.4f}")


# Create image paths
def create_image_path(row):
    patient_id = row['patient_id']
    image_id = row['image_id']
    return f"/kaggle/input/rsna-breast-cancer-detection/train_images/{patient_id}/{image_id}.dcm"

train_df['file_path'] = train_df.apply(create_image_path, axis=1)
print(f"Found {train_df['file_path'].apply(os.path.exists).sum()} existing DICOM files")

# Create MLO-CC pairs for each patient
def create_mammography_pairs(df):
    pairs = []
    
    for (patient_id, laterality), group in df.groupby(['patient_id', 'laterality']):
        mlo_images = group[group['view'] == 'MLO']
        cc_images = group[group['view'] == 'CC']
        
        # We need at least one MLO and one CC view for each breast
        if len(mlo_images) > 0 and len(cc_images) > 0:
            mlo_row = mlo_images.iloc[0]
            cc_row = cc_images.iloc[0]
            
            pairs.append({
                'patient_id': patient_id,
                'laterality': laterality,
                'cancer': group['cancer'].max(),  # Use max cancer status
                'age': group['age'].iloc[0],
                'mlo_path': mlo_row['file_path'],
                'cc_path': cc_row['file_path'],
                'mlo_image_id': mlo_row['image_id'],
                'cc_image_id': cc_row['image_id'],
            })
    
    return pd.DataFrame(pairs)

# Create the pairs
pairs_df = create_mammography_pairs(train_df)
print(f"Created {len(pairs_df)} MLO-CC pairs")
print(f"Cancer rate in pairs: {pairs_df['cancer'].mean():.4f}")

# Handle missing values
pairs_df['age'] = pairs_df['age'].fillna(pairs_df['age'].median())
pairs_df['laterality_L'] = (pairs_df['laterality'] == 'L').astype(int)
pairs_df['view_mlo'] = 1  # This is constant for our paired data

print("\nPairs dataframe:")
print(pairs_df.head())


# Dataset class for loading and processing DICOM images
class RSNADataset(Dataset):
    def __init__(self, df, transform=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
    
    def load_dicom(self, file_path):
        try:
            dicom = pydicom.dcmread(file_path)
            data = dicom.pixel_array
            
            # Handle different photometric interpretations
            if hasattr(dicom, 'PhotometricInterpretation') and dicom.PhotometricInterpretation == 'MONOCHROME1':
                data = np.max(data) - data  # Invert if needed
            
            # Normalize to 0-255
            data = data.astype(np.float32)
            data = data - np.min(data)
            data = data / (np.max(data) + 1e-8)
            data = (data * 255).astype(np.uint8)
            
            return data
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a gray image as fallback
            return np.ones((512, 512), dtype=np.uint8) * 128
    
    def apply_windowing(self, image, center=150, width=400):
        """Apply DICOM windowing for better contrast"""
        min_val = max(center - width // 2, 0)
        max_val = min(center + width // 2, 255)
        image = np.clip(image, min_val, max_val)
        image = (image - min_val) / (max_val - min_val + 1e-8)
        return (image * 255).astype(np.uint8)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        try:
            # Load MLO and CC views
            mlo_image = self.load_dicom(row['mlo_path'])
            cc_image = self.load_dicom(row['cc_path'])
            
            # Apply windowing for better visualization
            mlo_image = self.apply_windowing(mlo_image)
            cc_image = self.apply_windowing(cc_image)
            
            # Convert to 3-channel (RGB)
            if len(mlo_image.shape) == 2:
                mlo_image = np.stack([mlo_image] * 3, axis=-1)
            if len(cc_image.shape) == 2:
                cc_image = np.stack([cc_image] * 3, axis=-1)
            
            # Apply transforms (augmentations)
            if self.transform:
                mlo_image = self.transform(image=mlo_image)['image']
                cc_image = self.transform(image=cc_image)['image']
            
            # Metadata features (age, view type, laterality)
            meta_features = torch.tensor([
                row['age'] / 100.0,  # Normalized age
                row['view_mlo'],     # Always 1 for MLO in our pairs
                row['laterality_L']  # 1 for left, 0 for right
            ], dtype=torch.float32)
            
            if self.is_train:
                cancer_label = torch.tensor(row['cancer'], dtype=torch.float32)
                return mlo_image, cc_image, meta_features, cancer_label
            else:
                return mlo_image, cc_image, meta_features, row['patient_id']
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return dummy data if there's an error
            dummy_image = torch.zeros((3, 512, 512), dtype=torch.float32)
            dummy_meta = torch.tensor([0.5, 1, 0], dtype=torch.float32)
            if self.is_train:
                return dummy_image, dummy_image, dummy_meta, torch.tensor(0.0, dtype=torch.float32)
            else:
                return dummy_image, dummy_image, dummy_meta, -1

print("Dataset class defined successfully!")


# Cross-attention module for combining MLO and CC features
class MultiModalCrossAttention(nn.Module):
    def __init__(self, feat_dim=512, num_heads=8):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        
        # Attention mechanisms for both views
        self.mlo_attention = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.cc_attention = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.meta_proj = nn.Linear(3, feat_dim)  # Project metadata to feature dimension
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        
    def forward(self, mlo_feat, cc_feat, meta_feat):
        # Project metadata to same dimension as image features
        meta_proj = self.meta_proj(meta_feat).unsqueeze(1)
        
        # Apply attention: use metadata as query, image features as key/value
        mlo_attended, _ = self.mlo_attention(meta_proj, mlo_feat, mlo_feat)
        cc_attended, _ = self.cc_attention(meta_proj, cc_feat, cc_feat)
        
        # Combine attended features
        combined = self.norm1(mlo_attended + cc_attended)
        combined = self.norm2(combined + meta_proj)
        
        return combined.squeeze(1), mlo_attended.squeeze(1), cc_attended.squeeze(1)

# Main model architecture
class RSNAModel(nn.Module):
    def __init__(self, backbone_name='tf_efficientnet_b0', feat_dim=512):
        super().__init__()
        
        # Separate backbones for MLO and CC views
        self.backbone_mlo = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, global_pool=''
        )
        self.backbone_cc = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, global_pool=''
        )
        
        # Get the output feature size of the backbone
        backbone_out = self._get_backbone_output_size(backbone_name)
        
        # Projection layers to match feature dimensions
        self.mlo_proj = nn.Linear(backbone_out, feat_dim)
        self.cc_proj = nn.Linear(backbone_out, feat_dim)
        
        # Cross-attention module
        self.cross_attention = MultiModalCrossAttention(feat_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output probability between 0-1
        )
    
    def _get_backbone_output_size(self, backbone_name):
        # Return feature size based on backbone type
        if 'b0' in backbone_name:
            return 1280
        elif 'b1' in backbone_name:
            return 1280
        elif 'b2' in backbone_name:
            return 1408
        elif 'b3' in backbone_name:
            return 1536
        elif 'b4' in backbone_name:
            return 1792
        else:
            return 1280  # Default for efficientnet_b0
        
    def forward(self, mlo_images, cc_images, meta_features):
        batch_size = mlo_images.shape[0]
        
        # Extract features from MLO view
        mlo_feat = self.backbone_mlo(mlo_images)
        mlo_feat = F.adaptive_avg_pool2d(mlo_feat, (1, 1)).view(batch_size, -1)
        mlo_feat = self.mlo_proj(mlo_feat)
        
        # Extract features from CC view
        cc_feat = self.backbone_cc(cc_images)
        cc_feat = F.adaptive_avg_pool2d(cc_feat, (1, 1)).view(batch_size, -1)
        cc_feat = self.cc_proj(cc_feat)
        
        # Apply cross-modal attention
        combined_feat, _, _ = self.cross_attention(
            mlo_feat.unsqueeze(1), 
            cc_feat.unsqueeze(1), 
            meta_features
        )
        
        # Final classification
        cancer_pred = self.classifier(combined_feat)
        return cancer_pred

print("Model architecture defined successfully!")


# Data augmentation and preprocessing transforms
def get_transforms():
    # Training transforms with augmentation
    train_transform = A.Compose([
        A.Resize(512, 512),  # Resize to 512x512
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
        A.RandomRotate90(p=0.5),  # Random 90-degree rotations
        A.ShiftScaleRotate(
            shift_limit=0.05,  # Small shifts
            scale_limit=0.05,  # Small scaling
            rotate_limit=10,   # Small rotations
            p=0.5
        ),
        # Mild blur augmentation
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=0.5),
            A.MotionBlur(blur_limit=3, p=0.5),
        ], p=0.3),
        # Normalize for ImageNet pretrained models
        A.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),  # Convert to tensor
    ])
    
    # Validation transforms (no augmentation)
    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform

print("Data transforms defined successfully!")


# Split data into training and validation sets
train_data, val_data = train_test_split(
    pairs_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=pairs_df['cancer']  # Maintain cancer distribution
)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Training cancer rate: {train_data['cancer'].mean():.4f}")
print(f"Validation cancer rate: {val_data['cancer'].mean():.4f}")

# Use smaller subsets for initial testing (remove these lines for full training)
train_data = train_data.sample(min(200, len(train_data)), random_state=42)
val_data = val_data.sample(min(50, len(val_data)), random_state=42)

print(f"\nUsing smaller subsets for testing:")
print(f"Training: {len(train_data)} samples")
print(f"Validation: {len(val_data)} samples")

# Create datasets and data loaders
train_transform, val_transform = get_transforms()

train_dataset = RSNADataset(train_data, transform=train_transform, is_train=True)
val_dataset = RSNADataset(val_data, transform=val_transform, is_train=True)

# Use num_workers=0 to avoid multiprocessing issues in Kaggle
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

print("Data loaders created successfully!")


# Training function for one epoch
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    batch_count = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, batch in enumerate(progress_bar):
        try:
            mlo_imgs, cc_imgs, meta, targets = batch
            mlo_imgs = mlo_imgs.to(device)
            cc_imgs = cc_imgs.to(device)
            meta = meta.to(device)
            targets = targets.to(device)
            
            # Skip batches with invalid data
            if torch.isnan(mlo_imgs).any() or torch.isnan(cc_imgs).any():
                continue
                
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(mlo_imgs, cc_imgs, meta)
            loss = criterion(outputs.squeeze(), targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            # Calculate metrics
            running_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            batch_count += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{(predicted == targets).float().mean().item():.4f}'
            })
            
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {e}")
            continue
    
    if batch_count == 0:
        return 0.0, 0.0
        
    epoch_loss = running_loss / batch_count
    epoch_acc = correct_predictions / total_samples
    
    return epoch_loss, epoch_acc

# Validation function for one epoch
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    batch_count = 0
    
    progress_bar = tqdm(dataloader, desc="Validation")
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            try:
                mlo_imgs, cc_imgs, meta, targets = batch
                mlo_imgs = mlo_imgs.to(device)
                cc_imgs = cc_imgs.to(device)
                meta = meta.to(device)
                targets = targets.to(device)
                
                if torch.isnan(mlo_imgs).any() or torch.isnan(cc_imgs).any():
                    continue
                    
                outputs = model(mlo_imgs, cc_imgs, meta)
                loss = criterion(outputs.squeeze(), targets)
                
                running_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)
                batch_count += 1
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{(predicted == targets).float().mean().item():.4f}'
                })
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    if batch_count == 0:
        return 0.0, 0.0
        
    epoch_loss = running_loss / batch_count
    epoch_acc = correct_predictions / total_samples
    
    return epoch_loss, epoch_acc

print("Training functions defined successfully!")


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model
model = RSNAModel('tf_efficientnet_b0', feat_dim=512).to(device)
print("Model initialized!")

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

print("Optimizer and scheduler defined!")

# Training loop
best_val_acc = 0.0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("\nStarting training...")
for epoch in range(5):  # Train for 5 epochs initially
    print(f'\nEpoch {epoch+1}/5')
    print('-' * 50)
    
    # Training phase
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    
    # Validation phase
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    
    # Update learning rate
    scheduler.step()
    
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
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'ğŸš€ New best model saved! Val Acc: {val_acc:.4f}')

print("\nTraining completed!")


# Plot training history
plt.figure(figsize=(15, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
plt.plot(val_losses, 'r-', label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid(True, alpha=0.3)

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(train_accs, 'b-', label='Train Accuracy', linewidth=2)
plt.plot(val_accs, 'r-', label='Val Accuracy', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Best validation accuracy: {best_val_acc:.4f}")


# Evaluation function
def evaluate_model(model, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    val_transform = get_transforms()[1]
    val_dataset = RSNADataset(val_data, transform=val_transform, is_train=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluation"):
            try:
                mlo_imgs, cc_imgs, meta, targets = batch
                mlo_imgs = mlo_imgs.to(device)
                cc_imgs = cc_imgs.to(device)
                meta = meta.to(device)
                targets = targets.to(device)
                
                if torch.isnan(mlo_imgs).any() or torch.isnan(cc_imgs).any():
                    continue
                    
                outputs = model(mlo_imgs, cc_imgs, meta)
                all_probs.extend(outputs.cpu().numpy())
                all_preds.extend((outputs.squeeze() > 0.5).float().cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
            except Exception as e:
                print(f"Evaluation error: {e}")
                continue
    
    all_probs = np.array(all_probs).flatten()
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Check if we have both classes
    unique_classes = np.unique(all_targets)
    print(f"Unique classes in validation set: {unique_classes}")
    print(f"Class distribution: {np.bincount(all_targets.astype(int))}")
    
    # Calculate metrics (only if we have both classes)
    if len(unique_classes) < 2:
        print("âš ï¸�  Warning: Only one class present in validation set. Cannot compute AUC.")
        auc = 0.0
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
    else:
        auc = roc_auc_score(all_targets, all_probs)
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    print(f"\nğŸ“Š Final Evaluation Results:")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Only plot confusion matrix if we have predictions
    if len(all_targets) > 0:
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(all_targets, all_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Cancer', 'Cancer'],
                    yticklabels=['No Cancer', 'Cancer'])
        plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    return auc, accuracy, all_probs, all_targets

# Load best model and evaluate
from sklearn.metrics import precision_score, recall_score, f1_score

model = RSNAModel('tf_efficientnet_b0', feat_dim=512).to(device)
model.load_state_dict(torch.load('best_model.pth'))

auc, accuracy, probs, targets = evaluate_model(model, device)


# Save the complete model for future use
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'backbone': 'tf_efficientnet_b0',
        'feat_dim': 512
    },
    'metrics': {
        'auc': auc,
        'accuracy': accuracy
    },
    'class_names': ['No Cancer', 'Cancer']
}, 'rsna_mammo_final_model.pth')

print("âœ… Model saved successfully as 'rsna_mammo_final_model.pth'")

# List all saved files
print("\nğŸ“� Saved files:")
!ls -lh *.pth




