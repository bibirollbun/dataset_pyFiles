import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
from tqdm import tqdm
import gc
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    # Paths
    TRAIN_CSV = '/kaggle/input/grand-xray-slam-division-a/train1.csv'
    TRAIN_IMG_DIR = '/kaggle/input/grand-xray-slam-division-a/train1'
    TEST_CSV = '/kaggle/input/grand-xray-slam-division-a/test1.csv'
    TEST_IMG_DIR = '/kaggle/input/grand-xray-slam-division-a/test1'
    
    # Model checkpoint paths
    CHECKPOINT_DIR = '/kaggle/working/checkpoints'
    BEST_MODEL_PATH = '/kaggle/working/best_model_fold{}.pth'
    EMA_MODEL_PATH = '/kaggle/working/ema_model_fold{}.pth'
    
    # Training parameters
    IMG_SIZE = 512  # Increased from 224 to 512
    BATCH_SIZE = 16  # Reduced due to larger image size
    EPOCHS = 8
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # K-Fold CV
    N_FOLDS = 5
    TRAIN_FOLDS = [0, 1, 2, 3, 4]  # Which folds to train
    
    # Early stopping
    PATIENCE = 3
    
    # EMA
    EMA_DECAY = 0.999
    
    # Mixed precision
    USE_AMP = True
    
    # Label columns for 14 thoracic conditions
    LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
              'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
              'Lung Opacity', 'No Finding', 'Pleural Effusion', 
              'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

# Create checkpoint directory
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

# Exponential Moving Average helper
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# Custom Dataset with proper dunder methods
class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, is_test=False):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_id'] if 'image_id' in self.df.columns else self.df.iloc[idx][0]
        img_path = os.path.join(self.img_dir, str(img_name))
        
        # Handle different image extensions
        if not os.path.exists(img_path):
            for ext in ['.jpg', '.jpeg', '.png', '.dcm']:
                if os.path.exists(img_path + ext):
                    img_path = img_path + ext
                    break
        
        # Load image and ensure RGB (handle grayscale)
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, img_name
        else:
            labels = self.df.iloc[idx][Config.LABELS].values.astype('float32')
            return image, torch.tensor(labels, dtype=torch.float32)

# Data Augmentation and Transforms
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(7),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

# CBAM Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# Model Definition with CBAM and Label Attention
class EfficientNetB0Model(nn.Module):
    def __init__(self, num_classes=14, pretrained=True, use_attention=True):
        super(EfficientNetB0Model, self).__init__()
        # Load pretrained EfficientNet-B0 with updated syntax
        if pretrained:
            self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Get feature dimension
        in_features = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Add CBAM attention
        self.use_attention = use_attention
        if use_attention:
            self.cbam = CBAM(in_features, reduction=16)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Label attention head
        self.label_embeddings = nn.Parameter(torch.randn(num_classes, in_features))
        self.attention = nn.MultiheadAttention(in_features, num_heads=4, batch_first=True)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone.features(x)
        
        # Apply CBAM if enabled
        if self.use_attention:
            features = self.cbam(features)
        
        # Global average pooling
        pooled = self.gap(features).flatten(1)
        
        # Label attention
        batch_size = pooled.size(0)
        label_emb = self.label_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        pooled_expanded = pooled.unsqueeze(1)
        attended, _ = self.attention(label_emb, pooled_expanded, pooled_expanded)
        attended = attended.mean(dim=1)
        
        # Combine pooled and attended features
        combined = pooled + attended
        
        # Final classification
        return self.classifier(combined)

# Calculate pos_weight for class imbalance
def calculate_pos_weight(train_df, labels):
    pos_counts = train_df[labels].sum(axis=0)
    neg_counts = len(train_df) - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-5)  # Add epsilon to avoid division by zero
    return torch.tensor(pos_weight.values, dtype=torch.float32)

# Training function with mixed precision
def train_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, ema, device):
    model.train()
    running_loss = 0.0
    
    with tqdm(dataloader, desc='Training', leave=False) as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            if Config.USE_AMP:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Update EMA
            if ema is not None:
                ema.update()
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step()
            
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# Validation function with per-label AUC
def validate_epoch(model, dataloader, criterion, device):
    from sklearn.metrics import roc_auc_score
    
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        with tqdm(dataloader, desc='Validation', leave=False) as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                if Config.USE_AMP:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                # Store predictions and labels
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())
                
                running_loss += loss.item() * images.size(0)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Calculate per-label and macro AUC
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    label_aucs = []
    for i, label_name in enumerate(Config.LABELS):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            label_aucs.append(auc)
        except:
            label_aucs.append(0.5)  # If AUC can't be calculated
    
    macro_auc = np.mean(label_aucs)
    
    return epoch_loss, macro_auc, label_aucs

# Test inference with TTA
def generate_predictions(model, dataloader, device, tta=True):
    model.eval()
    predictions = []
    image_ids = []
    
    with torch.no_grad():
        for images, img_names in tqdm(dataloader, desc='Generating predictions'):
            images = images.to(device)
            
            if Config.USE_AMP:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            
            probs = torch.sigmoid(outputs)
            
            # TTA: horizontal flip
            if tta:
                images_flipped = torch.flip(images, dims=[3])
                if Config.USE_AMP:
                    with torch.cuda.amp.autocast():
                        outputs_flipped = model(images_flipped)
                else:
                    outputs_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(outputs_flipped)
                probs = (probs + probs_flipped) / 2
            
            predictions.append(probs.cpu().numpy())
            image_ids.extend(img_names)
    
    predictions = np.vstack(predictions)
    return predictions, image_ids

# Main training pipeline
def train_fold(fold, train_df, val_df):
    print(f"\n{'='*60}")
    print(f"Training Fold {fold}")
    print(f"{'='*60}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    
    # Create datasets
    train_dataset = ChestXrayDataset(train_df, Config.TRAIN_IMG_DIR, 
                                     transform=get_transforms(train=True))
    val_dataset = ChestXrayDataset(val_df, Config.TRAIN_IMG_DIR, 
                                   transform=get_transforms(train=False))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                             shuffle=True, num_workers=Config.NUM_WORKERS,
                             pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                           shuffle=False, num_workers=Config.NUM_WORKERS,
                           pin_memory=True)
    
    # Initialize model
    model = EfficientNetB0Model(num_classes=len(Config.LABELS), pretrained=True, use_attention=True)
    model = model.to(Config.DEVICE)
    
    # Initialize EMA
    ema = EMA(model, decay=Config.EMA_DECAY)
    
    # Calculate pos_weight for class imbalance
    pos_weight = calculate_pos_weight(train_df, Config.LABELS).to(Config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer: AdamW
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, 
                           weight_decay=Config.WEIGHT_DECAY)
    
    # Scheduler: OneCycle
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=Config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if Config.USE_AMP else None
    
    # Training loop
    best_macro_auc = 0.0
    patience_counter = 0
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch [{epoch+1}/{Config.EPOCHS}]")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, 
                                scheduler, scaler, ema, Config.DEVICE)
        
        # Validate with EMA weights
        ema.apply_shadow()
        val_loss, macro_auc, label_aucs = validate_epoch(model, val_loader, criterion, Config.DEVICE)
        ema.restore()
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Macro AUC: {macro_auc:.4f}")
        
        # Print per-label AUCs
        print("Per-label AUCs:")
        for i, (label, auc) in enumerate(zip(Config.LABELS, label_aucs)):
            print(f"  {label}: {auc:.4f}", end=" | " if (i+1) % 3 != 0 else "\n")
        print()
        
        # Save best model based on macro AUC
        if macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
            patience_counter = 0
            
            # Save regular model
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH.format(fold))
            
            # Save EMA model
            ema.apply_shadow()
            torch.save(model.state_dict(), Config.EMA_MODEL_PATH.format(fold))
            ema.restore()
            
            print(f"✓ Best model saved! Macro AUC: {macro_auc:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print("-" * 60)
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return best_macro_auc

def main():
    print("="*60)
    print("EfficientNet-B0 with CBAM & Label Attention")
    print("Grand X-ray Slam Division A")
    print("="*60)
    print(f"\nDevice: {Config.DEVICE}")
    print(f"Image Size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print(f"Weight Decay: {Config.WEIGHT_DECAY}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Mixed Precision: {Config.USE_AMP}")
    print(f"EMA Decay: {Config.EMA_DECAY}\n")
    
    # Load training data
    print("Loading training data...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    print(f"Training samples: {len(train_df)}")
    print(f"Labels: {Config.LABELS}\n")
    
    # Create stratified K-folds
    print(f"Creating {Config.N_FOLDS}-fold MultilabelStratifiedKFold splits...")
    mskf = MultilabelStratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=42)
    
    fold_results = []
    
    # Train each fold
    for fold, (train_idx, val_idx) in enumerate(mskf.split(train_df, train_df[Config.LABELS])):
        if fold not in Config.TRAIN_FOLDS:
            continue
        
        train_data = train_df.iloc[train_idx]
        val_data = train_df.iloc[val_idx]
        
        best_auc = train_fold(fold, train_data, val_data)
        fold_results.append(best_auc)
    
    # Print fold summary
    print("\n" + "="*60)
    print("Cross-Validation Results")
    print("="*60)
    for fold, auc in enumerate(fold_results):
        print(f"Fold {fold}: Macro AUC = {auc:.4f}")
    print(f"\nMean Macro AUC: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print("="*60)
    
    # Generate test predictions (ensemble across folds)
    print("\nGenerating test predictions...")
    test_df = pd.read_csv(Config.TEST_CSV)
    test_dataset = ChestXrayDataset(test_df, Config.TEST_IMG_DIR, 
                                    transform=get_transforms(train=False), is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, num_workers=Config.NUM_WORKERS,
                            pin_memory=True)
    
    all_fold_predictions = []
    
    for fold in Config.TRAIN_FOLDS:
        print(f"Loading fold {fold} EMA model...")
        model = EfficientNetB0Model(num_classes=len(Config.LABELS), pretrained=False, use_attention=True)
        model.load_state_dict(torch.load(Config.EMA_MODEL_PATH.format(fold)))
        model = model.to(Config.DEVICE)
        
        predictions, image_ids = generate_predictions(model, test_loader, Config.DEVICE, tta=True)
        all_fold_predictions.append(predictions)
        
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    # Average predictions across folds
    final_predictions = np.mean(all_fold_predictions, axis=0)
    
    # Create submission
    submission_df = pd.DataFrame(final_predictions, columns=Config.LABELS)
    submission_df.insert(0, 'image_id', image_ids)
    submission_df.to_csv('/kaggle/working/submission.csv', index=False)
    
    print("\n✓ Submission file created: /kaggle/working/submission.csv")
    print("="*60)

if __name__ == '__main__':
    main()




