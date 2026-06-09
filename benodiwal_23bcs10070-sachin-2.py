import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    # Paths
    TRAIN_DIR = '/kaggle/input/petfinder-pawpularity-score/train'
    TRAIN_CSV = '/kaggle/input/petfinder-folds-dataset/train_folds.csv'
    OUTPUT_DIR = '/kaggle/working'
    
    # Model
    MODEL_NAME = 'tf_efficientnet_b4_ns'  # Noisy Student weights
    IMG_SIZE = 384
    NUM_CLASSES = 1
    
    # Training
    FOLD = 0  # Change this to train different folds (0-4)
    EPOCHS = 8
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-6
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Metadata features
    META_FEATURES = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 
                     'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 
                     'Info', 'Blur']

# Dataset
class PawpularityDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, row['Id'] + '.jpg')
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Get metadata (12 features) - convert to float explicitly
        meta_values = row[Config.META_FEATURES].values.astype(np.float32)
        meta = torch.from_numpy(meta_values)
        
        # Get target (normalize to 0-1 for BCE loss)
        if self.is_train:
            target = torch.tensor(row['Pawpularity'] / 100.0, dtype=torch.float32)
            return img, meta, target
        else:
            return img, meta

# Data augmentation
def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
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

# Model with metadata integration
class PawpularityModel(nn.Module):
    def __init__(self, model_name, num_meta_features=12, pretrained=True):
        super().__init__()
        
        # Load pre-trained backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Get number of features from backbone
        if hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            in_features = self.backbone.num_features
        
        # Regression head with metadata
        self.head = nn.Sequential(
            nn.Linear(in_features + num_meta_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
    def forward(self, img, meta):
        # Extract features from image
        img_features = self.backbone(img)
        
        # Concatenate with metadata
        combined = torch.cat([img_features, meta], dim=1)
        
        # Pass through regression head
        output = self.head(combined)
        return output

# Training function
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc='Training')
    for img, meta, target in pbar:
        img, meta, target = img.to(device), meta.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(img, meta).squeeze()
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return running_loss / len(loader)

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for img, meta, target in pbar:
            img, meta, target = img.to(device), meta.to(device), target.to(device)
            
            # Forward pass
            output = model(img, meta).squeeze()
            
            # Calculate loss
            loss = criterion(output, target)
            running_loss += loss.item()
            
            # Store predictions (convert back to 0-100 scale for RMSE)
            preds = torch.sigmoid(output) * 100.0
            predictions.extend(preds.cpu().numpy())
            targets.extend((target * 100.0).cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    # Calculate RMSE
    predictions = np.array(predictions)
    targets = np.array(targets)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    avg_loss = running_loss / len(loader)
    return avg_loss, rmse

def main():
    print("=" * 80)
    print(f"PetFinder Pawpularity - Training on Fold {Config.FOLD}")
    print("=" * 80)
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Image Size: {Config.IMG_SIZE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print(f"Epochs: {Config.EPOCHS}")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(Config.TRAIN_CSV)
    
    # DEBUG: Check column names and data types
    print("\n" + "=" * 80)
    print("DATA VALIDATION")
    print("=" * 80)
    print(f"Available columns: {df.columns.tolist()}")
    print(f"\nMetadata features to use: {Config.META_FEATURES}")
    
    # Check if all metadata features exist
    missing_features = [f for f in Config.META_FEATURES if f not in df.columns]
    if missing_features:
        print(f"\n❌ ERROR: Missing features: {missing_features}")
        print("\nAttempting to find correct column names...")
        # Try to find similar column names
        for missing in missing_features:
            similar = [col for col in df.columns if missing.lower() in col.lower()]
            if similar:
                print(f"  - '{missing}' might be: {similar}")
        raise ValueError("Please update Config.META_FEATURES with correct column names")
    
    # Check data types
    print("\nMetadata column types:")
    for feature in Config.META_FEATURES:
        dtype = df[feature].dtype
        print(f"  - {feature}: {dtype}")
        if dtype == 'object':
            print(f"    ⚠️ Warning: '{feature}' is object type, should be numeric")
            # Try to convert to numeric
            df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0).astype(int)
            print(f"    ✓ Converted to: {df[feature].dtype}")
    
    # Verify Pawpularity column
    print(f"\nTarget column 'Pawpularity': {df['Pawpularity'].dtype}")
    print(f"  Range: [{df['Pawpularity'].min()}, {df['Pawpularity'].max()}]")
    print(f"  Mean: {df['Pawpularity'].mean():.2f}")
    
    print("\n" + "=" * 80)
    
    train_df = df[df['fold'] != Config.FOLD].reset_index(drop=True)
    val_df = df[df['fold'] == Config.FOLD].reset_index(drop=True)
    
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Create datasets and dataloaders
    train_dataset = PawpularityDataset(
        train_df, Config.TRAIN_DIR, 
        transform=get_transforms(is_train=True)
    )
    val_dataset = PawpularityDataset(
        val_df, Config.TRAIN_DIR, 
        transform=get_transforms(is_train=False)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, 
        shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, 
        shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    # Create model
    print("\nInitializing model...")
    model = PawpularityModel(Config.MODEL_NAME, num_meta_features=12)
    model = model.to(Config.DEVICE)
    
    # Loss function - BCE with Logits (critical for this competition!)
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, 
                          weight_decay=Config.WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )
    
    # Training loop
    best_rmse = float('inf')
    
    print("\nStarting training...")
    print("=" * 80)
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        print("-" * 80)
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        
        # Validate
        val_loss, val_rmse = validate(model, val_loader, criterion, Config.DEVICE)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val RMSE: {val_rmse:.4f} | LR: {current_lr:.6f}")
        
        # Save best model
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            model_path = f'{Config.OUTPUT_DIR}/best_model_fold{Config.FOLD}.pth'
            
            # Save only the essential config as a regular dict (not class __dict__)
            config_dict = {
                'MODEL_NAME': Config.MODEL_NAME,
                'IMG_SIZE': Config.IMG_SIZE,
                'FOLD': Config.FOLD,
                'EPOCHS': Config.EPOCHS,
                'BATCH_SIZE': Config.BATCH_SIZE,
                'LEARNING_RATE': Config.LEARNING_RATE,
                'META_FEATURES': Config.META_FEATURES
            }
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse,
                'config': config_dict
            }, model_path)
            print(f"✓ Saved best model (RMSE: {best_rmse:.4f})")
    
    print("\n" + "=" * 80)
    print(f"Training completed! Best validation RMSE: {best_rmse:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()

