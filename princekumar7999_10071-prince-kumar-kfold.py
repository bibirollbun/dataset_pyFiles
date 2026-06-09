# Complete K-Fold Cross-Validation Workflow for SIIM-ISIC Melanoma Classification
# Using EfficientNet in PyTorch with StratifiedKFold

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import timm
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    seed = 42
    n_folds = 5
    img_size = 384
    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'tf_efficientnet_b3_ns'
    num_classes = 1
    save_path = '/kaggle/working/'

# Set random seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.seed)

# Custom Dataset Class
class MelanomaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['image_name']}.jpg")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(row['target'], dtype=torch.float32)
        
        return image, label

# Data Transforms
train_transform = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((Config.img_size, Config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Model Definition
class EfficientNetModel(nn.Module):
    def __init__(self, model_name=Config.model_name, num_classes=Config.num_classes, pretrained=True):
        super(EfficientNetModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

# Training Function
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, labels in tqdm(dataloader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# Validation Function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    auc_score = roc_auc_score(all_labels, all_preds)
    
    return epoch_loss, auc_score

# Main K-Fold Cross-Validation
def main():
    # Load training data
    train_df = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
    img_dir = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train'
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Target distribution:\n{train_df['target'].value_counts()}")
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=Config.n_folds, shuffle=True, random_state=Config.seed)
    fold_scores = []
    
    # K-Fold Cross-Validation Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['target'])):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{Config.n_folds}")
        print(f"{'='*60}")
        
        # Split data
        train_fold_df = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = train_df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Train size: {len(train_fold_df)}, Val size: {len(val_fold_df)}")
        
        # Create datasets and dataloaders
        train_dataset = MelanomaDataset(train_fold_df, img_dir, transform=train_transform)
        val_dataset = MelanomaDataset(val_fold_df, img_dir, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, 
                                 shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, 
                               shuffle=False, num_workers=2, pin_memory=True)
        
        # Initialize model, criterion, optimizer
        model = EfficientNetModel().to(Config.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                         factor=0.5, patience=2, verbose=True)
        
        # Training loop for this fold
        best_auc = 0.0
        best_model_path = os.path.join(Config.save_path, f'efficientnet_fold{fold+1}_best.pth')
        
        for epoch in range(Config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{Config.num_epochs}")
            
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, Config.device)
            val_loss, val_auc = validate(model, val_loader, criterion, Config.device)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            # Save best model for this fold
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': val_auc,
                    'fold': fold
                }, best_model_path)
                print(f"Best model saved with AUC: {best_auc:.4f}")
            
            scheduler.step(val_auc)
        
        print(f"\nFold {fold + 1} Best AUC: {best_auc:.4f}")
        fold_scores.append(best_auc)
        
        # Clean up memory
        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()
    
    # Print final results
    print(f"\n{'='*60}")
    print("K-Fold Cross-Validation Results")
    print(f"{'='*60}")
    for i, score in enumerate(fold_scores):
        print(f"Fold {i + 1} AUC: {score:.4f}")
    print(f"\nMean AUC across {Config.n_folds} folds: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"\nAll fold model weights saved to: {Config.save_path}")

if __name__ == '__main__':
    main()

