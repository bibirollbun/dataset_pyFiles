# %% [code]
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchmetrics
from tqdm.auto import tqdm

import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy


class Config:
    # Paths
    train_dir = '/kaggle/input/srifoton-25-machine-learning-competition/train/train'
    val_dir = '/kaggle/input/srifoton-25-machine-learning-competition/val/val'
    test_dir = '/kaggle/input/srifoton-25-machine-learning-competition/test/test'
    
    # Classes
    classes = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
    num_classes = 5
    
    # Model
    model_name = 'convnext_base.fb_in22k_ft_in1k'
    pretrained = True
    
    # Training
    batch_size = 8
    num_epochs = 20
    learning_rate = 1e-4
    weight_decay = 1e-4
    label_smoothing = 0.1
    
    # Augmentation
    img_size = 384
    mixup_alpha = 0.2
    cutmix_alpha = 1.0
    mixup_prob = 0.5
    
    # Others
    num_workers = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    checkpoint_path = 'best_model.pth'

cfg = Config()



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)


class CXRDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_train = is_train
        self.samples = []
        self.labels = []
        
        if is_train or root_dir == cfg.val_dir:
            # Training or validation with class folders
            for idx, class_name in enumerate(cfg.classes):
                class_dir = self.root_dir / class_name
                if class_dir.exists():
                    for img_path in class_dir.glob('*.jpeg'):
                        self.samples.append(img_path)
                        self.labels.append(idx)
                    for img_path in class_dir.glob('*.jpg'):
                        self.samples.append(img_path)
                        self.labels.append(idx)
                    for img_path in class_dir.glob('*.png'):
                        self.samples.append(img_path)
                        self.labels.append(idx)
        else:
            # Test set without labels
            for img_path in self.root_dir.glob('*.jpeg'):
                self.samples.append(img_path)
                self.labels.append(-1)
            for img_path in self.root_dir.glob('*.jpg'):
                self.samples.append(img_path)
                self.labels.append(-1)
            for img_path in self.root_dir.glob('*.png'):
                self.samples.append(img_path)
                self.labels.append(-1)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        if label == -1:
            return img, img_path.name
        
        return img, label



def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



def create_model():
    model = timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes
    )
    return model


def train_one_epoch(model, loader, criterion, optimizer, scheduler, mixup_fn, device, epoch):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{cfg.num_epochs} [Train]')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Apply Mixup/CutMix
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': running_loss / (batch_idx + 1), 
                         'lr': optimizer.param_groups[0]['lr']})
    
    scheduler.step()
    return running_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    
    # Use CrossEntropy for validation (no mixup)
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validating')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Calculate Macro F1 Score
    f1_metric = torchmetrics.F1Score(task='multiclass', num_classes=cfg.num_classes, average='macro')
    f1_score = f1_metric(all_preds, all_labels).item()
    
    # Calculate accuracy
    acc_metric = torchmetrics.Accuracy(task='multiclass', num_classes=cfg.num_classes)
    accuracy = acc_metric(all_preds, all_labels).item()
    
    avg_loss = running_loss / len(loader)
    
    return avg_loss, f1_score, accuracy



# ===================== Main Training Loop =====================
def train_model():
    print(f"Using device: {cfg.device}")
    print(f"Model: {cfg.model_name}")
    print("-" * 50)
    
    # Create datasets
    train_dataset = CXRDataset(cfg.train_dir, transform=get_train_transforms(), is_train=True)
    val_dataset = CXRDataset(cfg.val_dir, transform=get_val_transforms(), is_train=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = create_model().to(cfg.device)
    
    # Setup Mixup/CutMix
    mixup_fn = Mixup(
        mixup_alpha=cfg.mixup_alpha,
        cutmix_alpha=cfg.cutmix_alpha,
        prob=cfg.mixup_prob,
        mode='batch',
        label_smoothing=cfg.label_smoothing,
        num_classes=cfg.num_classes
    )
    
    # Loss function (SoftTargetCrossEntropy for mixup)
    criterion = SoftTargetCrossEntropy()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    
    # Scheduler (Cosine Annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.num_epochs,
        eta_min=1e-6
    )
    
    # Training loop
    best_f1 = 0.0
    train_losses = []
    val_losses = []
    val_f1_scores = []
    
    for epoch in range(cfg.num_epochs):
        print(f"\nEpoch {epoch+1}/{cfg.num_epochs}")
        print("-" * 30)
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, mixup_fn, cfg.device, epoch
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_f1, val_acc = validate(model, val_loader, cfg.device)
        val_losses.append(val_loss)
        val_f1_scores.append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val F1 Score: {val_f1:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        
        # Save best model based on F1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"New best F1 score: {best_f1:.4f}. Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_loss': val_loss,
            }, cfg.checkpoint_path)
    
    print("\n" + "="*50)
    print(f"Training completed! Best F1 Score: {best_f1:.4f}")
    
    return model


def generate_submission():
    print("\nGenerating submission file...")
    
    # Load best model
    model = create_model().to(cfg.device)
    checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create test dataset
    test_dataset = CXRDataset(cfg.test_dir, transform=get_val_transforms(), is_train=False)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    # Predictions
    predictions = []
    filenames = []
    
    with torch.no_grad():
        for images, names in tqdm(test_loader, desc='Inference'):
            images = images.to(cfg.device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            filenames.extend(names)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'Id': filenames,
        'Predicted': predictions
    })
    
    # Sort by filename to ensure correct order
    submission = submission.sort_values('Id')
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    print(f"Submission saved! Total predictions: {len(submission)}")
    
    # Print class distribution
    print("\nPredicted class distribution:")
    for i, class_name in enumerate(cfg.classes):
        count = (submission['Predicted'] == i).sum()
        print(f"{class_name}: {count}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(cfg)  # simpan config juga biar reproducible
    }, "final_model.pth")
    print("Model saved as final_model.pth")
    
    return submission



print("="*50)
print("CXR Classification Pipeline")
print("="*50)

# Train the model
model = train_model()

# Generate submission
submission_df = generate_submission()

print("\n" + "="*50)
print("Pipeline completed successfully!")
print("Files generated:")

