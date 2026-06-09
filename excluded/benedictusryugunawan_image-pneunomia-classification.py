import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import warnings
import torchvision.transforms as T
import cv2

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore') 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchmetrics
from tqdm.auto import tqdm

# Import image classification utilities from timm (PyTorch Image Models)
import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy


import torch

class Config:
    """
    Configuration class for all hyperparameters, paths, and settings.
    """

    # ---------------------------
    # Paths
    # ---------------------------
    train_dir = '/kaggle/input/srifoton-25-machine-learning-competition/train/train'
    val_dir   = '/kaggle/input/srifoton-25-machine-learning-competition/val/val'
    test_dir  = '/kaggle/input/srifoton-25-machine-learning-competition/test/test'

    # Checkpoints
    checkpoint_path = '/kaggle/input/my-trained-model/final_model.pth'  # uploaded dataset path
    final_model_path = 'final_model.pth'   # for saving after inference

    # ---------------------------
    # Classes
    # ---------------------------
    classes = [
        'Bacterial Pneumonia',
        'Corona Virus Disease',
        'Normal',
        'Tuberculosis',
        'Viral Pneumonia'
    ]
    num_classes = len(classes)

    # ---------------------------
    # Model
    # ---------------------------
    model_name = 'convnext_base.fb_in22k_ft_in1k'  # Pre-trained ConvNeXt Base
    pretrained = True

    # ---------------------------
    # Training Parameters
    # ---------------------------
    batch_size = 8
    num_epochs = 17
    learning_rate = 1e-4
    weight_decay = 1e-4
    label_smoothing = 0.1

    # ---------------------------
    # Early Stopping
    # ---------------------------
    early_stopping_patience = 7
    min_delta = 0.001

    # ---------------------------
    # Augmentation Parameters
    # ---------------------------
    img_size = 512  # Input size for model
    mixup_alpha = 0.2
    cutmix_alpha = 1.0
    mixup_prob = 0.5

    # ---------------------------
    # TTA (Test Time Augmentation)
    # ---------------------------
    use_tta = True
    tta_transforms = 4   # number of augmentations to apply during inference

    # ---------------------------
    # System Settings
    # ---------------------------
    num_workers = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42


cfg = Config()


class EarlyStopping:
    """Early stopping utility to stop training when validation score stops improving."""
    
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True, verbose=True):
        """
        Args:
            patience (int): Number of epochs with no improvement to wait before stopping
            min_delta (float): Minimum change to qualify as an improvement
            restore_best_weights (bool): Whether to restore model weights from best epoch
            verbose (bool): Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        self.best_model_state = None
        
    def __call__(self, score, model, epoch):
        """
        Check if early stopping criteria is met.
        
        Args:
            score (float): Current validation score (higher is better)
            model: The model being trained
            epoch (int): Current epoch number
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_model_state = model.state_dict().copy()
                
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience} (Best F1: {self.best_score:.4f} at epoch {self.best_epoch + 1})')
                
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'Early stopping triggered! Restoring best weights from epoch {self.best_epoch + 1}')
                    
        else:
            if self.verbose and score > self.best_score + self.min_delta:
                print(f'Validation score improved from {self.best_score:.4f} to {score:.4f}')
                
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_state = model.state_dict().copy()
    
    def restore_best_model(self, model):
        """Restore the best model weights."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def set_seed(seed):
    """Sets the seed for reproducibility across different libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Setting these to True/False as appropriate can sometimes affect performance
        # but ensures deterministic behavior.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(cfg.seed)


class CXRDataset(Dataset):
    """Custom Dataset for Chest X-Ray images."""
    def __init__(self, root_dir, transform=None, is_train=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_train = is_train
        self.samples = []
        self.labels = []
        
        # Determine how to load files based on the directory type
        if is_train or root_dir == cfg.val_dir:
            # Training or validation: Images are in class subfolders
            for idx, class_name in enumerate(cfg.classes):
                class_dir = self.root_dir / class_name
                if class_dir.exists():
                    # Collect all image files with common extensions
                    for ext in ['*.jpeg', '*.jpg', '*.png']:
                        for img_path in class_dir.glob(ext):
                            self.samples.append(img_path)
                            self.labels.append(idx)
        else:
            # Test set: Images are directly under the root_dir (no class subfolders)
            for ext in ['*.jpeg', '*.jpg', '*.png']:
                for img_path in self.root_dir.glob(ext):
                    self.samples.append(img_path)
                    self.labels.append(-1) # Placeholder for unknown label
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Open and convert to RGB (important for grayscale X-rays for pre-trained models)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        # Return filename instead of label for the test set
        if label == -1:
            return img, img_path.name
        
        return img, label


def get_train_transforms():
    """Returns a sequence of aggressive data augmentations for training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        # Normalization using ImageNet mean/std (standard practice for pre-trained models)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms():
    """Returns a simple resizing and normalization for validation/testing."""
    return transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_model():
    """Initializes a pre-trained model from the timm library."""
    model = timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes # Adjust the final layer for the number of classes
    )
    return model


# def train_one_epoch(model, loader, criterion, optimizer, scheduler, mixup_fn, device, epoch):
#     """Performs a single training epoch."""
#     model.train()
#     running_loss = 0.0
#     pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{cfg.num_epochs} [Train]')
    
#     for batch_idx, (images, labels) in enumerate(pbar):
#         images = images.to(device)
#         labels = labels.to(device)
        
#         # Apply Mixup/CutMix on the batch
#         if mixup_fn is not None:
#             images, labels = mixup_fn(images, labels)
        
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels) # Uses SoftTargetCrossEntropy for mixed labels
        
#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         # Update metrics
#         running_loss += loss.item()
        
#         # Update progress bar
#         pbar.set_postfix({
#             'loss': running_loss / (batch_idx + 1), 
#             'lr': optimizer.param_groups[0]['lr']
#         })
    
#     scheduler.step() # Step the learning rate scheduler after the epoch
#     return running_loss / len(loader)

# def validate(model, loader, device):
#     """Evaluates the model on the validation set."""
#     model.eval()
#     all_preds = []
#     all_labels = []
#     running_loss = 0.0
    
#     # Use standard CrossEntropyLoss for validation (since labels aren't mixed)
#     criterion = nn.CrossEntropyLoss()
    
#     with torch.no_grad():
#         pbar = tqdm(loader, desc='Validating')
#         for images, labels in pbar:
#             images = images.to(device)
#             labels = labels.to(device)
            
#             outputs = model(images)
#             loss = criterion(outputs, labels)
            
#             running_loss += loss.item()
            
#             # Collect predictions and labels
#             preds = torch.argmax(outputs, dim=1)
#             all_preds.append(preds.cpu())
#             all_labels.append(labels.cpu())
    
#     all_preds = torch.cat(all_preds)
#     all_labels = torch.cat(all_labels)
    
#     # Calculate Macro F1 Score (The target competition metric)
#     f1_metric = torchmetrics.F1Score(
#         task='multiclass', 
#         num_classes=cfg.num_classes, 
#         average='macro'
#     )
#     f1_score = f1_metric(all_preds, all_labels).item()
    
#     # Calculate standard accuracy
#     acc_metric = torchmetrics.Accuracy(
#         task='multiclass', 
#         num_classes=cfg.num_classes
#     )
#     accuracy = acc_metric(all_preds, all_labels).item()
    
#     avg_loss = running_loss / len(loader)
    
#     return avg_loss, f1_score, accuracy


# def train_model():
#     """Sets up the data, model, optimizer, and executes the training loop with early stopping."""
#     print(f"Using device: {cfg.device}")
#     print(f"Model: {cfg.model_name}")
#     print(f"Early stopping patience: {cfg.early_stopping_patience} epochs")
#     print(f"Minimum improvement threshold: {cfg.min_delta}")
#     print("-" * 50)
    
#     # Create datasets and print counts
#     train_dataset = CXRDataset(cfg.train_dir, transform=get_train_transforms(), is_train=True)
#     val_dataset = CXRDataset(cfg.val_dir, transform=get_val_transforms(), is_train=False)
    
#     print(f"Train samples: {len(train_dataset)}")
#     print(f"Val samples: {len(val_dataset)}")
    
#     # Create dataloaders
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=cfg.batch_size, 
#         shuffle=True, 
#         num_workers=cfg.num_workers,
#         pin_memory=True
#     )
    
#     val_loader = DataLoader(
#         val_dataset, 
#         batch_size=cfg.batch_size, 
#         shuffle=False, 
#         num_workers=cfg.num_workers,
#         pin_memory=True
#     )
    
#     # Create model and move to device
#     model = create_model().to(cfg.device)
    
#     # Setup Mixup/CutMix data augmentation
#     mixup_fn = Mixup(
#         mixup_alpha=cfg.mixup_alpha,
#         cutmix_alpha=cfg.cutmix_alpha,
#         prob=cfg.mixup_prob,
#         mode='batch',
#         label_smoothing=cfg.label_smoothing,
#         num_classes=cfg.num_classes
#     )
    
#     # Loss function: SoftTargetCrossEntropy handles soft labels from Mixup
#     criterion = SoftTargetCrossEntropy()
    
#     # Optimizer: AdamW is commonly used in vision tasks
#     optimizer = optim.AdamW(
#         model.parameters(), 
#         lr=cfg.learning_rate,
#         weight_decay=cfg.weight_decay
#     )
    
#     # Scheduler: Cosine Annealing reduces LR smoothly
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(
#         optimizer,
#         T_max=cfg.num_epochs,
#         eta_min=1e-6
#     )
    
#     # Initialize Early Stopping
#     early_stopping = EarlyStopping(
#         patience=cfg.early_stopping_patience,
#         min_delta=cfg.min_delta,
#         restore_best_weights=True,
#         verbose=True
#     )
    
#     # Training history tracking
#     best_f1 = 0.0
#     training_history = {
#         'train_loss': [],
#         'val_loss': [],
#         'val_f1': [],
#         'val_accuracy': []
#     }
    
#     # Start Training Loop
#     for epoch in range(cfg.num_epochs):
#         print(f"\nEpoch {epoch+1}/{cfg.num_epochs}")
#         print("-" * 30)
        
#         # Train
#         train_loss = train_one_epoch(
#             model, train_loader, criterion, optimizer, scheduler, mixup_fn, cfg.device, epoch
#         )
        
#         # Validate
#         val_loss, val_f1, val_acc = validate(model, val_loader, cfg.device)
        
#         # Store history
#         training_history['train_loss'].append(train_loss)
#         training_history['val_loss'].append(val_loss)
#         training_history['val_f1'].append(val_f1)
#         training_history['val_accuracy'].append(val_acc)
        
#         print(f"Train Loss: {train_loss:.4f}")
#         print(f"Val Loss: {val_loss:.4f}")
#         print(f"Val F1 Score: {val_f1:.4f}")
#         print(f"Val Accuracy: {val_acc:.4f}")
        
#         # Save best model based on F1 score (the primary metric)
#         if val_f1 > best_f1:
#             best_f1 = val_f1
#             print(f"New best F1 score: {best_f1:.4f}. Saving model...")
#             # Save checkpoint
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'best_f1': best_f1,
#                 'val_loss': val_loss,
#                 'training_history': training_history
#             }, cfg.checkpoint_path)
        
#         # Check early stopping
#         early_stopping(val_f1, model, epoch)
        
#         if early_stopping.early_stop:
#             print(f"\nEarly stopping at epoch {epoch+1}")
#             # Restore best model weights
#             early_stopping.restore_best_model(model)
#             # Save the final model with restored weights
#             torch.save({
#                 'epoch': early_stopping.best_epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'best_f1': early_stopping.best_score,
#                 'val_loss': val_loss,
#                 'training_history': training_history,
#                 'early_stopped': True,
#                 'stopped_at_epoch': epoch + 1
#             }, cfg.checkpoint_path)
#             break
    
#     print("\n" + "="*50)
#     if early_stopping.early_stop:
#         print(f"Training stopped early at epoch {epoch+1}")
#         print(f"Best F1 Score: {early_stopping.best_score:.4f} (achieved at epoch {early_stopping.best_epoch + 1})")
#     else:
#         print(f"Training completed! Best F1 Score: {best_f1:.4f}")
    
#     return model


def predict_with_tta(model, img_path, transforms, device):
    """Apply TTA and average predictions"""
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    preds = []
    with torch.no_grad():
        for t in transforms:
            augmented = t(image=img)
            img_tensor = augmented['image'].unsqueeze(0).to(device)
            output = model(img_tensor)  # logits
            preds.append(torch.softmax(output, dim=1))  # simpan probabilitas
    
    # Rata-rata prediksi
    avg_pred = torch.mean(torch.stack(preds), dim=0)
    return avg_pred


def generate_submission():
    print("\nGenerating submission file...")
    
    # Load best model
    model = create_model().to(cfg.device)
    checkpoint = torch.load("/kaggle/input/mihu-mihu-weight-model/Mihu Mihu Weight Model/mihu mihu_WeightÂ Model.pth", map_location=cfg.device)
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
print("CXR Classification Pipeline with Early Stopping - Start")
print("="*50)

# 1. Train the model
# Note: The function saves the best model during training
# trained_model = train_model()

# 2. Generate submission file using the best checkpoint
submission_df = generate_submission()

print("\n" + "="*50)
print("Pipeline completed successfully! ðŸŽ‰")
print("Files generated:")
print(f"  - {cfg.checkpoint_path}: Best trained model checkpoint")
print("  - submission.csv: Predictions for Kaggle submission")
print("="*50)

