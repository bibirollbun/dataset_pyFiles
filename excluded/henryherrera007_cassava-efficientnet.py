import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        # Check if the file extension is not '.jpg'
        if not filename.lower().endswith('.jpg'):
            print(os.path.join(dirname, filename))


import sys
import os
import random
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import timm
import math
import matplotlib.pyplot as plt

# Define paths for data and outputs
PATHS = {
    'TRAIN_CSV': '/kaggle/input/cassava-leaf-disease-classification/train.csv',
    'TEST_CSV': '/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv',
    'DISEASE_MAP': '/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json',
    'TRAIN_IMAGES': '/kaggle/input/cassava-leaf-disease-classification/train_images',
    'TEST_IMAGES': '/kaggle/input/cassava-leaf-disease-classification/test_images',
    'OUTPUT': '/kaggle/working/submission.csv',
    'WEIGHTS': '/kaggle/working/weights',
    'PLOTS': '/kaggle/working/plots'  # New path for saving plots
}

# Create weights and plots directories
os.makedirs(PATHS['WEIGHTS'], exist_ok=True)
os.makedirs(PATHS['PLOTS'], exist_ok=True)


# Custom Scheduler with warmup and cosine decay
class WarmupCosineScheduler(_LRScheduler):
    """
    Scheduler with warmup and cosine decay as specified in requirements
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, max_lr=2e-4, final_lr=3.17e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            factor = alpha
            # Ensure factor is positive
            factor = max(0, factor)
            return [self.min_lr + factor * (self.max_lr - self.min_lr) for _ in self.base_lrs]
        else:
            # Cosine decay from max_lr to final_lr
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            progress = min(1.0, progress)
            # Add epsilon to avoid numerical issues
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.final_lr + cosine_factor * (self.max_lr - self.final_lr) for _ in self.base_lrs]


# Sigmoid Focal Loss with Label Smoothing
class SigmoidFocalLossWithLabelSmoothing(nn.Module):
    """
    Sigmoid Focal Loss with Label Smoothing as specified in requirements
    """
    def __init__(self, gamma=2.0, alpha=0.25, smoothing=0.1, reduction='mean'):
        super(SigmoidFocalLossWithLabelSmoothing, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        
        # Apply label smoothing
        smoothed_targets = torch.zeros_like(logits)
        smoothed_targets.fill_(self.smoothing / (num_classes - 1))
        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Calculate focal loss with label smoothing
        targets_one_hot = torch.zeros_like(logits)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        pt = (1 - probs) * targets_one_hot + probs * (1 - targets_one_hot)
        focal_weight = (self.alpha * targets_one_hot + (1 - self.alpha) * (1 - targets_one_hot)) * pt.pow(self.gamma)
        
        loss = -focal_weight * (
            smoothed_targets * torch.log(probs + 1e-8) + 
            (1 - smoothed_targets) * torch.log(1 - probs + 1e-8)
        )
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class Config:
    """Configuration class for model training and inference."""
    
    def __init__(self):
        # Model parameters
        self.seed: int = 719
        self.model_name: str = 'tf_efficientnet_b4.ns_jft_in1k'  # Updated model name to avoid deprecation warning
        self.image_size: int = 512
        self.drop_connect_rate: float = 0.4  # Custom drop connect rate
        self.dropout_rate: float = 0.5      # Dropout rate for custom head
        
        # Training parameters - MODIFIED FOR DEMO RUN
        self.num_epochs: int = 15           # Reduced to 15 epochs for demo run
        self.train_batch_size: int = 16
        self.valid_batch_size: int = 32
        
        # Learning rate parameters
        self.min_lr: float = 1e-6
        self.max_lr: float = 2e-4
        self.final_lr: float = 3.17e-6
        self.warmup_epochs: int = 3
        
        # Loss function parameters
        self.focal_gamma: float = 2.0
        self.focal_alpha: float = 0.25
        self.label_smoothing: float = 0.1
        
        # Early stopping parameters
        self.early_stopping_patience: int = 5
        self.early_stopping_min_delta: float = 0.001
        
        # Other parameters
        self.num_workers: int = 4
        self.grad_accum_steps: int = 1
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_folds: int = 5
        
        # MODIFIED FOR DEMO RUN - Only use a single fold
        self.used_epochs: list = list(range(15))  # Consider all 15 epochs
        self.used_folds: list = [0]               # Only use fold 0 for testing
        
        # Normalization parameters (Global mean and std for 2020 Cassava dataset)
        self.mean: list = [0.4253, 0.4559, 0.3395]  # Updated for Cassava dataset
        self.std: list = [0.2236, 0.2261, 0.2339]   # Updated for Cassava dataset
        self.weight_decay: float = 1e-4
        
        # TTA parameters
        self.tta_patches: int = 6  # 4 overlapping + 2 center crops
        self.tta_augmentations: int = 2  # 2 augmentations per patch
        
        # Checkpoint parameters
        self.save_checkpoint_freq: int = 5  # Save checkpoint every N epochs
        self.save_best_model: bool = True   # Save best model during training

    @staticmethod
    def load_disease_map(path: str) -> Dict[str, str]:
        """Load disease mapping from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class CassavaDataset(Dataset):
    """Dataset class for Cassava Leaf Disease Classification."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str,
        transforms: Optional[A.Compose] = None,
        output_label: bool = True
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.data_root = Path(data_root)
        self.output_label = output_label
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Optional[int]]:
        image_id = self.df.iloc[index]['image_id']
        if not str(image_id).lower().endswith(('.jpg', '.jpeg', '.png')):
            image_id = f"{image_id}.jpg"
        
        image_path = self.data_root / image_id
        try:
            image = self._load_image(str(image_path))
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            print(f"Trying alternative extensions...")
            
            for ext in ['.jpg', '.jpeg', '.png']:
                try:
                    alt_path = self.data_root / f"{self.df.iloc[index]['image_id']}{ext}"
                    if alt_path.exists():
                        image = self._load_image(str(alt_path))
                        break
                except:
                    continue
            else:
                print(f"Could not load image with any extension. Using blank image.")
                image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        if self.transforms:
            image = self.transforms(image=image)['image']
        
        if self.output_label:
            target = self.df.iloc[index]['label']
            return image, target
        return image
    
    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        """Load and convert image to RGB."""
        image = np.array(Image.open(path).convert('RGB'))
        if image is None:
            raise ValueError(f"Failed to load image at {path}")
        return image


class CustomEfficientNet(nn.Module):
    """
    Custom EfficientNet with modified drop path rate and custom head
    with global average pooling and dropout
    """
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        drop_connect_rate: float = 0.4,  # This will be mapped to drop_path_rate
        dropout_rate: float = 0.5
    ):
        super().__init__()
        # Create base model - using drop_path_rate instead of drop_connect_rate
        try:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                drop_path_rate=drop_connect_rate  # This is the key change
            )
            
            # Get the number of features from the last layer
            if hasattr(self.model, 'classifier'):
                n_features = self.model.classifier.in_features
                self.model.classifier = nn.Identity()
            elif hasattr(self.model, 'fc'):
                n_features = self.model.fc.in_features
                self.model.fc = nn.Identity()
            else:
                raise AttributeError("Model doesn't have a standard classifier or fc layer")
            
            # Create a custom head with global average pooling and dropout
            self.custom_head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(n_features, num_classes)
            )
            
        except Exception as e:
            print(f"Error creating model: {e}")
            print(f"Trying fallback to a different model architecture...")
            # Fallback to a different EfficientNet version
            self.model = timm.create_model(
                'tf_efficientnet_b3.ns_jft_in1k',  # Fallback model
                pretrained=True,
                drop_path_rate=drop_connect_rate
            )
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
            
            self.custom_head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(n_features, num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the base model
        features = self.model(x)
        # Forward pass through the custom head
        return self.custom_head(features)


class DataTransforms:
    """Data augmentation and preprocessing transforms."""
    
    @staticmethod
    def get_train_transforms(config: Config) -> A.Compose:
        """Return training data augmentation transforms."""
        return A.Compose([
            A.RandomResizedCrop(config.image_size, config.image_size),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=0.2,
                sat_shift_limit=0.2,
                val_shift_limit=0.2,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.5
            ),
            A.Normalize(
                mean=config.mean,
                std=config.std,
                max_pixel_value=255.0,
                p=1.0
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=config.image_size // 8,
                max_width=config.image_size // 8,
                min_holes=5,
                min_height=config.image_size // 16,
                min_width=config.image_size // 16,
                fill_value=0,
                p=0.5
            ),
            ToTensorV2(p=1.0),
        ], p=1.)
    
    @staticmethod
    def get_valid_transforms(config: Config) -> A.Compose:
        """Return validation data preprocessing transforms."""
        return A.Compose([
            A.CenterCrop(config.image_size, config.image_size, p=1.),
            A.Resize(config.image_size, config.image_size),
            A.Normalize(
                mean=config.mean,
                std=config.std,
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2(p=1.0),
        ], p=1.)
    
    @staticmethod
    def get_specific_tta_transforms(config: Config, aug_type: int) -> A.Compose:
        """Return specific TTA transforms based on augmentation type."""
        if aug_type == 0:  # No augmentation (center crop)
            return A.Compose([
                A.CenterCrop(config.image_size, config.image_size),
                A.Normalize(mean=config.mean, std=config.std, max_pixel_value=255.0),
                ToTensorV2(),
            ])
        elif aug_type == 1:  # Horizontal flip
            return A.Compose([
                A.CenterCrop(config.image_size, config.image_size),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=config.mean, std=config.std, max_pixel_value=255.0),
                ToTensorV2(),
            ])
        elif aug_type == 2:  # Vertical flip
            return A.Compose([
                A.CenterCrop(config.image_size, config.image_size),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=config.mean, std=config.std, max_pixel_value=255.0),
                ToTensorV2(),
            ])
        elif aug_type == 3:  # Transpose
            return A.Compose([
                A.CenterCrop(config.image_size, config.image_size),
                A.Transpose(p=1.0),
                A.Normalize(mean=config.mean, std=config.std, max_pixel_value=255.0),
                ToTensorV2(),
            ])
        elif aug_type == 4:  # Rotate 90
            return A.Compose([
                A.CenterCrop(config.image_size, config.image_size),
                A.Rotate(limit=(90, 90), p=1.0),
                A.Normalize(mean=config.mean, std=config.std, max_pixel_value=255.0),
                ToTensorV2(),
            ])
        else:  # Rotate 270
            return A.Compose([
                A.CenterCrop(config.image_size, config.image_size),
                A.Rotate(limit=(270, 270), p=1.0),
                A.Normalize(mean=config.mean, std=config.std, max_pixel_value=255.0),
                ToTensorV2(),
            ])


class EarlyStopping:
    """Early stopping implementation with model weights restoration."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        restore_best_weights: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_weights = None
        self.best_epoch = -1
    
    def __call__(self, model: nn.Module, val_score: float, epoch: int) -> bool:
        """Update early stopping state based on validation score."""
        if self.best_score is None:
            self.best_score = val_score
            self.best_weights = model.state_dict().copy()
            self.best_epoch = epoch
            return False
        
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            self.best_weights = model.state_dict().copy()
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def restore_model(self, model: nn.Module) -> None:
        """Restore model to best weights."""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print(f"Restored model to best weights from epoch {self.best_epoch}.")


class Trainer:
    """Class for model training with warmup + cosine decay and early stopping."""
    
    def __init__(self, model: nn.Module, config: Config):
        self.model = model
        self.config = config
        self.criterion = SigmoidFocalLossWithLabelSmoothing(
            gamma=config.focal_gamma,
            alpha=config.focal_alpha,
            smoothing=config.label_smoothing
        )
        self.scaler = GradScaler()
        self.device = torch.device(config.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.min_lr,
            weight_decay=config.weight_decay
        )
        
        # Initialize scheduler with warmup and cosine decay
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=config.warmup_epochs,
            max_epochs=config.num_epochs,
            min_lr=config.min_lr,
            max_lr=config.max_lr,
            final_lr=config.final_lr
        )
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        batch_count = 0
        
        # Initialize optimizer gradients at the beginning
        self.optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc='Training')
        
        for batch_idx, (images, batch_targets) in enumerate(pbar):
            batch_count += 1
            images = images.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Forward pass with automatic mixed precision
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, batch_targets)
            
            # Scale the loss and perform backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Record loss and predictions
            total_loss += loss.item()
            
            # Get predicted class
            _, predictions = torch.max(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_targets.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Make sure to update optimizer for any remaining gradients
        if batch_count % self.config.grad_accum_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        # Compute metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        average_loss = total_loss / batch_count
        
        return average_loss, accuracy
    
    @torch.no_grad()
    def validate(self, valid_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate the model on the validation set."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        batch_count = 0
        
        pbar = tqdm(valid_loader, desc='Validating')
        
        for images, batch_targets in pbar:
            batch_count += 1
            images = images.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, batch_targets)
            
            # Record loss and predictions
            total_loss += loss.item()
            
            # Get predicted class
            _, predictions = torch.max(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_targets.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        average_loss = total_loss / batch_count
        
        return average_loss, accuracy


def plot_training_curves(history: Dict, fold: int) -> None:
    """Plot training and validation loss and accuracy curves."""
    # Create figure with 2 subplots
    plt.figure(figsize=(12, 10))
    
    # Plot training and validation loss
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Curves - Fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot training and validation accuracy
    plt.subplot(2, 1, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'Accuracy Curves - Fold {fold}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plot_path = os.path.join(PATHS['PLOTS'], f'training_curves_fold_{fold}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved training curves plot to {plot_path}")
    
    # Plot learning rate curve
    plt.figure(figsize=(10, 5))
    # Extract learning rates from history if available
    if 'learning_rates' in history:
        plt.plot(history['learning_rates'])
        plt.title(f'Learning Rate Schedule - Fold {fold}')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        lr_plot_path = os.path.join(PATHS['PLOTS'], f'lr_curve_fold_{fold}.png')
        plt.savefig(lr_plot_path)
        plt.close()
        print(f"Saved learning rate curve plot to {lr_plot_path}")


def train_fold(fold: int, train_data: pd.DataFrame, valid_data: pd.DataFrame, config: Config, data_root: str) -> Dict:
    """Train a single fold with early stopping and weight restoration."""
    device = torch.device(config.device)
    
    # Create datasets
    train_dataset = CassavaDataset(
        train_data,
        data_root=os.path.join(data_root, PATHS['TRAIN_IMAGES']),
        transforms=DataTransforms.get_train_transforms(config)
    )
    
    valid_dataset = CassavaDataset(
        valid_data,
        data_root=os.path.join(data_root, PATHS['TRAIN_IMAGES']),
        transforms=DataTransforms.get_valid_transforms(config)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Initialize the model with custom drop connect rate and custom head
    model = CustomEfficientNet(
        config.model_name,
        train_data.label.nunique(),
        pretrained=True,
        drop_connect_rate=config.drop_connect_rate,
        dropout_rate=config.dropout_rate
    ).to(device)
    
    # Initialize the trainer
    trainer = Trainer(model, config)
    
    best_val_acc = 0
    best_val_loss = float('inf')
    model_states = {}
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'learning_rates': []
    }
    
    # Training loop
    for epoch in range(config.num_epochs):
        print(f'Epoch {epoch + 1}/{config.num_epochs}')
        
        # Train for one epoch
        train_loss, train_acc = trainer.train_epoch(train_loader)
        
        # Validate the model
        val_loss, val_acc = trainer.validate(valid_loader)
        
        # Get current learning rate
        current_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Update scheduler
        trainer.scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # Print metrics
        print(f'Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')
        print(f'Valid Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # Save checkpoint at specified frequency
        if (epoch + 1) % config.save_checkpoint_freq == 0 or epoch == config.num_epochs - 1:
            checkpoint_path = os.path.join(
                PATHS['WEIGHTS'],
                f'{config.model_name}_fold_{fold}_epoch_{epoch}_checkpoint.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'config': {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
            }, checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')
        
        # Update best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            
            # Save best model weights
            if config.save_best_model:
                best_model_path = os.path.join(
                    PATHS['WEIGHTS'],
                    f'{config.model_name}_fold_{fold}_best.pth'
                )
                torch.save(model.state_dict(), best_model_path)
                print(f'Saved best model weights: {best_model_path}')
            
            # Save checkpoint for weights loading during inference
            checkpoint_path = os.path.join(
                PATHS['WEIGHTS'],
                f'{config.model_name}_fold_{fold}_{epoch}'
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Saved checkpoint for inference: {checkpoint_path}')
        
        # Check early stopping
        if trainer.early_stopping(model, val_acc, epoch):
            print(f'Early stopping triggered at epoch {epoch + 1}')
            # Restore best weights
            trainer.early_stopping.restore_model(model)
            break
        
        # Save model state for specified epochs
        if epoch in config.used_epochs:
            model_states[epoch] = model.state_dict().copy()
            print(f'Saved model state for epoch {epoch}')
    
    # Plot training curves after training is complete
    plot_training_curves(history, fold)
    
    # Save final model
    final_model_path = os.path.join(
        PATHS['WEIGHTS'],
        f'{config.model_name}_fold_{fold}_final.pth'
    )
    torch.save(model.state_dict(), final_model_path)
    print(f'Saved final model weights: {final_model_path}')
    
    # Clean up
    torch.cuda.empty_cache()
    
    return {
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'history': history,
        'model_states': model_states,
        'best_epoch': trainer.early_stopping.best_epoch
    }


@torch.no_grad()
def process_image_for_tta(
    model: nn.Module,
    image: np.ndarray,
    config: Config,
    device: torch.device
) -> torch.Tensor:
    """
    Process a single image for Test Time Augmentation using the specified patch and augmentation strategy.
    
    1. Extract 4 overlapping patches from corners
    2. Extract 2 center crops
    3. Apply specified augmentations to each patch
    4. Average predictions across all patches and augmentations
    """
    h, w = image.shape[:2]
    patch_size = config.image_size
    
    # Handle images smaller than patch_size
    if h < patch_size or w < patch_size:
        # Resize image to be at least patch_size x patch_size
        new_h = max(patch_size, h)
        new_w = max(patch_size, w)
        resized_image = np.array(Image.fromarray(image).resize((new_w, new_h)))
        image = resized_image
        h, w = image.shape[:2]
    
    # Extract 4 overlapping patches from corners
    patches = [
        image[:patch_size, :patch_size],                      # Top-left
        image[:patch_size, w-patch_size:],                    # Top-right
        image[h-patch_size:, :patch_size],                    # Bottom-left
        image[h-patch_size:, w-patch_size:],                  # Bottom-right
        image[h//2-patch_size//2:h//2+patch_size//2,          # Center
              w//2-patch_size//2:w//2+patch_size//2],
        image[h//2-patch_size//2:h//2+patch_size//2,          # Center (duplicate)
              w//2-patch_size//2:w//2+patch_size//2]
    ]
    
    # Apply different augmentations to each patch
    predictions = []
    
    # Process each patch
    for i, patch in enumerate(patches):
        # For the first 4 patches, apply 2 different augmentations
        if i < 4:
            for aug_type in range(1, 3):  # Augmentation types 1 and 2
                transform = DataTransforms.get_specific_tta_transforms(config, aug_type)
                transformed = transform(image=patch)['image']
                
                # Get prediction
                pred = model(transformed.unsqueeze(0).to(device))
                predictions.append(pred.softmax(dim=1))
        else:
            # For center crops, use no augmentation
            transform = DataTransforms.get_specific_tta_transforms(config, 0)
            transformed = transform(image=patch)['image']
            
            # Get prediction
            pred = model(transformed.unsqueeze(0).to(device))
            predictions.append(pred.softmax(dim=1))
    
    # Average all predictions
    final_pred = torch.mean(torch.cat(predictions, dim=0), dim=0, keepdim=True)
    return final_pred
    
def main():
    """
    Main function to run the Cassava Leaf Disease Classification training pipeline.
    This function:
    1. Initializes configuration
    2. Sets random seeds
    3. Loads and prepares data
    4. Runs k-fold cross-validation training
    5. Makes predictions on test set
    6. Creates submission file
    """
    # Initialize configuration
    config = Config()
    
    # Set seeds for reproducibility
    seed_everything(config.seed)
    
    # Load disease mapping
    disease_map = Config.load_disease_map(PATHS['DISEASE_MAP'])
    print(f"Disease mapping: {disease_map}")
    
    # Read training data
    train_df = pd.read_csv(PATHS['TRAIN_CSV'])
    print(f"Training data shape: {train_df.shape}")
    print(f"Class distribution:\n{train_df['label'].value_counts()}")
    
    # Setup k-fold cross-validation
    skf = StratifiedKFold(
        n_splits=config.num_folds,
        shuffle=True,
        random_state=config.seed
    )
    
    # Dictionary to store results for each fold
    fold_results = {}
    
    # Run k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        print(f"\n{'='*50}")
        print(f"Fold {fold}")
        print(f"{'='*50}")
        
        # Check if we should process this fold
        if fold not in config.used_folds:
            print(f"Skipping fold {fold} as per configuration.")
            continue
        
        # Split data for this fold
        train_data = train_df.iloc[train_idx].reset_index(drop=True)
        valid_data = train_df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Training on {len(train_data)} samples, validating on {len(valid_data)} samples")
        
        # Train the model for this fold
        fold_result = train_fold(fold, train_data, valid_data, config, '')
        
        # Store results
        fold_results[fold] = fold_result
        
        print(f"\nFold {fold} best validation accuracy: {fold_result['best_val_acc']:.4f}")
        print(f"Fold {fold} best validation loss: {fold_result['best_val_loss']:.4f}")
        print(f"Fold {fold} best epoch: {fold_result['best_epoch']}")
    
    # Print overall results
    print("\n==== Overall Results ====")
    if fold_results:
        # Calculate average metrics across folds
        avg_val_acc = sum(res['best_val_acc'] for res in fold_results.values()) / len(fold_results)
        avg_val_loss = sum(res['best_val_loss'] for res in fold_results.values()) / len(fold_results)
        
        print(f"Average best validation accuracy: {avg_val_acc:.4f}")
        print(f"Average best validation loss: {avg_val_loss:.4f}")
        
        # Optionally perform inference on test set
        test_inference(config, fold_results)
    else:
        print("No folds were processed.")

def test_inference(config, fold_results):
    """
    Perform inference on the test set using trained models.
    
    Args:
        config: Configuration object
        fold_results: Dictionary of results from each fold
    """
    print("\n==== Test Inference ====")
    
    # Read test data
    test_df = pd.read_csv(PATHS['TEST_CSV'])
    print(f"Test data shape: {test_df.shape}")
    
    # Initialize device
    device = torch.device(config.device)
    
    # Create test dataset
    test_dataset = CassavaDataset(
        test_df,
        data_root=PATHS['TEST_IMAGES'],
        transforms=DataTransforms.get_valid_transforms(config),
        output_label=False
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Ensemble predictions from multiple folds
    all_predictions = []
    
    # For each fold
    for fold in fold_results.keys():
        print(f"Inferencing with fold {fold} model...")
        
        # Load the best model for this fold
        model = CustomEfficientNet(
            config.model_name,
            len(Config.load_disease_map(PATHS['DISEASE_MAP'])),
            pretrained=False,
            drop_connect_rate=config.drop_connect_rate,
            dropout_rate=config.dropout_rate
        ).to(device)
        
        best_model_path = os.path.join(
            PATHS['WEIGHTS'],
            f'{config.model_name}_fold_{fold}_best.pth'
        )
        
        # Check if best model exists
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model from {best_model_path}")
        else:
            # Try to load from the used_epochs list
            best_epoch = fold_results[fold]['best_epoch']
            checkpoint_path = os.path.join(
                PATHS['WEIGHTS'],
                f'{config.model_name}_fold_{fold}_{best_epoch}'
            )
            
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path))
                print(f"Loaded checkpoint from {checkpoint_path}")
            else:
                print(f"Warning: Could not find best model for fold {fold}, using final model")
                final_path = os.path.join(
                    PATHS['WEIGHTS'],
                    f'{config.model_name}_fold_{fold}_final.pth'
                )
                model.load_state_dict(torch.load(final_path))
        
        # Set model to evaluation mode
        model.eval()
        
        # Get predictions for this fold
        fold_preds = []
        
        with torch.no_grad():
            for images in tqdm(test_loader, desc=f"Predicting fold {fold}"):
                images = images.to(device)
                outputs = model(images)
                probs = outputs.softmax(dim=1)
                fold_preds.extend(probs.cpu().numpy())
        
        all_predictions.append(np.array(fold_preds))
    
    # Average predictions across folds
    if all_predictions:
        final_predictions = np.mean(all_predictions, axis=0)
        
        # Get the most likely class for each sample
        predicted_classes = np.argmax(final_predictions, axis=1)
        
        # Create submission dataframe
        submission_df = pd.DataFrame({
            'image_id': test_df['image_id'],
            'label': predicted_classes
        })
        
        # Save submission file
        submission_df.to_csv(PATHS['OUTPUT'], index=False)
        print(f"Saved submission file to {PATHS['OUTPUT']}")
    else:
        print("No predictions were made.")

if __name__ == "__main__":
    main()

