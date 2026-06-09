import sys
sys.path.append("../input/tez-lib/")

import random
import numpy as np
import torch
import os
import torch.nn as nn
from tez import Tez, TezConfig
import tez
import albumentations
import pandas as pd
import cv2
import timm
from sklearn import metrics
from tez.callbacks import EarlyStopping
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
import dataclasses

@dataclasses.dataclass
class TrainingConfig:
    """Configuration class for training parameters"""
    batch_size: int = 8
    image_size: int = 384
    epochs: int = 4
    learning_rate: float = 2.5e-5
    weight_decay: float = 0.01
    dropout_rate: float = 0.5
    patience: int = 2
    num_folds: int = 2

def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed
seed_everything(42)


class PawpularDataset(torch.utils.data.Dataset):
    """Dataset class for Pawpularity competition"""
    
    def __init__(
        self, 
        image_paths: List[str], 
        dense_features: np.ndarray, 
        targets: np.ndarray, 
        augmentations: Optional[albumentations.Compose]
    ):
        self.image_paths = image_paths
        self.dense_features = dense_features
        self.targets = targets
        self.augmentations = augmentations
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        # Load and process image
        image = cv2.imread(self.image_paths[item])
        if image is None:
            raise ValueError(f"Could not load image: {self.image_paths[item]}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentations
        if self.augmentations is not None:
            image = self.augmentations(image=image)["image"]
            
        # Transpose and normalize image
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(self.dense_features[item], dtype=torch.float),
            "targets": torch.tensor(self.targets[item], dtype=torch.float),
        }


class PawpularModel(nn.Module):
    """Neural network model for Pawpularity prediction"""
    
    def __init__(self, config: TrainingConfig, model_name: str = "resnet50"):
        super().__init__()
        self.config = config
        
        # Backbone model
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True, 
            in_chans=3,
            num_classes=0  # Remove final classification layer
        )
        
        # Get feature dimension from backbone
        backbone_features = self.backbone(torch.randn(1, 3, config.image_size, config.image_size)).shape[1]
        
        # Additional layers
        self.dropout = nn.Dropout(config.dropout_rate)
        self.output_layer = nn.Linear(backbone_features + 12, 1)  # 12 dense features
        
        self.step_scheduler_after = "epoch"

    def monitor_metrics(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor, 
        loss: torch.Tensor
    ) -> Dict[str, torch.Tensor]:  # Changed to return tensors
        """Calculate and return metrics"""
        with torch.no_grad():
            # Calculate RMSE and keep it as a tensor
            rmse = torch.sqrt(loss)
            return {"rmse": rmse}

    def optimizer_scheduler(self) -> Tuple[torch.optim.Optimizer, Any]:
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10, 
            T_mult=1, 
            eta_min=1e-6, 
            last_epoch=-1
        )
        
        return optimizer, scheduler

    def forward(
        self, 
        image: torch.Tensor, 
        features: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass"""
        # Extract image features
        image_features = self.backbone(image)
        image_features = self.dropout(image_features)
        
        # Combine with dense features
        combined_features = torch.cat([image_features, features], dim=1)
        combined_features = self.dropout(combined_features)
        
        # Final prediction
        predictions = self.output_layer(combined_features)
        
        # Calculate loss if targets provided
        if targets is not None:
            loss = nn.MSELoss()(predictions, targets.view(-1, 1))
            metrics = self.monitor_metrics(predictions, targets, loss)
            return predictions, loss, metrics
        
        return predictions, torch.tensor(0.0), {}


def get_augmentations(config: TrainingConfig) -> Tuple[albumentations.Compose, albumentations.Compose]:
    """Get training and validation augmentations"""
    
    train_aug = albumentations.Compose([
        albumentations.LongestMaxSize(config.image_size, p=1),
        albumentations.PadIfNeeded(
            config.image_size, config.image_size, p=1, border_mode=0
        ),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.1),
        albumentations.Rotate(limit=180, p=0.5),
        albumentations.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5
        ),
        albumentations.HueSaturationValue(
            hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
        ),
        albumentations.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ], p=1.0)

    valid_aug = albumentations.Compose([
        albumentations.LongestMaxSize(config.image_size, p=1),
        albumentations.PadIfNeeded(
            config.image_size, config.image_size, p=1, border_mode=0
        ),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ], p=1.0)
    
    return train_aug, valid_aug


def train_fold(
    fold: int, 
    df: pd.DataFrame, 
    config: TrainingConfig,
    model_name: str = "resnet50"
) -> None:
    """Train model for a single fold"""
    
    print(f"Training fold: {fold}")
    
    # Prepare data
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    dense_features = [
        'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
        'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
    ]
    
    train_img_paths = [
        f"../input/petfinder-pawpularity-score/train/{x}.jpg" 
        for x in df_train["Id"].values
    ]
    valid_img_paths = [
        f"../input/petfinder-pawpularity-score/train/{x}.jpg" 
        for x in df_valid["Id"].values
    ]
    
    # Get augmentations
    train_aug, valid_aug = get_augmentations(config)
    
    # Create datasets
    train_dataset = PawpularDataset(
        image_paths=train_img_paths,
        dense_features=df_train[dense_features].values,
        targets=df_train.Pawpularity.values,
        augmentations=train_aug,
    )
    
    valid_dataset = PawpularDataset(
        image_paths=valid_img_paths,
        dense_features=df_valid[dense_features].values,
        targets=df_valid.Pawpularity.values,
        augmentations=valid_aug,
    )
    
    # Initialize model
    model = PawpularModel(config, model_name)
    model = Tez(model)
    
    # Configure training
    tez_config = TezConfig(
        training_batch_size=config.batch_size,
        validation_batch_size=2 * config.batch_size,
        epochs=config.epochs,
        step_scheduler_after="epoch",
        step_scheduler_metric="valid_rmse",
        fp16=True,
        val_strategy="batch",
        val_steps=900,
    )
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor="valid_rmse",
        model_path=f"model_f{fold}.bin",
        patience=config.patience,
        mode="min",
        save_weights_only=True,
    )
    
    # Train model
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        callbacks=[early_stopping],
        config=tez_config,
    )
    
    print(f"Training fold {fold} complete\n")


# Initialize configuration
config = TrainingConfig()

# Load data
df = pd.read_csv("/kaggle/input/kfold-petfinder-my-pawpularity-contest/train_5folds.csv")
print(f"Dataset loaded with {len(df)} samples")
print(f"Data columns: {df.columns.tolist()}")
print(f"Fold distribution:\n{df['kfold'].value_counts().sort_index()}")

# Train all folds
for fold in range(config.num_folds):
    train_fold(fold, df, config)

print("All folds training completed!")

