import sys
# Append external library path to system modules
sys.path.append("../input/tez-lib/")


# import timm
# from pprint import pprint
# model_names = timm.list_models(pretrained=True)
# pprint(model_names)


# Essential libraries for reproducibility and deep learning
import random
import numpy as np
import torch
import os

def set_random_seed(seed_value):
    """
    Initialize all random number generators with a fixed seed
    to ensure reproducible results across runs
    """
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    
# Set seed for reproducibility
RANDOM_SEED = 42
set_random_seed(RANDOM_SEED)


# Core dependencies for model training and evaluation
import torch.nn as nn
from tez import Tez, TezConfig
import tez
import albumentations
import pandas as pd
import cv2
import numpy as np
import timm
import torch.nn as nn
from sklearn import metrics
import torch
from tez.callbacks import EarlyStopping
from tqdm import tqdm


# Configuration parameters for model training
class TrainingConfig:
    batch_size = 8
    image_size = 384
    epochs = 1  # Increase to 10 for full training
    fold = 0


class PetImageDataset:
    """
    Custom Dataset class for loading pet images along with metadata features
    and target popularity scores
    """
    
    def __init__(self, img_paths, metadata_features, target_scores, transform_pipeline):
        self.img_paths = img_paths
        self.metadata_features = metadata_features
        self.target_scores = target_scores
        self.transform_pipeline = transform_pipeline
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Load and convert image from BGR to RGB
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transformations if provided
        if self.transform_pipeline is not None:
            transformed = self.transform_pipeline(image=img)
            img = transformed["image"]
        
        # Transpose image to channels-first format (C, H, W)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        
        # Extract metadata and target for current sample
        metadata = self.metadata_features[idx, :]
        target = self.target_scores[idx]
        
        # Return dictionary with tensors
        return {
            "image": torch.tensor(img, dtype=torch.float),
            "features": torch.tensor(metadata, dtype=torch.float),
            "targets": torch.tensor(target, dtype=torch.float),
        }


class PetPopularityPredictor(nn.Module):
    """
    Neural network model for predicting pet popularity scores
    Uses ResNet50 as backbone with custom regression head
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize pre-trained ResNet50 backbone
        self.backbone = timm.create_model("resnet50", pretrained=True, in_chans=3)
        
        # Regularization layer to prevent overfitting
        self.dropout_layer = nn.Dropout(0.5)
        
        # Final regression layer for popularity score prediction
        self.regression_head = nn.Linear(1000, 1)
        
        # Scheduler configuration
        self.step_scheduler_after = "epoch"

    def monitor_metrics(self, predictions, ground_truth, loss_value):
        """Calculate and return RMSE metric"""
        rmse_score = loss_value
        # Handle NaN values in loss
        if str(rmse_score) == 'nan':
            rmse_score = float('inf')
        return {"rmse": rmse_score}

    def optimizer_scheduler(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=2.5e-05, 
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10, 
            T_mult=1, 
            eta_min=1e-6, 
            last_epoch=-1
        )
        return optimizer, scheduler

    def forward(self, image, features, targets=None):
        """
        Forward pass through the network
        Args:
            image: Input pet image tensor
            features: Metadata features (currently not concatenated)
            targets: Ground truth popularity scores (optional)
        Returns:
            predictions, loss, metrics
        """
        # Pass image through backbone
        x = self.backbone(image)
        
        # Apply dropout for regularization
        x = self.dropout_layer(x)
        
        # Get final predictions
        predictions = self.regression_head(x)

        # Calculate loss and metrics if targets provided
        if targets is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(predictions, targets.view(-1, 1))
            metrics = self.monitor_metrics(predictions, targets, loss)
            return predictions, loss, metrics
        
        return predictions, 0, {}


# Data augmentation pipeline for training set
training_transforms = albumentations.Compose(
    [
        # Resize while maintaining aspect ratio
        albumentations.LongestMaxSize(TrainingConfig.image_size, p=1),
        albumentations.PadIfNeeded(
            TrainingConfig.image_size, 
            TrainingConfig.image_size, 
            p=1, 
            border_mode=0
        ),
        
        # Geometric augmentations
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.1),
        albumentations.Rotate(limit=180, p=0.5),
        albumentations.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.1, 
            rotate_limit=45, 
            p=0.5
        ),
        
        # Color augmentations
        albumentations.HueSaturationValue(
            hue_shift_limit=0.2, 
            sat_shift_limit=0.2, 
            val_shift_limit=0.2, 
            p=0.5
        ),
        albumentations.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), 
            contrast_limit=(-0.1, 0.1), 
            p=0.5
        ),
        
        # Normalization using ImageNet statistics
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

# Validation transforms - only resize and normalize
validation_transforms = albumentations.Compose(
    [
        # Maintain aspect ratio during resize
        albumentations.LongestMaxSize(TrainingConfig.image_size, p=1),
        albumentations.PadIfNeeded(
            TrainingConfig.image_size, 
            TrainingConfig.image_size, 
            p=1, 
            border_mode=0
        ),
        
        # Apply same normalization as training
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)


# Load pre-split dataset with k-fold cross validation
data_df = pd.read_csv("/kaggle/input/same-old-creating-folds/train_5folds.csv")
data_df.head()


# Training loop for fold 0
current_fold = 0
print(f'Starting training for fold: {current_fold}')

# Set current fold
TrainingConfig.fold = 0

# Split data into training and validation sets based on fold
train_df = data_df[data_df.kfold != TrainingConfig.fold].reset_index(drop=True)
val_df = data_df[data_df.kfold == TrainingConfig.fold].reset_index(drop=True)

# Define metadata feature columns
metadata_cols = [
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
    'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
]

# Construct image file paths
train_image_paths = [
    f"../input/petfinder-pawpularity-score/train/{img_id}.jpg" 
    for img_id in train_df["Id"].values
]
val_image_paths = [
    f"../input/petfinder-pawpularity-score/train/{img_id}.jpg" 
    for img_id in val_df["Id"].values
]

# Initialize training dataset with augmentations
train_data = PetImageDataset(
    img_paths=train_image_paths,
    metadata_features=train_df[metadata_cols].values,
    target_scores=train_df.Pawpularity.values,
    transform_pipeline=training_transforms,
)

# Initialize validation dataset
val_data = PetImageDataset(
    img_paths=val_image_paths,
    metadata_features=val_df[metadata_cols].values,
    target_scores=val_df.Pawpularity.values,
    transform_pipeline=validation_transforms,
)

# Initialize model and wrap with Tez
network = PetPopularityPredictor()
network = Tez(network)

# Configure training parameters
train_config = TezConfig(
    training_batch_size=TrainingConfig.batch_size,
    validation_batch_size=2 * TrainingConfig.batch_size,
    epochs=TrainingConfig.epochs,
    step_scheduler_after="epoch",
    step_scheduler_metric="valid_rmse",
    fp16=True,
    val_strategy="batch",
    val_steps=900,
)

# Setup early stopping callback
early_stop = EarlyStopping(
    monitor="valid_rmse",
    model_path=f"model_f{TrainingConfig.fold}.bin",
    patience=4,
    mode="min",
    save_weights_only=True,
)

# Start training process
network.fit(
    train_data,
    valid_dataset=val_data,
    callbacks=[early_stop],
    config=train_config,
)

print(f'Completed training for fold: {current_fold}')





































