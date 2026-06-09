import sys

sys.path.append("../input/tez-lib/")


import os
import random
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import albumentations
from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping

# --- CONFIGURATION ---
class CFG:
    seed = 42
    model_name = 'resnet50'
    image_size = 384
    batch_size = 8
    epochs = 10
    lr = 2.5e-05
    weight_decay = 0.01
    num_folds = 5
    patience = 4
    
    # Paths
    input_dir = "/kaggle/input/"
    train_csv_path = os.path.join(input_dir, "aditya-10148-kfold-petfinder-my/train_5folds.csv")
    train_img_dir = os.path.join(input_dir, "petfinder-pawpularity-score/train/")

# --- SEEDING ---
def seed_everything(seed_value):
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True

seed_everything(CFG.seed)


class PetFinderDataset:
    """Custom dataset for loading images, metadata, and targets."""
    def __init__(self, image_paths, dense_features, targets, augmentations):
        self.image_paths = image_paths
        self.dense_features = dense_features
        self.targets = targets
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        # Load image and convert to RGB
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentations
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
            
        # Transpose for PyTorch format (C, H, W)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        # Get dense features and target
        features = self.dense_features[index, :]
        target = self.targets[index]
        
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float),
            "targets": torch.tensor(target, dtype=torch.float),
        }


class PawpularityNet(nn.Module):
    """Neural network for predicting Pawpularity."""
    def __init__(self):
        super().__init__()
        # Image backbone
        self.backbone = timm.create_model(CFG.model_name, pretrained=True, in_chans=3)
        num_backbone_features = self.backbone.get_classifier().in_features
        
        # The final layer is removed to use the backbone as a feature extractor
        self.backbone.reset_classifier(0)

        # Head for combining features and making a prediction
        num_dense_features = 12
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_backbone_features + num_dense_features, 1)
        )
        
    def monitor_metrics(self, outputs, targets, loss):
        # RMSE is the square root of MSE, which is our loss
        return {"rmse": torch.sqrt(loss)}

    def optimizer_scheduler(self):
        opt = torch.optim.AdamW(self.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=1, eta_min=1e-6
        )
        return opt, sch

    def forward(self, image, features, targets=None):
        # Feature extraction
        image_features = self.backbone(image)
        
        # Combine image and dense features
        combined_features = torch.cat([image_features, features], dim=1)
        
        # Prediction
        output = self.head(combined_features)

        if targets is not None:
            loss = nn.MSELoss()(output, targets.view(-1, 1))
            metrics = self.monitor_metrics(output, targets, loss)
            return output, loss, metrics
        
        return output, 0, {}


# --- DATA LOADING ---
df = pd.read_csv(CFG.train_csv_path)
dense_features = [
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
    'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
]

# --- AUGMENTATIONS ---
train_aug = albumentations.Compose([
    albumentations.LongestMaxSize(CFG.image_size, p=1),
    albumentations.PadIfNeeded(CFG.image_size, CFG.image_size, p=1, border_mode=cv2.BORDER_CONSTANT),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.1),
    albumentations.Rotate(limit=180, p=0.5),
    albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
    albumentations.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
    albumentations.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
])

valid_aug = albumentations.Compose([
    albumentations.LongestMaxSize(CFG.image_size, p=1),
    albumentations.PadIfNeeded(CFG.image_size, CFG.image_size, p=1, border_mode=cv2.BORDER_CONSTANT),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
])

# --- CROSS-VALIDATION LOOP ---
for fold in range(CFG.num_folds):
    print(f"{'='*20} FOLD {fold} {'='*20}")
    
    # Split data
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # Create datasets
    train_img_paths = [os.path.join(CFG.train_img_dir, f"{x}.jpg") for x in df_train["Id"].values]
    valid_img_paths = [os.path.join(CFG.train_img_dir, f"{x}.jpg") for x in df_valid["Id"].values]

    train_dataset = PetFinderDataset(
        image_paths=train_img_paths,
        dense_features=df_train[dense_features].values,
        targets=df_train.Pawpularity.values,
        augmentations=train_aug,
    )
    
    valid_dataset = PetFinderDataset(
        image_paths=valid_img_paths,
        dense_features=df_valid[dense_features].values,
        targets=df_valid.Pawpularity.values,
        augmentations=valid_aug,
    )
    
    # Initialize model and trainer
    model = PawpularityNet()
    trainer = Tez(model)
    
    # Configure training process
    config = TezConfig(
        training_batch_size=CFG.batch_size,
        validation_batch_size=2 * CFG.batch_size,
        epochs=CFG.epochs,
        step_scheduler_after="epoch",
        step_scheduler_metric="valid_rmse",
        fp16=True,
    )
    
    es_callback = EarlyStopping(
        monitor="valid_rmse",
        model_path=f"model_fold_{fold}.bin",
        patience=CFG.patience,
        mode="min",
        save_weights_only=True,
    )
    
    # Run training
    trainer.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        callbacks=[es_callback],
        config=config,
    )
    print(f"Fold {fold} training complete.\n")





































