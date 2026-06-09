import sys
sys.path.append("../input/tez-lib/")

import random
import numpy as np
import torch
import os

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(42)

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

class args:
    batch_size = 8
    image_size = 384
    epochs = 10
    fold = 0


class CustomDataset:
    def __init__(self, image_paths, dense_features, targets, augmentations):
        self.image_paths = image_paths
        self.dense_features = dense_features
        self.targets = targets
        self.augmentations = augmentations
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
            
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        features = self.dense_features[item, :]
        targets = self.targets[item]
        
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.float),
        }


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()        
        self.model = timm.create_model("resnet50", pretrained=True, in_chans=3)
        
        # Get the number of features from the model
        n_features = self.model.get_classifier().in_features
        
        # Remove the default classifier
        self.model.reset_classifier(0)
        
        self.dropout = nn.Dropout(0.5)
        # Output should be 1 for binary classification with sigmoid
        self.out = nn.Linear(n_features, 1)
        
        self.step_scheduler_after = "epoch"

    def monitor_metrics(self, outputs, targets):
        """Calculate ROC AUC score with NaN handling"""
        outputs_np = outputs.sigmoid().cpu().detach().numpy().flatten()
        targets_np = targets.cpu().detach().numpy().flatten()
        
        try:
            # Check if we have both classes in targets
            if len(np.unique(targets_np)) < 2:
                auc = 0.5
            else:
                auc = metrics.roc_auc_score(targets_np, outputs_np)
                # Handle NaN cases
                if np.isnan(auc) or np.isinf(auc):
                    auc = 0.5
        except:
            auc = 0.5
        
        # Return as tensor for Tez compatibility
        return {"auc": torch.tensor(auc, device=outputs.device)}

    def optimizer_scheduler(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return opt, sch

    def forward(self, image, features, targets=None):
        # Get features from backbone
        x = self.model(image)
        x = self.dropout(x)
        # Output logits (no activation here)
        x = self.out(x)

        if targets is not None:
            # Use BCEWithLogitsLoss for binary classification (more stable)
            loss = nn.BCEWithLogitsLoss()(x.view(-1), targets.view(-1))
            metrics = self.monitor_metrics(x, targets)
            return x, loss, metrics
        return x, 0, {}


train_aug = albumentations.Compose(
    [
        albumentations.LongestMaxSize(args.image_size, p=1),
        albumentations.PadIfNeeded(args.image_size, args.image_size, p=1, border_mode=0),
        
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
    ],
    p=1.0,
)

valid_aug = albumentations.Compose(
    [
        albumentations.LongestMaxSize(args.image_size, p=1),
        albumentations.PadIfNeeded(args.image_size, args.image_size, p=1, border_mode=0),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

df = pd.read_csv("/kaggle/input/deepak-10092-pawpularity/train_5folds.csv")
df.head()

i = 0
print(f'training fold: {i} start')
args.fold = 0
df_train = df[df.kfold != args.fold].reset_index(drop=True)
df_valid = df[df.kfold == args.fold].reset_index(drop=True)

dense_features = []

# REPLACE 'Id' WITH YOUR ACTUAL COLUMN NAME
# Common options: 'Id', 'image_id', 'image', etc.
train_img_paths = [f"/kaggle/input/siim-isic-melanoma-classification/jpeg/train/{x}.jpg" for x in df_train["image_name"].values]
valid_img_paths = [f"/kaggle/input/siim-isic-melanoma-classification/jpeg/train/{x}.jpg" for x in df_valid["image_name"].values]

train_dataset = CustomDataset(
    image_paths=train_img_paths,
    dense_features=df_train[dense_features].values,
    targets=df_train.target.values,
    augmentations=train_aug,
)

valid_dataset = CustomDataset(
    image_paths=valid_img_paths,
    dense_features=df_valid[dense_features].values,
    targets=df_valid.target.values,
    augmentations=valid_aug,
)

model = CustomModel()
model = Tez(model)

config = TezConfig(
    training_batch_size=args.batch_size,
    validation_batch_size=2 * args.batch_size,
    epochs=args.epochs,
    step_scheduler_after="epoch",
    step_scheduler_metric="valid_auc",
    fp16=True,
    val_strategy="batch",
    val_steps=900,
)

es = EarlyStopping(
    monitor="valid_auc",
    model_path=f"model_f{args.fold}.bin",
    patience=4,
    mode="max",
    save_weights_only=True,
)

model.fit(
    train_dataset,
    valid_dataset=valid_dataset,
    callbacks=[es],
    config=config,
)

print(f'training fold: {i} complete')































