import sys
sys.path.append("../input/tez-lib/")


import torch.nn as nn
import tez
import albumentations
import pandas as pd
import cv2
import numpy as np
import timm
import torch.nn as nn
import torch
import random
import numpy as np
import albumentations as A
import torch
import os


from tqdm import tqdm
from tez.callbacks import EarlyStopping
from tez import Tez, TezConfig
from sklearn import metrics


def set_global_seed(seed_value):
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True

set_global_seed(42)


class ProcessArguments:
    sample_batch_size = 8
    sample_image_size = 384
    sample_epochs = 1 #10
    sample_fold = 0


class CustomDataset:
    def __init__(self, image_paths, dense_features, targets, augmentations=None):
        self.img_paths = image_paths
        self.tabular_features = dense_features
        self.labels = targets
        self.transforms = augmentations

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # Read image
        image_file = self.img_paths[index]
        image_data = cv2.imread(image_file)
        if image_data is None:
            raise FileNotFoundError(f"Could not load image: {image_file}")

        # Convert to RGB
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        # Apply augmentations (if available)
        if self.transforms:
            augmented_result = self.transforms(image=image_data)
            image_data = augmented_result["image"]

        # Convert (H, W, C) → (C, H, W)
        image_data = np.moveaxis(image_data, -1, 0).astype(np.float32)

        # Collect tabular features and labels
        features_vec = self.tabular_features[index, :]
        label_value = self.labels[index]

        return {
            "image": torch.tensor(image_data, dtype=torch.float32),
            "features": torch.tensor(features_vec, dtype=torch.float32),
            "targets": torch.tensor(label_value, dtype=torch.float32),
        }


import torch
import torch.nn as nn
import timm

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Base model backbone
        self.backbone = timm.create_model(
            "resnet50", pretrained=True, in_chans=3
        )

        self.dropout_layer = nn.Dropout(0.5)
        self.output_layer = nn.Linear(1000, 1)

        # Used by Tez
        self.step_scheduler_after = "epoch"

    def monitor_metrics(self, outputs, targets, loss_value):
        metric_val = loss_value
        if str(metric_val) == "nan":
            metric_val = float("inf")
        return {"rmse": metric_val}

    def optimizer_scheduler(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=2.5e-5, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return optimizer, scheduler

    def forward(self, image, features, targets=None):
        # match dataset keys exactly: "image", "features", "targets"
        x = self.backbone(image)
        x = self.dropout_layer(x)

        # if you decide to use dense features, uncomment this:
        # x = torch.cat([x, features], dim=1)

        preds = self.output_layer(x)

        if targets is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(preds, targets.view(-1, 1).type_as(preds))
            metrics = self.monitor_metrics(preds, targets, loss)
            return preds, loss, metrics

        return preds, 0, {}



# --- Training augmentations ---
train_augmentations = A.Compose(
    [
        A.LongestMaxSize(max_size=ProcessArguments.sample_image_size, p=1.0),
        A.PadIfNeeded(
            min_height=ProcessArguments.sample_image_size,
            min_width=ProcessArguments.sample_image_size,
            border_mode=0,
            p=1.0
        ),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Rotate(limit=180, p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            p=0.5
        ),

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
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
    ],
    p=1.0,
)

# --- Validation augmentations ---
valid_augmentations = A.Compose(
    [
        A.LongestMaxSize(max_size=ProcessArguments.sample_image_size, p=1.0),
        A.PadIfNeeded(
            min_height=ProcessArguments.sample_image_size,
            min_width=ProcessArguments.sample_image_size,
            border_mode=0,
            p=1.0
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
    ],
    p=1.0,
)


df = pd.read_csv("/kaggle/input/shreyas-10155-same-old-creating-folds/train_5folds.csv")


from tez.callbacks import EarlyStopping
from tez import Tez, TezConfig

# --- Training Setup ---
fold_idx = 0
print(f"Training fold: {fold_idx} start")

# Assign current fold
ProcessArguments.sample_fold = fold_idx

# --- Split data ---
df_train = df[df.kfold != ProcessArguments.sample_fold].reset_index(drop=True)
df_valid = df[df.kfold == ProcessArguments.sample_fold].reset_index(drop=True)

dense_features = []  # No dense features used for now

# --- Image paths ---
train_img_paths = [
    f"/kaggle/input/siim-isic-melanoma-classification/jpeg/train/{img_name}.jpg"
    for img_name in df_train["image_name"].values
]
valid_img_paths = [
    f"/kaggle/input/siim-isic-melanoma-classification/jpeg/train/{img_name}.jpg"
    for img_name in df_valid["image_name"].values
]

# --- Datasets ---
train_dataset = CustomDataset(
    image_paths=train_img_paths,
    dense_features=df_train[dense_features].values,
    targets=df_train.target.values,
    augmentations=train_augmentations,
)

valid_dataset = CustomDataset(
    image_paths=valid_img_paths,
    dense_features=df_valid[dense_features].values,
    targets=df_valid.target.values,
    augmentations=valid_augmentations,
)

# --- Model ---
model = CustomModel()
tez_model = Tez(model)

# --- Configuration ---
config = TezConfig(
    training_batch_size=ProcessArguments.sample_batch_size,
    validation_batch_size=2 * ProcessArguments.sample_batch_size,
    epochs=ProcessArguments.sample_epochs,
    step_scheduler_after="epoch",
    step_scheduler_metric="valid_rmse",
    fp16=True,
    val_strategy="batch",
    val_steps=900,
)

# --- Early stopping ---
early_stopper = EarlyStopping(
    monitor="valid_rmse",
    model_path=f"model_f{ProcessArguments.sample_fold}.bin",
    patience=4,
    mode="min",
    save_weights_only=True,
)

# --- Train ---
tez_model.fit(
    train_dataset,
    valid_dataset=valid_dataset,
    callbacks=[early_stopper],
    config=config,
)

print(f"Training fold: {fold_idx} complete")










