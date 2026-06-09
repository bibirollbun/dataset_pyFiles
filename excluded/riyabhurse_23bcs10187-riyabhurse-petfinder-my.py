import sys
# sys is a built-in Python module that lets you interact with the Python runtime environment.
sys.path.append("../input/tez-lib/")
# sys.path is a list of directories Python searches when you import a module.
# sys.path.append("../input/tez-lib/") adds a new directory to that list.


import random
import numpy as np
import pandas as pd
import torch
import os
import cv2
import timm
import albumentations
import torch.nn as nn
from sklearn import metrics
from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping
from tqdm import tqdm


# import timm
# from pprint import pprint
# model_names = timm.list_models(pretrained=True)
# pprint(model_names)



# ===========================
# âš™ï¸� Reproducibility
# ===========================
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)

# ===========================
# âš™ï¸� Training Configuration
# ===========================
class args:
    batch_size = 16
    image_size = 380
    epochs = 10
    fold = 0


# ===========================
# ğŸ§  Dataset Definition
# ===========================
class PawpularDataset:
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


# ===========================
# ğŸ§© Model Definition
# ===========================
class PawpularModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Stronger backbone: EfficientNet
        self.model = timm.create_model("tf_efficientnet_b4_ns", pretrained=True, in_chans=3)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()

        # Metadata network
        self.meta = nn.Sequential(
            nn.Linear(12, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Fusion + final output
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(in_features + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, image, features, targets=None):
        x = self.model(image)
        meta_out = self.meta(features)
        x = torch.cat([x, meta_out], dim=1)
        x = self.fc(x)

        if targets is not None:
            loss = nn.MSELoss()(x, targets.view(-1, 1))
            rmse = torch.sqrt(loss)
            return x, loss, {"rmse": rmse}
        return x, 0, {}

    def optimizer_scheduler(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=5, eta_min=1e-6
        )
        return opt, sch# ===========================
# ğŸ§  Dataset Definition
# ===========================
class PawpularDataset:
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


# ===========================
# ğŸ§ª Data Augmentations
# ===========================
train_aug = albumentations.Compose(
    [
        albumentations.RandomResizedCrop(args.image_size, args.image_size, scale=(0.8, 1.0), p=1.0),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.2),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=25, p=0.5),
        albumentations.RandomBrightnessContrast(p=0.5),
        albumentations.HueSaturationValue(p=0.3),
        albumentations.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    ],
    p=1.0,
)

valid_aug = albumentations.Compose(
    [
        albumentations.CenterCrop(args.image_size, args.image_size, p=1.0),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    ],
    p=1.0,
)


# ===========================
# ğŸ“‚ Load Data
# ===========================
df = pd.read_csv("/kaggle/input/same-old-creating-folds/train_5folds.csv")

dense_features = [
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
    'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
]


# ===========================
# ğŸ�‹ï¸� 5-Fold Training Loop
# ===========================
for i in range(5):
    print(f"ğŸŒ€ Training Fold: {i}")
    args.fold = i

    df_train = df[df.kfold != args.fold].reset_index(drop=True)
    df_valid = df[df.kfold == args.fold].reset_index(drop=True)

    train_img_paths = [f"../input/petfinder-pawpularity-score/train/{x}.jpg" for x in df_train["Id"].values]
    valid_img_paths = [f"../input/petfinder-pawpularity-score/train/{x}.jpg" for x in df_valid["Id"].values]

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

    model = PawpularModel()
    model = Tez(model)

    config = TezConfig(
        training_batch_size=args.batch_size,
        validation_batch_size=2 * args.batch_size,
        epochs=args.epochs,
        step_scheduler_after="epoch",
        step_scheduler_metric="valid_rmse",
        fp16=True,
        val_strategy="epoch",
    )

    es = EarlyStopping(
        monitor="valid_rmse",
        model_path=f"best_model_fold{i}.bin",
        patience=3,
        mode="min",
        save_weights_only=True,
    )

    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        callbacks=[es],
        config=config,
    )

    print(f"âœ… Fold {i} training complete.\n")

print("ğŸ�‰ All folds complete! Best models saved as best_model_fold0-4.bin")






































