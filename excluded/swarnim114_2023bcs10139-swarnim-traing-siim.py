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
    torch.backends.cudnn.deterministic = False  # Changed for speed
    torch.backends.cudnn.benchmark = True  # Enable for speed
    
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
    batch_size = 32  # Increased for speed
    image_size = 224  # Reduced further
    epochs = 3  # Much fewer epochs
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
        self.model = timm.create_model("resnet34", pretrained=True, in_chans=3)  # Lighter model
        
        n_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(n_features + 1, 1)
        
        self.step_scheduler_after = "epoch"

    def monitor_metrics(self, outputs, targets, loss):
        valid_binaryloss = loss
        if str(valid_binaryloss) == 'nan':
            valid_binaryloss = float('inf')
        return {"binaryloss": valid_binaryloss}

    def optimizer_scheduler(self):
        opt = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=0.01)  # Higher LR
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
        return opt, sch

    def forward(self, image, features, targets=None):
        x = self.model(image)
        x = self.dropout(x)
        x = torch.cat([x, features], dim=1)
        x = self.out(x)

        if targets is not None:
            loss = nn.BCEWithLogitsLoss()(x, targets.view(-1, 1).type_as(x))
            metrics = self.monitor_metrics(x, targets, loss)
            return x, loss, metrics
        return x, 0, {}


# Minimal augmentations for speed
train_aug = albumentations.Compose(
    [
        albumentations.Resize(args.image_size, args.image_size, p=1),  # Direct resize
        
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Rotate(limit=20, p=0.3),  # Reduced
        
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
        albumentations.Resize(args.image_size, args.image_size, p=1),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)


df = pd.read_csv("/kaggle/input/2023bcs10139-swarnim-kfold-melonama/train_5folds.csv")
df.head()


print(f'Training fold: {args.fold} start')

df_train = df[df.kfold != args.fold].reset_index(drop=True)
df_valid = df[df.kfold == args.fold].reset_index(drop=True)

dense_features = ['age_approx']

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
    validation_batch_size=args.batch_size * 2,
    epochs=args.epochs,
    step_scheduler_after="epoch",
    step_scheduler_metric="valid_binaryloss",
    fp16=True,
    val_strategy="epoch",
)

es = EarlyStopping(
    monitor="valid_binaryloss",
    model_path=f"model_f{args.fold}.bin",
    patience=2,
    mode="min",
    save_weights_only=True,
)

model.fit(
    train_dataset,
    valid_dataset=valid_dataset,
    callbacks=[es],
    config=config,
)

print(f'Training fold: {args.fold} complete')
print(f'Best model saved at: model_f{args.fold}.bin')


































