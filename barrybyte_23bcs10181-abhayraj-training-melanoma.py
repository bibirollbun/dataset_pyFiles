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
    
seed_everything(69)


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
    epochs = 2
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
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(1000, 1)
        self.step_scheduler_after = "epoch"


    def monitor_metrics(self, outputs, targets, loss):
        valid_binaryloss = loss
        if str(valid_binaryloss) == 'nan':
            valid_binaryloss = float('inf')
        return {"binaryloss": valid_binaryloss}

    def optimizer_scheduler(self):
        opt = torch.optim.AdamW(self.parameters(), lr=2.5e-05, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return opt,sch

    def forward(self, image, features, targets=None):

        x = self.model(image)
        x = self.dropout(x)
        x = self.out(x)

        if targets is not None:
            loss = nn.BCEWithLogitsLoss()(x, targets.view(-1, 1).type_as(x))
            metrics = self.monitor_metrics(x, targets, loss)
            return x, loss, metrics
        return x, 0, {}




train_aug = albumentations.Compose([
    albumentations.LongestMaxSize(args.image_size, p=1),
    albumentations.PadIfNeeded(args.image_size, args.image_size, p=1, border_mode=cv2.BORDER_REFLECT_101),

    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.1),
    albumentations.RandomRotate90(p=0.2),  
    albumentations.ShiftScaleRotate(
        shift_limit=0.1, scale_limit=0.15, rotate_limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.5
    ),

    albumentations.OneOf([
        albumentations.HueSaturationValue(20, 30, 20, p=0.7),
        albumentations.RGBShift(10, 10, 10, p=0.3),
    ], p=0.5),

    albumentations.RandomBrightnessContrast(0.15, 0.15, p=0.5),
    albumentations.CLAHE(clip_limit=2.0, p=0.2),  

    albumentations.OneOf([
        albumentations.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        albumentations.MotionBlur(blur_limit=3, p=0.2),
    ], p=0.2),

    albumentations.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
    ),
], p=1.0)

valid_aug = albumentations.Compose([
    albumentations.LongestMaxSize(args.image_size, p=1),
    albumentations.PadIfNeeded(args.image_size, args.image_size, p=1, border_mode=cv2.BORDER_REFLECT_101),
    albumentations.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
    ),
], p=1.0)



df = pd.read_csv("/kaggle/input/23bcs10181-abhayraj-kfold-melanoma/train_5folds.csv")
df.head()


i=0
print(f'training fold: {i} start')
args.fold = 0
df_train = df[df.kfold != args.fold].reset_index(drop=True)
df_valid = df[df.kfold == args.fold].reset_index(drop=True)
dense_features = [
    'age_approx'
]
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
    step_scheduler_metric="valid_binaryloss",
    fp16=True,
    # fp16=False,
    val_strategy="batch",
    val_steps=900,
)

es = EarlyStopping(
    monitor="valid_binaryloss",
    model_path=f"model_f{args.fold}.bin",
    patience=4,#3,
    mode="min",
    save_weights_only=True,
)

model.fit(
    train_dataset,
    valid_dataset=valid_dataset,
    callbacks=[es],
    config=config,
)
print(f'training fold: {i} complete')




