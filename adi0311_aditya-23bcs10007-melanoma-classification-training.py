import sys
sys.path.append("../input/tez-lib/tez-main")


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


df = pd.read_csv("/kaggle/input/aditya-23bcs10007-melanoma-classification-kfolds/train_5folds.csv")
df.head()


class args:
    batch_size = 8
    image_size = 384
    epochs = 10
    fold = 0


class MelanomaDataset:
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
        
        image = image.astype(np.float32) / 255.0   
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        features = self.dense_features[item, :].astype(np.float32)
        targets = self.targets[item]
        
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.float),
        }


class MelanomaModel(nn.Module):
    def __init__(self):
        super().__init__()        
        self.model = timm.create_model("resnet50", pretrained=True, num_classes=0)#resnet50#resnet101#eca_nfnet_l1#resnest101e

        
        self.dropout = nn.Dropout(0.5)# increase dropout

        self.out = nn.Linear(2066, 1)

        
        self.step_scheduler_after = "epoch"


    def monitor_metrics(self, outputs, targets, loss):
        return {"bce_loss": loss}

    def optimizer_scheduler(self):
        opt = torch.optim.AdamW(self.parameters(), lr=2.5e-05, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return opt,sch

    def forward(self, image, features, targets=None):

        x = self.model(image)
        x = self.dropout(x)
        x = torch.cat([x, features], dim=1)
        x = self.dropout(x)
        x = self.out(x)

        if targets is not None:
            targets = targets.view(-1, 1).float()
            loss = nn.BCEWithLogitsLoss()(x, targets)
            metrics = self.monitor_metrics(x, targets, loss)
            return x, loss, metrics
        return x, 0, {}


train_aug = albumentations.Compose(
    [

        albumentations.LongestMaxSize(args.image_size, p=1),
        albumentations.PadIfNeeded(args.image_size,args.image_size, p=1,border_mode=0),
        
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
        albumentations.PadIfNeeded(args.image_size,args.image_size, p=1,border_mode=0),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)


df["diagnosis"] = df["diagnosis"].fillna("unknown")
df["anatom_site_general_challenge"] = df["anatom_site_general_challenge"].fillna("unknown")

# One-hot encode desired columns, dtype uint8 -> ensures 0/1 integers and no float NaNs
to_encode = ["diagnosis", "anatom_site_general_challenge"]
df_encoded = pd.get_dummies(df, columns=to_encode, dtype=np.uint8)

# Keep 'sex' and 'age_approx' as numeric features (fill NaN)
if "sex" in df_encoded.columns:
    df_encoded["sex"] = df_encoded["sex"].replace({"male":1, "female":0}).fillna(0).astype(np.float32)
else:
    # If sex was a boolean column or similar
    if "sex" in df.columns:
        df_encoded["sex"] = df["sex"].fillna(0).astype(np.float32)
    else:
        df_encoded["sex"] = 0.0  # fallback

if "age_approx" in df_encoded.columns:
    df_encoded["age_approx"] = df_encoded["age_approx"].fillna(df_encoded["age_approx"].median()).astype(np.float32)
else:
    df_encoded["age_approx"] = 0.0

# define dense feature list (you can adapt to use only existing columns)
dense_features = [
    "sex", "age_approx"
]
# add diagnosis and anatom_site dummies present in df_encoded
for c in df_encoded.columns:
    if c.startswith("diagnosis_") or c.startswith("anatom_site_general_challenge_"):
        dense_features.append(c)

# final feature matrix
X = df_encoded[dense_features].fillna(0).values.astype(np.float32)
y = df_encoded["target"].values.astype(np.float32)
image_names = df_encoded["image_name"].values  # adjust if the column is named differently

print("Num features:", X.shape[1])
print("Feature names sample:", dense_features[:10])


df=df_encoded


for i in range(5):
    print(f"training fold: {i} start")
    args.fold = i

    df_train = df[df.kfold != args.fold].reset_index(drop=True)
    df_valid = df[df.kfold == args.fold].reset_index(drop=True)

    dense_features = [
        'sex', 'age_approx',
        'diagnosis_atypical melanocytic proliferation',
        'diagnosis_cafe-au-lait macule',
        'diagnosis_lentigo NOS',
        'diagnosis_lichenoid keratosis',
        'diagnosis_melanoma',
        'diagnosis_nevus',
        'diagnosis_seborrheic keratosis',
        'diagnosis_solar lentigo',
        'diagnosis_unknown',
        'anatom_site_general_challenge_head/neck',
        'anatom_site_general_challenge_lower extremity',
        'anatom_site_general_challenge_oral/genital',
        'anatom_site_general_challenge_palms/soles',
        'anatom_site_general_challenge_torso',
        'anatom_site_general_challenge_unknown',
        'anatom_site_general_challenge_upper extremity'
    ]

    train_img_paths = [
        f"/kaggle/input/siim-isic-melanoma-classification/jpeg/train/{x}.jpg"
        for x in df_train["image_name"].values
    ]
    valid_img_paths = [
        f"/kaggle/input/siim-isic-melanoma-classification/jpeg/train/{x}.jpg"
        for x in df_valid["image_name"].values
    ]
    
    train_feats = df_train[dense_features].fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
    valid_feats = df_valid[dense_features].fillna(0).clip(-1e6, 1e6).values.astype(np.float32)

    train_targets = df_train["target"].values.astype(np.float32)
    valid_targets = df_valid["target"].values.astype(np.float32)
    
    train_dataset = MelanomaDataset(
        image_paths=train_img_paths,
        dense_features=train_feats,
        targets=train_targets,
        augmentations=train_aug,
    )

    valid_dataset = MelanomaDataset(
        image_paths=valid_img_paths,
        dense_features=valid_feats,
        targets=valid_targets,
        augmentations=valid_aug,
    )

    model = MelanomaModel()  # must use BCEWithLogitsLoss inside
    model = Tez(model)

    config = TezConfig(
        training_batch_size=args.batch_size,
        validation_batch_size=2 * args.batch_size,
        epochs=args.epochs,
        step_scheduler_after="epoch",
        step_scheduler_metric="valid_bce_loss",  # updated metric
        fp16=True,
        val_strategy="batch",
        val_steps=max(1, len(valid_dataset) // (2 * args.batch_size)),
    )

    es = EarlyStopping(
        monitor="valid_bce_loss",  # updated to match metric name
        model_path=f"model_f{i}.bin",
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

    print(f"training fold: {i} complete")





