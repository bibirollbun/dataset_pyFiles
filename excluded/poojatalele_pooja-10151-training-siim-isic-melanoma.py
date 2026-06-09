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
    epochs = 1
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
            # "features": torch.tensor(features, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.float),
        }


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()        
        self.model = timm.create_model("tf_efficientnet_b3_ns", pretrained=True, in_chans=3)

        
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

# # # === PATCH E: stronger backbone + metadata head ===
# import timm, torch
# import torch.nn as nn
# import torch.optim as optim

# class CustomModel(nn.Module):
#     def __init__(self, model_name="tf_efficientnet_b3_ns", n_meta=0, pretrained=True):
#         super().__init__()
#         self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
#         in_feats = self.backbone.num_features

#         self.use_meta = n_meta > 0
#         if self.use_meta:
#             self.meta = nn.Sequential(
#                 nn.Linear(n_meta, 32),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(0.2),
#             )
#             in_feats += 32

#         self.head = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(in_feats, 1)
#         )

#         # Tez expects a loss function on the model
#         self.loss_fn = nn.BCEWithLogitsLoss()   # you can overwrite with pos_weight later

#     def forward(self, image=None, features=None, targets=None, **kwargs):
#         """
#         Tez passes batch via kwargs: image, features, targets.
#         Support both your older (x_img/x_meta) style and Tez style.
#         """
#         # Fallbacks for any legacy names, if present
#         x_img = kwargs.get("x_img", image)
#         x_meta = kwargs.get("x_meta", features)

#         f = self.backbone(x_img)                   # [B, C]
#         if self.use_meta and x_meta is not None:
#             m = self.meta(x_meta)                  # [B, 32]
#             f = torch.cat([f, m], dim=1)

#         logits = self.head(f).view(-1)             # [B]

#         loss = None
#         if targets is not None:
#             loss = self.loss_fn(logits, targets.float())

#         return logits, loss

#     def optimizer_scheduler(self, *args, **kwargs):
#         """
#         Tez-compat: some versions call with (train_len, epochs),
#         others call with no args. Handle both.
#         """
#         import math
#         epochs = kwargs.get("epochs", None)
#         if epochs is None and len(args) >= 2:
#             # legacy: (train_len, epochs)
#             epochs = args[1]
#         if epochs is None:
#             epochs = 10  # safe default; Tez may override lr per-epoch anyway
    
#         optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
#         return optimizer, scheduler
    
#     # Optional shims for older Tez variants that look for these names:
#     def optimizer(self):
#         return optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
    
#     def scheduler(self, optimizer):
#         return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


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


df = pd.read_csv("/kaggle/input/creating-folds-siim-isic-melanoma/train_5folds.csv")
df.head()


from sklearn.model_selection import StratifiedGroupKFold

if ("kfold" not in df.columns) or (df["kfold"].nunique() < 2):
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    df["kfold"] = -1
    for f, (tr, va) in enumerate(sgkf.split(df, df["target"], groups=df["patient_id"])):
        df.loc[va, "kfold"] = f

meta_cat = ["sex", "anatom_site_general_challenge"]
df = pd.get_dummies(df, columns=meta_cat, dummy_na=True)

meta_onehot_cols = [c for c in df.columns if c.startswith("sex_") or c.startswith("anatom_site_general_challenge_")]


for i in [0,1,2]:
    print(f"training fold: {i} start")
    args.fold = i
    df_train = df[df.kfold != i].reset_index(drop=True)
    df_valid = df[df.kfold == i].reset_index(drop=True)

    N_pos = int((df_train.target == 1).sum())
    N_neg = int((df_train.target == 0).sum())
    pos_weight = torch.tensor([N_neg / max(1, N_pos)], dtype=torch.float32)
    
    age_mean = df_train["age_approx"].mean()
    age_std  = df_train["age_approx"].std() + 1e-6
    df_train.loc[:, "age_approx"] = (df_train["age_approx"] - age_mean) / age_std
    df_valid.loc[:, "age_approx"] = (df_valid["age_approx"] - age_mean) / age_std

    if True:  
        pos_idx = df_train.index[df_train.target == 1]
        neg_idx = df_train.index[df_train.target == 0]
        rep = max(1, len(neg_idx) // max(1, len(pos_idx)))
        new_idx = list(neg_idx) + list(np.repeat(pos_idx, rep))
        df_train = df_train.loc[new_idx].sample(frac=1.0, random_state=42).reset_index(drop=True)

    dense_features = ["age_approx"] + meta_onehot_cols

    df_train["age_approx"] = df_train["age_approx"].fillna(0.0)
    df_valid["age_approx"] = df_valid["age_approx"].fillna(0.0)
    
    for c in dense_features:
        df_train[c] = pd.to_numeric(df_train[c], errors="coerce").fillna(0.0)
        df_valid[c] = pd.to_numeric(df_valid[c], errors="coerce").fillna(0.0)
    
    X_train_meta = np.asarray(df_train[dense_features].values, dtype=np.float32)
    X_valid_meta = np.asarray(df_valid[dense_features].values, dtype=np.float32)

    train_img_paths = [f"/kaggle/input/siim-isic-melanoma-classification/jpeg/train/{x}.jpg"
                   for x in df_train["image_name"].values]
    valid_img_paths = [f"/kaggle/input/siim-isic-melanoma-classification/jpeg/train/{x}.jpg"
                   for x in df_valid["image_name"].values]

    train_dataset = CustomDataset(
        image_paths=train_img_paths,
        dense_features=X_train_meta,
        targets=df_train.target.values,
        augmentations=train_aug,
    )
    
    valid_dataset = CustomDataset(
        image_paths=valid_img_paths,
        dense_features=X_valid_meta,
        targets=df_valid.target.values,
        augmentations=valid_aug,
    )

    n_meta = len(dense_features)
    model = CustomModel() 
    model = Tez(model)
    
    config = TezConfig(
    training_batch_size=args.batch_size,
    validation_batch_size=2 * args.batch_size,
    epochs=args.epochs,
    step_scheduler_after="epoch",
    step_scheduler_metric="valid_binaryloss",
    fp16=True,
    )
    
    es = EarlyStopping(
        monitor="valid_binaryloss",
        model_path=f"model_f{i}.bin",
        patience=4,
        mode="min",
        save_weights_only=True,
    )
    model.fit(train_dataset, valid_dataset=valid_dataset, callbacks=[es], config=config)
    print(f"training fold: {i} complete")

