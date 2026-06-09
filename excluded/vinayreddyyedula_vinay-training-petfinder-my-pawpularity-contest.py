import sys
sys.path.append("../input/tez-lib/")


# import timm
# from pprint import pprint
# model_names = timm.list_models(pretrained=True)
# pprint(model_names)


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
    epochs = 12 #10
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
    def __init__(self, pos_weight: float = None):
        super().__init__()
        self.backbone = timm.create_model("resnet50", pretrained=True, in_chans=3, num_classes=0)
        in_features = getattr(self.backbone, "num_features", 2048)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(in_features, 1)
        self.step_scheduler_after = "epoch"

        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float))
        else:
            self.pos_weight = None

    def monitor_metrics(self, outputs, targets, loss):
        with torch.no_grad():
            # collect per-batch safely; impute neutral AUC=0.5 if single-class
            probs = torch.sigmoid(outputs).detach().view(-1)
            t = targets.detach().view(-1)
            t_np = t.cpu().numpy()
            p_np = probs.cpu().numpy()
            if len(np.unique(t_np)) < 2:
                auc_val = 0.5
            else:
                try:
                    auc_val = metrics.roc_auc_score(t_np, p_np)
                except Exception:
                    auc_val = 0.5
        return {
            "binaryloss": loss.detach(),
            "auc": torch.tensor(auc_val, device=outputs.device, dtype=outputs.dtype),
        }

    def optimizer_scheduler(self):
        opt = torch.optim.AdamW(self.parameters(), lr=2.5e-05, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return opt, sch

    def forward(self, image, features, targets=None):
        feats = self.backbone(image)
        x = self.dropout(feats)
        logits = self.out(x)
        if targets is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight) if self.pos_weight is not None else nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, targets.view(-1, 1).type_as(logits))
            metrics_dict = self.monitor_metrics(logits, targets, loss)
            return logits, loss, metrics_dict
        return logits, 0, {}



train_aug = albumentations.Compose(
    [
#         albumentations.Resize(args.image_size, args.image_size, p=1),
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
#         albumentations.Resize(args.image_size, args.image_size, p=1),
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


df = pd.read_csv("/kaggle/input/vinay-reddy-10083/train_5folds.csv")
df.head()


i = 0
print(f"training fold: {i} start")
args.fold = 0

# pick valid target column and normalize to 0/1 ints
target_candidates = ["Custom", "target", "label", "is_malignant", "malignant", "cancer"]
target_col = next((c for c in target_candidates if c in df.columns), None)
if target_col is None:
    raise KeyError(f"Target column not found. Tried: {target_candidates}. Available: {list(df.columns)}")
df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).clip(0, 1)

# split folds
df_train = df[df.kfold != args.fold].reset_index(drop=True)
df_valid = df[df.kfold == args.fold].reset_index(drop=True)

dense_features = []

# build paths
train_img_paths = [f"/kaggle/input/siim-isic-melanoma-classification/jpeg/train/{x}.jpg" for x in df_train["image_name"].values]
valid_img_paths = [f"/kaggle/input/siim-isic-melanoma-classification/jpeg/train/{x}.jpg" for x in df_valid["image_name"].values]

# datasets
train_dataset = CustomDataset(
    image_paths=train_img_paths,
    dense_features=df_train[dense_features].values,
    targets=df_train[target_col].values,
    augmentations=train_aug,
)

valid_dataset = CustomDataset(
    image_paths=valid_img_paths,
    dense_features=df_valid[dense_features].values,
    targets=df_valid[target_col].values,
    augmentations=valid_aug,
)

# class weight for BCEWithLogitsLoss
pos = int((df_train[target_col] == 1).sum())
neg = int((df_train[target_col] == 0).sum())
pos_weight = float(neg / max(pos, 1))

# model + trainer
model = CustomModel(pos_weight=pos_weight)
model = Tez(model)

config = TezConfig(
    training_batch_size=args.batch_size,
    validation_batch_size=2 * args.batch_size,
    epochs=args.epochs,
    step_scheduler_after="epoch",
    # IMPORTANT: do not pass a metric to CosineAnnealingWarmRestarts; step without metric to avoid NaN issues
    step_scheduler_metric=None,
    fp16=True,
    val_strategy="epoch",
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
print(f"training fold: {i} complete")






































