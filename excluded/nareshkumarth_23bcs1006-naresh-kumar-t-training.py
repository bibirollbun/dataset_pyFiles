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
        else:
            image = cv2.resize(image, (args.image_size, args.image_size)).astype(np.float32)
            image = image / 255.0
            image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = np.ascontiguousarray(image)

        features = self.dense_features[item, :] if (self.dense_features is not None and self.dense_features.size) else np.zeros((0,), dtype=np.float32)
        targets = self.targets[item]

        return {
            "image": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.float),
        }




import sklearn.metrics as skm

class CustomModel(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        # backbone returns features (no classifier head)
        self.backbone = timm.create_model("resnet50", pretrained=True, in_chans=3, num_classes=0, global_pool="avg")
        nf = getattr(self.backbone, "num_features", 2048)

        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(nf, 1)

        if pos_weight is not None:
            # BCEWithLogitsLoss expects pos_weight on the positive class
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

        self.step_scheduler_after = "epoch"

    def optimizer_scheduler(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-6)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1, eta_min=1e-6)
        return opt, sch

    def forward(self, image, features, targets=None):
        feats = self.backbone(image)
        feats = self.dropout(feats)
        logits = self.out(feats).view(-1)

        if targets is not None:
            t = targets.view(-1).type_as(logits)
            loss = self.loss_fn(logits, t)
            with torch.no_grad():
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                try:
                    auc = float(skm.roc_auc_score(t.detach().cpu().numpy(), probs))
                except Exception:
                    auc = 0.5
            return logits.unsqueeze(1), loss, {
                "binaryloss": loss.detach(), # CHANGED from "valid_binaryloss"
                "valid_auc": torch.tensor(auc, device=logits.device)
            }
        return logits.unsqueeze(1), 0, {}




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


df = pd.read_csv("/kaggle/input/naresh-kumar-t-23bcs10006-k-folds/train_5folds.csv")
df.head()


i=0
print(f'training fold: {i} start')
args.fold = 0
df_train = df[df.kfold != args.fold].reset_index(drop=True)
df_valid = df[df.kfold == args.fold].reset_index(drop=True)
pos = df_train.target.sum()
neg = len(df_train) - pos
pos_weight = torch.tensor((neg / (pos + 1e-6))).float().cuda() if torch.cuda.is_available() else torch.tensor((neg / (pos + 1e-6))).float()

dense_features = [
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

model = CustomModel(pos_weight=pos_weight)
model = Tez(model)

config = TezConfig(
    training_batch_size=args.batch_size,
    validation_batch_size=2 * args.batch_size,
    epochs=args.epochs,
    step_scheduler_after="epoch",
    step_scheduler_metric="valid_binaryloss", # <--- CHANGE TO "valid_binaryloss"
    fp16=True,
    val_strategy="epoch",
)

es = EarlyStopping(
    monitor="valid_binaryloss",
    model_path=f"model_f{args.fold}.bin",
    patience=4,
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




