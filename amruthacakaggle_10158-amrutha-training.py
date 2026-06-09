import sys
sys.path.append("../input/tez-lib/")

import os
import random
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import timm
import albumentations

from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)

class CFG:
    batch_size = 8
    image_size = 384
    epochs = 10  
    fold = 0
    model_name = "resnet50"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_dense = 12

print("Device:", CFG.device)



class PawpularDataset:
    def __init__(self, image_paths, dense_features=None, targets=None, augmentations=None):
        self.image_paths = image_paths
        self.dense_features = dense_features if dense_features is not None else np.zeros((len(image_paths), 0), dtype=np.float32)
        self.targets = targets
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((CFG.image_size, CFG.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        image = image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1))

        features = self.dense_features[idx].astype(np.float32)
        if self.targets is None:
            target = np.array(0.0, dtype=np.float32)
        else:
            target = np.array(self.targets[idx], dtype=np.float32)

        return {
            "image": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float),
            "targets": torch.tensor(target, dtype=torch.float),
        }



class PawpularModel(nn.Module):
    def __init__(self, backbone_name="resnet50", n_dense=12, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, in_chans=3)
        feat_dim = self.backbone.num_features

        self.dropout = nn.Dropout(0.5)
        self.out = nn.Sequential(
            nn.Linear(feat_dim + n_dense, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

        self.step_scheduler_after = "epoch"

    def monitor_metrics(self, outputs, targets, loss):
        rmse = torch.sqrt(loss.detach())
        return {"rmse": rmse}

    def optimizer_scheduler(self):
        opt = torch.optim.AdamW(self.parameters(), lr=2.5e-05, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
        return opt, sch

    def forward(self, image, features, targets=None):
        x = self.backbone(image)                   
        x = torch.cat([x, features], dim=1)        
        x = self.dropout(x)
        x = self.out(x)                          

        if targets is not None:
            loss = nn.MSELoss()(x, targets.view(-1,1))
            metrics = self.monitor_metrics(x, targets, loss)
            return x, loss, metrics
        return x, torch.tensor(0.).to(x.device), {}



train_aug = albumentations.Compose([
    albumentations.LongestMaxSize(CFG.image_size, p=1),
    albumentations.PadIfNeeded(CFG.image_size, CFG.image_size, p=1, border_mode=0),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.1),
    albumentations.Rotate(limit=180, p=0.5),
    albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
    albumentations.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
    albumentations.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=0.5),
    albumentations.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], max_pixel_value=255.0, p=1.0),
], p=1.0)

valid_aug = albumentations.Compose([
    albumentations.LongestMaxSize(CFG.image_size, p=1),
    albumentations.PadIfNeeded(CFG.image_size, CFG.image_size, p=1, border_mode=0),
    albumentations.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], max_pixel_value=255.0, p=1.0),
], p=1.0)



df = pd.read_csv("/kaggle/input/same-old-creating-folds/train_5folds.csv")
print("Data rows:", df.shape[0])
display(df.head())

dense_features = [
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
    'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
]

fold = CFG.fold
train_df = df[df.kfold != fold].reset_index(drop=True)
valid_df = df[df.kfold == fold].reset_index(drop=True)

train_img_paths = [f"../input/petfinder-pawpularity-score/train/{x}.jpg" for x in train_df.Id.values]
valid_img_paths = [f"../input/petfinder-pawpularity-score/train/{x}.jpg" for x in valid_df.Id.values]

train_targets = train_df.Pawpularity.values
valid_targets = valid_df.Pawpularity.values

train_features = train_df[dense_features].values
valid_features = valid_df[dense_features].values

train_ds = PawpularDataset(train_img_paths, dense_features=train_features, targets=train_targets, augmentations=train_aug)
valid_ds = PawpularDataset(valid_img_paths, dense_features=valid_features, targets=valid_targets, augmentations=valid_aug)

print("Train size:", len(train_ds), "Valid size:", len(valid_ds))



model = PawpularModel(backbone_name=CFG.model_name, n_dense=CFG.n_dense, pretrained=True)
tez_model = Tez(model)

config = TezConfig(
    training_batch_size=CFG.batch_size,
    validation_batch_size=2 * CFG.batch_size,
    epochs=CFG.epochs,
    step_scheduler_after="epoch",
    step_scheduler_metric="valid_rmse",
    fp16=True,
    val_strategy="epoch",   
)

model_path = f"10158_model_fold{fold}.bin"
es = EarlyStopping(
    monitor="valid_rmse",
    model_path=model_path,
    patience=4,
    mode="min",
    save_weights_only=False
)

tez_model.fit(train_ds, valid_dataset=valid_ds, callbacks=[es], config=config)

print("Training done. Best weights saved to:", model_path)



from torch.utils.data import DataLoader

# Validation dataset
valid_ds = PawpularDataset(valid_img_paths, dense_features=valid_features, targets=None, augmentations=valid_aug)

# Wrap in DataLoader
valid_loader = DataLoader(valid_ds, batch_size=CFG.batch_size, shuffle=False)

# Predict
preds_list = []
for batch in valid_loader:
    images = batch['image'].to(CFG.device, dtype=torch.float)
    features = batch['features'].to(CFG.device, dtype=torch.float)
    
    with torch.no_grad():
        outputs, _, _ = tez_model.model(images, features)
    
    preds_list.extend(outputs.detach().cpu().squeeze().numpy().tolist())

# Convert to numpy array
preds = np.array(preds_list, dtype=np.float32)

# Compute RMSE
rmse = np.sqrt(((preds - valid_targets) ** 2).mean())
print(f"Fold {fold} Validation RMSE: {rmse:.4f}")





