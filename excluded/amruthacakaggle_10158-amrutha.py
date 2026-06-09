import sys
sys.path.append("../input/tez-lib/")  # tez library

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
    n_folds = 5
    model_name = "resnet50"
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
            image = self.augmentations(image=image)["image"]

        image = image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

        features = self.dense_features[idx].astype(np.float32)
        target = np.array(0.0, dtype=np.float32) if self.targets is None else np.array(self.targets[idx], dtype=np.float32)

        return {
            "image": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float),
            "targets": torch.tensor(target, dtype=torch.float)
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

    def forward(self, image, features, targets=None):
        x = self.backbone(image)
        x = torch.cat([x, features], dim=1)
        x = self.dropout(x)
        x = self.out(x)

        if targets is not None:
            loss = nn.MSELoss()(x, targets.view(-1, 1))
            metrics = self.monitor_metrics(x, targets, loss)
            return x, loss, metrics
        return x, torch.tensor(0.), {}

    def monitor_metrics(self, outputs, targets, loss):
        rmse = torch.sqrt(loss.detach())
        return {"rmse": rmse}

    def optimizer_scheduler(self):
        opt = torch.optim.AdamW(self.parameters(), lr=2.5e-05, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
        return opt, sch



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



# Load train folds CSV
df = pd.read_csv("/kaggle/input/same-old-creating-folds/train_5folds.csv")
print("Train folds shape:", df.shape)
display(df.head())

# Load test CSV
test_df = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/test.csv")
print("Test shape:", test_df.shape)
display(test_df.head())



from torch.utils.data import DataLoader

dense_features = [
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
    'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
]

oof_preds = np.zeros(len(df), dtype=np.float32)
test_preds = np.zeros(len(test_df), dtype=np.float32)

for fold in range(CFG.n_folds):
    print(f"\n=== Fold {fold} ===")
    seed_everything(42 + fold)

    # Split train/valid
    train_df_fold = df[df.kfold != fold].reset_index(drop=True)
    valid_df_fold = df[df.kfold == fold].reset_index(drop=True)

    train_img_paths = [f"../input/petfinder-pawpularity-score/train/{x}.jpg" for x in train_df_fold.Id.values]
    valid_img_paths = [f"../input/petfinder-pawpularity-score/train/{x}.jpg" for x in valid_df_fold.Id.values]

    train_targets = train_df_fold.Pawpularity.values
    valid_targets = valid_df_fold.Pawpularity.values

    train_features = train_df_fold[dense_features].values
    valid_features = valid_df_fold[dense_features].values

    # Datasets
    train_ds = PawpularDataset(train_img_paths, dense_features=train_features, targets=train_targets, augmentations=train_aug)
    valid_ds = PawpularDataset(valid_img_paths, dense_features=valid_features, targets=valid_targets, augmentations=valid_aug)

    # Model
    model = PawpularModel(backbone_name=CFG.model_name, n_dense=len(dense_features), pretrained=True)
    tez_model = Tez(model)

    config = TezConfig(
        training_batch_size=CFG.batch_size,
        validation_batch_size=2*CFG.batch_size,
        epochs=CFG.epochs,
        step_scheduler_after="epoch",
        step_scheduler_metric="valid_rmse",
        fp16=True,
        val_strategy="epoch"
    )

    model_path = f"model_f{fold}.bin"
    es = EarlyStopping(
        monitor="valid_rmse",
        model_path=model_path,
        patience=4,
        mode="min",
        save_weights_only=False   # important! save full model
    )

    # Fit
    tez_model.fit(train_ds, valid_dataset=valid_ds, callbacks=[es], config=config)

    # Load best weights for inference
    if os.path.exists(model_path):
        print(f"Loading model weights: {model_path}")
        tez_model.load(model_path)
    else:
        print(f"Warning: model file not found: {model_path} (skipping)")

    # -------------------
    # OOF predictions
    valid_ds_pred = PawpularDataset(valid_img_paths, dense_features=valid_features, targets=None, augmentations=valid_aug)
    valid_loader = DataLoader(valid_ds_pred, batch_size=CFG.batch_size, shuffle=False)

    preds_list = []
    tez_model.model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            images = batch["image"].to(CFG.device, dtype=torch.float)
            features = batch["features"].to(CFG.device, dtype=torch.float)
            outputs, _, _ = tez_model.model(images, features)
            preds_list.extend(outputs.detach().cpu().squeeze().numpy().tolist())

    preds_valid = np.array(preds_list, dtype=np.float32)
    oof_preds[df[df.kfold == fold].index.values] = preds_valid

    # -------------------
    # Test predictions
    test_img_paths = [f"../input/petfinder-pawpularity-score/test/{x}.jpg" for x in test_df.Id.values]
    test_features = np.zeros((len(test_df), len(dense_features)), dtype=np.float32)
    for i, col in enumerate(dense_features):
        if col in test_df.columns:
            test_features[:, i] = test_df[col].values

    test_ds_pred = PawpularDataset(test_img_paths, dense_features=test_features, targets=None, augmentations=valid_aug)
    test_loader = DataLoader(test_ds_pred, batch_size=CFG.batch_size, shuffle=False)

    preds_test_list = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(CFG.device, dtype=torch.float)
            features = batch["features"].to(CFG.device, dtype=torch.float)
            outputs, _, _ = tez_model.model(images, features)
            preds_test_list.extend(outputs.detach().cpu().squeeze().numpy().tolist())

    preds_test_fold = np.array(preds_test_list, dtype=np.float32)
    test_preds += preds_test_fold / CFG.n_folds

    # Fold RMSE
    fold_rmse = np.sqrt(((preds_valid - valid_targets) ** 2).mean())
    print(f"Fold {fold} RMSE: {fold_rmse:.4f}")




oof_rmse = np.sqrt(((oof_preds - df.Pawpularity.values) ** 2).mean())
print(f"OOF RMSE (all folds): {oof_rmse:.4f}")

# Prepare submission
submission = pd.DataFrame({
    "Id": test_df.Id.values,
    "Pawpularity": test_preds
})

# Clip predictions to [0, 100] (valid range)
submission["Pawpularity"] = submission["Pawpularity"].clip(0, 100)

# Save CSV
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
display(submission.head())





