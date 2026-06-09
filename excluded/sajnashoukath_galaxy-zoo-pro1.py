# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os, zipfile

base = "/kaggle/input/galaxy-zoo-the-galaxy-challenge"
work = "/kaggle/working"

# Unzip training images
with zipfile.ZipFile(os.path.join(base, "images_training_rev1.zip"), "r") as z:
    z.extractall(work)

# Unzip test images
with zipfile.ZipFile(os.path.join(base, "images_test_rev1.zip"), "r") as z:
    z.extractall(work)

# Unzip labels
with zipfile.ZipFile(os.path.join(base, "training_solutions_rev1.zip"), "r") as z:
    z.extractall(work)

# Unzip zero benchmark (submission template)
with zipfile.ZipFile(os.path.join(base, "all_zeros_benchmark.zip"), "r") as z:
    z.extractall(work)

print(os.listdir(work))


# Full corrected notebook script: Galaxy Zoo - EfficientNet B4 (single model) ----------------

# 0) Install timm if not present (uncomment in Kaggle if needed)
# !pip install -q timm

import os
import cv2
import random
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm  # recommended, stable API for many backbones

from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

# -------------------------------
# Config
# -------------------------------
IMG_SIZE = 256
BATCH_SIZE = 24
EPOCHS = 10
LR = 3e-4
SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "/kaggle/working/images_training_rev1"
TEST_DIR  = "/kaggle/working/images_test_rev1"
LABELS    = "/kaggle/working/training_solutions_rev1.csv"   # Galaxy Zoo ground-truth prob vectors

CHECKPOINT_PATH = "/kaggle/working/efficientnet_b4_seed1337.pth"
PREDS_PATH      = "/kaggle/working/preds_eff_b4.csv"

# -------------------------------
# Utilities
# -------------------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

# -------------------------------
# Dataset
# -------------------------------
class GalaxyDataset(Dataset):
    def __init__(self, galaxy_ids, img_dir, labels=None, augment=False):
        """
        galaxy_ids: array-like of ids (no .jpg extension)
        img_dir: directory containing <id>.jpg files
        labels: None or numpy array of shape (N, 37)
        augment: whether to apply training augmentations
        """
        self.galaxy_ids = list(galaxy_ids)
        self.img_dir = img_dir
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.augment = augment

        self.aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(360),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.1,0.1,0.1,0.1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])

        self.basic = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.galaxy_ids)

    def __getitem__(self, idx):
        gid = self.galaxy_ids[idx]
        img_path = os.path.join(self.img_dir, f"{gid}.jpg")

        # read
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            # fallback: return zero image if missing (avoid crash) but warn
            # (You may prefer to raise instead)
            # print(f"WARN: missing {img_path}")
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.augment:
            img_t = self.aug(img)
        else:
            img_t = self.basic(img)

        if self.labels is not None:
            label = torch.from_numpy(self.labels[idx]).float()  # shape (37,)
            return img_t, label
        return img_t

# -------------------------------
# Load labels and split
# -------------------------------
df = pd.read_csv(LABELS)
all_ids = df["GalaxyID"].values
targets = df.drop("GalaxyID", axis=1).values  # shape (N, 37)

X_train_ids, X_val_ids, y_train, y_val = train_test_split(
    all_ids, targets, test_size=0.1, random_state=SEED, stratify=None
)

train_ds = GalaxyDataset(X_train_ids, TRAIN_DIR, y_train, augment=True)
val_ds   = GalaxyDataset(X_val_ids,   TRAIN_DIR, y_val,   augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# -------------------------------
# Model (timm EfficientNet-B4)
# -------------------------------
class GalaxyEfficientNetB4(nn.Module):
    def __init__(self, pretrained=True, num_outputs=37):
        super().__init__()
        # create timm model; when num_classes set, timm will build classifier with correct out dim
        self.model = timm.create_model("efficientnet_b4", pretrained=pretrained, num_classes=num_outputs)

    def forward(self, x):
        return self.model(x)

model = GalaxyEfficientNetB4(pretrained=True, num_outputs=37).to(DEVICE)

# Sanity check: ensure model has parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model params:", total_params, "Trainable:", trainable_params)
assert total_params > 0, "Model has zero parameters - constructor failed."

# -------------------------------
# Optimizer / Scheduler / Loss / AMP
# -------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
# Galaxy labels are soft probability vectors across groups — MSE is OK; you can try BCEWithLogitsLoss after sigmoid if desired
criterion = nn.MSELoss()
scaler = GradScaler()

# small helper to select correct autocast device
use_cuda = DEVICE == "cuda"

# -------------------------------
# Validation helper
# -------------------------------
def validate(model, loader, device):
    model.eval()
    running_loss = 0.0
    n = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_cuda:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
            else:
                with torch.amp.autocast(device_type="cpu"):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
    return running_loss / n

# -------------------------------
# Training loop
# -------------------------------
for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0.0
    n = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} train")
    for imgs, labels in pbar:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        if use_cuda:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
        else:
            with torch.amp.autocast(device_type="cpu"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        pbar.set_postfix(train_loss=train_loss / n)

    scheduler.step()

    val_loss = validate(model, val_loader, DEVICE)
    print(f"Epoch {epoch} Train loss: {train_loss / n:.4f}  |  Val loss: {val_loss:.4f}")

    # optionally save best by val loss
    torch.save(model.state_dict(), CHECKPOINT_PATH)

# final save
torch.save(model.state_dict(), CHECKPOINT_PATH)
print(f"Saved checkpoint to {CHECKPOINT_PATH}")

# -------------------------------
# TTA prediction on test set
# -------------------------------
def tta_predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="TTA predict"):
            # loader yields images (no labels)
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            imgs = imgs.to(device)
            batch_preds = []

            # base
            if use_cuda:
                with torch.amp.autocast(device_type="cuda"):
                    out = model(imgs)
            else:
                with torch.amp.autocast(device_type="cpu"):
                    out = model(imgs)
            batch_preds.append(out)

            # horizontal flip
            imgs_h = torch.flip(imgs, dims=[3])
            if use_cuda:
                with torch.amp.autocast(device_type="cuda"):
                    out_h = model(imgs_h)
            else:
                with torch.amp.autocast(device_type="cpu"):
                    out_h = model(imgs_h)
            batch_preds.append(out_h)

            # vertical flip
            imgs_v = torch.flip(imgs, dims=[2])
            if use_cuda:
                with torch.amp.autocast(device_type="cuda"):
                    out_v = model(imgs_v)
            else:
                with torch.amp.autocast(device_type="cpu"):
                    out_v = model(imgs_v)
            batch_preds.append(out_v)

            # average predictions (logits/regression outputs)
            batch_mean = torch.stack(batch_preds, dim=0).mean(dim=0)
            preds.append(batch_mean.cpu().numpy())

    return np.vstack(preds)

# Build test dataset & loader
sub_template = pd.read_csv("/kaggle/working/all_zeros_benchmark.csv")  # your provided template
test_ids = sub_template["GalaxyID"].values
test_ds = GalaxyDataset(test_ids, TEST_DIR, labels=None, augment=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# run TTA predict
preds = tta_predict(model, test_loader, DEVICE)

# Postprocess: clamp and normalize per groups as you had
preds = np.clip(preds, 0.0, 1.0)
groups = [
    (0,3),(3,5),(5,8),(8,11),(11,15),
    (15,18),(18,25),(25,28),(28,31),(31,37)
]
for start, end in groups:
    s = preds[:, start:end].sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    preds[:, start:end] /= s

# Save submission
out = sub_template.copy()
out.iloc[:, 1:] = preds  # keep GalaxyID column
out.to_csv(PREDS_PATH, index=False)
out.to_csv("submission_single_effb4.csv", index=False)
print(f"Saved preds to {PREDS_PATH} and submission_single_effb4.csv")

# Clean up
gc.collect()
torch.cuda.empty_cache()





