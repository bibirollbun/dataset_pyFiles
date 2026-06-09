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


# ============================================================
# STRONG EfficientNet-B4 model for Galaxy Zoo
# ============================================================

import os, cv2, random, gc
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

# -------------------- CONFIG --------------------

IMG_SIZE = 256
BATCH_SIZE = 16   # smaller for B4
EPOCHS = 25
LR = 2e-4
SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "/kaggle/working/images_training_rev1"
TEST_DIR  = "/kaggle/working/images_test_rev1"
LABELS    = "/kaggle/working/training_solutions_rev1.csv"

BEST_CKPT = "/kaggle/working/effb4_strong_best.pth"
SUB_PATH  = "/kaggle/working/submission_effb4_strong.csv"

# -------------------- SEEDING --------------------

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

# -------------------- DATASET & TRANSFORMS --------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(360),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

class GalaxyDataset(Dataset):
    def _init_(self, galaxy_ids, img_dir, labels=None, transform=None):
        self.galaxy_ids = galaxy_ids
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform

    def _len_(self):
        return len(self.galaxy_ids)

    def _getitem_(self, idx):
        gid = self.galaxy_ids[idx]
        img_path = os.path.join(self.img_dir, f"{gid}.jpg")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label

        return img

# -------------------- LOAD DATA --------------------

df = pd.read_csv(LABELS)
all_ids = df["GalaxyID"].values
targets = df.drop("GalaxyID", axis=1).values

X_train_ids, X_val_ids, y_train, y_val = train_test_split(
    all_ids, targets, test_size=0.1, random_state=1337
)

train_ds = GalaxyDataset(X_train_ids, TRAIN_DIR, y_train, transform=train_transform)
val_ds   = GalaxyDataset(X_val_ids,   TRAIN_DIR, y_val,   transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# -------------------- MODEL --------------------

class StrongEffB4(nn.Module):
    def _init_(self):
        super()._init_()
        self.backbone = models.efficientnet_b4(pretrained=True)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 37)
        )

    def forward(self, x):
        return self.backbone(x)

model = StrongEffB4().to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.MSELoss()
scaler = GradScaler()

best_val_loss = float("inf")

# -------------------- TRAINING --------------------

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0

    for imgs, labels in tqdm(train_loader, desc=f"[EffB4-strong] Epoch {epoch+1}/{EPOCHS} - train"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss_sum += loss.item()

    scheduler.step()
    train_loss = train_loss_sum / len(train_loader)

    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"[EffB4-strong] Epoch {epoch+1}/{EPOCHS} - val"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            val_loss_sum += loss.item()

    val_loss = val_loss_sum / len(val_loader)
    print(f"[EffB4-strong] Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), BEST_CKPT)
        print(f"  ðŸ’¾ New best model saved with val_loss={best_val_loss:.5f}")

print(f"\nLoading best checkpoint from {BEST_CKPT}")
model.load_state_dict(torch.load(BEST_CKPT, map_location=DEVICE))

# -------------------- TTA PREDICTION --------------------

def tta_predict(model, loader):
    model.eval()
    preds = []

    with torch.no_grad():
        for imgs in tqdm(loader, desc="[EffB4-strong] TTA predicting"):
            imgs = imgs.to(DEVICE)

            outs = []

            with autocast():
                o = model(imgs)
            outs.append(o)

            imgs_h = torch.flip(imgs, dims=[3])
            with autocast():
                o_h = model(imgs_h)
            outs.append(o_h)

            imgs_v = torch.flip(imgs, dims=[2])
            with autocast():
                o_v = model(imgs_v)
            outs.append(o_v)

            mean_out = torch.stack(outs, dim=0).mean(dim=0)
            preds.append(mean_out.cpu().numpy())

    return np.vstack(preds)

sub_template = pd.read_csv("/kaggle/working/all_zeros_benchmark.csv")
test_ids = sub_template["GalaxyID"].values

test_ds = GalaxyDataset(test_ids, TEST_DIR, labels=None, transform=val_transform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

preds = tta_predict(model, test_loader)

# -------------------- GROUP NORMALIZATION --------------------

preds = np.clip(preds, 0.0, 1.0)
groups = [
    (0,3),(3,5),(5,8),(8,11),(11,15),
    (15,18),(18,25),(25,28),(28,31),(31,37)
]

for s, e in groups:
    g = preds[:, s:e].sum(axis=1, keepdims=True)
    g[g == 0] = 1.0
    preds[:, s:e] /= g

sub = sub_template.copy()
sub.iloc[:, 1:] = preds
sub.to_csv(SUB_PATH, index=False)
sub.to_csv("submission_effb4_strong.csv", index=False)

print("\nâœ… Wrote strong EffB4 submission to submission_effb4_strong.csv")







