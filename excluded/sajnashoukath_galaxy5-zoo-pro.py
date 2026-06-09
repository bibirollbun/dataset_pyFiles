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


# Full corrected script for EffB4 with 9-view TTA (fixes empty-param error)
# Requires timm: uncomment and run `!pip install -q timm` on Kaggle if missing

import os, cv2, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

# ---------------- CONFIG ----------------
IMG_SIZE   = 256
BATCH_SIZE = 16
EPOCHS     = 70
LR         = 2e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "/kaggle/working/images_training_rev1"
TEST_DIR  = "/kaggle/working/images_test_rev1"
LABELS    = "/kaggle/working/training_solutions_rev1.csv"
BENCH     = "/kaggle/working/all_zeros_benchmark.csv"
BEST_CKPT = "/kaggle/working/effb4_T10_best.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------- SEEDING ----------------
def seed_everything(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(1337)

# ---------------- TRANSFORMS ----------------
train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(360),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.5)
])

val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------------- DATASET (fixed dunders) ----------------
class GalaxyDataset(Dataset):
    def __init__(self, galaxy_ids, img_dir, labels=None, transform=None):
        self.galaxy_ids = list(galaxy_ids)
        self.img_dir = img_dir
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.galaxy_ids)

    def __getitem__(self, idx):
        gid = self.galaxy_ids[idx]
        img_path = os.path.join(self.img_dir, f"{gid}.jpg")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            # fallback zero image
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            label = torch.from_numpy(self.labels[idx]).float()
            return img, label
        return img

# ---------------- LOAD DATA ----------------
df = pd.read_csv(LABELS)
all_ids = df["GalaxyID"].values
targets = df.drop("GalaxyID", axis=1).values  # shape (N,37)

X_train_ids, X_val_ids, y_train, y_val = train_test_split(
    all_ids, targets, test_size=0.1, random_state=42
)

train_ds = GalaxyDataset(X_train_ids, TRAIN_DIR, y_train, transform=train_tf)
val_ds   = GalaxyDataset(X_val_ids,   TRAIN_DIR, y_val,   transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ---------------- MODEL (timm) ----------------
class StrongEffB4(nn.Module):
    def __init__(self, pretrained=True, num_classes=37):
        super().__init__()
        # timm reliably lets you set num_classes
        self.model = timm.create_model("efficientnet_b4", pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

model = StrongEffB4(pretrained=True, num_classes=37).to(DEVICE)

# ------------- SANITY CHECK BEFORE OPTIMIZER -------------
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model params:", total_params, "Trainable:", trainable_params)
assert total_params > 0, "Model has zero parameters â€” constructor didn't run correctly. Re-run class cell."

# ---------------- OPTIM / SCHED / LOSS / AMP ----------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.MSELoss()
scaler = GradScaler()

# ---------------- TRAINING LOOP ----------------
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0
    n_train = 0

    for imgs, labels in tqdm(train_loader, desc=f"[EffB4-T10] Epoch {epoch+1}/{EPOCHS} - train"):
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        # autocast with explicit device type
        if DEVICE == "cuda":
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

        train_loss_sum += loss.item() * imgs.size(0)
        n_train += imgs.size(0)

    scheduler.step()
    train_loss = train_loss_sum / max(1, n_train)

    # validation
    model.eval()
    val_loss_sum = 0.0
    n_val = 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"[EffB4-T10] Epoch {epoch+1}/{EPOCHS} - val"):
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            if DEVICE == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
            else:
                with torch.amp.autocast(device_type="cpu"):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * imgs.size(0)
            n_val += imgs.size(0)

    val_loss = val_loss_sum / max(1, n_val)
    print(f"[EffB4-T10] Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), BEST_CKPT)
        print(f"  ðŸ’¾ New best model saved with val_loss={best_val_loss:.6f}")

# ---------------- LOAD BEST ----------------
print(f"\nLoading best checkpoint from {BEST_CKPT}")
model.load_state_dict(torch.load(BEST_CKPT, map_location=DEVICE))
model.eval()

# ---------------- 9-VIEW TTA PREDICTION ----------------
def tta_9view(model, imgs):
    # imgs: (B,C,H,W) on device
    aug_imgs = [
        imgs,
        torch.flip(imgs, [3]),                    # horiz
        torch.flip(imgs, [2]),                    # vert
        torch.rot90(imgs, 1, [2,3]),
        torch.rot90(imgs, 2, [2,3]),
        torch.rot90(imgs, 3, [2,3]),
        torch.flip(torch.rot90(imgs, 1, [2,3]), [3]),
        torch.flip(torch.rot90(imgs, 2, [2,3]), [3]),
        torch.flip(torch.rot90(imgs, 3, [2,3]), [3]),
    ]
    outs = []
    for a in aug_imgs:
        # use autocast for inference as well
        if DEVICE == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                outs.append(model(a))
        else:
            with torch.amp.autocast(device_type="cpu"):
                outs.append(model(a))
    return torch.stack(outs).mean(0)

sub_template = pd.read_csv(BENCH)
test_ids = sub_template["GalaxyID"].values
test_ds = GalaxyDataset(test_ids, TEST_DIR, labels=None, transform=val_tf)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

all_preds = []
with torch.no_grad():
    for imgs in tqdm(test_loader, desc="[EffB4-T10] TTA predicting"):
        # DataLoader returns images only because labels=None
        if isinstance(imgs, (list, tuple)):
            imgs = imgs[0]
        imgs = imgs.to(DEVICE, non_blocking=True)
        preds = tta_9view(model, imgs)
        all_preds.append(preds.cpu().numpy())

preds = np.vstack(all_preds)

# ---------------- GROUP NORMALIZATION ----------------
preds = np.clip(preds, 0.0, 1.0)
groups = [
    (0,3),(3,5),(5,8),(8,11),(11,15),
    (15,18),(18,25),(25,28),(28,31),(31,37)
]
for s,e in groups:
    g = preds[:, s:e].sum(axis=1, keepdims=True)
    g[g == 0] = 1.0
    preds[:, s:e] /= g

sub = sub_template.copy()
sub.iloc[:, 1:] = preds
sub.to_csv("submission_effb4_T10.csv", index=False)
print("\nâœ… Saved submission_effb4_T10.csv")



import os
BEST_CKPT = "/kaggle/working/effb4_T10_best.pth"

# 1) Does the file exist?
print("exists:", os.path.exists(BEST_CKPT))

# 2) File size (helps check it's not empty)
if os.path.exists(BEST_CKPT):
    print("size (MB):", os.path.getsize(BEST_CKPT) / 1024**2)

# 3) List recent files in working dir
print("working dir contents (sample):", sorted(os.listdir("/kaggle/working"))[-20:])



# show current working directory and disk usage
import os, sys, subprocess
print("pwd:", os.getcwd())
print("cwd contents:", sorted(os.listdir("."))[:200])

# show /kaggle/working contents (where we save)
wk = "/kaggle/working"
print("/kaggle/working exists:", os.path.exists(wk))
if os.path.exists(wk):
    print("working dir files (full):", sorted(os.listdir(wk))[:200])

# show free disk space (MB)
stat = os.statvfs(wk if os.path.exists(wk) else ".")
free_mb = stat.f_bavail * stat.f_frsize / 1024**2
print("free space (MB):", round(free_mb,2))





