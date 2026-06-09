# A) imports, seed, device, speed flags
import os, math, random, glob, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# speed tweaks
import torch.backends.cudnn as cudnn
cudnn.benchmark = True  # lets cudnn pick fastest conv algo for 32x32

print("Device:", DEVICE)


# B) extract /kaggle/input/cifar-10/*.7z → /kaggle/working/{train,test}
!apt -yq install libarchive-dev >/dev/null
!pip -q install libarchive >/dev/null
import libarchive.public

base_in  = "/kaggle/input/cifar-10"
train_out = "/kaggle/working/train"
test_out  = "/kaggle/working/test"

os.makedirs(train_out, exist_ok=True)
os.makedirs(test_out, exist_ok=True)

# extract train once
if len(glob.glob(train_out+"/*.png")) < 50000:
    for _ in libarchive.public.file_pour(f"{base_in}/train.7z"): pass

# extract test once (300,000 files; allow ~2–3 min)
if len(glob.glob(test_out+"/*.png")) < 300000:
    for _ in libarchive.public.file_pour(f"{base_in}/test.7z"): pass

labels_df = pd.read_csv(f"{base_in}/trainLabels.csv")  # columns: id, label
classes = sorted(labels_df.label.unique().tolist())    # list of class names
cls_to_idx = {c:i for i,c in enumerate(classes)}

# stratified split 45k/5k
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=5000, random_state=SEED)
idx_train, idx_val = next(sss.split(labels_df["id"], labels_df["label"]))
labels_tr = labels_df.iloc[idx_train].reset_index(drop=True)
labels_va = labels_df.iloc[idx_val].reset_index(drop=True)

# transforms (CutMix happens in the loop; these are base TFMs)
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

tfm_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2,0.2,0.2,0.1),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
    T.RandomErasing(p=0.25, scale=(0.02,0.2), ratio=(0.3,3.3), value=0)
])

tfm_test = T.Compose([
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

class CifarFiles(Dataset):
    def __init__(self, df, root, transform):
        self.df = df
        self.root = Path(root)
        self.tfm = transform
        self.has_label = "label" in df.columns

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / f"{row['id']}.png").convert("RGB")
        x = self.tfm(img)
        if self.has_label:
            y = cls_to_idx[row["label"]]
            return x, torch.tensor(y, dtype=torch.long)
        else:
            return x, int(row["id"])

train_ds = CifarFiles(labels_tr, train_out, tfm_train)
val_ds   = CifarFiles(labels_va, train_out, tfm_test)

BS = 128
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BS, shuffle=False, num_workers=2, pin_memory=True)

print("Train/Val sizes:", len(train_ds), len(val_ds), "Classes:", classes)


# C) model: scratch ResNet-ish for CIFAR-10 (random init)
def conv3(in_c, out_c, s=1): return nn.Conv2d(in_c, out_c, 3, stride=s, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = conv3(in_c, out_c, stride)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = conv3(out_c, out_c, 1)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.down  = None
        if stride != 1 or in_c != out_c:
            self.down = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = self.act(out + identity)
        return out

class SmallResNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(conv3(3,64), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        # 32x32
        self.s1 = nn.Sequential(BasicBlock(64,64,1), BasicBlock(64,64,1))
        # 16x16
        self.s2 = nn.Sequential(BasicBlock(64,128,2), BasicBlock(128,128,1))
        # 8x8
        self.s3 = nn.Sequential(BasicBlock(128,256,2), BasicBlock(256,256,1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(256, num_classes)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.s1(x); x = self.s2(x); x = self.s3(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

model = SmallResNetCIFAR(num_classes=10).to(DEVICE)
sum(p.numel() for p in model.parameters()), model.__class__.__name__


# D) AdamW + warmup→cosine (Challenge requirement)
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

EPOCHS        = 40
BASE_LR       = 3e-3
MIN_LR_FACTOR = 0.01
WARMUP_EPOCHS = 5
WEIGHT_DECAY  = 5e-4

optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, betas=(0.9,0.999), weight_decay=WEIGHT_DECAY)

warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
cosine = CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS-WARMUP_EPOCHS), eta_min=BASE_LR*MIN_LR_FACTOR)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

print("Using AdamW + Linear warmup → CosineAnnealingLR")


# E) Train with CutMix + AMP (torch.amp) and validate — CORRECTED

import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast  # NEW API

criterion = nn.CrossEntropyLoss()

# ----- AMP setup (works on GPU/CPU) -----
AMP_ENABLED = (DEVICE.type == "cuda")
scaler = GradScaler("cuda" if AMP_ENABLED else "cpu")

# ----- CutMix helpers -----
USE_CUTMIX   = True
CUTMIX_ALPHA = 1.0
CUTMIX_P     = 0.5

def rand_bbox(W, H, lam):
    cut_w = int(W * math.sqrt(1 - lam))
    cut_h = int(H * math.sqrt(1 - lam))
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    x1 = max(cx - cut_w // 2, 0); y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W); y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2

def apply_cutmix(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    B, _, H, W = x.size()
    index = torch.randperm(B, device=x.device)
    x_shuf, y_shuf = x[index], y[index]
    x1, y1, x2, y2 = rand_bbox(W, H, lam)
    x[:, :, y1:y2, x1:x2] = x_shuf[:, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, y, y_shuf, lam

@torch.no_grad()
def evaluate(loader, use_model=None):
    m = model if use_model is None else use_model
    m.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with autocast("cuda", dtype=torch.float16, enabled=AMP_ENABLED):  # NEW
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            logits = m(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total

best_val = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    for x, y in train_loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        use_cm = USE_CUTMIX and (random.random() < CUTMIX_P)
        if use_cm:
            x, y_a, y_b, lam = apply_cutmix(x, y, alpha=CUTMIX_ALPHA)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=torch.float16, enabled=AMP_ENABLED):  # NEW
            logits = model(x)
            if use_cm:
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    scheduler.step()

    vloss, vacc = evaluate(val_loader)
    print(f"Epoch {epoch:03d} | val_acc={vacc:.4f} | val_loss={vloss:.4f} | lr={scheduler.get_last_lr()[0]:.5f}")

    if vacc > best_val:
        best_val = vacc
        torch.save(model.state_dict(), "best_model.pth")

print(f"Best val acc: {best_val:.4f}")


# F) Offline evaluation on the official CIFAR-10 test set (requires Internet=ON)
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.amp import autocast
import torch
import torch.nn as nn

# in case it's not defined above:
AMP_ENABLED = (DEVICE.type == "cuda")

# dataset + loader
test_official = CIFAR10(
    root="/kaggle/working/data",
    train=False,
    download=True,
    transform=tfm_test,      # uses the same normalization as train/val
)
test_loader_official = DataLoader(
    test_official, batch_size=BS, shuffle=False, num_workers=2, pin_memory=True
)

# load best checkpoint and eval
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

criterion = nn.CrossEntropyLoss()
loss_sum, correct, total = 0.0, 0, 0

with torch.no_grad():
    for x, y in test_loader_official:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        # NEW AMP API
        with autocast("cuda", dtype=torch.float16, enabled=AMP_ENABLED):
            logits = model(x)
            loss = criterion(logits, y)

        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)

ta = correct / total
tl = loss_sum / total
print(f"Official CIFAR-10 test accuracy (≈ Kaggle score): {ta:.4f}")


# G) Predict Kaggle test set and create submission.csv
from torch.utils.data import DataLoader
from torch.amp import autocast
from pathlib import Path
import numpy as np
import pandas as pd
import glob
import torch

# in case it's not defined above:
AMP_ENABLED = (DEVICE.type == "cuda")

# gather test ids in numeric order
test_paths = sorted(glob.glob(test_out + "/*.png"), key=lambda p: int(Path(p).stem))
test_ids = pd.DataFrame({"id": [int(Path(p).stem) for p in test_paths]})

# dataset/loader (uses the same CifarFiles and tfm_test defined earlier)
test_ds = CifarFiles(test_ids, test_out, tfm_test)
test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=2, pin_memory=True)

# load best model and eval
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

preds = []
with torch.no_grad():
    for x, idx in test_loader:
        x = x.to(DEVICE, non_blocking=True)
        # NEW AMP API
        with autocast("cuda", dtype=torch.float16, enabled=AMP_ENABLED):
            logits = model(x)
        preds.append(logits.argmax(1).cpu().numpy())

preds = np.concatenate(preds)

# map indices -> class names and save
submission = pd.DataFrame({
    "id": test_ids["id"].values,
    "label": [classes[i] for i in preds]
})
submission.to_csv("submission.csv", index=False)
print(submission.head(), "\nSaved: submission.csv")


history = []  # put this before the training loop

# inside your epoch loop, after you compute vacc/vloss:
history.append((epoch, vacc, vloss, scheduler.get_last_lr()[0]))

# after training:
print("epoch | val_acc | val_loss | lr")
for e, a, l, lr in history:
    print(f"{e:03d}   | {a:.4f}  | {l:.4f}  | {lr:.5f}")


