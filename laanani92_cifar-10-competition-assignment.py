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


# CIFAR-10 from scratch (PyTorch) — local macOS (M-series) + create submission from your local test folder
# No pretrained weights. Trains on torchvision CIFAR-10; infers on your local Kaggle test images.
# Paths below are set to your OneDrive locations.

import os, glob, random, pathlib
from contextlib import nullcontext
from typing import List
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# ========= Local paths (from your message) =========
BASE = pathlib.Path("/Users/la_anani/Library/CloudStorage/OneDrive-Personal/Documents/HBKU subjects/Applied Deep Learning/cifar-10")
LOCAL_TRAIN_DIR = BASE / "train"                   # not used (we train on torchvision's CIFAR-10)
LOCAL_TEST_DIR  = BASE / "test"                    # folder with test PNGs
LOCAL_SAMPLE_SUB = BASE / "sampleSubmission.csv"   # sampleSubmission.csv (note the camel case)
OUTPUT_SUBMISSION = BASE / "submission.csv"        # where we'll write the CSV

# ========= Reproducibility =========
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(42)

# ========= Device selection (MPS > CUDA > CPU) =========
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Device:", device)

# Use AMP only on CUDA (MPS autocast can be finicky for training)
use_amp = (device.type == "cuda")
def autocast_ctx():
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# ========= CIFAR-10 (official) via torchvision =========
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

try:
    randaug = transforms.RandAugment(num_ops=2, magnitude=9)
except AttributeError:
    class _Identity:
        def __call__(self, x): return x
    randaug = _Identity()

train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    randaug,
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

test_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

data_root = './data'  # torchvision cache
train_set = datasets.CIFAR10(root=data_root, train=True,  download=True, transform=train_tfms)
test_set  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tfms)
classes = train_set.classes  # ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
print("Classes:", classes)

BATCH_SIZE = 128
NUM_WORKERS = 2 if device.type != "mps" else 0  # mps sometimes happier with 0 workers
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=False)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

# ========= Model (ResNet-9, from scratch) =========
def conv_bn(in_ch, out_ch, ks=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.down  = (stride != 1 or in_ch != out_ch)
        if self.down:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        if self.down: x = self.proj(x)
        return F.relu(x + y, inplace=True)

class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            conv_bn(3, 64),
            conv_bn(64, 128),
            nn.MaxPool2d(2),
        )
        self.block1 = nn.Sequential(ResBlock(128, 128), conv_bn(128, 256, stride=2, padding=1))
        self.block2 = nn.Sequential(ResBlock(256, 256), conv_bn(256, 512, stride=2, padding=1))
        self.block3 = ResBlock(512, 512)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(512, num_classes)
        )
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):  nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu'); 
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.stem(x); x = self.block1(x); x = self.block2(x); x = self.block3(x); x = self.head(x)
        return x

model = ResNet9().to(device)
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

# ========= Optim / Loss / Scheduler =========
EPOCHS = 200
LR = 0.2
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ========= Train / Eval =========
def train_one_epoch():
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            logits = model(imgs)
            loss = criterion(logits, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward(); optimizer.step()
        bs = labels.size(0)
        loss_sum += loss.item() * bs
        total += bs
        correct += (logits.argmax(1) == labels).sum().item()
    scheduler.step()
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast_ctx():
            logits = model(imgs)
            loss = criterion(logits, labels)
        bs = labels.size(0)
        loss_sum += loss.item() * bs
        total += bs
        correct += (logits.argmax(1) == labels).sum().item()
    return loss_sum/total, correct/total

best_acc, best_path = 0.0, "best_cifar10_from_scratch.pth"
for epoch in range(1, EPOCHS+1):
    tr_loss, tr_acc = train_one_epoch()
    te_loss, te_acc = evaluate(test_loader)
    if te_acc > best_acc:
        best_acc = te_acc
        torch.save({"model": model.state_dict(), "acc": best_acc, "epoch": epoch}, best_path)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d}/{EPOCHS} | train_acc={tr_acc:.4f} | test_acc={te_acc:.4f} | best={best_acc:.4f}")
print("Best official CIFAR-10 test acc (offline):", best_acc)

# ========= Build submission from YOUR local test/ + sampleSubmission.csv =========
assert LOCAL_SAMPLE_SUB.exists(), f"sampleSubmission.csv not found at: {LOCAL_SAMPLE_SUB}"
assert LOCAL_TEST_DIR.exists(),    f"test/ folder not found at: {LOCAL_TEST_DIR}"

# Load best weights
ckpt = torch.load(best_path, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

# Read sample submission and infer column names (usually 'id','label')
sample_sub = pd.read_csv(LOCAL_SAMPLE_SUB)
id_col = "id" if "id" in sample_sub.columns else sample_sub.columns[0]
label_col = "label" if "label" in sample_sub.columns else (sample_sub.columns[1] if len(sample_sub.columns) > 1 else "label")

# Collect test files (png/jpg just in case)
test_files: List[str] = sorted(glob.glob(str(LOCAL_TEST_DIR / "*.png"))) + sorted(glob.glob(str(LOCAL_TEST_DIR / "*.jpg")))
print(f"Found {len(test_files)} test images under {LOCAL_TEST_DIR}")

class LocalTestDS(Dataset):
    def __init__(self, files, tfms):
        self.files = files; self.tfms = tfms
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        fp = self.files[i]
        img = Image.open(fp).convert("RGB")
        return self.tfms(img), os.path.basename(fp)

test_ds = LocalTestDS(test_files, test_tfms)
test_loader_local = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

preds, names = [], []
with torch.no_grad():
    for imgs, fnames in test_loader_local:
        imgs = imgs.to(device)
        with autocast_ctx():
            logits = model(imgs)
        y = logits.argmax(1).cpu().numpy().tolist()
        preds.extend([classes[i] for i in y])
        names.extend(fnames)

def file_to_id(name: str):
    stem = os.path.splitext(name)[0]
    return int(stem) if str(stem).isdigit() else stem

ids = [file_to_id(n) if id_col == "id" else n for n in names]
pred_df = pd.DataFrame({id_col: ids, label_col: preds})

# Ensure correct row order by merging into the sample frame
merged = sample_sub[[id_col]].merge(pred_df, on=id_col, how="left")
# Fill any missing with a safe default (rare, e.g., file missing)
if merged[label_col].isna().any():
    merged[label_col] = merged[label_col].fillna(classes[0])

OUTPUT_SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUTPUT_SUBMISSION, index=False)
print(f"\nWrote submission to: {OUTPUT_SUBMISSION}")
print(merged.head())

