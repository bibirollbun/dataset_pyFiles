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


# ==============================================================
# CIFAR-10 (Kaggle) — ResNet18 from scratch (no pre-trained)
# Trains 100 epochs, evaluates, and writes /kaggle/working/submission.csv
# Framework: PyTorch
# ==============================================================

import os, re, glob, math, time, csv, random, sys
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T

# ----------------------------
# Repro & device
# ----------------------------
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ----------------------------
# Config
# ----------------------------
EPOCHS = 100
BATCH_SIZE = 128
BASE_LR = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
LABEL_SMOOTH = 0.1
NUM_WORKERS = 2

CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# ----------------------------
# Data (official CIFAR-10 train/test for offline validation)
# ----------------------------
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

train_tfms = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.TrivialAugmentWide(),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

test_tfms = T.Compose([
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

train_set = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tfms)
test_official = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tfms)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(test_official, batch_size=256, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# ----------------------------
# Model (ResNet18 adapted to 32x32, no pretrain)
# ----------------------------
def resnet18_cifar(num_classes=10):
    model = torchvision.models.resnet18(weights=None)  # NO PRETRAIN
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model = resnet18_cifar().to(DEVICE)

# ----------------------------
# Optimizer / LR / Loss (Label smoothing)
# ----------------------------
optimizer = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM,
                            weight_decay=WEIGHT_DECAY, nesterov=True)

def cosine_lr(step, total_steps, base_lr, final_lr=0.0):
    step = min(step, total_steps)
    cos = (1 + math.cos(math.pi * step / total_steps)) / 2
    return final_lr + (base_lr - final_lr) * cos

class SmoothCE(nn.Module):
    def __init__(self, smoothing=0.1): super().__init__(); self.smoothing = smoothing
    def forward(self, logits, targets):
        n = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

criterion = SmoothCE(LABEL_SMOOTH)

# New AMP API
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE=="cuda"))

# ----------------------------
# Train / Evaluate
# ----------------------------
def evaluate(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(DEVICE=="cuda")):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)
    return loss_sum/total, correct/total

best_acc, global_step = 0.0, 0
total_steps = EPOCHS * len(train_loader)

for epoch in range(1, EPOCHS+1):
    model.train()
    run_loss = run_correct = run_total = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        lr = cosine_lr(global_step, total_steps, BASE_LR)
        for pg in optimizer.param_groups: pg['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(DEVICE=="cuda")):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        run_loss += loss.item() * y.size(0)
        run_correct += (logits.argmax(1) == y).sum().item()
        run_total += y.size(0)
        global_step += 1

    train_loss = run_loss / run_total
    train_acc  = run_correct / run_total
    val_loss, val_acc = evaluate(val_loader)

    print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({"model": model.state_dict()}, "best_resnet18_cifar10.pth")

print(f"Best est. Kaggle score (official test acc): {best_acc:.4f}")

# Reload best
ckpt = torch.load("best_resnet18_cifar10.pth", map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval().to(DEVICE)

# ==============================================================
# Kaggle test-set → predictions → /kaggle/working/submission.csv
# ==============================================================

import subprocess

WORK_DIR = Path("/kaggle/working")
INPUT_DIR = Path("/kaggle/input")
TEST_DIR  = Path("./test")  # we'll normalize to have images here

def find_test7z():
    cands = list(INPUT_DIR.rglob("test.7z"))
    return cands[0] if cands else None

# Extract if needed (pure Python, no apt)
if not TEST_DIR.exists() or not any(TEST_DIR.iterdir()):
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    test7z = find_test7z()
    if test7z is None:
        raise FileNotFoundError("Could not find test.7z under /kaggle/input. "
                                "Attach the competition dataset to your notebook.")
    try:
        import py7zr
    except ImportError:
        print("Installing py7zr...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "py7zr"])
        import py7zr

    print(f"Extracting {test7z} -> {TEST_DIR} ...")
    with py7zr.SevenZipFile(str(test7z), mode='r') as z:
        z.extractall(path=str(TEST_DIR))
    print("Done extracting.")

# Some archives create ./test/test; normalize
inner = TEST_DIR / "test"
if inner.exists() and any(inner.iterdir()):
    TEST_DIR = inner

# Build file list
def list_test_files(d: Path):
    files = []
    for pat in ("*.png","*.jpg","*.jpeg"):
        files.extend(sorted(d.glob(pat)))
    return files

files = list_test_files(TEST_DIR)
if len(files) == 0:
    raise RuntimeError(f"No images found under {TEST_DIR}. Check extraction.")
print("Found test images:", len(files), "Sample:", [p.name for p in files[:5]])

# Filename → numeric id
id_regex = re.compile(r"(\d+)")
def num_id(p: Path):
    m = id_regex.search(p.stem)
    if not m:
        raise ValueError(f"No numeric id in filename: {p.name}")
    return int(m.group(1))

# Dataset / loader for Kaggle test
class KaggleTest(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = sorted(paths, key=num_id)
        self.t = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        x = Image.open(p).convert("RGB")
        return (self.t(x) if self.t else x), num_id(p)

kaggle_loader = DataLoader(KaggleTest(files, test_tfms),
                           batch_size=256, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)

# Predict (AMP new API)
rows = []
with torch.no_grad(), torch.amp.autocast("cuda", enabled=(DEVICE=="cuda")):
    for x, ids in kaggle_loader:
        x = x.to(DEVICE)
        logits = model(x)
        preds = logits.argmax(1).cpu().numpy()
        rows.extend([(int(ids[k]), CIFAR10_CLASSES[int(preds[k])]) for k in range(len(ids))])

# Write CSV in working dir
rows.sort(key=lambda t: t[0])
sub_path = WORK_DIR / "submission.csv"
with open(sub_path, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id","label"]); w.writerows(rows)

print(f"Wrote {sub_path} with {len(rows)} rows.")
assert len(rows) == len(files), "Row count mismatch (CSV rows != #images)."

# ----------------------------
# Short printable report snippet
# ----------------------------
print("\n=== REPORT SNIPPET ===")
print("Title: CIFAR-10 Classification — ResNet-18 (from scratch)")
print("Framework: PyTorch (assignment allows Keras/PyTorch/TensorFlow; I used PyTorch).")
print("Training: 100 epochs, SGD(m=0.9, wd=5e-4), cosine LR 0.1→0, label smoothing 0.1, batch 128, TrivialAugment.")
print(f"Validation (official CIFAR-10 test) accuracy: ~{best_acc:.4f} (proxy for Kaggle score).")
print("Submission: Created /kaggle/working/submission.csv with id,label for the 300k Kaggle test images.")



# CIFAR-10 Residual Network from basic layers (no torchvision models)
# - Residual connections implemented explicitly in ResidualBlock.
# - Built from Conv2d/BatchNorm2d/ReLU/AvgPool/Linear — no ResNet18 import.
# - Randomly initialized weights (default PyTorch init).

import os, math, time, argparse
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# -----------------------------
# Model
# -----------------------------

class ResidualBlock(nn.Module):
    """Basic residual block: conv3x3 -> BN -> ReLU -> conv3x3 -> BN, with skip.
       If in/out channels differ or stride>1, uses 1x1 projection for skip.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.proj  = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.proj is not None:
            identity = self.proj(x)
        out += identity
        out = F.relu(out, inplace=True)
        return out


class SimpleResNet(nn.Module):
    """A small ResNet-style network for CIFAR-10 (32x32).
       Stem: 3x3 conv (no maxpool). Stages: [2,2,2] blocks with channels [64,128,256].
       Global average pool -> Linear(256,10).
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 64, stride=1),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),  # downsample to 16x16
            ResidualBlock(128, 128, stride=1),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2), # downsample to 8x8
            ResidualBlock(256, 256, stride=1),
        )
        self.head = nn.Linear(256, num_classes)

        # Weight init (Kaiming for convs, 1 for BN)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        # global average pooling
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

# -----------------------------
# Training / Eval
# -----------------------------

def get_loaders(data_root: str, batch: int = 128, workers: int = 2):
    # CIFAR-10 mean/std
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    train_tfms = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tfms = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tfms)
    testset  = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tfms)
    train_loader = DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(testset,  batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader

def accuracy(logits, targets):
    preds = logits.argmax(1)
    return (preds == targets).float().mean().item()

def train_one_epoch(model, loader, opt, scaler, loss_fn, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            logits = model(images)
            loss = loss_fn(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, targets) * bs
        n += bs
    return total_loss / n, total_acc / n

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = loss_fn(logits, targets)
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy(logits, targets) * bs
        n += bs
    return total_loss / n, total_acc / n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    
    # use parse_known_args instead of parse_args
    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_loaders(args.data_root, args.batch, args.workers)

    model = SimpleResNet(num_classes=10).to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.2f}M")

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, scaler, loss_fn, device)
        te_loss, te_acc = evaluate(model, test_loader, loss_fn, device)
        scheduler.step()
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), "simple_resnet_cifar10_best.pth")
        print(f"[Epoch {epoch:03d}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={te_loss:.4f} val_acc={te_acc:.4f} | best_acc={best_acc:.4f}")
    print("Training done. Best val_acc:", best_acc)

if __name__ == "__main__":
    main()


