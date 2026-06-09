# Basics
import os, random, glob, csv
from pathlib import Path
import numpy as np
from PIL import Image

# Torch / TorchVision
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models

# Reproducibility
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  =

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# CIFAR-10 normalization
C10_MEAN = (0.4914, 0.4822, 0.4465)
C10_STD  = (0.2470, 0.2435, 0.2616)

# Class order (use for submission mapping)
IDX_TO_CLASS = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
print("Class order:", IDX_TO_CLASS)



# Step 1: Train/Val data, this is 45k/5k split

# Transforms
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),     # mild color jitter
    transforms.ToTensor(),
    transforms.Normalize(C10_MEAN, C10_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])
eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(C10_MEAN, C10_STD),
])

# Load official CIFAR-10 training set (50k)
root = "./data"
full_train = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)

# Split: 45k train / 5k val 
val_size = 5000
train_size = len(full_train) - val_size
g = torch.Generator().manual_seed(seed)
train_set, val_set = random_split(full_train, [train_size, val_size], generator=g)
val_set.dataset.transform = eval_tf   # no augs on val

# Dataloaders
BATCH_SIZE = 256    
NUM_WORKERS = 2
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

# Quick sanity check
xb, yb = next(iter(train_loader))
print("Train batch:", xb.shape, yb.shape)
print("Train/Val sizes:", len(train_set), len(val_set))



# Step 2: Model (random init) with CIFAR stem
# ResNet-18 RANDOM init, adapted stem for 32x32 3x3 stride1, no maxpool
model = models.resnet18(weights=None, num_classes=10) ## no pre-trained weights were used
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()

# Explicit Kaiming init =
def init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None: nn.init.zeros_(m.bias)
model.apply(init_kaiming)

model = model.to(device)
print("Model ready.")



# Step 3: Training config (200 epochs, cosine, warmup, EMA)

EPOCHS = 200
base_lr = 0.2 * (BATCH_SIZE / 256)   # LR scales with batch size
optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9,
                            weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Simple EMA of weights (optional but helpful)
use_ema = True
ema_decay = 0.999
if use_ema:
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

@torch.no_grad()
def ema_update():
    if not use_ema: return
    for k, v in model.state_dict().items():
        ema[k].mul_(ema_decay).add_(v.detach(), alpha=1-ema_decay)

best_acc, best_path = 0.0, "best_resnet18_cifar10.pth"



# Step 4: Train loop

# ===== Cell 5: Train loop with SAFE EMA =====
import torch

# --- EMA settings ---
use_ema = True
ema_decay = 0.999

# Initialize EMA snapshot of current weights
if use_ema:
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

@torch.no_grad()
def ema_update():
    """Update EMA weights; only average float tensors, copy others (e.g., Long buffers)."""
    if not use_ema:
        return
    for k, v in model.state_dict().items():
        if v.dtype.is_floating_point:
            ema[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
        else:
            ema[k] = v.detach().clone()

def run_epoch(loader, train=True):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
                ema_update()  # keep EMA synced after each optimizer step
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total

# --- Training loop with 5-epoch warmup + cosine scheduler ---
best_acc, best_path = 0.0, "best_resnet18_cifar10.pth"
warmup_epochs = 5

for epoch in range(EPOCHS):
    # linear warmup
    if epoch < warmup_epochs:
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr * (epoch + 1) / warmup_epochs

    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader,   train=False)
    scheduler.step()

    # save best (main weights)
    if va_acc > best_acc:
        best_acc = va_acc
        torch.save({"model": model.state_dict()}, best_path)

    print(f"Epoch {epoch+1:03d}/{EPOCHS} | "
          f"train {tr_acc:.4f}/{tr_loss:.4f} | "
          f"val {va_acc:.4f}/{va_loss:.4f} | best {best_acc:.4f}")

print("Best val acc:", best_acc)



# Offline accuracy on  CIFAR-10 test which is not mandatory
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

C10_MEAN=(0.4914,0.4822,0.4465); C10_STD=(0.2470,0.2435,0.2616)
test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(C10_MEAN, C10_STD)])
test_official = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)
test_loader = DataLoader(test_official, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

# Prefer EMA weights; otherwise use best checkpoint
if 'ema' in globals():
    model.load_state_dict(ema, strict=False)
else:
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)

model.eval()
total = correct = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)

offline_acc = correct / total
print("Official CIFAR-10 test accuracy (offline estimate):", offline_acc)



# Robust extract + verify + flatten, then recount
!pip -q install py7zr
import os, glob, shutil, py7zr
from pathlib import Path

# 1) Locate the archive
cands = glob.glob("/kaggle/input/**/test.7z", recursive=True)
print("Candidates:", cands)
assert cands, "Couldn't find test.7z. Make sure the CIFAR-10 competition is attached as an input."
test7z = cands[0]
print("Using:", test7z, "| size (bytes):", os.path.getsize(test7z))

# 2) Clean output dir
outdir = "test"
if os.path.exists(outdir):
    shutil.rmtree(outdir)
os.makedirs(outdir, exist_ok=True)

# 3) Peek inside the archive, then extract
with py7zr.SevenZipFile(test7z, mode="r") as z:
    names = z.getnames()
    print("First 10 archive entries:", names[:10])
    z.extractall(path=outdir)

# 4) Count recursively (handles nested folders inside the archive)
pngs_recursive = glob.glob(os.path.join(outdir, "**", "*.png"), recursive=True)
print("PNG files found (recursive):", len(pngs_recursive))

# 5) If files are nested (e.g., test/test/1.png), flatten into ./test
moved = 0
for p in pngs_recursive:
    dest = os.path.join(outdir, os.path.basename(p))
    if os.path.abspath(p) != os.path.abspath(dest):
        if not os.path.exists(dest):
            shutil.move(p, dest)
            moved += 1
print("Moved from nested dirs:", moved)

# 6) Remove empty subfolders under ./test (optional)
for root, dirs, files in os.walk(outdir, topdown=False):
    if root == outdir: 
        continue
    if not os.listdir(root):
        os.rmdir(root)

# 7) Final count should be ~300000
final_count = len(glob.glob(os.path.join(outdir, "*.png")))
print("Final PNG count in ./test:", final_count)



# Run inference on 300k test images and make submssion file
import csv, glob
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Load EMA (if present) or best checkpoint; set to eval
if 'ema' in globals():
    model.load_state_dict(ema, strict=False)
else:
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)
model.eval()

# Same normalization as training/eval
test_tf_final = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(C10_MEAN, C10_STD),
])

# Sorted by numeric filename so ids align (1.png, 2.png, ...)
test_files = sorted(glob.glob('test/*.png'), key=lambda p: int(Path(p).stem))

class TestFolder(Dataset):
    def __init__(self, files, tfm):
        self.files = files; self.tfm = tfm
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        fp = self.files[i]
        img = Image.open(fp).convert('RGB')
        return self.tfm(img), int(Path(fp).stem)

test_loader_big = DataLoader(TestFolder(test_files, test_tf_final),
                             batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

IDX_TO_CLASS = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

rows = []
with torch.no_grad():
    for x, ids in test_loader_big:
        x = x.to(device)
        pred = model(x).argmax(1).cpu().tolist()
        for i, p in zip(ids.tolist(), pred):
            rows.append((i, IDX_TO_CLASS[p]))

# Write CSV with exact header id,label
rows.sort(key=lambda t: t[0])
with open('submission.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id','label'])
    w.writerows(rows)

print("Wrote submission.csv with", len(rows), "rows.")



import pandas as pd

df = pd.read_csv("submission.csv")
print("shape:", df.shape)                        # (300000, 2)
print("columns:", list(df.columns))              # ['id','label']
print("id range:", df['id'].min(), df['id'].max())
print("unique labels:", sorted(df['label'].unique()))
assert set(df['label'].unique()) <= set(['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])
assert df['id'].is_monotonic_increasing
assert not df.isna().any().any()
print("✅ CSV looks good.")


