!pip install -q py7zr



# ============================================================
# Cell 1: Ultra-fast extraction + config + datasets + loaders
# ============================================================

import os, math, random, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import py7zr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 1337
def set_seed(seed=SEED):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
set_seed(SEED)

print("="*70)
print("ULTRA-FAST EXTRACTION WITH PY7ZR")
print("="*70)

# ----------------------------
# Extract train.7z -> /kaggle/working/
# ----------------------------
train_7z = next((p for p in ["../input/cifar-10/train.7z",
                             "/kaggle/input/cifar-10/train.7z"]
                 if Path(p).exists()), None)
assert train_7z, "train.7z not found"
with py7zr.SevenZipFile(train_7z, "r") as z:
    z.extractall(path="/kaggle/working/")
train_dir = Path("/kaggle/working/train")
train_count = len(list(train_dir.glob("*.png")))
print(f"âœ“ Extracted {train_count} training images -> {train_dir}")

# ----------------------------
# Labels
# ----------------------------
labels_csv = next((p for p in ["../input/cifar-10/trainLabels.csv",
                               "/kaggle/input/cifar-10/trainLabels.csv"]
                   if Path(p).exists()), None)
assert labels_csv, "trainLabels.csv not found"
train_labels = pd.read_csv(labels_csv)
classes = sorted(train_labels["label"].unique())
print(f"âœ“ Classes ({len(classes)}): {classes}")
print(f"âœ“ Training samples: {len(train_labels):,}")

# ----------------------------
# Extract test.7z -> /kaggle/working/
# ----------------------------
test_7z = next((p for p in ["../input/cifar-10/test.7z",
                            "/kaggle/input/cifar-10/test.7z"]
                if Path(p).exists()), None)
assert test_7z, "test.7z not found"
with py7zr.SevenZipFile(test_7z, "r") as z:
    z.extractall(path="/kaggle/working/")

# possible layouts
test_dir1 = Path("/kaggle/working/test")
test_dir2 = Path("/kaggle/working/test/test")
if test_dir1.exists() and any(test_dir1.glob("*.png")):
    test_path = test_dir1
elif test_dir2.exists() and any(test_dir2.glob("*.png")):
    test_path = test_dir2
else:
    # fallback: find the folder with most PNGs under /kaggle/working
    pngs = list(Path("/kaggle/working").rglob("*.png"))
    from collections import Counter
    parent_counts = Counter(p.parent for p in pngs)
    test_path, _ = parent_counts.most_common(1)[0]
test_count = len(list(test_path.glob("*.png")))
print(f"âœ“ Extracted {test_count} test images -> {test_path}")
print("âœ“ Examples exist:", (test_path / "1.png").exists(), (test_path / "300000.png").exists())

print("\n" + "="*70)
print("âœ… EXTRACTION COMPLETE")
print("="*70)

# ----------------------------
# Config
# ----------------------------
class CFG:
    img_size = 32
    num_classes = 10
    epochs = 120          # bump to 100-120 for â‰¥0.94
    batch_size = 512
    lr = 3e-3
    weight_decay = 1e-4
    warmup_epochs = 5
    num_workers = 4
    use_amp = True
    cutmix_alpha = 1.0
    cutmix_prob  = 0.5
    cutmix_warmup_epochs = 5   # CutMix off for first N epochs
    val_ratio = 0.05
    ema_decay = 0.999
    tta_times = 4
    min_lr = 1e-5

CFG.device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", CFG.device)

# ----------------------------
# Class maps
# ----------------------------
cls2idx = {c:i for i,c in enumerate(classes)}
idx2cls = {i:c for c,i in cls2idx.items()}

# ----------------------------
# Datasets
# ----------------------------
class Cifar10Train(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.dir = Path(img_dir)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.dir / f"{r['id']}.png").convert("RGB")
        if self.transform: img = self.transform(img)
        return img, cls2idx[r["label"]]

class Cifar10Test(Dataset):
    def __init__(self, img_dir, transform=None):
        self.dir = Path(img_dir)
        self.ids = sorted([p.stem for p in self.dir.glob("*.png")], key=lambda x:int(x))
        self.transform = transform
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        img_id = self.ids[i]
        img = Image.open(self.dir / f"{img_id}.png").convert("RGB")
        if self.transform: img = self.transform(img)
        return img, img_id

# ----------------------------
# Transforms  (IMPORTANT: normalization added)
# ----------------------------
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD  = [0.2470, 0.2435, 0.2616]

train_tfms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(CFG.img_size, padding=4, padding_mode="reflect"),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

test_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# ----------------------------
# CutMix helpers + collate
# ----------------------------
def rand_bbox(W, H, lam):
    import math, numpy as np
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
    cx = np.random.randint(W); cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w//2, 0, W); y1 = np.clip(cy - cut_h//2, 0, H)
    x2 = np.clip(cx + cut_w//2, 0, W); y2 = np.clip(cy + cut_h//2, 0, H)
    return x1, y1, x2, y2

def cutmix_collate(batch):
    import numpy as np, torch
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = torch.tensor([b[1] for b in batch], dtype=torch.long)
    if np.random.rand() < CFG.cutmix_prob and CFG.cutmix_alpha > 0:
        lam = np.random.beta(CFG.cutmix_alpha, CFG.cutmix_alpha)
        B, C, H, W = images.size()
        idx = torch.randperm(B)
        x1,y1,x2,y2 = rand_bbox(W, H, lam)
        images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
        lam = 1 - ((x2-x1)*(y2-y1)/(W*H))
        return images, (targets, targets[idx], lam)
    else:
        return images, (targets, targets, 1.0)

# ----------------------------
# Split + loaders
# ----------------------------
full_train_df = train_labels.copy()
n_train = int(len(full_train_df) * (1 - CFG.val_ratio))
n_val   = len(full_train_df) - n_train

tr_idx, va_idx = random_split(
    range(len(full_train_df)), [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED)
)
train_df = full_train_df.iloc[tr_idx.indices].reset_index(drop=True)
val_df   = full_train_df.iloc[va_idx.indices].reset_index(drop=True)

TRAIN_DIR = train_dir
TEST_DIR  = test_path

train_ds = Cifar10Train(train_df, TRAIN_DIR, transform=train_tfms)
val_ds   = Cifar10Train(val_df,   TRAIN_DIR, transform=test_tfms)

train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                          num_workers=CFG.num_workers, pin_memory=True,
                          collate_fn=cutmix_collate, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False,
                          num_workers=CFG.num_workers, pin_memory=True)

print(f"Train/Val sizes: {len(train_ds)} / {len(val_ds)}")
print("ðŸ‘‰ Run Cell 2 to train, then Cell 3 to create submission.csv")

# (Optional) tiny overfit sanity loader (uncomment in Cell 2 to use)
tiny_idx = list(range(min(256, len(train_ds))))
tiny_loader = DataLoader(torch.utils.data.Subset(train_ds, tiny_idx),
                         batch_size=128, shuffle=True, num_workers=2)



## check the test file only 
from pathlib import Path
import re

def find_test_dir():
    cand = [Path("/kaggle/working/test"), Path("/kaggle/working/test/test")]
    for p in cand:
        if p.exists() and any(p.glob("*.png")):
            # Make sure files are numeric 1..300000
            sample = next((x for x in p.glob("*.png")), None)
            if sample and re.fullmatch(r"\d+\.png", sample.name):
                return p
    # fallback: search the tree
    all_png_dirs = {}
    for png in Path("/kaggle/working").rglob("*.png"):
        if re.fullmatch(r"\d+\.png", png.name):
            all_png_dirs[png.parent] = all_png_dirs.get(png.parent, 0) + 1
    assert all_png_dirs, "No numeric test PNGs found after extraction."
    # choose the dir with most numeric pngs
    test_dir = max(all_png_dirs, key=all_png_dirs.get)
    return test_dir

TEST_DIR = find_test_dir()
print("TEST_DIR =", TEST_DIR)
print("Has 1.png? ", (TEST_DIR / "1.png").exists(), " | Has 300000.png? ", (TEST_DIR / "300000.png").exists())



# ============================================================
# Cell 2: Custom model + training (AdamW, cosine+warmup, CutMix, AMP, EMA)
# Depends on Cell 1 globals: CFG, train_loader, val_loader, tiny_loader (optional)
# ============================================================

import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ------------------------------------------------------------
# Toggle: evaluate with EMA or live model (start with live!)
# ------------------------------------------------------------
EVAL_WITH_EMA = False   # set True later (e.g., after epoch 20) if you want

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c // r, 1)
        self.fc2 = nn.Conv2d(c // r, c, 1)
    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w

class XBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, expansion=2):
        super().__init__()
        mid = out_c * expansion
        self.use_res = (in_c == out_c and stride == 1)
        self.pw1 = nn.Conv2d(in_c, mid, 1, bias=False); self.bn1 = nn.BatchNorm2d(mid)
        self.dw  = nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False); self.bn2 = nn.BatchNorm2d(mid)
        self.pw2 = nn.Conv2d(mid, out_c, 1, bias=False); self.bn3 = nn.BatchNorm2d(out_c)
        self.se  = SE(out_c, r=8); self.act = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity() if self.use_res else nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm2d(out_c)
        )
    def forward(self, x):
        out = self.act(self.bn1(self.pw1(x)))
        out = self.act(self.bn2(self.dw(out)))
        out = self.bn3(self.pw2(out))
        out = self.se(out)
        return F.relu(out + self.shortcut(x), inplace=True)

class CifarXNet(nn.Module):
    def __init__(self, num_classes=10, width=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(XBlock(width, width), XBlock(width, width))
        self.stage2 = nn.Sequential(XBlock(width, width*2, stride=2), XBlock(width*2, width*2))
        self.stage3 = nn.Sequential(XBlock(width*2, width*4, stride=2), XBlock(width*4, width*4))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width*4, num_classes))
        self._init_weights()
    def _init_weights(self):
        # REQUIREMENT: Randomly initialized weights (no pretraining)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.stem(x); x = self.stage1(x); x = self.stage2(x); x = self.stage3(x)
        return self.head(x)

# ------------------------------------------------------------
# EMA
# ------------------------------------------------------------
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = CifarXNet(num_classes=CFG.num_classes).to(CFG.device)
        self.ema.load_state_dict(model.state_dict())
        self.decay = decay
        for p in self.ema.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        d = self.decay; msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point: v.copy_(v * d + (1. - d) * msd[k])

def build_warmup_cosine_scheduler(optim, num_epochs, warmup_epochs, min_lr, base_lr):
    def lr_lambda(e):
        if e < warmup_epochs: return (e + 1) / max(1, warmup_epochs)
        p = (e - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        cos = 0.5 * (1 + math.cos(math.pi * p))
        return (min_lr / base_lr) + (1 - (min_lr / base_lr)) * cos
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

criterion = nn.CrossEntropyLoss()

def cutmix_criterion(pred, targets_tuple):
    y1, y2, lam = targets_tuple
    return lam * criterion(pred, y1) + (1 - lam) * criterion(pred, y2)

def accuracy(output, target):
    return (output.argmax(1) == target).float().mean().item()

# AMP compatibility (old/new PyTorch)
GradScaler = torch.cuda.amp.GradScaler if hasattr(torch.cuda, "amp") else torch.amp.GradScaler
amp_context = torch.cuda.amp.autocast if hasattr(torch.cuda.amp, "autocast") else torch.amp.autocast

def evaluate(model, ema, loader):
    net = ema.ema if EVAL_WITH_EMA else model
    net.eval()
    losses, accs = [], []
    with torch.no_grad(), amp_context(enabled=CFG.use_amp):
        for batch in loader:
            # val loader yields (images, targets)
            if len(batch) == 2 and not isinstance(batch[1], (tuple, list)):
                images, targets = batch
            else:
                images, targets = batch[0], batch[1][0]
            images, targets = images.to(CFG.device), targets.to(CFG.device)
            logits = net(images)
            losses.append(criterion(logits, targets).item())
            accs.append(accuracy(logits, targets))
    return float(np.mean(losses)), float(np.mean(accs))

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------
model = CifarXNet(num_classes=CFG.num_classes).to(CFG.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
scheduler = build_warmup_cosine_scheduler(optimizer, CFG.epochs, CFG.warmup_epochs, CFG.min_lr, CFG.lr)
scaler = GradScaler(enabled=CFG.use_amp)
ema = ModelEMA(model, decay=CFG.ema_decay)

best_acc = 0.0
for epoch in range(CFG.epochs):
    model.train()
    total_loss = total_acc = n = 0
    start = time.time()

    # CutMix warm-up: disable for first N epochs
    use_cutmix = (epoch >= CFG.cutmix_warmup_epochs)

    for images, targets_tuple in train_loader:
        images = images.to(CFG.device)
        if not use_cutmix:
            y = targets_tuple[0]
            targets_tuple = (y, y, 1.0)
        targets_tuple = tuple(t.to(CFG.device) if torch.is_tensor(t) else t for t in targets_tuple)

        optimizer.zero_grad(set_to_none=True)
        with amp_context(enabled=CFG.use_amp):
            logits = model(images)
            loss = cutmix_criterion(logits, targets_tuple)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        ema.update(model)

        total_loss += loss.item()
        total_acc  += accuracy(logits.detach(), targets_tuple[0]); n += 1

    scheduler.step()
    val_loss, val_acc = evaluate(model, ema, val_loader)
    lr_now = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1:03d}/{CFG.epochs} | lr={lr_now:.6f} | "
          f"train_loss={total_loss/n:.4f} train_acc={total_acc/n:.4f} | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {time.time()-start:.1f}s")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best.pth")        # live model
        torch.save(ema.ema.state_dict(), "best_ema.pth")  # optional

print(f"\nâœ… Best val_acc (live model): {best_acc:.4f}")
print("ðŸ’¾ Saved checkpoints: best.pth (live), best_ema.pth (EMA)")



# ============================================================
# Cell 3: Load best (live) model, TTA inference, write submission.csv (300,000 rows)
# Depends on Cell 1 globals: CFG, Cifar10Test, TEST_DIR, test_tfms, idx2cls
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# --- redeclare model (safe if already defined) ---
class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c // r, 1); self.fc2 = nn.Conv2d(c // r, c, 1)
    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1); w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w)); return x * w

class XBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, expansion=2):
        super().__init__()
        mid = out_c * expansion
        self.use_res = (in_c == out_c and stride == 1)
        self.pw1 = nn.Conv2d(in_c, mid, 1, bias=False); self.bn1 = nn.BatchNorm2d(mid)
        self.dw  = nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False); self.bn2 = nn.BatchNorm2d(mid)
        self.pw2 = nn.Conv2d(mid, out_c, 1, bias=False); self.bn3 = nn.BatchNorm2d(out_c)
        self.se  = SE(out_c, r=8); self.act = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity() if self.use_res else nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm2d(out_c)
        )
    def forward(self, x):
        out = self.act(self.bn1(self.pw1(x)))
        out = self.act(self.bn2(self.dw(out)))
        out = self.bn3(self.pw2(out)); out = self.se(out)
        return F.relu(out + self.shortcut(x), inplace=True)

class CifarXNet(nn.Module):
    def __init__(self, num_classes=10, width=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(XBlock(width, width), XBlock(width, width))
        self.stage2 = nn.Sequential(XBlock(width, width*2, stride=2), XBlock(width*2, width*2))
        self.stage3 = nn.Sequential(XBlock(width*2, width*4, stride=2), XBlock(width*4, width*4))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width*4, CFG.num_classes))
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.stem(x); x = self.stage1(x); x = self.stage2(x); x = self.stage3(x)
        return self.head(x)

# AMP compatibility
amp_context = torch.cuda.amp.autocast if hasattr(torch.cuda.amp, "autocast") else torch.amp.autocast

# Build test loader
test_ds = Cifar10Test(TEST_DIR, transform=test_tfms)
test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False,
                         num_workers=CFG.num_workers, pin_memory=True)

# Load best *live* model checkpoint (not EMA)
model = CifarXNet(num_classes=CFG.num_classes).to(CFG.device)
state = torch.load("best.pth", map_location="cpu")
model.load_state_dict(state); model.eval()

@torch.no_grad()
def predict_tta(model, loader, tta_times=4):
    probs_list, ids_all = [], []
    for images, ids in loader:
        images = images.to(CFG.device)
        with amp_context(enabled=CFG.use_amp):
            logits = model(images)
            prob = F.softmax(logits, dim=1)
            for _ in range(tta_times - 1):
                flipped = torch.flip(images, dims=[3])  # horizontal flip
                prob += F.softmax(model(flipped), dim=1)
            prob /= tta_times
        probs_list.append(prob.cpu()); ids_all.extend(ids)
    probs = torch.cat(probs_list, dim=0)
    preds = probs.argmax(1).numpy()
    return ids_all, preds

test_ids, test_preds_idx = predict_tta(model, test_loader, tta_times=CFG.tta_times)
test_preds = [idx2cls[int(i)] for i in test_preds_idx]

sub = pd.DataFrame({"id": test_ids, "label": test_preds})
sub["id"] = sub["id"].astype(int)
sub = sub.sort_values("id").reset_index(drop=True)
assert len(sub) == 300_000, f"Submission must have 300,000 rows, got {len(sub)}"

sub.to_csv("submission.csv", index=False)
print(sub.head(), "\n"); print(sub.tail(), "\n")
print(f"âœ… Wrote submission.csv with {len(sub)} rows")


