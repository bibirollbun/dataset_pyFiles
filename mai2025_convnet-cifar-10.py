# =========================
# Cell 1: Config & Setup
# =========================
import os, io, re, time, math, glob, random, shutil, copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast          # ← new AMP API
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18
from torchvision.transforms import RandAugment, RandomErasing

@dataclass
class Cfg:
    work_dir: str = "./work"
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    img_size: int = 32
    batch_size: int = 256
    num_workers: int = 4

    # Train
    epochs: int = 150                      # ↑ longer schedule for SGD+cosine
    base_lr: float = 0.20                  # SGD base LR
    weight_decay: float = 5e-4
    momentum: float = 0.9
    nesterov: bool = True

    # Aug & regularization
    use_mixup: bool = True
    use_cutmix: bool = True                # enable both
    mix_alpha: float = 0.2
    cut_alpha: float = 1.0                 # stronger CutMix

    # EMA & AMP
    use_ema: bool = True
    ema_decay: float = 0.999
    use_amp: bool = True

    # Submission
    submit_stream: bool = True             # True = fast (no extraction)
    tta_hflip: bool = True                 # TTA on submit
    submit_batch: int = 1024

cfg = Cfg()
os.makedirs(cfg.work_dir, exist_ok=True)

# Repro & device
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(cfg.seed)
device = torch.device(cfg.device)
print("Device:", device)

# CIFAR-10 normalization & classes
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)
CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# =========================
# Cell 2: Data & Dataloaders
# =========================
# Strong but cheap augmentations
train_tfms = T.Compose([
    T.RandomCrop(cfg.img_size, padding=4),
    T.RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=9),                 # NEW
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
    RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),  # NEW
])
test_tfms  = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])

train_set = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tfms)
test_set  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tfms)

def to_onehot(y, num_classes=10):
    return F.one_hot(y, num_classes=num_classes).float()

def mixup_cutmix_collate(batch):
    images, labels = zip(*batch)
    x = torch.stack(images, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    y_oh = to_onehot(y, 10)

    if cfg.use_cutmix and random.random() < 0.5:
        lam = np.random.beta(cfg.cut_alpha, cfg.cut_alpha)  # α=1.0
        B, C, H, W = x.size()
        idx = torch.randperm(B)
        cut_w = int(W * math.sqrt(1 - lam))
        cut_h = int(H * math.sqrt(1 - lam))
        cx = np.random.randint(W); cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W); x2 = np.clip(cx + cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H); y2 = np.clip(cy + cut_h // 2, 0, H)
        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam_adj = 1 - ((x2 - x1) * (y2 - y1)) / (W * H)
        y_oh = lam_adj * y_oh + (1 - lam_adj) * y_oh[idx]
    elif cfg.use_mixup:
        lam = np.random.beta(cfg.mix_alpha, cfg.mix_alpha)  # α=0.2
        B = x.size(0); idx = torch.randperm(B)
        x = lam * x + (1 - lam) * x[idx]
        y_oh = lam * y_oh + (1 - lam) * y_oh[idx]

    return x, y_oh

train_loader = DataLoader(
    train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
    num_workers=cfg.num_workers, pin_memory=True,
    collate_fn=mixup_cutmix_collate if (cfg.use_mixup or cfg.use_cutmix) else None,
)
val_loader = DataLoader(
    test_set, batch_size=1024, shuffle=False,
    num_workers=cfg.num_workers, pin_memory=True
)



# =========================
# Cell 3: Model, Optimizer, LR, EMA, AMP
# =========================
# Model — NO PRETRAIN (random init)
model = resnet18(weights=None, num_classes=10).to(device)
# (optional) re-init head explicitly
nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
nn.init.zeros_(model.fc.bias)

# Weight decay: apply to weights only (not BN/bias)
def split_weights(m):
    decay, no_decay = [], []
    for name, p in m.named_parameters():
        if not p.requires_grad: 
            continue
        if p.ndim == 1 or name.endswith(".bias"):  # BN/bias
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay,    "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

optimizer = torch.optim.SGD(
    split_weights(model), lr=cfg.base_lr,
    momentum=cfg.momentum, nesterov=cfg.nesterov
)

# 5-epoch warmup + cosine
def cosine_with_warmup_lambda(epoch):
    warmup = 5
    if epoch < warmup:
        return float(epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, (cfg.epochs - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_with_warmup_lambda)

# Loss (supports soft labels)
def soft_ce_loss(logits, targets):
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()

# AMP (new API)
scaler = GradScaler('cuda', enabled=cfg.use_amp)

# EMA (safe deepcopy-based)
class ModelEMA:
    def __init__(self, model, decay=cfg.ema_decay):
        self.decay = decay
        self.ema = copy.deepcopy(model).to(next(model.parameters()).device)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k in esd.keys():
            if esd[k].dtype.is_floating_point:
                esd[k].mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                esd[k].copy_(msd[k])

    def store(self, model):
        self._backup = copy.deepcopy(model.state_dict())

    def copy_to(self, model):
        model.load_state_dict(self.ema.state_dict(), strict=True)

    def restore(self, model):
        if self._backup is not None:
            model.load_state_dict(self._backup, strict=True)
            self._backup = None

ema = ModelEMA(model) if cfg.use_ema else None
ce = nn.CrossEntropyLoss()



# =========================
# Cell 4: Train / Eval / Loop
# =========================
def _train_acc_from_logits(logits, targets):
    preds = logits.argmax(1)
    tgt = targets.argmax(1) if targets.ndim == 2 else targets
    return (preds == tgt).float().mean().item()

def train_one_epoch(epoch):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    t0 = time.time()
    for images, targets in train_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda', enabled=cfg.use_amp):
            logits = model(images)
            loss = soft_ce_loss(logits, targets) if targets.ndim == 2 else ce(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if ema: ema.update(model)

        bs = images.size(0)
        running_loss += loss.item() * bs
        running_acc  += _train_acc_from_logits(logits.detach(), targets) * bs
        n += bs

    scheduler.step()
    return running_loss / n, running_acc / n, time.time() - t0

@torch.no_grad()
def evaluate(use_ema=True):
    if use_ema and ema:
        ema.store(model); ema.copy_to(model)
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = ce(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean().item()
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc  += acc * bs
        n += bs
    if use_ema and ema:
        ema.restore(model)
    return total_loss / n, total_acc / n

# Training loop
best_acc = 0.0
best_path = os.path.join(cfg.work_dir, "best_model.pth")

for epoch in range(cfg.epochs):
    tr_loss, tr_acc, dt = train_one_epoch(epoch)
    val_loss, val_acc = evaluate(use_ema=True)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save((ema.ema if (cfg.use_ema and ema) else model).state_dict(), best_path)

    lr_now = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:03d}/{cfg.epochs} | lr {lr_now:.5f} | "
          f"train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
          f"val_loss {val_loss:.4f} acc {val_acc:.4f} | {dt:.1f}s")

print("Best Val Acc:", best_acc)



# =========================
# Cell 5: Offline test accuracy (same split as val_loader)
# =========================
assert os.path.exists(best_path), f"Checkpoint not found: {best_path}"
test_loader_off = DataLoader(test_set, batch_size=1024, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

eval_model = resnet18(weights=None, num_classes=10).to(device).eval()
state = torch.load(best_path, map_location=device)
eval_model.load_state_dict(state, strict=True)
eval_model.eval()

correct, total = 0, 0
with torch.inference_mode():
    for xb, yb in test_loader_off:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        pred = eval_model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total   += yb.size(0)
final_acc = correct / total
print(f"[OFFLINE] CIFAR-10 test accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")



# =========================
# Cell 6  Extract 
# =========================
!pip -q install py7zr
import py7zr

# Find archive
candidates = glob.glob('/kaggle/input/**/test.7z', recursive=True)
archive_path = candidates[0]

shutil.rmtree('test', ignore_errors=True)
os.makedirs('test', exist_ok=True)
with py7zr.SevenZipFile(archive_path, mode='r') as z:
    z.extractall(path='test')

png_files = glob.glob('test/**/*.png', recursive=True)
print("PNG files:", len(png_files))
assert len(png_files) == 300000, f"Expected 300000, got {len(png_files)}"




# =========================
# Cell 7 : submit
# =========================
# Build dataset/loader
val_tfms = T.Compose([
    T.Resize((cfg.img_size, cfg.img_size), antialias=True),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])

class KaggleTestDataset(Dataset):
    def __init__(self, root="test", transform=None):
        self.transform = transform
        files = glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
        def keyfn(p):
            b = os.path.basename(p)
            m = re.match(r"(\d+)", os.path.splitext(b)[0]); 
            return int(m.group(1)) if m else 10**12
        files.sort(key=keyfn)
        self.paths = files
        self.ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in self.paths]
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        x = self.transform(Image.open(self.paths[i]).convert("RGB"))
        return x, self.ids[i]

test_ds = KaggleTestDataset(transform=val_tfms)
test_loader_submit = DataLoader(test_ds, batch_size=cfg.submit_batch, shuffle=False,
                                num_workers=cfg.num_workers, pin_memory=True)


# Predict
submit_model = resnet18(weights=None, num_classes=10).to(device).eval()
submit_model.load_state_dict(torch.load(best_path, map_location=device), strict=True)
submit_model.eval()

all_ids, all_preds = [], []
with torch.inference_mode():
    for xb, idb in test_loader_submit:
        xb = xb.to(device, non_blocking=True)
        logits = submit_model(xb)
        if cfg.tta_hflip:
            logits = (logits + submit_model(torch.flip(xb, dims=[3]))) / 2.0
        preds = logits.argmax(1).cpu().numpy()
        all_ids.append(idb.numpy()); all_preds.append(preds)

ids = np.concatenate(all_ids)
labels_idx = np.concatenate(all_preds)
labels = [CLASS_NAMES[i] for i in labels_idx]
sub = pd.DataFrame({"id": ids, "label": labels}).sort_values("id")

assert sub.shape[0] == 300000
assert sub["id"].min() == 1 and sub["id"].max() == 300000
assert sub["id"].nunique() == 300000

out_csv = os.path.join(cfg.work_dir, "submission.csv")
sub.to_csv(out_csv, index=False)
print(sub.head(), "\nWrote:", out_csv)

