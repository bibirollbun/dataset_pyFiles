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
from torch.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as T
from torchvision.transforms import RandAugment, RandomErasing

# -------------------------
# Assignment Requirements:
# - Random init (no pretrain): satisfied by custom net below
# - Advanced optimizer: AdamW  ✅
# - Cosine LR w/ warmup: SequentialLR(LinearLR → CosineAnnealingLR) ✅
# - Advanced augmentation: CutMix (and explicit CutOut) ✅
# -------------------------

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
    epochs: int = 200               # more runway for cosine
    base_lr: float = 3e-3           # AdamW typical starting point
    weight_decay: float = 0.02      # decoupled weight decay
    # Optimizer-specific
    adamw_betas: tuple = (0.9, 0.999)
    adamw_eps: float = 1e-8

    # Aug & regularization
    use_mixup: bool = True
    use_cutmix: bool = True
    mix_alpha: float = 0.2
    cut_alpha: float = 1.0
    use_cutout: bool = True         # explicit CutOut toggle
    cutout_holes: int = 1
    cutout_max_frac: float = 0.5    # max hole side as fraction of H/W

    # EMA & AMP
    use_ema: bool = True
    ema_decay: float = 0.999
    use_amp: bool = True

    # LR schedule
    warmup_epochs: int = 10         # Linear warmup steps
    cosine_min_lr: float = 1e-6     # floor on cosine

    # Submission
    submit_stream: bool = True
    tta_hflip: bool = True
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
# Explicit CutOut (advanced aug) implementation
class Cutout(nn.Module):
    """
    CutOut: zero out one or more random square patches in the image.
    - holes: number of squares
    - max_frac: max side length as a fraction of image H/W
    Applies per-sample on a batch tensor (C,H,W) or PIL->tensor in transforms.
    """
    def __init__(self, holes=1, max_frac=0.5):
        super().__init__()
        self.holes = holes
        self.max_frac = max_frac

    def forward(self, img):
        # Works on tensor images after ToTensor (C,H,W)
        if not torch.is_tensor(img):
            return img
        C, H, W = img.shape
        for _ in range(self.holes):
            side = int(self.max_frac * min(H, W) * random.random())
            if side <= 0: 
                continue
            cx = random.randrange(W)
            cy = random.randrange(H)
            x1 = max(cx - side // 2, 0); x2 = min(cx + side // 2, W)
            y1 = max(cy - side // 2, 0); y2 = min(cy + side // 2, H)
            img[:, y1:y2, x1:x2] = 0.0
        return img


# Strong but stable augmentations (advanced: RandAugment + CutOut + optional RandomErasing)
train_tfms = T.Compose([
    T.RandomCrop(cfg.img_size, padding=4),
    T.RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=7),         # tuned down a bit for stability
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
    Cutout(holes=cfg.cutout_holes, max_frac=cfg.cutout_max_frac) if cfg.use_cutout else nn.Identity(),
    RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
])

test_tfms  = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])

train_set = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tfms)
test_set  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tfms)

def to_onehot(y, num_classes=10):
    return F.one_hot(y, num_classes=num_classes).float()

def onehot_smooth(y, n_classes=10, eps=0.1):
    y_oh = F.one_hot(y, n_classes).float()
    return y_oh * (1 - eps) + eps / n_classes

def mixup_cutmix_collate(batch):
    # Advanced augmentation at batch level (CutMix / MixUp)
    images, labels = zip(*batch)
    x = torch.stack(images, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    y_oh = to_onehot(y, 10)

    if cfg.use_cutmix and random.random() < 0.5:
        # ---- CutMix (advanced aug) ----
        lam = np.random.beta(cfg.cut_alpha, cfg.cut_alpha)
        B, C, H, W = x.size()
        idx = torch.randperm(B)

        cut_w = int(W * math.sqrt(1 - lam))
        cut_h = int(H * math.sqrt(1 - lam))
        cx = random.randrange(W)
        cy = random.randrange(H)
        x1 = max(cx - cut_w // 2, 0); x2 = min(cx + cut_w // 2, W)
        y1 = max(cy - cut_h // 2, 0); y2 = min(cy + cut_h // 2, H)

        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam_adj = 1.0 - ((x2 - x1) * (y2 - y1)) / float(W * H)
        y_oh = lam_adj * y_oh + (1 - lam_adj) * y_oh[idx]

    elif cfg.use_mixup:
        # ---- MixUp (also advanced) ----
        lam = np.random.beta(cfg.mix_alpha, cfg.mix_alpha)
        B = x.size(0); idx = torch.randperm(B)
        x = lam * x + (1 - lam) * x[idx]
        y_oh = lam * y_oh + (1 - lam) * y_oh[idx]

    else:
        # No batch-level aug -> use label smoothing
        y_oh = onehot_smooth(y, 10, eps=0.1)

    return x, y_oh

train_loader = DataLoader(
    train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
    num_workers=cfg.num_workers, pin_memory=True,
    collate_fn=mixup_cutmix_collate,
)
val_loader = DataLoader(
    test_set, batch_size=1024, shuffle=False,
    num_workers=cfg.num_workers, pin_memory=True
)



# =========================
# Cell 3: Model (custom), Optimizer (AdamW), LR (Warmup+Cosine), EMA, AMP
# =========================
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler

# -------- Custom residual network (random init, no pretrain) --------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

        # Random init (He/Kaiming); zero-init last BN in residual branch
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.proj is not None:
            identity = self.proj(identity)
        out = self.relu(out + identity)
        return out

def _make_layer(in_ch, out_ch, num_blocks, first_stride):
    layers = [BasicBlock(in_ch, out_ch, stride=first_stride)]
    for _ in range(num_blocks - 1):
        layers.append(BasicBlock(out_ch, out_ch, stride=1))
    return nn.Sequential(*layers)

class SmallResNet(nn.Module):
    """
    CIFAR-10 residual net (custom, random-init):
      Stem: 3x3, 64
      Stages: [64]x3, [128]x3, [256]x3, [512]x3 (stride=2 at first block of stages 2–4)
      Head: GAP -> Dropout(0.1) -> FC(512, 10)
    """
    def __init__(self, num_classes=10, p_drop=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _make_layer(64,   64,  num_blocks=3, first_stride=1)  # 32x32
        self.layer2 = _make_layer(64,   128, num_blocks=3, first_stride=2)  # 16x16
        self.layer3 = _make_layer(128,  256, num_blocks=3, first_stride=2)  # 8x8
        self.layer4 = _make_layer(256,  512, num_blocks=3, first_stride=2)  # 4x4
        self.pool   = nn.AdaptiveAvgPool2d((1,1))
        self.drop   = nn.Dropout(p=p_drop)
        self.fc     = nn.Linear(512, num_classes)

        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        return x

# Instantiate model
model = SmallResNet(num_classes=10).to(device)

# -------- AdamW with proper weight-decay grouping (exclude BN/bias) --------
def split_weights(m):
    decay, no_decay = [], []
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):  # BN/bias -> no weight decay
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay,    "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

optimizer = torch.optim.AdamW(
    split_weights(model),
    lr=cfg.base_lr,
    betas=cfg.adamw_betas,
    eps=cfg.adamw_eps
)

# -------- Cosine annealing with warmup (SequentialLR) --------
# NOTE: LinearLR requires 0 < start_factor <= 1. Use a tiny epsilon to approximate 0.
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1e-3,             # was 0.0 -> invalid; start near zero safely
    end_factor=1.0,
    total_iters=cfg.warmup_epochs
)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max(1, cfg.epochs - cfg.warmup_epochs),
    eta_min=cfg.cosine_min_lr
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[cfg.warmup_epochs]
)

# -------- Losses (supports soft targets from MixUp/CutMix) --------
def soft_ce_loss(logits, targets):
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()

ce = nn.CrossEntropyLoss()

# -------- AMP (CUDA-only) --------
scaler = GradScaler(
    device='cuda' if device.type == 'cuda' else 'cpu',
    enabled=(cfg.use_amp and device.type == 'cuda')
)

# -------- EMA (unchanged) --------
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
        msd = model.state_dict(); esd = self.ema.state_dict()
        for k in esd.keys():
            if esd[k].dtype.is_floating_point:
                esd[k].mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                esd[k].copy_(msd[k])

    def store(self, model):  self._backup = copy.deepcopy(model.state_dict())
    def copy_to(self, model): model.load_state_dict(self.ema.state_dict(), strict=True)
    def restore(self, model):
        if self._backup is not None:
            model.load_state_dict(self._backup, strict=True); self._backup = None

ema = ModelEMA(model) if cfg.use_ema else None



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

        with autocast(device_type='cuda', enabled=(cfg.use_amp and device.type=='cuda')):
            logits = model(images)
            loss = soft_ce_loss(logits, targets) if targets.ndim == 2 else ce(logits, targets)

        scaler.scale(loss).backward()
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    # Note: With SequentialLR, optimizer.param_groups[0]['lr'] updates per epoch
    lr_now = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:03d}/{cfg.epochs} | lr {lr_now:.6f} | "
          f"train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
          f"val_loss {val_loss:.4f} acc {val_acc:.4f} | {dt:.1f}s")

print("Best Val Acc:", best_acc)




# =========================
# Cell 5: Offline test accuracy (same split as val_loader)
# =========================
assert os.path.exists(best_path), f"Checkpoint not found: {best_path}"
test_loader_off = DataLoader(test_set, batch_size=1024, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

eval_model = SmallResNet(num_classes=10).to(device).eval()
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
# Cell 6: Kaggle test extraction (unchanged)
# =========================
!pip -q install py7zr
import py7zr
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
# Cell 7: Kaggle submission (unchanged, with hflip TTA)
# =========================
import os, re, glob
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch

png_count = len(glob.glob('test/**/*.png', recursive=True))
if png_count != 300000:
    raise RuntimeError(
        f"'test' folder not ready: found {png_count} PNGs. "
        "Run Cell 6 to extract the Kaggle test set, then rerun this cell."
    )

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
            m = re.match(r"(\d+)", os.path.splitext(b)[0])
            return int(m.group(1)) if m else 10**12
        files.sort(key=keyfn)
        self.paths = files
        self.ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in self.paths]
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        x = self.transform(Image.open(self.paths[i]).convert("RGB"))
        return x, self.ids[i]

test_ds = KaggleTestDataset(transform=val_tfms)
test_loader_submit = DataLoader(
    test_ds,
    batch_size=cfg.submit_batch,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True
)

assert os.path.exists(best_path), f"Checkpoint not found: {best_path}. Train first."
submit_model = SmallResNet(num_classes=10).to(device).eval()
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
assert sub.shape[0] == 300000, f"Expected 300000 rows, got {sub.shape[0]}"
assert sub["id"].min() == 1 and sub["id"].max() == 300000, "IDs must span 1..300000"
assert sub["id"].nunique() == 300000, "IDs must be unique"

out_csv = os.path.join(cfg.work_dir, "submission.csv")
os.makedirs(cfg.work_dir, exist_ok=True)
sub.to_csv(out_csv, index=False)
print(sub.head(), "\nWrote:", out_csv)

