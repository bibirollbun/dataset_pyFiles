# ============================================================
# Plant Seedlings — One-Block Training with Rich Logging (Fixed CAM)
# Backbones: customcnn / resnet18 / convnext_tiny
# Features: OneCycle, Mixup/CutMix, per-class metrics, LR track,
#           t-SNE features dump, Grad-CAM samples, resource profile
# ============================================================

import os, math, random, time, csv, json
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms as T, models
from tqdm.notebook import tqdm # Use notebook-friendly tqdm


@dataclass
class CFG:
    project_name: str = "seedlings-3backbones"
    # === choose model & pretrained ===
    model_name: str = "customcnn"    # "customcnn" | "resnet18" | "convnext_tiny"
    use_pretrained: bool = False     # True (ImageNet) or False (scratch)
    # ================================
    seed: int = 42
    num_workers: int = 0             # 0 is more stable on Kaggle
    epochs: int = 50                 # Recommended 50 for scratch, 30 for pretrained
    img_size: int = 224
    batch_size: int = 128
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    debug: bool = False              # Use a small subset of data for quick tests

    # LR scheduler: OneCycle
    use_onecycle: bool = True
    max_lr: float = 3e-3
    onecycle_pct_start: float = 0.15
    onecycle_div_factor: float = 25.0
    onecycle_final_div: float = 1e4
    # (Alternative Cosine scheduler, not used by default)
    lr: float = 1e-3
    min_lr: float = 1e-6
    warmup_epochs: int = 2

    # Mixup / CutMix (disabled during last `aug_off_frac` of training)
    use_mixup_cutmix: bool = True
    mixup_alpha: float = 0.1
    cutmix_alpha: float = 0.1
    aug_off_frac: float = 0.2

    # Optional fine-tune @288px
    do_finetune_288: bool = True
    finetune_epochs: int = 3
    finetune_lr: float = 5e-5

    # Inference
    do_tta: bool = True              # Use Test-Time Augmentation (horizontal flip)
    save_dir: str = "/kaggle/working"
    num_classes: int = None          # Will be set automatically based on data

    # ==== Rich logging for analysis ====
    log_dir: str = "/kaggle/working/metrics"
    enable_per_class: bool = True    # Log per-class metrics & confusion matrix
    enable_lr_track: bool = True     # Record learning rate per step
    enable_feature_dump: bool = True # Export validation set features for t-SNE
    enable_cam: bool = True          # Export Grad-CAM samples
    cam_samples: int = 12            # Number of CAM samples to export
    measure_resource: bool = True    # Profile model parameters/throughput/memory

cfg = CFG()


def set_seed(s=42):
    """Sets the seed for random, numpy, and torch for reproducibility."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True
set_seed(cfg.seed)

def fmt_time(sec):
    """Formats seconds into a MM:SS string."""
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

# --- Data paths ---
KAGGLE_INPUT = Path("/kaggle/input")
ROOT = KAGGLE_INPUT / "plant-seedlings-classification"
TRAIN = ROOT / "train"
TEST = ROOT / "test"
SAMPLE = ROOT / "sample_submission.csv"

# A quick check to ensure the dataset is available
assert TRAIN.exists(), "Please add the `Plant Seedlings Classification` dataset from Kaggle."


class DS(Dataset):
    """Custom PyTorch Dataset for loading seedling images."""
    def __init__(self, items, tfm):
        self.items, self.tfm = items, tfm
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("RGB")
        x = self.tfm(img)
        return x, y, str(p)

def build_split(val_ratio=0.1):
    """Scans the training data and creates stratified train/validation splits."""
    classes = sorted([d.name for d in TRAIN.iterdir() if d.is_dir()])
    c2i = {c: i for i, c in enumerate(classes)}
    items = []
    for c in classes:
        for p in (TRAIN / c).glob("*.*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                items.append((p, c2i[c]))
    y = np.array([b for _, b in items])
    idx = np.arange(len(items))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=cfg.seed)
    tr, va = next(sss.split(idx, y))
    return classes, [items[i] for i in tr], [items[i] for i in va]

def get_tfms(size, strong=True):
    """Returns image transformations for training and validation."""
    if strong:
        # Strong augmentations for the main training phase
        train = T.Compose([
            T.Resize(int(size * 1.14)),
            T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(3/4, 4/3)),
            T.RandomHorizontalFlip(0.5),
            T.RandAugment(2, 9),
            T.ToTensor(),
            T.RandomErasing(p=0.1),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        # Weaker augmentations for the final epochs or fine-tuning
        train = T.Compose([
            T.Resize(int(size * 1.14)),
            T.RandomResizedCrop(size, scale=(0.9, 1.0), ratio=(3/4, 4/3)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.RandomErasing(p=0.1),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    # Validation transforms (no augmentation, just resize and crop)
    valid = T.Compose([
        T.Resize(int(size * 1.14)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return train, valid

# Create the train/validation split
classes, train_items, valid_items = build_split(0.1)
cfg.num_classes = len(classes)
if cfg.debug: # If debugging, use a smaller subset
    train_items = train_items[:1000]
    valid_items = valid_items[:200]

print(f"Dataset: seedlings | classes={cfg.num_classes} | train={len(train_items)} | valid={len(valid_items)}")
print(f"Classes: {classes}")

# Create transforms and datasets
train_tfm, valid_tfm = get_tfms(cfg.img_size, strong=True)
train_ds = DS(train_items, train_tfm)
valid_ds = DS(valid_items, valid_tfm)

# Create dataloaders
train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True)

# Calculate soft class weights to handle class imbalance
freq = Counter([y for _, y in train_items])
gamma = 0.5 # Smoothing factor
w = torch.tensor([(1.0 / freq[i])**gamma for i in range(cfg.num_classes)], dtype=torch.float)
w = (w / w.mean())
print(f"Calculated class weights: {w.numpy().round(2)}")


class SqueezeExcite(nn.Module):
    """A simple Squeeze-and-Excitation block."""
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c // r, 1)
        self.fc2 = nn.Conv2d(c // r, c, 1)
    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.silu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w

class DWSeparable(nn.Module):
    """A Depthwise Separable Convolution block with optional Squeeze-Excite."""
    def __init__(self, in_c, out_c, k=3, s=1, p=1, se=True, drop=0.1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, k, s, p, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU()
        self.se = SqueezeExcite(out_c) if se else nn.Identity()
        self.dropout = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
    def forward(self, x):
        x = self.dw(x); x = self.pw(x); x = self.bn(x); x = self.act(x)
        x = self.se(x); x = self.dropout(x)
        return x

class CustomCNNS(nn.Module):
    """A custom lightweight CNN with ~1.3M parameters."""
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(),
        )
        self.stage1 = nn.Sequential(DWSeparable(64, 64), DWSeparable(64, 64))
        self.down1 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.stage2 = nn.Sequential(DWSeparable(128, 128, drop=0.15), DWSeparable(128, 128, drop=0.15))
        self.down2 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.stage3 = nn.Sequential(DWSeparable(256, 256, drop=0.2), DWSeparable(256, 256, drop=0.2))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, num_classes)
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.down1(x)
        x = self.stage2(x); x = self.down2(x)
        x = self.stage3(x)
        return self.head(x)

def _try_download_weights(build_fn_newapi, build_fn_legacy, name):
    """Helper to download pretrained weights, compatible with old/new torchvision APIs."""
    os.environ.setdefault("TORCH_HOME", "/kaggle/working/.cache/torch")
    os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
    try:
        m = build_fn_newapi()
        print(f"[weights] {name}: new API loaded.")
        return m
    except Exception as e1:
        print(f"[weights] {name}: new API failed -> {e1}")
        if build_fn_legacy is None: return None
        try:
            m = build_fn_legacy()
            print(f"[weights] {name}: legacy pretrained=True loaded.")
            return m
        except Exception as e2:
            print(f"[weights] {name}: legacy failed -> {e2}")
            return None

def build_resnet18(num_classes, pretrained=True):
    """Builds a ResNet-18 model with a modified classifier head."""
    if not pretrained:
        m = models.resnet18(weights=None)
    else:
        def newapi():
            from torchvision.models import ResNet18_Weights
            return models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        def legacy(): return models.resnet18(pretrained=True)
        m = _try_download_weights(newapi, legacy, "resnet18") or models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def build_convnext_tiny(num_classes, pretrained=True):
    """Builds a ConvNeXt-Tiny model with a modified classifier head."""
    if not pretrained:
        m = models.convnext_tiny(weights=None)
    else:
        def newapi():
            from torchvision.models import ConvNeXt_Tiny_Weights
            return models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        def legacy(): return models.convnext_tiny(pretrained=True)
        m = _try_download_weights(newapi, legacy, "convnext_tiny") or models.convnext_tiny(weights=None)
    in_f = m.classifier[2].in_features
    m.classifier[2] = nn.Linear(in_f, num_classes)
    return m

def build_model(name: str, num_classes: int, use_pretrained: bool):
    """A factory function to build the selected model."""
    name = name.lower()
    if name == "customcnn":     return CustomCNNS(num_classes)
    if name == "resnet18":      return build_resnet18(num_classes, pretrained=use_pretrained)
    if name == "convnext_tiny": return build_convnext_tiny(num_classes, pretrained=use_pretrained)
    raise ValueError(f"Unknown model: {name}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(cfg.model_name, cfg.num_classes, cfg.use_pretrained).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
criterion = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=cfg.label_smoothing)
scaler = torch.amp.GradScaler('cuda')

if cfg.use_onecycle:
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.max_lr, epochs=cfg.epochs, steps_per_epoch=steps_per_epoch,
        pct_start=cfg.onecycle_pct_start, div_factor=cfg.onecycle_div_factor,
        final_div_factor=cfg.onecycle_final_div, anneal_strategy='cos'
    )
else: # Fallback to a cosine scheduler with warmup
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = max(1, len(train_loader) * cfg.warmup_epochs)
    def lr_lambda(step):
        if step < warmup_steps: return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(cfg.min_lr / cfg.lr, 0.5 * (1 + math.cos(math.pi * prog)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


os.makedirs(cfg.log_dir, exist_ok=True)
class CSVLogger:
    """A simple CSV logger for training metrics."""
    def __init__(self, path, header):
        self.path = path
        self.header = header
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)
    def log(self, row_dict):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row_dict.get(h, "") for h in self.header])

trainlog = CSVLogger(
    os.path.join(cfg.log_dir, "train_curve.csv"),
    ["timestamp", "epoch", "phase", "loss", "acc", "f1", "lr", "lam", "time_sec", "model", "pretrained"]
)

# --- Resource quick profile ---
if cfg.measure_resource:
    with torch.no_grad():
        dummy = torch.randn(1, 3, cfg.img_size, cfg.img_size).to(device)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        
        # Warm-up pass
        t0 = time.time()
        _ = model(dummy)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.time()
        warm_latency = t1 - t0

        # Throughput test
        rep = 8
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(rep): _ = model(dummy)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t1 = time.time()
        thr = rep / (t1 - t0 + 1e-9)

        # Save results
        resource_data = {
            "model": cfg.model_name, "pretrained": bool(cfg.use_pretrained),
            "params_M": round(params_m, 3), "single_pass_s": round(warm_latency, 4),
            "throughput_iter_per_s": round(thr, 2), "img_size": cfg.img_size,
        }
        with open(os.path.join(cfg.log_dir, "resource.json"), "w") as f:
            json.dump(resource_data, f, indent=2)
        print(f"[Resource] Model: {cfg.model_name} | Params: {params_m:.2f}M | Throughput: {thr:.1f} iter/s")


def rand_bbox(W, H, lam):
    """Generates a random bounding box for CutMix."""
    cut_rat = math.sqrt(1. - lam)
    cw, ch = int(W * cut_rat), int(H * cut_rat)
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1, y1 = np.clip(cx - cw // 2, 0, W), np.clip(cy - ch // 2, 0, H)
    x2, y2 = np.clip(cx + cw // 2, 0, W), np.clip(cy + ch // 2, 0, H)
    return x1, y1, x2, y2

def apply_mixup(x, y, alpha):
    """Applies Mixup augmentation."""
    if alpha <= 0: return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def apply_cutmix(x, y, alpha):
    """Applies CutMix augmentation."""
    if alpha <= 0: return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    bs, C, H, W = x.size()
    idx = torch.randperm(bs, device=x.device)
    x1, y1, x2, y2 = rand_bbox(W, H, lam)
    xx = x.clone()
    xx[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return xx, y, y[idx], lam

def criterion_mc(crit, logits, y1, y2, lam):
    """Loss function for Mixup/CutMix. It's a linear combination of two losses."""
    if y2 is None: return crit(logits, y1)
    return lam * crit(logits, y1) + (1 - lam) * crit(logits, y2)


@torch.no_grad()
def validate(return_details: bool = False, epoch_idx: int = 0):
    """Evaluates the model on the validation set."""
    model.eval()
    loss_sum = acc_sum = n = 0
    all_logits = []
    all_true = []
    
    for x, y, _ in valid_loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda'):
            lo = model(x)
            ls = criterion(lo, y)
        
        bs = x.size(0)
        loss_sum += ls.item() * bs
        acc_sum += (lo.argmax(1) == y).float().sum().item()
        n += bs
        all_logits.append(lo.cpu())
        all_true.append(y.cpu())
        
    logits = torch.cat(all_logits, 0).numpy()
    true = torch.cat(all_true, 0).numpy()
    pred = logits.argmax(1)
    
    f1 = f1_score(true, pred, average="macro")
    val_loss, val_acc = loss_sum / n, acc_sum / n

    details = {}
    if return_details and cfg.enable_per_class:
        # Generate and save detailed reports
        rep = classification_report(true, pred, target_names=classes, output_dict=True, zero_division=0)
        cm = confusion_matrix(true, pred, labels=list(range(cfg.num_classes)))
        pd.DataFrame(rep).to_csv(os.path.join(cfg.log_dir, f"per_class_report_epoch{epoch_idx:03d}.csv"))
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(cfg.log_dir, f"confusion_matrix_epoch{epoch_idx:03d}.csv"))
        details["report_dict"] = rep
        details["confusion"] = cm.tolist()
        
    return (val_loss, val_acc, f1) if not return_details else (val_loss, val_acc, f1, details)


best_acc = 0.0
best_path = Path(cfg.save_dir) / f"{cfg.project_name}_{cfg.model_name}_{'pt' if cfg.use_pretrained else 'scratch'}_best.pt"

for epoch in range(cfg.epochs):
    model.train()
    # Check if we are in the final phase to turn off strong augmentations
    late_phase = (epoch >= int(cfg.epochs * (1 - cfg.aug_off_frac)))
    if late_phase:
        train_tfm, _ = get_tfms(cfg.img_size, strong=False)
        train_ds.tfm = train_tfm

    loss_sum = acc_sum = n = 0
    t0 = time.time()
    
    pbar = tqdm(train_loader, desc=f"[{cfg.model_name}] Epoch {epoch+1}/{cfg.epochs}")
    for x, y, _ in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        # Apply Mixup/CutMix
        y2, lam = None, 1.0
        if cfg.use_mixup_cutmix and not late_phase:
            if random.random() < 0.5 and cfg.mixup_alpha > 0:
                x, y, y2, lam = apply_mixup(x, y, cfg.mixup_alpha)
            elif cfg.cutmix_alpha > 0:
                x, y, y2, lam = apply_cutmix(x, y, cfg.cutmix_alpha)

        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            lo = model(x)
            ls = criterion_mc(criterion, lo, y, y2, lam)

        # Backward pass
        scaler.scale(ls).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Update running metrics
        bs = x.size(0)
        loss_sum += ls.item() * bs
        acc_sum += (lo.argmax(1) == y).float().sum().item()
        n += bs
        pbar.set_postfix(loss=f"{loss_sum/n:.4f}", acc=f"{acc_sum/n:.4f}")

    # --- End of Epoch ---
    tr_loss, tr_acc = loss_sum / n, acc_sum / n
    va_loss, va_acc, va_f1, _ = validate(return_details=True, epoch_idx=epoch + 1)
    
    print(f"[{cfg.model_name}] [{epoch+1:02d}/{cfg.epochs}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
          f"val_loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} | time={fmt_time(time.time()-t0)}")

    # Log metrics to CSV
    log_payload = {"timestamp": datetime.now().isoformat(timespec="seconds"), "epoch": epoch + 1,
                   "time_sec": f"{int(time.time()-t0)}", "model": cfg.model_name, "pretrained": int(cfg.use_pretrained)}
    trainlog.log({**log_payload, "phase": "train", "loss": f"{tr_loss:.6f}", "acc": f"{tr_acc:.6f}", "lr": f"{optimizer.param_groups[0]['lr']:.6e}"})
    trainlog.log({**log_payload, "phase": "valid", "loss": f"{va_loss:.6f}", "acc": f"{va_acc:.6f}", "f1": f"{va_f1:.6f}"})

    # Save best model
    if va_acc > best_acc:
        best_acc = va_acc
        torch.save({"model": model.state_dict(), "classes": classes}, best_path)
        print(f"  -> Saved new best (val_acc={best_acc:.4f}) to {best_path}")


if cfg.do_finetune_288:
    print(f"\n[FT] Starting fine-tuning at 288px for {cfg.finetune_epochs} epochs...")
    # Load the best model from the previous stage
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)

    # Update datasets with new image size and weaker augmentations
    ft_train_tfm, ft_valid_tfm = get_tfms(288, strong=False)
    train_ds.tfm = ft_train_tfm
    valid_ds.tfm = ft_valid_tfm
    
    # Use a smaller batch size for the larger image resolution to avoid memory issues
    ft_batch_size = max(64, cfg.batch_size // 2)
    train_loader = DataLoader(train_ds, batch_size=ft_batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=ft_batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
                              
    # Use a new optimizer with a small learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.finetune_lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda')

    for e in range(cfg.finetune_epochs):
        model.train()
        t0 = time.time()
        loss_sum = acc_sum = n = 0
        pbar = tqdm(train_loader, desc=f"FT Epoch {e+1}/{cfg.finetune_epochs}", leave=False)
        for x, y, _ in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                lo = model(x)
                ls = criterion(lo, y)
            scaler.scale(ls).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            bs = x.size(0)
            loss_sum += ls.item() * bs
            acc_sum += (lo.argmax(1) == y).float().sum().item()
            n += bs
            pbar.set_postfix(loss=f"{loss_sum/n:.4f}", acc=f"{acc_sum/n:.4f}")
            
        tr_loss, tr_acc = loss_sum / n, acc_sum / n
        va_loss, va_acc, va_f1, _ = validate(return_details=True, epoch_idx=cfg.epochs + e + 1)
        
        print(f"[FT {e+1}/{cfg.finetune_epochs}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} | time={fmt_time(time.time()-t0)}")
        
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(), "classes": classes}, best_path)
            print(f"  -> Saved new best after FT (val_acc={best_acc:.4f})")


print("\n[Inference] Preparing for test set prediction...")
# Load the final best model
ckpt = torch.load(best_path, map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval().to(device)

# Use the appropriate image size for evaluation
eval_size = 288 if cfg.do_finetune_288 else cfg.img_size
eval_tfm = T.Compose([
    T.Resize(int(eval_size * 1.14)),
    T.CenterCrop(eval_size),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

@torch.no_grad()
def tta_predict(pil, tfm):
    """Performs Test-Time Augmentation (TTA) prediction."""
    xs = [tfm(pil)]
    if cfg.do_tta:
        xs.append(T.functional.hflip(xs[0].clone())) # Add horizontally flipped version
    x = torch.stack(xs).to(device)
    with torch.amp.autocast('cuda'):
        # Average predictions from original and augmented images
        pr = model(x).softmax(1).mean(0).cpu().numpy()
    return pr

TEST_DIR = TEST if TEST.exists() else None
if TEST_DIR is not None:
    test_paths = sorted([p for p in TEST_DIR.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"]])
    rows = []
    for p in tqdm(test_paths, desc="Infer test"):
        pil = Image.open(p).convert("RGB")
        pred = int(np.argmax(tta_predict(pil, eval_tfm)))
        rows.append({"file": p.name, "species": classes[pred]})
    
    sub = pd.DataFrame(rows)
    out_csv = Path(cfg.save_dir) / f"submission_{cfg.model_name}_{'pt' if cfg.use_pretrained else 'scratch'}.csv"
    sub.to_csv(out_csv, index=False)
    print(f">> Saved submission to: {out_csv}")
    display(sub.head())
else:
    print("Test directory not found, skipping submission generation.")


def get_feature_extractor(model):
    """Returns a hook to extract features from the model's penultimate layer."""
    name = cfg.model_name.lower()
    model.eval()
    if name == "resnet18":
        backbone = nn.Sequential(*(list(model.children())[:-1])) # Exclude the final fc layer
        def fe(x): return backbone(x).view(x.size(0), -1)
        return fe
    elif name == "convnext_tiny":
        feats = nn.Sequential(model.features, model.avgpool)
        def fe(x): return feats(x).view(x.size(0), -1)
        return fe
    elif name == "customcnn":
        stem = nn.Sequential(model.stem, model.stage1, model.down1, model.stage2, model.down2, model.stage3)
        gap = nn.AdaptiveAvgPool2d(1)
        def fe(x): return gap(stem(x)).view(x.size(0), -1)
        return fe
    else:
        raise ValueError("Feature extractor not defined for this model.")

if cfg.enable_feature_dump:
    print("\n[Features] Dumping validation set features for t-SNE/UMAP...")
    fe = get_feature_extractor(model)
    feats, ys, files = [], [], []
    for x, y, paths in tqdm(valid_loader, desc="Extracting val features"):
        with torch.no_grad():
            f = fe(x.to(device)).cpu().numpy()
        feats.append(f)
        ys.append(y.numpy())
        files.extend(list(paths))
    
    feats = np.concatenate(feats, 0)
    ys = np.concatenate(ys, 0)
    
    # Save features, labels, and file index
    np.save(os.path.join(cfg.log_dir, "val_feats.npy"), feats)
    np.save(os.path.join(cfg.log_dir, "val_labels.npy"), ys)
    pd.DataFrame({"path": files, "label": [classes[i] for i in ys]}).to_csv(
        os.path.join(cfg.log_dir, "val_index.csv"), index=False
    )
    print(f"[Features] Saved! Features shape: {feats.shape}, Labels shape: {ys.shape}")


def get_cam_target_layer(model):
    """Identifies the target layer for Grad-CAM for different architectures."""
    name = cfg.model_name.lower()
    if name == "resnet18":
        return model.layer4[-1].conv2
    elif name == "convnext_tiny":
        # Target the last depthwise conv in the final stage
        try:
            return model.features[-1][-1].block[0]
        except Exception: # Fallback for different torchvision versions
            for m in model.features.modules():
                if isinstance(m, nn.Conv2d) and m.groups == m.in_channels:
                    layer = m
            if layer is None: raise RuntimeError("No depthwise conv found for ConvNeXt CAM.")
            return layer
    elif name == "customcnn":
        return model.stage3[-1].dw
    else:
        return None

def build_cam_handle(model):
    """Sets up forward and backward hooks to capture gradients and activations."""
    target_layer = get_cam_target_layer(model)
    if target_layer is None:
        print(f"[CAM] Not supported for model {cfg.model_name}, skipping.")
        return None, None
    
    activations = {}
    def fwd_hook(m, inp, out): activations["feat"] = out.detach()
    def bwd_hook(m, gin, gout): activations["grad"] = gout[0].detach()
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_backward_hook(bwd_hook)
    return activations, (h1, h2)

def save_cam_image(pil_img, cam_01, path_out, alpha=0.4):
    """Overlays the CAM heatmap on the original image and saves it."""
    try:
        import cv2 # Use OpenCV for colormapping if available
        rgb = np.array(pil_img)
        heat = (cam_01 * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[:, :, ::-1] # BGR to RGB
        over = (alpha * heat + (1 - alpha) * rgb).clip(0, 255).astype(np.uint8)
        Image.fromarray(over).save(path_out)
    except ImportError: # Fallback if cv2 is not installed
        rgb = np.array(pil_img).astype(np.float32)
        heat = np.zeros_like(rgb); heat[..., 0] = cam_01 * 255.0 # Simple red heatmap
        over = (alpha * heat + (1 - alpha) * rgb).clip(0, 255).astype(np.uint8)
        Image.fromarray(over).save(path_out)

if cfg.enable_cam:
    cam_dir = os.path.join(cfg.log_dir, "cam")
    os.makedirs(cam_dir, exist_ok=True)
    act, hooks = build_cam_handle(model)
    
    if act is not None:
        print("\n[CAM] Exporting Grad-CAM samples...")
        model.eval()
        cnt = 0
        # Iterate through validation data to find samples
        for x, y, paths in valid_loader:
            for i in range(x.size(0)):
                if cnt >= cfg.cam_samples: break
                xi, yi = x[i:i + 1].to(device), y[i:i + 1].to(device)
                pil = Image.open(paths[i]).convert("RGB").resize((eval_size, eval_size))
                
                # Get model output for the target class
                model.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    lo = model(xi)
                    score = lo[0, yi.item()]
                score.backward() # Backpropagate to get gradients
                
                feat, grad = act.get("feat"), act.get("grad")
                if feat is None or grad is None: continue

                # Grad-CAM calculation
                weights = grad.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((feat * weights).sum(1, keepdim=True))
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6) # Normalize to [0,1]
                cam = T.functional.resize(cam, [eval_size, eval_size])[0, 0].cpu().numpy()

                outp = os.path.join(cam_dir, f"{cnt:02d}_{classes[yi.item()]}.jpg")
                save_cam_image(pil, cam, outp)
                cnt += 1
            if cnt >= cfg.cam_samples: break
        
        for h in hooks: h.remove() # Clean up hooks
        print(f"[CAM] Saved {cnt} images to -> {cam_dir}")

print("\nAll done")

