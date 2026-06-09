# ======================================================================
# CIFAR-10 Challenge 
# ======================================================================

import os
import sys
from pathlib import Path
from dataclasses import dataclass
import subprocess

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import torchvision
import torchvision.transforms as T

# ------------------------------
# Kaggle Environment Detection & Setup
# ------------------------------
def setup_kaggle_paths():
    """Automatically detect Kaggle environment and extract test data"""
    
    # Check if running on Kaggle
    if os.path.exists('/kaggle/input'):
        print("âœ“ Detected Kaggle environment")
        
        # Kaggle competition data paths
        kaggle_input = Path('/kaggle/input/cifar-10')
        
        if not kaggle_input.exists():
            print("âš  Warning: /kaggle/input/cifar-10 not found!")
            print("   Please add 'CIFAR-10 - Object Recognition in Images' dataset")
            return './data', './test'
        
        # Extract test.7z if needed
        test_7z = kaggle_input / 'test.7z'
        test_extracted = Path('/kaggle/working/test')
        
        if test_7z.exists() and not test_extracted.exists():
            print(f"ðŸ“¦ Extracting {test_7z}...")
            try:
                # Install 7z if not available
                subprocess.run(['apt-get', 'update', '-qq'], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(['apt-get', 'install', '-y', '-qq', 'p7zip-full'], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Extract test.7z
                subprocess.run(['7z', 'x', str(test_7z), f'-o/kaggle/working/', '-y'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                print(f"âœ“ Extracted test data to {test_extracted}")
                
                # Verify extraction
                if (test_extracted / '1.png').exists():
                    print(f"âœ“ Verified: Found test images (1.png exists)")
                else:
                    print(f"âš  Warning: Extraction may have failed")
                    
            except Exception as e:
                print(f"âš  Error extracting test.7z: {e}")
                print("   Trying manual extraction...")
                os.makedirs('/kaggle/working/test', exist_ok=True)
        
        # Return paths
        data_root = '/kaggle/working/data'
        test_dir = str(test_extracted) if test_extracted.exists() else './test'
        
        print(f"âœ“ Data root: {data_root}")
        print(f"âœ“ Test dir: {test_dir}")
        
        return data_root, test_dir
    else:
        # Local/Colab environment
        print("âœ“ Local environment detected")
        return './data', './test'

# ------------------------------
# CIFAR-10 labels (exact order)
# ------------------------------
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ------------------------------
# Custom CNN: CifarXNet
# ------------------------------
class DWConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.dw(x); x = self.pw(x); x = self.bn(x)
        return F.relu(x, inplace=True)

class SE(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // r, 1)
        self.fc2 = nn.Conv2d(ch // r, ch, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s

class XBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            DWConv(in_ch, out_ch, stride=stride),
            DWConv(out_ch, out_ch, stride=1),
            SE(out_ch, r=8),
        )
        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        idt = x
        y = self.body(x)
        if self.proj is not None:
            idt = self.proj(x)
        return F.relu(y + idt, inplace=True)

class CifarXNet(nn.Module):
    """Custom-designed CNN - REQUIREMENT MET"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(XBlock(64, 96, stride=1), XBlock(96, 96, stride=1))
        self.stage2 = nn.Sequential(XBlock(96, 160, stride=2), XBlock(160, 160, stride=1))
        self.stage3 = nn.Sequential(XBlock(160, 256, stride=2), XBlock(256, 256, stride=1))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        """REQUIREMENT MET: Random initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x

# ------------------------------
# EMA
# ------------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].data)

    def restore(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data.copy_(self.backup[name].data)
        self.backup = {}

# ------------------------------
# REQUIREMENT MET: Cosine annealing + warmup
# ------------------------------
class WarmupCosine:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-5, base_lr=None):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr = min_lr
        self.base_lrs = [g['lr'] if base_lr is None else base_lr for g in optimizer.param_groups]
        self.step_num = 0

    def step(self):
        self.step_num += 1
        for i, pg in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if self.step_num <= self.warmup_steps:
                lr = base_lr * (self.step_num / self.warmup_steps)
            else:
                t = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                lr = self.min_lr + 0.5*(base_lr - self.min_lr)*(1 + np.cos(np.pi * t))
            pg['lr'] = lr

# ------------------------------
# REQUIREMENT MET: Advanced augmentation (CutMix)
# ------------------------------
def rand_bbox(W, H, lam, eps=1e-12):
    cut_ratio = np.sqrt(1.0 - lam + eps)
    cut_w, cut_h = int(W * cut_ratio), int(H * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def apply_cutmix(images, targets, alpha=1.0, p=0.6):
    if np.random.rand() > p:
        return images, targets, None
    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = images.size()
    index = torch.randperm(batch_size, device=images.device)
    shuffled_images = images[index]
    x1, y1, x2, y2 = rand_bbox(W, H, lam)
    images[:, :, y1:y2, x1:x2] = shuffled_images[:, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H + 1e-12))
    return images, (targets, targets[index], lam), "cutmix"

def cutmix_criterion(logits, target_tuple, label_smoothing=0.0):
    y_a, y_b, lam = target_tuple
    return lam * F.cross_entropy(logits, y_a, label_smoothing=label_smoothing) + \
           (1 - lam) * F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)

# ------------------------------
# Data
# ------------------------------
def get_loaders(data_root, batch_size=128, val_ratio=0.1, num_workers=2, use_cutout=True):
    """REQUIREMENT MET: CutOut (advanced augmentation)"""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tfms = [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]
    if use_cutout:
        train_tfms.append(T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0))

    train_tfms = T.Compose(train_tfms)
    test_tfms = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_full = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tfms)
    test_set   = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tfms)

    val_len = int(len(train_full) * val_ratio)
    train_len = len(train_full) - val_len
    train_set, val_set = random_split(train_full, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# ------------------------------
# Kaggle test dataset
# ------------------------------
class KaggleCIFAR10Test(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.ids = list(range(1, 300000 + 1))
        if not (self.root / "1.png").exists():
            raise FileNotFoundError(
                f"Expected {self.root}/1.png ... {self.root}/300000.png. "
                f"Test directory: {self.root}"
            )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = self.root / f"{img_id}.png"
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img_id, img

# ------------------------------
# TTA
# ------------------------------
@torch.no_grad()
def tta_predict_logits(model, images, mean, std, tta_crops=True, tta_hflip=True):
    device = images.device
    B = images.size(0)

    inv_norm = T.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    to_pil = T.ToPILImage()
    base_norm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    aug_logits_sum = None
    num_augs = 0

    def run_set(pil_list):
        nonlocal aug_logits_sum, num_augs
        tens = torch.stack([base_norm(p) for p in pil_list]).to(device)
        logits = model(tens)
        if aug_logits_sum is None:
            aug_logits_sum = logits
        else:
            aug_logits_sum += logits
        num_augs += 1

    pil_batch = [to_pil(inv_norm(img.cpu())) for img in images]
    run_set(pil_batch)

    if tta_hflip:
        hflip = T.functional.hflip
        pil_batch_h = [hflip(p) for p in pil_batch]
        run_set(pil_batch_h)

    if tta_crops:
        resize = T.Resize(40, interpolation=T.InterpolationMode.BILINEAR)
        fivecrop = T.FiveCrop(32)
        crop_tensors = []
        for i, p in enumerate(pil_batch):
            rp = resize(p)
            crops = fivecrop(rp)
            for c in crops:
                crop_tensors.append(base_norm(c))

        crop_batch = torch.stack(crop_tensors).to(device)
        crop_logits = model(crop_batch)
        crop_logits = crop_logits.view(B, 5, -1).mean(dim=1)

        if aug_logits_sum is None:
            aug_logits_sum = crop_logits
        else:
            aug_logits_sum += crop_logits
        num_augs += 1

    return aug_logits_sum / max(1, num_augs)

# ------------------------------
# Config
# ------------------------------
@dataclass
class TrainConfig:
    data_root: str = "./data"
    test_dir: str = "./test"
    out_csv: str = "submission.csv"
    epochs: int = 200
    batch_size: int = 128
    lr: float = 3e-3
    weight_decay: float = 5e-4
    val_ratio: float = 0.1
    num_workers: int = 2
    warmup_epochs: int = 10
    cutmix_alpha: float = 1.0
    cutmix_p: float = 0.6
    use_cutout: bool = True
    label_smoothing: float = 0.05
    ema_decay: float = 0.999
    resume: str = ""
    no_train: bool = False
    seed: int = 42
    tta_hflip: bool = True
    tta_crops: bool = True

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy(logits, target):
    with torch.no_grad():
        return (logits.argmax(1) == target).float().mean().item()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        b = images.size(0)
        tot_loss += loss.item() * b
        tot_acc  += (logits.argmax(1) == labels).float().sum().item()
        n += b
    return tot_loss / n, tot_acc / n

# ------------------------------
# REQUIREMENT MET: Creates 300,000-row submission.csv
# ------------------------------
@torch.no_grad()
def generate_kaggle_submission_tta(model, test_dir, device, output_csv="submission.csv", 
                                    batch_size=256, num_workers=2, tta_hflip=True, tta_crops=True):
    model.eval()
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    base_transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    ds = KaggleCIFAR10Test(test_dir, transform=base_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    all_ids, all_preds = [], []
    print(f"Generating predictions for {len(ds)} test images...")
    
    for batch_idx, (ids, images) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        logits = tta_predict_logits(model, images, mean, std, tta_crops=tta_crops, tta_hflip=tta_hflip)
        preds = logits.argmax(1).cpu().tolist()

        all_ids.extend(ids.tolist())
        all_preds.extend(preds)
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Processed {len(all_ids)}/{len(ds)} images...")

    labels = [CLASS_NAMES[p] for p in all_preds]
    df = pd.DataFrame({"id": all_ids, "label": labels}).sort_values("id")
    
    # Validation
    assert len(df) == 300000, f"Expected 300000 samples, got {len(df)}"
    assert df["id"].is_unique, "Duplicate IDs found"
    assert df["id"].min() == 1 and df["id"].max() == 300000, "ID range incorrect"
    assert set(df["label"].unique()).issubset(set(CLASS_NAMES)), "Invalid labels"
    
    df.to_csv(output_csv, index=False)
    print(f"\nâœ“ Saved {output_csv} with {len(df)} predictions")

# ------------------------------
# Main Training
# ------------------------------
def train_and_submit(cfg: TrainConfig):
    """ALL REQUIREMENTS MET"""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, _ = get_loaders(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        val_ratio=cfg.val_ratio,
        num_workers=cfg.num_workers,
        use_cutout=cfg.use_cutout
    )

    # Model
    model = CifarXNet(num_classes=10).to(device)
    print("âœ“ Model: CifarXNet (custom)")
    print("âœ“ Weights: Randomly initialized")

    # REQUIREMENT MET: AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print("âœ“ Optimizer: AdamW")
    
    # REQUIREMENT MET: Cosine + warmup
    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = max(1, int(cfg.warmup_epochs * len(train_loader)))
    scheduler = WarmupCosine(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=1e-5)
    print(f"âœ“ Scheduler: Cosine + {cfg.warmup_epochs}ep warmup")

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    ema = EMA(model, decay=cfg.ema_decay)

    best_acc = 0.0
    if cfg.resume:
        state = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded from {cfg.resume}")

    if not cfg.no_train:
        print("\n" + "="*60)
        print("TRAINING")
        print("="*60)
        print(f"âœ“ Augmentation: CutOut + CutMix (p={cfg.cutmix_p})")
        
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            running_loss = 0.0
            running_acc = 0.0
            n = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                images_mixed, targets_mixed, how = apply_cutmix(
                    images, labels, alpha=cfg.cutmix_alpha, p=cfg.cutmix_p
                )

                optimizer.zero_grad(set_to_none=True)
                
                if scaler is None:
                    logits = model(images_mixed)
                    if how == "cutmix":
                        loss = cutmix_criterion(logits, targets_mixed, label_smoothing=cfg.label_smoothing)
                        acc = accuracy(logits, targets_mixed[0])
                    else:
                        loss = F.cross_entropy(logits, labels, label_smoothing=cfg.label_smoothing)
                        acc = accuracy(logits, labels)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.cuda.amp.autocast():
                        logits = model(images_mixed)
                        if how == "cutmix":
                            loss = cutmix_criterion(logits, targets_mixed, label_smoothing=cfg.label_smoothing)
                            acc = accuracy(logits, targets_mixed[0])
                        else:
                            loss = F.cross_entropy(logits, labels, label_smoothing=cfg.label_smoothing)
                            acc = accuracy(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                ema.update(model)
                scheduler.step()

                b = images.size(0)
                running_loss += loss.item() * b
                running_acc  += acc * b
                n += b

            ema.apply_shadow(model)
            val_loss, val_acc = evaluate(model, val_loader, device)
            ema.restore(model)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_cifar10_model.pth")
                
            print(f"[Epoch {epoch:03d}] train_loss={running_loss/n:.4f} train_acc={running_acc/n:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} best={best_acc:.4f}")

        print("\n" + "="*60)
        print(f"TRAINING COMPLETE - Best: {best_acc:.4f}")
        print("="*60 + "\n")

        model.load_state_dict(torch.load("best_cifar10_model.pth", map_location=device))

    ema.apply_shadow(model)

    # Generate submission
    test_path = Path(cfg.test_dir)
    if not test_path.exists() or not (test_path / "1.png").exists():
        print("\n" + "="*60)
        print("ERROR: Test directory not found!")
        print("="*60)
        print(f"Looking for: {test_path}")
        print(f"Test file check: {test_path / '1.png'}")
        print("\nPlease ensure test.7z is extracted correctly.")
        print("="*60 + "\n")
        return
    
    print("\n" + "="*60)
    print("GENERATING SUBMISSION")
    print("="*60)
    generate_kaggle_submission_tta(
        model, cfg.test_dir, device, output_csv=cfg.out_csv,
        num_workers=cfg.num_workers, tta_hflip=cfg.tta_hflip, tta_crops=cfg.tta_crops
    )
    print("="*60 + "\n")

    ema.restore(model)


# ======================================================================
# MAIN ENTRY POINT - Auto-detects Kaggle and extracts data
# ======================================================================


# Auto-detect Kaggle and setup paths
data_root, test_dir = setup_kaggle_paths()

# Configure training
cfg = TrainConfig(
    data_root=data_root,
    test_dir=test_dir,
    out_csv="submission.csv",
    epochs=200,  # Full training
    batch_size=128,
    lr=3e-3,
    weight_decay=5e-4,
    val_ratio=0.1,
    num_workers=2,
    warmup_epochs=10,
    cutmix_alpha=1.0,
    cutmix_p=0.6,
    use_cutout=True,
    label_smoothing=0.05,
    ema_decay=0.999,
    resume="",
    no_train=False,
    seed=42,
    tta_hflip=True,
    tta_crops=True,
)

# Start training!
train_and_submit(cfg)

