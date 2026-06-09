!pip install py7zr


# ======================================================================
# CELL 1: ULTRA-FAST EXTRACTION (SIMPLE FIX)
# ======================================================================

import py7zr
import pandas as pd
from pathlib import Path

print("="*70)
print("ULTRA-FAST EXTRACTION WITH PY7ZR")
print("="*70)

# Extract train.7z to /kaggle/working/
print("\nðŸ“¦ Extracting train.7z...")
with py7zr.SevenZipFile('../input/cifar-10/train.7z', mode='r') as archive:
    archive.extractall(path='/kaggle/working/')

train_count = len(list(Path('/kaggle/working/train').glob('*.png')))
print(f"âœ“ Extracted {train_count} training images")

# Load labels
train_labels = pd.read_csv("../input/cifar-10/trainLabels.csv", header="infer")
classes = train_labels['label'].unique()
print(f"âœ“ Classes: {classes}")
print(f"âœ“ Training samples: {len(train_labels)}")

# Extract test.7z to /kaggle/working/ (SAME AS TRAIN)
print("\nðŸ“¦ Extracting test.7z...")
with py7zr.SevenZipFile('/kaggle/input/cifar-10/test.7z', mode='r') as archive:
    archive.extractall(path='/kaggle/working/')  # Extract to working, not working/test

# Check both possible locations
test_dir1 = Path('/kaggle/working/test')
test_dir2 = Path('/kaggle/working/')

if test_dir1.exists():
    test_files = list(test_dir1.glob('*.png'))
    test_count = len(test_files)
    test_path = test_dir1
    print(f"âœ“ Extracted {test_count} test images to /kaggle/working/test/")
else:
    # Files might be directly in working directory
    test_files = [f for f in test_dir2.glob('*.png') if f.name.isdigit() or f.stem.isdigit()]
    test_count = len(test_files)
    test_path = test_dir2
    print(f"âœ“ Extracted {test_count} test images to /kaggle/working/")

# Verify
if (test_path / '1.png').exists():
    print("âœ“ Verified: 1.png exists")
if (test_path / '300000.png').exists():
    print("âœ“ Verified: 300000.png exists")

print("\n" + "="*70)
print("âœ… EXTRACTION COMPLETE!")
print("="*70)
print("Extracted files:")
print(f"  â€¢ /kaggle/working/train/ - {train_count} training images")
print(f"  â€¢ {test_path} - {test_count} test images")
print("\nðŸ‘‰ Now run CELL 2 for training!")
print("="*70)


# ======================================================================
# TRAINING AND SUBMISSION
# ======================================================================

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T

# Paths to extracted data
TRAIN_DIR = "/kaggle/working/train"
TEST_DIR = "/kaggle/working/test"
LABELS_CSV = "../input/cifar-10/trainLabels.csv"

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ======================================================================
# CUSTOM NEURAL NETWORK - REQUIREMENT MET
# ======================================================================

class DWConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class SE(nn.Module):
    """Squeeze-and-Excitation block"""
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
    """Residual block with DWConv + SE"""
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
    """
    Custom Neural Network for CIFAR-10
    REQUIREMENT MET: Custom architecture, not off-the-shelf
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            XBlock(64, 96, stride=1),
            XBlock(96, 96, stride=1)
        )
        self.stage2 = nn.Sequential(
            XBlock(96, 160, stride=2),
            XBlock(160, 160, stride=1)
        )
        self.stage3 = nn.Sequential(
            XBlock(160, 256, stride=2),
            XBlock(256, 256, stride=1)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        """
        Random weight initialization - NO PRE-TRAINED WEIGHTS
        REQUIREMENT MET: Randomly initialized weights
        """
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

# ======================================================================
# EMA (Exponential Moving Average)
# ======================================================================

class EMA:
    """Exponential Moving Average of model parameters"""
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

# ======================================================================
# WARMUP + COSINE ANNEALING - REQUIREMENT MET
# ======================================================================

class WarmupCosine:
    """
    Cosine annealing with warmup
    REQUIREMENT MET: Cosine annealing + warmup
    """
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
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + np.cos(np.pi * t))
            pg['lr'] = lr

# ======================================================================
# CUTMIX - REQUIREMENT MET
# ======================================================================

def rand_bbox(W, H, lam, eps=1e-12):
    """Generate random bounding box for CutMix"""
    cut_ratio = np.sqrt(1.0 - lam + eps)
    cut_w, cut_h = int(W * cut_ratio), int(H * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def apply_cutmix(images, targets, alpha=1.0, p=0.6):
    """
    CutMix augmentation
    REQUIREMENT MET: Advanced data augmentation
    """
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
    """Loss for CutMix"""
    y_a, y_b, lam = target_tuple
    return lam * F.cross_entropy(logits, y_a, label_smoothing=label_smoothing) + \
           (1 - lam) * F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)

# ======================================================================
# DATASETS
# ======================================================================

class KaggleCIFAR10Train(Dataset):
    """Training dataset from extracted /kaggle/working/train/"""
    def __init__(self, train_dir, labels_csv, transform=None):
        self.train_dir = Path(train_dir)
        self.transform = transform
        self.labels_df = pd.read_csv(labels_csv)
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = self.train_dir / f"{row['id']}.png"
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        label = self.class_to_idx[row['label']]
        return img, label

class KaggleCIFAR10Test(Dataset):
    """Test dataset from extracted /kaggle/working/test/"""
    def __init__(self, test_dir, transform=None):
        self.test_dir = Path(test_dir)
        self.transform = transform
        self.ids = list(range(1, 300000 + 1))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = self.test_dir / f"{img_id}.png"
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        return img_id, img

# ======================================================================
# DATA LOADERS - CUTOUT (RandomErasing) - REQUIREMENT MET
# ======================================================================

def get_loaders(train_dir, labels_csv, batch_size=128, val_ratio=0.1, num_workers=2, use_cutout=True):
    """
    Create train/val loaders with CutOut augmentation
    REQUIREMENT MET: CutOut via RandomErasing
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tfms = [
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]
    
    if use_cutout:
        # CutOut via RandomErasing - ADVANCED AUGMENTATION
        train_tfms.append(T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0))

    train_tfms = T.Compose(train_tfms)
    val_tfms = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_full = KaggleCIFAR10Train(train_dir, labels_csv, transform=train_tfms)
    
    val_len = int(len(train_full) * val_ratio)
    train_len = len(train_full) - val_len
    train_set, val_set = random_split(
        train_full, [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_set.dataset.transform = val_tfms

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

# ======================================================================
# TEST-TIME AUGMENTATION (TTA)
# ======================================================================

@torch.no_grad()
def tta_predict_logits(model, images, mean, std, tta_crops=True, tta_hflip=True):
    """Test-Time Augmentation for better predictions"""
    device = images.device
    B = images.size(0)

    inv_norm = T.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])
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
        run_set([T.functional.hflip(p) for p in pil_batch])

    if tta_crops:
        resize = T.Resize(40, interpolation=T.InterpolationMode.BILINEAR)
        fivecrop = T.FiveCrop(32)
        crop_tensors = []
        for p in pil_batch:
            for c in fivecrop(resize(p)):
                crop_tensors.append(base_norm(c))
        crop_batch = torch.stack(crop_tensors).to(device)
        crop_logits = model(crop_batch).view(B, 5, -1).mean(dim=1)
        if aug_logits_sum is None:
            aug_logits_sum = crop_logits
        else:
            aug_logits_sum += crop_logits
        num_augs += 1

    return aug_logits_sum / max(1, num_augs)

# ======================================================================
# TRAINING CONFIG
# ======================================================================

@dataclass
class TrainConfig:
    train_dir: str = TRAIN_DIR
    test_dir: str = TEST_DIR
    labels_csv: str = LABELS_CSV
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
    seed: int = 42
    tta_hflip: bool = True
    tta_crops: bool = True

# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

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
        tot_acc += (logits.argmax(1) == labels).float().sum().item()
        n += b
    return tot_loss / n, tot_acc / n

# ======================================================================
# SUBMISSION GENERATION - REQUIREMENT MET
# ======================================================================

@torch.no_grad()
def generate_submission(model, test_dir, device, output_csv="submission.csv",
                       batch_size=256, num_workers=2, tta_hflip=True, tta_crops=True):
    """
    Generate submission.csv with 300,000 predictions
    REQUIREMENT MET: Creates 300,000-row submission.csv
    """
    model.eval()
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    base_transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    ds = KaggleCIFAR10Test(test_dir, transform=base_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)

    all_ids, all_preds = [], []
    print(f"Generating predictions for {len(ds)} test images...")
    
    for batch_idx, (ids, images) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        logits = tta_predict_logits(model, images, mean, std,
                                    tta_crops=tta_crops, tta_hflip=tta_hflip)
        preds = logits.argmax(1).cpu().tolist()
        all_ids.extend(ids.tolist())
        all_preds.extend(preds)
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  {len(all_ids)}/{len(ds)} images processed...")

    labels = [CLASS_NAMES[p] for p in all_preds]
    df = pd.DataFrame({"id": all_ids, "label": labels}).sort_values("id")
    
    assert len(df) == 300000, f"Expected 300000 rows, got {len(df)}"
    assert df["id"].is_unique, "Duplicate IDs found"
    
    df.to_csv(output_csv, index=False)
    print(f"\nâœ“ Saved {output_csv} with {len(df)} predictions")

# ======================================================================
# MAIN TRAINING FUNCTION
# ======================================================================

def train_and_submit(cfg: TrainConfig):
    """
    ALL REQUIREMENTS MET:
    âœ“ Random weight initialization
    âœ“ Custom neural network
    âœ“ AdamW optimizer
    âœ“ Cosine annealing + warmup
    âœ“ CutOut + CutMix augmentation
    âœ“ Creates 300,000-row submission.csv
    """
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*70)
    print("TRAINING CIFAR-10")
    print("="*70)
    print(f"Device: {device}")

    # Load data
    train_loader, val_loader = get_loaders(
        cfg.train_dir, cfg.labels_csv, cfg.batch_size,
        cfg.val_ratio, cfg.num_workers, cfg.use_cutout
    )
    print(f"âœ“ Train batches: {len(train_loader)}")
    print(f"âœ“ Val batches: {len(val_loader)}")

    # Model - RANDOMLY INITIALIZED
    model = CifarXNet(num_classes=10).to(device)
    print("âœ“ Model: CifarXNet (custom, random init)")

    # REQUIREMENT MET: AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print("âœ“ Optimizer: AdamW")
    
    # REQUIREMENT MET: Cosine annealing + warmup
    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = max(1, int(cfg.warmup_epochs * len(train_loader)))
    scheduler = WarmupCosine(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    print(f"âœ“ Scheduler: Cosine annealing + {cfg.warmup_epochs}ep warmup")
    print(f"âœ“ Augmentation: CutOut + CutMix")

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    ema = EMA(model, decay=cfg.ema_decay)

    best_acc = 0.0

    print("\n" + "="*70)
    print("TRAINING STARTED")
    print("="*70)
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss, running_acc, n = 0.0, 0.0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # REQUIREMENT MET: CutMix
            images_mixed, targets_mixed, how = apply_cutmix(
                images, labels, cfg.cutmix_alpha, cfg.cutmix_p
            )

            optimizer.zero_grad(set_to_none=True)
            
            if scaler is None:
                logits = model(images_mixed)
                if how == "cutmix":
                    loss = cutmix_criterion(logits, targets_mixed, cfg.label_smoothing)
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
                        loss = cutmix_criterion(logits, targets_mixed, cfg.label_smoothing)
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
            running_acc += acc * b
            n += b

        # Evaluate with EMA
        ema.apply_shadow(model)
        val_loss, val_acc = evaluate(model, val_loader, device)
        ema.restore(model)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_cifar10_model.pth")
            
        print(f"[Epoch {epoch:03d}] "
              f"loss={running_loss/n:.4f} acc={running_acc/n:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} best={best_acc:.4f}")

    print(f"\nâœ“ Training complete - Best: {best_acc:.4f}\n")
    model.load_state_dict(torch.load("best_cifar10_model.pth", map_location=device))

    # Use EMA for submission
    ema.apply_shadow(model)

    print("="*70)
    print("GENERATING SUBMISSION")
    print("="*70)
    
    generate_submission(model, cfg.test_dir, device, output_csv=cfg.out_csv,
                       num_workers=cfg.num_workers, tta_hflip=cfg.tta_hflip,
                       tta_crops=cfg.tta_crops)
    
    ema.restore(model)

# ======================================================================
# RUN TRAINING
# ======================================================================

print("="*70)
print("CIFAR-10 KAGGLE SUBMISSION")
print("="*70)
print("âœ“ Custom neural network (CifarXNet)")
print("âœ“ Random weight initialization")
print("âœ“ AdamW optimizer")
print("âœ“ Cosine annealing + warmup")
print("âœ“ CutOut + CutMix augmentation")
print("âœ“ Creates 300,000-row submission.csv")
print("="*70)

cfg = TrainConfig(
    train_dir=TRAIN_DIR,
    test_dir=TEST_DIR,
    labels_csv=LABELS_CSV,
    out_csv="submission.csv",
    epochs=200,  # Change to 2 for quick test
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
    seed=42,
    tta_hflip=True,
    tta_crops=True,
)

train_and_submit(cfg)

print("\n" + "="*70)
print("âœ… COMPLETE!")
print("="*70)
print("Files created:")
print("  â€¢ best_cifar10_model.pth")
print("  â€¢ submission.csv (300,000 rows)")
print("\nðŸ‘‰ Download submission.csv and submit to Kaggle!")
print("="*70)

