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


%%writefile cifar10_resnet_baseline.py
"""
Assignment 4 — CIFAR10 Residual CNN (PyTorch) Baseline
(random init, residual connections)
"""

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

# -----------------------------
# Utils
# -----------------------------

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# Model: Basic ResNet for CIFAR-10
# -----------------------------

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, drop_prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        self.drop_prob = drop_prob

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.drop_prob > 0.0:
            out = F.dropout(out, p=self.drop_prob, training=self.training)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SmallResNet(nn.Module):
    """A compact ResNet designed for CIFAR-10 (32x32)."""
    def __init__(self, block: nn.Module, num_blocks: List[int], num_classes: int = 10, base: int = 64,
                 drop_prob: float = 0.0):
        super().__init__()
        self.in_planes = base
        self.conv1 = nn.Conv2d(3, base, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base)
        self.layer1 = self._make_layer(block, base, num_blocks[0], stride=1, drop_prob=drop_prob)
        self.layer2 = self._make_layer(block, base * 2, num_blocks[1], stride=2, drop_prob=drop_prob)
        self.layer3 = self._make_layer(block, base * 4, num_blocks[2], stride=2, drop_prob=drop_prob)
        self.layer4 = self._make_layer(block, base * 4, num_blocks[3], stride=2, drop_prob=drop_prob)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base * 4 * block.expansion, num_classes)

        # Kaiming init to keep it purely random-init (no pretrain)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, num_blocks, stride, drop_prob):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, drop_prob))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def resnet18tiny(num_classes=10, base=64, drop_prob=0.0):
    # 2-2-2-2 blocks: light variant for CIFAR-10
    return SmallResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, base=base, drop_prob=drop_prob)

# -----------------------------
# Data
# -----------------------------

def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    return train_tf, test_tf

class KaggleCIFAR10Train(Dataset):
    """Use Kaggle competition bundle (train.7z + trainLabels.csv)."""
    def __init__(self, train_dir: str, labels_csv: str, transform, id_filter: List[int] | None = None):
        super().__init__()
        import csv as _csv
        id2label = {}
        with open(labels_csv, "r") as f:
            r = _csv.reader(f)
            next(r)  # header
            for row in r:
                _id = int(row[0]); _lab = row[1]
                id2label[_id] = CIFAR10_CLASSES.index(_lab)
        self.id2label = id2label

        all_ids = [int(p.stem) for p in Path(train_dir).glob("*.png")]
        all_ids.sort()
        if id_filter is None:
            self.ids = all_ids
        else:
            filt = set(id_filter)
            self.ids = [i for i in all_ids if i in filt]
        self.train_dir = Path(train_dir)
        self.transform = transform

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        _id = self.ids[idx]
        img = Image.open(self.train_dir / f"{_id}.png").convert("RGB")
        x = self.transform(img)
        y = self.id2label[_id]
        return x, y

def _ensure_extract_7z(src_7z: str, out_dir: str):
    """Extracts 7z to out_dir if not already extracted (handles nested /train/train layout later)."""
    out = Path(out_dir)
    # Skip if we already see PNGs either directly under out_dir or under out_dir/train
    if out.exists() and (any(out.glob("*.png")) or any((out / "train").glob("*.png"))):
        return
    out.mkdir(parents=True, exist_ok=True)
    os.system(f"7z x {src_7z} -o{out_dir} -y > /dev/null")

def _resolve_image_dir(root_dir: str) -> Path:
    """Handle Kaggle 7z layout: sometimes files land under train/train/*.png."""
    p = Path(root_dir)
    if any(p.glob("*.png")):
        return p
    if (p / "train").exists() and any((p / "train").glob("*.png")):
        return p / "train"
    return p  # fallback

def build_dataloaders(data_root: str, batch_size: int, num_workers: int, val_split: int,
                      use_kaggle_bundle: bool = False, kaggle_input_root: str = "/kaggle/input/cifar-10") -> Tuple[DataLoader, DataLoader]:
    train_tf, test_tf = get_transforms()

    if use_kaggle_bundle:
        train_7z = Path(kaggle_input_root) / "train.7z"
        labels_csv = Path(kaggle_input_root) / "trainLabels.csv"
        extract_root = Path("/kaggle/working/train")
        _ensure_extract_7z(str(train_7z), str(extract_root))
        train_dir = _resolve_image_dir(str(extract_root))

        all_ids = sorted([int(p.stem) for p in train_dir.glob("*.png")])
        if len(all_ids) == 0:
            raise RuntimeError(f"No training PNGs found under {train_dir}. Check extraction.")
        rng = np.random.default_rng(42)
        rng.shuffle(all_ids)
        if val_split > 0:
            val_ids = all_ids[:val_split]
            tr_ids = all_ids[val_split:]
            train_ds = KaggleCIFAR10Train(str(train_dir), str(labels_csv), transform=train_tf, id_filter=tr_ids)
            val_ds = KaggleCIFAR10Train(str(train_dir), str(labels_csv), transform=test_tf, id_filter=val_ids)
        else:
            train_ds = KaggleCIFAR10Train(str(train_dir), str(labels_csv), transform=train_tf)
            val_ds = KaggleCIFAR10Train(str(train_dir), str(labels_csv), transform=test_tf, id_filter=all_ids[:5000])
    else:
        # Fallback: official torchvision CIFAR-10
        train_full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
        if val_split > 0:
            indices = list(range(len(train_full)))
            np.random.shuffle(indices)
            val_idx = indices[:val_split]
            train_idx = indices[val_split:]
            train_ds = Subset(train_full, train_idx)
            val_ds = Subset(datasets.CIFAR10(root=data_root, train=True, download=False, transform=test_tf), val_idx)
        else:
            train_ds = train_full
            val_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader

# -----------------------------
# Training & Evaluation
# -----------------------------

def accuracy(output, target):
    with torch.no_grad():
        pred = output.argmax(dim=1)
        return (pred == target).float().mean().item()

def train_one_epoch(model, loader, optimizer, scaler, device, label_smoothing=0.0):
    model.train(); total_loss=0.0; total_acc=0.0; n=0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(x)
            if label_smoothing > 0:
                n_classes = logits.size(1)
                with torch.no_grad():
                    true_dist = torch.zeros_like(logits)
                    true_dist.fill_(label_smoothing / (n_classes - 1))
                    true_dist.scatter_(1, y.data.unsqueeze(1), 1 - label_smoothing)
                loss = torch.mean(torch.sum(-true_dist * F.log_softmax(logits, dim=1), dim=1))
            else:
                loss = F.cross_entropy(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        total_loss += loss.item()*x.size(0)
        total_acc += accuracy(logits, y)*x.size(0)
        n += x.size(0)
    return total_loss/n, total_acc/n

def evaluate(model, loader, device):
    model.eval(); total_loss=0.0; total_acc=0.0; n=0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="val", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x); loss = F.cross_entropy(logits, y)
            total_loss += loss.item()*x.size(0)
            total_acc += accuracy(logits, y)*x.size(0)
            n += x.size(0)
    return total_loss/n, total_acc/n

# -----------------------------
# Inference on Kaggle test dir
# -----------------------------

class KaggleCIFAR10Test(Dataset):
    def __init__(self, test_dir: str, transform):
        self.paths = sorted(list(Path(test_dir).glob("*.png")), key=lambda p: int(p.stem))
        self.transform = transform
        assert len(self.paths) > 0, "No PNG files found in test_dir."
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), int(p.stem)

def predict_and_write_csv(model, test_dir: str, submission_path: str, batch_size: int, num_workers: int, device):
    _, test_tf = get_transforms()
    ds = KaggleCIFAR10Test(test_dir, test_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    model.eval(); ids=[]; labels=[]
    with torch.no_grad():
        for x, idxs in tqdm(loader, desc="predict"):
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            ids.extend(idxs.numpy().tolist())
            labels.extend([CIFAR10_CLASSES[p] for p in preds])
    with open(submission_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id","label"])
        for i, lab in zip(ids, labels): w.writerow([i, lab])
    print(f"Saved submission to {submission_path} with {len(ids)} rows.")

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--val-split", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="resnet18tiny")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--test-dir", type=str, default="./test")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/best.pt")
    parser.add_argument("--submission", type=str, default="./submission.csv")
    parser.add_argument("--use-kaggle-bundle", action="store_true")
    parser.add_argument("--kaggle-input-root", type=str, default="/kaggle/input/cifar-10")
    parser.add_argument("--auto-extract-test", action="store_true",
                        help="If set with --predict and --use-kaggle-bundle, auto-extract test.7z to /kaggle/working/test")
    args, _ = parser.parse_known_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Build model
    if args.model == "resnet18tiny":
        model = resnet18tiny(base=64, drop_prob=args.dropout)
    else:
        raise ValueError("Unknown model")
    model = model.to(device)

    if args.predict:
        if args.auto_extract_test and args.use_kaggle_bundle:
            test_7z = Path(args.kaggle_input_root) / "test.7z"
            out = Path("/kaggle/working/test")
            if not out.exists() or not any(out.glob("*.png")):
                out.mkdir(parents=True, exist_ok=True)
                os.system(f"7z x {test_7z} -o{out} -y > /dev/null")
        assert os.path.isfile(args.ckpt), f"Checkpoint not found: {args.ckpt}"
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model"])
        predict_and_write_csv(model, args.test_dir, args.submission, args.batch_size, args.num_workers, device)
        return

    # Data
    train_loader, val_loader = build_dataloaders(
        args.data_root, args.batch_size, args.num_workers, args.val_split,
        use_kaggle_bundle=args.use_kaggle_bundle, kaggle_input_root=args.kaggle_input_root
    )
    print(f"Data ready: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val")

    # Optim & sched
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_acc = 0.0
    ckpt_dir = Path("./checkpoints"); ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, args.label_smoothing)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "acc": best_acc, "epoch": epoch}, ckpt_dir / "best.pt")
            print(f"Saved new best with acc={best_acc:.4f}")

    print(f"Best val acc: {best_acc:.4f}. Checkpoint at {ckpt_dir/'best.pt'}")

if __name__ == "__main__":
    main()



import torch, subprocess, textwrap
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# remove any previous extraction
# !rm -rf /kaggle/working/test


# !mkdir -p /kaggle/working/test
# !7z x -mmt=8 -aos /kaggle/input/cifar-10/test.7z -o/kaggle/working/test -y
!find /kaggle/working/test/test -maxdepth 1 -name "*.png" | wc -l


!python -u cifar10_resnet_baseline.py \
  --use-kaggle-bundle --kaggle-input-root /kaggle/input/cifar-10 \
  --epochs 60 --batch-size 128 --val-split 5000 --num-workers 4



# confirm GPU & a few files exist
import torch, glob, os
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Some test files:", len(glob.glob("/kaggle/working/test/*.png")))
print("Nested test files:", len(glob.glob("/kaggle/working/test/test/*.png")))
print("CKPT exists:", os.path.isfile("/kaggle/working/checkpoints/best.pt"))




# run prediction (unbuffered so you see the tqdm bar)
!python -u cifar10_resnet_baseline.py --predict \
  --test-dir /kaggle/working/test/test \
  --ckpt /kaggle/working/checkpoints/best.pt \
  --submission /kaggle/working/submission.csv \
  --batch-size 512 --num-workers 4

!head -n 5 /kaggle/working/submission.csv

