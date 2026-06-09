%%writefile /kaggle/working/train.py
import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from torchvision.models import densenet121, DenseNet121_Weights

# ======================
# Dataset
# ======================
class ChestXRayDataset(Dataset):
    def __init__(self, df, img_size=(1048, 1048), is_test=False, transforms=None):
        self.df = df
        self.img_size = img_size
        self.is_test = is_test
        self.label_columns = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
            'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
            'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
        ]
        self.image_dir = '/kaggle/input/grand-xray-slam-division-a/train1/' if not is_test else '/kaggle/input/grand-xray-slam-division-a/test1/'
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['Image_name'])
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)

        img = cv2.resize(img, self.img_size)

        if self.transforms:
            img = self.transforms(img)

        if not self.is_test:
            labels = row[self.label_columns].values.astype(np.float32)
            return img, torch.tensor(labels)
        return img

# ======================
# Model
# ======================
def create_model(num_classes=14):
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    for param in model.features.denseblock4.parameters():
        param.requires_grad = True
    return model


# ======================
# Main DDP Training
# ======================
def main():
    dist.init_process_group(backend="nccl")  # NCCL = best for multi-GPU
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # --- Load data ---
    train_df = pd.read_csv('/kaggle/input/grand-xray-slam-division-a/train1.csv')
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df['No Finding']
    )

    img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ChestXRayDataset(train_data, img_size=(1048, 1048), transforms=img_transforms)
    val_dataset = ChestXRayDataset(val_data, img_size=(1048, 1048), transforms=img_transforms)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, sampler=val_sampler, num_workers=4)

    # --- Model, loss, optimizer ---
    model = create_model().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)

    num_epochs = 10

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        # ---- Train ----
        model.train()
        train_preds, train_labels = [], []
        running_train_loss = 0.0

        for images, labels in tqdm(train_loader, disable=(dist.get_rank() != 0)):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            train_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
            train_labels.append(labels.detach().cpu().numpy())

        # ---- Validation ----
        model.eval()
        val_preds, val_labels = [], []
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                val_preds.append(torch.sigmoid(outputs).cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        if dist.get_rank() == 0:  # Only rank 0 prints
            train_preds = np.vstack(train_preds)
            train_labels = np.vstack(train_labels)
            val_preds = np.vstack(val_preds)
            val_labels = np.vstack(val_labels)

            from sklearn.metrics import roc_auc_score
            train_auc = roc_auc_score(train_labels, train_preds, average='macro')
            val_auc = roc_auc_score(val_labels, val_preds, average='macro')

            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss {running_train_loss/len(train_dataset):.4f} | Train AUC {train_auc:.4f} | "
                  f"Val Loss {running_val_loss/len(val_dataset):.4f} | Val AUC {val_auc:.4f}")

    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), "best_model.pth")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()



# !torchrun --nproc_per_node=4 /kaggle/working/train.py


%%writefile /kaggle/working/inference.py
import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from torchvision.models import densenet121, DenseNet121_Weights

# ======================
# Dataset
# ======================
class ChestXRayTestDataset(Dataset):
    def __init__(self, df, img_size=(512, 512), transforms=None):
        self.df = df
        self.img_size = img_size
        self.image_dir = '/kaggle/input/grand-xray-slam-division-a/test1/'
        self.transforms = transforms
        self.label_columns = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
            'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
            'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['Image_name'])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transforms:
            img = self.transforms(img)

        return img, row['Image_name']

# ======================
# Model
# ======================
def create_model(num_classes=14):
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model

# ======================
# Inference
# ======================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    test_files = os.listdir('/kaggle/input/grand-xray-slam-division-a/test1/')
    test_df = pd.DataFrame({"Image_name": test_files})

    img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = ChestXRayTestDataset(test_df, img_size=(512, 512), transforms=img_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Load model
    model = create_model()
    model.load_state_dict(torch.load("/kaggle/working/best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    all_probs = []
    all_names = []

    with torch.no_grad():
        for images, names in tqdm(test_loader, desc="Inference", leave=True):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.append(probs)
            all_names.extend(names)

    all_probs = np.vstack(all_probs)
    submission = pd.DataFrame(all_probs, columns=test_dataset.label_columns)
    submission.insert(0, "Image_name", all_names)

    submission.to_csv("/kaggle/working/submission.csv", index=False)
    print("✅ Saved submission.csv")

if __name__ == "__main__":
    main()



# !python /kaggle/working/inference.py


%%writefile /kaggle/working/train.py
"""
Improved train.py (image-only) for Grand X-Ray Slam
- fixes warnings from logs (Albumentations, AMP, DDP grad bucket view)
- uses modern torch.amp API, gradient_as_bucket_view for DDP
- uses Affine (Albumentations) instead of ShiftScaleRotate
- optimizer.zero_grad(set_to_none=True) for better perf
- CosineAnnealingLR scheduler (per-epoch)
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # recommended for DDP per-process thread control

import gc
import random
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import timm
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score

# ---------- helpers ----------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def free_memory():
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass

# ---------- labels ----------
LABEL_COLS = [
    'Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum',
    'Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion',
    'Pleural Other','Pneumonia','Pneumothorax','Support Devices'
]

# ---------- dataset ----------
class ChestXRayImageDataset(Dataset):
    def __init__(self, df, image_dir, img_size=1024, transforms=None, image_col='Image_Name', label_cols=LABEL_COLS):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        self.img_size = img_size
        self.image_col = image_col
        self.label_cols = label_cols

    def __len__(self):
        return len(self.df)

    def _read_img(self, fname):
        p = self.image_dir / fname
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row[self.image_col]
        image = self._read_img(fname)
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        labels = row[self.label_cols].values.astype(np.float32)
        return image, torch.tensor(labels, dtype=torch.float32)

# ---------- loss ----------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)
        focal = (1 - p_t) ** self.gamma * bce
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal

def hybrid_loss(logits, targets, pos_weight_tensor, alpha_bce=0.6, alpha_focal=0.4, focal_gamma=2.0):
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)(logits, targets)
    focal = FocalLoss(gamma=focal_gamma)(logits, targets)
    return alpha_bce * bce + alpha_focal * focal

# ---------- train/val ----------
def train_one_epoch(model, loader, optimizer, scaler, device, pos_weight_tensor, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc='Train', leave=False)
    for step, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type):
            logits = model(images)
            loss = hybrid_loss(logits, targets, pos_weight_tensor)

        scaler.scale(loss / accumulation_steps).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(loss.item()) * images.size(0)
        all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

        pbar.set_postfix({'loss': running_loss / ((step + 1) * loader.batch_size)})

    epoch_loss = running_loss / len(loader.dataset)
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    return epoch_loss, preds, targets

def valid_one_epoch(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc='Valid', leave=False)
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    return preds, targets

def compute_per_class_auc(truths, preds, label_cols):
    per_class = []
    for i, name in enumerate(label_cols):
        try:
            a = roc_auc_score(truths[:, i], preds[:, i])
        except Exception:
            a = float('nan')
        per_class.append(a)
    macro = float(np.nanmean(per_class))
    return macro, per_class

# ---------- main ----------
def main():
    parser = ArgumentParser()
    parser.add_argument('--data-csv', type=str, default='/kaggle/input/grand-xray-slam-division-a/train1.csv')
    parser.add_argument('--image-dir', type=str, default='/kaggle/input/grand-xray-slam-division-a/train1/')
    parser.add_argument('--out-dir', type=str, default='/kaggle/working/')
    parser.add_argument('--img-size', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--use-ddp', action='store_true')
    parser.add_argument('--backbone', type=str, default='convnext_large')
    args = parser.parse_args()

    seed_everything(42)
    os.makedirs(args.out_dir, exist_ok=True)

    # performance flags
    torch.backends.cudnn.benchmark = True

    # DDP init
    use_ddp = args.use_ddp
    if use_ddp:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    # read csv
    df = pd.read_csv(args.data_csv)
    # detect image column
    image_col_candidates = ['Image_Name','Image_name','Image']
    image_col = None
    for c in image_col_candidates:
        if c in df.columns:
            image_col = c
            break
    if image_col is None:
        for c in df.columns:
            if c.lower() == 'image_name' or c.lower() == 'image':
                image_col = c
                break
    if image_col is None:
        raise ValueError("Couldn't find image column in CSV. Expected 'Image_Name' or 'Image_name' or 'Image'.")

    # ensure labels exist (case-insensitive)
    label_cols = []
    for lc in LABEL_COLS:
        if lc in df.columns:
            label_cols.append(lc)
        else:
            matched = [c for c in df.columns if c.lower() == lc.lower()]
            if matched:
                label_cols.append(matched[0])
            else:
                raise ValueError(f"Label column {lc} not found in CSV. Available: {df.columns.tolist()}")

    # rename image col
    df = df.rename(columns={image_col: 'Image_Name'})

    # compute pos_weight per class
    total = len(df)
    label_sums = df[label_cols].sum(axis=0).values.astype(np.float32)
    negs = total - label_sums
    pos_weight = (negs / (label_sums + 1e-6)).astype(np.float32)
    pos_weight_tensor = torch.from_numpy(pos_weight).to(device)

    # split: grouped by patient if available
    if 'Patient_ID' in df.columns:
        groups = df['Patient_ID'].values
        gkf = GroupKFold(n_splits=args.n_folds)
        splits = list(gkf.split(df, df[label_cols], groups))
        train_idx, val_idx = splits[args.fold]
    else:
        stratify_col = df[label_cols[0]] if label_cols[0] in df.columns else None
        train_idx, val_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=stratify_col)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # transforms (Affine used instead of ShiftScaleRotate)
    train_transforms = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.92, 1.08), rotate=(-10, 10), shear=0, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.CLAHE(p=0.3),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])
    val_transforms = A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])

    # datasets & loaders
    train_ds = ChestXRayImageDataset(train_df, args.image_dir, img_size=args.img_size, transforms=train_transforms, image_col='Image_Name', label_cols=label_cols)
    val_ds = ChestXRayImageDataset(val_df, args.image_dir, img_size=args.img_size, transforms=val_transforms, image_col='Image_Name', label_cols=label_cols)

    if use_ddp:
        train_sampler = DistributedSampler(train_ds)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                            shuffle=False, num_workers=args.workers, pin_memory=True)

    # model init + DDP with gradient_as_bucket_view to avoid grad/bucket warnings
    model = timm.create_model(args.backbone, pretrained=True, num_classes=len(label_cols))
    model.to(device)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True
        )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    # scaler: use new API, specify device type when CUDA
    try:
        if device.type == 'cuda':
            scaler = torch.amp.GradScaler(device_type='cuda')
        else:
            scaler = torch.amp.GradScaler()
    except TypeError:
        # fallback for older torch versions
        scaler = torch.amp.GradScaler()

    best_auc = 0.0

    for epoch in range(args.epochs):
        if use_ddp:
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_preds, train_targets = train_one_epoch(
            model, train_loader, optimizer, scaler, device, pos_weight_tensor, accumulation_steps=args.accumulation
        )

        val_preds, val_targets = valid_one_epoch(model, val_loader, device)

        macro_auc, per_class = compute_per_class_auc(val_targets, val_preds, label_cols)

        if local_rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | TrainLoss: {train_loss:.4f} | Val AUC (macro): {macro_auc:.4f}")
            for ln, a in zip(label_cols, per_class):
                print(f"  {ln}: {a:.4f}")
            if macro_auc > best_auc:
                best_auc = macro_auc
                ckpt_path = os.path.join(args.out_dir, f"best_{args.backbone}_fold{args.fold}_img{args.img_size}.pth")
                state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save({'state_dict': state, 'auc': best_auc}, ckpt_path)
                print("Saved checkpoint:", ckpt_path)

        # scheduler step per epoch
        try:
            scheduler.step()
        except Exception:
            pass

        free_memory()

    if use_ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()



# # debug / quick: smaller image, fewer epochs
# python /kaggle/working/train.py --data-csv /kaggle/input/grand-xray-slam-division-a/train1.csv \
#   --image-dir /kaggle/input/grand-xray-slam-division-a/train1/ \
#   --img-size 512 --batch-size 16 --epochs 2 --workers 4 --backbone convnext_small



!torchrun --nproc_per_node=4 /kaggle/working/train.py --use-ddp --img-size 512 --batch-size 16 --epochs 10 --workers 8 --backbone convnext_large


%%writefile /kaggle/working/inference.py
"""
Inference (image-only) with optional TTA and checkpoint ensembling.
Usage examples:
  python inference.py --image-dir /kaggle/input/grand-xray-slam-division-a/test1/ --ckpt "/kaggle/working/*best*.pth" --img-size 1024 --batch-size 8 --tta-scales "1024,800" --tta-flip
"""

import os
import glob
import argparse
from pathlib import Path
from tqdm.auto import tqdm

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

LABEL_COLS = [
    'Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum',
    'Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion',
    'Pleural Other','Pneumonia','Pneumothorax','Support Devices'
]

class TestImageDataset(Dataset):
    def __init__(self, image_dir, files, transforms=None, img_size=1024):
        self.image_dir = Path(image_dir)
        self.files = files
        self.transforms = transforms
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def _read(self, fname):
        p = self.image_dir / fname
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            return img
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = self._read(fname)
        if self.transforms:
            img = self.transforms(image=img)['image']
        return img, fname

def load_state_stripped(path, map_location='cpu'):
    ck = torch.load(path, map_location=map_location)
    if isinstance(ck, dict) and 'state_dict' in ck:
        state = ck['state_dict']
    elif isinstance(ck, dict) and 'model' in ck:
        state = ck['model']
    elif isinstance(ck, dict):
        state = ck
    else:
        state = ck
    new_state = {}
    for k, v in state.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state[new_key] = v
    return new_state

@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    sigmoid = nn.Sigmoid()
    probs_list = []
    names = []
    for imgs, fnames in tqdm(loader, desc="Inf", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = sigmoid(logits).cpu().numpy()
        probs_list.append(probs)
        names.extend(fnames)
    probs = np.vstack(probs_list)
    return probs, names

def build_transform(img_size, hflip=False):
    tr = [A.Resize(img_size, img_size)]
    if hflip:
        tr.append(A.HorizontalFlip(p=1.0))
    tr += [
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ]
    return A.Compose(tr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None, help="glob or single path")
    parser.add_argument('--backbone', type=str, default='convnext_large')
    parser.add_argument('--img-size', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--tta-scales', type=str, default=None, help="e.g. '1024,800'")
    parser.add_argument('--tta-flip', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--out', type=str, default='/kaggle/working/submission.csv')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # gather checkpoints
    if args.ckpt:
        ckpts = sorted(glob.glob(args.ckpt))
    else:
        ckpts = sorted(glob.glob('/kaggle/working/*best*.pth')) + sorted(glob.glob('/kaggle/working/*.pth'))
    if len(ckpts) == 0:
        raise FileNotFoundError("No checkpoints found. Provide --ckpt pattern or place pth files in /kaggle/working/")

    files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    if len(files) == 0:
        raise FileNotFoundError(f"No images found in {args.image_dir}")

    if args.tta_scales:
        scales = [int(x.strip()) for x in args.tta_scales.split(',') if x.strip()]
    else:
        scales = [args.img_size]

    ensemble_preds = None
    for ckpt in ckpts:
        print("Loading checkpoint:", ckpt)
        # load model
        model = timm.create_model(args.backbone, pretrained=False, num_classes=len(LABEL_COLS))
        sd = load_state_stripped(ckpt, map_location='cpu')
        model.load_state_dict(sd, strict=False)
        model.to(device)
        model.eval()

        model_accum = None
        tta_count = 0

        for scale in scales:
            # normal
            tr = build_transform(scale, hflip=False)
            ds = TestImageDataset(args.image_dir, files, transforms=tr, img_size=scale)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            probs, names = run_inference(model, loader, device)
            if model_accum is None:
                model_accum = probs.copy()
            else:
                model_accum += probs
            tta_count += 1

            # flip
            if args.tta_flip:
                trf = build_transform(scale, hflip=True)
                dsf = TestImageDataset(args.image_dir, files, transforms=trf, img_size=scale)
                loaderf = DataLoader(dsf, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
                probs_f, _ = run_inference(model, loaderf, device)
                model_accum += probs_f
                tta_count += 1

        model_accum = model_accum / float(tta_count)
        if ensemble_preds is None:
            ensemble_preds = model_accum.copy()
        else:
            ensemble_preds += model_accum

        # free GPU
        del model, sd
        torch.cuda.empty_cache()

    ensemble_preds = ensemble_preds / float(len(ckpts))
    out_df = pd.DataFrame(ensemble_preds, columns=LABEL_COLS)
    out_df.insert(0, 'Image_Name', names)
    out_df.to_csv(args.out, index=False)
    print("Saved submission:", args.out)

if __name__ == '__main__':
    main()



!python /kaggle/working/inference.py \
  --image-dir /kaggle/input/grand-xray-slam-division-a/test1/ \
  --ckpt "/kaggle/working/*best*.pth" \
  --backbone convnext_large --img-size 1024 --batch-size 64 --tta-scales "1024,800" --tta-flip \
  --out /kaggle/working/submission.csv

