# ===============================
# Cell 1 — Setup / Dataset / Utils (with light train-time aug)
# ===============================
import os, cv2, math, random, numpy as np, pandas as pd
from typing import List
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from PIL import Image

import timm

# TPU (PyTorch/XLA)
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

# -------------------------
# Config
# -------------------------
DATA_DIR   = "/kaggle/input/grand-xray-slam-division-a"
TRAIN_CSV  = f"{DATA_DIR}/train1.csv"
TRAIN_DIR  = f"{DATA_DIR}/train1"
TEST_DIR   = f"{DATA_DIR}/test1"
OUT_CSV    = "/kaggle/working/submission.csv"

LABEL_COLUMNS: List[str] = [
    'Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum',
    'Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion',
    'Pleural Other','Pneumonia','Pneumothorax','Support Devices'
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = (380, 380)  # EfficientNet-B4 input
TO_DROP = set(["00025979_008_001.jpg", "00048043_001_002.jpg"])

# -------------------------
# Transforms
# -------------------------
def get_train_transforms(img_size=(380, 380)):
    H, W = img_size
    return T.Compose([
        # 안전한 의료형 약증강
        T.RandomResizedCrop(size=(H, W), scale=(0.85, 1.0), ratio=(0.95, 1.05), interpolation=InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=5, interpolation=InterpolationMode.BILINEAR, fill=0),
        T.RandomAutocontrast(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_test_transforms(img_size=(380, 380)):
    H, W = img_size
    return T.Compose([
        T.Resize(size=(H, W), interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# -------------------------
# Dataset (now with train-time aug)
# -------------------------
class XRayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, img_size=(380, 380), is_train=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.img_size = img_size
        self.is_train = is_train
        self.tfms = get_train_transforms(img_size) if is_train else get_test_transforms(img_size)

    def __len__(self):
        return len(self.df)

    def _imread_rgb(self, path):
        # 원본 해상도 유지(증강이 크기/크롭/회전을 처리)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            # 비정상 파일은 검은 이미지로 대체
            H, W = self.img_size
            img = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["Image_name"])
        img = self._imread_rgb(img_path)

        # PIL로 변환 후 torchvision 증강 수행
        img = Image.fromarray(img)
        img = self.tfms(img)  # Tensor [C,H,W]

        if "split" in self.df.columns and not self.is_train:
            return row["Image_name"], img

        labels = torch.tensor(row[LABEL_COLUMNS].values.astype(np.float32))
        return img, labels

class TestDataset(Dataset):
    def __init__(self, image_dir: str, img_size=(380, 380)):
        self.image_dir = image_dir
        self.img_size = img_size
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")])
        self.tfms = get_test_transforms(img_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        p = os.path.join(self.image_dir, name)
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            H, W = self.img_size
            img = np.zeros((H, W, 3), dtype=np.uint8)
            img = Image.fromarray(img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        img = self.tfms(img)
        return name, img

# (Optional) FocalLoss kept (unused)
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = (1 - pt) ** self.gamma * bce
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

@torch.no_grad()
def compute_macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        return float("nan")

print("Cell 1 ready ✔ (with light train-time aug)")



# ===============================
# Cell 2 — Ensemble Inference (EMA B4 + EMA ViT), TTA hflip x2 (bf16→fp32 fix)
# ===============================
import os, contextlib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
import pandas as pd
import numpy as np

# TPU (PyTorch/XLA)
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from torch_xla.amp import autocast as xla_autocast

# ----- 체크포인트 경로 -----
B4_EMA_PATH  = "/kaggle/input/929591-pth/ema_b4_final.pth"
VIT_EMA_PATH = "/kaggle/input/vit-data/ema_vitb16_384.pth"
assert os.path.exists(B4_EMA_PATH),  f"Not found: {B4_EMA_PATH}"
assert os.path.exists(VIT_EMA_PATH), f"Not found: {VIT_EMA_PATH}"

# ----- Device / Loader -----
device = xm.xla_device()
batch_size  = 24
num_workers = 8

# 셀1의 TestDataset 재사용
test_ds = TestDataset(TEST_DIR, img_size=(384, 384))
test_loader_cpu = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
test_loader = pl.MpDeviceLoader(test_loader_cpu, device)

xm.master_print("Using TPU device:", device)

# ----- Model build & load -----
# EfficientNet-B4 (380)
model_b4 = timm.create_model(
    'tf_efficientnet_b4_ns', pretrained=False,
    num_classes=len(LABEL_COLUMNS), drop_rate=0.0
).to(device)
state_b4 = torch.load(B4_EMA_PATH, map_location='cpu')
model_b4.load_state_dict(state_b4, strict=False)
model_b4.eval()

# ViT-B/16 (384)
model_vit = timm.create_model(
    'vit_base_patch16_384', pretrained=False,
    num_classes=len(LABEL_COLUMNS)
).to(device)
state_vit = torch.load(VIT_EMA_PATH, map_location='cpu')
model_vit.load_state_dict(state_vit, strict=False)
model_vit.eval()

# ----- Helpers -----
def resize_batch(x, size_hw):
    if list(x.shape[-2:]) == list(size_hw):
        return x
    return F.interpolate(x, size=size_hw, mode='bilinear', align_corners=False)

@torch.no_grad()
def predict_probs_tta(model, imgs, in_size, use_amp=True):
    """
    Equal-weight TTA = original + hflip (2-pass). Returns float32 probs.
    """
    preds_accum = 0.0
    cnt = 0
    x1 = resize_batch(imgs, in_size)       # original
    x2 = torch.flip(x1, dims=[3])          # hflip

    for xx in (x1, x2):
        with xla_autocast(device=device, dtype=torch.bfloat16) if use_amp else contextlib.nullcontext():
            logits = model(xx)
            probs = torch.sigmoid(logits)
        preds_accum += probs
        cnt += 1

    # ✅ numpy로 보낼 때 bf16 이슈 방지: float32로 캐스팅
    return (preds_accum / cnt).to(torch.float32)

# ----- Inference: (B4 + ViT)/2, TTA flip x2 -----
submission_rows = []
use_amp = True

with torch.no_grad():
    pbar = tqdm(test_loader, desc="[Ensemble Infer] (B4 + ViT)/2, TTA hflip x2", unit="batch")
    for names, imgs in pbar:
        imgs = imgs.to(device, non_blocking=True)

        probs_b4  = predict_probs_tta(model_b4,  imgs, (380, 380), use_amp=use_amp)
        probs_vit = predict_probs_tta(model_vit, imgs, (384, 384), use_amp=use_amp)

        probs = 0.5 * (probs_b4 + probs_vit)                # equal-weight
        probs = probs.cpu().numpy()                          # now safe (fp32)

        for n, p in zip(names, probs):
            submission_rows.append([n] + p.tolist())
        xm.mark_step()

sub_df = pd.DataFrame(submission_rows, columns=["Image_name"] + LABEL_COLUMNS)
sub_df.to_csv(OUT_CSV, index=False)
xm.master_print(f"✅ Saved submission: {OUT_CSV}")


