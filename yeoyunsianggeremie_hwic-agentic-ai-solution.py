import os
import psutil  # For CPU affinity

# CPU affinity (pin to specific cores to prevent resource overlap)
# psutil.Process(os.getpid()).cpu_affinity([30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44])
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# BASE_DIR = "task/whale-categorization-playground" if not os.getenv('KAGGLE_KERNEL_RUN_TYPE') else "/kaggle/input/whale-categorization-playground"

# coding: utf-8
# Whale Identification (whale-categorization-playground)
# Single-file pipeline using ONLY ConvNeXt V2 Tiny (timm).
# v5 changes: Closed-set training (exclude 'new_whale' from classes/loss), widened τ search for gating, and per-iteration LR scheduler stepping.

import os
import math
import time
import json
import random
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

# -----------------------------
# Paths, logging, and config
# -----------------------------
BASE_DIR = Path("/kaggle/input/whale-categorization-playground") # Path("task/whale-categorization-playground")
OUT_DIR = Path(".") # BASE_DIR / "outputs" / "7_3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT_DIR / "code_7_3_v5.txt"   # per instruction
SUB_PATH = OUT_DIR / "submission_5.csv"  # per instruction

# Place logging.basicConfig at the start
logging.basicConfig(
    filename=str(LOG_FILE),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logging.getLogger("").addHandler(console)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    logging.info("HF_TOKEN detected; public timm weights typically do not require auth. Proceeding.")

# -----------------------------
# Determinism and environment
# -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def env_summary():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_cpu = os.cpu_count() or 1
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        logging.info(f"CUDA available: True | GPU: {props.name} | VRAM: {props.total_memory/(1024**3):.1f} GB")
    else:
        logging.info("CUDA available: False. Running on CPU will be slow.")
    logging.info(f"Detected CPU cores: {n_cpu}")
    return device, n_cpu

# -----------------------------
# I/O and hashing
# -----------------------------
def md5_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def read_csvs():
    train_csv = BASE_DIR / "train.csv"
    sample_sub = BASE_DIR / "sample_submission.csv"
    assert train_csv.exists(), f"Missing {train_csv}"
    assert sample_sub.exists(), f"Missing {sample_sub}"
    df = pd.read_csv(train_csv)
    ss = pd.read_csv(sample_sub)
    return df, ss

# -----------------------------
# Image transforms
# -----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def letterbox_resize(img: Image.Image, size: int = 384, pad_color: int = 114) -> Image.Image:
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = img.resize((nw, nh), resample=Image.BICUBIC)
    new_img = Image.new("RGB", (size, size), (pad_color, pad_color, pad_color))
    top = (size - nh) // 2
    left = (size - nw) // 2
    new_img.paste(img_resized, (left, top))
    return new_img

class TwoCropTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform
    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)

def build_transforms(img_size: int = 384):
    train_aug = T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.Lambda(lambda im: letterbox_resize(im, size=img_size)),
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10, interpolation=T.InterpolationMode.BICUBIC, fill=tuple([114]*3)),
        T.RandomPerspective(distortion_scale=0.1, p=0.15, interpolation=T.InterpolationMode.BICUBIC, fill=tuple([114]*3)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value="random"),
    ])
    train_transform = TwoCropTransform(train_aug)

    valid_transform = T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.Lambda(lambda im: letterbox_resize(im, size=img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, valid_transform

# -----------------------------
# Dataset
# -----------------------------
class WhaleDataset(Dataset):
    # df expects columns: Image, label_idx (for training/validation)
    def __init__(self, df: pd.DataFrame, root: Path, transform, return_label=True, two_crop=True):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.return_label = return_label
        self.two_crop = two_crop
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row["Image"]
        with Image.open(img_path) as im:
            im = im.convert("RGB") if im.mode != "RGB" else im
            if self.two_crop:
                im1, im2 = self.transform(im)
                if self.return_label:
                    return (im1, im2), int(row["label_idx"])
                else:
                    return (im1, im2), row["Image"]
            else:
                im1 = self.transform(im)
                if self.return_label:
                    return im1, int(row["label_idx"])        # no None in batch
                else:
                    return im1, row["Image"]

# -----------------------------
# Deduplication and split (closed-set)
# -----------------------------
def compute_md5_for_df(df: pd.DataFrame, img_dir: Path) -> pd.DataFrame:
    md5s = []
    for img in df["Image"].tolist():
        md5s.append(md5_of_file(img_dir / img))
    out = df.copy()
    out["md5"] = md5s
    return out

def deduplicate_by_md5(train_df: pd.DataFrame, train_dir: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    logging.info("MD5 deduplication started (train set).")
    t0 = time.time()
    df = compute_md5_for_df(train_df, train_dir)
    keep_mask = np.ones(len(df), dtype=bool)
    for md5, idxs in df.groupby("md5").indices.items():
        ids = df.loc[list(idxs), "Id"].tolist()
        uniq = list(set(ids))
        if len(uniq) == 1:
            continue
        non_new = [x for x in uniq if x != "new_whale"]
        if len(non_new) == 1:
            cand_idx = df[(df["md5"] == md5) & (df["Id"] == non_new[0])].index[0]
            for j in idxs:
                if j != cand_idx:
                    keep_mask[j] = False
        else:
            for j in idxs:
                keep_mask[j] = False
    df_dedup = df[keep_mask].drop_duplicates(subset=["md5"], keep="first").reset_index(drop=True)
    md5_to_id = dict(zip(df_dedup["md5"], df_dedup["Id"]))
    logging.info(f"Deduped train: {len(train_df)} -> {len(df_dedup)} in {time.time()-t0:.1f}s")
    return df_dedup, md5_to_id

def closed_set_split_fold0(df: pd.DataFrame, seed: int = 42, new_whale_valid_frac: float = 0.20):
    # Keep all non-'new_whale' IDs in training; hold out 1 image per ID for validation.
    val_rows, train_rows = [], []
    for gid, g in df.groupby("Id"):
        if gid == "new_whale":
            continue
        if len(g) >= 2:
            sel = g.sample(1, random_state=seed)
            rem = g.drop(sel.index)
            val_rows.append(sel)
            train_rows.append(rem)
        else:
            train_rows.append(g)
    train_df = pd.concat(train_rows, axis=0) if len(train_rows) else df.iloc[0:0]
    val_df = pd.concat(val_rows, axis=0) if len(val_rows) else df.iloc[0:0]
    # Add a portion of new_whale to validation for threshold tuning
    nw_df = df[df["Id"] == "new_whale"].copy()
    if len(nw_df):
        val_nw = nw_df.sample(max(1, int(len(nw_df) * new_whale_valid_frac)), random_state=seed)
        val_df = pd.concat([val_df, val_nw], axis=0)
    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    logging.info(f"Closed-set Split fold-0: train={len(train_df)}, valid={len(val_df)} (kept all labeled IDs)")
    return train_df, val_df

# -----------------------------
# Labels (exclude 'new_whale' from training)
# -----------------------------
def fit_label_encoder_no_newwhale(train_ids: List[str]) -> Tuple[Dict[str, int], List[str]]:
    uniq = sorted(set([u for u in train_ids if u != "new_whale"]))
    id2idx = {k: i for i, k in enumerate(uniq)}
    idx2id = uniq[:]  # no 'new_whale' inside the classifier head
    return id2idx, idx2id

# -----------------------------
# GeM pooling
# -----------------------------
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, clamp=True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.clamp = clamp
    def forward(self, x):
        if self.clamp:
            x = x.clamp(min=self.eps)
        x = x.pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.pow(1.0 / self.p)
        return x

# -----------------------------
# Sub-center ArcFace (AMP-safe)
# -----------------------------
class SubCenterArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, K: int = 2, s: float = 30.0, m: float = 0.30, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features * K, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

    def forward(self, features, labels):
        W = F.normalize(self.weight, dim=1)
        logits_all = F.linear(features, W)  # (N, C*K)
        N = logits_all.size(0)
        C = self.out_features
        K = self.K
        logits_ck = logits_all.view(N, C, K)
        logits = logits_ck.max(dim=2).values  # (N, C)

        idx = torch.arange(N, device=features.device, dtype=torch.long)
        labels = labels.long()
        cos_theta = logits[idx, labels].clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        one = torch.ones_like(cos_theta)
        sin_theta = torch.sqrt((one - cos_theta * cos_theta).clamp(min=0.0))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = cos_theta_m.to(logits.dtype)  # ensure dtype match for index_put

        output = logits.clone()
        output[idx, labels] = cos_theta_m
        output = output * self.s
        return output

# -----------------------------
# SupCon Loss
# -----------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        # features: (N,2,D)
        device = features.device
        N = features.shape[0]
        features = F.normalize(features, dim=-1)
        f1, f2 = features[:, 0], features[:, 1]
        feats = torch.cat([f1, f2], dim=0)  # (2N,D)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # (N,N)
        mask = torch.cat([torch.cat([mask, mask], dim=1),
                          torch.cat([mask, mask], dim=1)], dim=0)  # (2N,2N)
        anchor_dot_contrast = torch.matmul(feats, feats.T) / self.temperature
        logits_mask = torch.ones_like(anchor_dot_contrast) - torch.eye(2*N, device=device)
        anchor_dot_contrast = anchor_dot_contrast * logits_mask
        exp_logits = torch.exp(anchor_dot_contrast)
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1).clamp_min(1.0))
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()

# -----------------------------
# Model wrapper
# -----------------------------
class WhaleModel(nn.Module):
    def __init__(self, backbone_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384", embed_dim=512, num_classes=1000, drop_path=0.15, head_dropout=0.10, subcenters=2):
        super().__init__()
        # ConvNeXt V2 Tiny ONLY
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, drop_path_rate=drop_path)
        self.gem = GeM(p=3.0, eps=1e-6)
        in_ch = getattr(self.backbone, "num_features", 768)
        self.embed = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.Dropout(p=head_dropout),
        )
        self.arc = SubCenterArcMarginProduct(embed_dim, num_classes, K=subcenters, s=30.0, m=0.30)
    def forward(self, x, labels=None):
        feats = self.backbone.forward_features(x)
        feats = self.embed(feats)
        pooled = self.gem(feats)
        emb = self.fc(pooled)
        emb = F.normalize(emb, dim=1)
        if labels is not None:
            logits = self.arc(emb, labels)
            return emb, logits
        else:
            return emb

# -----------------------------
# Optimizer and per-iteration scheduler
# -----------------------------
def create_optimizer(model: WhaleModel, lr_backbone=2e-4, lr_head=1e-3, weight_decay=0.05):
    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone"):
            backbone_params.append(p)
        else:
            head_params.append(p)
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ], weight_decay=weight_decay)
    return opt

# Per-iteration warmup+cosine with static base_lrs; step once per optimizer step.
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.min_lr = float(min_lr)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]  # snapshot base LRs
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        step = self.last_epoch + 1  # number of scheduler.step() calls
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                lr = base_lr * step / self.warmup_steps
            else:
                t = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + math.cos(math.pi * t))
            lrs.append(lr)
        return lrs

# -----------------------------
# Train/Validate utilities
# -----------------------------
def map_per_image(actual: str, predicted: List[str]) -> float:
    if actual in predicted[:5]:
        return 1.0 / (predicted.index(actual) + 1)
    return 0.0

def mapk(actuals: List[str], preds: List[List[str]], k: int = 5) -> float:
    return float(np.mean([map_per_image(a, p[:k]) for a, p in zip(actuals, preds)]))

def train_one_epoch(model, loader, optimizer, scheduler, scaler, supcon_loss_fn, ce_loss_fn, device, grad_clip=1.0):
    # Per-iteration LR schedule stepping
    model.train()
    total_loss, n_batches = 0.0, 0
    for (x1, x2), y in loader:
        x1 = x1.to(device).to(memory_format=torch.channels_last)
        x2 = x2.to(device).to(memory_format=torch.channels_last)
        y = y.to(device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            emb1, logits = model(x1, labels=y)
            loss_ce = ce_loss_fn(logits, y)
            emb2 = model(x2)
            feats = torch.stack([emb1, emb2], dim=1)
            loss_sc = supcon_loss_fn(feats, y)
            loss = loss_ce + 0.1 * loss_sc
        if torch.cuda.is_available():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()
        total_loss += float(loss.detach().cpu())
        n_batches += 1
    return total_loss / max(1, n_batches)

def unpack_batch_xy(batch):
    # Support both ((x1, x2), y) and (x1, y)
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        a, y = batch
        if isinstance(a, (list, tuple)) and len(a) == 2 and torch.is_tensor(a[0]):
            return a[0], y
        elif torch.is_tensor(a):
            return a, y
    raise ValueError("Unexpected batch structure for (x, y).")

@torch.no_grad()
def validate_classification(model, loader, ce_loss_fn, device):
    model.eval()
    total_loss, n_samples = 0.0, 0
    for batch in loader:
        x1, y = unpack_batch_xy(batch)
        x1 = x1.to(device).to(memory_format=torch.channels_last)
        y = y.to(device, dtype=torch.long)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            emb, logits = model(x1, labels=y)
            loss = ce_loss_fn(logits, y)
        total_loss += float(loss.detach().cpu()) * x1.size(0)
        n_samples += x1.size(0)
    return total_loss / max(1, n_samples)

@torch.no_grad()
def extract_embeddings_with_labels(model, loader, device):
    model.eval()
    embs_list, labels_list = [], []
    for batch in loader:
        x1, y = unpack_batch_xy(batch)
        x1 = x1.to(device).to(memory_format=torch.channels_last)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            emb = model(x1)
        embs_list.append(emb.detach().cpu())
        labels_list.extend(y.detach().cpu().numpy().tolist())
    embs = torch.cat(embs_list, dim=0).numpy()
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
    return embs, labels_list

@torch.no_grad()
def extract_embeddings_with_names(model, df_src: pd.DataFrame, root: Path, transform, device, bs=16, nw=4):
    class EDS(Dataset):
        def __init__(self, df, root, tf):
            self.df = df.reset_index(drop=True)
            self.root = root; self.tf = tf
        def __len__(self): return len(self.df)
        def __getitem__(self, i):
            r = self.df.iloc[i]
            with Image.open(self.root / r["Image"]) as im:
                im = im.convert("RGB") if im.mode != "RGB" else im
                x = self.tf(im)
            return x, r["Image"]
    dl = DataLoader(EDS(df_src, root, transform), batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    model.eval()
    embs, names = [], []
    for x, ns in dl:
        x = x.to(device).to(memory_format=torch.channels_last)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            e = model(x)
        embs.append(e.detach().cpu())
        names.extend(list(ns))
    embs = torch.cat(embs, 0).numpy()
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
    return embs, names

@torch.no_grad()
def compute_prototypes(embeddings: np.ndarray, labels_idx: np.ndarray, idx2id: List[str]) -> Dict[str, np.ndarray]:
    id_to_vecs = defaultdict(list)
    for emb, idx in zip(embeddings, labels_idx):
        lbl = idx2id[int(idx)]
        id_to_vecs[lbl].append(emb)
    prototypes = {}
    for k, v in id_to_vecs.items():
        arr = np.stack(v, axis=0)
        proto = arr.mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-10)
        prototypes[k] = proto.astype(np.float32)
    return prototypes

def cosine_rank(prototypes: Dict[str, np.ndarray], query_embs: np.ndarray, topk=5):
    ids = list(prototypes.keys())
    proto_mat = np.stack([prototypes[i] for i in ids], axis=1)  # (D, Nids)
    sims = query_embs @ proto_mat  # (Q, Nids)
    topk_idx = np.argsort(-sims, axis=1)[:, :topk]
    topk_ids = [[ids[j] for j in row] for row in topk_idx]
    max_sim = sims.max(axis=1)
    return topk_ids, max_sim

def tune_new_whale_threshold(valid_embs, valid_labels_true: List[str], prototypes: Dict[str, np.ndarray]):
    # Wider τ search for closed-set training with open-set gating
    topk_ids, max_sim = cosine_rank(prototypes, valid_embs, topk=5)
    best_tau, best_map = 0.0, -1.0
    for tau in np.linspace(0.20, 0.95, 38):
        preds = []
        for ids_row, s in zip(topk_ids, max_sim):
            if s < tau:
                out = ["new_whale"] + ids_row[:4]
            else:
                out = ids_row[:5]
                if "new_whale" not in out:
                    out[-1] = "new_whale"
            preds.append(out)
        score = mapk(valid_labels_true, preds, k=5)
        if score > best_map:
            best_map, best_tau = score, float(tau)
    logging.info(f"Tuned threshold tau={best_tau:.3f}, MAP@5={best_map:.4f}")
    return best_tau, best_map

# -----------------------------
# Main pipeline
# -----------------------------
def run_pipeline(DEBUG: bool):
    seed_everything(42)
    mode = "DEBUG" if DEBUG else "FULL"
    logging.info(f"=== Running pipeline in {mode} mode (closed-set training) ===")
    device, n_cpu = env_summary()
    torch.backends.cudnn.benchmark = torch.cuda.is_available()

    train_dir = BASE_DIR / "train"
    test_dir = BASE_DIR / "test"

    # Load CSVs
    df_train_raw, df_sample = read_csvs()

    # Deduplicate train by MD5
    df_train_dedup, md5_to_id_train = deduplicate_by_md5(df_train_raw, train_dir)

    # Closed-set fold-0 split
    df_tr, df_val = closed_set_split_fold0(df_train_dedup, seed=42)

    # Remove 'new_whale' from training supervision
    df_tr = df_tr[df_tr["Id"] != "new_whale"].reset_index(drop=True)

    # Label encoding WITHOUT 'new_whale'
    id2idx, idx2id = fit_label_encoder_no_newwhale(df_tr["Id"].tolist())
    df_tr = df_tr.copy()
    df_tr["label_idx"] = df_tr["Id"].map(id2idx).astype(int)

    # Known-only validation for CE
    df_val_known = df_val[df_val["Id"] != "new_whale"].copy()
    if len(df_val_known):
        df_val_known["label_idx"] = df_val_known["Id"].map(id2idx).astype(int)

    # DEBUG sampling
    if DEBUG:
        one_per_class = df_tr.groupby("label_idx").head(1)
        if len(one_per_class) > 0.5 * len(df_tr):
            logging.warning("DEBUG set would exceed 50% of train; skipping DEBUG and returning.")
            return
        target = max(1000, len(one_per_class))
        rem = df_tr.drop(one_per_class.index)
        need = max(0, target - len(one_per_class))
        sampled = rem.sample(min(len(rem), need), random_state=42)
        df_tr = pd.concat([one_per_class, sampled], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
        epochs = 1
        logging.info(f"DEBUG training rows: {len(df_tr)}")
    else:
        epochs = 24

    # Transforms and datasets
    IMG_SIZE = 384
    tr_tf, val_tf = build_transforms(img_size=IMG_SIZE)
    ds_tr = WhaleDataset(df_tr, train_dir, transform=tr_tf, return_label=True, two_crop=True)        # ((x1,x2), y)
    ds_val_ce = WhaleDataset(df_val_known, train_dir, transform=val_tf, return_label=True, two_crop=False) if len(df_val_known) else None

    # Loaders
    batch_size = 16
    num_workers = min(8, max(1, (os.cpu_count() or 2) - 1))
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    dl_val_ce = DataLoader(ds_val_ce, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True) if ds_val_ce is not None else None

    logging.info(f"Train size: {len(df_tr)} | Valid size (full, incl new_whale): {len(df_val)} | Classes (no 'new_whale'): {len(idx2id)}")

    # Model
    model = WhaleModel(
        backbone_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384",
        embed_dim=512,
        num_classes=len(idx2id),
        drop_path=0.15,
        head_dropout=0.10,
        subcenters=2,
    )
    model = model.to(device).to(memory_format=torch.channels_last)
    logging.info("Loaded ConvNeXt V2 Tiny backbone with pretrained weights and custom GeM+ArcFace+SupCon heads (closed-set).")

    # Optimizer and per-iteration scheduler
    optimizer = create_optimizer(model, lr_backbone=2e-4, lr_head=1e-3, weight_decay=0.05)
    steps_per_epoch = max(1, len(dl_tr))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(0.15 * total_steps))
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    supcon_loss_fn = SupConLoss(temperature=0.07, base_temperature=0.07)
    ce_loss_fn = nn.CrossEntropyLoss()

    # Train with early stopping
    best_val, best_epoch, bad_epochs, patience = float("inf"), -1, 0, 3
    best_state = None
    t0 = time.time()
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, dl_tr, optimizer, scheduler, scaler, supcon_loss_fn, ce_loss_fn, device)
        if dl_val_ce is not None and len(df_val_known):
            val_loss = validate_classification(model, dl_val_ce, ce_loss_fn, device)
        else:
            val_loss = float("nan")
        logging.info(f"[{mode}] Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | val_loss(known-only)={val_loss:.4f}")

        if (not DEBUG) and epoch == 0 and (math.isnan(train_loss) or (not math.isnan(val_loss) and math.isnan(val_loss))):
            logging.warning("NaN detected after epoch 1 in FULL mode; aborting training and proceeding to inference.")
            break

        # Early stopping only if we have a numeric val_loss
        if not math.isnan(val_loss):
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}. Best epoch={best_epoch}, best val_loss={best_val:.4f}")
                break

    elapsed = (time.time() - t0) / 60.0
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=True)
    logging.info(f"[{mode}] Training complete. Best epoch={best_epoch} | best val_loss={best_val:.4f} | time={elapsed:.1f} min")

    # Retrieval validation and threshold tuning on FULL validation (incl 'new_whale')
    dl_tr_embed = DataLoader(WhaleDataset(df_tr, train_dir, transform=val_tf, return_label=True, two_crop=False),
                             batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    embs_tr, y_tr_idx = extract_embeddings_with_labels(model, dl_tr_embed, device)
    prototypes = compute_prototypes(embs_tr, np.array(y_tr_idx), idx2id)
    logging.info(f"Built {len(prototypes)} prototypes (closed-set, no 'new_whale').")

    embs_val, val_names = extract_embeddings_with_names(model, df_val, train_dir, val_tf, device, bs=batch_size, nw=num_workers)
    name2id = dict(zip(df_val["Image"], df_val["Id"]))
    val_true_ids = [name2id[nm] for nm in val_names]

    tau, map5_val = tune_new_whale_threshold(embs_val, val_true_ids, prototypes)
    logging.info(f"[{mode}] Fold-0 retrieval MAP@5={map5_val:.4f} (tau={tau:.3f})")

    # Save best model
    model_path = OUT_DIR / f"convnextv2_tiny_fold0_best_v5.pth"
    torch.save(model.state_dict(), model_path)
    logging.info(f"Saved model to {model_path}")

    # DEBUG: do not create submission
    if DEBUG:
        logging.info("DEBUG mode: submission skipped per requirements.")
        return

    # Inference on test
    test_imgs = sorted([p.name for p in (BASE_DIR / "test").glob("*.jpg")])
    logging.info(f"Test images: {len(test_imgs)}")

    logging.info("Computing test MD5s for exact-match shortcut...")
    test_dir = BASE_DIR / "test"
    test_md5 = {img: md5_of_file(test_dir / img) for img in test_imgs}
    md5_hits = sum(1 for img in test_imgs if test_md5[img] in md5_to_id_train)
    logging.info(f"MD5 exact matches in test: {md5_hits}/{len(test_imgs)}")

    class TestDataset(Dataset):
        def __init__(self, images: List[str], root: Path, transform):
            self.images = images
            self.root = root
            self.transform = transform
        def __len__(self):
            return len(self.images)
        def __getitem__(self, idx):
            img_name = self.images[idx]
            with Image.open(self.root / img_name) as im:
                im = im.convert("RGB") if im.mode != "RGB" else im
                x = self.transform(im)
            return x, img_name

    test_tf = T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.Lambda(lambda im: letterbox_resize(im, size=IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    ds_test = TestDataset(test_imgs, test_dir, test_tf)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model.eval()
    test_embs_list, test_names = [], []
    with torch.no_grad():
        for x, names in dl_test:
            x = x.to(device).to(memory_format=torch.channels_last)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                emb = model(x)
            test_embs_list.append(emb.detach().cpu())
            test_names.extend(list(names))
    test_embs = torch.cat(test_embs_list, dim=0).numpy()
    test_embs = test_embs / (np.linalg.norm(test_embs, axis=1, keepdims=True) + 1e-10)

    # Retrieval
    if len(prototypes) == 0:
        logging.warning("No prototypes computed; predicting 'new_whale' for all.")
        preds = [["new_whale"] * 5 for _ in test_names]
    else:
        topk_ids_all, max_sim_all = cosine_rank(prototypes, test_embs, topk=5)
        preds = []
        for img_name, ids_row, s in zip(test_names, topk_ids_all, max_sim_all):
            exact = md5_to_id_train.get(test_md5[img_name], None)
            if exact is not None:
                if exact == "new_whale":
                    out = ["new_whale"] + ids_row[:4]
                else:
                    out = [exact] + [i for i in ids_row if i != exact][:4]
                    if len(out) < 5:
                        out = out + ["new_whale"] * (5 - len(out))
                preds.append(out)
                continue
            if s < tau:
                out = ["new_whale"] + ids_row[:4]
            else:
                out = ids_row[:5]
                if "new_whale" not in out:
                    out[-1] = "new_whale"
            preds.append(out)

    sub = pd.DataFrame({"Image": test_names, "Id": [" ".join(p) for p in preds]})
    sub.sort_values("Image", inplace=True)
    sub.to_csv(SUB_PATH, index=False)
    logging.info(f"Submission written to: {SUB_PATH}")

    # Distribution logging
    top1 = [row.split(" ")[0] for row in sub["Id"].tolist()]
    vc = Counter(top1)
    logging.info(f"Top-1 distribution (top 10): {json.dumps(dict(vc.most_common(10)), indent=2)}")
    frac_new = vc.get("new_whale", 0) / max(1, len(sub))
    logging.info(f"Fraction 'new_whale' at rank-1: {frac_new:.3f}")

# -----------------------------
# Entry point: run twice (DEBUG then FULL)
# -----------------------------
if __name__ == "__main__":
    run_pipeline(DEBUG=True)
    run_pipeline(DEBUG=False)

