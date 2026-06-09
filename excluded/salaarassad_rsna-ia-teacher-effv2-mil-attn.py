import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.serialization import add_safe_globals

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

import torchvision.transforms as VT
import torchvision.transforms.functional as VF


CFG = {
    # Preprocessed volumes
    "PREPROC_INDEX": "/kaggle/input/preprocessingv5/preproc_teacher/preproc_teacher_index.csv",
    "NPZ_DIR": "/kaggle/input/preprocessingv5/preproc_teacher",

    # RSNA labels & localizers
    "TRAIN_LABELS": "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv",
    "LOCALIZERS_CSV": "/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv",

    # Training hyperparameters
    "epochs": 12,
    "freeze_encoder_epochs": 3,
    "batch_series": 2,
    "num_workers": 2,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    "aneurysm_weight": 13.0,
    "label_smoothing": 0.02,
    "seed": 18,

    # MIL / memory safety
    "max_s_train": 96,
    "max_s_val": 128,
    "slice_sample": "uniform",   # "uniform" | "center"
    "encoder_chunk": 32,
    "channels_last": True,
    "cuda_alloc_conf": "expandable_segments:True,max_split_size_mb:128",

    # AMP & TTA
    "use_amp": True,
    "tta_hflip": True,
    "lambda_attn": 0.08,

    # CV / splits
    "folds": 3,

    # Device / fallback
    "fallback_to_cpu_if_cuda_unhealthy": True,

    # Checkpointing
    "ckpt_dir": "/kaggle/working",
    "best_ckpt_name": "model_teacher_effs_mil_best.pt",
}

Path(CFG["ckpt_dir"]).mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. DEVICE, SEEDING, HELPERS
# ============================================================
def is_cuda_healthy() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.synchronize()
        _ = torch.empty(1, device="cuda")
        torch.cuda.synchronize()
        return True
    except Exception as e:
        print(f"[WARN] CUDA health check failed: {type(e).__name__}: {e}")
        return False


# Env for CUDA allocation
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ and CFG["cuda_alloc_conf"]:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CFG["cuda_alloc_conf"]

_cuda_ok = is_cuda_healthy()
if not _cuda_ok and CFG["fallback_to_cpu_if_cuda_unhealthy"]:
    print("[INFO] Falling back to CPU due to CUDA health check failure.")
    device = "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int = 42, allow_cuda_seed: bool = True):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if allow_cuda_seed and torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except RuntimeError as e:
            print(f"[WARN] Skipping CUDA seeding ({type(e).__name__}: {e})")

set_seed(CFG["seed"], allow_cuda_seed=(device == "cuda"))

if CFG["channels_last"] and device == "cuda":
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

def get_base_model(m):
    return m.module if hasattr(m, "module") else m

def unwrap_state_dict(m):
    return get_base_model(m).state_dict()

# Safe torch.load for numpy scalars (PyTorch >= 2.6)
add_safe_globals([np.core.multiarray.scalar, np.dtype])

# ============================================================
# 2. LABELS & BASIC HELPERS
# ============================================================
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]
NUM_LABELS = len(LABEL_COLS)
MODALITY_TO_ID = {"CTA": 0, "MRA": 1}

def load_labels_frame(labels_csv: str, uid_col: str = "SeriesInstanceUID") -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    needed = [uid_col] + LABEL_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Label CSV missing columns: {missing}")
    return df[needed].copy()



# ============================================================
# 3. AUGMENTATIONS, LOCALIZER MAP, DATASET & COLLATE
# ============================================================
class SliceAug:
    """Mild per-slice augmentations (3xHxW) on CPU."""
    def __init__(self, p=0.75):
        self.p = p
        self.jitter = VT.ColorJitter(brightness=0.15, contrast=0.15)
    def __call__(self, x_np):  # (3,H,W) float32 in [0,1]
        if np.random.rand() > self.p:
            return x_np
        x = torch.from_numpy(x_np)
        if np.random.rand() < 0.4:
            angle = float(np.random.uniform(-7.0, 7.0))
            x = VF.rotate(x, angle, interpolation=VT.InterpolationMode.BILINEAR)
        if np.random.rand() < 0.5:
            x = self.jitter(x)
        if np.random.rand() < 0.25:
            x = VT.RandomErasing(
                p=0.35, scale=(0.01, 0.05), ratio=(0.3, 3.0),
                value=0.0, inplace=False
            )(x)
        return x.clamp_(0, 1).numpy()

SLICE_AUG = SliceAug(p=0.75)

def build_series_to_roi_sops(localizers_csv: str) -> dict:
    loc = pd.read_csv(localizers_csv)
    if "SOPInstanceUID" not in loc.columns:
        raise ValueError("train_localizers.csv missing SOPInstanceUID column")
    if "any" in loc.columns:
        roi_loc = loc[loc["any"] == 1]
    else:
        roi_loc = loc
    series_to_roi = (
        roi_loc.groupby("SeriesInstanceUID")["SOPInstanceUID"]
        .apply(lambda x: set(map(str, x.values)))
        .to_dict()
    )
    return series_to_roi

class RSNATeacherSeriesNPZ(Dataset):
    """
    NPZ layout (from preproc_teacher):
      - volume: [Z,3,H,W] uint8
      - sops:   [Z] SOPInstanceUID strings
      - spacing, modality, series_uid, bbox, z_bounds (some ignored here)
    index CSV needs: SeriesInstanceUID, Modality, npz_path
    """
    def __init__(
        self,
        index_csv: str,
        labels_df: pd.DataFrame,
        series_to_roi_sops: dict,
        uid_col: str = "SeriesInstanceUID",
        npz_path_col: str = "npz_path",
        modality_col: str = "Modality",
        load_into_mem: bool = False,
        dtype: torch.dtype = torch.float32,
        do_aug: bool = False,
    ):
        self.meta = pd.read_csv(index_csv)
        need_cols = {"SeriesInstanceUID", npz_path_col, modality_col}
        if not need_cols.issubset(self.meta.columns):
            raise ValueError(f"index CSV must include {need_cols}")
        self.uid_col = uid_col
        self.npz_path_col = npz_path_col
        self.modality_col = modality_col
        self.labels_df = labels_df.set_index(uid_col)
        self.series_to_roi_sops = series_to_roi_sops or {}
        self.load_into_mem = load_into_mem
        self.dtype = dtype
        self.do_aug = do_aug

        self.cache = {}
        if load_into_mem:
            for _, row in self.meta.iterrows():
                uid = str(row[self.uid_col])
                p = row[self.npz_path_col]
                data = np.load(p, allow_pickle=True)
                self.cache[uid] = {k: data[k] for k in data.files}

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        uid = str(row[self.uid_col])
        p = row[self.npz_path_col]

        if self.load_into_mem and uid in self.cache:
            data = self.cache[uid]
        else:
            data = np.load(p, allow_pickle=True)

        if "volume" not in data.files:
            raise ValueError(f"{p} missing 'volume' array")

        vol = data["volume"]  # [Z,3,H,W] uint8
        if vol.ndim != 4:
            raise ValueError(f"{uid}: expected 4D volume, got {vol.shape}")

        slices = vol.astype(np.float32) / 255.0  # [Z,3,H,W]

        sops = None
        if "sops" in data.files:
            sops_arr = data["sops"]
            sops = [str(s) for s in sops_arr.tolist()]
            if len(sops) != slices.shape[0]:
                if len(sops) == 1:
                    sops = [sops[0] for _ in range(slices.shape[0])]
                else:
                    sops = sops[:slices.shape[0]]
                    while len(sops) < slices.shape[0]:
                        sops.append(sops[-1])

        if self.do_aug:
            for i in range(slices.shape[0]):
                slices[i] = SLICE_AUG(slices[i])

        modality_str = str(row[self.modality_col]).upper()
        modality_id = MODALITY_TO_ID.get(modality_str, 0)

        # labels
        y = self.labels_df.loc[uid, LABEL_COLS].astype(float).values
        has_aneurysm = (y[LABEL_COLS.index("Aneurysm Present")] == 1.0)

        S = slices.shape[0]
        attn_target = np.zeros((S,), dtype=np.float32)
        if has_aneurysm and sops is not None and uid in self.series_to_roi_sops:
            roi_sops = self.series_to_roi_sops[uid]
            for i, sop in enumerate(sops):
                if sop in roi_sops:
                    attn_target[i] = 1.0
            # neighbor dilation
            roi_idx = np.where(attn_target > 0.5)[0]
            for i in roi_idx:
                if i - 1 >= 0:
                    attn_target[i-1] = max(attn_target[i-1], 0.5)
                if i + 1 < S:
                    attn_target[i+1] = max(attn_target[i+1], 0.5)

        return {
            "uid": uid,
            "slices": torch.from_numpy(slices).to(self.dtype),  # [S,3,H,W]
            "modality_id": torch.tensor(modality_id, dtype=torch.long),
            "labels": torch.tensor(y, dtype=torch.float32),
            "attn_target": torch.from_numpy(attn_target),
            "modality_str": modality_str,
        }

def _choose_indices(n: int, k: int, mode: str = "uniform", jitter: bool = False):
    k = int(min(max(k, 1), n))
    if k == n:
        return np.arange(n, dtype=np.int64)
    if mode == "center":
        mid = n // 2; half = k // 2
        s = max(0, mid - half); e = min(n, s + k); s = e - k
        return np.arange(s, e, dtype=np.int64)
    grid = np.linspace(0, n - 1, k + 2)[1:-1]
    if jitter:
        noise = np.random.uniform(-0.10, 0.10, size=grid.shape) * max(n-1, 1)
        grid = np.clip(grid + noise, 0, n-1)
    idx = np.unique(np.round(grid).astype(np.int64))
    while len(idx) < k:
        idx = np.unique(np.append(idx, np.random.randint(0, n)))
    return idx[:k]

def collate_series_batch(batch):
    if len(batch) == 0:
        return None
    max_s = max(x["slices"].shape[0] for x in batch)
    B = len(batch)
    C, H, W = batch[0]["slices"].shape[1:]

    batch_slices = torch.zeros((B, max_s, C, H, W), dtype=batch[0]["slices"].dtype)
    batch_mask   = torch.zeros((B, max_s), dtype=torch.bool)
    batch_attn_t = torch.zeros((B, max_s), dtype=torch.float32)
    labels       = torch.zeros((B, NUM_LABELS), dtype=torch.float32)
    modality_ids = torch.zeros((B,), dtype=torch.long)
    uids         = []

    for i, item in enumerate(batch):
        s = item["slices"]; n = s.shape[0]
        batch_slices[i, :n] = s
        batch_mask[i, :n]   = True
        labels[i]           = item["labels"]
        modality_ids[i]     = item["modality_id"]
        uids.append(item["uid"])
        a = item["attn_target"]
        batch_attn_t[i, :min(n, a.shape[0])] = a[:n]

    return {
        "uids": uids,
        "slices": batch_slices,
        "mask": batch_mask,
        "labels": labels,
        "modality_ids": modality_ids,
        "attn_target": batch_attn_t,
    }

def cap_slices_on_cpu(batch_slices, batch_mask, max_s: int,
                      mode: str = "uniform", stochastic: bool = False,
                      attn_target=None):
    """
    ROI-aware capping: always keep ROI slices (attn_target > 0.5),
    then fill up to max_s from non-ROI.
    """
    assert batch_slices.device.type == "cpu"
    B, S, C, H, W = batch_slices.shape
    n_valid = [int(batch_mask[i].sum().item()) for i in range(B)]
    k_list  = [min(max_s, max(1, n)) for n in n_valid]
    K = max(k_list) if B > 0 else 1

    out_slices = torch.zeros((B, K, C, H, W), dtype=batch_slices.dtype)
    out_masks  = torch.zeros((B, K), dtype=torch.bool)
    out_attn_t = torch.zeros((B, K), dtype=torch.float32) if attn_target is not None else None

    for i in range(B):
        n_i = n_valid[i]; k_i = k_list[i]
        if n_i <= 0:
            out_slices[i, 0] = batch_slices[i, 0]
            out_masks[i, 0]  = True
            if out_attn_t is not None:
                out_attn_t[i, 0] = 0.0
            continue

        valid_idx = np.arange(n_i)
        roi_idx = np.array([], dtype=np.int64)
        if attn_target is not None:
            roi_idx = (attn_target[i, :n_i] > 0.5).nonzero(as_tuple=True)[0].cpu().numpy()
        roi_idx = np.unique(roi_idx)

        if k_i <= len(roi_idx):
            idx = roi_idx[:k_i]
        else:
            non_roi = np.setdiff1d(valid_idx, roi_idx)
            k_rest = k_i - len(roi_idx)
            if len(non_roi) > 0:
                base_idx = _choose_indices(len(non_roi), k_rest, mode=mode, jitter=stochastic)
                idx_rest = non_roi[base_idx]
                idx = np.concatenate([roi_idx, idx_rest])
            else:
                idx = roi_idx
        while len(idx) < k_i:
            idx = np.append(idx, idx[-1])
        idx = idx[:k_i]

        idx_t = torch.as_tensor(idx, dtype=torch.long)
        sel = batch_slices[i, :n_i][idx_t]
        out_slices[i, :k_i] = sel
        out_masks[i, :k_i]  = True
        if out_attn_t is not None:
            out_attn_t[i, :k_i] = attn_target[i, :n_i][idx_t]

    return out_slices, out_masks, out_attn_t


# ============================================================
# 4. MODEL, LOSSES, METRICS
# ============================================================
class EfficientNetV2S_Encoder(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_v2_s(weights=weights)
        self.features = model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            h = self.features(x); h = self.avgpool(h).flatten(1)
            self.out_dim = h.shape[1]

    def forward_slices(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x); h = self.avgpool(h).flatten(1)
        return self.drop(h)

class TinyAttentionMIL(nn.Module):
    def __init__(self, dim: int, hidden: int = 256, drop: float = 0.1):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Dropout(drop),
            nn.Linear(hidden, 1, bias=False),
        )
    def forward(self, H, mask):
        A = self.attn(H).squeeze(-1)      # [B,S]
        A = A.masked_fill(~mask, float("-inf"))
        A = torch.softmax(A, dim=1)
        Z = torch.einsum("bs,bsd->bd", A, H)
        return Z, A

class AneurysmMILModel(nn.Module):
    def __init__(
        self,
        mil_hidden: int = 256,
        head_hidden: int = 768,
        use_modality_embed: bool = True,
        modality_embed_dim: int = 16,
        encoder_pretrained: bool = True,
        encoder_dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = EfficientNetV2S_Encoder(pretrained=encoder_pretrained,
                                               dropout=encoder_dropout)
        enc_dim = self.encoder.out_dim
        self.mil = TinyAttentionMIL(dim=enc_dim, hidden=mil_hidden, drop=0.1)
        self.use_modality_embed = use_modality_embed
        head_in = enc_dim + (modality_embed_dim if use_modality_embed else 0)
        if use_modality_embed:
            self.mod_embed = nn.Embedding(num_embeddings=len(MODALITY_TO_ID),
                                          embedding_dim=modality_embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(head_hidden, NUM_LABELS),
        )

    def forward(self, slices, mask, modality_ids=None):
        B, S, C, H, W = slices.shape
        chunk = max(1, int(CFG.get("encoder_chunk", 64)))
        feats_list = []
        for start in range(0, S, chunk):
            end = min(S, start + chunk)
            x = slices[:, start:end].reshape(-1, C, H, W)
            x = x.contiguous()
            if CFG.get("channels_last", False) and x.is_cuda and torch.cuda.device_count() <= 1:
                x = x.contiguous(memory_format=torch.channels_last)
            h = self.encoder.forward_slices(x)
            feats_list.append(h.view(B, end - start, -1))
        feats = torch.cat(feats_list, dim=1)   # [B,S,D]
        Z, attn = self.mil(feats, mask)        # Z:[B,D], attn:[B,S]
        if self.use_modality_embed and modality_ids is not None:
            Z = torch.cat([Z, self.mod_embed(modality_ids)], dim=1)
        logits = self.head(Z)
        probs = torch.sigmoid(logits)
        return {"logits": logits, "probs": probs, "attn": attn, "bag": Z}

class WeightedBCELossLS(nn.Module):
    def __init__(self, aneurysm_weight: float = 13.0, smoothing: float = 0.02):
        super().__init__()
        w = torch.ones(NUM_LABELS, dtype=torch.float32)
        w[LABEL_COLS.index("Aneurysm Present")] = aneurysm_weight
        self.register_buffer("w", w)
        self.smoothing = smoothing
    def forward(self, logits, targets):
        w = self.w if self.w.device == logits.device else self.w.to(logits.device)
        eps = self.smoothing
        targets = targets * (1 - eps) + 0.5 * eps
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (loss * w).mean()

def attention_supervision_loss(attn, attn_target, mask, alpha: float = 0.1):
    attn = attn * mask
    tgt  = attn_target * mask
    sums = tgt.sum(dim=1, keepdim=True)
    valid = (sums.squeeze(1) > 0)
    if not valid.any():
        return attn.sum() * 0.0
    tgt_norm = torch.zeros_like(tgt)
    tgt_norm[valid] = tgt[valid] / (sums[valid] + 1e-8)
    loss = -(tgt_norm[valid] * (attn[valid] + 1e-8).log()).sum(dim=1).mean()
    return alpha * loss

def columnwise_auc(y_true: np.ndarray, y_prob: np.ndarray, label_names):
    aucs = {}
    for i, name in enumerate(label_names):
        yt, yp = y_true[:, i], y_prob[:, i]
        if len(np.unique(yt)) < 2:
            aucs[name] = np.nan
        else:
            aucs[name] = roc_auc_score(yt, yp)
    return aucs

def weighted_mean_auc(aucs_dict: dict) -> float:
    num = den = 0.0
    for k, v in aucs_dict.items():
        if np.isnan(v):
            continue
        w = 13.0 if k == "Aneurysm Present" else 1.0
        num += w * v; den += w
    return num / den if den > 0 else np.nan


# ============================================================
# 5. LOAD LABELS, SPLIT DATA, BUILD DATALOADERS
# ============================================================
labels_df = load_labels_frame(CFG["TRAIN_LABELS"])
index_df  = pd.read_csv(CFG["PREPROC_INDEX"])

# fix npz paths if needed
if "npz_path" in index_df.columns:
    def fix_path(p):
        fname = Path(p).name
        return str(Path(CFG["NPZ_DIR"]) / fname)
    index_df["npz_path"] = index_df["npz_path"].apply(fix_path)
else:
    raise ValueError("preproc_teacher_index.csv must have 'npz_path' column")

# keep only labeled series
index_labeled = index_df.merge(labels_df[["SeriesInstanceUID"]],
                               on="SeriesInstanceUID", how="inner")
tmp = labels_df.set_index("SeriesInstanceUID").loc[
    index_labeled["SeriesInstanceUID"], ["Aneurysm Present"]
].reset_index()
index_labeled = index_labeled.merge(tmp, on="SeriesInstanceUID", how="left")

y_strat = (
    index_labeled["Modality"].astype(str) + "_" +
    tmp["Aneurysm Present"].astype(int).astype(str)
).values

vc = pd.Series(y_strat).value_counts()
safe_folds = int(min(CFG["folds"], max(2, vc.min()))) if len(vc) > 0 else 2

if safe_folds < 2 or len(index_labeled) < 3:
    rng = np.random.default_rng(CFG["seed"])
    perm = rng.permutation(len(index_labeled))
    cut = max(1, int(0.8 * len(index_labeled)))
    train_idx, val_idx = perm[:cut], perm[cut:]
else:
    skf = StratifiedKFold(n_splits=safe_folds, shuffle=True,
                          random_state=CFG["seed"])
    train_idx, val_idx = next(skf.split(index_labeled, y_strat))

train_meta = index_labeled.iloc[train_idx].reset_index(drop=True)
val_meta   = index_labeled.iloc[val_idx].reset_index(drop=True)

if len(val_meta) == 0 and len(train_meta) > 0:
    val_meta = train_meta.tail(1).copy()
    train_meta = train_meta.iloc[:-1].reset_index(drop=True)

TRAIN_SPLIT = "/kaggle/working/train_split_teacher.csv"
VAL_SPLIT   = "/kaggle/working/val_split_teacher.csv"
train_meta.to_csv(TRAIN_SPLIT, index=False)
val_meta.to_csv(VAL_SPLIT, index=False)
print(f"[SPLIT] train: {len(train_meta)} | val: {len(val_meta)} | folds used: {safe_folds}")

series_to_roi_sops = build_series_to_roi_sops(CFG["LOCALIZERS_CSV"])

train_ds = RSNATeacherSeriesNPZ(
    TRAIN_SPLIT,
    labels_df=labels_df,
    series_to_roi_sops=series_to_roi_sops,
    load_into_mem=False,
    do_aug=True,
)
val_ds = RSNATeacherSeriesNPZ(
    VAL_SPLIT,
    labels_df=labels_df,
    series_to_roi_sops=series_to_roi_sops,
    load_into_mem=False,
    do_aug=False,
)

train_dl = DataLoader(
    train_ds,
    batch_size=CFG["batch_series"],
    sampler=RandomSampler(train_ds),
    num_workers=CFG["num_workers"],
    pin_memory=True,
    collate_fn=collate_series_batch,
    persistent_workers=(CFG["num_workers"] > 0),
)
val_dl = DataLoader(
    val_ds,
    batch_size=CFG["batch_series"],
    sampler=SequentialSampler(val_ds),
    num_workers=CFG["num_workers"],
    pin_memory=True,
    collate_fn=collate_series_batch,
    persistent_workers=(CFG["num_workers"] > 0),
)

analysis_dl = DataLoader(
    val_ds,
    batch_size=1,
    sampler=SequentialSampler(val_ds),
    num_workers=0,
    pin_memory=True,
    collate_fn=collate_series_batch,
)


# ============================================================
# 6. MODEL, OPTIMIZER, EMA & TRAIN LOOP
# ============================================================
model = AneurysmMILModel(
    mil_hidden=256,
    head_hidden=768,
    use_modality_embed=True,
    modality_embed_dim=16,
    encoder_pretrained=True,
    encoder_dropout=0.2,
).to(device)

NUM_GPUS = torch.cuda.device_count() if device == "cuda" else 0
if device == "cuda" and NUM_GPUS > 1:
    print(f"Using DataParallel on {NUM_GPUS} GPUs")
    model = nn.DataParallel(model)
    CFG["channels_last"] = False  # channels_last + DataParallel is messy

if CFG["channels_last"] and device == "cuda":
    get_base_model(model).to(memory_format=torch.channels_last)

# Freeze BN for stability
def freeze_bn(m):
    for module in m.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

freeze_bn(get_base_model(model))

criterion = WeightedBCELossLS(
    aneurysm_weight=CFG["aneurysm_weight"],
    smoothing=CFG["label_smoothing"],
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CFG["lr"],
    weight_decay=CFG["weight_decay"],
)

steps_per_epoch = max(1, len(train_dl))
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-4,
    div_factor=10.0,
    total_steps=CFG["epochs"] * steps_per_epoch,
    pct_start=0.15,
    final_div_factor=10.0,
    anneal_strategy="cos",
)

scaler = torch.amp.GradScaler(
    'cuda',
    enabled=(device == "cuda" and CFG["use_amp"])
)

# Warm-up: freeze encoder
for p in get_base_model(model).encoder.parameters():
    p.requires_grad = False

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            k: v.clone().detach()
            for k, v in get_base_model(model).state_dict().items()
            if v.dtype.is_floating_point
        }
    @torch.no_grad()
    def update(self, model):
        sd = get_base_model(model).state_dict()
        for k, v in self.shadow.items():
            v.mul_(self.decay).add_(sd[k].detach(), alpha=1.0 - self.decay)
    @torch.no_grad()
    def copy_to(self, model):
        sd = get_base_model(model).state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v)

ema = EMA(model, decay=0.999)

best_wauc = -1.0
best_path = str(Path(CFG["ckpt_dir"]) / CFG["best_ckpt_name"])
no_improve = 0
EARLY_STOP = 6

def forward_pass(bag_slices, bag_mask, mods):
    return model(bag_slices, bag_mask, mods)

def run_epoch(dataloader, train=True, stochastic_bag=False, tta=False, lambda_attn=0.0):
    if len(dataloader) == 0:
        return 0.0, float("nan"), {}, 0.0, 0.0

    model.train(train)
    total_loss = total_loss_cls = total_loss_attn = 0.0
    probs_cat, labels_cat = [], []

    max_s_cap   = CFG["max_s_train"] if train else CFG["max_s_val"]
    sample_mode = CFG["slice_sample"]

    for batch in dataloader:
        if batch is None:
            continue

        slices_cpu = batch["slices"]
        mask_cpu   = batch["mask"]
        y_cpu      = batch["labels"]
        mods_cpu   = batch["modality_ids"]
        attn_t_cpu = batch["attn_target"]

        slices_cpu, mask_cpu, attn_t_cpu = cap_slices_on_cpu(
            slices_cpu, mask_cpu, max_s=max_s_cap,
            mode=sample_mode, stochastic=(train and stochastic_bag),
            attn_target=attn_t_cpu,
        )

        slices = slices_cpu.to(device, non_blocking=True)
        mask   = mask_cpu.to(device, non_blocking=True)
        y      = y_cpu.to(device, non_blocking=True)
        mods   = mods_cpu.to(device, non_blocking=True)
        attn_t = attn_t_cpu.to(device, non_blocking=True)

        bs = slices.size(0)

        if train:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device == "cuda" and CFG["use_amp"])):
                out = forward_pass(slices, mask, mods)
                loss_cls = criterion(out["logits"], y)
                if lambda_attn > 0:
                    loss_attn = attention_supervision_loss(
                        out["attn"], attn_t, mask, alpha=lambda_attn
                    )
                else:
                    loss_attn = torch.tensor(0.0, device=device)
                loss = loss_cls + loss_attn
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
        else:
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=(device == "cuda" and CFG["use_amp"])):
                    if not tta:
                        out = forward_pass(slices, mask, mods)
                        loss_cls = criterion(out["logits"], y)
                    else:
                        out1 = forward_pass(slices, mask, mods)
                        if CFG["tta_hflip"]:
                            slices_flipped = slices.flip(-1)
                            out2 = forward_pass(slices_flipped, mask, mods)
                            logits = 0.5 * (out1["logits"] + out2["logits"])
                            probs = torch.sigmoid(logits)
                            out = {"logits": logits, "probs": probs}
                        else:
                            out = out1
                        loss_cls = criterion(out["logits"], y)
                    loss_attn = torch.tensor(0.0, device=device)
                    loss = loss_cls

        total_loss      += float(loss.detach().cpu()) * bs
        total_loss_cls  += float(loss_cls.detach().cpu()) * bs
        total_loss_attn += float(loss_attn.detach().cpu()) * bs

        probs_cat.append(out["probs"].detach().cpu().numpy())
        labels_cat.append(y.detach().cpu().numpy())

    if len(labels_cat) == 0:
        return 0.0, float("nan"), {}, 0.0, 0.0

    y_true_np = np.vstack(labels_cat)
    y_prob_np = np.vstack(probs_cat)
    aucs = columnwise_auc(y_true_np, y_prob_np, LABEL_COLS)
    wauc = weighted_mean_auc(aucs)

    N = len(dataloader.dataset)
    avg_loss = total_loss / N
    avg_cls  = total_loss_cls / N
    avg_attn = total_loss_attn / N
    return avg_loss, wauc, aucs, avg_cls, avg_attn

@torch.no_grad()
def evaluate_with_ema(dataloader):
    backup = {k: v.clone() for k, v in get_base_model(model).state_dict().items()}
    ema.copy_to(model)
    va_loss, va_wauc, va_aucs, _, _ = run_epoch(dataloader, train=False, tta=True)
    get_base_model(model).load_state_dict(backup, strict=True)
    return va_loss, va_wauc, va_aucs

# history logging
history = []
hist_path = Path(CFG["ckpt_dir"]) / "training_history_teacher.csv"

for epoch in range(1, CFG["epochs"] + 1):
    if epoch == CFG["freeze_encoder_epochs"] + 1:
        for p in get_base_model(model).encoder.parameters():
            p.requires_grad = True

    if   epoch <= 4:  lambda_attn = 0.0
    elif epoch <= 8:  lambda_attn = CFG["lambda_attn"] * 0.5
    else:             lambda_attn = CFG["lambda_attn"]

    tr_loss, tr_wauc, _, tr_cls, tr_attn = run_epoch(
        train_dl, train=True, stochastic_bag=True, lambda_attn=lambda_attn
    )
    va_loss, va_wauc, va_aucs = evaluate_with_ema(val_dl)

    print(
        f"Epoch {epoch:02d} | train loss {tr_loss:.4f} "
        f"(cls {tr_cls:.4f}, attn {tr_attn:.4f}) | "
        f"val wAUC {va_wauc:.4f} | "
        f"val Aneurysm AUC {va_aucs.get('Aneurysm Present', np.nan):.4f}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": float(tr_loss),
        "train_cls_loss": float(tr_cls),
        "train_attn_loss": float(tr_attn),
        "train_wAUC": float(tr_wauc) if tr_wauc is not None else np.nan,
        "val_loss": float(va_loss),
        "val_wAUC": float(va_wauc),
        "val_auc_aneurysm": float(va_aucs.get("Aneurysm Present", np.nan)),
    })
    pd.DataFrame(history).to_csv(hist_path, index=False)

    if not np.isnan(va_wauc) and va_wauc > best_wauc:
        best_wauc = va_wauc
        no_improve = 0
        torch.save({
            "state_dict": unwrap_state_dict(model),
            "best_wauc": best_wauc,
            "label_order": LABEL_COLS,
        }, best_path)
        print(f"  ↳ Saved best to {best_path}")
    else:
        no_improve += 1
        if no_improve >= EARLY_STOP:
            print("Early stopping (no val wAUC improvement).")
            last_path = str(Path(CFG["ckpt_dir"]) / "model_teacher_effs_mil_last.pt")
            torch.save({
                "state_dict": unwrap_state_dict(model),
                "best_wauc": float(best_wauc),
                "label_order": list(LABEL_COLS),
            }, last_path)
            break

    last_path = str(Path(CFG["ckpt_dir"]) / "model_teacher_effs_mil_last.pt")
    torch.save({
        "state_dict": unwrap_state_dict(model),
        "epoch": int(epoch),
        "label_order": list(LABEL_COLS),
    }, last_path)

print("Best weighted AUC:", best_wauc)

# Reload history for plots
hist_df = pd.read_csv(hist_path)
plt.figure(figsize=(6,4))
plt.plot(hist_df["epoch"], hist_df["train_loss"], marker="o", label="Train loss")
plt.plot(hist_df["epoch"], hist_df["val_loss"], marker="o", label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(hist_df["epoch"], hist_df["val_wAUC"], marker="o", label="Val wAUC")
plt.plot(hist_df["epoch"], hist_df["val_auc_aneurysm"], marker="o", label="Val Aneurysm AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Validation AUCs over epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 7. ANALYSIS: LOAD BEST MODEL, COLLECT PREDICTIONS
# ============================================================

best_path = "/kaggle/input/efficientnetv2-mil-att-v3/model_teacher_effs_mil_best.pt"
state = torch.load(best_path, map_location=device, weights_only=False)
state_dict = state["state_dict"] if "state_dict" in state else state

model = AneurysmMILModel(
    mil_hidden=256,
    head_hidden=768,
    use_modality_embed=True,
    modality_embed_dim=16,
    encoder_pretrained=False,
    encoder_dropout=0.2,
).to(device)
model.load_state_dict(state_dict, strict=True)
model.eval()
base_model = get_base_model(model)

ANEUR_IDX = LABEL_COLS.index("Aneurysm Present")
print("Model reloaded. Aneurysm label index:", ANEUR_IDX)

def collect_series_predictions(dataloader, max_cases=None, thresh=0.5):
    records = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if batch is None:
                continue
            if max_cases is not None and i >= max_cases:
                break
            uids   = batch["uids"]
            slices = batch["slices"].to(device)
            mask   = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            mods   = batch["modality_ids"].to(device)

            out   = model(slices, mask, mods)
            probs = out["probs"].cpu().numpy()[0]
            attn  = out["attn"].cpu().numpy()[0]
            mask_np = batch["mask"].cpu().numpy()[0].astype(bool)

            y_true_full = labels.cpu().numpy()[0]
            uid   = uids[0]
            mod_id = int(batch["modality_ids"].cpu().numpy()[0])

            gt   = int(y_true_full[ANEUR_IDX])
            p    = float(probs[ANEUR_IDX])
            pred = int(p >= thresh)

            if   gt == 1 and pred == 1: cat = "TP"
            elif gt == 0 and pred == 1: cat = "FP"
            elif gt == 1 and pred == 0: cat = "FN"
            else:                       cat = "TN"

            records.append({
                "uid": uid,
                "modality_id": mod_id,
                "y_true": gt,
                "prob": p,
                "pred": pred,
                "category": cat,
                "attn": attn[mask_np],
            })
    return records

records = collect_series_predictions(analysis_dl, max_cases=None, thresh=0.5)
print(f"Collected {len(records)} series for analysis.")

# Global metrics
y_true = np.array([r["y_true"] for r in records])
y_prob = np.array([r["prob"] for r in records])
y_pred = np.array([r["pred"] for r in records])

cm = confusion_matrix(y_true, y_pred, labels=[0,1])
tn, fp, fn, tp = cm.ravel()
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(cm)
print(f"\nTN={tn}, FP={fp}, FN={fn}, TP={tp}")

sens = tp / (tp + fn + 1e-8)
spec = tn / (tn + fp + 1e-8)
prec = tp / (tp + fp + 1e-8)
print(f"\nSensitivity (Recall+): {sens:.3f}")
print(f"Specificity (Recall-): {spec:.3f}")
print(f"Precision:             {prec:.3f}")

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=["No aneurysm", "Aneurysm"]))

roc_auc = roc_auc_score(y_true, y_prob)
print(f"ROC AUC (Aneurysm Present): {roc_auc:.3f}")

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["No aneurysm", "Aneurysm"])
disp.plot(values_format="d")
plt.title("Confusion Matrix — Aneurysm Present")
plt.show()

fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Aneurysm Present")
plt.legend()
plt.grid(True)
plt.show()

prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)
plt.figure()
plt.plot(rec_curve, prec_curve, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve — Aneurysm Present")
plt.legend()
plt.grid(True)
plt.show()

# Per-modality metrics
mod_ids = np.array([r["modality_id"] for r in records])
for mid, name in [(0, "CTA"), (1, "MRA")]:
    mask = (mod_ids == mid)
    if mask.sum() == 0:
        continue
    print(f"\n=== {name} only ===")
    y_true_m = y_true[mask]
    y_prob_m = y_prob[mask]
    y_pred_m = (y_prob_m >= 0.5).astype(int)
    cm_m = confusion_matrix(y_true_m, y_pred_m, labels=[0,1])
    print("Confusion Matrix:\n", cm_m)
    print("ROC AUC:", roc_auc_score(y_true_m, y_prob_m))



# ============================================================
# 8. GRAD-CAM + ATTENTION VISUALIZATION + DASHBOARD
# ============================================================
# Install once in the notebook
!pip install grad-cam -q

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

NPZ_DIR = Path(CFG["NPZ_DIR"])

class MILGradCAMWrapper(nn.Module):
    def __init__(self, mil_model, modality_id: int):
        super().__init__()
        self.mil_model = mil_model
        self.modality_id = modality_id
    def forward(self, x):
        device_local = next(self.mil_model.parameters()).device
        x = x.to(device_local)
        S = x.shape[0]
        B = 1
        bag  = x.unsqueeze(0)
        mask = torch.ones((B,S), dtype=torch.bool, device=device_local)
        mods = torch.full((B,), self.modality_id, dtype=torch.long, device=device_local)
        out  = self.mil_model(bag, mask, mods)
        return out["logits"]



checkpoint_path = "/kaggle/input/efficientnetv2-mil-att-v3/model_teacher_effs_mil_best.pt"
print("Loading best checkpoint from:", checkpoint_path)

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

base_model = AneurysmMILModel(
    mil_hidden=256,
    head_hidden=768,
    use_modality_embed=True,
    modality_embed_dim=16,
    encoder_pretrained=False,   
    encoder_dropout=0.2,
).to(device)

# Load weights
missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

base_model.eval()
print("Checkpoint loaded successfully.")

    
target_layers = [base_model.encoder.features[-1]]
cta_wrapper   = MILGradCAMWrapper(base_model, MODALITY_TO_ID["CTA"]).to(device)
mra_wrapper   = MILGradCAMWrapper(base_model, MODALITY_TO_ID["MRA"]).to(device)
cam = GradCAM(model=cta_wrapper, target_layers=target_layers)

def run_gradcam_for_uid_ax(uid: str,
                           modality_str: str = "CTA",
                           slice_idx: int = None,
                           target_label_idx: int = ANEUR_IDX,
                           ax=None):
    npz_path = NPZ_DIR / f"{uid}.npz"
    if not npz_path.exists():
        print(f"[WARN] NPZ not found: {npz_path}")
        return
    data = np.load(npz_path, allow_pickle=True)
    vol  = data["volume"].astype(np.float32) / 255.0
    Z, C, H, W = vol.shape
    if slice_idx is None:
        slice_idx = Z // 2
    slice_idx = max(0, min(slice_idx, Z-1))
    slice_chw = vol[slice_idx]
    slice_hwc = np.transpose(slice_chw, (1,2,0))

    mod_str_u = modality_str.upper()
    if mod_str_u == "CTA":
        cam.model = cta_wrapper
    elif mod_str_u == "MRA":
        cam.model = mra_wrapper
    else:
        print(f"[WARN] Unknown modality {modality_str}, defaulting to CTA.")
        cam.model = cta_wrapper

    input_tensor = torch.from_numpy(slice_chw).unsqueeze(0)
    targets = [ClassifierOutputTarget(target_label_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = show_cam_on_image(slice_hwc, grayscale_cam, use_rgb=True)

    if ax is None:
        plt.figure(figsize=(6,6))
        plt.imshow(visualization)
        plt.axis("off")
        plt.title(f"Grad-CAM: {modality_str}, UID={uid}, slice={slice_idx}")
        plt.show()
    else:
        ax.imshow(visualization)
        ax.axis("off")

def pick_examples(records, per_cat=3):
    buckets = defaultdict(list)
    for r in records:
        if len(buckets[r["category"]]) < per_cat:
            buckets[r["category"]].append(r)
    return buckets

examples = pick_examples(records, per_cat=3)

def plot_attention_example(rec, label_str="Aneurysm Present"):
    uid  = rec["uid"]
    attn = rec["attn"]
    npz_path = NPZ_DIR / f"{uid}.npz"
    if not npz_path.exists():
        print(f"[WARN] NPZ not found for {uid} at {npz_path}, skipping.")
        return
    data = np.load(npz_path, allow_pickle=True)
    vol  = data["volume"]
    Z, C, H, W = vol.shape
    S     = len(attn)
    z_idx = int(np.argmax(attn)) if S > 0 else 0
    z_idx = min(z_idx, Z-1)
    img = vol[z_idx,0].astype(np.float32) / 255.0

    fig, axes = plt.subplots(1,2, figsize=(12,4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(
        f"{uid}\ncat={rec['category']}, prob={rec['prob']:.3f}, gt={rec['y_true']}"
    )
    axes[0].axis("off")
    axes[1].plot(np.arange(S), attn, marker="o")
    axes[1].axvline(z_idx, linestyle="--")
    axes[1].set_xlabel("Slice index")
    axes[1].set_ylabel("Attention weight")
    axes[1].set_title(f"Attention over slices ({label_str})")
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()

def show_gradcam_examples_from_records(examples, category: str, k: int = 3):
    recs = examples.get(category, [])
    if not recs:
        print(f"No records found for category={category}")
        return
    recs = recs[:k]
    print(f"\n=== {category} Grad-CAM examples (up to {k}) ===")
    for i, rec in enumerate(recs):
        uid   = rec["uid"]
        attn  = rec["attn"]
        y     = rec["y_true"]
        prob  = rec["prob"]
        slice_idx = int(np.argmax(attn)) if len(attn) > 0 else 0
        modality_str = index_df.loc[
            index_df["SeriesInstanceUID"] == uid, "Modality"
        ].iloc[0]
        print(f"[{category} #{i}] UID={uid}, Modality={modality_str}, y={y}, prob={prob:.3f}, slice_idx={slice_idx}")
        run_gradcam_for_uid_ax(uid, modality_str=modality_str,
                               slice_idx=slice_idx, target_label_idx=ANEUR_IDX)
        plot_attention_example(rec, label_str="Aneurysm Present")


def plot_performance_dashboard(hist_df, records, index_df,
                               figsize=(12,10), thresh=0.5):
    y_true = np.array([r["y_true"] for r in records])
    y_prob = np.array([r["prob"] for r in records])
    y_pred = (y_prob >= thresh).astype(int)

    cm  = confusion_matrix(y_true, y_pred, labels=[0,1])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    # choose example: FN > FP > TP
    cat_order = ["FN","FP","TP"]
    chosen_rec = None
    for cat in cat_order:
        cand = [r for r in records if r["category"] == cat]
        if cand:
            chosen_rec = cand[0]
            break
    if chosen_rec is None and records:
        chosen_rec = records[0]

    fig, axes = plt.subplots(2,2, figsize=figsize)

    # (1,1) Train vs val loss (+ optional val_wAUC)
    ax = axes[0,0]
    ax.plot(hist_df["epoch"], hist_df["train_loss"], marker="o", label="Train loss")
    ax.plot(hist_df["epoch"], hist_df["val_loss"], marker="o", label="Val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend(); ax.grid(True)
    if "val_wAUC" in hist_df.columns:
        ax2 = ax.twinx()
        ax2.plot(hist_df["epoch"], hist_df["val_wAUC"],
                 marker="s", linestyle="--", alpha=0.7, label="Val wAUC")
        ax2.set_ylabel("Val wAUC")
        ax2.legend(loc="lower right")

    # (1,2) ROC
    ax = axes[0,1]
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0,1],[0,1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — Aneurysm Present")
    ax.legend(); ax.grid(True)

    # (2,1) Confusion matrix
    ax = axes[1,0]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No aneurysm", "Aneurysm"])
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix")

    # (2,2) Grad-CAM example
    ax = axes[1,1]
    if chosen_rec is not None:
        uid   = chosen_rec["uid"]
        attn  = chosen_rec["attn"]
        prob  = chosen_rec["prob"]
        gt    = chosen_rec["y_true"]
        cat   = chosen_rec["category"]
        slice_idx = int(np.argmax(attn)) if len(attn) > 0 else 0
        modality_str = index_df.loc[
            index_df["SeriesInstanceUID"] == uid, "Modality"
        ].iloc[0]
        run_gradcam_for_uid_ax(uid=uid, modality_str=modality_str,
                               slice_idx=slice_idx, target_label_idx=ANEUR_IDX, ax=ax)
        ax.set_title(
            f"Grad-CAM ({cat})\nUID={uid}, Mod={modality_str}, "
            f"gt={gt}, prob={prob:.3f}, slice={slice_idx}"
        )
    else:
        ax.text(0.5,0.5,"No records available", ha="center", va="center")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


for cat in ["TP", "FP", "FN"]:
    show_gradcam_examples_from_records(examples, category=cat, k=3)


plot_performance_dashboard(hist_df, records, index_df)




