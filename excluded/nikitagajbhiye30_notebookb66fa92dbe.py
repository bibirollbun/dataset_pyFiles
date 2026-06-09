from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import os
import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# =====================
# CONFIG (ajuste aqui)
# =====================
TRAIN_CSV = "/kaggle/input/grand-xray-slam-division-a/train1.csv"
TEST_CSV  = "/kaggle/input/grand-xray-slam-division-a/sample_submission_1.csv"   # precisa ter 'Image_name'
TRAIN_DIR = "/kaggle/input/grand-xray-slam-division-a/train1"
TEST_DIR  = "/kaggle/input/grand-xray-slam-division-a/test1"

OUT_DIR    = "/kaggle/working/"
IMAGE_SIZE = 320
CHANNELS   = 1            # grayscale
OVERWRITE  = True
N_WORKERS  = max(1, (os.cpu_count() or 2) - 1)

LABEL_COLUMNS = [
    'Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum',
    'Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion',
    'Pleural Other','Pneumonia','Pneumothorax','Support Devices'
]

# ==== Verificações de diretório (patch) ====  # TRAIN_DIR_CHECK_PATCH
from pathlib import Path
import os

def _assert_exists_dir(path_str, hint=""):
    p = Path(path_str)
    if not p.exists():
        print(f"[ERRO] Diretório não encontrado: {p}")
        # tentar auto-descobrir
        candidates = [x for x in Path("/kaggle/input").glob("*") if x.is_dir() and "xray" in x.name.lower() or "grand" in x.name.lower()]
        if candidates:
            print("[DICA] Datasets potenciais em /kaggle/input:", [c.name for c in candidates])
        if hint:
            print("Hint:", hint)
        raise FileNotFoundError(str(p))

_assert_exists_dir(TRAIN_DIR, "Ajuste TRAIN_DIR para o dataset correto em /kaggle/input")
_assert_exists_dir(TEST_DIR, "Ajuste TEST_DIR para o dataset correto em /kaggle/input")



# =====================
# Helpers
# =====================
def _ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _validate_files(file_paths: List[Path]):
    missing = [str(p) for p in file_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} imagens não encontradas. Ex.: {missing[:3]}")

def _compute_slices(n: int, n_workers: int, chunk_size: int | None = None) -> List[Tuple[int,int]]:
    if chunk_size and chunk_size > 0:
        slices = []
        for s in range(0, n, chunk_size):
            e = min(n, s + chunk_size)
            if s < e:
                slices.append((s, e))
        return slices
    # fallback: dividir por workers
    n_workers = max(1, min(n_workers, n))
    base = n // n_workers
    rem  = n % n_workers
    slices = []
    start = 0
    for w in range(n_workers):
        size = base + (1 if w < rem else 0)
        end = start + size
        if start < end:
            slices.append((start, end))
        start = end
    return slices

def _worker_write_slice(
    out_imgs: str,
    files_chunk: List[str],
    idx_start: int,
    image_size: int,
):
    # Reabrir memmap neste processo
    arr = open_memmap(out_imgs, mode='r+')
    H = W = image_size

    # Array scratch para evitar realocações
    scratch = np.empty((H, W), dtype=np.uint8)

    for k, path_str in enumerate(files_chunk):
        i = idx_start + k
        # Leitura grayscale rápida
        img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Falha ao ler: {path_str}")

        # Resize (INTER_AREA é adequado para downscale)
        if img.shape[0] != H or img.shape[1] != W:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        # Garantir dtype
        if img.dtype != np.uint8:
            img = img.astype(np.uint8, copy=False)

        scratch[...] = img
        # Escrever (C,H,W) com C=1
        arr[i, 0, :, :] = scratch

    # Fechar view do memmap neste processo
    del arr
    return True

def _pack_split(csv_path: str, img_dir: str, is_train: bool):
    csv_path = Path(csv_path); img_dir = Path(img_dir)
    out_dir  = Path(OUT_DIR);  _ensure_outdir(out_dir)

    df = pd.read_csv(csv_path)
    if 'Image_name' not in df.columns:
        raise KeyError("CSV precisa conter a coluna 'Image_name'.")

    n = len(df)
    files: List[Path] = [img_dir / str(nm) for nm in df['Image_name'].tolist()]
    _validate_files(files)

    c, h, w = CHANNELS, IMAGE_SIZE, IMAGE_SIZE

    if is_train:
        out_imgs = out_dir / f"images_train_{IMAGE_SIZE}_c1_uint8.npy"
        out_lbls = out_dir / "labels_train.npy"
        out_idx  = out_dir / "index_train.csv"
    else:
        out_imgs = out_dir / f"images_test_{IMAGE_SIZE}_c1_uint8.npy"
        out_lbls = None
        out_idx  = out_dir / "index_test.csv"

    if out_imgs.exists() and not OVERWRITE:
        raise FileExistsError(f"{out_imgs} já existe. Defina OVERWRITE=True para sobrescrever.")

    print(f"=== PACK {'TRAIN' if is_train else 'TEST'} ===")
    print(f"N={n} | {h}x{w} | C={c} | dtype=uint8 -> {out_imgs}")

    # Criar arquivo .npy memmap e pré-alocar
    arr = open_memmap(str(out_imgs), mode='w+', dtype=np.uint8, shape=(n, c, h, w))
    del arr  # será reaberto pelos workers em 'r+'

    # Estratégia de progresso:
    # - blocos menores deixam a barra mais suave; 512/1024 são bons valores.
    # - ajuste chunk_size se quiser barras mais granulares.
    chunk_size = 1024
    slices = _compute_slices(n, N_WORKERS, chunk_size=chunk_size)
    paths_str = [str(p) for p in files]

    total_done = 0
    with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(slices))) as ex, \
         tqdm(total=n, desc="Empacotando (paralelo)", unit="img", dynamic_ncols=True) as pbar:
        futures = []
        for (s, e) in slices:
            fut = ex.submit(
                _worker_write_slice,
                str(out_imgs),
                paths_str[s:e],
                s,
                IMAGE_SIZE,
            )
            futures.append((fut, e - s))

        # Conforme cada slice concluir, atualizamos o progresso
        for fut, size in futures:
            fut.result()  # Propaga erro, se houver
            total_done += size
            pbar.update(size)

    # CSV rápido
    if is_train:
        # Checar colunas
        missing = [c for c in LABEL_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(f"Colunas faltantes no CSV de treino: {missing}")

        out_df = pd.DataFrame({
            "idx": np.arange(n, dtype=np.int32),
            "path": paths_str,  # opcional: salvar apenas Image_name
        })
        for col in LABEL_COLUMNS:
            out_df[col] = df[col].astype(np.float32).values
        out_df.to_csv(out_idx, index=False)

        # Labels separados (compatível com treino)
        labels = df[LABEL_COLUMNS].astype(np.float32).values
        np.save(out_lbls, labels)
        print(f"Labels salvos em: {out_lbls}")

    else:
        out_df = pd.DataFrame({
            "idx": np.arange(n, dtype=np.int32),
            "path": paths_str,
        })
        out_df.to_csv(out_idx, index=False)

    print(f"Imagens salvas em: {out_imgs}")
    print(f"Índice salvo em:  {out_idx}")

# =====================
# Main
# =====================
if __name__ == "__main__":
    _pack_split(TRAIN_CSV, TRAIN_DIR, is_train=True)
    _pack_split(TEST_CSV,  TEST_DIR,  is_train=False)
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import warnings, random, json, os, sys, math
import numpy as np
import pandas as pd
import timm
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as TF
from copy import deepcopy

# =========================
# CONFIG
# =========================
OUT_DIR       = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_TRAIN_NPY = r"/kaggle/working/images_train_320_c1_uint8.npy"
IMG_TEST_NPY  = r"/kaggle/working/images_test_320_c1_uint8.npy"
LBL_TRAIN_NPY = r"/kaggle/working/labels_train.npy"
INDEX_TRAIN_CSV = r"/kaggle/working/index_train.csv"
INDEX_TEST_CSV  = r"/kaggle/working/index_test.csv"
TRAIN1_CSV      = r"/kaggle/input/grand-xray-slam-division-a/train1.csv"
TEST_CSV_PATH   = r"/kaggle/input/grand-xray-slam-division-a/sample_submission_1.csv"
RUNS = Path("/kaggle/working/Runs")
RUNS.mkdir(parents=True, exist_ok=True)
PATIENT_MAP_JSON = OUT_DIR/"patient_map.json"

# Básico
BATCH_SIZE = 64
EPOCHS     = 10
SEED       = 42
K_FOLDS    = 5
NUM_WORKERS = 0

# Opt (LLRD/WD/etc.)
BASE_LR_HEAD    = 1e-3
BASE_LR_BACKB   = 1e-4
LR_DECAY_STAGE  = 0.5
WEIGHT_DECAY    = 5e-2
WARMUP_STEPS    = 800
CLIP_NORM       = 1.0
USE_AMP         = True

# Modelo (ConvNeXt)
DROP_RATE       = 0.10
DROP_PATH_RATE  = 0.10

LABEL_COLUMNS = [
    'Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum',
    'Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion',
    'Pleural Other','Pneumonia','Pneumothorax','Support Devices'
]
# Loss config
LOSS_MODE        = "bce" # "bce" | "focal"
FOCAL_GAMMA      = 1.5
FOCAL_ALPHA      = None  # None | float | list de tamanho C *None se usando AUTO_ALPHA
AUTO_ALPHA_MODE  = None  # None | "pos_prior" | "effective_num"
CB_BETA          = 0.999
ALPHA_CLIP_MIN   = 0.05
ALPHA_CLIP_MAX   = 0.95

# Early stop
EARLYSTOP_PATIENCE  = 1
EARLYSTOP_MIN_DELTA = 1e-4

# Normalização (ImageNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# NF exclusividade (apenas para métrica opcional)
APPLY_NF_EXCL_IN_VAL = False
T_NF_VAL = 0.5

# Reg. exclusividade
LAMBDA_EXCL_START = 0.0
LAMBDA_EXCL_TARGET = 0.0
ANNEAL_EPOCHS = 1

# DRY RUN
DRY_RUN = True
DRY_RUN_N_BATCHES_TRAIN = 3
DRY_RUN_N_BATCHES_VAL   = 3
DRY_RUN_N = 256         # máx. imagens para inferir
DRY_RUN_FOLDS = 1       # máx. folds no ensemble
DRY_RUN_RANDOM = False  # True para amostra aleatória

# EMA
USE_EMA   = True
EMA_DECAY = 0.999

# Aug fracas
AUG_WEAK          = False
AUG_DEG           = 7
AUG_TRANSLATE_FR  = 0.02
AUG_SCALE_FR      = 0.05
AUG_HFLIP_PROB    = 0.0 #Usar com cuidado, muda significado semântico.

# WD dif.
WD_BACKBONE = 5e-2
WD_HEAD     = 1e-2

# Accum
GRAD_ACCUM_STEPS = 1

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_pbar(iterable, desc):
    disable_env = os.environ.get("TQDM_DISABLE", "")
    disable_flag = (disable_env == "1")
    try:
        is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    except Exception:
        is_tty = False
    disable = disable_flag or (not is_tty)
    try:
        return tqdm(iterable, desc=desc, leave=False, ncols=80, file=sys.stdout, disable=disable)
    except Exception:
        return iterable

# Exclusividade NF (para métrica/regularização)
def apply_nf_excl_strict_with_tnf(probs: np.ndarray, label_columns: List[str], t_nf: float):
    out = probs.copy()
    idx_nf = label_columns.index("No Finding")
    pick_nf = (probs[:, idx_nf] >= float(t_nf))
    if np.any(pick_nf):
        out[pick_nf, :] = 0.0
        out[pick_nf, idx_nf] = 1.0
    out[~pick_nf, idx_nf] = 0.0
    return out

def exclusivity_regularizer(p: torch.Tensor, y: torch.Tensor, idx_nf: int,
                            cond_on_ynf: bool = True) -> torch.Tensor:
    mask = torch.ones(p.size(1), dtype=torch.bool, device=p.device)
    mask[idx_nf] = False
    p_any = 1.0 - torch.prod(1.0 - p[:, mask], dim=1)
    reg = p[:, idx_nf] * p_any
    if cond_on_ynf: reg = reg * y[:, idx_nf]
    return reg.mean()

def lambda_excl_at_epoch(epoch: int) -> float:
    if epoch <= 1: return float(LAMBDA_EXCL_START)
    if epoch >= ANNEAL_EPOCHS: return float(LAMBDA_EXCL_TARGET)
    t = (epoch-1) / max(ANNEAL_EPOCHS-1, 1)
    return float(LAMBDA_EXCL_START + t*(LAMBDA_EXCL_TARGET - LAMBDA_EXCL_START))

def safe_macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, y_pred, average='macro'))
    except Exception:
        aucs = []
        C = y_true.shape[1]
        for c in range(C):
            yt = y_true[:, c]; yp = y_pred[:, c]
            if len(np.unique(yt)) == 2:
                aucs.append(roc_auc_score(yt, yp))
        return float(np.mean(aucs)) if aucs else float("nan")

# =========================
# Dataset .NPY
# =========================

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

class NpyDatasetCHW(Dataset):
    def __init__(self, images_path, labels_path=None, to_rgb=True, normalize=True):
        self.images_path = str(images_path)
        self.labels_path = labels_path
        self.to_rgb = to_rgb
        self.normalize = normalize        
        # Memmap: baixo uso de RAM
        self._arr = np.load(self.images_path, mmap_mode='r')  # (N, C, H, W), uint8
        assert self._arr.ndim == 4, f"Esperado (N,C,H,W); obtido {self._arr.shape}"
        self.n, self.c, self.h, self.w = self._arr.shape
        assert self.c in (1, 3), f"C deve ser 1 ou 3; obtido {self.c}"
        self._labels = None
        if self.labels_path is not None:
            y = np.load(self.labels_path)
            if y.dtype != np.float32:
                y = y.astype(np.float32, copy=False)
            assert y.shape[0] == self.n, "#imgs e #labels diferentes"
            self._labels = y
            
    def __len__(self):
        return self.n
        
    def __getitem__(self, idx):
        img = self._arr[idx]                  # (C,H,W), uint8
        if not img.flags.writeable:
            img = img.copy()                  # pequena, só 1 imagem
        if img.shape[0] == 1 and self.to_rgb:
            img = np.repeat(img, 3, axis=0)   # (1,H,W) -> (3,H,W)
        x = torch.from_numpy(img).float()     # (C,H,W)
        if self._arr.dtype == np.uint8:
            x = x / 255.0
        if self.normalize:
            if x.shape[0] == 3:
                x = (x - IMAGENET_MEAN) / IMAGENET_STD
            else:
                x = (x - 0.5) / 0.5
        if self._labels is None:
            return x
        else:
            y = torch.from_numpy(self._labels[idx]).float()
            return x, y

class NpyDatasetWithGroups(NpyDatasetCHW):
    def __init__(self, images_path: str, labels_path: Optional[str], groups: np.ndarray,to_rgb: bool=True, normalize: bool=True):
        super().__init__(images_path, labels_path, to_rgb, normalize)
        assert len(groups) == self.n
        self.groups_arr = groups.astype(np.int64)
    def __getitem__(self, idx: int):
        x = super().__getitem__(idx)
        if isinstance(x, tuple):
            img, y = x; g = torch.tensor(self.groups_arr[idx], dtype=torch.int64); return (img, y, g)
        else:
            img = x; g = torch.tensor(self.groups_arr[idx], dtype=torch.int64); return (img, g)

# =========================
# Split patient-wise
# =========================
def load_patient_groups_for_npy(index_csv: str, train1_csv: str):
    if not Path(index_csv).exists():
        raise FileNotFoundError(f"index_train.csv não encontrado: {index_csv}")
    idx_df = pd.read_csv(index_csv)
    name_col = None
    for cand in ["Image_name","Image_Name","image_name","path"]:
        if cand in idx_df.columns: name_col=cand; break
    if name_col is None:
        raise ValueError("index_train.csv precisa conter 'Image_name' ou 'path'.")
    if name_col == "path":
        idx_df["Image_name"] = idx_df["path"].map(lambda p: Path(str(p)).name)
    else:
        idx_df["Image_name"] = idx_df[name_col].map(lambda p: Path(str(p)).name)
    tr1 = pd.read_csv(train1_csv).rename(columns={"Image_Name":"Image_name","PatientId":"Patient_ID"})
    if "Patient_ID" not in tr1.columns or "Image_name" not in tr1.columns:
        raise ValueError("train1.csv precisa ter 'Patient_ID' e 'Image_name'.")
    merged = idx_df.merge(tr1[["Image_name","Patient_ID"]], on="Image_name", how="left")
    patients = merged["Patient_ID"].astype(str).values
    uniq = pd.unique(patients); pid2int = {p:i for i,p in enumerate(uniq)}
    groups = np.array([pid2int[p] for p in patients], dtype=np.int64)
    meta = {"n_samples": int(len(merged)), "n_patients": int(len(uniq))}
    return groups, meta

def build_patient_stratification(labels_all: np.ndarray, groups: np.ndarray) -> np.ndarray:
    idx_nf = LABEL_COLUMNS.index('No Finding')
    df = pd.DataFrame({"group": groups, "NF": labels_all[:, idx_nf].astype(int)})
    return df.groupby("group")["NF"].max().reindex(df["group"]).to_numpy().astype(int)

def split_patientwise(groups: np.ndarray, labels_all: np.ndarray, n_splits: int, seed: int):
    pat_target = build_patient_stratification(labels_all, groups)
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_idx, va_idx in sgkf.split(np.arange(len(groups)), y=pat_target, groups=groups):
            yield tr_idx, va_idx
        return
    except Exception:
        pass
    # fallback
    rng = np.random.RandomState(seed)
    df_p = pd.DataFrame({"group": groups, "t": pat_target}).drop_duplicates("group")
    pos_p = df_p[df_p["t"]==1]["group"].values; rng.shuffle(pos_p)
    neg_p = df_p[df_p["t"]==0]["group"].values; rng.shuffle(neg_p)
    pos_folds = np.array_split(pos_p, n_splits); neg_folds = np.array_split(neg_p, n_splits)
    for i in range(n_splits):
        va_groups = set(np.concatenate([pos_folds[i], neg_folds[i]]).tolist())
        va_mask = np.array([g in va_groups for g in groups])
        va_idx = np.where(va_mask)[0]; tr_idx = np.where(~va_mask)[0]
        yield tr_idx, va_idx

# =========================
# Modelo + Opt
# =========================
def create_model(num_classes=14, in_chans=3):
    try:
        model = timm.create_model("convnext_tiny", pretrained=True,num_classes=num_classes, in_chans=in_chans,drop_rate=DROP_RATE, drop_path_rate=DROP_PATH_RATE)
    except Exception:
        model = timm.create_model("convnext_tiny", pretrained=False,num_classes=num_classes, in_chans=in_chans,drop_rate=DROP_RATE, drop_path_rate=DROP_PATH_RATE)
    for p in model.parameters(): p.requires_grad = True
    return model

def _add_param_group(params, module, lr, wd):
    if module is None: return
    decay, no_decay = [], []
    for n, p in module.named_parameters():
        if not p.requires_grad: continue
        if p.ndim < 2 or any(k in n.lower() for k in ["bias","norm","bn","ln","gamma","beta"]):
            no_decay.append(p)
        else:
            decay.append(p)
    if decay:    params.append({"params": decay, "lr": lr, "weight_decay": wd})
    if no_decay: params.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})

def build_optimizer_llrd(model: nn.Module):
    assert hasattr(model, "stem") and hasattr(model, "stages") and hasattr(model, "head")
    param_groups = []
    _add_param_group(param_groups, model.head, lr=BASE_LR_HEAD, wd=WD_HEAD)
    stages = list(model.stages)
    stage_lrs = [BASE_LR_BACKB * (LR_DECAY_STAGE ** 3),BASE_LR_BACKB * (LR_DECAY_STAGE ** 2),BASE_LR_BACKB * (LR_DECAY_STAGE ** 1),BASE_LR_BACKB * (LR_DECAY_STAGE ** 0),]
    for stg, lr in zip(stages, stage_lrs):
        _add_param_group(param_groups, stg, lr=lr, wd=WD_BACKBONE)
    stem_lr = BASE_LR_BACKB * (LR_DECAY_STAGE ** 4)
    _add_param_group(param_groups, model.stem, lr=stem_lr, wd=WD_BACKBONE)
    return torch.optim.AdamW(param_groups)

def build_warmup_cosine(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# =========================
# Losses
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma); self.reduction = reduction
        if isinstance(alpha, (list,tuple,np.ndarray)):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer("alpha", alpha if isinstance(alpha, torch.Tensor) else None)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = targets*p + (1.0-targets)*(1.0-p)
        loss = (1.0-pt).pow(self.gamma) * bce
        if self.alpha is not None:
            ap, an = self.alpha, 1.0-self.alpha if self.alpha.ndim==0 else (self.alpha, 1.0-self.alpha)
            a = targets*ap + (1.0-targets)*an
            loss = a*loss
        return loss.mean() if self.reduction=="mean" else loss.sum()

def compute_alpha_from_labels(labels: np.ndarray, mode: str, beta: float,
                              clip_min: float, clip_max: float) -> np.ndarray:
    pos_counts = labels.sum(axis=0).astype(np.float64); n = float(labels.shape[0])
    if mode == "pos_prior":
        p = np.clip(pos_counts/max(n,1.0), 0.0, 1.0); alpha = 1.0 - p
    elif mode == "effective_num":
        w = np.zeros_like(pos_counts, dtype=np.float64); has_pos = pos_counts>0
        w[has_pos] = (1.0-beta)/(1.0-np.power(beta, pos_counts[has_pos]))
        alpha = w/(w.max() if w.max()>0 else 1.0)
    else:
        raise ValueError("AUTO_ALPHA_MODE inválido")
    return np.clip(alpha, clip_min, clip_max).astype(np.float32)

def build_criterion(device, labels_for_alpha: Optional[np.ndarray], n_classes: int):
    if str(LOSS_MODE).lower() == "bce":
        return nn.BCEWithLogitsLoss()
    alpha_tensor = None
    if FOCAL_ALPHA is not None:
        if isinstance(FOCAL_ALPHA, (list,tuple,np.ndarray)):
            arr = np.asarray(FOCAL_ALPHA, dtype=np.float32); assert arr.shape[0]==n_classes
            alpha_tensor = torch.tensor(arr, dtype=torch.float32, device=device)
        else:
            alpha_tensor = torch.tensor(float(FOCAL_ALPHA), dtype=torch.float32, device=device)
    elif AUTO_ALPHA_MODE is not None and labels_for_alpha is not None:
        alpha_vec = compute_alpha_from_labels(labels_for_alpha, AUTO_ALPHA_MODE, CB_BETA,
                                              ALPHA_CLIP_MIN, ALPHA_CLIP_MAX)
        print(f"[AUTO α] modo={AUTO_ALPHA_MODE} | alpha(min,max,mean)=({alpha_vec.min():.3f}, {alpha_vec.max():.3f}, {alpha_vec.mean():.3f})")
        alpha_tensor = torch.tensor(alpha_vec, dtype=torch.float32, device=device)
    return FocalLoss(alpha=alpha_tensor, gamma=float(FOCAL_GAMMA), reduction="mean")

# =========================
# EMA helper
# =========================
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters(): p.requires_grad=False
        self.decay = decay
    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay; msd = model.state_dict(); esd = self.ema.state_dict()
        for k in esd.keys(): esd[k].mul_(d).add_(msd[k], alpha=1.0-d)
            
# =========================
# Validação / Treino
# =========================
def validate(model, loader, criterion, device, return_preds=False):
    model.eval(); loss_sum=0.0; preds=[]; labels_all=[]; groups=[]
    def _unpack(batch):
        if isinstance(batch, (tuple,list)):
            if len(batch)==3: return batch[0], batch[1], batch[2]
            if len(batch)==2: x,y = batch; return x,y,None
        raise ValueError("Batch em formato inesperado")
    pbar = make_pbar(loader, "Val")
    for ib, batch in enumerate(pbar):
        x,y,g = _unpack(batch)
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=(USE_AMP and device.type=='cuda')):
            logits = model(x); loss = criterion(logits, y); ps = torch.sigmoid(logits)
        loss_sum += loss.item()*x.size(0)
        preds.append(ps.detach().cpu().numpy()); labels_all.append(y.detach().cpu().numpy())
        if g is not None: groups.extend(g.detach().cpu().numpy().tolist())
        if DRY_RUN and (ib+1)>=DRY_RUN_N_BATCHES_VAL: break
    epoch_loss = loss_sum/len(loader.dataset)
    preds = np.vstack(preds); labels_all = np.vstack(labels_all)
    auc_raw_img = safe_macro_auc(labels_all, preds)
    auc_gated_img = auc_raw_img
    if APPLY_NF_EXCL_IN_VAL:
        preds_g = apply_nf_excl_strict_with_tnf(preds, LABEL_COLUMNS, T_NF_VAL)
        auc_gated_img = safe_macro_auc(labels_all, preds_g)
    if len(groups)>0:
        groups = np.asarray(groups)
        df_img = pd.DataFrame({"group": groups})
        for j,c in enumerate(LABEL_COLUMNS):
            df_img[c+"_p"]=preds[:,j]; df_img[c+"_y"]=labels_all[:,j]
        grp = df_img.groupby("group", sort=False)
        Yp = grp[[f"{c}_p" for c in LABEL_COLUMNS]].mean().to_numpy()
        Yt = grp[[f"{c}_y" for c in LABEL_COLUMNS]].max().to_numpy()
        auc_raw_pat = safe_macro_auc(Yt, Yp); auc_gated_pat = auc_raw_pat
        if APPLY_NF_EXCL_IN_VAL:
            Yp_g = apply_nf_excl_strict_with_tnf(Yp, LABEL_COLUMNS, T_NF_VAL)
            auc_gated_pat = safe_macro_auc(Yt, Yp_g)
    else:
        groups = np.array([], dtype=np.int64)
        auc_raw_pat = auc_raw_img; auc_gated_pat = auc_gated_img
    if return_preds:
        return (epoch_loss, auc_raw_img, auc_gated_img, auc_raw_pat, auc_gated_pat,preds, labels_all, groups)
    return epoch_loss, auc_raw_img, auc_gated_img, auc_raw_pat, auc_gated_pat

def weak_aug_batch(x: torch.Tensor) -> torch.Tensor:
    if not AUG_WEAK: return x
    B,C,H,W = x.shape; out=x.clone()
    for i in range(B):
        angle = float(torch.empty(1).uniform_(-AUG_DEG, AUG_DEG))
        tx  = int(torch.empty(1).uniform_(-AUG_TRANSLATE_FR*W, AUG_TRANSLATE_FR*W))
        ty  = int(torch.empty(1).uniform_(-AUG_TRANSLATE_FR*H, AUG_TRANSLATE_FR*H))
        scale = float(1.0 + torch.empty(1).uniform_(-AUG_SCALE_FR, AUG_SCALE_FR))
        out[i] = TF.affine(out[i], angle=angle, translate=[tx,ty], scale=scale, shear=[0.0,0.0],interpolation=InterpolationMode.BILINEAR)
        if AUG_HFLIP_PROB>0 and torch.rand(1).item()<AUG_HFLIP_PROB: out[i]=TF.hflip(out[i])
    return out

def train_one_epoch(model, loader, optimizer, criterion, device,lambda_excl: float, scaler: torch.amp.GradScaler,scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,step_scheduler_per_batch: bool = True,ema_helper: ModelEMA | None = None):
    model.train(); loss_sum=0.0; preds=[]; labels_all=[]; groups=[]
    idx_nf = LABEL_COLUMNS.index("No Finding")
    reg_meter=0.0; n_seen=0
    pbar = make_pbar(loader, "Train"); optimizer.zero_grad(set_to_none=True)
    for ib, batch in enumerate(pbar):
        if isinstance(batch,(tuple,list)):
            if len(batch)==3: x,y,g=batch
            elif len(batch)==2: x,y=batch; g=None
            else: raise ValueError("Batch com estrutura inesperada")
        else: raise ValueError("Batch não é tupla/lista")
        x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
        x=weak_aug_batch(x)
        with torch.amp.autocast('cuda', enabled=(USE_AMP and device.type=='cuda')):
            logits=model(x); base_loss=criterion(logits,y); p=torch.sigmoid(logits)
            reg = exclusivity_regularizer(p,y,idx_nf,cond_on_ynf=True); loss=base_loss + lambda_excl*reg
        loss = loss / max(1, GRAD_ACCUM_STEPS)
        scaler.scale(loss).backward()
        if CLIP_NORM and CLIP_NORM>0:
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        if ((ib+1) % max(1, GRAD_ACCUM_STEPS))==0:
            scale_before = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            did_step = (scaler.get_scale() >= scale_before)
            if USE_EMA and ema_helper is not None and did_step: ema_helper.update(model)
            if step_scheduler_per_batch and scheduler is not None and did_step: scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        bs=x.size(0); loss_sum += loss.item()*bs*max(1, GRAD_ACCUM_STEPS)
        reg_meter += reg.item()*bs; n_seen+=bs
        preds.append(p.detach().cpu().numpy()); labels_all.append(y.detach().cpu().numpy())
        if g is not None: groups.extend(g.numpy().tolist())
        try: pbar.set_postfix(loss=float(loss.item()*max(1, GRAD_ACCUM_STEPS)), reg=float((lambda_excl*reg).item()))
        except Exception: pass
        if DRY_RUN and (ib+1)>=DRY_RUN_N_BATCHES_TRAIN: break
    epoch_loss = loss_sum/len(loader.dataset); mean_reg = reg_meter/max(n_seen,1)
    preds=np.vstack(preds); labels_all=np.vstack(labels_all); groups=np.asarray(groups) if len(groups)>0 else np.array([],dtype=np.int64)
    auc_raw_img = safe_macro_auc(labels_all, preds); auc_gated_img = auc_raw_img
    if APPLY_NF_EXCL_IN_VAL:
        preds_g = apply_nf_excl_strict_with_tnf(preds, LABEL_COLUMNS, T_NF_VAL)
        auc_gated_img = safe_macro_auc(labels_all, preds_g)
    if groups.size>0:
        df_img = pd.DataFrame({"group": groups})
        for j,c in enumerate(LABEL_COLUMNS):
            df_img[c+"_p"]=preds[:,j]; df_img[c+"_y"]=labels_all[:,j]
        grp=df_img.groupby("group",sort=False)
        Yp=grp[[f"{c}_p" for c in LABEL_COLUMNS]].mean().to_numpy()
        Yt=grp[[f"{c}_y" for c in LABEL_COLUMNS]].max().to_numpy()
        auc_raw_pat=safe_macro_auc(Yt,Yp); auc_gated_pat=auc_raw_pat
        if APPLY_NF_EXCL_IN_VAL:
            Yp_g=apply_nf_excl_strict_with_tnf(Yp,LABEL_COLUMNS,T_NF_VAL); auc_gated_pat=safe_macro_auc(Yt,Yp_g)
    else:
        auc_raw_pat=auc_raw_img; auc_gated_pat=auc_gated_img
    return (epoch_loss, auc_raw_img, auc_gated_img, auc_raw_pat, auc_gated_pat, mean_reg)

# =========================
# Threshold search (F1)
# =========================
def compute_best_thresholds(y_true: np.ndarray, y_prob: np.ndarray,label_columns: List[str], n_steps:int=101) -> Dict:
    C = y_true.shape[1]; ths = np.zeros(C, dtype=np.float32); grid = np.linspace(0,1,n_steps)
    for c in range(C):
        best_f1, best_t = -1.0, 0.5
        yt = y_true[:,c].astype(np.int32); yp=y_prob[:,c]
        if len(np.unique(yt))<2: ths[c]=0.5; continue
        for t in grid:
            f1 = f1_score(yt, (yp>=t).astype(np.uint8), zero_division=0)
            if f1>best_f1: best_f1, best_t = f1, float(t)
        ths[c]=best_t
    idx_nf = label_columns.index("No Finding")
    return {"label_order": label_columns, "idx_nf": idx_nf,"thresholds": ths.tolist(), "meta": {"n_val": int(y_true.shape[0])}}

# =========================
# Treino por fold
# =========================
def train_fold(fold_id: int, train_idx: np.ndarray, val_idx: np.ndarray,labels_all: np.ndarray, groups_all: np.ndarray, device: torch.device):
    print(f"\n========== FOLD {fold_id+1}/{K_FOLDS} ==========")
    ds_full = NpyDatasetWithGroups(IMG_TRAIN_NPY, LBL_TRAIN_NPY, groups=groups_all, to_rgb=True, normalize=True)    
    pin_mem = (device.type=='cuda')
    ds_tr = NpyDatasetCHW(IMG_TRAIN_NPY, labels_path=LBL_TRAIN_NPY, to_rgb=True, normalize=True)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,pin_memory=(device.type=='cuda'))
    ds_va = NpyDatasetCHW(IMG_TRAIN_NPY, labels_path=LBL_TRAIN_NPY, to_rgb=True, normalize=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_mem)
    criterion = build_criterion(device, labels_all[train_idx], n_classes=len(LABEL_COLUMNS))
    model = create_model(num_classes=len(LABEL_COLUMNS), in_chans=3).to(device)
    optimizer = build_optimizer_llrd(model)
    total_steps = len(dl_tr) * (1 if DRY_RUN else EPOCHS)
    scheduler = build_warmup_cosine(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=(USE_AMP and device.type=='cuda'))
    ema_helper = ModelEMA(model, EMA_DECAY) if USE_EMA else None
    best_auc=-1.0; no_improve=0
    best_path = RUNS / f"best_model_fold{fold_id}.pth"    
    history=[]; local_epochs = 1 if DRY_RUN else EPOCHS
    for epoch in range(1, local_epochs+1):
        lam_excl = lambda_excl_at_epoch(epoch)
        print(f"\n--- Fold {fold_id+1} | Epoch {epoch}/{local_epochs} ---")
        print(f"λ_excl={lam_excl:.3f} | DROP_RATE={DROP_RATE} | WD(head/back)={WD_HEAD}/{WD_BACKBONE} | "f"LRs(head={BASE_LR_HEAD:.1e}, back={BASE_LR_BACKB:.1e}, decay={LR_DECAY_STAGE}) | "f"EMA={USE_EMA} | AUG_WEAK={AUG_WEAK} | ACCUM={GRAD_ACCUM_STEPS}")
        tr_loss, tr_auc_raw_img, tr_auc_gated_img, tr_auc_raw_pat, tr_auc_gated_pat, tr_reg = \
            train_one_epoch(model, dl_tr, optimizer, criterion, device,lambda_excl=lam_excl, scaler=scaler, scheduler=scheduler,step_scheduler_per_batch=True, ema_helper=ema_helper)
        va = validate(ema_helper.ema if (USE_EMA and ema_helper is not None) else model, dl_va, criterion, device)
        va_loss, va_auc_raw_img, va_auc_gated_img, va_auc_raw_pat, va_auc_gated_pat = va
        print(f"[Fold {fold_id}] Train | loss={tr_loss:.4f} | AUROC(img raw)={tr_auc_raw_img:.4f} | AUROC(pat raw)={tr_auc_raw_pat:.4f} | reg={tr_reg:.4f}")
        print(f"[Fold {fold_id}] Val   | loss={va_loss:.4f} | AUROC(img raw)={va_auc_raw_img:.4f} | AUROC(pat raw)={va_auc_raw_pat:.4f}")
        history.append({"epoch": epoch,"train_loss": tr_loss,"train_auc_raw_img": tr_auc_raw_img, "train_auc_raw_pat": tr_auc_raw_pat,"train_reg": tr_reg,"val_loss": va_loss, "val_auc_raw_img": va_auc_raw_img, "val_auc_raw_pat": va_auc_raw_pat})
        score_for_es = va_auc_raw_img if not np.isnan(va_auc_raw_img) else va_auc_gated_img
        if score_for_es > best_auc + EARLYSTOP_MIN_DELTA:
            best_auc = score_for_es; no_improve=0
            to_save = (ema_helper.ema if (USE_EMA and ema_helper is not None) else model)
            torch.save(to_save.state_dict(), best_path)
            print(f"[Fold {fold_id}] BEST AUROC(img raw)={best_auc:.4f} salvo em {best_path}")
        else:
            no_improve += 1
            print(f"[Fold {fold_id}] Sem melhora ({no_improve}/{EARLYSTOP_PATIENCE}).")
        if not DRY_RUN and no_improve >= EARLYSTOP_PATIENCE:
            print(f"[Fold {fold_id}] Early stopping."); break

    # carrega best e gera OOF do fold
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state)
    if USE_EMA and ema_helper is not None: ema_helper.ema.load_state_dict(state)
    (va_loss, va_auc_raw_img, va_auc_gated_img, va_auc_raw_pat, va_auc_gated_pat,va_probs, va_labels, va_groups) = validate(ema_helper.ema if (USE_EMA and ema_helper is not None) else model,dl_va, criterion, device, return_preds=True)

    # thresholds por imagem (F1)
    thr_info = compute_best_thresholds(va_labels, va_probs, LABEL_COLUMNS, n_steps=201)
    thr_path = RUNS / f"thresholds_fold{fold_id}.json"
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(thr_info, f, indent=2)
    print(f"[Fold {fold_id}] Thresholds salvos: {thr_path}")
    return best_path

# =========================
# K-FOLD
# =========================
def kfold_train_and_ensemble():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type=='cuda': torch.backends.cudnn.benchmark=True
    labels_all = np.load(LBL_TRAIN_NPY)
    groups, meta = load_patient_groups_for_npy(INDEX_TRAIN_CSV, TRAIN1_CSV)
    assert len(groups) == labels_all.shape[0]
    with open(PATIENT_MAP_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[Groups] N={meta['n_samples']} | pacientes únicos={meta['n_patients']} | mapa salvo em {PATIENT_MAP_JSON}")
    fold_paths = []
    for fold_id, (tr_idx, va_idx) in enumerate(split_patientwise(groups, labels_all, K_FOLDS, SEED)):
        best_path = train_fold(fold_id, tr_idx, va_idx, labels_all, groups, device)
        fold_paths.append(best_path)
        if DRY_RUN:
            print("[DRY RUN] Encerrando após 1 fold."); break

    # ===== Inferência ENSEMBLE por imagem (padrão) =====
    test_names = pd.read_csv(TEST_CSV_PATH)['Image_name'].values  # ou TEST_CSV_PATH['Image_name']
    ds_te = NpyDatasetCHW(IMG_TEST_NPY, labels_path=None, to_rgb=True, normalize=True)
    # Subamostragem segura (mantém alinhamento dataset <-> test_names)
    if DRY_RUN:
        from torch.utils.data import Subset    
        n_total = len(ds_te)
        n_use = min(DRY_RUN_N, n_total)
        if DRY_RUN_RANDOM:
            rng = np.random.default_rng(SEED)
            idx = np.sort(rng.choice(n_total, size=n_use, replace=False))
        else:
            idx = np.arange(n_use, dtype=int)    
        ds_te = Subset(ds_te, idx)
        test_names = test_names[idx]
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,pin_memory=(device.type=='cuda'))

    logits_all_folds = []
    for fold_id, path in enumerate(fold_paths):
        print(f"[Ensemble] Fold {fold_id}: inferindo com {path}")
        model = create_model(num_classes=len(LABEL_COLUMNS), in_chans=3).to(device)
        state = torch.load(path, map_location=device); model.load_state_dict(state); model.eval()    
        fold_logits = []
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=(USE_AMP and device.type=='cuda')):
            for x in make_pbar(dl_te, f"Test fold {fold_id}"):
                if isinstance(x, (tuple, list)): x = x[0]
                x = x.to(device, non_blocking=True)
                logits = model(x)  # (B, C)
                fold_logits.append(logits.detach().cpu().float().numpy())
        logits_all_folds.append(np.vstack(fold_logits))  # (N, C) por fold    
    logits_all_folds = np.stack(logits_all_folds, axis=0)   # (K, N, C)
    logits_mean = logits_all_folds.mean(axis=0)             # (N, C)
    
    # Probabilidades = sigmoid(média dos logits)
    probs_mean = torch.sigmoid(torch.from_numpy(logits_mean)).numpy()  # (N, C)    
    assert probs_mean.shape[0] == len(test_names), \
        f"N de imagens divergente: probs={probs_mean.shape[0]} vs names={len(test_names)}"    
    sub_probs = pd.DataFrame({"Image_name": test_names})
    for i, col in enumerate(LABEL_COLUMNS):
        sub_probs[col] = probs_mean[:, i]    
    probs_path = Path("/kaggle/working/submission.csv")
    sub_probs.to_csv(probs_path, index=False)
    print("Submissão (sigmoid da MÉDIA dos LOGITS):", probs_path)

if __name__ == "__main__":
    kfold_train_and_ensemble()

