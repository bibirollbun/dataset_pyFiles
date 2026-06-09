# ==== Config & Setup ====
import os
import random
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Tuple, Dict, List

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

@dataclass
class Config:
    BASE_PATH: str = "/kaggle/input/playground-series-s5e11"
    TRAIN_FILE: str = "train.csv"
    TEST_FILE: str = "test.csv"
    TARGET: str = "loan_paid_back"
    N_SPLITS: int = 5
    SEED: int = 42
    EPOCHS: int = 30
    PATIENCE: int = 6
    BATCH_SIZE: int = 1024
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-5
    DROPOUT: float = 0.25
    EMA_DECAY: float = 0.99
    # Previous MLP-specific param left for compatibility; not used by FTTransformer
    N_BLOCKS: int = 10
    # FT-Transformer hyperparameters
    DIM: int = 128
    DEPTH: int = 4
    HEADS: int = 8
    DIM_HEAD: int = 16
    ATTN_DROPOUT: float = 0.1
    FF_DROPOUT: float = 0.1
CFG = Config()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_df = pd.read_csv(f"{CFG.BASE_PATH}/{CFG.TRAIN_FILE}")
test_df = pd.read_csv(f"{CFG.BASE_PATH}/{CFG.TEST_FILE}")

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG.SEED)
print(f"Using device: {DEVICE}")


# ==== Feature extraction (single function) ====
from typing import Optional


def _stratified_kfold():
    return StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=CFG.SEED)

def _mean_encode_oof(train_cat: pd.Series, y: np.ndarray, test_cat: pd.Series, smoothing: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    # Out-of-fold target mean encoding with smoothing to reduce leakage
    y = y.astype(float)
    global_mean = y.mean()
    oof = np.zeros(len(train_cat), dtype=float)
    skf = _stratified_kfold()
    
    for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
        tr_c = train_cat.iloc[tr_idx].reset_index(drop=True)  # Reset index to align with tr_y
        tr_y = y[tr_idx]
        counts = tr_c.value_counts()
        means = tr_c.groupby(tr_c).apply(lambda s: tr_y[s.index].mean())
        means = means.reindex(counts.index)
        smooth = (means * counts + smoothing * global_mean) / (counts + smoothing)
        mapping = smooth.to_dict()
        oof[va_idx] = train_cat.iloc[va_idx].map(mapping).fillna(global_mean).values
    
    # Fit on full train for test mapping
    counts_full = train_cat.value_counts()
    means_full = train_cat.groupby(train_cat).apply(lambda s: y[s.index].mean())
    means_full = means_full.reindex(counts_full.index)
    smooth_full = (means_full * counts_full + smoothing * global_mean) / (counts_full + smoothing)
    mapping_full = smooth_full.to_dict()
    test_enc = test_cat.map(mapping_full).fillna(global_mean).values
    
    return oof.astype(np.float32), test_enc.astype(np.float32)

def features_extraction(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    # Split columns
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    cat_cols = [c for c in train_df.columns if c not in numeric_cols + [target]]

    # Copy and basic NA handling
    tr = train_df.copy()
    te = test_df.copy()

    # Fill numeric
    medians = tr[numeric_cols].median(numeric_only=True)
    tr[numeric_cols] = tr[numeric_cols].fillna(medians)
    te[numeric_cols] = te[numeric_cols].fillna(medians)

    # Fill categoricals
    for c in cat_cols:
        tr[c] = tr[c].astype("category").cat.add_categories(["Unknown"]).fillna("Unknown")
        te[c] = te[c].astype("category").cat.add_categories(["Unknown"]).fillna("Unknown")

    y = tr[target].values.astype(np.float32)

    # Requested important-features-related engineering
    # 1) employment_status target mean and count encodings
    if "employment_status" in cat_cols:
        emp_oof_mean, emp_test_mean = _mean_encode_oof(tr["employment_status"], y, te["employment_status"], smoothing=20.0)
        tr["mean_employment_status"] = emp_oof_mean
        te["mean_employment_status"] = emp_test_mean
        emp_count = tr["employment_status"].value_counts()
        tr["count_employment_status"] = tr["employment_status"].map(emp_count).fillna(0).astype(np.float32)
        te["count_employment_status"] = te["employment_status"].map(emp_count).fillna(0).astype(np.float32)
        # Combined mean*count
        tr["orig_mean_employment_status_orig_count_employment_status"] = tr["mean_employment_status"] * tr["count_employment_status"]
        te["orig_mean_employment_status_orig_count_employment_status"] = te["mean_employment_status"] * te["count_employment_status"]
    else:
        tr["mean_employment_status"] = 0.0
        te["mean_employment_status"] = 0.0
        tr["count_employment_status"] = 0.0
        te["count_employment_status"] = 0.0
        tr["orig_mean_employment_status_orig_count_employment_status"] = 0.0
        te["orig_mean_employment_status_orig_count_employment_status"] = 0.0

    # 2) grade_subgrade target mean (for numeric interaction with debt_to_income_ratio)
    if "grade_subgrade" in cat_cols:
        gs_oof_mean, gs_test_mean = _mean_encode_oof(tr["grade_subgrade"], y, te["grade_subgrade"], smoothing=20.0)
        tr["mean_grade_subgrade"] = gs_oof_mean
        te["mean_grade_subgrade"] = gs_test_mean
    else:
        tr["mean_grade_subgrade"] = 0.0
        te["mean_grade_subgrade"] = 0.0

    # 3) Interactions requested
    # numeric x categorical (using mean encodings for numeric interaction)
    if "debt_to_income_ratio" in numeric_cols:
        tr["debt_to_income_ratio-employment_status"] = tr["debt_to_income_ratio"].astype(np.float32) * tr["mean_employment_status"].astype(np.float32)
        te["debt_to_income_ratio-employment_status"] = te["debt_to_income_ratio"].astype(np.float32) * te["mean_employment_status"].astype(np.float32)
        tr["debt_to_income_ratio-grade_subgrade"] = tr["debt_to_income_ratio"].astype(np.float32) * tr["mean_grade_subgrade"].astype(np.float32)
        te["debt_to_income_ratio-grade_subgrade"] = te["debt_to_income_ratio"].astype(np.float32) * te["mean_grade_subgrade"].astype(np.float32)
    else:
        tr["debt_to_income_ratio-employment_status"] = 0.0
        te["debt_to_income_ratio-employment_status"] = 0.0
        tr["debt_to_income_ratio-grade_subgrade"] = 0.0
        te["debt_to_income_ratio-grade_subgrade"] = 0.0

    # categorical crosses
    def _cross(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a.astype(str) + "|" + b.astype(str)).astype("category")

    if set(["gender", "employment_status"]).issubset(cat_cols):
        tr["gender-employment_status"] = _cross(tr["gender"], tr["employment_status"])  # requested name
        te["gender-employment_status"] = _cross(te["gender"], te["employment_status"])  # requested name
    if set(["employment_status", "grade_subgrade"]).issubset(cat_cols):
        tr["employment_status-grade_subgrade"] = _cross(tr["employment_status"], tr["grade_subgrade"])  # requested name
        te["employment_status-grade_subgrade"] = _cross(te["employment_status"], te["grade_subgrade"])  # requested name

    # Prepare final feature lists
    extra_numeric = [
        "mean_employment_status",
        "count_employment_status",
        "orig_mean_employment_status_orig_count_employment_status",
        "debt_to_income_ratio-employment_status",
        "debt_to_income_ratio-grade_subgrade",
        "mean_grade_subgrade",
    ]
    final_numeric = [c for c in numeric_cols if c != target] + extra_numeric

    extra_cats = []
    if "gender-employment_status" in tr.columns:
        extra_cats.append("gender-employment_status")
    if "employment_status-grade_subgrade" in tr.columns:
        extra_cats.append("employment_status-grade_subgrade")

    final_cats = cat_cols + extra_cats
    final_cats = list(dict.fromkeys(final_cats))  # dedupe preserving order

    # Encoders for OHE path (kept for reference / comparison)
    ohe_kwargs = {"handle_unknown": "ignore"}
    try:
        ohe = OneHotEncoder(sparse_output=False, **ohe_kwargs)
    except TypeError:
        ohe = OneHotEncoder(sparse=False, **ohe_kwargs)
    scaler = StandardScaler()

    # Fit encoders on train
    ohe.fit(tr[final_cats])
    scaler.fit(tr[final_numeric])

    # Transform (OHE path)
    X_train = np.hstack([
        scaler.transform(tr[final_numeric]) if len(final_numeric) else np.empty((len(tr), 0)),
        ohe.transform(tr[final_cats]) if len(final_cats) else np.empty((len(tr), 0)),
    ]).astype(np.float32)
    X_test = np.hstack([
        scaler.transform(te[final_numeric]) if len(final_numeric) else np.empty((len(te), 0)),
        ohe.transform(te[final_cats]) if len(final_cats) else np.empty((len(te), 0)),
    ]).astype(np.float32)

    feature_names = [f"num::{c}" for c in final_numeric]
    feature_names += [f"cat::{c}" for c in ohe.get_feature_names_out(final_cats).tolist()]

    return X_train, X_test, y, feature_names





# ==== FT-Transformer Model for Tabular Data ====
# We adapt core components from ft_transformer.py
from einops import rearrange, repeat

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

def FeedForward(dim, mult=4, dropout=0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)

class TransformerLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)
        self.ff = FeedForward(dim, dropout=ff_dropout)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x

class FTTransformer(nn.Module):
    def __init__(self, categories: List[int], num_continuous: int, cfg: Config, dim_out: int = 1):
        super().__init__()
        self.num_categories = len(categories)
        self.num_continuous = num_continuous
        self.dim = cfg.DIM
        self.depth = cfg.DEPTH
        self.heads = cfg.HEADS
        self.dim_head = cfg.DIM_HEAD

        # Embeddings for categorical
        total_tokens = sum(categories)
        self.register_buffer('categories_offset', torch.tensor([0] + list(np.cumsum(categories)[:-1])))
        self.categorical_embeds = nn.Embedding(total_tokens, self.dim)

        # Numerical embedding: simple linear projection
        if self.num_continuous > 0:
            self.numerical_proj = nn.Linear(self.num_continuous, self.dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(self.dim, self.heads, self.dim_head, cfg.ATTN_DROPOUT, cfg.FF_DROPOUT)
            for _ in range(self.depth)
        ])

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, dim_out)
        )

    def forward(self, x_categ: torch.Tensor, x_numer: torch.Tensor):
        # x_categ shape: (batch, num_categories) with raw category indices per column
        # offset each column's indices into joint embedding space
        if self.num_categories > 0:
            x_categ = x_categ + self.categories_offset.to(x_categ.device)
            cat_embed = self.categorical_embeds(x_categ)  # (b, num_cat, dim)
        else:
            cat_embed = torch.empty(x_categ.size(0), 0, self.dim, device=x_categ.device)

        if self.num_continuous > 0:
            num_embed = self.numerical_proj(x_numer).unsqueeze(1)  # (b,1,dim)
        else:
            num_embed = torch.empty(x_categ.size(0), 0, self.dim, device=x_categ.device)

        tokens = torch.cat([cat_embed, num_embed], dim=1)  # (b, num_cat + maybe 1, dim)
        b = tokens.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        for layer in self.layers:
            x = layer(x)
        cls_final = x[:, 0]
        return self.head(cls_final)  # (b, 1)

print("FTTransformer class defined.")


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def prepare_transformer_inputs(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str):
    """Prepare categorical index tensors and continuous tensors for transformer.
    We reuse the engineered features but split raw columns.
    Returns:
        categ_train, categ_test, numer_train, numer_test, y, categories_cardinalities, num_cont_cols
    """
    # Identify numeric and categorical base columns (excluding target)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    cat_cols = [c for c in train_df.columns if c not in numeric_cols + [target]]

    tr = train_df.copy()
    te = test_df.copy()

    # Fill missing values
    medians = tr[numeric_cols].median(numeric_only=True)
    tr[numeric_cols] = tr[numeric_cols].fillna(medians)
    te[numeric_cols] = te[numeric_cols].fillna(medians)
    for c in cat_cols:
        tr[c] = tr[c].astype('category').cat.add_categories(['Unknown']).fillna('Unknown')
        te[c] = te[c].astype('category').cat.add_categories(['Unknown']).fillna('Unknown')

    y = tr[target].values.astype(np.float32)

    # Category mapping to indices per column
    categories_cardinalities = []
    categ_train_cols = []
    categ_test_cols = []
    for c in cat_cols:
        tr_cat = tr[c].cat.codes.astype(np.int64)
        te_cat = te[c].cat.codes.astype(np.int64)
        categories_cardinalities.append(int(tr[c].nunique()))
        categ_train_cols.append(tr_cat)
        categ_test_cols.append(te_cat)

    if len(categ_train_cols):
        categ_train = np.vstack(categ_train_cols).T  # (n_samples, n_cat)
        categ_test = np.vstack(categ_test_cols).T
    else:
        categ_train = np.zeros((len(tr), 0), dtype=np.int64)
        categ_test = np.zeros((len(te), 0), dtype=np.int64)

    # Numerical: use numeric_cols directly (standardize)
    scaler = StandardScaler()
    numer_train = scaler.fit_transform(tr[numeric_cols]).astype(np.float32)
    numer_test = scaler.transform(te[numeric_cols]).astype(np.float32)

    return categ_train, categ_test, numer_train, numer_test, y, categories_cardinalities, numeric_cols


def train_one_fold_transformer(
    fold: int,
    categ_all: np.ndarray,
    numer_all: np.ndarray,
    y_all: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    categ_test: np.ndarray,
    numer_test: np.ndarray,
    categories_cardinalities: List[int],
    cfg: Config,
    device: torch.device,
):
    Xc_tr, Xc_val = categ_all[train_idx], categ_all[val_idx]
    Xn_tr, Xn_val = numer_all[train_idx], numer_all[val_idx]
    y_tr, y_val = y_all[train_idx], y_all[val_idx]

    # Datasets
    train_ds = TensorDataset(
        torch.from_numpy(Xc_tr).long(),
        torch.from_numpy(Xn_tr).float(),
        torch.from_numpy(y_tr.reshape(-1, 1)).float()
    )
    val_ds = TensorDataset(
        torch.from_numpy(Xc_val).long(),
        torch.from_numpy(Xn_val).float(),
        torch.from_numpy(y_val.reshape(-1, 1)).float()
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4)

    model = FTTransformer(categories=categories_cardinalities, num_continuous=numer_all.shape[1], cfg=cfg, dim_out=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    ema = EMA(model, decay=cfg.EMA_DECAY) if cfg.EMA_DECAY > 0 else None

    best_auc = -np.inf
    epochs_no_improve = 0
    best_state = None

    print(f"\n===== Fold {fold} / {cfg.N_SPLITS} (Transformer) =====")
    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xc, xn, yb in train_loader:
            xc = xc.to(device)
            xn = xn.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xc, xn)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)
            train_loss += loss.item() * xc.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        if ema is not None:
            ema.apply_shadow(model)
        val_logits_list = []
        with torch.no_grad():
            for xc, xn, yb in val_loader:
                xc = xc.to(device)
                xn = xn.to(device)
                logits = model(xc, xn)
                val_logits_list.append(logits.cpu().numpy())
        val_logits = np.vstack(val_logits_list).ravel()
        val_probs = 1 / (1 + np.exp(-val_logits))
        val_auc = roc_auc_score(y_val, val_probs)
        if ema is not None:
            ema.restore(model)

        msg = f"Epoch {epoch:02d}: train_loss={train_loss:.4f} val_auc={val_auc:.5f}"
        if val_auc > best_auc + 1e-6:
            best_auc = val_auc
            epochs_no_improve = 0
            if ema is not None:
                ema.apply_shadow(model)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if ema is not None:
                ema.restore(model)
            msg += " [best]"
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.PATIENCE:
                print("Early stopping.")
                break
        print(msg)

    print(f"Best AUC fold {fold}: {best_auc:.5f}")

    # Load best state
    model.load_state_dict(best_state)
    model.eval()

    # OOF predictions
    val_logits_list = []
    with torch.no_grad():
        for xc, xn, yb in val_loader:
            xc = xc.to(device)
            xn = xn.to(device)
            logits = model(xc, xn)
            val_logits_list.append(logits.cpu().numpy())
    val_logits = np.vstack(val_logits_list).ravel()
    oof_probs = 1 / (1 + np.exp(-val_logits))

    # Test predictions
    test_ds = TensorDataset(
        torch.from_numpy(categ_test).long(),
        torch.from_numpy(numer_test).float()
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_logits_list = []
    with torch.no_grad():
        for xc, xn in test_loader:
            xc = xc.to(device)
            xn = xn.to(device)
            logits = model(xc, xn)
            test_logits_list.append(logits.cpu().numpy())
    test_logits = np.vstack(test_logits_list).ravel()
    test_probs = 1 / (1 + np.exp(-test_logits))

    return oof_probs.astype(np.float32), test_probs.astype(np.float32), float(best_auc)


# Build OHE features once (for reference)
X_all, X_test, y_all, feature_names = features_extraction(train_df, test_df, CFG.TARGET)
print(f"X_all: {X_all.shape}, X_test: {X_test.shape}, n_features: {len(feature_names)}")

# Build Transformer-friendly inputs
Xc_all, Xc_test, Xn_all, Xn_test, y_all_tf, categories_cardinalities, num_cont_cols = prepare_transformer_inputs(train_df, test_df, CFG.TARGET)
# Overwrite y_all to ensure training uses the same target vector
y_all = y_all_tf
print(f"Transformer inputs -> Xc_all: {Xc_all.shape}, Xn_all: {Xn_all.shape}; Xc_test: {Xc_test.shape}, Xn_test: {Xn_test.shape}")
print(f"Categories per feature: {categories_cardinalities}")


# ==== 5-Fold Training Loop (FT-Transformer) ====
from tqdm.auto import tqdm

skf = StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=CFG.SEED)

fold_aucs = []
oof_preds = np.zeros(len(Xc_all), dtype=np.float32)
test_preds_folds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(Xc_all, y_all), start=1):
    oof_probs_fold, test_probs_fold, best_auc = train_one_fold_transformer(
        fold=fold,
        categ_all=Xc_all,
        numer_all=Xn_all,
        y_all=y_all,
        train_idx=train_idx,
        val_idx=val_idx,
        categ_test=Xc_test,
        numer_test=Xn_test,
        categories_cardinalities=categories_cardinalities,
        cfg=CFG,
        device=DEVICE,
    )
    oof_preds[val_idx] = oof_probs_fold
    test_preds_folds.append(test_probs_fold)
    fold_aucs.append(best_auc)

cv_auc = roc_auc_score(y_all, oof_preds)
print('Per fold AUCs:', fold_aucs)
print(f"Mean fold AUC: {np.mean(fold_aucs):.5f}")
print(f"OOF AUC: {cv_auc:.5f}")



# Average test predictions
test_preds = np.mean(np.vstack(test_preds_folds), axis=0)


# ==== Make submission ====
submission = pd.DataFrame({
    "id": test_df["id"].values,
    "loan_paid_back": test_preds.astype(np.float32),
})
submission_path = "submission.csv"  # saved in /kaggle/working during Kaggle runs
submission.to_csv(submission_path, index=False)
print(f"Saved submission to: {submission_path}")
submission.head()


submission.shape

