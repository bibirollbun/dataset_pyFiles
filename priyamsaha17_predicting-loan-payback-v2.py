# Standard libs
import os
import math
import random
import time
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

# sklearn utils
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# torch imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# classical models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

# progress bar
from tqdm.auto import tqdm

print("torch:", torch.__version__, "lgb:", bool(lgb), "xgb:", bool(xgb), "catboost:", bool(CatBoostClassifier))


# Ensure reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Config identical to your provided snippet (paths may need to be changed to match your environment)
class Config:
    # file paths (adjust if needed)
    train_path = "/kaggle/input/playground-series-s5e11/train.csv"
    test_path = "/kaggle/input/playground-series-s5e11/test.csv"
    sample_path = "/kaggle/input/playground-series-s5e11/sample_submission.csv"
    orig_path = "/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv"

    # column names
    id_col = "id"
    target_col = "loan_paid_back"

    # training params
    n_splits = 4
    batch_size = 2048
    max_epochs = 30
    early_stopping_patience = 3
    lr = 1e-3
    weight_decay = 1e-4
    dropout = 0.1
    hidden_dim = 128
    residual_depth = 4
    seed = 42

    device = "cuda" if torch.cuda.is_available() else "cpu"

# Apply seed
set_seed(Config.seed)
print("Device:", Config.device)


def feature_engineering(df: pd.DataFrame, num_features: List[str], cat_features: List[str], orig: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    out = df.copy()

    global_mean = orig[Config.target_col].mean()
    global_count = 0.0

    # Mean and count encoding using original data
    for c in num_features + cat_features:
        if c not in orig.columns:
            out[f"{c}_org_mean"] = global_mean
            out[f"{c}_org_count"] = global_count
            continue

        col_mean = orig.groupby(c)[Config.target_col].agg("mean").rename(f"{c}_org_mean").reset_index()
        col_count = orig.groupby(c)[Config.target_col].agg("count").rename(f"{c}_org_count").reset_index()

        out = out.merge(col_mean, on=c, how="left")
        out[f"{c}_org_mean"] = out[f"{c}_org_mean"].fillna(global_mean)

        out = out.merge(col_count, on=c, how="left")
        out[f"{c}_org_count"] = out[f"{c}_org_count"].fillna(global_count)

    # Polynomial and log transforms
    for c in num_features:
        out[f"Log_{c}"] = np.log1p(np.maximum(out[c].astype(float).fillna(0.0), 0.0))
        out[f"{c}_sq"] = np.square(out[c].astype(float).fillna(0.0))
        out[f"{c}_cbrt"] = np.cbrt(out[c].astype(float).fillna(0.0))

    # Financial interaction features
    if all(col in out.columns for col in ['loan_amount', 'annual_income']):
        out['loan_income_ratio'] = out['loan_amount'] / (out['annual_income'] + 1)
    if all(col in out.columns for col in ['loan_amount', 'credit_score']):
        out['loan_credit_product'] = out['loan_amount'] * out['credit_score']
    if all(col in out.columns for col in ['debt_to_income_ratio', 'interest_rate']):
        out['debt_income_interest'] = out['debt_to_income_ratio'] * out['interest_rate']

    # Grade features if present
    if "grade_subgrade" in out.columns:
        try:
            out["grade_number"] = out["grade_subgrade"].astype(str).str[1].astype("float64")
        except Exception:
            out["grade_number"] = np.nan
        try:
            grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
            out["grade_rank"] = out["grade_subgrade"].astype(str).str[0].map(grade_map).astype("float64")
        except Exception:
            out["grade_rank"] = np.nan
        if "interest_rate" in out.columns:
            out["grade_risk"] = out["grade_rank"] * out["interest_rate"]

    # Ordinal mapping for domain-driven categoricals
    if "education_level" in out.columns:
        edu_map = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4, "Other": 0}
        out["education_level_ord"] = out["education_level"].map(edu_map).fillna(-1).astype(int)

    if "employment_status" in out.columns:
        emp_map = {"Unemployed": 0, "Student": 1, "Retired": 2, "Self-employed": 3, "Employed": 4}
        out["employment_status_ord"] = out["employment_status"].map(emp_map).fillna(-1).astype(int)

    # Factorize low-cardinality numerics into categories
    highcard = ["annual_income", "loan_amount"]
    lowcard = [c for c in num_features if c not in highcard]
    numtocat_features = []

    for c in lowcard:
        if c not in out.columns:
            continue
        codes, _ = pd.factorize(out[c].astype(str))
        out[f"{c}_cat"] = pd.Categorical(codes)
        numtocat_features.append(f"{c}_cat")

    for c in highcard:
        if c not in out.columns:
            continue
        out[f"{c}_round"] = pd.Categorical(out[c].round(0).astype(str))
        out[f"{c}_thousands"] = pd.Categorical(out[c].round(-3).astype(str))
        numtocat_features.extend([f"{c}_round", f"{c}_thousands"])

    # Cast provided categoricals
    for c in cat_features:
        if c in out.columns:
            out[c] = out[c].astype("category")

    # Combine all derived categorical-like features
    all_cats = numtocat_features + [c for c in cat_features if c in out.columns]

    # Frequency encoding
    for c in all_cats:
        freqs = out[c].value_counts(normalize=True)
        out[f"{c}_fe"] = out[c].map(freqs).astype("float64")

    # Updated feature lists
    updated_num = out.select_dtypes(exclude=["category", "object", "bool"]).columns.tolist()
    updated_cat = out.select_dtypes(include=["category"]).columns.tolist()

    return out, updated_num, updated_cat


# TabDataset (same as your snippet)
class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32) if y is not None else None
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx: int):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]

# ResidualBlock & MLPResNet (kept as provided)
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.Dropout(dropout)
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(x + self.block(x))

class MLPResNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, depth: int, dropout: float):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        blocks = []
        for _ in range(depth):
            blocks.append(ResidualBlock(hidden, dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        x = self.head(x)
        return x.squeeze(1)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str) -> float:
    model.train()
    running = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    return running / len(loader.dataset)

def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, np.ndarray]:
    model.eval()
    preds = []
    ys = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            prob = torch.sigmoid(logits)
            preds.append(prob.detach().cpu().numpy())
            ys.append(yb.detach().cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(ys)
    auc = roc_auc_score(y_true, y_pred)
    return auc, y_pred


def train_kfold_resnet(X: np.ndarray, y: np.ndarray, X_test: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Trains the ResNet using StratifiedKFold exactly as in the original snippet.
    Returns:
      - oof (preds for train)
      - test_pred (averaged over folds)
      - models_info: list of dict per fold with 'best_auc' and optionally model state
    """
    oof = np.zeros(X.shape[0], dtype=np.float32)
    test_pred = np.zeros(X_test.shape[0], dtype=np.float32)
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    models_info = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n===== Fold {fold+1}/{cfg.n_splits} =====")
        X_tr, X_val = X[trn_idx], X[val_idx]
        y_tr, y_val = y[trn_idx], y[val_idx]

        dtr = TabDataset(X_tr, y_tr)
        dval = TabDataset(X_val, y_val)
        dte = TabDataset(X_test, None)

        tr_loader = DataLoader(dtr, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(dval, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        te_loader = DataLoader(dte, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

        model = MLPResNet(in_dim=X.shape[1], hidden=cfg.hidden_dim, depth=cfg.residual_depth, dropout=cfg.dropout).to(cfg.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        best_auc = -np.inf
        patience = 0
        best_state = None

        for epoch in range(cfg.max_epochs):
            _ = train_one_epoch(model, tr_loader, optimizer, criterion, cfg.device)
            val_auc, val_pred = evaluate(model, val_loader, cfg.device)

            print(f"Epoch {epoch+1}: val AUC = {val_auc:.6f}")

            if val_auc > best_auc:
                best_auc = val_auc
                patience = 0
                # store best state (CPU)
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                oof[val_idx] = val_pred
            else:
                patience += 1

            if patience >= cfg.early_stopping_patience:
                print("Early stopping")
                break

        # restore best
        model.load_state_dict({k: v.to(cfg.device) for k, v in best_state.items()})
        model.eval()

        # predict test fold
        fold_preds = []
        with torch.no_grad():
            for xb in te_loader:
                xb = xb.to(cfg.device)
                prob = torch.sigmoid(model(xb)).detach().cpu().numpy()
                fold_preds.append(prob)
        if len(fold_preds) == 0:
            fold_preds = [np.zeros(X_test.shape[0], dtype=np.float32)]
        test_pred += np.concatenate(fold_preds) / cfg.n_splits

        print(f"Fold {fold+1} best val AUC: {best_auc:.6f}")
        models_info.append({"fold": fold, "best_auc": best_auc})
        # free memory
        del model, optimizer, criterion, tr_loader, val_loader
        torch.cuda.empty_cache()

    return oof, test_pred, models_info


# ===========================================
# Load dataframes (adjust paths if needed)
# ===========================================
train_df = pd.read_csv(Config.train_path, index_col=Config.id_col)
test_df  = pd.read_csv(Config.test_path, index_col=Config.id_col)
orig_df  = pd.read_csv(Config.orig_path)

# Separate target and features
y = train_df[Config.target_col].values.astype(np.float32)
X_base = train_df.drop(columns=[Config.target_col]).reset_index(drop=True)
T_base = test_df.reset_index(drop=True)

# Identify numeric and categorical features
num_features = X_base.select_dtypes(exclude=["object", "bool", "category"]).columns.tolist()
cat_features = X_base.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

# Concatenate for joint engineering
combined = pd.concat([X_base, T_base], axis=0, ignore_index=True)

# Perform the provided feature engineering
engineered, updated_num, updated_cat = feature_engineering(
    combined, num_features, cat_features, orig_df
)

# Split back
X_eng = engineered.iloc[:len(X_base)].reset_index(drop=True)
T_eng = engineered.iloc[len(X_base):].reset_index(drop=True)

# Select numeric features ONLY (network input)
used_numeric = X_eng.select_dtypes(exclude=["category", "object", "bool"]).columns.tolist()
print("Number of numeric features used:", len(used_numeric))

# ==========================================================
# Standard scaling (same output variable names as original)
# ==========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_eng[used_numeric].values)
T_scaled = scaler.transform(T_eng[used_numeric].values)

print("Shapes -> X_scaled:", X_scaled.shape, "T_scaled:", T_scaled.shape)

# ==========================================================
# ============ CONTRASTIVE LEARNING SECTION ================
# ==========================================================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

tf.random.set_seed(42)

# ---------- Encoder + Projection Head ----------
def make_encoder(input_dim, proj_dim=128, hidden=(256,128)):
    inp = keras.Input(shape=(input_dim,))
    x = inp
    for h in hidden:
        x = layers.Dense(h, activation="relu")(x)
        x = layers.BatchNormalization()(x)
    rep = layers.Dense(128, activation="relu", name="rep")(x)
    z   = layers.Dense(proj_dim, name="proj")(rep)
    return keras.Model(inp, [rep, z], name="encoder")

# ---------- Tabular Augmentations ----------
def augment(X, noise_std=0.01, mask_p=0.1):
    X1 = X + np.random.normal(scale=noise_std, size=X.shape)
    X2 = X + np.random.normal(scale=noise_std, size=X.shape)
    m1 = np.random.rand(*X.shape) < mask_p
    m2 = np.random.rand(*X.shape) < mask_p
    X1[m1] = 0
    X2[m2] = 0
    return X1.astype("float32"), X2.astype("float32")

# ---------- NT-Xent Loss ----------
class NTXent(tf.keras.losses.Loss):
    def __init__(self, temp=0.5):
        super().__init__()
        self.temp = temp
    def call(self, zis, zjs):
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)
        N = tf.shape(zis)[0]
        reps = tf.concat([zis, zjs], axis=0)
        sim  = tf.matmul(reps, reps, transpose_b=True) / self.temp
        # mask self-similarity
        mask = tf.eye(2*N)
        sim  = sim - 1e9 * mask
        labels = tf.concat([tf.range(N,2*N), tf.range(0,N)], axis=0)
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, sim, from_logits=True
            )
        )

# -----------------------------------------------------------
# Train contrastive encoder on scaled numeric features
# -----------------------------------------------------------
Xc = X_scaled.astype("float32")
input_dim = Xc.shape[1]
encoder = make_encoder(input_dim)

optimizer = keras.optimizers.Adam(1e-3)
loss_fn   = NTXent(temp=0.5)

batch = 512 if Xc.shape[0] > 15000 else 128
epochs = 10  # can increase to 20–50 for stronger embeddings
steps = max(1, Xc.shape[0] // batch)

for e in range(epochs):
    idx = np.random.permutation(Xc.shape[0])
    Xs  = Xc[idx]
    losses = []
    for s in range(steps):
        xb = Xs[s*batch:(s+1)*batch]
        x1, x2 = augment(xb)
        with tf.GradientTape() as tape:
            _, z1 = encoder(x1, training=True)
            _, z2 = encoder(x2, training=True)
            loss = loss_fn(z1, z2)
        grads = tape.gradient(loss, encoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, encoder.trainable_variables))
        losses.append(float(loss))
    print(f"Contrastive Epoch {e+1}/{epochs} - Loss {np.mean(losses):.4f}")

# -----------------------------------------------------------
# Extract embeddings and CONCATENATE to original features
# -----------------------------------------------------------
rep_model = keras.Model(encoder.input, encoder.get_layer("rep").output)

X_rep = rep_model.predict(X_scaled, batch_size=256)
T_rep = rep_model.predict(T_scaled, batch_size=256)

# CONCAT
X_scaled = np.concatenate([X_scaled, X_rep], axis=1)
T_scaled = np.concatenate([T_scaled, T_rep], axis=1)

print("After contrastive learning:")
print("X_scaled:", X_scaled.shape)
print("T_scaled:", T_scaled.shape)


# Train ResNet with K-Fold (using provided Config)
oof_resnet, test_resnet, resnet_models_info = train_kfold_resnet(X_scaled, y, T_scaled, Config)
print("ResNet OOF AUC:", roc_auc_score(y, oof_resnet))


# We will train classical models using the same StratifiedKFold splits to produce OOF preds.
# Important: use the SAME features used by the ResNet => used_numeric

def oof_train_classical(models: Dict[str, object], X_num: np.ndarray, y: np.ndarray, X_test_num: np.ndarray, cfg: Config):
    """
    models: dict of name -> sklearn-like estimator (must implement fit() and predict_proba())
    X_num: numpy array (n_samples, n_features)
    returns:
        results: dict with keys per model:
            'oof' : OOF predictions (proba)
            'test': averaged test predictions
            'auc' : OOF AUC
            'model_objs': list of fitted models per fold (optional)
    """
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    n = X_num.shape[0]
    results = {}
    for name, estimator in models.items():
        print(f"\nTraining model: {name}")
        oof = np.zeros(n, dtype=np.float32)
        test_pred = np.zeros(X_test_num.shape[0], dtype=np.float32)
        model_folds = []
        for fold, (trn_idx, val_idx) in enumerate(skf.split(X_num, y)):
            X_tr, X_val = X_num[trn_idx], X_num[val_idx]
            y_tr, y_val = y[trn_idx], y[val_idx]

            # clone estimator to avoid state sharing
            import copy
            clf = copy.deepcopy(estimator)
            # Fit
            clf.fit(X_tr, y_tr)
            # Predict proba (try predict_proba else decision_function)
            if hasattr(clf, "predict_proba"):
                valp = clf.predict_proba(X_val)[:,1]
                testp = clf.predict_proba(X_test_num)[:,1]
            else:
                # fallback
                try:
                    valp_scores = clf.decision_function(X_val)
                    testp_scores = clf.decision_function(X_test_num)
                    valp = 1/(1+np.exp(-valp_scores))
                    testp = 1/(1+np.exp(-testp_scores))
                except Exception:
                    valp = clf.predict(X_val)
                    testp = clf.predict(X_test_num)

            oof[val_idx] = valp
            test_pred += testp / cfg.n_splits
            model_folds.append(clf)
            val_auc = roc_auc_score(y_val, valp)
            print(f"  Fold {fold+1} val AUC: {val_auc:.6f}")

        overall_auc = roc_auc_score(y, oof)
        print(f"Model {name} OOF AUC: {overall_auc:.6f}")
        results[name] = {"oof": oof, "test": test_pred, "auc": overall_auc, "models": model_folds}
    return results

# Prepare models to train (you can adjust hyperparams)
models_to_run = {}

# Logistic Regression (on scaled inputs)
models_to_run['logreg'] = LogisticRegression(max_iter=1000, random_state=Config.seed)

# RandomForest (use reasonable settings)
models_to_run['rf'] = RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=Config.seed)

# ExtraTrees
models_to_run['et'] = ExtraTreesClassifier(n_estimators=200, n_jobs=2, random_state=Config.seed)

# HistGradientBoosting (sklearn)
models_to_run['hgb'] = HistGradientBoostingClassifier(random_state=Config.seed)

# LightGBM (if available) - use sklearn API LGBMClassifier
if lgb is not None:
    models_to_run['lgb'] = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, n_jobs=2, random_state=Config.seed)

# XGBoost (if available)
if xgb is not None:
    models_to_run['xgb'] = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', verbosity=0, tree_method='hist', random_state=Config.seed)

# CatBoost (if available)
if CatBoostClassifier is not None:
    # note: we did label-free feature_eng, we pass label-encoded inputs, so CatBoost will treat them as numeric
    models_to_run['catboost'] = CatBoostClassifier(iterations=500, learning_rate=0.05, random_seed=Config.seed, verbose=200)

# Call training for classical models
# For logistic regression we must use scaled input (X_scaled), for tree-based models raw numeric works fine.
# We'll pass X_scaled for LR and X_scaled for other models too (safe), because the features are numeric and scaled is acceptable.
X_num_all = X_scaled  # use the scaled numeric matrix for all classical models for consistency
T_num_all = T_scaled

classical_results = oof_train_classical(models_to_run, X_num_all, y, T_num_all, Config)


# Collect all results (include resnet)
all_results = {}

# ResNet
all_results['resnet'] = {"oof": oof_resnet, "test": test_resnet, "auc": roc_auc_score(y, oof_resnet)}

# Classical
for k,v in classical_results.items():
    all_results[k] = {"oof": v["oof"], "test": v["test"], "auc": v["auc"]}

# Sort by AUC
ranking = sorted([(k, v['auc']) for k,v in all_results.items()], key=lambda x: x[1], reverse=True)
print("Model ranking by OOF AUC:")
for name, auc in ranking:
    print(f"{name:10s}  AUC: {auc:.6f}")


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ---- Basic checks ----
if 'all_results' not in globals():
    raise RuntimeError("all_results dict not found. Ensure previous cells produced `all_results`.")
if 'y' not in globals():
    raise RuntimeError("True labels `y` not found. Ensure `y` (train targets) is available.")

model_names = sorted(all_results.keys())
print("Models available for ensembling:", model_names)

# ---- Build meta-level matrices ----
meta_oof = np.vstack([all_results[m]['oof'] for m in model_names]).T   # (n_train, n_models)
meta_test = np.vstack([all_results[m]['test'] for m in model_names]).T  # (n_test, n_models)
print("meta_oof shape:", meta_oof.shape, "meta_test shape:", meta_test.shape)

# ---- Meta-model training (CV stacking) ----
CFG_FOLDS = StratifiedKFold(n_splits=Config.n_splits, shuffle=True, random_state=Config.seed)

def train_meta(meta_X, y, meta_X_test, meta_model='logistic'):
    oof_meta = np.zeros(meta_X.shape[0], dtype=float)
    test_meta_folds = np.zeros((meta_X_test.shape[0], CFG_FOLDS.n_splits), dtype=float)
    fold_i = 0
    for tr_idx, val_idx in CFG_FOLDS.split(meta_X, y):
        X_tr_m, X_val_m = meta_X[tr_idx], meta_X[val_idx]
        y_tr_m, y_val_m = y[tr_idx], y[val_idx]

        if meta_model == 'logistic':
            clf = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=2000)
            clf.fit(X_tr_m, y_tr_m)
            val_pred = clf.predict_proba(X_val_m)[:,1]
            test_pred = clf.predict_proba(meta_X_test)[:,1]
        elif meta_model == 'lgb':
            if lgb is None:
                raise RuntimeError("lightgbm not available for meta model")
            clf = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, random_state=Config.seed)
            clf.fit(X_tr_m, y_tr_m, eval_set=[(X_val_m, y_val_m)], early_stopping_rounds=50, verbose=False)
            val_pred = clf.predict_proba(X_val_m)[:,1]
            test_pred = clf.predict_proba(meta_X_test)[:,1]
        else:
            raise ValueError("Unsupported meta_model: choose 'logistic' or 'lgb'")

        oof_meta[val_idx] = val_pred
        test_meta_folds[:, fold_i] = test_pred
        fold_i += 1

    test_meta = test_meta_folds.mean(axis=1)
    auc = roc_auc_score(y, oof_meta)
    return oof_meta, test_meta, auc

print("\nTraining logistic meta-learner...")
meta_oof_log, meta_test_log, meta_auc_log = train_meta(meta_oof, y, meta_test, meta_model='logistic')
print("Logistic meta OOF AUC:", meta_auc_log)

meta_oof_lgb, meta_test_lgb, meta_auc_lgb = None, None, -np.inf
if lgb is not None:
    try:
        print("Training LightGBM meta-learner...")
        meta_oof_lgb, meta_test_lgb, meta_auc_lgb = train_meta(meta_oof, y, meta_test, meta_model='lgb')
        print("LightGBM meta OOF AUC:", meta_auc_lgb)
    except Exception as e:
        print("LightGBM meta training failed:", e)

# ---- Greedy ensemble selection (Caruana-style) ----
from sklearn.metrics import roc_auc_score
def greedy_ensemble(oof_dict, y_true, max_iters=50, verbose=True):
    names = list(oof_dict.keys())
    preds = np.vstack([oof_dict[n] for n in names])  # shape (n_models, n_train)
    n_models, n_train = preds.shape
    ensemble = np.zeros(n_train, dtype=float)
    selected = []
    best_auc = 0.0
    for it in range(max_iters):
        best_local_auc = -1.0
        best_idx = None
        for i in range(n_models):
            # candidate ensemble if we add model i once more
            if len(selected) == 0:
                candidate = preds[i]
            else:
                candidate = (ensemble * len(selected) + preds[i]) / (len(selected) + 1)
            auc = roc_auc_score(y_true, candidate)
            if auc > best_local_auc:
                best_local_auc = auc
                best_idx = i
        if best_local_auc <= best_auc + 1e-12:
            if verbose:
                print(f"No improvement on iteration {it}, stopping. best_auc={best_auc:.6f}")
            break
        # accept best
        selected.append(names[best_idx])
        if len(selected) == 1:
            ensemble = preds[best_idx].copy()
        else:
            ensemble = (ensemble * (len(selected)-1) + preds[best_idx]) / len(selected)
        best_auc = best_local_auc
        if verbose:
            print(f"Iteration {it+1}: added {names[best_idx]} -> ensemble AUC {best_auc:.6f}")
    return selected, ensemble, best_auc

# Prepare oof dict from all_results
oof_dict = {n: all_results[n]['oof'] for n in model_names}
print("\nRunning greedy ensemble selection...")
selected_models, greedy_oof, greedy_auc = greedy_ensemble(oof_dict, y, max_iters=50, verbose=True)
print("Greedy selection finished. AUC:", greedy_auc)
print("Selected (in order):", selected_models)

# Derive test preds for greedy by averaging selected model test preds (with repetitions)
test_dict = {n: all_results[n]['test'] for n in model_names}
if len(selected_models) > 0:
    test_greedy = np.zeros_like(test_dict[selected_models[0]])
    for sel in selected_models:
        test_greedy += test_dict[sel]
    test_greedy /= len(selected_models)
else:
    test_greedy = np.zeros(meta_test.shape[0])

# ---- Compare & pick final ensemble ----
candidates = {
    'logistic_meta': (meta_oof_log, meta_test_log, meta_auc_log),
    'lgb_meta': (meta_oof_lgb, meta_test_lgb, meta_auc_lgb),
    'greedy': (greedy_oof, test_greedy, greedy_auc)
}

best_name = None
best_auc = -np.inf
best_oof = None
best_test = None
for k, (oof_arr, test_arr, auc_val) in candidates.items():
    if oof_arr is None:
        continue
    print(f"Candidate {k} -> AUC: {auc_val:.6f}")
    if auc_val > best_auc:
        best_auc = auc_val
        best_name = k
        best_oof = oof_arr
        best_test = test_arr

print(f"\nSelected final ensemble strategy: {best_name} with OOF AUC = {best_auc:.6f}")

# ---- Optionally save submission (if sample/test_df available) ----
try:
    if 'sample' in globals() and sample is not None:
        submission = sample.copy()
        submission[Config.target_col] = best_test
        out_file = "submission_advanced_ensemble.csv"
        submission.to_csv(out_file, index=False)
        print("Saved submission to", out_file)
    elif 'test_df' in globals():
        # save id + prediction
        sub = pd.DataFrame({ 'id': test_df.index, Config.target_col: best_test })
        out_file = "submission_advanced_ensemble.csv"
        sub.to_csv(out_file, index=False)
        print("Saved submission to", out_file)
    else:
        print("No sample/test_df available in workspace; skipping file save.")
except Exception as e:
    print("Failed to write submission:", e)

# ---- final reporting ----
print("Final chosen ensemble:", best_name)
print("Final OOF AUC:", best_auc)


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# ---- Build ranking of base models by AUC ----
ranking = sorted(
    [(name, all_results[name]['auc']) for name in all_results.keys()],
    key=lambda x: x[1],
    reverse=True
)

plt.figure(figsize=(9,7))

# ---- Plot top K base models ----
TOP_K = min(6, len(ranking))

for name, auc_val in ranking[:TOP_K]:
    fpr, tpr, _ = roc_curve(y, all_results[name]['oof'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.4f})", alpha=0.8)

# ---- Plot final chosen ensemble ----
fpr, tpr, _ = roc_curve(y, best_oof)
plt.plot(
    fpr, tpr,
    label=f"{best_name} (AUC={roc_auc_score(y, best_oof):.4f})",
    color='black',
    linewidth=2.5
)

# ---- Plot chance line ----
plt.plot([0,1], [0,1], '--', color='gray')

# ---- Labels and formatting ----
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Base Models vs Final Ensemble")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10, loc='lower right')

plt.show()

