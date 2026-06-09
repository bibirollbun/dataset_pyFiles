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


# If your environment allows installs and RDKit isn't present, you may uncomment:
# !pip -q install rdkit-pypi lightgbm xgboost

import os, sys, gc, math, time, random, glob, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

# Optional boosters (handled gracefully if not installed)
try:
    import lightgbm as lgb
    LIGHTGBM_INSTALLED = True
except Exception:
    LIGHTGBM_INSTALLED = False

try:
    import xgboost as xgb
    XGBOOST_INSTALLED = True
except Exception:
    XGBOOST_INSTALLED = False

# Optional chemistry toolkit (RDKit)
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

TARGETS = ["Tg","FFV","Tc","Density","Rg"]


class CFG:
    n_splits = 5
    use_scaffold_cv = True
    fp_nbits = 2048
    fp_radius = 2
    add_rdkit_descriptors = True
    use_supplements = True
    verbose = 1
    max_train_rows_debug = None   # e.g., 2000 for quick debugging

    # Base models to include
    use_lgb = True
    use_xgb = True
    use_linear = True
    use_mlp = True

    # LightGBM
    lgb_params = dict(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=256,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1e-2,
        reg_lambda=1e-1,
        min_child_weight=1e-3,
        objective="mae",
        n_jobs=-1,
        verbose=-1
    )
    # XGBoost
    xgb_params = dict(
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1e-2,
        reg_lambda=1e-1,
        objective="reg:squarederror",
        n_jobs=-1
    )

    # Linear & MLP
    ridge_alphas = np.logspace(-3, 3, 13)
    mlp_hidden = (512, 256)
    mlp_alpha = 1e-4
    mlp_max_iter = 150
    mlp_batch_size = 256

cfg = CFG()

def find_file(patterns):
    candidates = []
    for root in [".", "/kaggle/input", "/kaggle/working"]:
        for pat in patterns:
            candidates.extend(glob.glob(os.path.join(root, "**", pat), recursive=True))
    if candidates:
        candidates = sorted(candidates, key=lambda p: (len(p.split(os.sep)), len(p)))
        return candidates[0]
    return None

TRAIN_PATH = find_file(["train.csv"])
TEST_PATH = find_file(["test.csv"])
SAMPLE_SUB_PATH = find_file(["sample_submission.csv"])
SUPP_DIR = find_file(["train_supplement"])
if SUPP_DIR and os.path.isfile(SUPP_DIR):
    SUPP_DIR = None

print("TRAIN_PATH:", TRAIN_PATH)
print("TEST_PATH:", TEST_PATH)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("SUPP_DIR:", SUPP_DIR)
print("RDKit available:", RDKIT_AVAILABLE)
print("LightGBM installed:", LIGHTGBM_INSTALLED)
print("XGBoost installed:", XGBOOST_INSTALLED)



assert TRAIN_PATH is not None, "train.csv not found"
assert TEST_PATH is not None, "test.csv not found"

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

if cfg.max_train_rows_debug is not None:
    train = train.sample(cfg.max_train_rows_debug, random_state=RANDOM_SEED).reset_index(drop=True)

for col in ["id","SMILES"]:
    assert col in train.columns, f"Train missing {col}"
    assert col in test.columns, f"Test missing {col}"

# Optional: read any supplemental CSVs in train_supplement/
supp_frames = []
if cfg.use_supplements and SUPP_DIR and os.path.isdir(SUPP_DIR):
    print("Loading supplements from:", SUPP_DIR)
    for fname in sorted(glob.glob(os.path.join(SUPP_DIR, "*.csv"))):
        try:
            df = pd.read_csv(fname)
            if "SMILES" not in df.columns:
                continue
            keep_cols = ["SMILES"] + [t for t in TARGETS if t in df.columns]
            df = df[keep_cols].copy()
            df["__source__"] = os.path.basename(fname)
            supp_frames.append(df)
            print(f"  - {os.path.basename(fname)} -> {df.shape}, cols={list(df.columns)}")
        except Exception as e:
            print("  ! Failed to load", fname, "->", e)

if supp_frames:
    supp = pd.concat(supp_frames, axis=0, ignore_index=True)
    supp = supp.drop_duplicates(subset=["SMILES"] + [c for c in TARGETS if c in supp.columns])
    # keep rows with at least one label
    has_any = np.zeros(len(supp), dtype=bool)
    for t in TARGETS:
        if t in supp.columns:
            has_any |= supp[t].notnull().values
    supp_labeled = supp[has_any].copy()
    print("Supplement (labeled) shape:", supp_labeled.shape)
else:
    supp = None
    supp_labeled = None

print("Train shape:", train.shape, " Test shape:", test.shape)
display(train.head(3))
display(test.head(3))


# --- PATCH: rebuild features with a single shared vectorizer ---

from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

def smiles_to_mol_safe(s):
    if not RDKIT_AVAILABLE:
        return None
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        Chem.SanitizeMol(m, catchErrors=True)
        return m
    except Exception:
        return None

# RDKit descriptor names (optional)
RD_DESCS = []
if RDKIT_AVAILABLE:
    RD_DESCS = [d[0] for d in Descriptors.descList]

def mol_2d_descriptors(mol):
    if not RDKIT_AVAILABLE or mol is None:
        return []
    out = []
    for name, fn in Descriptors.descList:
        try:
            out.append(fn(mol))
        except Exception:
            out.append(np.nan)
    return out

def morgan_fp(mol, nBits=2048, radius=2):
    if not RDKIT_AVAILABLE or mol is None:
        return np.zeros(nBits, dtype=np.uint8)
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        import numpy as _np
        arr = _np.zeros((nBits,), dtype=_np.uint8)
        from rdkit import DataStructs as _DataStructs
        _DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return np.zeros(nBits, dtype=np.uint8)

def smiles_basic_stats(s):
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    return {
        "len": len(s),
        "num_digits": sum(ch.isdigit() for ch in s),
        "num_upper": sum(ch.isupper() for ch in s),
        "num_lower": sum(ch.islower() for ch in s),
        "num_brackets": s.count("[") + s.count("]"),
        "num_branch": s.count("(") + s.count(")"),
        "num_ring": sum(s.count(str(d)) for d in range(10)),
        "num_hash": s.count("#"),
        "num_plus": s.count("+"),
        "num_minus": s.count("-"),
        "num_equals": s.count("="),
        "num_dot": s.count("."),
        "num_slash": s.count("/") + s.count("\\"),
        "num_arom_lower": sum(s.count(ch) for ch in "bcnops"),
        "num_arom_upper": sum(s.count(ch) for ch in "BCNOPS"),
    }

def build_features_joint(train_df, test_df, add_unlabeled_smiles=None):
    # 1) Fit CountVectorizer on combined corpus (train + test + unlabeled)
    train_smi = train_df["SMILES"].fillna("").astype(str)
    test_smi  = test_df["SMILES"].fillna("").astype(str)
    corpus = pd.concat([train_smi, test_smi], ignore_index=True)
    if add_unlabeled_smiles is not None and len(add_unlabeled_smiles) > 0:
        corpus = pd.concat([corpus, pd.Series(add_unlabeled_smiles, dtype=str)], ignore_index=True)

    cv = CountVectorizer(analyzer="char", ngram_range=(2,3), min_df=2, max_features=20000)
    cv.fit(corpus)

    # helper to featurize one dataframe
    def featurize(df):
        smiles = df["SMILES"].fillna("").astype(str)

        # SMILES stats
        stat_df = pd.DataFrame([smiles_basic_stats(s) for s in smiles]).astype(float)
        stat_mat = sparse.csr_matrix(stat_df.values)
        feat_names = list(stat_df.columns)

        # n-gram features (shared vocab)
        cv_mat = cv.transform(smiles)
        cv_names = [f"cv_{t}" for t in cv.get_feature_names_out()]

        mats = [stat_mat, cv_mat]
        feat_names += cv_names

        # RDKit descriptors + fingerprints (same length for train/test)
        if RDKIT_AVAILABLE:
            mols = [smiles_to_mol_safe(s) for s in smiles]
            if cfg.add_rdkit_descriptors:
                descs = [mol_2d_descriptors(m) for m in mols]
                rd_desc_mat = sparse.csr_matrix(np.nan_to_num(np.array(descs, dtype=float)))
                mats.append(rd_desc_mat)
                feat_names += [f"rd_{n}" for n in RD_DESCS]
            fps = [morgan_fp(m, nBits=cfg.fp_nbits, radius=cfg.fp_radius) for m in mols]
            rd_fp_mat = sparse.csr_matrix(np.array(fps, dtype=np.uint8))
            mats.append(rd_fp_mat)
            feat_names += [f"fp_{i}" for i in range(cfg.fp_nbits)]

        X = sparse.hstack(mats, format="csr")
        return X, feat_names

    X_tr, feat_names = featurize(train_df)
    X_te, _          = featurize(test_df)
    return X_tr, X_te, feat_names

# Prepare optional unlabeled smiles
unlabeled_smiles = None
if 'supp' in globals() and supp is not None and "SMILES" in supp.columns:
    tmp = supp.copy()
    for t in TARGETS:
        if t in tmp.columns:
            tmp[t] = tmp[t].notnull()
    labeled_mask = np.zeros(len(tmp), dtype=bool)
    for t in TARGETS:
        if t in supp.columns:
            labeled_mask |= tmp[t].values
    unlabeled_smiles = supp.loc[~labeled_mask, "SMILES"].dropna().unique().tolist()

# Rebuild features with shared CV
X_train, X_test, feat_names = build_features_joint(train, test, add_unlabeled_smiles=unlabeled_smiles)
print("Rebuilt features with shared vocabulary.")
print("X_train:", X_train.shape, "| X_test:", X_test.shape)



from sklearn.model_selection import KFold, GroupKFold

def generate_scaffold(smiles):
    if not RDKIT_AVAILABLE:
        return "NA"
    try:
        mol = smiles_to_mol_safe(smiles)
        if mol is None:
            return "NA"
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core, isomericSmiles=False)
    except Exception:
        return "NA"

def _scaffold_groups(df):
    """Return array of group labels (one per row) for scaffold CV."""
    if not cfg.use_scaffold_cv or not RDKIT_AVAILABLE:
        # either user disabled scaffold CV or RDKit missing
        return None
    scaffolds = df["SMILES"].fillna("").apply(generate_scaffold)
    return scaffolds.values

def get_cv_splits(df):
    """Robust CV splitter: prefer scaffold CV, gracefully fallback if needed."""
    groups = _scaffold_groups(df)
    if groups is not None:
        uniq = pd.Series(groups).nunique()
        if uniq >= 2:
            n_splits_eff = min(cfg.n_splits, uniq)
            if cfg.verbose:
                print(f"Using GroupKFold (scaffolds): {uniq} groups, n_splits={n_splits_eff}")
            gkf = GroupKFold(n_splits=n_splits_eff)
            return list(gkf.split(df, groups=groups, y=None))
        else:
            if cfg.verbose:
                print(f"Scaffold CV fallback: only {uniq} group(s). Switching to regular KFold.")
    # Regular KFold fallback
    kf = KFold(n_splits=min(cfg.n_splits, max(2, len(df)//5)), shuffle=True, random_state=RANDOM_SEED)
    return list(kf.split(df))

CV_SPLITS = get_cv_splits(train)
print("Prepared CV splits:", len(CV_SPLITS))



def calc_weights(y_valid_df):
    """w_k ∝ (1/sqrt(n_k)) / R_k, normalized so sum(w_k)=T."""
    weights, ranges, counts = {}, {}, {}
    eps = 1e-9
    T = len(TARGETS)
    raw = []
    for t in TARGETS:
        if t not in y_valid_df.columns:
            weights[t] = 0.0
            continue
        y = y_valid_df[t].dropna().values
        n_k = max(len(y), 1)
        R_k = max(np.nanpercentile(y, 99.5) - np.nanpercentile(y, 0.5), eps)
        counts[t] = n_k
        ranges[t] = R_k
        raw.append((t, (1.0 / (np.sqrt(n_k) + eps)) / (R_k + eps)))
    s = sum(w for _, w in raw) + eps
    for t, w in raw:
        weights[t] = w * T / s
    return weights, ranges, counts

def wmae_score(y_true_df, y_pred_df):
    weights, _, _ = calc_weights(y_true_df)
    total = 0.0
    for t in TARGETS:
        if t not in y_true_df.columns or t not in y_pred_df.columns:
            continue
        mask = y_true_df[t].notnull().values
        if mask.sum() == 0:
            continue
        mae = mean_absolute_error(y_true_df.loc[mask, t], y_pred_df.loc[mask, t])
        total += weights[t] * mae
    return total



# === Base learner definitions (all in one place) ===
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
from scipy import sparse as sp

def _to_csr64(X):
    """Ensure CSR float64 for sparse matrices; pass-through for dense."""
    if sp.isspmatrix(X):
        if not isinstance(X, sp.csr_matrix):
            X = X.tocsr()
        if X.dtype != np.float64:
            X = X.astype(np.float64)
        return X
    return X.astype(np.float64) if isinstance(X, np.ndarray) and X.dtype != np.float64 else X

# --- LightGBM ---
def fit_lgb(X_tr, y_tr, X_va, y_va):
    if not LIGHTGBM_INSTALLED or not cfg.use_lgb:
        return None, np.zeros(X_va.shape[0])
    model = lgb.LGBMRegressor(**cfg.lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
    )
    preds = model.predict(X_va, num_iteration=getattr(model, "best_iteration_", None))
    return model, preds

# --- XGBoost ---
def fit_xgb(X_tr, y_tr, X_va, y_va):
    if not XGBOOST_INSTALLED or not cfg.use_xgb:
        return None, np.zeros(X_va.shape[0])
    model = xgb.XGBRegressor(**cfg.xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, early_stopping_rounds=200)
    preds = model.predict(X_va)
    return model, preds

# --- Linear (sparse-safe SGD L2 instead of RidgeCV) ---
def fit_linear_ridge(X_tr, y_tr, X_va, y_va=None):
    if not cfg.use_linear:
        return None, np.zeros(X_va.shape[0], dtype=float)

    X_tr = _to_csr64(X_tr)
    X_va = _to_csr64(X_va)

    scaler = StandardScaler(with_mean=False)
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    alpha_grid = np.logspace(-6, -2, 9)
    best_mae, best_model, best_pred = np.inf, None, None

    for alpha in alpha_grid:
        model = SGDRegressor(
            loss="squared_error", penalty="l2", alpha=alpha,
            max_iter=3000, tol=1e-4, learning_rate="invscaling", eta0=0.01,
            random_state=RANDOM_SEED
        )
        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_va_s)
        if y_va is None:
            best_model, best_pred = model, preds
            break
        mae = mean_absolute_error(y_va, preds)
        if mae < best_mae:
            best_mae, best_model, best_pred = mae, model, preds

    return (scaler, best_model), best_pred

def predict_linear(pair, X):
    scaler, model = pair
    X = _to_csr64(X) if sp.isspmatrix(X) or isinstance(X, np.ndarray) else X
    return model.predict(scaler.transform(X))

# --- MLP ---
def fit_mlp(X_tr, y_tr, X_va):
    if not cfg.use_mlp:
        return None, np.zeros(X_va.shape[0])
    X_tr_d = X_tr.toarray() if sp.isspmatrix(X_tr) else X_tr
    X_va_d = X_va.toarray() if sp.isspmatrix(X_va) else X_va
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr_s = scaler.fit_transform(X_tr_d)
    X_va_s = scaler.transform(X_va_d)
    model = MLPRegressor(
        hidden_layer_sizes=cfg.mlp_hidden,
        alpha=cfg.mlp_alpha,
        random_state=RANDOM_SEED,
        max_iter=cfg.mlp_max_iter,
        batch_size=cfg.mlp_batch_size,
        early_stopping=True, n_iter_no_change=20, verbose=False
    )
    model.fit(X_tr_s, y_tr)
    preds = model.predict(X_va_s)
    return (scaler, model), preds

def predict_mlp(pair, X):
    scaler, model = pair
    Xd = X.toarray() if sp.isspmatrix(X) else X
    return model.predict(scaler.transform(Xd))

# --- Train one target (with stacking) ---
def train_one_target(y_name, X_tr, X_te, df, cv_splits):
    y_all = df[y_name].values if y_name in df.columns else np.full(len(df), np.nan)
    oof = np.full(len(df), np.nan, dtype=float)
    fold_mae, fold_models, test_meta_preds = [], [], []

    X_te_csr = _to_csr64(X_te) if sp.isspmatrix(X_te) or isinstance(X_te, np.ndarray) else X_te

    for fold, (tr_idx, va_idx) in enumerate(cv_splits, 1):
        tr_mask = ~np.isnan(y_all[tr_idx])
        va_mask = ~np.isnan(y_all[va_idx])
        tr_idx2, va_idx2 = tr_idx[tr_mask], va_idx[va_mask]
        if len(tr_idx2) == 0 or len(va_idx2) == 0:
            if cfg.verbose:
                print(f"[{y_name}] Fold {fold}: no labels, skip.")
            fold_models.append(None)
            continue

        X_tr_f, X_va_f = X_tr[tr_idx2], X_tr[va_idx2]
        y_tr_f, y_va_f = y_all[tr_idx2], y_all[va_idx2]
        X_tr_f, X_va_f = _to_csr64(X_tr_f), _to_csr64(X_va_f)

        base_va_preds, base_te_preds = [], []
        models_this_fold = {}

        # LightGBM
        if LIGHTGBM_INSTALLED and cfg.use_lgb:
            m, p_va = fit_lgb(X_tr_f, y_tr_f, X_va_f, y_va_f)
            p_te = m.predict(X_te_csr, num_iteration=getattr(m, "best_iteration_", None))
            models_this_fold["lgb"] = m
            base_va_preds.append(p_va); base_te_preds.append(p_te)

        # XGBoost
        if XGBOOST_INSTALLED and cfg.use_xgb:
            m, p_va = fit_xgb(X_tr_f, y_tr_f, X_va_f, y_va_f)
            p_te = m.predict(X_te_csr)
            models_this_fold["xgb"] = m
            base_va_preds.append(p_va); base_te_preds.append(p_te)

        # Linear
        if cfg.use_linear:
            m, p_va = fit_linear_ridge(X_tr_f, y_tr_f, X_va_f, y_va=y_va_f)
            p_te = predict_linear(m, X_te_csr)
            models_this_fold["sgd_l2"] = m
            base_va_preds.append(p_va); base_te_preds.append(p_te)

        # MLP
        if cfg.use_mlp:
            m, p_va = fit_mlp(X_tr_f, y_tr_f, X_va_f)
            p_te = predict_mlp(m, X_te_csr)
            models_this_fold["mlp"] = m
            base_va_preds.append(p_va); base_te_preds.append(p_te)

        # Stack predictions with Ridge meta-learner
        P_va = np.vstack(base_va_preds).T
        P_te = np.vstack(base_te_preds).T

        meta = RidgeCV(alphas=cfg.ridge_alphas, fit_intercept=True, cv=5)
        meta.fit(P_va, y_va_f)
        oof_pred = meta.predict(P_va)
        test_meta_preds.append(meta.predict(P_te))
        models_this_fold["meta"] = meta

        oof[va_idx2] = oof_pred
        mae = mean_absolute_error(y_va_f, oof_pred)
        fold_mae.append(mae)
        if cfg.verbose:
            print(f"[{y_name}] Fold {fold} MAE: {mae:.5f}")

        fold_models.append(models_this_fold)

    pred_test = np.vstack(test_meta_preds).mean(axis=0) if test_meta_preds else np.zeros(X_te_csr.shape[0])
    return {"oof": oof, "pred_test": pred_test, "fold_mae": fold_mae, "fold_models": fold_models}



oof_df = pd.DataFrame(index=train.index)
preds_test = pd.DataFrame(index=test.index)

for t in TARGETS:
    if t not in train.columns:
        print(f"Skipping {t}: not in train.")
        continue
    res = train_one_target(t, X_train, X_test, train, CV_SPLITS)
    oof_df[t] = res["oof"]
    preds_test[t] = res["pred_test"]
    print(f"[{t}] Mean fold MAE: {np.nanmean(res['fold_mae']):.5f}")

cv_wmae = wmae_score(train[TARGETS], oof_df[TARGETS])
print(f"\nOverall CV wMAE (approx): {cv_wmae:.6f}")



# sub = pd.DataFrame({"id": test["id"].values})
# for t in TARGETS:
#     sub[t] = preds_test[t].values if t in preds_test.columns else 0.0

# sub = sub[["id"] + TARGETS]
# sub.to_csv("submission.csv", index=False)
# print("Wrote submission.csv")
# display(sub.head())



import pandas as pd

# Build submission DataFrame from test ids and predictions
required_cols = ["id","Tg","FFV","Tc","Density","Rg"]

sub = pd.DataFrame({"id": test["id"].values})
for c in required_cols[1:]:
    if c in preds_test.columns:
        sub[c] = preds_test[c].values
    else:
        sub[c] = 0.0  # fallback in case a target wasn't predicted

# 1) Ensure column order and types
sub = sub[required_cols].copy()
for c in required_cols:
    if c == "id":
        sub[c] = pd.to_numeric(sub[c], errors="raise", downcast="integer")
    else:
        sub[c] = pd.to_numeric(sub[c], errors="raise")  # floats are fine

# 2) Basic sanity checks
assert list(sub.columns) == required_cols, "Wrong column order/header."
assert sub.isna().sum().sum() == 0, "NaNs found in submission."
assert sub["id"].is_unique, "Duplicate ids in submission."

# 3) (Optional) round to fixed decimals
for c in ["Tg","FFV","Tc","Density","Rg"]:
    sub[c] = sub[c].astype(float)
    # sub[c] = sub[c].round(6)  # uncomment if you want consistent precision

# 4) Save correctly: commas, header, NO index
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv")
print(sub.head(3))



per_target_mae = {}
for t in TARGETS:
    if t in train.columns:
        mask = train[t].notnull() & oof_df[t].notnull()
        per_target_mae[t] = mean_absolute_error(train.loc[mask, t], oof_df.loc[mask, t]) if mask.sum()>0 else np.nan

weights, ranges, counts = calc_weights(train[TARGETS])

diag = pd.DataFrame({
    "MAE": pd.Series(per_target_mae),
    "count(n)": pd.Series(counts),
    "range(robust)": pd.Series(ranges),
    "weight(w_k)": pd.Series(weights),
}).loc[TARGETS]

print(diag)
diag.to_csv("cv_diagnostics.csv")
print("Wrote cv_diagnostics.csv")





