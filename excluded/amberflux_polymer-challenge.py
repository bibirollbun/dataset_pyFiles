# ==== CELL 0: Setup & global flags ====
import os, numpy as np, pandas as pd

# Competition paths (unchanged)
COMP_DIR = "/kaggle/input/neurips-open-polymer-prediction-2025"

# GPU usage hint (XGB will auto-use if True)
USE_GPU = True

# Targets (kept the same pattern you used)
TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]

# Optional weights for reporting (not used in scoring, just for your local print)
W = {t: 1.0 for t in TARGETS}

# Feature cache to speed reruns
FEAT_CACHE = "/kaggle/working/feat_cache"
os.makedirs(FEAT_CACHE, exist_ok=True)

print("Setup ok. COMP_DIR:", COMP_DIR)



import warnings, os
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"



# ==== CELL 1: Load data (robust, unchanged structure) ====
import pandas as pd, numpy as np, os

train_path = f"{COMP_DIR}/train.csv"
test_path  = f"{COMP_DIR}/test.csv"
sample_path= f"{COMP_DIR}/sample_submission.csv"

assert os.path.exists(train_path), "train.csv not found"
assert os.path.exists(test_path),  "test.csv not found"

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)
sample_df = pd.read_csv(sample_path)

# Normalize SMILES / id casing if needed
if "SMILES" not in train.columns:
    for c in train.columns:
        if "smile" in c.lower(): train.rename(columns={c:"SMILES"}, inplace=True); break
if "SMILES" not in test.columns:
    for c in test.columns:
        if "smile" in c.lower(): test.rename(columns={c:"SMILES"}, inplace=True); break

if "id" not in test.columns:
    # create safe id if the comp's test.csv lacks it
    test.insert(0, "id", np.arange(len(test)))

# Verify targets exist in train (they will be partially NaN; we mask later)
for t in TARGETS:
    assert t in train.columns, f"Target missing in train: {t}"

print("Train:", train.shape, "| Test:", test.shape)
print("Train columns:", list(train.columns)[:20])



# ==== CELL 2: Engineered features (counts, composition, autocorr, FFT) + cache ====
import re, numpy as np, pandas as pd
from pathlib import Path

def basic_smiles_stats(s):
    # Counts of common atoms/fragments; vectorized via str.count
    feats = {}
    feats["smiles_len"] = len(s)
    # single-char counts first to avoid double-counting multi-char later
    for ch in ["C","O","N","F","P","S","B","I"]:
        feats[f"cnt_{ch}"] = s.count(ch)
    # two-char halogens explicitly
    for frag in ["Cl","Br"]:
        feats[f"cnt_{frag}"] = len(re.findall(frag, s))
    # ring/branch markers
    feats["cnt_open_paren"] = s.count("(")
    feats["cnt_close_paren"] = s.count(")")
    feats["cnt_equal"] = s.count("=")
    feats["cnt_hash"] = s.count("#")
    feats["cnt_digits"] = sum(ch.isdigit() for ch in s)
    # ratios
    L = max(feats["smiles_len"], 1)
    feats.update({f"ratio_{k}": v / L for k, v in feats.items() if k.startswith("cnt_")})
    return feats

# Discrete autocorrelation over a simple token stream (character-level)
def acf_counts(s, max_lag=6):
    arr = np.frombuffer(s.encode("utf-8"), dtype=np.uint8)  # quick & deterministic
    if arr.size == 0:
        return {f"acf_lag_{k}": 0.0 for k in range(1, max_lag+1)}
    out = {}
    mu = arr.mean()
    var = np.var(arr)
    for k in range(1, max_lag+1):
        if arr.size - k <= 0 or var == 0:
            out[f"acf_lag_{k}"] = 0.0
        else:
            out[f"acf_lag_{k}"] = float(np.dot(arr[:-k]-mu, arr[k:]-mu) / ((arr.size - k) * var + 1e-12))
    return out

# FFT energy over the same byte stream (periodicity proxy)
def fft_energy(s, bands=(2,4,8,16)):
    arr = np.frombuffer(s.encode("utf-8"), dtype=np.uint8).astype(np.float32)
    if arr.size == 0:
        return {f"fft_band_{b}": 0.0 for b in bands}
    spec = np.fft.rfft(arr - arr.mean())
    power = (spec.real**2 + spec.imag**2)
    # split cumulative energy at band edges
    out = {}
    L = power.size
    for b in bands:
        idx = min(int(L * (1.0 / b)), L-1) if L>1 else 0
        out[f"fft_band_{b}"] = float(power[:idx+1].sum() / (power.sum() + 1e-12))
    return out

def featurize_df(df):
    rows = []
    for s in df["SMILES"].fillna("").astype(str).values:
        base = basic_smiles_stats(s)
        base.update(acf_counts(s, max_lag=6))
        base.update(fft_energy(s, bands=(2,4,8,16)))
        rows.append(base)
    F = pd.DataFrame(rows, index=df.index)
    # clean any inf/nan
    F = F.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return F

# Cache-aware
tr_cache = Path(FEAT_CACHE)/"X_train_base.parquet"
te_cache = Path(FEAT_CACHE)/"X_test_base.parquet"

if tr_cache.exists() and te_cache.exists():
    X_train = pd.read_parquet(tr_cache)
    X_testF = pd.read_parquet(te_cache)
    print("Loaded base features from cache:", X_train.shape, X_testF.shape)
else:
    X_train = featurize_df(train)
    X_testF = featurize_df(test)
    X_train.to_parquet(tr_cache)
    X_testF.to_parquet(te_cache)
    print("Built base features:", X_train.shape, X_testF.shape)




# ==== CELL 2.5: TF-IDF (char 2–4) + TruncatedSVD embeddings ====
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

_smiles_tr = train["SMILES"].fillna("").astype(str).values
_smiles_te = test["SMILES"].fillna("").astype(str).values

VEC = TfidfVectorizer(analyzer="char", ngram_range=(2,4), min_df=2,
                      max_features=75000, dtype=np.float32)
X_tfidf_tr = VEC.fit_transform(_smiles_tr)
X_tfidf_te = VEC.transform(_smiles_te)
print("TF-IDF shapes:", X_tfidf_tr.shape, X_tfidf_te.shape)

SVD_K = 128
svd = TruncatedSVD(n_components=SVD_K, random_state=1337)
SV_tr = svd.fit_transform(X_tfidf_tr).astype(np.float32)
SV_te = svd.transform(X_tfidf_te).astype(np.float32)

svd_cols = [f"svd_{i:03d}" for i in range(SVD_K)]
SV_tr_df = pd.DataFrame(SV_tr, columns=svd_cols, index=train.index)
SV_te_df = pd.DataFrame(SV_te, columns=svd_cols, index=test.index)

# Augment engineered features with SVD embeddings (keep names)
X_train = pd.concat([X_train.reset_index(drop=True), SV_tr_df.reset_index(drop=True)], axis=1)
X_testF = pd.concat([X_testF.reset_index(drop=True), SV_te_df.reset_index(drop=True)], axis=1)
print("Augmented features:", X_train.shape, X_testF.shape)



# ==== CELL 2.6 (optional): Lean RDKit descriptors (auto-skip if RDKit missing) ====
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    def rdkit_feats(smiles):
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return dict(
                rd_mw=0, rd_tpsa=0, rd_logp=0, rd_hba=0, rd_hbd=0,
                rd_rotb=0, rd_ring=0, rd_fusedring=0, rd_heavy=0
            )
        return dict(
            rd_mw=Descriptors.MolWt(m),
            rd_tpsa=Descriptors.TPSA(m),
            rd_logp=Descriptors.MolLogP(m),
            rd_hba=rdMolDescriptors.CalcNumHBA(m),
            rd_hbd=rdMolDescriptors.CalcNumHBD(m),
            rd_rotb=rdMolDescriptors.CalcNumRotatableBonds(m),
            rd_ring=rdMolDescriptors.CalcNumRings(m),
            rd_fusedring=rdMolDescriptors.CalcNumSpiroAtoms(m) + rdMolDescriptors.CalcNumBridgeheadAtoms(m),
            rd_heavy=m.GetNumHeavyAtoms(),
        )

    def featurize_rdkit(df):
        rows = []
        for s in df["SMILES"].fillna("").astype(str).values:
            rows.append(rdkit_feats(s))
        F = pd.DataFrame(rows, index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return F

    rd_tr = featurize_rdkit(train)
    rd_te = featurize_rdkit(test)
    X_train = pd.concat([X_train, rd_tr], axis=1)
    X_testF = pd.concat([X_testF, rd_te], axis=1)
    print("RDKit descriptors added:", rd_tr.shape[1], "columns")
except Exception as e:
    print("RDKit not available or failed; skipping RDKit descriptors. Reason:", str(e))



# ==== CELL 3: Training — LGBM + XGB (GPU/CPU safe) + MLP + TF-IDF KNN, with stacking ====
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb, xgboost as xgb

# Config
FOLDS   = 5
SEEDS   = [42, 1337, 2025]
KNN_K   = 35
LGB_NE  = 1200
XGB_NE  = 1200
MLP_HID = (96,)

def weighted_mae(per_target_mae: dict, weights: dict) -> float:
    num = den = 0.0
    for k, v in per_target_mae.items():
        if v is None or (isinstance(v, float) and np.isnan(v)): 
            continue
        w = float(weights.get(k, 1.0))
        num += w * float(v); den += w
    return num/den if den>0 else np.nan

# ---- LightGBM trainer ----
def train_lgb_once(X_tr, y_tr, X_va, y_va, X_te, seed=1337):
    model = lgb.LGBMRegressor(
        objective="mae",
        boosting_type="goss",
        top_rate=0.2, other_rate=0.1,
        n_estimators=LGB_NE,
        learning_rate=0.035,
        num_leaves=96,
        max_bin=127,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              eval_metric="l1",
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    return model.predict(X_va), model.predict(X_te)

# ---- XGBoost trainer (with GPU → CPU fallback) ----
def train_xgb_once(X_tr, y_tr, X_va, y_va, X_te, seed=1337):
    def _make(tree_method, predictor):
        return xgb.XGBRegressor(
            objective="reg:absoluteerror",
            n_estimators=XGB_NE,
            max_depth=8,
            learning_rate=0.035,
            subsample=0.8,
            colsample_bytree=0.8,
            max_bin=256,
            reg_alpha=1.0,
            reg_lambda=2.0,
            tree_method=tree_method,
            predictor=predictor,
            random_state=seed,
            n_jobs=-1,
            eval_metric="mae",
            early_stopping_rounds=100,
            verbosity=0
        )
    # Try GPU first
    try:
        model = _make("gpu_hist", "gpu_predictor")
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    except Exception as e:
        print("⚠️ XGB GPU unavailable, falling back to CPU:", str(e).splitlines()[0])
        model = _make("hist", "auto")
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    return model.predict(X_va), model.predict(X_te)

# ---- MLP trainer ----
def train_mlp_once(X_tr, y_tr, X_va, y_va, X_te, seed=1337):
    scaler = StandardScaler()
    X_trs, X_vas, X_tes = scaler.fit_transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)
    mlp = MLPRegressor(hidden_layer_sizes=MLP_HID,
                       activation="relu",
                       solver="adam",
                       alpha=1e-4,
                       learning_rate_init=1e-3,
                       max_iter=400,
                       random_state=seed,
                       early_stopping=True,
                       n_iter_no_change=20,
                       verbose=False)
    mlp.fit(X_trs, y_tr)
    return mlp.predict(X_vas), mlp.predict(X_tes)

# ---- TF-IDF KNN helper ----
def knn_neighbor_preds(T_tr, y_tr, T_va, T_te, k=35):
    n_neighbors = min(k, len(y_tr))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(T_tr)
    d_va, i_va = nn.kneighbors(T_va); w_va = 1.0 - d_va
    val_pred  = (w_va * y_tr[i_va]).sum(axis=1) / (w_va.sum(axis=1) + 1e-8)
    d_te, i_te = nn.kneighbors(T_te); w_te = 1.0 - d_te
    test_pred = (w_te * y_tr[i_te]).sum(axis=1) / (w_te.sum(axis=1) + 1e-8)
    return val_pred.astype(float), test_pred.astype(float)

# ---- Main CV loop ----
n_test = len(test)
preds = np.zeros((n_test, len(TARGETS)), dtype=float)
per_target_mae = {}
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

for j, t in enumerate(TARGETS):
    print(f"\nTraining target: {t}")
    y_all = pd.to_numeric(train[t], errors="coerce").values
    idx = np.where(~np.isnan(y_all))[0]
    if idx.size < 20:
        per_target_mae[t] = np.nan
        preds[:, j] = np.nan
        print(f"[{t}] skipped (too few labels)")
        continue

    X_sub = X_train.iloc[idx].reset_index(drop=True)
    y_sub = y_all[idx].astype(float)
    X_te  = X_testF.copy()

    # drop constant columns
    keep_cols = X_sub.nunique()[lambda s: s > 1].index
    X_sub, X_te = X_sub[keep_cols], X_te.reindex(columns=keep_cols, fill_value=0)

    folds = list(kf.split(X_sub))

    # KNN on TF-IDF
    T_full, T_te = X_tfidf_tr[idx], X_tfidf_te
    oof_k = np.full(len(X_sub), np.nan); p_k = np.zeros(n_test)
    for tr, va in folds:
        pv_k, pt_k = knn_neighbor_preds(T_full[tr], y_sub[tr], T_full[va], T_te, k=KNN_K)
        oof_k[va] = pv_k; p_k += pt_k / FOLDS

    # Seed loop for LGB/XGB/MLP
    oof_l_sum, p_l_sum = np.zeros(len(X_sub)), np.zeros(n_test)
    oof_x_sum, p_x_sum = np.zeros(len(X_sub)), np.zeros(n_test)
    oof_m_sum, p_m_sum = np.zeros(len(X_sub)), np.zeros(n_test)

    for seed in SEEDS:
        oof_l, p_l = np.full(len(X_sub), np.nan), np.zeros(n_test)
        oof_x, p_x = np.full(len(X_sub), np.nan), np.zeros(n_test)
        oof_m, p_m = np.full(len(X_sub), np.nan), np.zeros(n_test)

        for tr, va in folds:
            X_tr, X_va = X_sub.iloc[tr], X_sub.iloc[va]
            y_tr, y_va = y_sub[tr], y_sub[va]

            pv_l, pt_l = train_lgb_once(X_tr, y_tr, X_va, y_va, X_te, seed=seed)
            pv_x, pt_x = train_xgb_once(X_tr, y_tr, X_va, y_va, X_te, seed=seed)
            pv_m, pt_m = train_mlp_once(X_tr.values, y_tr, X_va.values, y_va, X_te.values, seed=seed)

            oof_l[va], oof_x[va], oof_m[va] = pv_l, pv_x, pv_m
            p_l += pt_l / FOLDS; p_x += pt_x / FOLDS; p_m += pt_m / FOLDS

        oof_l_sum += np.nan_to_num(oof_l, nan=0.0); p_l_sum += p_l
        oof_x_sum += np.nan_to_num(oof_x, nan=0.0); p_x_sum += p_x
        oof_m_sum += np.nan_to_num(oof_m, nan=0.0); p_m_sum += p_m

    # Average across seeds
    nS = float(len(SEEDS))
    oof_l_avg, p_l_avg = oof_l_sum/nS, p_l_sum/nS
    oof_x_avg, p_x_avg = oof_x_sum/nS, p_x_sum/nS
    oof_m_avg, p_m_avg = oof_m_sum/nS, p_m_sum/nS

    # Stack with Ridge
    oof_stack = np.vstack([oof_l_avg, oof_x_avg, oof_m_avg, oof_k]).T
    ok = np.isfinite(oof_stack).all(axis=1) & np.isfinite(y_sub)
    meta = Ridge(alpha=1e-3)
    meta.fit(oof_stack[ok], y_sub[ok])
    preds[:, j] = meta.predict(np.vstack([p_l_avg, p_x_avg, p_m_avg, p_k]).T)

    per_target_mae[t] = mean_absolute_error(y_sub[ok], meta.predict(oof_stack[ok]))
    print(f"  {t} OOF MAE: {per_target_mae[t]:.5f}")

print("\nPer-target OOF MAE:", per_target_mae)
print("Macro avg MAE:", np.mean(list(per_target_mae.values())))



# --- Build submission DataFrame ---
sub = pd.DataFrame({"id": test["id"].values})
for i, t in enumerate(TARGETS):
    col = preds[:, i]
    # replace NaN or inf with 0.0 fallback
    col = np.where(np.isfinite(col), col, 0.0)
    sub[t] = col

print("\nSubmission preview:\n", sub.head())



# ==== CELL 4: Build & validate submission (keeps your schema) ====
import os, numpy as np, pandas as pd

required = sample_df.columns.tolist()  # ['id','Tg','FFV','Tc','Density','Rg']

# Construct submission frame
sub = pd.DataFrame({"id": test["id"].values})
for i, t in enumerate(TARGETS):
    col = preds[:, i] if i < preds.shape[1] else np.full(len(test), 0.0)
    if not np.isfinite(col).any():
        med = float(pd.to_numeric(train[t], errors="coerce").median(skipna=True)) if t in train.columns else 0.0
        col = np.full(len(test), med)
    else:
        med = float(np.nanmedian(col))
        col = np.where(np.isfinite(col), col, med)
    sub[t] = col.astype(float)

# Align schema and ids
for c in required:
    if c not in sub.columns:
        sub[c] = np.nan
sub = sub[required].copy()

test_df = pd.read_csv(f"{COMP_DIR}/test.csv")
sub = sub[sub["id"].isin(test_df["id"])]
missing = set(test_df["id"]) - set(sub["id"])
if missing:
    sub = pd.concat([sub, pd.DataFrame({"id": sorted(missing)})], ignore_index=True)
sub = sub.sort_values("id").reset_index(drop=True)

# Final numeric cleanup
for c in required:
    if c == "id": continue
    s = pd.to_numeric(sub[c], errors="coerce")
    sub[c] = (s.fillna(float(s.median(skipna=True))) if not s.isna().all() else 0.0).astype(float)
    assert np.isfinite(sub[c]).all(), f"Non-finite values in {c}"

# Save
os.makedirs("/kaggle/working", exist_ok=True)
out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("Saved:", out_path, "shape:", sub.shape)
print(sub.head())



# ==== CELL 5: Sanity checks ====
print("NaN counts per column in submission:")
print(sub.isna().sum())

# Ensure required columns present and in order
required = ["id","Tg","FFV","Tc","Density","Rg"]
missing = [c for c in required if c not in sub.columns]
assert not missing, f"Missing required columns: {missing}"

# Kaggle preview
sub.head()



# ==== SAFE METRICS PRINT (guarded) ====
import numpy as np
import pandas as pd

def _to_safe_float(x):
    try:
        if x is None:
            return None
        x = float(x)
        if np.isnan(x):
            return None
        return x
    except Exception:
        return None

try:
    # oof_mae may have numpy floats / NaNs, convert to JSON-safe dict
    safe_mae = {str(k): _to_safe_float(v) for k, v in (oof_mae if isinstance(oof_mae, dict) else {}).items()}
    print("Per-target OOF MAE (safe):", safe_mae)
except Exception as e:
    print("Per-target OOF MAE print skipped due to:", repr(e))



# ==== FINAL CELL: write only, no heavy prints ====
import os
os.makedirs("/kaggle/working", exist_ok=True)
sub_path = "/kaggle/working/submission.csv"
sub.to_csv(sub_path, index=False)
print("✅ Saved:", sub_path, "shape:", sub.shape)


