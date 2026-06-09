# ==============================================================
# Setup
# ==============================================================
!pip install -q lightgbm catboost xgboost

import os, gc, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# ==============================================================
# Config
# ==============================================================
DATA_DIR = "/kaggle/input/playground-series-s5e10"
TARGET = "accident_risk"
ID = "id"
SEED = 42
VAL_SIZE = 0.2         # 80/20 split
N_BAGS = 3             # bagging per base model
CORR_DROP_THR = 0.995  # buang fitur sangat kolinear
TOP_VAR = 8000         # early variance filter (jaga RAM)
POLY_DEG = 4           # polynomial degree (aman utk RAM)

np.random.seed(SEED)

# ==============================================================
# Load
# ==============================================================
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

X = train.drop([TARGET, ID], axis=1)
y = train[TARGET].astype(np.float32)
X_test_raw = test.drop([ID], axis=1)

categorical_features = ["road_type", "lighting", "weather", "time_of_day"]
binary_features      = ["road_signs_present", "public_road", "holiday", "school_season"]
numerical_features   = ["num_lanes", "curvature", "speed_limit", "num_reported_accidents"]

# ==============================================================
# Train/Validation split (tanpa kebocoran)
# ==============================================================
X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(
    X, y, test_size=VAL_SIZE, random_state=SEED,
    stratify=pd.qcut(y, 10, duplicates="drop")
)
print(f"Train: {len(X_tr_raw)}, Valid: {len(X_va_raw)}, Test: {len(X_test_raw)}")

# ==============================================================
# Target Encoding & Binning (fit di TRAIN saja → anti-leak)
# ==============================================================
def fit_te(train_df, y_tr, cols):
    mappings = {}
    global_mean = float(y_tr.mean())
    for c in cols:
        s = train_df[c].astype(str)
        means = y_tr.groupby(s).mean()
        mappings[c] = (means, global_mean)
    return mappings

def apply_te(df, mappings):
    out = pd.DataFrame(index=df.index)
    for c, (means, gmean) in mappings.items():
        out[f"{c}_te"] = df[c].astype(str).map(means).fillna(gmean).astype(np.float32)
    return out

def fit_quantile_bins(series, q=50):
    edges = np.unique(np.nanquantile(series.astype(float), np.linspace(0,1,q+1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges

def apply_bins(series, edges):
    return pd.cut(series.astype(float), bins=edges, labels=False, include_lowest=True)

# ==============================================================
# Feature Engineering (banyak tapi RAM-safe)
# ==============================================================
def fe_transform(df_raw, te_map=None, bin_edges=None, is_train=False):
    df = df_raw.copy()

    # Base numerik
    for c in numerical_features:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    # Frequency encoding kategori + biner (tanpa target)
    for c in categorical_features + binary_features:
        freq = df[c].value_counts(normalize=True)
        df[f"{c}_freq"] = df[c].map(freq).astype(np.float32)

    # Target Encoding (pakai mapping dari TRAIN)
    if te_map is not None:
        te_df = apply_te(df, te_map)
        df = pd.concat([df, te_df], axis=1)

    # Binning pakai edges dari TRAIN
    if bin_edges is not None:
        for col in ["curvature", "speed_limit"]:
            try:
                df[f"{col}_bin"] = apply_bins(df[col], bin_edges[col]).astype(np.float32)
            except Exception:
                pass

    # Transform dasar numeric
    for col in numerical_features:
        df[f"log1p_{col}"] = np.log1p(np.clip(df[col], a_min=0, a_max=None)).astype(np.float32)
        df[f"sqrt_{col}"]  = np.sqrt(np.abs(df[col])).astype(np.float32)

    # Rasio & selisih antar numeric
    for i in range(len(numerical_features)):
        for j in range(i+1, len(numerical_features)):
            a, b = numerical_features[i], numerical_features[j]
            df[f"{a}_div_{b}"]   = (df[a] / (df[b].replace(0, np.nan) + 1e-6)).astype(np.float32)
            df[f"{a}_minus_{b}"] = (df[a] - df[b]).astype(np.float32)

    # Interaksi kategori-encoding (freq/te) × numerik
    enc_cols = [f"{c}_freq" for c in categorical_features] + [f"{c}_te" for c in categorical_features if f"{c}_te" in df.columns]
    for e in enc_cols:
        if e in df.columns:
            for num in numerical_features:
                df[f"{e}_x_{num}"] = (df[e] * df[num]).astype(np.float32)

    # Interaksi antar encoding kategori (freq & te)
    enc_cols_present = [c for c in enc_cols if c in df.columns]
    for i in range(len(enc_cols_present)):
        for j in range(i+1, len(enc_cols_present)):
            a, b = enc_cols_present[i], enc_cols_present[j]
            df[f"{a}_x_{b}"] = (df[a] * df[b]).astype(np.float32)

    # Polynomial pada basis ringan (numerik + 4 enc kolom) agar RAM aman
    poly_base = numerical_features + [c for c in enc_cols_present][:4]
    poly = PolynomialFeatures(degree=POLY_DEG, include_bias=False)
    try:
        poly_mat = poly.fit_transform(df[poly_base])
        poly_cols = [f"poly_{i}" for i in range(poly_mat.shape[1])]
        poly_df = pd.DataFrame(poly_mat, columns=poly_cols, index=df.index).astype(np.float32)
        df = pd.concat([df, poly_df], axis=1)
    except Exception:
        pass

    # Random projections moderat (300)
    rng = np.random.RandomState(123 if is_train else 321)
    rp = rng.normal(size=(len(df), 300)).astype(np.float32)
    rp_df = pd.DataFrame(rp, columns=[f"rp_{i}" for i in range(300)], index=df.index)
    df = pd.concat([df, rp_df], axis=1)

    # Bersih & numerik only
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.select_dtypes(include=[np.number]).astype(np.float32)

    gc.collect()
    return df

# ----- FIT objek dari TRAIN (anti-leak) -----
te_map = fit_te(X_tr_raw, y_tr, categorical_features)
bin_edges = {}
for c in ["curvature", "speed_limit"]:
    bin_edges[c] = fit_quantile_bins(X_tr_raw[c], q=50)

# ----- Apply FE -----
X_tr_fe = fe_transform(X_tr_raw, te_map, bin_edges, is_train=True)
X_va_fe = fe_transform(X_va_raw, te_map, bin_edges, is_train=False)
X_te_fe = fe_transform(X_test_raw, te_map, bin_edges, is_train=False)

print("Shapes FE  | Train:", X_tr_fe.shape, "| Valid:", X_va_fe.shape, "| Test:", X_te_fe.shape)

# ==============================================================
# Seleksi fitur berlapis + buang multikolinearitas
# ==============================================================
def drop_constant(df):
    keep = [c for c in df.columns if df[c].std(ddof=0) > 0]
    return df[keep]

def variance_filter(df, topk=TOP_VAR):
    var = df.var().sort_values(ascending=False)
    keep = var.head(min(topk, len(var))).index
    return df[keep]

def drop_multicollinear(df, thr=CORR_DROP_THR):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > thr)]
    return df.drop(columns=to_drop), to_drop

# Step 1: drop constant
X_tr_sel = drop_constant(X_tr_fe)
X_va_sel = X_va_fe[X_tr_sel.columns].copy()
X_te_sel = X_te_fe[X_tr_sel.columns].copy()

# Step 2: variance filter
X_tr_sel = variance_filter(X_tr_sel, topk=TOP_VAR)
X_va_sel = X_va_sel[X_tr_sel.columns].copy()
X_te_sel = X_te_sel[X_tr_sel.columns].copy()

# Step 3: drop multicollinear
X_tr_final, dropped = drop_multicollinear(X_tr_sel, thr=CORR_DROP_THR)
X_va_final = X_va_sel[X_tr_final.columns].copy()
X_te_final = X_te_sel[X_tr_final.columns].copy()

print("Selected features:", X_tr_final.shape[1], "| Dropped MC cols:", len(dropped))

# ==============================================================
# GPU/CPU fallback helpers
# ==============================================================
def fit_lgbm_gpu_cpu(Xtr, ytr, Xva, yva, seed):
    params = dict(
        objective="regression",
        learning_rate=0.03,
        n_estimators=6000,
        num_leaves=48,
        min_child_samples=100,
        reg_lambda=12.0,
        reg_alpha=1.0,
        feature_fraction=0.6,
        bagging_fraction=0.7,
        bagging_freq=1,
        random_state=seed
    )
    # try GPU first
    try:
        mdl = LGBMRegressor(device="gpu", **params)
        mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                callbacks=[early_stopping(stopping_rounds=300), log_evaluation(period=200)])
        return mdl
    except Exception as e:
        print(f"[LGBM] GPU fallback to CPU because: {e}")
        mdl = LGBMRegressor(**params)
        mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                callbacks=[early_stopping(stopping_rounds=300), log_evaluation(period=200)])
        return mdl

def fit_xgb_gpu_cpu(Xtr, ytr, Xva, yva, seed):
    # try GPU first
    try:
        mdl = XGBRegressor(
            tree_method="gpu_hist",
            n_estimators=3000, learning_rate=0.03,
            max_depth=8, min_child_weight=8.0,
            subsample=0.7, colsample_bytree=0.6,
            reg_lambda=12.0, reg_alpha=1.0,
            objective="reg:squarederror",
            random_state=seed
        )
        mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=200)
        return mdl
    except Exception as e:
        print(f"[XGB] GPU fallback to CPU because: {e}")
        mdl = XGBRegressor(
            tree_method="hist",
            n_estimators=3000, learning_rate=0.03,
            max_depth=8, min_child_weight=8.0,
            subsample=0.7, colsample_bytree=0.6,
            reg_lambda=12.0, reg_alpha=1.0,
            objective="reg:squarederror",
            random_state=seed
        )
        mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=200)
        return mdl

def fit_cat_gpu_cpu(Xtr, ytr, Xva, yva, seed):
    # try GPU first
    try:
        mdl = CatBoostRegressor(
            iterations=3000, learning_rate=0.03, depth=8,
            l2_leaf_reg=14.0, subsample=0.7, bootstrap_type='Bernoulli',
            loss_function="RMSE", task_type="GPU",
            random_seed=seed, early_stopping_rounds=300, verbose=200
        )
        mdl.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
        return mdl
    except Exception as e:
        print(f"[CAT] GPU fallback to CPU because: {e}")
        mdl = CatBoostRegressor(
            iterations=3000, learning_rate=0.03, depth=8,
            l2_leaf_reg=14.0, subsample=0.7, bootstrap_type='Bernoulli',
            loss_function="RMSE", task_type="CPU",
            random_seed=seed, early_stopping_rounds=300, verbose=200
        )
        mdl.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
        return mdl

# ==============================================================
# Base Models + Bagging (train on TRAIN, eval on VALID)
# ==============================================================
def fit_predict_bagging_lgbm(Xtr, ytr, Xva, yva, Xte, n_bags=N_BAGS, seed=SEED):
    preds_va = np.zeros(len(Xva), dtype=np.float32)
    preds_te = np.zeros(len(Xte), dtype=np.float32)
    for b in range(n_bags):
        mdl = fit_lgbm_gpu_cpu(Xtr, ytr, Xva, yva, seed+b)
        preds_va += mdl.predict(Xva).astype(np.float32)/n_bags
        preds_te += mdl.predict(Xte).astype(np.float32)/n_bags
    return preds_va, preds_te

def fit_predict_bagging_cat(Xtr, ytr, Xva, yva, Xte, n_bags=N_BAGS, seed=SEED):
    preds_va = np.zeros(len(Xva), dtype=np.float32)
    preds_te = np.zeros(len(Xte), dtype=np.float32)
    for b in range(n_bags):
        mdl = fit_cat_gpu_cpu(Xtr, ytr, Xva, yva, seed+b)
        preds_va += mdl.predict(Xva).astype(np.float32)/n_bags
        preds_te += mdl.predict(Xte).astype(np.float32)/n_bags
    return preds_va, preds_te

def fit_predict_bagging_xgb(Xtr, ytr, Xva, yva, Xte, n_bags=N_BAGS, seed=SEED):
    preds_va = np.zeros(len(Xva), dtype=np.float32)
    preds_te = np.zeros(len(Xte), dtype=np.float32)
    for b in range(n_bags):
        mdl = fit_xgb_gpu_cpu(Xtr, ytr, Xva, yva, seed+b)
        preds_va += mdl.predict(Xva).astype(np.float32)/n_bags
        preds_te += mdl.predict(Xte).astype(np.float32)/n_bags
    return preds_va, preds_te

def fit_predict_bagging_rf(Xtr, ytr, Xva, yva, Xte, n_bags=N_BAGS, seed=SEED):
    preds_va = np.zeros(len(Xva), dtype=np.float32)
    preds_te = np.zeros(len(Xte), dtype=np.float32)
    for b in range(n_bags):
        mdl = RandomForestRegressor(
            n_estimators=600, max_depth=None, min_samples_leaf=2,
            max_features=0.5, n_jobs=-1, random_state=seed + b, oob_score=False
        )
        mdl.fit(Xtr, ytr)
        preds_va += mdl.predict(Xva).astype(np.float32)/n_bags
        preds_te += mdl.predict(Xte).astype(np.float32)/n_bags
    return preds_va, preds_te

print("\nTraining base models with bagging...")
va_lgb, te_lgb = fit_predict_bagging_lgbm(X_tr_final, y_tr, X_va_final, y_va, X_te_final)
va_cat, te_cat = fit_predict_bagging_cat (X_tr_final, y_tr, X_va_final, y_va, X_te_final)
va_xgb, te_xgb = fit_predict_bagging_xgb (X_tr_final, y_tr, X_va_final, y_va, X_te_final)
va_rf , te_rf  = fit_predict_bagging_rf  (X_tr_final, y_tr, X_va_final, y_va, X_te_final)

print("\nBase RMSE (Validation):")
print("  LGBM :", np.sqrt(mean_squared_error(y_va, va_lgb)))
print("  CAT  :", np.sqrt(mean_squared_error(y_va, va_cat)))
print("  XGB  :", np.sqrt(mean_squared_error(y_va, va_xgb)))
print("  RF   :", np.sqrt(mean_squared_error(y_va, va_rf )))

# ==============================================================
# Stacking (meta-model: Linear Regression)
# ==============================================================
meta_va = pd.DataFrame({"lgb": va_lgb, "cat": va_cat, "xgb": va_xgb, "rf": va_rf})
meta_te = pd.DataFrame({"lgb": te_lgb, "cat": te_cat, "xgb": te_xgb, "rf": te_rf})

meta = LinearRegression()
meta.fit(meta_va, y_va)

va_stack = meta.predict(meta_va)
te_stack = meta.predict(meta_te)

rmse_stack = np.sqrt(mean_squared_error(y_va, va_stack))
print("\nStacking RMSE (Validation):", rmse_stack)

# ==============================================================
# Submission
# ==============================================================
sub = pd.DataFrame({ID: test[ID], TARGET: np.clip(te_stack, 0, 1)})
sub.to_csv("submission.csv", index=False)
print("\n✅ Submission saved: submission.csv")


