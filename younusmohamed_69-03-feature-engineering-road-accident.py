import os, gc, time, sys, math, json, warnings, pathlib, textwrap, random
warnings.filterwarnings("ignore")

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb  # optional; kept for completeness

from catboost import CatBoostRegressor, Pool
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, PolynomialFeatures

VERSION = "14"               
CUTOFF_HOURS = 11.5           # global wall-clock cutoff; progress saved
N_SPLITS = 5
SEED = 42

# "lgbm", "xgb", "cat", "hgb", "rf", "et", "gbr", "ridge", "lasso", "enet"
MODELS_TO_RUN = ["enet"]  

GPU_MODEL_SET  = ["lgbm", "xgb", "cat", "hgb"]      
CPU_MODEL_SET  = ["rf", "et", "gbr", "ridge", "lasso", "enet"]

# Feature-Engineering tiers to run
FE_SETS_TO_RUN = [
    {"key": "fe1_eda"},
    {"key": "fe2_eda_plus"},
    {"key": "fe3_richer"},
    {"key": "fe4_max"}
]

# Preprocessing grid (same idea as your last code)
# encoder_type: "native_cats" (for LGB/CAT), "onehot", "ordinal", "target_mean", "freq"
# num_transform: list among ["none","log1p_acc","poly","bin_curvature","standardize","interact_cs"]
# rare_thresh: merge categories < frac into "RARE" (for non-native encoders only)
PREPROCESS_TO_RUN = [
    {"key":"native_base",  "encoder_type":"native_cats", "num_transform":["none"],                      "rare_thresh":None},
    {"key":"oh_base",      "encoder_type":"onehot",      "num_transform":["none"],                      "rare_thresh":None},
    {"key":"oh_rare",      "encoder_type":"onehot",      "num_transform":["none"],                      "rare_thresh":0.005},
    {"key":"oh_poly",      "encoder_type":"onehot",      "num_transform":["poly","interact_cs"],        "rare_thresh":None},
    {"key":"oh_bin",       "encoder_type":"onehot",      "num_transform":["bin_curvature","interact_cs"],"rare_thresh":None},
    {"key":"ord_std",      "encoder_type":"ordinal",     "num_transform":["standardize"],               "rare_thresh":0.005},
    {"key":"tgtm_mean",    "encoder_type":"target_mean", "num_transform":["interact_cs"],               "rare_thresh":0.005},
    {"key":"freq_poly",    "encoder_type":"freq",        "num_transform":["poly","log1p_acc"],          "rare_thresh":0.005},
]


DATA_DIR = "/kaggle/input/playground-series-s5e10"
OUT_SUB_DIR = "submissions"
OUT_OOF_DIR = "oof"
OUT_RES_DIR = "results"
os.makedirs(OUT_SUB_DIR, exist_ok=True)
os.makedirs(OUT_OOF_DIR, exist_ok=True)
os.makedirs(OUT_RES_DIR, exist_ok=True)

# Global timer
_GLOBAL_START = time.time()
_CUTOFF_SECS = CUTOFF_HOURS * 3600.0
def time_left_ok(): return (time.time() - _GLOBAL_START) < _CUTOFF_SECS
def now_min(): return round((time.time() - _GLOBAL_START)/60.0, 2)

print(f"[INFO] VERSION={VERSION} | CUT-OFF={CUTOFF_HOURS}h | N_SPLITS={N_SPLITS} | SEED={SEED}")
print(f"[INFO] Models this run: {MODELS_TO_RUN}")
print(f"[INFO] FE tiers: {[p['key'] for p in FE_SETS_TO_RUN]}")
print(f"[INFO] Preprocess combos: {[p['key'] for p in PREPROCESS_TO_RUN]}")

train_path = os.path.join(DATA_DIR, "train.csv")
test_path  = os.path.join(DATA_DIR, "test.csv")
sub_path   = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(train_path), train_path
assert os.path.exists(test_path), test_path

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)
sample_sub = pd.read_csv(sub_path) if os.path.exists(sub_path) else pd.DataFrame({"id": test["id"], "accident_risk": 0.0})

print(train.shape, test.shape, sample_sub.shape)
display(train.head(3))
display(test.head(3))


TARGET = "accident_risk"
ID_COL = "id"

# Column groups
all_cols = [c for c in train.columns if c != TARGET]
cat_cols = [c for c in all_cols if train[c].dtype == "object"]
bool_cols = [c for c in all_cols if train[c].dtype == bool]
num_cols  = [c for c in all_cols if c not in cat_cols + bool_cols + [ID_COL]]

# Treat bool as categorical for native-cat libs; also keep numeric {0,1} if needed
cat_cols_all_base = cat_cols + bool_cols
num_cols_all_base = [c for c in num_cols if c != ID_COL]

features = [c for c in train.columns if c not in [TARGET]]
print("Base categorical-like:", cat_cols_all_base)
print("Base numeric-like:", num_cols_all_base)

y = train[TARGET].values
test_ids = test[ID_COL].values

# CV splitter (bin target for balance)
y_bins = pd.qcut(y, q=20, duplicates="drop").astype(str)
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
print("[INFO] KFold ready.")


def rmse(a, b):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(a, b, squared=False)

def evaluate_metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    return {
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mae":  mean_absolute_error(y_true, y_pred),
        "r2":   r2_score(y_true, y_pred)
    }

def save_oof_and_submission(model_key, oof_pred, test_pred):
    oof_df = pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "oof_pred": oof_pred})
    oof_path = os.path.join(OUT_OOF_DIR, f"oof_{model_key}_v{VERSION}.csv")
    oof_df.to_csv(oof_path, index=False)
    sub_df = sample_sub.copy()
    sub_df[TARGET] = np.clip(test_pred, 0.0, 1.0)
    sub_path = os.path.join(OUT_SUB_DIR, f"submission_{model_key}_v{VERSION}.csv")
    sub_df.to_csv(sub_path, index=False)
    return oof_path, sub_path

def merge_rare_levels(sr: pd.Series, frac=0.005):
    if frac is None:
        return sr
    vc = sr.value_counts(normalize=True)
    rare = vc[vc < frac].index
    return sr.mask(sr.isin(rare), "RARE")

def kfold_target_mean_encoding(train_df, test_df, col, target, n_splits=5, seed=42):
    """
    Leakage-safe OOF target mean encoding.
    IMPORTANT: cast group key to 'object' to avoid Pandas Categorical fillna/type issues.
    """
    # Work on an object (string-like) view of the column to be safe
    tr_key = train_df[col].astype("object")
    te_key = test_df[col].astype("object")

    global_mean = float(train_df[target].mean())
    oof = pd.Series(np.nan, index=train_df.index, dtype=float)

    kf_local = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf_local.split(train_df):
        # Compute means using only training folds
        fold_means = train_df[target].iloc[tr_idx].groupby(tr_key.iloc[tr_idx]).mean()
        # Map on validation keys (as object), coerce to float, then fillna with float
        vals = tr_key.iloc[va_idx].map(fold_means)
        vals = vals.astype(float)
        vals = vals.fillna(global_mean)
        oof.iloc[va_idx] = vals.values

    # Test encoding from full-train means
    full_means = train_df[target].groupby(tr_key).mean()
    test_enc = te_key.map(full_means).astype(float).fillna(global_mean)

    return oof.astype(float), test_enc.astype(float)

def frequency_encoding(train_df, test_df, col):
    freq = train_df[col].astype("object").value_counts()
    tr = train_df[col].astype("object").map(freq).astype(float)
    te = test_df[col].astype("object").map(freq).fillna(0).astype(float)
    return tr, te

def cyclical_map_time_of_day(df):
    mapping = {"morning": 1, "afternoon": 2, "evening": 3}
    x = df["time_of_day"].map(mapping).fillna(2).astype(float)
    ang = (x - x.min()) / (x.max() - x.min()) * 2 * math.pi
    return np.sin(ang), np.cos(ang)

# --- dedup utilities used elsewhere ---
def ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    counts = {}
    new_cols = []
    for c in df.columns:
        if c in counts:
            counts[c] += 1
            new_cols.append(f"{c}__dup{counts[c]}")
        else:
            counts[c] = 0
            new_cols.append(c)
    if len(set(new_cols)) != len(new_cols):
        for i, c in enumerate(new_cols):
            new_cols[i] = f"{c}__u{i}"
    df = df.copy()
    df.columns = new_cols
    return df

def unique_keep_order(seq):
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out


def add_common_flags(df):
    for b in ["road_signs_present","public_road","holiday","school_season"]:
        if b in df.columns:
            df[f"{b}_int"] = df[b].astype(int)
    if "lighting" in df.columns:
        df["is_night"] = (df["lighting"]=="night").astype(int)
        df["is_dim"]   = (df["lighting"]=="dim").astype(int)
    if "weather" in df.columns:
        df["is_foggy"] = (df["weather"]=="foggy").astype(int)
        df["is_clear"] = (df["weather"]=="clear").astype(int)
        df["is_rainy"] = (df["weather"]=="rainy").astype(int)
    if "speed_limit" in df.columns:
        df["is_high_speed"] = (df["speed_limit"]>=60).astype(int)
    if "num_lanes" in df.columns:
        df["lanes_ge3"] = (df["num_lanes"]>=3).astype(int)

def add_cyclical_time(df):
    if "time_of_day" in df.columns:
        s, c = cyclical_map_time_of_day(df)  # make sure this helper exists
        df["tod_sin"] = s
        df["tod_cos"] = c

def add_numeric_basics(df):
    if {"curvature","speed_limit"}.issubset(df.columns):
        df["curv_x_speed"] = df["curvature"] * df["speed_limit"]
        df["curv_over_speed"] = df["curvature"] / (df["speed_limit"] + 1e-6)
        if "num_lanes" in df.columns:
            df["speed_over_lanes"] = df["speed_limit"] / (df["num_lanes"] + 1e-6)
    if "num_reported_accidents" in df.columns:
        df["num_reported_accidents_log1p"] = np.log1p(df["num_reported_accidents"])

def add_bins(df):
    if "curvature" in df.columns:
        df["curvature_bin10"] = pd.qcut(df["curvature"], q=10, duplicates="drop").astype(str)
    if "speed_limit" in df.columns:
        df["speed_bin5"] = pd.qcut(df["speed_limit"].rank(method="first"), q=5, duplicates="drop").astype(str)

def oof_mean_encode(train_df, test_df, cols):
    out_tr = pd.DataFrame(index=train_df.index); out_te = pd.DataFrame(index=test_df.index)
    for c in cols:
        # be robust vs Categorical dtypes
        tr_enc, te_enc = kfold_target_mean_encoding(
            train_df.assign(**{c: train_df[c].astype(str)}),
            test_df.assign(**{c: test_df[c].astype(str)}),
            c, TARGET, n_splits=N_SPLITS, seed=SEED
        )
        out_tr[f"tmean__{c}"] = tr_enc.astype(float)
        out_te[f"tmean__{c}"] = te_enc.astype(float)
    return out_tr, out_te

def frequency_encoding_df(train_df, test_df, cols):
    out_tr = pd.DataFrame(index=train_df.index); out_te = pd.DataFrame(index=test_df.index)
    for c in cols:
        tr_enc, te_enc = frequency_encoding(train_df, test_df, c)  # returns Series
        out_tr[f"freq__{c}"] = tr_enc.astype(float)
        out_te[f"freq__{c}"] = te_enc.astype(float)
    return out_tr, out_te

def cross_oof_mean(train_df, test_df, pairs):
    out_tr = pd.DataFrame(index=train_df.index); out_te = pd.DataFrame(index=test_df.index)
    for a,b in pairs:
        key = f"{a}__{b}"
        tr_ab = train_df[a].astype(str) + "|" + train_df[b].astype(str)
        te_ab = test_df[a].astype(str)  + "|" + test_df[b].astype(str)
        tmp_tr = train_df.copy(); tmp_te = test_df.copy()
        tmp_tr[key] = tr_ab; tmp_te[key] = te_ab
        tr_enc, te_enc = kfold_target_mean_encoding(tmp_tr, tmp_te, key, TARGET, n_splits=N_SPLITS, seed=SEED)
        out_tr[f"tmean__{key}"] = tr_enc.astype(float)
        out_te[f"tmean__{key}"] = te_enc.astype(float)
    return out_tr, out_te

def poly_expand(df, base_cols, degree=2, prefix="poly"):
    from sklearn.preprocessing import PolynomialFeatures
    use_cols = [c for c in base_cols if c in df.columns]
    if not use_cols:
        return pd.DataFrame(index=df.index)
    X = df[use_cols].copy()
    for c in use_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in use_cols:
        col = X[c]
        if col.isna().any():
            med = col.median()
            if np.isnan(med):
                med = 0.0
            X[c] = col.fillna(med)
    pf = PolynomialFeatures(degree=degree, include_bias=False)
    M = pf.fit_transform(X.values)
    names = pf.get_feature_names_out(use_cols)
    out = pd.DataFrame(M, index=df.index, columns=[f"{prefix}__{n}" for n in names])
    add_cols = [c for c in out.columns if c not in df.columns]
    return out[add_cols]

def build_fe_set(fe_key, base_train, base_test, cat_cols_all):
    tr = base_train.copy(); te = base_test.copy()

    add_common_flags(tr); add_common_flags(te)
    add_cyclical_time(tr); add_cyclical_time(te)
    add_numeric_basics(tr); add_numeric_basics(te)
    add_bins(tr); add_bins(te)

    added_cols = []

    if fe_key == "fe1_eda":
        cats1 = [c for c in ["lighting","weather","holiday","public_road","road_type","time_of_day"] if c in tr.columns]
        t_tr, t_te = oof_mean_encode(tr, te, cats1)
        f_tr, f_te = frequency_encoding_df(tr, te, cats1)
        tr = pd.concat([tr, t_tr, f_tr], axis=1); te = pd.concat([te, t_te, f_te], axis=1)
        added_cols += list(t_tr.columns) + list(f_tr.columns)

    if fe_key in ["fe2_eda_plus", "fe3_richer", "fe4_max"]:
        cats1 = [c for c in ["lighting","weather","holiday","public_road","road_type","time_of_day"] if c in tr.columns]
        t_tr, t_te = oof_mean_encode(tr, te, cats1)
        f_tr, f_te = frequency_encoding_df(tr, te, cats1)
        tr = pd.concat([tr, t_tr, f_tr], axis=1); te = pd.concat([te, t_te, f_te], axis=1)
        added_cols += list(t_tr.columns) + list(f_tr.columns)

        pairs = []
        for a in ["lighting","weather","road_type"]:
            for b in ["time_of_day","public_road","holiday"]:
                if a in tr.columns and b in tr.columns:
                    pairs.append((a,b))
        ct_tr, ct_te = cross_oof_mean(tr, te, pairs[:4])
        tr = pd.concat([tr, ct_tr], axis=1); te = pd.concat([te, ct_te], axis=1)
        added_cols += list(ct_tr.columns)

        base_num = [c for c in ["curvature","speed_limit","num_lanes","num_reported_accidents_log1p","curv_x_speed","speed_over_lanes"] if c in tr.columns]
        for c in base_num:
            tr[f"{c}__sq"] = tr[c]**2; te[f"{c}__sq"] = te[c]**2
        added_cols += [f"{c}__sq" for c in base_num]

    if fe_key in ["fe3_richer", "fe4_max"]:
        cats_more_a = [c for c in ["road_type","lighting","weather"] if c in tr.columns]
        cats_more_b = [c for c in ["time_of_day","holiday","public_road","school_season"] if c in tr.columns]
        pairs_more = [(a,b) for a in cats_more_a for b in cats_more_b]
        ct2_tr, ct2_te = cross_oof_mean(tr, te, pairs_more[:12])
        tr = pd.concat([tr, ct2_tr], axis=1); te = pd.concat([te, ct2_te], axis=1)

        for b in ["curvature_bin10","speed_bin5"]:
            if b in tr.columns:
                # FIX: name the new columns explicitly
                fe_tr, fe_te = frequency_encoding(tr, te, b)
                tr = pd.concat([tr, pd.DataFrame({f"freq__{b}": fe_tr})], axis=1)
                te = pd.concat([te, pd.DataFrame({f"freq__{b}": fe_te})], axis=1)

        base_poly = [c for c in ["curvature","speed_limit","num_lanes",
                                 "num_reported_accidents_log1p","curv_x_speed","speed_over_lanes"]
                     if c in tr.columns]
        if len(base_poly) >= 2:
            poly_tr = poly_expand(tr, base_poly, degree=2, prefix="poly2")
            poly_te = poly_expand(te, base_poly, degree=2, prefix="poly2")
            tr = pd.concat([tr, poly_tr], axis=1)
            te = pd.concat([te, poly_te], axis=1)

    if fe_key == "fe4_max":
        cats_all = [c for c in cat_cols_all if c in tr.columns]
        extra_pairs = []
        for i,a in enumerate(cats_all):
            for j,b in enumerate(cats_all):
                if j > i:
                    extra_pairs.append((a,b))
        ct3_tr, ct3_te = cross_oof_mean(tr, te, extra_pairs[:20])
        tr = pd.concat([tr, ct3_tr], axis=1); te = pd.concat([te, ct3_te], axis=1)

        wide_poly = [c for c in ["curvature","speed_limit","num_lanes",
                                 "num_reported_accidents_log1p","curv_x_speed","speed_over_lanes",
                                 "tod_sin","tod_cos"] if c in tr.columns]
        if len(wide_poly) >= 2:
            poly_tr2 = poly_expand(tr, wide_poly, degree=2, prefix="poly2w")
            poly_te2 = poly_expand(te, wide_poly, degree=2, prefix="poly2w")
            tr = pd.concat([tr, poly_tr2], axis=1)
            te = pd.concat([te, poly_te2], axis=1)

    tr = ensure_unique_columns(tr)
    te = ensure_unique_columns(te)

    cat_work = [c for c in tr.columns if (c in cat_cols_all or c in ["curvature_bin10","speed_bin5"])]
    num_work = [c for c in tr.columns if c not in cat_work + [TARGET]]

    meta = {"fe_key": fe_key, "fe_cols_added": [c for c in tr.columns if c not in base_train.columns],
            "cat_cols_work": cat_work, "num_cols_work": num_work}
    return tr, te, meta


def build_views_for_spec(spec, train_df, test_df, cat_cols_all, num_cols_all):
    key = spec["key"]
    enc = spec["encoder_type"]
    num_tx = spec["num_transform"] if isinstance(spec["num_transform"], list) else [spec["num_transform"]]
    rare_thr = spec.get("rare_thresh", None)

    tr = train_df.copy()
    te = test_df.copy()

    # De-dup DF columns early (in case FE introduced dups)
    tr = ensure_unique_columns(tr)
    te = ensure_unique_columns(te)

    # ----- Rare level merge (for non-native encoders) -----
    cats_for_encode = cat_cols_all.copy()
    if enc != "native_cats" and rare_thr is not None:
        for c in cats_for_encode:
            if c in tr.columns:
                tr[c] = merge_rare_levels(tr[c], rare_thr)
                te[c] = merge_rare_levels(te[c], rare_thr)

    # Numeric transforms per preprocessing spec
    if "log1p_acc" in num_tx and "num_reported_accidents" in tr.columns:
        tr["num_reported_accidents_log1p_prep"] = np.log1p(tr["num_reported_accidents"])
        te["num_reported_accidents_log1p_prep"] = np.log1p(te["num_reported_accidents"])

    if "interact_cs" in num_tx and {"curvature","speed_limit"}.issubset(tr.columns):
        tr["curv_x_speed_prep"] = tr["curvature"] * tr["speed_limit"]
        te["curv_x_speed_prep"] = te["curvature"] * te["speed_limit"]

    if "bin_curvature" in num_tx and "curvature" in tr.columns:
        tr["curvature_bin_prep"] = pd.qcut(tr["curvature"], q=10, duplicates="drop").astype(str)
        te["curvature_bin_prep"] = pd.qcut(te["curvature"], q=10, duplicates="drop").astype(str)
        cats_for_encode = cats_for_encode + ["curvature_bin_prep"]

    # Ensure cat list is unique (order-preserving)
    cats_for_encode = unique_keep_order([c for c in cats_for_encode if c in tr.columns])

    # View for native categorical libs
    train_lgb = tr.copy()
    test_lgb  = te.copy()
    for c in cats_for_encode:
        if c in train_lgb.columns:
            train_lgb[c] = train_lgb[c].astype("category")
            test_lgb[c]  = test_lgb[c].astype("category")
    lgb_features = [c for c in train_lgb.columns if c != TARGET]

    if enc == "onehot":
        use_nums = [c for c in tr.columns if (c in num_cols_all) or c.endswith("_prep")]
        # unique combined selection (order-preserving)
        sel_cols = unique_keep_order(cats_for_encode + use_nums)
        onehot = ColumnTransformer(
            transformers=[("oh", OneHotEncoder(sparse=False, handle_unknown="ignore"), cats_for_encode)],
            remainder="passthrough"
        )
        X_oh = onehot.fit_transform(tr[sel_cols])
        X_test_oh = onehot.transform(te[sel_cols])
        enc_names = list(onehot.get_feature_names_out())
        X_enc, X_test_enc = X_oh, X_test_oh

    elif enc == "ordinal":
        use_nums = [c for c in tr.columns if (c in num_cols_all) or c.endswith("_prep")]
        ord_cols = cats_for_encode
        sel_cols = unique_keep_order(ord_cols + use_nums)
        ord_enc = ColumnTransformer(
            transformers=[("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ord_cols)],
            remainder="passthrough"
        )
        X_ord = ord_enc.fit_transform(tr[sel_cols])
        X_test_ord = ord_enc.transform(te[sel_cols])
        if "standardize" in num_tx:
            scaler = StandardScaler(with_mean=False)
            X_ord = scaler.fit_transform(X_ord); X_test_ord = scaler.transform(X_test_ord)
        enc_names = list(ord_enc.get_feature_names_out())
        X_enc, X_test_enc = X_ord, X_test_ord

    elif enc == "target_mean":
        # Force object dtype for all categorical keys BEFORE calling the encoder
        for c in cats_for_encode:
            if c in tr.columns:
                tr[c] = tr[c].astype("object")
                te[c] = te[c].astype("object")
    
        tr_tm = pd.DataFrame(index=tr.index); te_tm = pd.DataFrame(index=te.index)
        for c in cats_for_encode:
            tr_tm[c], te_tm[c] = kfold_target_mean_encoding(tr, te, c, TARGET, n_splits=N_SPLITS, seed=SEED)
    
        # append numerics (including *_prep made earlier in this spec)
        use_nums = [c for c in tr.columns if (c in num_cols_all) or c.endswith("_prep")]
        for c in use_nums:
            tr_tm[c] = tr[c].values
            te_tm[c] = te[c].values
    
        X_enc = tr_tm.values.astype(float)
        X_test_enc = te_tm.values.astype(float)
        enc_names = list(tr_tm.columns)


    elif enc == "freq":
        tr_fe = pd.DataFrame(index=tr.index); te_fe = pd.DataFrame(index=te.index)
        for c in cats_for_encode:
            tr_fe[c], te_fe[c] = frequency_encoding(tr, te, c)
        use_nums = [c for c in tr.columns if (c in num_cols_all) or c.endswith("_prep")]
        for c in use_nums:
            tr_fe[c] = tr[c].values; te_fe[c] = te[c].values
        X_enc = tr_fe.values.astype(float); X_test_enc = te_fe.values.astype(float)
        enc_names = list(tr_fe.columns)

    elif enc == "native_cats":
        X_enc = None; X_test_enc = None; enc_names = None

    else:
        raise ValueError(f"Unknown encoder_type: {enc}")

    return {
        "spec_key": key,
        "encoder_type": enc,
        "X_enc": X_enc,
        "X_test_enc": X_test_enc,
        "enc_names": enc_names,
        "train_lgb": train_lgb,
        "test_lgb": test_lgb,
        "lgb_features": [c for c in lgb_features if c != TARGET],
        "cat_cols_for_view": cats_for_encode
    }


def build_model(name, use_gpu=True, lgb_features=None):
    name = name.lower()
    rng = SEED

    if name == "lgbm":
        params = dict(
            n_estimators=5000,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=128,
            min_data_in_leaf=64,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=0.5,
            objective="rmse",
            random_state=rng
        )
        params.update(device_type=("gpu" if use_gpu else "cpu"))
        mono = []
        if lgb_features is not None:
            for f in lgb_features:
                mono.append(1 if f in ("curvature", "speed_limit") else 0)
        params.update(monotone_constraints=mono if mono else None)
        model = lgb.LGBMRegressor(**params)

    elif name == "xgb":
        params = dict(
            n_estimators=5000,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=0.7,
            min_child_weight=10,
            random_state=rng,
            tree_method="gpu_hist" if use_gpu else "hist",
            predictor="gpu_predictor" if use_gpu else None
        )
        model = xgb.XGBRegressor(**params)

    elif name == "cat":
        params = dict(
            iterations=5000,
            depth=8,
            learning_rate=0.03,
            loss_function="RMSE",
            random_state=rng,
            l2_leaf_reg=6.0,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            verbose=False
        )
        if use_gpu:
            params.update(task_type="GPU")
        else:
            params.update(rsm=0.8)
        model = CatBoostRegressor(**params)

    elif name == "hgb":
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=None,
            max_iter=2000,
            max_bins=255,
            min_samples_leaf=64,
            l2_regularization=0.1,
            random_state=rng
        )

    elif name == "rf":
        model = RandomForestRegressor(
            n_estimators=1000, max_depth=None, min_samples_leaf=5, n_jobs=-1, random_state=rng
        )

    elif name == "et":
        model = ExtraTreesRegressor(
            n_estimators=1000, max_depth=None, min_samples_leaf=4, n_jobs=-1, random_state=rng
        )

    elif name == "gbr":
        model = GradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, n_estimators=3000, max_depth=4, subsample=0.8, random_state=rng
        )

    elif name == "ridge":
        model = Pipeline(steps=[("est", Ridge(alpha=2.0, random_state=rng))])

    elif name == "lasso":
        model = Pipeline(steps=[("est", Lasso(alpha=0.001, random_state=rng, max_iter=10000))])

    elif name == "enet":
        model = Pipeline(steps=[("est", ElasticNet(alpha=0.001, l1_ratio=0.2, random_state=rng, max_iter=10000))])

    else:
        raise ValueError(f"Unknown model name: {name}")

    return model



results_rows = []

for fe_spec in FE_SETS_TO_RUN:
    if not time_left_ok():
        print(f"[STOP] Time limit reached at {now_min()} min BEFORE FE '{fe_spec['key']}'.")
        break

    fe_key = fe_spec["key"]
    print(f"\n================ Feature Set: {fe_key} ================")

    # Build FE tier
    train_fe, test_fe, fe_meta = build_fe_set(fe_key, train, test, cat_cols_all_base)
    print(f"[INFO] {fe_key}: +{len(fe_meta['fe_cols_added'])} engineered features")

    # Define per-FE categorical & numeric working lists
    cat_cols_all = fe_meta["cat_cols_work"]
    num_cols_all = [c for c in fe_meta["num_cols_work"] if c not in [ID_COL, TARGET]]

    # Native (for LGB/CAT) copies with proper dtype
    train_nat = train_fe.copy(); test_nat = test_fe.copy()
    for c in cat_cols_all:
        if c in train_nat.columns:
            train_nat[c] = train_nat[c].astype("category")
            test_nat[c]  = test_nat[c].astype("category")

    for prep in PREPROCESS_TO_RUN:
        if not time_left_ok():
            print(f"[STOP] Time limit reached at {now_min()} min BEFORE preprocessing '{prep['key']}'.")
            break

        print(f"\n----- Preprocessing: {prep['key']} (enc={prep['encoder_type']}, num={prep['num_transform']}) -----")
        # Build encoded views for THIS FE dataset
        views = build_views_for_spec(prep, train_nat, test_nat, cat_cols_all, num_cols_all)

        for mdl_name in MODELS_TO_RUN:
            if not time_left_ok():
                print(f"[STOP] Time limit reached at {now_min()} min. Saving progress and exiting model loop.")
                break

            use_gpu = mdl_name in GPU_MODEL_SET
            model_key_base = f"{mdl_name}{'_gpu' if use_gpu else '_cpu'}"
            model_key = f"{model_key_base}__{fe_key}__{prep['key']}"

            print(f"\n[MODEL] {model_key} | time={now_min()} min")

            try:
                model = build_model(mdl_name, use_gpu=use_gpu, lgb_features=views.get("lgb_features", None))
            except Exception as e:
                print(f"[SKIP] Could not build {model_key}: {e}")
                continue

            oof_pred = np.zeros(len(train), dtype=float)
            test_pred_folds = np.zeros((len(test), N_SPLITS), dtype=float)

            for fold, (tr_idx, va_idx) in enumerate(kf.split(train[ID_COL], y_bins), 1):
                if not time_left_ok():
                    print(f"[STOP] Time limit reached mid-model at {now_min()} min. Saving partial results for {model_key}.")
                    break

                y_tr, y_va = y[tr_idx], y[va_idx]
                enc_type = views["encoder_type"]

                if mdl_name == "lgbm":
                    if enc_type == "native_cats":
                        dtr = lgb.Dataset(
                            train_nat.iloc[tr_idx][views["lgb_features"]],
                            label=y_tr,
                            categorical_feature=[c for c in cat_cols_all if c in views["lgb_features"]],
                            free_raw_data=False
                        )
                        dva = lgb.Dataset(
                            train_nat.iloc[va_idx][views["lgb_features"]],
                            label=y_va,
                            categorical_feature=[c for c in cat_cols_all if c in views["lgb_features"]],
                            free_raw_data=False
                        )
                        params = model.get_params()
                        drop_keys = {"monotone_constraints", "importance_type"}
                        fit_params = {k: v for k, v in params.items() if k not in drop_keys}
                        bst = lgb.train(
                            fit_params,
                            dtr,
                            valid_sets=[dtr, dva],
                            valid_names=["train","valid"],
                            callbacks=[
                                lgb.early_stopping(stopping_rounds=200, verbose=False),
                                lgb.log_evaluation(period=200)
                            ]
                        )
                        oof_pred[va_idx] = bst.predict(train_nat.iloc[va_idx][views["lgb_features"]], num_iteration=bst.best_iteration)
                        test_pred_folds[:, fold-1] = bst.predict(test_nat[views["lgb_features"]], num_iteration=bst.best_iteration)
                        del dtr, dva, bst
                    else:
                        # sklearn LGBM on encoded matrix
                        X_enc = views["X_enc"]; X_test_enc = views["X_test_enc"]
                        est = build_model("lgbm", use_gpu=use_gpu, lgb_features=None)
                        est.fit(
                            X_enc[tr_idx], y_tr,
                            eval_set=[(X_enc[va_idx], y_va)],
                            eval_metric="rmse",
                            callbacks=[
                                lgb.early_stopping(stopping_rounds=200, verbose=False),
                                lgb.log_evaluation(period=200)
                            ]
                        )
                        oof_pred[va_idx] = est.predict(X_enc[va_idx])
                        test_pred_folds[:, fold-1] = est.predict(X_test_enc)
                        del est

                elif mdl_name == "cat":
                    if enc_type == "native_cats":
                        train_pool = Pool(train_nat.iloc[tr_idx][views["lgb_features"]], y_tr,
                                          cat_features=[c for c in cat_cols_all if c in views["lgb_features"]])
                        valid_pool = Pool(train_nat.iloc[va_idx][views["lgb_features"]], y_va,
                                          cat_features=[c for c in cat_cols_all if c in views["lgb_features"]])
                        params = model.get_params()
                        cat = CatBoostRegressor(**params)
                        cat.fit(train_pool, eval_set=valid_pool, verbose=False, use_best_model=True, early_stopping_rounds=200)
                        oof_pred[va_idx] = cat.predict(valid_pool)
                        test_pred_folds[:, fold-1] = cat.predict(Pool(test_nat[views["lgb_features"]],
                                                                      cat_features=[c for c in cat_cols_all if c in views["lgb_features"]]))
                        del train_pool, valid_pool, cat
                    else:
                        X_enc = views["X_enc"]; X_test_enc = views["X_test_enc"]
                        params = model.get_params()
                        cat = CatBoostRegressor(**params)
                        cat.fit(X_enc[tr_idx], y_tr, eval_set=(X_enc[va_idx], y_va),
                                verbose=False, use_best_model=True, early_stopping_rounds=200)
                        oof_pred[va_idx] = cat.predict(X_enc[va_idx])
                        test_pred_folds[:, fold-1] = cat.predict(X_test_enc)
                        del cat

                elif mdl_name == "xgb":
                    X_enc = views["X_enc"]; X_test_enc = views["X_test_enc"]
                    if enc_type == "native_cats":
                        # quick one-hot fallback if someone adds xgb to MODELS_TO_RUN
                        oh_cols = cat_cols_all
                        onehot = ColumnTransformer(
                            transformers=[("oh", OneHotEncoder(sparse=False, handle_unknown="ignore"), oh_cols)],
                            remainder="passthrough"
                        )
                        X_all = onehot.fit_transform(train_nat[oh_cols + num_cols_all])
                        X_te_all = onehot.transform(test_nat[oh_cols + num_cols_all])
                        X_enc = X_all; X_test_enc = X_te_all
                    dtr = xgb.DMatrix(X_enc[tr_idx], label=y_tr)
                    dva = xgb.DMatrix(X_enc[va_idx], label=y_va)
                    dte = xgb.DMatrix(X_test_enc)
                    params = model.get_params()
                    n_estimators = params.pop("n_estimators")
                    xg = xgb.train(
                        params,
                        dtr,
                        num_boost_round=n_estimators,
                        evals=[(dtr,"train"), (dva,"valid")],
                        verbose_eval=False,
                        early_stopping_rounds=200
                    )
                    oof_pred[va_idx] = xg.predict(dva, iteration_range=(0, xg.best_iteration+1))
                    test_pred_folds[:, fold-1] = xg.predict(dte, iteration_range=(0, xg.best_iteration+1))
                    del dtr, dva, dte, xg

                else:
                    # sklearn models (RF/ET/GBR/HistGB/Ridge/Lasso/ENet) need matrices
                    X_enc = views["X_enc"]; X_test_enc = views["X_test_enc"]
                    if X_enc is None or X_test_enc is None:
                        oh = ColumnTransformer(
                            transformers=[("oh", OneHotEncoder(sparse=False, handle_unknown="ignore"), cat_cols_all)],
                            remainder="passthrough"
                        )
                        X_all = oh.fit_transform(train_nat[cat_cols_all + num_cols_all])
                        X_te_all = oh.transform(test_nat[cat_cols_all + num_cols_all])
                        X_enc = X_all; X_test_enc = X_te_all

                    est = build_model(mdl_name, use_gpu=False)
                    est.fit(X_enc[tr_idx], y_tr)
                    oof_pred[va_idx] = est.predict(X_enc[va_idx])
                    test_pred_folds[:, fold-1] = est.predict(X_test_enc)
                    del est

                gc.collect()

            # aggregate & save
            test_pred = np.nanmean(test_pred_folds, axis=1)
            oof_metrics = evaluate_metrics(y, np.nan_to_num(oof_pred, nan=np.nanmean(oof_pred)))
            oof_path, sub_path = save_oof_and_submission(model_key, oof_pred, test_pred)

            print(f"[DONE] {model_key} | RMSE={oof_metrics['rmse']:.6f} | MAE={oof_metrics['mae']:.6f} | R2={oof_metrics['r2']:.4f}")
            print(f"       OOF: {oof_path} | SUB: {sub_path}")

            row = dict(model=model_key, fe_key=fe_key, prep=prep["key"], folds=N_SPLITS, **oof_metrics, time_min=now_min())
            results_rows.append(row)

            if not time_left_ok():
                print(f"[STOP] Time limit reached after completing {model_key}.")
                break

# Save combined leaderboard
results_df = pd.DataFrame(results_rows).sort_values(["rmse","model"])
res_csv_path = os.path.join(OUT_RES_DIR, f"results_01_v{VERSION}.csv")
results_df.to_csv(res_csv_path, index=False)
print(f"\n[RESULTS] Saved leaderboard -> {res_csv_path}")
display(results_df.head(20))


if len(results_rows) == 0:
    print("[WARN] No models were trained. Nothing to plot.")
else:
    top = results_df.nsmallest(10, "rmse")
    plt.figure(figsize=(10,7))
    labels = [f"{m}\n[{f} | {p}]" for m,f,p in zip(top["model"], top["fe_key"], top["prep"])]
    plt.barh(range(len(top)), top["rmse"].values)
    plt.yticks(range(len(top)), labels, fontsize=9)
    plt.gca().invert_yaxis()
    plt.title(f"Top-10 by RMSE (v{VERSION})")
    plt.xlabel("RMSE")
    plt.tight_layout()
    fig_path = os.path.join(OUT_RES_DIR, f"results_01_v{VERSION}.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved -> {fig_path}")

