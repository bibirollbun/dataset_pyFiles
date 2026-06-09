!pip install -U scikit-learn
!pip install -U focal-loss


import os
import psutil  # For CPU affinity

# CPU affinity (pin to specific cores to prevent resource overlap)
# psutil.Process(os.getpid()).cpu_affinity([42, 43, 44, 45, 46, 47])
# os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-f0338f46-3cf2-5aef-8d17-037aaab5d278"

# coding: utf-8
"""
Single-file Kaggle script (v9) for: task/playground-series-s5e10
Model: GRANDE (GPU). Uses only GRANDE.

Refinements vs v8:
- Train-only preprocessing to eliminate leakage: detect column types and fit both QuantileTransformer
  and OneHotEncoder on Xtr_raw (training fold) only.
- Full one-hot (drop=None) + handle_unknown='ignore' to avoid information loss.

Other retained fixes:
- Coherent TF stack on NumPy 2.x if TF import fails (tensorflow==2.19.1, protobuf==4.25.3, ml_dtypes==0.5.1).
- Install GRANDE without forcing TF downgrades (--no-deps).
- Patch category-encoders tag API to return dict-like _get_tags.
- Version-safe OneHotEncoder constructor (sparse_output vs sparse).
- Use float32 policy to avoid einsum dtype mismatch inside GRANDE.
- Version-safe RMSE: prefer root_mean_squared_error if available; else sqrt(MSE).

Logging: task/playground-series-s5e10/outputs/2_8/code_2_8_v9.txt
Submission: task/playground-series-s5e10/outputs/2_8/submission_9.csv
"""

import os
import sys
import time
import json
import subprocess
import logging
from typing import Tuple, List, Dict

# --------------------------------------------------------------------------
# Logging configuration (required at script start)
# --------------------------------------------------------------------------
BASE_DIR = "task/playground-series-s5e10" if not os.getenv('KAGGLE_KERNEL_RUN_TYPE') else "/kaggle/input/playground-series-s5e10"
OUT_DIR = "." # os.path.join(BASE_DIR, "outputs", "2_8")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "code_2_8_v9.txt")

logging.basicConfig(
    filename=LOG_PATH,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logging.info("Initialized logging to %s", LOG_PATH)

# --------------------------------------------------------------------------
# Global flags and constants
# --------------------------------------------------------------------------
DEBUG = True  # Runs DEBUG first, then FULL
SEED = 42
LOW_CARD_THRESHOLD = 20
TARGET_COL = "accident_risk"
ID_COL = "id"

# --------------------------------------------------------------------------
# Environment preparation
# --------------------------------------------------------------------------
# Purpose: Configure CUDA visibility and TF memory behavior.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Light imports (pure Python/NumPy)
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.metrics import mean_squared_error
import sklearn

def run(cmd: list):
    logging.info("Executing: %s", " ".join(cmd))
    subprocess.check_call(cmd)

# Purpose: Ensure TensorFlow is importable and ABI-compatible with NumPy 2.x on Python 3.12.
def ensure_tf_stack():
    try:
        import tensorflow as tf  # noqa: F401
        logging.info("TensorFlow import OK on first attempt.")
        return
    except Exception as e:
        logging.info("TensorFlow import failed: %s", str(e).splitlines()[-1] if str(e) else "ImportError")
        logging.info("Aligning TF/NumPy/protobuf/ml_dtypes stack...")
        run([sys.executable, "-m", "pip", "uninstall", "-y",
             "tf-keras", "keras", "tensorflow", "tensorflow-cpu", "tensorflow-intel",
             "ml_dtypes", "protobuf"])
        run([sys.executable, "-m", "pip", "install", "--no-cache-dir",
             "protobuf==4.25.3", "ml_dtypes==0.5.1", "tensorflow==2.19.1"])
        logging.info("Re-attempting TensorFlow import after alignment...")

ensure_tf_stack()

# Import TF and set dtype policy
import tensorflow as tf
from tensorflow.keras import mixed_precision

# Note: GRANDE’s internals produced a dtype mismatch with mixed_bfloat16 in tf.einsum.
# To ensure stable training, we use float32 policy so tensors share uniform dtype.
mixed_precision.set_global_policy("float32")
tf.random.set_seed(SEED)
np.random.seed(SEED)

# GPU setup
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
logging.info("TensorFlow version: %s | GPUs: %s | Mixed precision policy: %s",
             tf.__version__, [d.name for d in gpus] if gpus else "CPU-only", mixed_precision.global_policy())

# Purpose: Ensure GRANDE is installed and importable without downgrading TF.
def ensure_grande():
    try:
        import importlib.metadata as importlib_metadata  # py3.8+
    except Exception:
        import importlib_metadata  # type: ignore
    try:
        from GRANDE import GRANDE  # noqa: F401
        try:
            ver = importlib_metadata.version("GRANDE")
        except Exception:
            ver = "unknown"
        logging.info("GRANDE already importable (version=%s).", ver)
    except ImportError:
        logging.info("Installing/upgrading GRANDE (no-deps to preserve TF stack)...")
        run([sys.executable, "-m", "pip", "install", "--upgrade", "--no-deps", "GRANDE"])
        from GRANDE import GRANDE  # noqa: F401
        try:
            ver = importlib_metadata.version("GRANDE")
        except Exception:
            ver = "unknown"
        logging.info("GRANDE import succeeded (version=%s).", ver)

# Ensure category-encoders present (GRANDE uses it internally)
def ensure_category_encoders():
    try:
        import category_encoders  # noqa: F401
        logging.info("category-encoders already importable.")
    except ImportError:
        logging.info("Installing category-encoders...")
        run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "category-encoders>=2.6"])

ensure_grande()
ensure_category_encoders()

# Patch category_encoders for sklearn tag compatibility (must return dict, not Tags)
def patch_category_encoders():
    try:
        import category_encoders as ce
        # Patch BaseEncoder._get_tags to return a dict with the expected custom key
        try:
            from category_encoders.utils import BaseEncoder
            orig = getattr(BaseEncoder, "_get_tags", None)
            def _get_tags(self):
                try:
                    tags = orig(self) if callable(orig) else {}
                except Exception:
                    tags = {}
                if not isinstance(tags, dict):
                    tags = {}
                if "supervised_encoder" not in tags:
                    tags["supervised_encoder"] = getattr(self, "supervised", False)
                return tags
            BaseEncoder._get_tags = _get_tags  # type: ignore[attr-defined]
            logging.info("Patched category_encoders BaseEncoder._get_tags (dict-return).")
        except Exception as e:
            logging.info("BaseEncoder patch skipped: %s", repr(e))
        # Patch OrdinalEncoder as an extra safety net
        try:
            if hasattr(ce, "ordinal") and hasattr(ce.ordinal, "OrdinalEncoder"):
                OE = ce.ordinal.OrdinalEncoder
                if not hasattr(OE, "_get_tags") or not callable(getattr(OE, "_get_tags")):
                    def _get_tags(self):
                        return {"supervised_encoder": False}
                    OE._get_tags = _get_tags  # type: ignore[attr-defined]
                    logging.info("Patched category_encoders OrdinalEncoder._get_tags (dict-return).")
        except Exception as e:
            logging.info("OrdinalEncoder patch skipped: %s", repr(e))
    except ImportError:
        logging.info("category_encoders not available; patch not applied.")

patch_category_encoders()

from GRANDE import GRANDE  # final import after patches

# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def read_data(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test CSVs from BASE_DIR."""
    train_path = os.path.join(base_dir, "train.csv")
    test_path = os.path.join(base_dir, "test.csv")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    logging.info("Loaded train/test. train.shape=%s, test.shape=%s", train.shape, test.shape)
    return train, test

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add generator-aligned booleans and interactions."""
    out = df.copy()
    if "lighting" in out.columns:
        out["night"] = (out["lighting"] == "night").astype("int8")
    else:
        out["night"] = 0
    if "weather" in out.columns:
        out["not_clear"] = (out["weather"] != "clear").astype("int8")
    else:
        out["not_clear"] = 0
    if "speed_limit" in out.columns:
        out["speed60"] = (out["speed_limit"] >= 60).astype("int8")
    else:
        out["speed60"] = 0
    if "num_reported_accidents" in out.columns:
        out["acc_gt2"] = (pd.to_numeric(out["num_reported_accidents"], errors="coerce").fillna(-1).values > 2).astype("int8")
    else:
        out["acc_gt2"] = 0
    if "curvature" in out.columns:
        curv = pd.to_numeric(out["curvature"], errors="coerce")
        out["curv_x_night"] = curv * out["night"].astype(np.float32)
        out["curv_x_speed60"] = curv * out["speed60"].astype(np.float32)
    else:
        out["curv_x_night"] = 0.0
        out["curv_x_speed60"] = 0.0
    return out

def identify_column_types(df: pd.DataFrame, target_col: str, id_col: str) -> Tuple[List[str], List[str]]:
    """Return (numeric_bool_cols, low-card_cat_cols)."""
    cols = [c for c in df.columns if c not in [target_col, id_col]]
    cat_cols = [c for c in cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    low_card = [c for c in cat_cols if df[c].nunique(dropna=True) <= LOW_CARD_THRESHOLD]
    num_bool = [c for c in cols if c not in low_card]
    return num_bool, low_card

def parse_version_tuple(v: str) -> tuple:
    parts = []
    for p in v.split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])

# === Patch: train-only preprocessing and full one-hot (no drop-first) ===
def build_ohe(df_train: pd.DataFrame, ohe_cols: List[str]) -> OneHotEncoder:
    """Version-safe OneHotEncoder: fit on TRAIN ONLY; use full one-hot (drop=None)."""
    use_sparse_output = parse_version_tuple(sklearn.__version__) >= parse_version_tuple("1.2.0")
    if use_sparse_output:
        ohe = OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False)
    else:
        ohe = OneHotEncoder(handle_unknown="ignore", drop=None, sparse=False)
    if ohe_cols:
        ohe.fit(df_train[ohe_cols].astype(str))
    return ohe

def build_quantile(df_train: pd.DataFrame, num_cols: List[str]) -> QuantileTransformer:
    n_q = min(2000, max(10, len(df_train)))
    qt = QuantileTransformer(
        n_quantiles=n_q,
        output_distribution="normal",
        subsample=100000,
        random_state=SEED,
        copy=True,
    )
    if num_cols:
        qt.fit(df_train[num_cols])
    return qt

def transform_frame(df: pd.DataFrame, num_cols: List[str], ohe_cols: List[str],
                    qt: QuantileTransformer, ohe: OneHotEncoder) -> Tuple[np.ndarray, List[str]]:
    parts = []
    feat_names = []
    if num_cols:
        Xn = qt.transform(df[num_cols]).astype(np.float32)
        parts.append(Xn)
        feat_names.extend(num_cols)
    if ohe_cols:
        Xc = ohe.transform(df[ohe_cols].astype(str))
        if hasattr(Xc, "toarray"):
            Xc = Xc.toarray()
        Xc = Xc.astype(np.float32)
        parts.append(Xc)
        cats = ohe.categories_
        for col, cat_list in zip(ohe_cols, cats):
            for cat in list(cat_list):  # include all categories (drop=None)
                feat_names.append(f"{col}__{cat}")
    X = np.concatenate(parts, axis=1) if parts else np.zeros((len(df), 0), dtype=np.float32)
    return X, feat_names

def make_group_ids_from_raw(df_raw: pd.DataFrame, target_col: str, id_col: str) -> np.ndarray:
    cols = [c for c in df_raw.columns if c not in [target_col, id_col]]
    key_df = df_raw[cols].copy()
    key_df = key_df.fillna("<NA>").astype(str)
    hashed = pd.util.hash_pandas_object(key_df, index=False).astype(np.int64).values
    return hashed

def make_feature_key_df(df_raw: pd.DataFrame, target_col: str, id_col: str) -> pd.DataFrame:
    cols = [c for c in df_raw.columns if c not in [target_col, id_col]]
    key_df = df_raw[cols].copy()
    for c in key_df.columns:
        key_df[c] = key_df[c].astype(str).replace({"nan": "<NA>", "None": "<NA>"})
    return key_df

def build_memorization_map(df_train_raw: pd.DataFrame, y: pd.Series, min_count: int = 3) -> Dict[str, float]:
    key_df = make_feature_key_df(df_train_raw, TARGET_COL, ID_COL)
    key_series = key_df.apply(lambda r: json.dumps(tuple(r.values.tolist())), axis=1)
    grp = pd.DataFrame({"key": key_series, "y": y.values}).groupby("key")
    stats = grp.agg(["mean", "count"])
    stats.columns = ["mean", "count"]
    stats = stats[stats["count"] >= min_count]
    mapping = stats["mean"].to_dict()
    logging.info("Memorization map built: %d keys (min_count=%d).", len(mapping), min_count)
    return mapping

def apply_memorization_override(df_raw: pd.DataFrame, preds: np.ndarray, mem_map: Dict[str, float]) -> Tuple[np.ndarray, int]:
    if len(mem_map) == 0:
        return preds, 0
    key_df = make_feature_key_df(df_raw, TARGET_COL, ID_COL)
    keys = key_df.apply(lambda r: json.dumps(tuple(r.values.tolist())), axis=1).values
    overrides = 0
    out = preds.copy()
    for i, k in enumerate(keys):
        if k in mem_map:
            out[i] = mem_map[k]
            overrides += 1
    return out, overrides

# Version-safe RMSE (sklearn >=1.4 has root_mean_squared_error; in 1.6+, 'squared' arg was removed)
from sklearn import metrics as skmetrics
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if hasattr(skmetrics, "root_mean_squared_error"):
        return float(skmetrics.root_mean_squared_error(y_true, y_pred))
    else:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def build_fold0_groups(train_raw_with_target: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    groups = make_group_ids_from_raw(train_raw_with_target, TARGET_COL, ID_COL)
    gkf = GroupKFold(n_splits=5)
    fold_iter = gkf.split(np.zeros(len(groups)), np.zeros(len(groups)), groups=groups)
    train_idx, val_idx = next(fold_iter)  # fold 0
    return train_idx, val_idx

def deduplicate_within_train(df_train_raw: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    key_df = make_feature_key_df(df_train_raw, TARGET_COL, ID_COL)
    key_series = key_df.apply(lambda r: json.dumps(tuple(r.values.tolist())), axis=1)
    tmp = df_train_raw.copy()
    tmp["__key"] = key_series.values
    tmp["__y"] = y_train.values
    agg = tmp.groupby("__key")["__y"].agg(["mean", "count"])
    uniq = tmp.drop_duplicates("__key", keep="first").copy()
    y_avg = agg.loc[uniq["__key"].values, "mean"].astype(np.float32).values
    counts = agg.loc[uniq["__key"].values, "count"].astype(np.int32).values
    uniq = uniq.drop(columns=["__key", "__y"]).reset_index(drop=True)
    y_avg_series = pd.Series(y_avg, name="y")
    counts_series = pd.Series(counts, name="count")
    logging.info("Deduplicated train split: %d -> %d unique keys (%.2fx reduction).",
                 len(y_train), len(uniq), (len(y_train) / max(1, len(uniq))))
    return uniq, y_avg_series, counts_series

def get_best_epoch_if_available(model) -> int:
    best_epoch = -1
    if hasattr(model, "history"):
        hist = getattr(model, "history")
        if isinstance(hist, dict) and "val_loss" in hist:
            arr = np.array(hist["val_loss"], dtype=np.float32)
            if len(arr) > 0 and not np.isnan(arr).all():
                best_epoch = int(np.nanargmin(arr))
        elif hasattr(hist, "history") and isinstance(hist.history, dict) and "val_loss" in hist.history:
            arr = np.array(hist.history["val_loss"], dtype=np.float32)
            if len(arr) > 0 and not np.isnan(arr).all():
                best_epoch = int(np.nanargmin(arr))
    if hasattr(model, "best_epoch"):
        try:
            be = int(getattr(model, "best_epoch"))
            if be >= 0:
                best_epoch = be
        except Exception:
            pass
    if best_epoch < 0:
        logging.info("Best epoch not exposed by GRANDE; logging -1.")
    return best_epoch

# --------------------------------------------------------------------------
# GRANDE hyperparameters and args (per recommendations)
# --------------------------------------------------------------------------
def grande_params():
    params = {
        "depth": 5,
        "n_estimators": 1400,
        "learning_rate_index": 0.01,
        "learning_rate_values": 0.01,
        "learning_rate_leaf": 0.007,
        "learning_rate_weights": 0.004,
        "optimizer": "adam",
        "cosine_decay_steps": 400,
        "loss": "mse",
        "focal_loss": False,
        "temperature": 0.0,
        "from_logits": True,
        "use_class_weights": False,
        "dropout": 0.10,
        "selected_variables": 0.8,
        "data_subset_fraction": 1.0,
    }
    args = {
        "epochs": 300,
        "early_stopping_epochs": 30,
        "batch_size": 4096,
        "cat_idx": [],  # we provide numeric matrix (we still patch CE for GRANDE internals)
        "objective": "regression",
        "random_seed": SEED,
        "verbose": 1,
    }
    return params, args

# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def run_pipeline(debug_flag: bool):
    mode = "DEBUG" if debug_flag else "FULL"
    logging.info("======== Running mode: %s ========", mode)

    # Load data and engineer features
    train_df, test_df = read_data(BASE_DIR)
    assert TARGET_COL in train_df.columns, f"Missing target column {TARGET_COL}"
    assert ID_COL in train_df.columns and ID_COL in test_df.columns, "Missing id column"

    train_raw = engineer_features(train_df.drop(columns=[TARGET_COL], errors="ignore"))
    test_raw = engineer_features(test_df.copy())

    # Fold 0 via group hashing (hash of non-id features)
    tr_idx, va_idx = build_fold0_groups(pd.concat([train_raw, train_df[[TARGET_COL]]], axis=1))
    logging.info("Fold0 indices: train=%d, valid=%d", len(tr_idx), len(va_idx))

    # Split
    y_all = train_df[TARGET_COL].astype(np.float32)
    Xtr_raw, Xva_raw = train_raw.iloc[tr_idx].reset_index(drop=True), train_raw.iloc[va_idx].reset_index(drop=True)
    ytr, yva = y_all.iloc[tr_idx].reset_index(drop=True), y_all.iloc[va_idx].reset_index(drop=True)

    # DEBUG sampling rule
    if debug_flag:
        max_debug = 1000
        if len(Xtr_raw) > 2 * max_debug:
            sel = np.random.RandomState(SEED).choice(len(Xtr_raw), size=max_debug, replace=False)
            Xtr_raw = Xtr_raw.iloc[sel].reset_index(drop=True)
            ytr = ytr.iloc[sel].reset_index(drop=True)
            logging.info("DEBUG train subsample size: %d", len(Xtr_raw))
        else:
            logging.warning("DEBUG sample would exceed 50%% of train split; skipping DEBUG mode.")
            return

    # Fold-safe duplicate averaging
    Xtr_raw, ytr, dup_counts = deduplicate_within_train(Xtr_raw, ytr)

    # === Train-only column typing (no leakage) ===
    num_cols, ohe_cols = identify_column_types(Xtr_raw, target_col=TARGET_COL, id_col=ID_COL)
    num_cols = [c for c in num_cols if c != ID_COL]
    ohe_cols = [c for c in ohe_cols if c != ID_COL]
    logging.info("Feature columns: numeric/boolean=%d, low-card categoricals=%d", len(num_cols), len(ohe_cols))

    # Fit preprocessors on TRAIN ONLY
    ohe = build_ohe(Xtr_raw, ohe_cols)
    qt = build_quantile(Xtr_raw, num_cols)

    # Transform splits using train-fitted preprocessors
    Xtr, feat_names = transform_frame(Xtr_raw, num_cols, ohe_cols, qt, ohe)
    Xva, _ = transform_frame(Xva_raw, num_cols, ohe_cols, qt, ohe)
    logging.info("Transformed shapes: Xtr=%s, Xva=%s, n_features=%d", Xtr.shape, Xva.shape, Xtr.shape[1])

    # Memorization maps
    mem_map_tr = build_memorization_map(Xtr_raw, ytr, min_count=3)
    mem_map_full = build_memorization_map(engineer_features(train_df.drop(columns=[TARGET_COL])), train_df[TARGET_COL].astype(np.float32), min_count=3)
    global_mean = float(ytr.mean())
    logging.info("Global mean (train split): %.5f", global_mean)

    # Params
    params, args = grande_params()
    if debug_flag:
        args = {**args, "epochs": 1}
    logging.info("GRANDE params/args: %s | %s", json.dumps(params), json.dumps(args))

    # NaN warm-up (FULL only)
    nan_abort = False
    if not debug_flag:
        params_warm, args_warm = grande_params()
        args_warm["epochs"] = 1
        model_warm = GRANDE(params=params_warm, args=args_warm)
        logging.info("Warm-up: fitting 1 epoch to check for NaN loss...")
        t0 = time.time()
        model_warm.fit(X_train=Xtr, y_train=ytr.values, X_val=Xva, y_val=yva.values)
        warm_preds = model_warm.predict(Xva).astype(np.float32).reshape(-1)
        warm_preds = np.clip(warm_preds, 0.0, 1.0).astype(np.float32)
        warm_preds = np.round(warm_preds, 2)
        warm_preds_over, _ = apply_memorization_override(Xva_raw, warm_preds, mem_map_tr)
        warm_rmse = rmse(yva.values, warm_preds_over)
        elapsed = time.time() - t0
        logging.info("Warm-up val RMSE=%.6f in %.1fs.", warm_rmse, elapsed)
        if np.isnan(warm_rmse) or np.isnan(warm_preds_over).any():
            nan_abort = True
            logging.warning("Detected NaN after warm-up epoch; will SKIP further training and proceed to inference.")

    # Train (fold 0)
    model = None
    fit_time = 0.0
    if not nan_abort:
        model = GRANDE(params=params, args=args)
        logging.info("Fitting GRANDE on fold 0 ...")
        t0 = time.time()
        model.fit(X_train=Xtr, y_train=ytr.values, X_val=Xva, y_val=yva.values)
        fit_time = time.time() - t0
        logging.info("Training completed in %.1fs.", fit_time)
    else:
        logging.info("Training skipped due to NaN detection.")

    # Validation
    if model is not None:
        va_preds = model.predict(Xva).astype(np.float32).reshape(-1)
    else:
        va_preds = np.full(len(Xva), global_mean, dtype=np.float32)
    va_preds = np.clip(va_preds, 0.0, 1.0).astype(np.float32)
    va_preds = np.round(va_preds, 2)
    va_preds_over, n_over = apply_memorization_override(Xva_raw, va_preds, mem_map_tr)
    val_rmse = rmse(yva.values, va_preds_over)
    logging.info("Fold0 validation: RMSE=%.6f | overrides=%d/%d", val_rmse, n_over, len(va_preds_over))

    best_epoch = -1
    if model is not None:
        best_epoch = get_best_epoch_if_available(model)
    logging.info("Best epoch (if available) = %d | Total training time = %.1fs", best_epoch, fit_time)

    # Inference (skip in DEBUG)
    if debug_flag:
        logging.info("DEBUG mode: Skipping submission file creation.")
        return

    Xte_raw = engineer_features(pd.read_csv(os.path.join(BASE_DIR, "test.csv")))
    Xte, _ = transform_frame(Xte_raw, num_cols, ohe_cols, qt, ohe)
    logging.info("Transformed test shape: %s", Xte.shape)
    if model is not None:
        te_preds = model.predict(Xte).astype(np.float32).reshape(-1)
    else:
        te_preds = np.full(len(Xte), global_mean, dtype=np.float32)
    te_preds = np.clip(te_preds, 0.0, 1.0).astype(np.float32)
    te_preds = np.round(te_preds, 2)
    te_preds_over, n_over_te = apply_memorization_override(Xte_raw, te_preds, mem_map_full)
    logging.info("Test overrides via memorization: %d/%d", n_over_te, len(te_preds_over))

    sub = pd.DataFrame({
        ID_COL: pd.read_csv(os.path.join(BASE_DIR, "test.csv"))[ID_COL].astype(int).values,
        TARGET_COL: te_preds_over.astype(np.float32),
    })
    sub_path = os.path.join(OUT_DIR, "submission_9.csv")
    sub.to_csv(sub_path, index=False)
    logging.info("Wrote submission to %s (rows=%d).", sub_path, len(sub))


if __name__ == "__main__":
    # Run DEBUG first, then FULL.
    run_pipeline(debug_flag=DEBUG)
    run_pipeline(debug_flag=False)

