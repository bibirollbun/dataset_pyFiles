import os, gc, time, math, json, warnings, random, pathlib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet

VERSION = "HPO10"
CUTOFF_HOURS = 11.5
N_SPLITS = 5
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# Which models to tune (same keys as before)
# "lgbm", "xgb", "cat", "hgb", "rf", "et", "gbr", "ridge", "lasso", "enet"
MODELS_TO_RUN = ["lgbm"]

GPU_MODEL_SET  = ["lgbm", "xgb", "cat", "hgb"]
CPU_MODEL_SET  = ["rf", "et", "gbr", "ridge", "lasso", "enet"]

# Per-model HPO budgets (trials); reduce if short on time, increase if you can
HPO_BUDGET = {
    "xgb":   80,
    "lgbm":  80,
    "cat":   60,
    "hgb":   40,
    "rf":    40,
    "et":    40,
    "gbr":   40,
    "ridge": 30,
    "lasso": 30,
    "enet":  40,
}

TRY_OPTUNA = True


# ---- Paths ----
DATA_DIR = "/kaggle/input/playground-series-s5e10"
OUT_SUB_DIR = "submissions"
OUT_OOF_DIR = "oof"
OUT_RES_DIR = "results"
OUT_HPO_DIR = "hpo"
os.makedirs(OUT_SUB_DIR, exist_ok=True)
os.makedirs(OUT_OOF_DIR, exist_ok=True)
os.makedirs(OUT_RES_DIR, exist_ok=True)
os.makedirs(OUT_HPO_DIR, exist_ok=True)

# ---- Global timers ----
_GLOBAL_START = time.time()
_CUTOFF_SECS  = CUTOFF_HOURS * 3600.0
def time_left_ok(): return (time.time() - _GLOBAL_START) < _CUTOFF_SECS
def now_min(): return round((time.time() - _GLOBAL_START)/60.0, 2)

print(f"[INFO] VERSION={VERSION}  CUT-OFF={CUTOFF_HOURS}h  N_SPLITS={N_SPLITS}  SEED={SEED}")
print(f"[INFO] Models to tune: {MODELS_TO_RUN}")
train_path = os.path.join(DATA_DIR, "train.csv")
test_path  = os.path.join(DATA_DIR, "test.csv")
sub_path   = os.path.join(DATA_DIR, "sample_submission.csv")
assert os.path.exists(train_path)
assert os.path.exists(test_path)

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)
sample_sub = pd.read_csv(sub_path) if os.path.exists(sub_path) else pd.DataFrame({"id": test["id"], "accident_risk": 0.0})

print(train.shape, test.shape)
display(train.head(3)); display(test.head(3))


TARGET = "accident_risk"
ID_COL = "id"

all_cols = [c for c in train.columns if c != TARGET]
cat_cols = [c for c in all_cols if train[c].dtype == "object"]
bool_cols = [c for c in all_cols if train[c].dtype == bool]
num_cols  = [c for c in all_cols if c not in cat_cols + bool_cols + [ID_COL]]

cat_cols_all = cat_cols + bool_cols
num_cols_all = [c for c in num_cols if c != ID_COL]
features = [c for c in train.columns if c != TARGET]

y = train[TARGET].values
test_ids = test[ID_COL].values

# One-Hot view (used by XGB/HGB/RF/ET/GBR/Linear when needed)
onehot = ColumnTransformer(
    transformers=[("oh", OneHotEncoder(sparse=False, handle_unknown="ignore"), cat_cols_all)],
    remainder="passthrough"
)
X_oh = onehot.fit_transform(train[cat_cols_all + num_cols_all])
X_test_oh = onehot.transform(test[cat_cols_all + num_cols_all])
oh_feature_names = list(onehot.get_feature_names_out())

# Ordinal+Standardize view (good for linear baselines)
ord_std = ColumnTransformer(
    transformers=[("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols_all)],
    remainder="passthrough"
)
X_ord = ord_std.fit_transform(train[cat_cols_all + num_cols_all])
X_test_ord = ord_std.transform(test[cat_cols_all + num_cols_all])

scaler = StandardScaler(with_mean=False)
X_ord_std = scaler.fit_transform(X_ord)
X_test_ord_std = scaler.transform(X_test_ord)

# LightGBM native categorical view
train_lgb = train.copy(); test_lgb = test.copy()
for c in cat_cols_all:
    train_lgb[c] = train_lgb[c].astype("category")
    test_lgb[c]  = test_lgb[c].astype("category")
lgb_features = [c for c in train_lgb.columns if c != TARGET]

# CatBoost: use original strings and pass cat indices
cat_idx_for_catboost = [i for i, c in enumerate(features) if c in cat_cols_all]

print(f"[OH] X={X_oh.shape}, X_test={X_test_oh.shape}")
print(f"[ORD+STD] X={X_ord_std.shape}, X_test={X_test_ord_std.shape}")
print(f"[LGB] features={len(lgb_features)} (cats={len(cat_cols_all)})")
print(f"[CAT] cat_idx_first={cat_idx_for_catboost[:8]}")


def rmse(a, b): 
    return mean_squared_error(a, b, squared=False)

def evaluate_metrics(y_true, y_pred):
    return {"rmse": rmse(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)}

def save_oof_and_submission(model_key, oof_pred, test_pred):
    oof_df = pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "oof_pred": oof_pred})
    oof_path = os.path.join(OUT_OOF_DIR, f"oof_{model_key}_v{VERSION}.csv")
    oof_df.to_csv(oof_path, index=False)

    sub_df = sample_sub.copy()
    sub_df[TARGET] = np.clip(test_pred, 0.0, 1.0)
    sub_path = os.path.join(OUT_SUB_DIR, f"submission_{model_key}_v{VERSION}.csv")
    sub_df.to_csv(sub_path, index=False)
    return oof_path, sub_path

# CV (bin target for balance)
y_bins = pd.qcut(y, q=20, duplicates="drop").astype(str)
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

print("[INFO] KFold ready.")


def fit_predict_fold(mdl_name, use_gpu, tr_idx, va_idx, data_views, base_params):
    """
    Returns fold_oof_pred, fold_test_pred
    data_views: dict containing:
      - X_oh, X_test_oh
      - X_ord_std, X_test_ord_std
      - train_lgb, test_lgb, lgb_features
      - features, cat_idx_for_catboost
    base_params: dict of model hyperparameters
    """
    y_tr = y[tr_idx]; y_va = y[va_idx]

    if mdl_name == "lgbm":
        # Native categorical with lgb.train (fast + early stopping)
        dtr = lgb.Dataset(
            data_views["train_lgb"].iloc[tr_idx][data_views["lgb_features"]],
            label=y_tr,
            categorical_feature=[c for c in cat_cols_all if c in data_views["lgb_features"]],
            free_raw_data=False
        )
        dva = lgb.Dataset(
            data_views["train_lgb"].iloc[va_idx][data_views["lgb_features"]],
            label=y_va,
            categorical_feature=[c for c in cat_cols_all if c in data_views["lgb_features"]],
            free_raw_data=False
        )

        params = base_params.copy()
        n_estimators = params.pop("n_estimators", 5000)
        callbacks = [lgb.early_stopping(stopping_rounds=200, verbose=False)]
        bst = lgb.train(
            params, dtr, num_boost_round=n_estimators,
            valid_sets=[dtr, dva], valid_names=["train","valid"],
            callbacks=callbacks
        )
        oof = bst.predict(data_views["train_lgb"].iloc[va_idx][data_views["lgb_features"]], num_iteration=bst.best_iteration)
        test_pred = bst.predict(data_views["test_lgb"][data_views["lgb_features"]], num_iteration=bst.best_iteration)
        del dtr, dva, bst
        return oof, test_pred

    elif mdl_name == "cat":
        params = base_params.copy()
        if use_gpu:
            params.update(task_type="GPU")
        # Use native strings + categorical indices
        tr_pool = Pool(data_views["train_df"].iloc[tr_idx][data_views["features"]],
                       y_tr, cat_features=data_views["cat_idx_for_catboost"])
        va_pool = Pool(data_views["train_df"].iloc[va_idx][data_views["features"]],
                       y_va, cat_features=data_views["cat_idx_for_catboost"])
        model = CatBoostRegressor(**params)
        model.fit(tr_pool, eval_set=va_pool, verbose=False, use_best_model=True, early_stopping_rounds=200)
        oof = model.predict(va_pool)
        test_pred = model.predict(Pool(data_views["test_df"][data_views["features"]],
                                       cat_features=data_views["cat_idx_for_catboost"]))
        del tr_pool, va_pool, model
        return oof, test_pred

    elif mdl_name == "xgb":
        # One-hot matrix for XGB
        params = base_params.copy()
        n_estimators = params.pop("n_estimators", 5000)
        dtr = xgb.DMatrix(data_views["X_oh"][tr_idx], label=y_tr)
        dva = xgb.DMatrix(data_views["X_oh"][va_idx], label=y_va)
        dte = xgb.DMatrix(data_views["X_test_oh"])
        bst = xgb.train(
            params, dtr, num_boost_round=n_estimators,
            evals=[(dtr,"train"), (dva,"valid")],
            verbose_eval=False, early_stopping_rounds=200
        )
        oof = bst.predict(dva, iteration_range=(0, bst.best_iteration+1))
        test_pred = bst.predict(dte, iteration_range=(0, bst.best_iteration+1))
        del dtr, dva, dte, bst
        return oof, test_pred

    elif mdl_name == "hgb":
        # sklearn HGB -> one-hot matrix
        est = HistGradientBoostingRegressor(**base_params, random_state=SEED)
        est.fit(data_views["X_oh"][tr_idx], y_tr)
        oof = est.predict(data_views["X_oh"][va_idx])
        test_pred = est.predict(data_views["X_test_oh"])
        del est
        return oof, test_pred

    elif mdl_name == "rf":
        est = RandomForestRegressor(**base_params, n_jobs=-1, random_state=SEED)
        est.fit(data_views["X_oh"][tr_idx], y_tr)
        oof = est.predict(data_views["X_oh"][va_idx])
        test_pred = est.predict(data_views["X_test_oh"])
        del est
        return oof, test_pred

    elif mdl_name == "et":
        est = ExtraTreesRegressor(**base_params, n_jobs=-1, random_state=SEED)
        est.fit(data_views["X_oh"][tr_idx], y_tr)
        oof = est.predict(data_views["X_oh"][va_idx])
        test_pred = est.predict(data_views["X_test_oh"])
        del est
        return oof, test_pred

    elif mdl_name == "gbr":
        est = GradientBoostingRegressor(**base_params, random_state=SEED)
        est.fit(data_views["X_oh"][tr_idx], y_tr)
        oof = est.predict(data_views["X_oh"][va_idx])
        test_pred = est.predict(data_views["X_test_oh"])
        del est
        return oof, test_pred

    elif mdl_name in ["ridge","lasso","enet"]:
        # Linear family -> ordinal + standardize matrix
        if mdl_name == "ridge":
            est = Ridge(**base_params, random_state=SEED)
        elif mdl_name == "lasso":
            est = Lasso(**base_params, random_state=SEED, max_iter=10000)
        else:
            est = ElasticNet(**base_params, random_state=SEED, max_iter=10000)

        est.fit(data_views["X_ord_std"][tr_idx], y_tr)
        oof = est.predict(data_views["X_ord_std"][va_idx])
        test_pred = est.predict(data_views["X_test_ord_std"])
        del est
        return oof, test_pred

    else:
        raise ValueError(f"Unknown model {mdl_name}")


def _safe_lgbm_params(params, use_gpu: bool):
    """Ensure LGBM respects GPU limits; cap max_bin and, if still unsafe, fall back to CPU."""
    p = params.copy()
    if use_gpu:
        # GPU cannot run with max_bin > 255; also avoid tiny (<= 31) bins for stability
        if "max_bin" not in p or p["max_bin"] is None:
            p["max_bin"] = 255
        else:
            p["max_bin"] = int(min(max(63, p["max_bin"]), 255))
        p["device_type"] = "gpu"
    else:
        p["device_type"] = "cpu"
    return p

def default_spaces(mdl_name, use_gpu=True):
    rs = np.random.RandomState(SEED)

    if mdl_name == "xgb":
        return {
            "tree_method": "gpu_hist" if use_gpu else "hist",
            "predictor": "gpu_predictor" if use_gpu else "auto",
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_estimators": rs.randint(2000, 6001),
            "learning_rate": 10**rs.uniform(math.log10(0.01), math.log10(0.05)),
            "max_depth": int(rs.randint(4, 10)),
            "min_child_weight": float(rs.uniform(2, 20)),
            "subsample": float(rs.uniform(0.6, 0.95)),
            "colsample_bytree": float(rs.uniform(0.5, 0.95)),
            "lambda": 10**rs.uniform(math.log10(0.5), math.log10(5.0)),
            "alpha": float(rs.uniform(0.0, 2.0)),
            "grow_policy": "lossguide",
            "max_leaves": int(rs.randint(31, 256)),
        }

    if mdl_name == "lgbm":
        params = {
            "objective": "rmse",
            "n_estimators": rs.randint(2000, 6001),
            "learning_rate": 10**rs.uniform(math.log10(0.01), math.log10(0.05)),
            "num_leaves": int(rs.randint(31, 513)),
            "min_data_in_leaf": int(rs.randint(16, 257)),
            "feature_fraction": float(rs.uniform(0.6, 0.95)),
            "bagging_fraction": float(rs.uniform(0.6, 0.95)),
            "bagging_freq": int(rs.randint(1, 8)),
            "lambda_l1": 10**rs.uniform(math.log10(1e-3), math.log10(5.0)),
            "lambda_l2": 10**rs.uniform(math.log10(1e-3), math.log10(5.0)),
            "max_bin": int(rs.randint(63, 256)),  # <=255 for GPU
        }
        if rs.rand() < 0.5:
            mono = [(1 if f in ("curvature", "speed_limit") else 0) for f in lgb_features]
            params["monotone_constraints"] = mono
        return _safe_lgbm_params(params, use_gpu)

    if mdl_name == "cat":
        return {
            "iterations": rs.randint(3000, 6001),
            "depth": int(rs.randint(5, 11)),
            "learning_rate": float(rs.uniform(0.02, 0.06)),
            "loss_function": "RMSE",
            "l2_leaf_reg": float(rs.uniform(1.0, 15.0)),
            "bootstrap_type": "Bernoulli",
            "subsample": float(rs.uniform(0.6, 0.9)),
            "grow_policy": "Lossguide",
            "border_count": int(rs.randint(64, 255)),
            "random_seed": SEED,
            "verbose": False
        }

    if mdl_name == "hgb":
        return {
            "learning_rate": float(rs.uniform(0.02, 0.08)),
            "max_depth": None,
            "max_leaf_nodes": int(rs.randint(31, 256)),
            "min_samples_leaf": int(rs.randint(20, 257)),
            "l2_regularization": float(rs.uniform(0.0, 1.0)),
            "max_bins": int(rs.randint(127, 256)),
            "max_iter": int(rs.randint(600, 1201))
        }

    if mdl_name == "rf":
        return {
            "n_estimators": int(rs.randint(600, 1501)),
            "max_depth": None if rs.rand() < 0.5 else int(rs.randint(6, 30)),
            "min_samples_leaf": int(rs.randint(1, 11)),
            "max_features": float(rs.uniform(0.4, 1.0)) if rs.rand()<0.7 else "sqrt",
            "bootstrap": True
        }

    if mdl_name == "et":
        return {
            "n_estimators": int(rs.randint(600, 1501)),
            "max_depth": None if rs.rand() < 0.6 else int(rs.randint(6, 30)),
            "min_samples_leaf": int(rs.randint(1, 11)),
            "max_features": float(rs.uniform(0.4, 1.0)) if rs.rand()<0.7 else "sqrt"
        }

    if mdl_name == "gbr":
        return {
            "n_estimators": int(rs.randint(1500, 4001)),
            "learning_rate": float(rs.uniform(0.02, 0.08)),
            "max_depth": int(rs.randint(2, 7)),
            "subsample": float(rs.uniform(0.6, 0.95))
        }

    if mdl_name == "ridge":
        return {"alpha": 10**rs.uniform(math.log10(1e-4), math.log10(2.0))}

    if mdl_name == "lasso":
        return {"alpha": 10**rs.uniform(math.log10(5e-5), math.log10(5e-1))}

    if mdl_name == "enet":
        return {
            "alpha": 10**rs.uniform(math.log10(5e-5), math.log10(5e-1)),
            "l1_ratio": float(rs.uniform(0.05, 0.9)),
        }

    raise ValueError(mdl_name)


optuna = None
if TRY_OPTUNA:
    try:
        import optuna
    except Exception:
        optuna = None

def _sanitize_params_for_log(p: dict) -> dict:
    """Make params print-friendly (shorten very long fields)."""
    q = dict(p)  # shallow copy
    if "monotone_constraints" in q:
        mc = q["monotone_constraints"]
        try:
            q["monotone_constraints"] = f"<len={len(mc)}>"
        except Exception:
            q["monotone_constraints"] = "<list>"
    if "verbose" in q and isinstance(q["verbose"], bool):
        q["verbose"] = int(q["verbose"])
    return q

def _save_trial_artifacts(model_key: str, i: int, oof_pred: np.ndarray, test_pred: np.ndarray, model_dir: str):
    """Save OOF + SUB for a single trial, to both the per-model dir and the global submissions dir."""
    iter_tag = f"{model_key}__trial{i:03d}"
    # OOF (per-model dir)
    oof_path = os.path.join(model_dir, f"oof_{iter_tag}_v{VERSION}.csv")
    pd.DataFrame({ID_COL: train[ID_COL], TARGET: y, "oof_pred": oof_pred}).to_csv(oof_path, index=False)

    # SUB (per-model dir)
    sub_df = sample_sub.copy()
    sub_df[TARGET] = np.clip(test_pred, 0.0, 1.0)
    sub_path_model = os.path.join(model_dir, f"submission_{iter_tag}_v{VERSION}.csv")
    sub_df.to_csv(sub_path_model, index=False)

    # SUB (global submissions dir as well)
    sub_path_global = os.path.join(OUT_SUB_DIR, f"submission_{iter_tag}_v{VERSION}.csv")
    sub_df.to_csv(sub_path_global, index=False)

    return oof_path, sub_path_model, sub_path_global

def objective_once(mdl_name, use_gpu, params, data_views, return_preds=False):
    """Single OOF evaluation; optionally returns preds for saving."""
    oof_pred = np.zeros(len(train), dtype=float)
    test_pred_folds = np.zeros((len(test), N_SPLITS), dtype=float)

    # Safety for LGBM GPU params
    if mdl_name == "lgbm":
        params = _safe_lgbm_params(params, use_gpu)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train[ID_COL], y_bins), 1):
        if not time_left_ok():
            break
        try:
            oof, tpred = fit_predict_fold(mdl_name, use_gpu, tr_idx, va_idx, data_views, params)
        except lgb.basic.LightGBMError as e:
            if mdl_name == "lgbm" and use_gpu:
                params_cpu = params.copy(); params_cpu["device_type"] = "cpu"
                oof, tpred = fit_predict_fold(mdl_name, False, tr_idx, va_idx, data_views, params_cpu)
            else:
                raise e
        oof_pred[va_idx] = oof
        test_pred_folds[:, fold-1] = tpred
        gc.collect()

    # Handle partial fill
    if np.isnan(oof_pred).any():
        fillv = np.nanmean(oof_pred)
        oof_pred = np.nan_to_num(oof_pred, nan=fillv)

    score = rmse(y, oof_pred)
    if not return_preds:
        return score
    test_pred = np.nanmean(test_pred_folds, axis=1)
    return score, oof_pred, test_pred

def tune_model(mdl_name, use_gpu, budget, data_views, model_key):
    """
    Returns best_params, best_rmse, trial_rows (dicts with scores/params/paths).
    Saves OOF + SUB for each trial (both per-model dir and global submissions).
    """
    if not time_left_ok():
        return None, float("inf"), []

    trial_rows = []
    model_dir = os.path.join(OUT_HPO_DIR, f"{model_key}")
    os.makedirs(model_dir, exist_ok=True)
    log_txt_path = os.path.join(model_dir, f"hpo_log_{model_key}_v{VERSION}.txt")

    # ---- Optuna path ----
    if optuna is not None:
        def suggest_params(trial):
            if mdl_name == "xgb":
                return {
                    "tree_method": "gpu_hist" if use_gpu else "hist",
                    "predictor": "gpu_predictor" if use_gpu else "auto",
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "n_estimators": trial.suggest_int("n_estimators", 2000, 6000),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
                    "max_depth": trial.suggest_int("max_depth", 4, 10),
                    "min_child_weight": trial.suggest_float("min_child_weight", 2.0, 20.0),
                    "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
                    "lambda": trial.suggest_float("lambda", 0.5, 5.0, log=True),
                    "alpha": trial.suggest_float("alpha", 0.0, 2.0),
                    "grow_policy": "lossguide",
                    "max_leaves": trial.suggest_int("max_leaves", 31, 255),
                }
            if mdl_name == "lgbm":
                params = {
                    "objective": "rmse",
                    "n_estimators": trial.suggest_int("n_estimators", 2000, 6000),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 31, 512),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 16, 256),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.95),
                    "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
                    "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                    "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 5.0, log=True),
                    "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 5.0, log=True),
                    "max_bin": trial.suggest_int("max_bin", 63, 255),  # <=255 for GPU
                }
                if trial.suggest_categorical("use_monotone", [True, False]):
                    mono = [(1 if f in ("curvature","speed_limit") else 0) for f in lgb_features]
                    params["monotone_constraints"] = mono
                return _safe_lgbm_params(params, use_gpu)
            if mdl_name == "cat":
                return {
                    "iterations": trial.suggest_int("iterations", 3000, 6000),
                    "depth": trial.suggest_int("depth", 5, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.06),
                    "loss_function": "RMSE",
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
                    "bootstrap_type": "Bernoulli",
                    "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                    "grow_policy": "Lossguide",
                    "border_count": trial.suggest_int("border_count", 64, 254),
                    "random_seed": SEED,
                    "verbose": False
                }
            if mdl_name == "hgb":
                return {
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08),
                    "max_depth": None,
                    "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 31, 255),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 256),
                    "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
                    "max_bins": trial.suggest_int("max_bins", 127, 255),
                    "max_iter": trial.suggest_int("max_iter", 600, 1200),
                }
            if mdl_name == "rf":
                md = trial.suggest_categorical("max_depth", [None] + list(range(6, 31)))
                mf = trial.suggest_categorical("max_features", ["sqrt"]) if trial.suggest_float("mf_switch", 0, 1) < 0.3 else trial.suggest_float("max_features_f", 0.4, 1.0)
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 600, 1500),
                    "max_depth": md,
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "max_features": mf,
                    "bootstrap": True
                }
            if mdl_name == "et":
                md = trial.suggest_categorical("max_depth", [None] + list(range(6, 31)))
                mf = trial.suggest_categorical("max_features", ["sqrt"]) if trial.suggest_float("mf_switch", 0, 1) < 0.3 else trial.suggest_float("max_features_f", 0.4, 1.0)
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 600, 1500),
                    "max_depth": md,
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "max_features": mf
                }
            if mdl_name == "gbr":
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 1500, 4000),
                    "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08),
                    "max_depth": trial.suggest_int("max_depth", 2, 6),
                    "subsample": trial.suggest_float("subsample", 0.6, 0.95)
                }
            if mdl_name == "ridge":
                return {"alpha": trial.suggest_float("alpha", 1e-4, 2.0, log=True)}
            if mdl_name == "lasso":
                return {"alpha": trial.suggest_float("alpha", 5e-5, 5e-1, log=True)}
            if mdl_name == "enet":
                return {
                    "alpha": trial.suggest_float("alpha", 5e-5, 5e-1, log=True),
                    "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.9)
                }
            raise ValueError(mdl_name)

        study = optuna.create_study(direction="minimize")
        n_trials = int(HPO_BUDGET.get(mdl_name, 30))

        for i in range(n_trials):
            if not time_left_ok():
                break
            t0 = time.time()
            try:
                trial = study.ask()
                params = suggest_params(trial)
                pretty_params = json.dumps(_sanitize_params_for_log(params), sort_keys=True)
                print(f"[TRIAL] {model_key} #{i:03d} params={pretty_params}")
                score, oof_pred, test_pred = objective_once(mdl_name, use_gpu, params, DATA_VIEWS, return_preds=True)
                study.tell(trial, score)
            except Exception as e:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                print(f"[WARN] {model_key} trial {i} failed: {e}")
                with open(log_txt_path, "a") as f:
                    f.write(f"[FAIL] trial={i} err={e}\n")
                continue

            # SAVE per-trial artifacts (OOF + SUB, both locations)
            oof_path, sub_path_model, sub_path_global = _save_trial_artifacts(model_key, i, oof_pred, test_pred, model_dir)

            # Metrics + print + log
            mets = evaluate_metrics(y, oof_pred)
            took = round(time.time() - t0, 2)
            print(f"[SCORE] {model_key} #{i:03d} rmse={mets['rmse']:.6f} mae={mets['mae']:.6f} r2={mets['r2']:.4f} time={took}s")
            trial_rows.append({
                "model": model_key, "trial": i,
                "rmse": mets["rmse"], "mae": mets["mae"], "r2": mets["r2"],
                "time_sec": took,
                "params": json.dumps(params),
                "oof_path": oof_path,
                "sub_path_model": sub_path_model,
                "sub_path_global": sub_path_global
            })
            pd.DataFrame(trial_rows).to_csv(os.path.join(model_dir, f"hpo_trials_{model_key}_v{VERSION}.csv"), index=False)
            with open(log_txt_path, "a") as f:
                f.write(f"[TRIAL] i={i} params={pretty_params}\n")
                f.write(f"[SCORE] i={i} rmse={mets['rmse']:.6f} mae={mets['mae']:.6f} r2={mets['r2']:.4f} time={took}s\n")
                f.write(f"[FILES] oof={oof_path} sub_model={sub_path_model} sub_global={sub_path_global}\n")

        if len(study.trials) == 0:
            return None, float("inf"), trial_rows
        return study.best_params, study.best_value, trial_rows

    # ---- Random Search fallback ----
    best_params = None
    best_rmse = float("inf")
    n_trials = int(HPO_BUDGET.get(mdl_name, 30))
    for i in range(n_trials):
        if not time_left_ok():
            break
        t0 = time.time()
        params = default_spaces(mdl_name, use_gpu)
        pretty_params = json.dumps(_sanitize_params_for_log(params), sort_keys=True)
        print(f"[TRIAL] {model_key} #{i:03d} params={pretty_params}")

        try:
            score, oof_pred, test_pred = objective_once(mdl_name, use_gpu, params, DATA_VIEWS, return_preds=True)
        except Exception as e:
            print(f"[WARN] {model_key} trial {i} failed: {e}")
            with open(log_txt_path, "a") as f:
                f.write(f"[FAIL] trial={i} err={e}\n")
            continue

        if score < best_rmse:
            best_rmse = score
            best_params = params

        # SAVE per-trial artifacts (OOF + SUB, both locations)
        oof_path, sub_path_model, sub_path_global = _save_trial_artifacts(model_key, i, oof_pred, test_pred, model_dir)

        mets = evaluate_metrics(y, oof_pred)
        took = round(time.time() - t0, 2)
        print(f"[SCORE] {model_key} #{i:03d} rmse={mets['rmse']:.6f} mae={mets['mae']:.6f} r2={mets['r2']:.4f} time={took}s")
        trial_rows.append({
            "model": model_key, "trial": i,
            "rmse": mets["rmse"], "mae": mets["mae"], "r2": mets["r2"],
            "time_sec": took,
            "params": json.dumps(params),
            "oof_path": oof_path,
            "sub_path_model": sub_path_model,
            "sub_path_global": sub_path_global
        })
        pd.DataFrame(trial_rows).to_csv(os.path.join(model_dir, f"hpo_trials_{model_key}_v{VERSION}.csv"), index=False)
        with open(log_txt_path, "a") as f:
            f.write(f"[TRIAL] i={i} params={pretty_params}\n")
            f.write(f"[SCORE] i={i} rmse={mets['rmse']:.6f} mae={mets['mae']:.6f} r2={mets['r2']:.4f} time={took}s\n")
            f.write(f"[FILES] oof={oof_path} sub_model={sub_path_model} sub_global={sub_path_global}\n")

    return best_params, best_rmse, trial_rows


results_rows = []

# Bundle data views to pass around cleanly
DATA_VIEWS = {
    "X_oh": X_oh, "X_test_oh": X_test_oh,
    "X_ord_std": X_ord_std, "X_test_ord_std": X_test_ord_std,
    "train_lgb": train_lgb, "test_lgb": test_lgb, "lgb_features": lgb_features,
    "train_df": train, "test_df": test, "features": features,
    "cat_idx_for_catboost": cat_idx_for_catboost
}

for mdl_name in MODELS_TO_RUN:
    if not time_left_ok():
        print(f"[STOP] Time limit reached at {now_min()} min. Exiting HPO loop.")
        break

    use_gpu = mdl_name in GPU_MODEL_SET
    model_key = f"{mdl_name}{'_gpu' if use_gpu else '_cpu'}"
    budget = int(HPO_BUDGET.get(mdl_name, 30))
    print(f"\n[HPO] {model_key} | time={now_min()} min | budget={budget}")

    # ---- Hyperparameter tuning (also saves per-trial OOF/SUB) ----
    best_params, best_rmse, trial_rows = tune_model(
        mdl_name, use_gpu, budget, DATA_VIEWS, model_key
    )
    if best_params is None:
        print(f"[SKIP] No params found for {model_key} (likely cutoff).")
        continue

    # Persist best params
    hpo_json = os.path.join(OUT_HPO_DIR, f"hpo_{model_key}_v{VERSION}.json")
    with open(hpo_json, "w") as f:
        json.dump({"best_params": best_params, "oof_rmse": best_rmse}, f, indent=2)
    print(f"[HPO] Best RMSE={best_rmse:.6f} | saved -> {hpo_json}")

    # ---- Final OOF/Test with best params ----
    if not time_left_ok():
        print(f"[STOP] Time limit reached at {now_min()} min before final fit; skipping {model_key}.")
        break

    oof_pred = np.zeros(len(train), dtype=float)
    test_pred_folds = np.zeros((len(test), N_SPLITS), dtype=float)

    # LGBM GPU safety: clamp params and fallback to CPU if needed
    final_params = best_params.copy()
    if mdl_name == "lgbm":
        try:
            final_params = _safe_lgbm_params(final_params, use_gpu)
        except NameError:
            # if helper isn't present (older cell), do minimal clamping here
            final_params = final_params.copy()
            if use_gpu:
                final_params["device_type"] = "gpu"
                # GPU requires max_bin <= 255
                if final_params.get("max_bin", 255) > 255:
                    final_params["max_bin"] = 255

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train[ID_COL], y_bins), 1):
        if not time_left_ok():
            print(f"[STOP] Time limit reached during final CV for {model_key}.")
            break

        try:
            oof, tpred = fit_predict_fold(mdl_name, use_gpu, tr_idx, va_idx, DATA_VIEWS, final_params)
        except Exception as e:
            # targeted GPU→CPU fallback for LightGBM
            if mdl_name == "lgbm" and use_gpu:
                print(f"[WARN] {model_key} fold {fold}: GPU error -> retry on CPU. err={e}")
                params_cpu = final_params.copy()
                params_cpu["device_type"] = "cpu"
                oof, tpred = fit_predict_fold(mdl_name, False, tr_idx, va_idx, DATA_VIEWS, params_cpu)
            else:
                raise e

        oof_pred[va_idx] = oof
        test_pred_folds[:, fold-1] = tpred
        gc.collect()

    test_pred = np.nanmean(test_pred_folds, axis=1)
    if np.isnan(oof_pred).any():
        oof_pred = np.nan_to_num(oof_pred, nan=np.nanmean(oof_pred))

    oof_metrics = evaluate_metrics(y, oof_pred)

    # Save best-run OOF + SUB
    oof_path, sub_path = save_oof_and_submission(model_key, oof_pred, test_pred)

    print(f"[DONE] {model_key} | RMSE={oof_metrics['rmse']:.6f} | MAE={oof_metrics['mae']:.6f} | R2={oof_metrics['r2']:.4f}")
    print(f"       OOF: {oof_path} | SUB: {sub_path}")

    row = dict(model=model_key, folds=N_SPLITS, **oof_metrics, time_min=now_min(), hpo_rmse=best_rmse)
    results_rows.append(row)

# ---- Per-model and global trial logs (merge into a single CSV) ----
try:
    all_trial_csvs = []
    for mk in [f"{m}{'_gpu' if (m in GPU_MODEL_SET) else '_cpu'}" for m in MODELS_TO_RUN]:
        p = os.path.join(OUT_HPO_DIR, mk, f"hpo_trials_{mk}_v{VERSION}.csv")
        if os.path.exists(p):
            all_trial_csvs.append(pd.read_csv(p))
    if all_trial_csvs:
        all_trials_df = pd.concat(all_trial_csvs, ignore_index=True)
        all_trials_path = os.path.join(OUT_HPO_DIR, f"all_trials_v{VERSION}.csv")
        all_trials_df.to_csv(all_trials_path, index=False)
        print(f"[HPO] Saved all trial logs -> {all_trials_path}")
except Exception as e:
    print(f"[WARN] Could not save combined trials CSV: {e}")

# ---- Leaderboard + plot ----
if len(results_rows) > 0:
    results_df = pd.DataFrame(results_rows).sort_values("rmse")
    res_csv_path = os.path.join(OUT_RES_DIR, f"results_01_v{VERSION}.csv")
    results_df.to_csv(res_csv_path, index=False)
    print(f"\n[RESULTS] Saved leaderboard -> {res_csv_path}")
    display(results_df.head(20))

    top = results_df.nsmallest(10, "rmse")
    plt.figure(figsize=(9,6))
    plt.barh(range(len(top)), top["rmse"].values)
    plt.yticks(range(len(top)), top["model"].values, fontsize=9)
    plt.gca().invert_yaxis()
    plt.title(f"Top-10 by RMSE (HPO v{VERSION})")
    plt.xlabel("OOF RMSE")
    plt.tight_layout()
    fig_path = os.path.join(OUT_RES_DIR, f"results_01_v{VERSION}.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] Saved -> {fig_path}")
else:
    print("[WARN] No models completed before cutoff.")


if len(results_rows) > 0:
    results_df = pd.DataFrame(results_rows).sort_values("rmse")
    res_csv_path = os.path.join(OUT_RES_DIR, f"results_01_v{VERSION}.csv")
    results_df.to_csv(res_csv_path, index=False)
    print(f"\n[RESULTS] Saved leaderboard -> {res_csv_path}")
    display(results_df.head(20))

    top = results_df.nsmallest(10, "rmse")
    plt.figure(figsize=(9,6))
    plt.barh(range(len(top)), top["rmse"].values)
    plt.yticks(range(len(top)), top["model"].values, fontsize=9)
    plt.gca().invert_yaxis()
    plt.title(f"Top-10 by RMSE (HPO v{VERSION})")
    plt.xlabel("OOF RMSE")
    plt.tight_layout()
    fig_path = os.path.join(OUT_RES_DIR, f"results_01_v{VERSION}.png")
    plt.savefig(fig_path, bbox_inches="tight"); plt.show()
    print(f"[PLOT] Saved -> {fig_path}")
else:
    print("[WARN] No models completed before cutoff.")

trial_glob = os.path.join(OUT_HPO_DIR, f"all_trials_v{VERSION}.csv")
if os.path.exists(trial_glob):
    trials_df = pd.read_csv(trial_glob)
    print("\n[HPO] Top 10 trials across all models:")
    display(trials_df.sort_values("rmse").head(10))




