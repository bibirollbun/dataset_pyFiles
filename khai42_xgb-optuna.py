import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import warnings
warnings.filterwarnings('ignore')
import os, glob, json, math, gc, subprocess
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb


!pip -q install optuna catboost


SEED            = 42
FOLDS           = 5
SEEDS_FINAL     = [42, 2025]        
N_BINS          = 20                
LOG_SKEW_TH     = 1.0               
CLIP_QUANTILES  = (0.005, 0.995)    
np.random.seed(SEED)

def print_env():
    import platform
    print(f"Python: {platform.python_version()} | xgboost: {xgb.__version__}")
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"]).decode().strip()
        print("GPUs:\n" + out)
    except Exception:
        print("GPUs nvidia not available")

def _xgb_ge_20():
    try:
        major = int(xgb.__version__.split(".")[0])
        return major >= 2
    except Exception:
        return False

XGB_GE_20 = _xgb_ge_20()

def xgb_gpu_args():
    if XGB_GE_20:
        return {"tree_method": "hist", "device": "cuda"}
    else:
        return {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}

def safe_predict(model, X):
    best_it = getattr(model, "best_iteration", None)
    if best_it is not None:
        try:
            return model.predict(X, iteration_range=(0, best_it + 1))
        except TypeError:
            pass
    best_ntree = getattr(model, "best_ntree_limit", None)
    if best_ntree is not None:
        try:
            return model.predict(X, ntree_limit=best_ntree)
        except TypeError:
            pass
    return model.predict(X)

print_env()

def auto_path(filename: str) -> str:
    if os.path.exists(filename): 
        return filename
    m = glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
    if not m:
        raise FileNotFoundError(f"Could not locate {filename}. Add your dataset via 'Add data'.")
    return m[0]

train_path = auto_path("/kaggle/input/playground-series-s5e9/train.csv")
test_path  = auto_path("/kaggle/input/playground-series-s5e9/test.csv")
sub_path   = auto_path("/kaggle/input/playground-series-s5e9/sample_submission.csv")

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)
sample_sub = pd.read_csv(sub_path)

id_col = "id" if "id" in train.columns else train.columns[0]
target_col = "BeatsPerMinute" if "BeatsPerMinute" in train.columns else \
             [c for c in train.columns if c != id_col][-1]
feature_cols = [c for c in train.columns if c not in [id_col, target_col]]

def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred, squared=False)

def strat_bins(y: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    n_bins = max(2, min(n_bins, len(np.unique(y))))
    return pd.qcut(y, q=n_bins, labels=False, duplicates="drop").astype(int)

def preprocess_fit(df: pd.DataFrame, feature_cols: list) -> Dict[str, Any]:
    stats = {"clip": {}, "log_cols": []}
    numerics = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    q_low, q_high = CLIP_QUANTILES
    q = numerics.quantile([q_low, q_high])
    for c in feature_cols:
        lo = float(q.loc[q_low, c])
        hi = float(q.loc[q_high, c])
        stats["clip"][c] = (lo, hi)

    skew_s = numerics.skew(numeric_only=True)
    for c in feature_cols:
        col = numerics[c]
        if col.min() >= 0 and float(skew_s.get(c, 0.0)) >= LOG_SKEW_TH:
            stats["log_cols"].append(c)

    return stats

def preprocess_apply(df: pd.DataFrame, feature_cols: list, stats: Dict[str, Any]) -> pd.DataFrame:
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    # winsorize
    for c, (lo, hi) in stats["clip"].items():
        X[c] = X[c].clip(lower=lo, upper=hi)
    # log1p
    for c in stats["log_cols"]:
        X[c] = np.log1p(np.maximum(X[c], 0))
    return X.astype(np.float32)

pp_stats = preprocess_fit(train, feature_cols)
X = preprocess_apply(train, feature_cols, pp_stats)
y = train[target_col].astype(np.float32).values
X_test = preprocess_apply(test, feature_cols, pp_stats)

sub_target_cols = [c for c in sample_sub.columns if c != id_col]
assert len(sub_target_cols) == 1, "sample_submission.csv must have exactly one target col besides id"
sub_target_col = sub_target_cols[0]

meta = {
    "id_col": id_col,
    "target_col": target_col,
    "sub_target_col": sub_target_col,
    "feature_cols": feature_cols,
}
with open("/kaggle/working/columns_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"Train: {train.shape} | Test: {test.shape} | Features: {len(feature_cols)}")
print(f"ID: {id_col} | Target: {target_col} Sub target: {sub_target_col}")
print(f"Preprocess log1p {len(pp_stats['log_cols'])} features {CLIP_QUANTILES}")



import optuna

TUNE_MAX_ROWS   = 250_000 
TUNE_FOLDS      = 3
N_TRIALS        = 30      
EARLY_STOP      = 200

if len(train) > TUNE_MAX_ROWS:
    idx = np.random.RandomState(SEED).choice(len(train), size=TUNE_MAX_ROWS, replace=False)
    X_tune = X.iloc[idx].reset_index(drop=True)
    y_tune = y[idx]
else:
    X_tune, y_tune = X, y

y_bins_tune = strat_bins(y_tune, n_bins=N_BINS)
skf_tune = StratifiedKFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=SEED)

def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": 20000,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 6, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 8.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1e-1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "random_state": SEED,
        "n_jobs": -1,
    }
    params.update(xgb_gpu_args())

    rmses = []
    for trn_idx, val_idx in skf_tune.split(X_tune, y_bins_tune):
        X_tr, X_va = X_tune.iloc[trn_idx], X_tune.iloc[val_idx]
        y_tr, y_va = y_tune[trn_idx], y_tune[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            verbose=False,
            early_stopping_rounds=EARLY_STOP
        )

        y_hat = safe_predict(model, X_va)
        rmses.append(rmse(y_va, y_hat))

        del X_tr, X_va, y_tr, y_va, y_hat
        gc.collect()

    return float(np.mean(rmses))

pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0)
study = optuna.create_study(direction="minimize", pruner=pruner)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params = study.best_trial.params
print("\nBest params (Optuna):", best_params)
with open("/kaggle/working/best_params_xgb.json", "w") as f:
    json.dump(best_params, f, indent=2)
print("/kaggle/working/best_params_xgb.json")


with open("/kaggle/working/best_params_xgb.json") as f:
    best = json.load(f)
with open("/kaggle/working/columns_meta.json") as f:
    meta = json.load(f)
id_col, sub_target_col = meta["id_col"], meta["sub_target_col"]

final_params = {
    "n_estimators": 20000,
    "learning_rate": best.get("learning_rate", 0.05),
    "max_depth": int(best.get("max_depth", 8)),
    "subsample": best.get("subsample", 0.8),
    "colsample_bytree": best.get("colsample_bytree", 0.8),
    "min_child_weight": best.get("min_child_weight", 1.0),
    "reg_alpha": best.get("reg_alpha", 0.0),
    "reg_lambda": best.get("reg_lambda", 2.0),
    "gamma": best.get("gamma", 0.0),
    "n_jobs": -1,
}
final_params.update(xgb_gpu_args())

y_bins_full = strat_bins(y, n_bins=N_BINS)
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

all_oof = np.zeros(len(X), dtype=np.float32)
all_oof_seeds = []
test_preds_seeds = []
fold_rmse_seeds = []
fi_gain_sum = pd.Series(0.0, index=X.columns, dtype=float)

for seed in SEEDS_FINAL:
    seed_oof = np.zeros(len(X), dtype=np.float32)
    test_preds_folds = []
    fold_rmses = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y_bins_full), start=1):
        X_tr, X_va = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_va = y[trn_idx], y[val_idx]

        params = final_params.copy()
        params["random_state"] = seed

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            verbose=False,
            early_stopping_rounds=300
        )

        val_pred = safe_predict(model, X_va).astype(np.float32)
        seed_oof[val_idx] = val_pred
        score = rmse(y_va, val_pred)
        fold_rmses.append(score)
        print(f"[Seed {seed}] Fold {fold}/{FOLDS} RMSE: {score:.5f}")

        test_pred_fold = safe_predict(model, X_test).astype(np.float32)
        test_preds_folds.append(test_pred_fold)

        booster = model.get_booster()
        gain_map = booster.get_score(importance_type="gain")
        if set(gain_map.keys()) <= set(X.columns):
            gain_series = pd.Series(gain_map, dtype=float)
        else:
            names = booster.feature_names
            gain_vals = [gain_map.get(n, 0.0) for n in names]
            gain_series = pd.Series(gain_vals, index=names, dtype=float)
        gain_series = gain_series.reindex(X.columns).fillna(0.0)
        fi_gain_sum = fi_gain_sum.add(gain_series, fill_value=0.0)

        del X_tr, X_va, y_tr, y_va, model
        gc.collect()

    test_pred_seed = np.mean(np.vstack(test_preds_folds), axis=0).astype(np.float32)
    test_preds_seeds.append(test_pred_seed)
    all_oof_seeds.append(seed_oof)
    fold_rmse_seeds.append(fold_rmses)
    print(f"[Seed {seed}] Mean RMSE over folds: {np.mean(fold_rmses):.5f} ± {np.std(fold_rmses):.5f}")

    all_oof += seed_oof / len(SEEDS_FINAL)

oof_rmse = rmse(y, all_oof)
print(f"\n XGBoost OOF RMSE {oof_rmse:.5f}")


test_pred = np.mean(np.vstack(test_preds_seeds), axis=0).astype(np.float32)
submission = sample_sub.copy()
submission[sub_target_col] = test_pred
sub_out = "/kaggle/working/submission.csv"
submission.to_csv(sub_out, index=False)




