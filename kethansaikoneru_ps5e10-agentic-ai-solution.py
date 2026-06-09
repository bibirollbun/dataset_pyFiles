!pip install -U scikit-learn


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BASE_DIR = "task/playground-series-s5e10" if not os.getenv('KAGGLE_KERNEL_RUN_TYPE') else "/kaggle/input/playground-series-s5e10"
DEBUG = True  # Top-level flag; the pipeline will run sequentially twice: DEBUG=True then DEBUG=False

# Checklist (plan)
# - Build GroupKFold grouped by exact-X keys to prevent duplicate-X leakage; fixed seed and stratified-like quantiles via grouping
# - Engineer validated interactions and one-hot low-cardinality categoricals; align train/test features; clip [0,1]
# - Train XGBoost (CUDA), LightGBM (CPU), CatBoost (GPU); log per-fold RMSE and OOF; guard first-epoch metric on fold 0
# - Stack model OOF with Ridge; apply isotonic calibration; log final calibrated OOF
# - Run sequentially in DEBUG then FULL mode; write logs and produce a submission

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

import gc
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool


def ensure_dirs_and_logging():
    out_dir = "." # os.path.join(BASE_DIR, "outputs", "5")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "code_5_v14.txt")
    root_logger = logging.getLogger()
    # Remove existing FileHandlers to avoid duplicate logs into previous files
    for h in list(root_logger.handlers):
        if isinstance(h, logging.FileHandler):
            root_logger.removeHandler(h)
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    root_logger.addHandler(fh)
    return out_dir, log_path


def set_seed(seed: int = 42):
    np.random.seed(seed)


def load_comp_data():
    train_path = os.path.join(BASE_DIR, "train.csv")
    test_path = os.path.join(BASE_DIR, "test.csv")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def build_group_keys(df_features_raw: pd.DataFrame) -> np.ndarray:
    # Build exact-feature key to group duplicates deterministically
    df_key = df_features_raw.copy()
    for c in df_key.columns:
        if pd.api.types.is_float_dtype(df_key[c]):
            df_key[c] = df_key[c].map(lambda v: f"{v:.6f}")
        else:
            df_key[c] = df_key[c].astype(str)
    key_hash = pd.util.hash_pandas_object(df_key, index=False).astype(np.int64).values
    return key_hash


def compute_external_te_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    # Use external "simulated-roads-accident-data" means as TE-like predictors, centered by mean offset
    ext_dir = "/kaggle/input/simulated-roads-accident-data" # os.path.join(BASE_DIR, "external-data", "simulated-roads-accident-data")
    ext_candidates = [
        "synthetic_road_accidents_100k.csv",
        "synthetic_road_accidents_10k.csv",
        "synthetic_road_accidents_2k.csv",
    ]
    ext_path = None
    for fn in ext_candidates:
        p = os.path.join(ext_dir, fn)
        if os.path.exists(p):
            ext_path = p
            break
    if ext_path is None:
        return pd.DataFrame(index=train_df.index), pd.DataFrame(index=test_df.index)

    ext = pd.read_csv(ext_path)
    target_col = "accident_risk"
    cols_needed = ["lighting", "weather", "speed_limit"]
    present = [c for c in cols_needed if (c in train_df.columns and c in ext.columns)]
    if len(present) == 0 or target_col not in ext.columns:
        return pd.DataFrame(index=train_df.index), pd.DataFrame(index=test_df.index)

    train_mean = float(train_df[target_col].mean()) if target_col in train_df.columns else 0.352
    ext_mean = float(ext[target_col].mean())
    delta = ext_mean - train_mean

    te_train = pd.DataFrame(index=train_df.index)
    te_test = pd.DataFrame(index=test_df.index)

    for feat in present:
        means = ext.groupby(feat)[target_col].mean()
        means_adj = means - delta
        tr_map = train_df[feat].map(means_adj).fillna(train_mean)
        te_map = test_df[feat].map(means_adj).fillna(train_mean)
        te_train[f"te_ext_{feat}"] = tr_map.values.astype(np.float32)
        te_test[f"te_ext_{feat}"] = te_map.values.astype(np.float32)

    return te_train, te_test


def feature_engineer(train_df: pd.DataFrame, test_df: pd.DataFrame):
    target_col = "accident_risk"
    id_col = "id"

    te_train_ext, te_test_ext = compute_external_te_features(train_df, test_df)

    def add_engineered(df):
        df = df.copy()
        # Booleans -> int
        for c in df.columns:
            if df[c].dtype == bool:
                df[c] = df[c].astype(np.int8)

        # Cast numeric base columns when present
        for c in ["speed_limit", "curvature", "num_lanes", "num_reported_accidents"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Numeric engineered
        if "speed_limit" in df.columns:
            df["speed_limit_num"] = df["speed_limit"]
        if "curvature" in df.columns and "speed_limit_num" in df.columns:
            df["curv_speed"] = df["curvature"] * df["speed_limit_num"]
            df["high_speed"] = (df["speed_limit_num"] >= 60).astype(np.int8)
            df["high_speed_curv"] = df["curvature"] * df["high_speed"].astype(np.float32)
            df["curvature2"] = df["curvature"] ** 2
            df["curvature3"] = df["curvature"] ** 3
        if "num_lanes" in df.columns and "speed_limit_num" in df.columns:
            df["lane_speed_risk"] = (5 - df["num_lanes"]) * df["speed_limit_num"]
        if "num_reported_accidents" in df.columns and "num_lanes" in df.columns:
            df["accidents_per_lane"] = df["num_reported_accidents"] / (df["num_lanes"] + 1.0)

        # One-hot encode low-card cats; ensure matching columns exist
        cat_cols = [c for c in ["lighting", "weather", "time_of_day", "road_type", "speed_limit"] if c in df.columns]
        if len(cat_cols) > 0:
            df = pd.get_dummies(df, columns=cat_cols, dtype=np.uint8)

        # Interactions with high_speed for selected cats
        if "high_speed" in df.columns:
            lighting_cols = [c for c in df.columns if c.startswith("lighting_")]
            weather_cols = [c for c in df.columns if c.startswith("weather_")]
            for lc in lighting_cols:
                df[f"{lc}__x__high_speed"] = df[lc].astype(np.int8) * df["high_speed"].astype(np.int8)
            for wc in weather_cols:
                df[f"{wc}__x__high_speed"] = df[wc].astype(np.int8) * df["high_speed"].astype(np.int8)

        # Drop any residual non-numeric except id/target
        to_drop = [c for c in df.columns if c not in [target_col, id_col] and not pd.api.types.is_numeric_dtype(df[c])]
        if len(to_drop) > 0:
            df = df[[c for c in df.columns if c not in to_drop]]

        return df

    trX = add_engineered(train_df.drop(columns=[target_col], errors='ignore'))
    teX = add_engineered(test_df.copy())

    if te_train_ext.shape[1] > 0:
        trX = trX.join(te_train_ext)
        teX = teX.join(te_test_ext)

    # Align features: union of columns, excluding id
    all_cols = sorted(set(trX.columns).union(set(teX.columns)))
    feat_cols = [c for c in all_cols if c != "id"]
    trX = trX.reindex(columns=feat_cols, fill_value=0)
    teX = teX.reindex(columns=feat_cols, fill_value=0)

    y = train_df[target_col].values.astype(np.float32)
    ids_test = test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
    return trX, y, teX, ids_test, feat_cols


class XGBFirstEpochGuard(xgb.callback.TrainingCallback):
    def __init__(self, enable_check: bool, mode_tag: str):
        super().__init__()
        self.enable_check = enable_check
        self.mode_tag = mode_tag

    def after_iteration(self, model, epoch, evals_log):
        if not self.enable_check:
            return False
        if epoch == 0:
            first_val = None
            # evals_log: {dataset: {metric: [values]}}
            for dname, metrics in evals_log.items():
                if "valid" in dname or "eval" in dname:
                    for _, vals in metrics.items():
                        if len(vals) > 0:
                            first_val = vals[0]
                            break
                if first_val is not None:
                    break
            if first_val is not None and (np.isnan(first_val) or float(first_val) == 0.0):
                raise Exception(f"[{self.mode_tag}] Guard: First-epoch validation metric invalid for XGBoost (value={first_val})")
        return False


def lgb_first_epoch_guard(fold_zero_check: bool, mode_tag: str):
    def _callback(env):
        if not fold_zero_check:
            return
        if env.iteration == 0:
            val_metric = None
            for data_name, eval_name, result, _ in env.evaluation_result_list:
                if "valid" in data_name and eval_name in ("rmse", "l2_root"):
                    val_metric = result
                    break
            if val_metric is None and len(env.evaluation_result_list) > 0:
                val_metric = env.evaluation_result_list[0][2]
            if val_metric is not None and (np.isnan(val_metric) or float(val_metric) == 0.0):
                raise Exception(f"[{mode_tag}] Guard: First-epoch validation metric invalid for LightGBM (value={val_metric})")
    _callback.order = 10
    return _callback


def prepare_catboost_frames(train_df: pd.DataFrame, test_df: pd.DataFrame):
    target_col = "accident_risk"
    id_col = "id"
    cat_cols = [c for c in ["lighting", "weather", "time_of_day", "road_type"] if c in train_df.columns]

    def add_engineered_cat(df):
        df = df.copy()
        # Booleans -> int
        for c in df.columns:
            if df[c].dtype == bool:
                df[c] = df[c].astype(np.int8)
        # Cast numerics
        for c in ["speed_limit", "curvature", "num_lanes", "num_reported_accidents"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["speed_limit_num"] = df["speed_limit"] if "speed_limit" in df.columns else 0.0
        # Engineered numerics
        if "curvature" in df.columns and "speed_limit_num" in df.columns:
            df["curv_speed"] = df["curvature"] * df["speed_limit_num"]
            df["high_speed"] = (df["speed_limit_num"] >= 60).astype(np.int8)
            df["high_speed_curv"] = df["curvature"] * df["high_speed"].astype(np.float32)
            df["curvature2"] = df["curvature"] ** 2
            df["curvature3"] = df["curvature"] ** 3
        if "num_lanes" in df.columns and "speed_limit_num" in df.columns:
            df["lane_speed_risk"] = (5 - df["num_lanes"]) * df["speed_limit_num"]
        if "num_reported_accidents" in df.columns and "num_lanes" in df.columns:
            df["accidents_per_lane"] = df["num_reported_accidents"] / (df["num_lanes"] + 1.0)
        return df

    tr = add_engineered_cat(train_df.drop(columns=[target_col], errors="ignore"))
    te = add_engineered_cat(test_df.copy())

    feat_cols = [c for c in tr.columns if c != id_col]
    te = te.reindex(columns=feat_cols, fill_value=0)
    tr = tr[feat_cols]

    cat_idx = [tr.columns.get_loc(c) for c in cat_cols if c in tr.columns]
    return tr, te, cat_idx


def fit_catboost_oof(train_df: pd.DataFrame, test_df: pd.DataFrame, n_splits: int, seed: int, mode_tag: str, debug_flag: bool):
    target_col = "accident_risk"
    base_feat_cols = [c for c in train_df.columns if c not in ["id", target_col]]
    groups = build_group_keys(train_df[base_feat_cols])

    X_tr, X_te, cat_idx = prepare_catboost_frames(train_df, test_df)
    y = train_df[target_col].astype(np.float32).values

    oof = np.zeros(len(train_df), dtype=np.float32)
    test_pred = np.zeros(len(test_df), dtype=np.float32)

    gkf = GroupKFold(n_splits=n_splits)
    params = dict(
        loss_function="RMSE",
        depth=7,
        learning_rate=0.03,
        l2_leaf_reg=10.0,
        bagging_temperature=0.7,
        random_seed=seed,
        iterations=(4 if debug_flag else 6000),
        od_type="Iter",
        od_wait=(10 if debug_flag else 300),
        verbose=False,
        task_type="GPU"
    )

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y, groups=groups)):
        tr_pool = Pool(
            data=X_tr.iloc[tr_idx],
            label=y[tr_idx],
            cat_features=cat_idx
        )
        va_pool = Pool(
            data=X_tr.iloc[va_idx],
            label=y[va_idx],
            cat_features=cat_idx
        )
        te_pool = Pool(
            data=X_te,
            cat_features=cat_idx
        )

        model = CatBoostRegressor(**params)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)

        # Guard: check first-epoch validation metric for fold 0
        if fold_idx == 0:
            evals = model.get_evals_result()
            val_hist = None
            if "validation" in evals and "RMSE" in evals["validation"]:
                val_hist = evals["validation"]["RMSE"]
            elif "learn" in evals and "RMSE" in evals["learn"]:
                val_hist = evals["learn"]["RMSE"]
            if val_hist is not None and len(val_hist) > 0:
                first_val = float(val_hist[0])
                if np.isnan(first_val) or first_val == 0.0:
                    raise Exception(f"[{mode_tag}] Guard: First-epoch validation metric invalid for CatBoost (value={first_val})")

        pred_va = np.clip(model.predict(va_pool), 0.0, 1.0).astype(np.float32)
        oof[va_idx] = pred_va
        rmse_fold = root_mean_squared_error(y[va_idx], pred_va)
        logging.info(f"[{mode_tag}] Fold {fold_idx+1} CatBoost RMSE: {rmse_fold:.6f}")

        test_pred += np.clip(model.predict(te_pool), 0.0, 1.0).astype(np.float32) / n_splits

        del model, tr_pool, va_pool, te_pool
        gc.collect()

    rmse_oof = root_mean_squared_error(y, oof)
    logging.info(f"[{mode_tag}] OOF RMSE CatBoost: {rmse_oof:.6f}")
    return oof, test_pred, rmse_oof


def run_cv_training(X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, raw_train_df: pd.DataFrame,
                    mode_tag: str,
                    n_splits: int,
                    seed: int,
                    cfg_xgb: dict,
                    cfg_lgb: dict,
                    early_stopping_rounds: int,
                    debug_flag: bool):
    id_col = "id"
    target_col = "accident_risk"
    base_feat_cols = [c for c in raw_train_df.columns if c not in [id_col, target_col]]
    groups = build_group_keys(raw_train_df[base_feat_cols])

    gkf = GroupKFold(n_splits=n_splits)
    oof_xgb = np.zeros(len(y), dtype=np.float32)
    oof_lgb = np.zeros(len(y), dtype=np.float32)
    test_preds_xgb = np.zeros(X_test.shape[0], dtype=np.float32)
    test_preds_lgb = np.zeros(X_test.shape[0], dtype=np.float32)

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # XGBoost (CUDA) - constructor-based callbacks/early stopping (2.1+ API)
        xgb_guard = XGBFirstEpochGuard(enable_check=(fold_idx == 0), mode_tag=mode_tag)
        model_xgb = xgb.XGBRegressor(
            tree_method="hist",
            device="cuda",
            objective="reg:squarederror",
            random_state=seed,
            eval_metric="rmse",
            early_stopping_rounds=early_stopping_rounds,
            callbacks=[xgb_guard],
            **cfg_xgb
        )
        model_xgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)]
        )
        pred_va_xgb = np.clip(model_xgb.predict(X_va), 0.0, 1.0)
        oof_xgb[va_idx] = pred_va_xgb
        rmse_xgb = root_mean_squared_error(y_va, pred_va_xgb)
        logging.info(f"[{mode_tag}] Fold {fold_idx+1} XGBoost RMSE: {rmse_xgb:.6f}")
        test_preds_xgb += np.clip(model_xgb.predict(X_test), 0.0, 1.0) / n_splits

        # LightGBM (CPU)
        model_lgb = lgb.LGBMRegressor(
            objective="regression",
            random_state=seed,
            **cfg_lgb
        )
        lgb_es = lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
        lgb_guard = lgb_first_epoch_guard(fold_zero_check=(fold_idx == 0), mode_tag=mode_tag)
        model_lgb.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            callbacks=[lgb_es, lgb_guard]
        )
        pred_va_lgb = np.clip(model_lgb.predict(X_va, num_iteration=model_lgb.best_iteration_), 0.0, 1.0)
        oof_lgb[va_idx] = pred_va_lgb
        rmse_lgb = root_mean_squared_error(y_va, pred_va_lgb)
        logging.info(f"[{mode_tag}] Fold {fold_idx+1} LightGBM RMSE: {rmse_lgb:.6f}")
        test_preds_lgb += np.clip(model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration_), 0.0, 1.0) / n_splits

        del model_xgb, model_lgb, X_tr, X_va, y_tr, y_va
        gc.collect()

    rmse_oof_xgb = root_mean_squared_error(y, oof_xgb)
    rmse_oof_lgb = root_mean_squared_error(y, oof_lgb)
    logging.info(f"[{mode_tag}] OOF RMSE XGBoost: {rmse_oof_xgb:.6f}")
    logging.info(f"[{mode_tag}] OOF RMSE LightGBM: {rmse_oof_lgb:.6f}")

    # Return per-model OOF and predictions for stacking
    result = {
        "oof": {
            "xgb": oof_xgb,
            "lgb": oof_lgb,
            "rmse": {
                "xgb": rmse_oof_xgb,
                "lgb": rmse_oof_lgb,
            },
        },
        "test_pred": {
            "xgb": test_preds_xgb,
            "lgb": test_preds_lgb,
        }
    }
    return result


def stack_with_ridge(oof_dict: dict, test_dict: dict, y: np.ndarray, mode_tag: str, alpha: float = 1.0):
    cols = sorted(oof_dict.keys())
    X_meta = np.column_stack([oof_dict[k] for k in cols])
    T_meta = np.column_stack([test_dict[k] for k in cols])

    meta = Ridge(alpha=alpha, fit_intercept=True)
    meta.fit(X_meta, y)
    oof_stack = meta.predict(X_meta).astype(np.float32)
    test_stack = meta.predict(T_meta).astype(np.float32)

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(oof_stack, y)
    oof_cal = np.clip(iso.transform(oof_stack), 0.0, 1.0)
    test_cal = np.clip(iso.transform(test_stack), 0.0, 1.0)

    rmse_oof_stack = root_mean_squared_error(y, np.clip(oof_stack, 0.0, 1.0))
    rmse_oof_stack_cal = root_mean_squared_error(y, oof_cal)
    logging.info(f"[{mode_tag}] OOF RMSE Meta-Stack (Ridge) before calibration: {rmse_oof_stack:.6f}")
    logging.info(f"[{mode_tag}] Final OOF RMSE after isotonic calibration (stacked): {rmse_oof_stack_cal:.6f}")

    return {
        "oof_stack": np.clip(oof_stack, 0.0, 1.0),
        "test_stack": np.clip(test_stack, 0.0, 1.0),
        "oof_stack_cal": oof_cal,
        "test_stack_cal": test_cal,
        "meta_coef": dict(zip(cols, meta.coef_)),
        "rmse_oof_stack": rmse_oof_stack,
        "rmse_oof_stack_cal": rmse_oof_stack_cal
    }


def run_pipeline(debug_flag: bool, out_dir: str):
    mode_tag = "DEBUG" if debug_flag else "FULL"
    logging.info(f"[{mode_tag}] Starting mode")
    seed = 42
    set_seed(seed)
    train_df, test_df = load_comp_data()

    if debug_flag:
        n_debug = min(256, len(train_df))
        train_df = train_df.sample(n=n_debug, random_state=seed).reset_index(drop=True)

    # Feature engineering for XGB/LGB
    X, y, X_test, test_ids, feat_cols = feature_engineer(train_df.copy(), test_df.copy())

    n_splits = 3 if debug_flag else 5
    early_stopping_rounds = 10 if debug_flag else 100

    cfg_xgb = dict(
        n_estimators=(8 if debug_flag else 1400),
        learning_rate=0.05,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2.0,
        reg_lambda=1.5,
        reg_alpha=0.0,
        gamma=0.0,
    )
    cfg_lgb = dict(
        n_estimators=(16 if debug_flag else 1600),
        learning_rate=0.03,
        num_leaves=255,
        max_depth=-1,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        min_child_samples=30,
        reg_lambda=1.0,
        reg_alpha=0.0,
        n_jobs=-1
    )

    # Base models: XGB + LGB
    cv_result = run_cv_training(
        X=X,
        y=y,
        X_test=X_test,
        raw_train_df=train_df,
        mode_tag=mode_tag,
        n_splits=n_splits,
        seed=seed,
        cfg_xgb=cfg_xgb,
        cfg_lgb=cfg_lgb,
        early_stopping_rounds=early_stopping_rounds,
        debug_flag=debug_flag
    )

    # CatBoost (GPU) on raw categoricals + engineered numerics
    cat_oof, cat_test, rmse_cat = fit_catboost_oof(train_df, test_df, n_splits=n_splits, seed=seed, mode_tag=mode_tag, debug_flag=debug_flag)

    # Stack three models with Ridge + isotonic calibration
    oof_dict = {"cat": cat_oof, "lgb": cv_result["oof"]["lgb"], "xgb": cv_result["oof"]["xgb"]}
    test_dict = {"cat": cat_test, "lgb": cv_result["test_pred"]["lgb"], "xgb": cv_result["test_pred"]["xgb"]}
    stack_res = stack_with_ridge(oof_dict, test_dict, y=y, mode_tag=mode_tag, alpha=1.0)

    # Final validation results for this mode
    logging.info(f"[{mode_tag}] Final OOF RMSE (XGB): {cv_result['oof']['rmse']['xgb']:.6f}")
    logging.info(f"[{mode_tag}] Final OOF RMSE (LGB): {cv_result['oof']['rmse']['lgb']:.6f}")
    logging.info(f"[{mode_tag}] Final OOF RMSE (CatBoost): {rmse_cat:.6f}")
    logging.info(f"[{mode_tag}] Final OOF RMSE (Stacked calibrated): {stack_res['rmse_oof_stack_cal']:.6f}")

    # Submission for FULL only
    if not debug_flag:
        submission = pd.DataFrame({
            "id": test_ids,
            "accident_risk": stack_res["test_stack_cal"]
        })
        sub_path = os.path.join(out_dir, "submission_14.csv")
        submission.to_csv(sub_path, index=False)

    return {
        "cv_result": cv_result,
        "cat": {"oof": cat_oof, "test": cat_test, "rmse": rmse_cat},
        "stack": stack_res
    }


def main():
    out_dir, _ = ensure_dirs_and_logging()
    # First run: DEBUG=True
    run_pipeline(debug_flag=True, out_dir=out_dir)
    # Second run: FULL
    run_pipeline(debug_flag=False, out_dir=out_dir)


if __name__ == "__main__":
    main()

