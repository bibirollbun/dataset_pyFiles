# -----------------------------
# Imports, Globals, and Configuration
# -----------------------------

import os, math, warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.isotonic import IsotonicRegression

import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)


# -----------------------------
# Config
# -----------------------------
SEED = 42
N_FOLDS = 5
CV_SEEDS = [42]  
DATA_DIR = "/kaggle/input/playground-series-s5e9"
ID_COL = "id"
TARGET_COL = "BeatsPerMinute"

BASE_NUM_COLS = [
    'RhythmScore','AudioLoudness','VocalContent','AcousticQuality',
    'InstrumentalScore','LivePerformanceLikelihood','MoodScore',
    'TrackDurationMs','Energy'
]


# -----------------------------
# Utils
# -----------------------------

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

@dataclass
class CVResult:
    oof: np.ndarray
    test_pred: np.ndarray
    fold_metrics: List[float]
    best_iters: List[int]


# -----------------------------
# Feature Engineering
# -----------------------------

def compute_bin_edges(train_col: pd.Series, q: List[float]) -> np.ndarray:
    # unique quantiles, padded with -inf/inf
    qs = np.unique(np.clip(q, 0.0, 1.0))
    edges = np.quantile(train_col.values, qs)
    edges = np.unique(edges)
    # ensure strictly increasing edges; add tiny jitter if necessary
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-9
    return np.concatenate(([-np.inf], edges, [np.inf]))


def apply_bins(col: pd.Series, edges: np.ndarray) -> np.ndarray:
    # returns bin indices 0..len(edges)-2
    return np.digitize(col.values, edges) - 1


def build_features(train: pd.DataFrame, test: pd.DataFrame, base_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Create engineered features on train & test using train-only stats.
    - Pairwise products (including squares) and safe ratios (both directions)
    - Quantile-based bins (quartile & decile) computed from train only
    """
    tr = train.copy()
    te = test.copy()

    # Cast to float32 early for memory
    for c in base_cols:
        tr[c] = tr[c].astype(np.float32)
        te[c] = te[c].astype(np.float32)

    # Products & squares
    for i, c1 in enumerate(base_cols):
        v1_tr = tr[c1]
        v1_te = te[c1]
        tr[f"{c1}_sq"] = v1_tr * v1_tr
        te[f"{c1}_sq"] = v1_te * v1_te
        for j in range(i+1, len(base_cols)):
            c2 = base_cols[j]
            tr[f"{c1}_x_{c2}"] = v1_tr * tr[c2]
            te[f"{c1}_x_{c2}"] = v1_te * te[c2]

    # Safe ratios (both directions)
    eps = 1e-6
    for i, c1 in enumerate(base_cols):
        for j, c2 in enumerate(base_cols):
            if i == j:
                continue
            tr[f"{c1}_div_{c2}"] = tr[c1] / (tr[c2].abs() + eps)
            te[f"{c1}_div_{c2}"] = te[c1] / (te[c2].abs() + eps)

    # Quantile bins from train only
    quart_q = [0.25, 0.5, 0.75]
    dec_q   = [i/10 for i in range(1, 10)]
    for c in base_cols:
        edges4 = compute_bin_edges(tr[c], quart_q)
        edges10 = compute_bin_edges(tr[c], dec_q)
        tr[f"{c}_quartile"] = apply_bins(tr[c], edges4).astype(np.int8)
        te[f"{c}_quartile"] = apply_bins(te[c], edges4).astype(np.int8)
        tr[f"{c}_decile"] = apply_bins(tr[c], edges10).astype(np.int8)
        te[f"{c}_decile"] = apply_bins(te[c], edges10).astype(np.int8)

    # Final feature list: all except ID/TARGET
    feat_cols = [c for c in tr.columns if c not in [ID_COL, TARGET_COL]]
    return tr, te, feat_cols


# -----------------------------
# Model Trainers
# -----------------------------

def fit_lgbm(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, num_cols: List[str]) -> CVResult:
    params = dict(
        n_estimators=6000,
        learning_rate=0.02,
        num_leaves=127,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_samples=25,
        random_state=SEED,
        n_jobs=-1,
    )

    # Preprocessor: median imputation for numeric
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
    pre = ColumnTransformer([("num", num_pipe, num_cols)], remainder="drop")

    oof_sum = np.zeros(len(X), dtype=float)
    oof_cnt = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    fold_metrics = []
    best_iters = []

    for cv_seed in CV_SEEDS:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=cv_seed)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
            print(f"\n[LGBM] Seed {cv_seed} | Fold {fold}")
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            pre.fit(X_tr, y_tr)
            Xtr = pre.transform(X_tr)
            Xva = pre.transform(X_va)
            Xte = pre.transform(X_test)

            model = lgb.LGBMRegressor(**params)
            model.fit(
                Xtr, y_tr,
                eval_set=[(Xva, y_va)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=True)],
            )
            bi = int(model.best_iteration_)
            best_iters.append(bi)

            va_pred = model.predict(Xva, num_iteration=bi)
            oof_sum[va_idx] += va_pred
            oof_cnt[va_idx] += 1
            fold_rmse = rmse(y_va, va_pred)
            fold_metrics.append(fold_rmse)
            print(f"[LGBM] RMSE: {fold_rmse:.5f}")

            test_pred += model.predict(Xte, num_iteration=bi) / (N_FOLDS * len(CV_SEEDS))

    oof = oof_sum / np.maximum(oof_cnt, 1)
    print(f"[LGBM] CV RMSE: {rmse(y, oof):.5f}")
    return CVResult(oof, test_pred, fold_metrics, best_iters)


def fit_xgb(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, num_cols: List[str]) -> CVResult:
    params = dict(
        n_estimators=7000,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
    pre = ColumnTransformer([("num", num_pipe, num_cols)], remainder="drop")

    oof_sum = np.zeros(len(X), dtype=float)
    oof_cnt = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    fold_metrics = []
    best_iters = []

    for cv_seed in CV_SEEDS:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=cv_seed)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
            print(f"\n[XGB] Seed {cv_seed} | Fold {fold}")
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            pre.fit(X_tr, y_tr)
            Xtr = pre.transform(X_tr)
            Xva = pre.transform(X_va)
            Xte = pre.transform(X_test)

            model = XGBRegressor(**params)
            model.fit(
                Xtr, y_tr,
                eval_set=[(Xva, y_va)],
                eval_metric="rmse",
                verbose=False,
                early_stopping_rounds=300,
            )
            best_iter = int(model.best_iteration) if hasattr(model, "best_iteration") else params["n_estimators"]
            best_iters.append(best_iter)

            va_pred = model.predict(Xva, iteration_range=(0, best_iter))
            oof_sum[va_idx] += va_pred
            oof_cnt[va_idx] += 1
            fold_rmse = rmse(y_va, va_pred)
            fold_metrics.append(fold_rmse)
            print(f"[XGB] RMSE: {fold_rmse:.5f}")

            test_pred += model.predict(Xte, iteration_range=(0, best_iter)) / (N_FOLDS * len(CV_SEEDS))

    oof = oof_sum / np.maximum(oof_cnt, 1)
    print(f"[XGB] CV RMSE: {rmse(y, oof):.5f}")
    return CVResult(oof, test_pred, fold_metrics, best_iters)


def fit_cat(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, num_cols: List[str]) -> CVResult:
    params = dict(
        depth=8,
        learning_rate=0.03,
        n_estimators=8000,
        loss_function="RMSE",
        random_seed=SEED,
        l2_leaf_reg=3.0,
        subsample=0.9,
        rsm=0.9,
        verbose=200,
        allow_const_label=True,
    )

    # CatBoost: explicit median imputation
    imp = SimpleImputer(strategy="median")
    X_num = pd.DataFrame(imp.fit_transform(X[num_cols]), columns=num_cols, index=X.index)
    Xte_num = pd.DataFrame(imp.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)

    oof_sum = np.zeros(len(X), dtype=float)
    oof_cnt = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    fold_metrics = []
    best_iters = []

    for cv_seed in CV_SEEDS:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=cv_seed)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_num, y), 1):
            print(f"\n[CAT] Seed {cv_seed} | Fold {fold}")
            X_tr, X_va = X_num.iloc[tr_idx], X_num.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            train_pool = Pool(X_tr, y_tr)
            valid_pool = Pool(X_va, y_va)

            model = CatBoostRegressor(**params)
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

            best_iters.append(int(model.tree_count_))
            va_pred = model.predict(valid_pool)
            oof_sum[va_idx] += va_pred
            oof_cnt[va_idx] += 1
            fold_rmse = rmse(y_va, va_pred)
            fold_metrics.append(fold_rmse)
            print(f"[CAT] RMSE: {fold_rmse:.5f}")

            test_pred += model.predict(Xte_num) / (N_FOLDS * len(CV_SEEDS))

    oof = oof_sum / np.maximum(oof_cnt, 1)
    print(f"[CAT] CV RMSE: {rmse(y, oof):.5f}")
    return CVResult(oof, test_pred, fold_metrics, best_iters)


# -----------------------------
# Blending & Calibration
# -----------------------------

def two_stage_weight_search(y_true: np.ndarray, oof_mat: np.ndarray) -> Tuple[float, Tuple[float,float,float]]:
    # Stage 1: coarse grid
    best = (np.inf, (1.0, 0.0, 0.0))
    for a in np.linspace(0, 1, 21):
        for b in np.linspace(0, 1-a, int((1-a)/0.05)+1):
            c = 1.0 - a - b
            w = np.array([a, b, c])
            score = rmse(y_true, (oof_mat * w).sum(1))
            if score < best[0]:
                best = (score, (a, b, c))
    # Stage 2: fine grid around best
    a0, b0, c0 = best[1]
    fine = np.arange(-0.05, 0.0501, 0.005)
    best_f = best
    for da in fine:
        for db in fine:
            a = a0 + da; b = b0 + db; c = 1.0 - a - b
            if a < 0 or b < 0 or c < 0: 
                continue
            w = np.array([a, b, c])
            score = rmse(y_true, (oof_mat * w).sum(1))
            if score < best_f[0]:
                best_f = (score, (a, b, c))
    return best_f


# -----------------------------
# Main
# -----------------------------

def main():
    # Load
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    _ = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    assert TARGET_COL in train.columns, f"Target column '{TARGET_COL}' not found in train.csv"
    assert ID_COL in train.columns and ID_COL in test.columns, "ID column not found in train/test"

    # Build features (train-only stats used for bins)
    train_fe, test_fe, feature_cols = build_features(train, test, BASE_NUM_COLS)

    X = train_fe[feature_cols].copy()
    y = train_fe[TARGET_COL].astype(float).copy()
    X_test = test_fe[feature_cols].copy()

    # Numeric/categorical split (should be all numeric now)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Train shape: {X.shape} | Test shape: {X_test.shape} | Numeric features: {len(num_cols)}")

    # Optional target transform
    use_target_transform = True
    if use_target_transform:
        pt = PowerTransformer(method="yeo-johnson")
        y_t = pt.fit_transform(y.values.reshape(-1, 1)).ravel()
        y_fit = pd.Series(y_t, index=y.index)
        inv = lambda arr: pt.inverse_transform(arr.reshape(-1,1)).ravel()
    else:
        y_fit = y.copy()
        inv = lambda arr: arr

    # Train models (with CV seeds bagging if CV_SEEDS has >1 seed)
    lgb_cv = fit_lgbm(X, y_fit, X_test, num_cols)
    xgb_cv = fit_xgb(X, y_fit, X_test, num_cols)
    cat_cv = fit_cat(X, y_fit, X_test, num_cols)

    # Inverse-transform predictions back to BPM
    lgb_oof, lgb_test = inv(lgb_cv.oof), inv(lgb_cv.test_pred)
    xgb_oof, xgb_test = inv(xgb_cv.oof), inv(xgb_cv.test_pred)
    cat_oof, cat_test = inv(cat_cv.oof), inv(cat_cv.test_pred)

    print("\nModel CV RMSEs (post-inverse-transform):")
    print(f"  LGBM: {rmse(y, lgb_oof):.5f}")
    print(f"  XGB : {rmse(y, xgb_oof):.5f}")
    print(f"  CAT : {rmse(y, cat_oof):.5f}")

    # Adaptive clipping bounds from train target
    lo = float(np.quantile(y, 0.005))
    hi = float(np.quantile(y, 0.995))
    clip_lo = max(lo, 40.0)
    clip_hi = min(hi, 240.0)

    # Try three calibration modes and pick the best by OOF
    modes = []

    # Mode A: No isotonic
    A_oof = np.vstack([
        np.clip(lgb_oof, clip_lo, clip_hi),
        np.clip(xgb_oof, clip_lo, clip_hi),
        np.clip(cat_oof, clip_lo, clip_hi)
    ]).T
    A_test = np.vstack([
        np.clip(lgb_test, clip_lo, clip_hi),
        np.clip(xgb_test, clip_lo, clip_hi),
        np.clip(cat_test, clip_lo, clip_hi)
    ]).T
    modes.append(("none", A_oof, A_test))

    # Mode B: Per-model isotonic
    iso_lgb = IsotonicRegression(out_of_bounds="clip").fit(lgb_oof, y)
    iso_xgb = IsotonicRegression(out_of_bounds="clip").fit(xgb_oof, y)
    iso_cat = IsotonicRegression(out_of_bounds="clip").fit(cat_oof, y)
    B_oof = np.vstack([
        np.clip(iso_lgb.predict(lgb_oof), clip_lo, clip_hi),
        np.clip(iso_xgb.predict(xgb_oof), clip_lo, clip_hi),
        np.clip(iso_cat.predict(cat_oof), clip_lo, clip_hi)
    ]).T
    B_test = np.vstack([
        np.clip(iso_lgb.predict(lgb_test), clip_lo, clip_hi),
        np.clip(iso_xgb.predict(xgb_test), clip_lo, clip_hi),
        np.clip(iso_cat.predict(cat_test), clip_lo, clip_hi)
    ]).T
    modes.append(("per_model_iso", B_oof, B_test))

    # Evaluate modes with blending; keep the best
    best_global = (np.inf, None, None, None)  # (rmse, weights, test_mat, mode)
    for mode_name, oof_mat, test_mat in modes:
        score1, w1 = two_stage_weight_search(y.values, oof_mat)
        print(f"\n[{mode_name}] Best blend OOF RMSE: {score1:.5f} with weights LGBM={w1[0]:.3f}, XGB={w1[1]:.3f}, CAT={w1[2]:.3f}")

        # Final isotonic on blended predictions (often helps a bit)
        oof_blend = (oof_mat * np.array(w1)).sum(1)
        iso_final = IsotonicRegression(out_of_bounds="clip").fit(oof_blend, y.values)
        oof_blend_iso = iso_final.predict(oof_blend)
        score2 = rmse(y.values, oof_blend_iso)
        print(f"[{mode_name}] After final isotonic: OOF RMSE: {score2:.5f}")

        if score2 < best_global[0]:
            best_global = (score2, w1, (test_mat, iso_final), mode_name)

    best_rmse, best_w, (best_test_mat, best_iso), best_mode = best_global
    print(f"\nChosen mode: {best_mode} | OOF RMSE: {best_rmse:.5f} | Weights: LGBM={best_w[0]:.3f}, XGB={best_w[1]:.3f}, CAT={best_w[2]:.3f}")

    blended_test = (best_test_mat * np.array(best_w)).sum(1)
    blended_test = best_iso.predict(blended_test)
    blended_test = np.clip(blended_test, clip_lo, clip_hi)

    # Save submission
    submission = pd.DataFrame({ID_COL: test_fe[ID_COL], TARGET_COL: blended_test})
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")

if __name__ == "__main__":
    main()


