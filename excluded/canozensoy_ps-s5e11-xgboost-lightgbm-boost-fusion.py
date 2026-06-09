# ============================================================
# Libraries
# ============================================================

import os, gc, warnings, math, random
from pathlib import Path

import numpy as np
import pandas as pd

import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# ============================================================
# Config
# ============================================================

DATA_DIR = Path("/kaggle/input/playground-series-s5e11")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH  = DATA_DIR / "test.csv"

ID_COL = "id"
TARGET = "loan_paid_back"

N_FOLDS = 7
SEED = 42

# Boosting budget & early stopping used by xgb.cv to find a good n_estimators
NUM_BOOST_ROUND = 20000
EARLY_STOP = 50

# Small internal XGB ensemble: same model with slight param jitters and different seeds
ENSEMBLE_SEEDS = [42, 7, 19, 77, 123]
JITTERS = [
    dict(max_leaves=4,  min_child_weight=89, reg_alpha=1.4, reg_lambda=5.9),
    dict(max_leaves=4,  min_child_weight=82, reg_alpha=1.1, reg_lambda=6.3),
    dict(max_leaves=5,  min_child_weight=95, reg_alpha=1.6, reg_lambda=5.6),
    dict(max_leaves=5,  min_child_weight=88, reg_alpha=1.3, reg_lambda=6.1),
    dict(max_leaves=4,  min_child_weight=92, reg_alpha=1.2, reg_lambda=6.0),
]

# Base XGBoost params (hist + categorical)
# If you don't have GPU, set device='cpu' or remove the parameter.
BASE_PARAMS = {
    'tree_method': 'hist',
    'device': 'cuda',            # change to 'cpu' if needed
    'predictor': 'auto',
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'colsample_bylevel': 1.0,
    'colsample_bynode': 1.0,
    'gamma': 0.0,
    'scale_pos_weight': 1.0,
}

# Reproducible randomness for Python, NumPy, and hashing
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# Helpers
# ============================================================

def read_data():
    """Load train/test CSVs from Kaggle input path."""
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    return train, test


def target_encoding(train, test, cols, target_col, n_splits=10, seed=42):
    """
    Out-of-fold target mean encoding for leakage-safe training.
    - Uses StratifiedKFold for stable class balance per fold.
    - Applies global mapping to test.
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    te_train = pd.DataFrame(index=train.index)
    te_test  = pd.DataFrame(index=test.index)

    y = train[target_col].values
    for col in cols:
        oof_vals = np.zeros(len(train), dtype=float)
        for tr_idx, va_idx in kf.split(train, y):
            tr_fold = train.iloc[tr_idx]
            va_fold = train.iloc[va_idx]
            mean_map = tr_fold.groupby(col)[target_col].mean()
            oof_vals[va_idx] = va_fold[col].map(mean_map).astype(float)
        te_train[f"mean_{col}"] = oof_vals

        # Global mapping for test rows (no leakage)
        global_map = train.groupby(col)[target_col].mean()
        te_test[f"mean_{col}"] = test[col].map(global_map).astype(float)

    return te_train, te_test


def create_frequency_and_bins(train, test, cols, num_cols):
    """
    Lightweight encodings:
    - Frequency encoding for all columns.
    - Quantile bins (5/10/15) for numeric columns to add coarse order information.
    """
    tr_new = pd.DataFrame(index=train.index)
    te_new = pd.DataFrame(index=test.index)

    for col in cols:
        # Frequency (fallback to mean frequency for unseen test values)
        freq = train[col].value_counts()
        tr_new[f"{col}_freq"] = train[col].map(freq).astype(float)
        te_new[f"{col}_freq"] = test[col].map(freq).astype(float).fillna(freq.mean())

        # Quantile bins only for numeric columns; protect against constant columns
        if col in num_cols:
            for q in [5, 10, 15]:
                try:
                    tr_bins, bins = pd.qcut(
                        train[col], q=q, labels=False, retbins=True, duplicates="drop"
                    )
                    tr_new[f"{col}_bin{q}"] = tr_bins.astype(float)
                    te_new[f"{col}_bin{q}"] = pd.cut(
                        test[col], bins=bins, labels=False, include_lowest=True
                    ).astype(float)
                except Exception:
                    tr_new[f"{col}_bin{q}"] = 0.0
                    te_new[f"{col}_bin{q}"] = 0.0
    return tr_new, te_new


def enable_categoricals(df, cat_cols):
    """Cast listed columns to pandas 'category' so XGBoost / LightGBM can see them."""
    for c in cat_cols:
        if df[c].dtype.name != "category":
            df[c] = df[c].astype("category")
    return df


def do_cv_nround(train_df, features, target, base_params):
    """
    Estimate a good number of boosting rounds via xgb.cv with early stopping.
    Returns (best_round, best_auc) based on test-auc-mean peak.
    """
    dtrain = xgb.DMatrix(train_df[features], label=train_df[target], enable_categorical=True)
    cv = xgb.cv(
        params=base_params,
        dtrain=dtrain,
        nfold=N_FOLDS,
        num_boost_round=NUM_BOOST_ROUND,
        metrics='auc',
        verbose_eval=False,
        early_stopping_rounds=EARLY_STOP,
        seed=SEED,
        shuffle=True,
        stratified=True,
    )
    best_round = int(cv['test-auc-mean'].idxmax())
    best_auc   = float(cv['test-auc-mean'][best_round])
    print(f"[CV] Best round: {best_round} | Best CV AUC: {best_auc:.7f}")
    return best_round, best_auc


def oof_auc_for_n(X, y, n_estimators, params):
    """
    Compute OOF AUC for a given n_estimators and params using StratifiedKFold.
    This is a small micro-sweep to refine around the cv-chosen round.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X), dtype="float32")
    for tr_idx, va_idx in skf.split(X, y):
        X_tr_f, y_tr_f = X.iloc[tr_idx], y[tr_idx]
        X_va_f, y_va_f = X.iloc[va_idx], y[va_idx]
        m = XGBClassifier(**params, n_estimators=n_estimators, enable_categorical=True)
        m.fit(X_tr_f, y_tr_f)
        oof[va_idx] = m.predict_proba(X_va_f)[:, 1].astype("float32")
        del m
        gc.collect()
    return roc_auc_score(y, oof), oof


# ============================================================
# Main
# ============================================================

def main():
    train, test = read_data()
    print(f"train: {train.shape} | test: {test.shape}")

    # Minimal domain feature: extract numeric subgrade from 'grade_subgrade' (e.g., 'A7' -> 7)
    train['subgrade'] = train['grade_subgrade'].str[1:].astype(int)
    test['subgrade']  = test['grade_subgrade'].str[1:].astype(int)

    # Build feature list excluding ID and target
    base_cols = train.drop(columns=[TARGET, ID_COL]).columns.tolist()

    # Split into categorical vs numeric for later encodings
    cat_cols = [c for c in base_cols if train[c].dtype in ["object", "category"]]
    num_cols = [c for c in base_cols if c not in cat_cols]

    # Leakage-safe target encoding on all base columns (categorical & numeric)
    te_tr, te_te = target_encoding(train, test, base_cols, TARGET, n_splits=10, seed=SEED)

    # Frequency + quantile-bin encodings
    fq_tr, fq_te = create_frequency_and_bins(train, test, base_cols, num_cols)

    # Concatenate original features with encodings
    X_tr = pd.concat([train[base_cols], te_tr, fq_tr], axis=1)
    X_te = pd.concat([test[base_cols],  te_te, fq_te], axis=1)

    # Optional feature drops kept from prior experimentation
    drops = [
        "education_level", "loan_purpose", "grade_subgrade", "interest_rate", "marital_status",
        "employment_status_freq", "credit_score_bin5", "loan_amount_bin5", "debt_to_income_ratio_bin5"
    ]
    drops = [d for d in drops if d in X_tr.columns]
    X_tr = X_tr.drop(columns=drops, errors="ignore")
    X_te = X_te.drop(columns=drops, errors="ignore")

    # Ensure categorical dtypes
    cat_all = [c for c in X_tr.columns if X_tr[c].dtype in ["object", "category"]]
    X_tr = enable_categoricals(X_tr, cat_all)
    X_te = enable_categoricals(X_te, cat_all)

    # Align columns between train and test for safety
    common_cols = [c for c in X_tr.columns if c in X_te.columns]
    X_tr = X_tr[common_cols]
    X_te = X_te[common_cols]

    print(f"Final feature count: {X_tr.shape[1]}")

    y = train[TARGET].values

    # ========================================================
    # 1) XGBoost: find good n_estimators via cv + micro-sweep
    # ========================================================

    base_for_cv = BASE_PARAMS.copy()
    probe = dict(max_leaves=4, min_child_weight=89, reg_alpha=1.4, reg_lambda=5.9)
    base_for_cv.update(probe)
    best_round, best_auc = do_cv_nround(
        pd.concat([X_tr, train[[TARGET]]], axis=1),
        common_cols,
        TARGET,
        base_for_cv
    )

    strong = BASE_PARAMS.copy()
    strong.update(probe)
    strong["random_state"] = SEED

    # Slightly expanded candidate set around best_round
    candidates = [best_round - 10, best_round, best_round + 10, best_round + 20]
    candidates = [n for n in candidates if n > 50]  # safety
    print(f"[n-sweep] candidates: {candidates}")

    best_n, best_n_auc = None, -1.0
    best_n_oof = None

    for n in candidates:
        auc_n, oof_n = oof_auc_for_n(X_tr[common_cols], y, n_estimators=n, params=strong)
        print(f"  n_estimators={n} -> OOF AUC={auc_n:.6f}")
        if auc_n > best_n_auc:
            best_n_auc = auc_n
            best_n = n
            best_n_oof = oof_n

    n_estimators = int(best_n)
    print(f"[n-sweep] chosen n_estimators={n_estimators} | OOF AUC={best_n_auc:.6f}")

    # ========================================================
    # 2) XGBoost internal ensemble
    # ========================================================

    preds_xgb_test_list = []
    for seed, jitter in zip(ENSEMBLE_SEEDS, JITTERS):
        params = BASE_PARAMS.copy()
        params.update(jitter)
        params["random_state"] = seed

        model = XGBClassifier(
            **params,
            n_estimators=n_estimators,
            enable_categorical=True
        )
        model.fit(X_tr, y)
        pred = model.predict_proba(X_te)[:, 1].astype("float32")
        preds_xgb_test_list.append(pred)
        del model
        gc.collect()

    # Build OOF stack for same ensemble (for blend tuning)
    print("[OOF-XGB] Building internal OOF for blend beta …")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_stack_xgb = np.zeros((len(train), len(preds_xgb_test_list)), dtype="float32")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y), 1):
        X_tr_f, y_tr_f = X_tr.iloc[tr_idx], y[tr_idx]
        X_va_f, y_va_f = X_tr.iloc[va_idx], y[va_idx]
        for m_idx, jitter in enumerate(JITTERS):
            params = BASE_PARAMS.copy()
            params.update(jitter)
            params["random_state"] = ENSEMBLE_SEEDS[m_idx]

            m = XGBClassifier(
                **params,
                n_estimators=n_estimators,
                enable_categorical=True
            )
            m.fit(X_tr_f, y_tr_f)
            oof_stack_xgb[va_idx, m_idx] = m.predict_proba(X_va_f)[:, 1].astype("float32")
            del m
        auc_fold = roc_auc_score(y_va_f, oof_stack_xgb[va_idx].mean(axis=1))
        print(f"  Fold {fold}: mean-prob AUC = {auc_fold:.6f}")
        gc.collect()

    prob_oof_xgb = oof_stack_xgb.mean(axis=1)
    ranks_xgb = np.column_stack([
        pd.Series(oof_stack_xgb[:, i]).rank(method="average").values
        for i in range(oof_stack_xgb.shape[1])
    ])
    rank_oof_xgb = (ranks_xgb.mean(axis=1) - ranks_xgb.min()) / (ranks_xgb.max() - ranks_xgb.min() + 1e-12)

    print(f"[XGB] Pure mean-prob OOF AUC: {roc_auc_score(y, prob_oof_xgb):.6f}")

    # ========================================================
    # 3) LightGBM model on the same features
    # ========================================================

    # categorical features for LightGBM
    cat_lgb = [c for c in X_tr.columns if X_tr[c].dtype.name == "category"]

    # Slightly regularized LGB params (small tweak vs previous)
    params_lgb = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 80,
        "max_depth": -1,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 60,
        "lambda_l1": 0.0,
        "lambda_l2": 5.0,
        "verbose": -1,
        "seed": SEED,
    }

    oof_lgb = np.zeros(len(X_tr), dtype="float32")
    preds_lgb_test = np.zeros(len(X_te), dtype="float32")

    print("\n[LightGBM] StratifiedKFold training …")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y), 1):
        print(f"  Fold {fold}/{N_FOLDS}")
        X_tr_f, y_tr_f = X_tr.iloc[tr_idx], y[tr_idx]
        X_va_f, y_va_f = X_tr.iloc[va_idx], y[va_idx]

        lgb_train = lgb.Dataset(
            X_tr_f, label=y_tr_f,
            categorical_feature=cat_lgb or None,
            free_raw_data=False
        )
        lgb_valid = lgb.Dataset(
            X_va_f, label=y_va_f,
            categorical_feature=cat_lgb or None,
            free_raw_data=False
        )

        model_lgb = lgb.train(
            params_lgb,
            lgb_train,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=["train", "valid"],
            num_boost_round=5000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=200),
                lgb.log_evaluation(period=200)
            ]
        )

        oof_lgb[va_idx] = model_lgb.predict(
            X_va_f,
            num_iteration=model_lgb.best_iteration
        ).astype("float32")
        preds_lgb_test += model_lgb.predict(
            X_te,
            num_iteration=model_lgb.best_iteration
        ).astype("float32") / N_FOLDS

        del model_lgb, lgb_train, lgb_valid, X_tr_f, X_va_f, y_tr_f, y_va_f
        gc.collect()

    auc_lgb = roc_auc_score(y, oof_lgb)
    print(f"[LightGBM] OOF AUC: {auc_lgb:.6f}")

    ranks_lgb = pd.Series(oof_lgb).rank(method="average").values
    rank_oof_lgb = (ranks_lgb - ranks_lgb.min()) / (ranks_lgb.max() - ranks_lgb.min() + 1e-12)

    # ========================================================
    # 4) Joint blend search: XGB vs LGBM, prob vs rank
    #    (denser grid around high-performing region)
    # ========================================================

    print("\n[Blend-Search] XGB vs LGBM, prob vs rank")

    # XGB weight grid: fine steps between 0.60 and 0.80
    w_grid = np.linspace(0.60, 0.80, 9)  # 0.60, 0.625, ..., 0.80
    # Beta grid: finer control of prob vs rank mix
    beta_grid = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    best_combo_auc = -1.0
    best_w = None
    best_beta = None

    for w_xgb in w_grid:
        w_lgb = 1.0 - w_xgb
        # first combine model-wise probabilities and ranks
        prob_mix_model = w_xgb * prob_oof_xgb + w_lgb * oof_lgb
        rank_mix_model = w_xgb * rank_oof_xgb + w_lgb * rank_oof_lgb

        for beta in beta_grid:
            # final blend: between prob_mix and rank_mix
            final_oof = (1.0 - beta) * prob_mix_model + beta * rank_mix_model
            auc_blend = roc_auc_score(y, final_oof)
            print(f"  w_xgb={w_xgb:.3f}, beta={beta:.2f} -> OOF AUC={auc_blend:.6f}")
            if auc_blend > best_combo_auc:
                best_combo_auc = auc_blend
                best_w = float(w_xgb)
                best_beta = float(beta)

    print(f"\n[Blend-Search] Best w_xgb={best_w:.3f}, beta={best_beta:.2f}, OOF AUC={best_combo_auc:.6f}")

    # ========================================================
    # 5) Apply best blend to test predictions & save submission
    # ========================================================

    # XGB test: mean prob + rank
    prob_test_xgb = np.mean(preds_xgb_test_list, axis=0)
    ranks_te_xgb = np.column_stack([
        pd.Series(preds_xgb_test_list[i]).rank(method="average").values
        for i in range(len(preds_xgb_test_list))
    ])
    rank_test_xgb = (ranks_te_xgb.mean(axis=1) - ranks_te_xgb.min()) / (
        ranks_te_xgb.max() - ranks_te_xgb.min() + 1e-12
    )

    # LGB test: prob + rank
    prob_test_lgb = preds_lgb_test
    ranks_te_lgb = pd.Series(preds_lgb_test).rank(method="average").values
    rank_test_lgb = (ranks_te_lgb - ranks_te_lgb.min()) / (
        ranks_te_lgb.max() - ranks_te_lgb.min() + 1e-12
    )

    # model-wise mixture
    w_xgb = best_w
    w_lgb = 1.0 - w_xgb

    prob_mix_test_model = w_xgb * prob_test_xgb + w_lgb * prob_test_lgb
    rank_mix_test_model = w_xgb * rank_test_xgb + w_lgb * rank_test_lgb

    # final prob vs rank blend
    beta = best_beta
    final_pred = (1.0 - beta) * prob_mix_test_model + beta * rank_mix_test_model
    final_pred = np.clip(final_pred, 0.0, 1.0).astype("float32")

    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: final_pred})
    sub.to_csv("submission.csv", index=False)

    print("\n[DONE] Wrote submission.csv")
    print(f"XGB probe CV AUC: {best_auc:.6f} | XGB n-sweep best OOF AUC: {best_n_auc:.6f}")
    print(f"LGBM OOF AUC: {auc_lgb:.6f} | Final blended OOF AUC: {best_combo_auc:.6f}")


if __name__ == "__main__":
    main()


