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


# 1. Imports

import os, gc, warnings, math, random
from pathlib import Path

import numpy as np
import pandas as pd

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# 2. Config & experiment metadata


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


ENSEMBLE_SEEDS = [42, 7, 19, 77, 123]
JITTERS = [
    dict(max_leaves=4,  min_child_weight=89, reg_alpha=1.4, reg_lambda=5.9),
    dict(max_leaves=4,  min_child_weight=82, reg_alpha=1.1, reg_lambda=6.3),
    dict(max_leaves=5,  min_child_weight=95, reg_alpha=1.6, reg_lambda=5.6),
    dict(max_leaves=5,  min_child_weight=88, reg_alpha=1.3, reg_lambda=6.1),
    dict(max_leaves=4,  min_child_weight=92, reg_alpha=1.2, reg_lambda=6.0),
]

# Base XGBoost params (hist + categorical). Keep these stable while you test encodings.
BASE_PARAMS = {
    'tree_method': 'hist',
    'device': 'cuda',            # falls back if no GPU; see GPU fallback note below
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


# 3. Determinism helpers


os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


# 4. Helper functions

def read_data():
    """Load train/test CSVs from Kaggle input path.

    NOTE: If you run locally, replace DATA_DIR with your local path.
    """
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    return train, test


def target_encoding(train, test, cols, target_col, n_splits=10, seed=42):
    """
    Out-of-fold target mean encoding for leakage-safe training.

    Important notes (do not change code without understanding leakage):
    - Uses StratifiedKFold to maintain class balance per fold.
    - Applies mapping from train to test (global_map), which is leakage-safe.
    - This implementation does NOT include smoothing or prior regularization. For
      sparse categories or categories with very few counts, consider adding a
      smoothing term (e.g., Bayesian smoothing) to avoid overfitting.

    Practical tweaks to try (not applied here):
    - Add smoothing: mean = (count*mean_cat + k*global_mean) / (count + k)
    - Clip or floor encoded values for extreme categories.
    - Replace unseen test categories with global mean (already happens as NaN -> NaN
      so cast to float; you may want explicit fillna(train[target_col].mean())).
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
    - Frequency encoding for all columns (helps rare vs frequent separation).
    - Quantile bins (5/10/15) for numeric columns to add coarse order information.

    Practical tip: frequency encoding can leak if done with target information â€” here
    it is computed on the train only and mapped to test, which is safe. Still, in
    temporal problems prefer computing frequencies on earlier data only.
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
                    tr_bins, bins = pd.qcut(train[col], q=q, labels=False, retbins=True, duplicates="drop")
                    tr_new[f"{col}_bin{q}"] = tr_bins.astype(float)
                    te_new[f"{col}_bin{q}"] = pd.cut(test[col], bins=bins, labels=False, include_lowest=True).astype(float)
                except Exception:
                    # If qcut fails (constant column, too few unique values), use zeros.
                    tr_new[f"{col}_bin{q}"] = 0.0
                    te_new[f"{col}_bin{q}"] = 0.0
    return tr_new, te_new


def enable_categoricals(df, cat_cols):
    """Cast listed columns to pandas 'category' so XGBoost can use enable_categorical=True.

    NOTE: XGBoost's categorical support expects pandas.Categorical with consistent categories
    between train/test. If you see strange errors, ensure categories match or use label encoding.
    """
    for c in cat_cols:
        if df[c].dtype.name != "category":
            df[c] = df[c].astype("category")
    return df


def do_cv_nround(train_df, features, target, base_params):
    """
    Estimate a good number of boosting rounds via xgb.cv with early stopping.

    Returns (best_round, best_auc) based on test-auc-mean peak.

    Practical notes:
    - xgb.cv returns a DataFrame with 'test-auc-mean' per round. We pick the round
      with the maximum test-auc-mean. Using idxmax is robust but you could also
      pick the early_stopping best iteration (last row when early stop triggered).
    - If you run this on CPU, consider changing device/predictor params accordingly.
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

    This small micro-sweep refines n_estimators chosen by xgb.cv using a full OOF
    training pass. It's slower but gives a more realistic OOF estimate when
    fitting sklearn wrappers.
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
    return roc_auc_score(y, oof)


# 5. Main flow 


def main():
    train, test = read_data()
    print(f"train: {train.shape} | test: {test.shape}")

    # Minimal domain feature: extract numeric subgrade from 'grade_subgrade' (e.g., 'A7' -> 7)
    # If grade_subgrade sometimes contains NA or malformed values, consider protecting with .str[1:].fillna(0)
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
        "education_level","loan_purpose","grade_subgrade","interest_rate","marital_status",
        "employment_status_freq", "credit_score_bin5", "loan_amount_bin5", "debt_to_income_ratio_bin5"
    ]
    drops = [d for d in drops if d in X_tr.columns]
    X_tr = X_tr.drop(columns=drops, errors="ignore")
    X_te = X_te.drop(columns=drops, errors="ignore")

    # Ensure XGBoost categorical support by casting objects/categories
    cat_all = [c for c in X_tr.columns if X_tr[c].dtype in ["object","category"]]
    X_tr = enable_categoricals(X_tr, cat_all)
    X_te = enable_categoricals(X_te, cat_all)

    # Align columns between train and test for safety
    common_cols = [c for c in X_tr.columns if c in X_te.columns]
    X_tr = X_tr[common_cols]
    X_te = X_te[common_cols]

    print(f"Final feature count: {X_tr.shape[1]}")

    # Find a good n_estimators via xgb.cv using a representative jitter probe
    base_for_cv = BASE_PARAMS.copy()
    probe = dict(max_leaves=4, min_child_weight=89, reg_alpha=1.4, reg_lambda=5.9)
    base_for_cv.update(probe)
    best_round, best_auc = do_cv_nround(pd.concat([X_tr, train[[TARGET]]], axis=1), common_cols, TARGET, base_for_cv)

    # Micro-sweep around the cv peak to lock n_estimators (best, +10, +20)
    y = train[TARGET].values
    strong = BASE_PARAMS.copy()
    strong.update(probe)
    strong["random_state"] = SEED

    candidates = [best_round, best_round + 10, best_round + 20]
    best_n, best_n_auc = None, -1.0
    print(f"[n-sweep] candidates: {candidates}")
    for n in candidates:
        auc_n = oof_auc_for_n(X_tr[common_cols], y, n_estimators=n, params=strong)
        print(f"  n_estimators={n} -> OOF AUC={auc_n:.6f}")
        if auc_n > best_n_auc:
            best_n_auc = auc_n
            best_n = n

    n_estimators = int(best_n)
    print(f"[n-sweep] chosen n_estimators={n_estimators} | OOF AUC={best_n_auc:.6f}")

    # Train the internal ensemble (same n_estimators, jittered params, different seeds)
    preds = []
    for seed, jitter in zip(ENSEMBLE_SEEDS, JITTERS):
        params = BASE_PARAMS.copy()
        params.update(jitter)
        params["random_state"] = seed

        model = XGBClassifier(
            **params,
            n_estimators=n_estimators,
            enable_categorical=True
        )
        model.fit(X_tr, train[TARGET])
        pred = model.predict_proba(X_te)[:, 1].astype("float32")
        preds.append(pred)
        del model
        gc.collect()

    # Build OOF stack for the same ensemble to tune a simple convex blend (prob vs rank)
    print("[OOF] Building internal OOF for blend beta â€¦")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_stack = np.zeros((len(train), len(preds)), dtype="float32")

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
            oof_stack[va_idx, m_idx] = m.predict_proba(X_va_f)[:, 1].astype("float32")
            del m
        auc_fold = roc_auc_score(y_va_f, oof_stack[va_idx].mean(axis=1))
        print(f"  Fold {fold}: mean-prob AUC = {auc_fold:.6f}")
        gc.collect()

    # Tune beta for convex combination of mean probabilities and mean ranks (robustness)
    prob_oof = oof_stack.mean(axis=1)
    ranks = np.column_stack([pd.Series(oof_stack[:, i]).rank(method="average").values for i in range(oof_stack.shape[1])])
    rank_oof = (ranks.mean(axis=1) - ranks.min()) / (ranks.max() - ranks.min() + 1e-12)

    beta_grid = [0.20, 0.25, 0.30, 0.35, 0.40]
    best_beta, best_beta_auc = 0.25, -1.0
    for b in beta_grid:
        mix = (1-b)*prob_oof + b*rank_oof
        auc = roc_auc_score(y, mix)
        if auc > best_beta_auc:
            best_beta_auc = auc
            best_beta = b
    print(f"[Blend] Best beta={best_beta} | OOF AUC={best_beta_auc:.6f}")

    # Apply tuned beta to test predictions and write submission
    prob_test = np.mean(preds, axis=0)
    ranks_te = np.column_stack([pd.Series(preds[i]).rank(method="average").values for i in range(len(preds))])
    rank_test = (ranks_te.mean(axis=1) - ranks_te.min()) / (ranks_te.max() - ranks_te.min() + 1e-12)
    final_pred = (1-best_beta)*prob_test + best_beta*rank_test
    final_pred = np.clip(final_pred, 0.0, 1.0).astype("float32")

    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: final_pred})
    sub.to_csv("submission.csv", index=False)
    print("\n[DONE] Wrote submission.csv")
    print(f"CV ref AUC (probe): {best_auc:.6f} | OOF AUC (n-sweep best): {best_n_auc:.6f} | Blend OOF AUC: {best_beta_auc:.6f}")




# 9. Final: run if invoked as script

if __name__ == "__main__":
    main()


