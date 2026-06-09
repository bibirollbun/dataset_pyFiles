# ============================================================
#  S5E11 – XGBoost + CatBoost Ensemble
# ============================================================

import os
import gc
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_FOLDS_XGB = 10
N_FOLDS_TE  = 5        # for target encoding
BLEND_W_XGB = 0.7      # weight for XGBoost vs CatBoost

# ============================================================
#  1. Load data
# ============================================================

train_path   = "/kaggle/input/playground-series-s5e11/train.csv"
test_path    = "/kaggle/input/playground-series-s5e11/test.csv"
sample_path  = "/kaggle/input/playground-series-s5e11/sample_submission.csv"

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

print("Train shape:", train.shape)
print("Test  shape:", test.shape)

TARGET_COL = "loan_paid_back"
ID_COL     = "id" if "id" in train.columns else None

y = train[TARGET_COL].values
X = train.drop(columns=[TARGET_COL])
X_test = test.copy()

if ID_COL is not None:
    X = X.drop(columns=[ID_COL])
    X_test_ids = X_test[ID_COL].copy()
    X_test = X_test.drop(columns=[ID_COL])
else:
    X_test_ids = np.arange(len(X_test))

full = pd.concat([X, X_test], axis=0, ignore_index=True)

# ============================================================
#  2. Identify numeric & categorical columns
# ============================================================

cat_cols = full.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = full.select_dtypes(include=[np.number]).columns.tolist()

print("Categorical columns:", cat_cols)
print("Numeric columns:", num_cols)

# ============================================================
#  3. Basic numeric cleaning: clip outliers + log-skewed
# ============================================================

# clip extremes (1% – 99%) to reduce impact of outliers
for col in num_cols:
    q1 = full[col].quantile(0.01)
    q99 = full[col].quantile(0.99)
    full[col] = full[col].clip(q1, q99)

# log1p-transform skewed numeric features
skews = full[num_cols].skew().sort_values(ascending=False)
skewed_cols = skews[abs(skews) > 1.0].index.tolist()

for col in skewed_cols:
    full[f"{col}_log1p"] = np.log1p(full[col])
    num_cols.append(f"{col}_log1p")

print("Skewed cols log-transformed:", skewed_cols)

# ============================================================
#  4. Domain-ish advanced features (if columns exist)
# ============================================================

# NOTE: some of these may not exist, so we guard with if-statements
def safe_add_ratio(col_a, col_b, name):
    if col_a in full.columns and col_b in full.columns:
        full[name] = full[col_a] / (full[col_b].replace(0, np.nan))
        full[name] = full[name].fillna(full[name].median())
        num_cols.append(name)

def safe_add_product(col_a, col_b, name):
    if col_a in full.columns and col_b in full.columns:
        full[name] = full[col_a] * full[col_b]
        num_cols.append(name)

# common S5E11 columns 
cols = full.columns

if {"loan_amount", "annual_income"}.issubset(cols):
    safe_add_ratio("loan_amount", "annual_income", "loan_to_income")
    safe_add_ratio("annual_income", "loan_amount", "income_to_loan")

if {"annual_income", "debt_to_income_ratio"}.issubset(cols):
    safe_add_ratio("annual_income", "debt_to_income_ratio", "income_to_dti")

if {"interest_rate", "loan_amount"}.issubset(cols):
    safe_add_product("interest_rate", "loan_amount", "interest_x_loan")

if {"credit_score", "loan_amount"}.issubset(cols):
    safe_add_ratio("credit_score", "loan_amount", "credit_to_loan")

# more generic interactions: pairwise ratios for top numeric features
top_num = [c for c in num_cols if c in [
    "annual_income", "loan_amount", "credit_score", "interest_rate", "debt_to_income_ratio"
]]

for i, c1 in enumerate(top_num):
    for c2 in top_num[i+1:]:
        name = f"{c1}_over_{c2}"
        if name not in full.columns:
            full[name] = full[c1] / (full[c2].replace(0, np.nan))
            full[name] = full[name].replace([np.inf, -np.inf], np.nan)
            full[name] = full[name].fillna(full[name].median())
            num_cols.append(name)

print("Total numeric columns after FE:", len(num_cols))

# ============================================================
#  5. Frequency encoding for categoricals
# ============================================================

for col in cat_cols:
    freq = full[col].value_counts(normalize=True)
    full[f"{col}_freq"] = full[col].map(freq).astype(float)
    num_cols.append(f"{col}_freq")

# ============================================================
#  6. K-fold target encoding for categoricals (train-only)
# ============================================================

def add_target_encoding(full_df, y, cat_columns, n_splits=5, random_state=42):
    """Add K-fold target encoding for each categorical column."""
    from sklearn.model_selection import StratifiedKFold
    
    n_train = len(y)
    full_df = full_df.copy()
    te_cols = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for col in cat_columns:
        te_col = f"{col}_te"
        te_cols.append(te_col)
        full_df[te_col] = 0.0

        # compute out-of-fold encodings
        oof = np.zeros(n_train)
        for tr_idx, val_idx in skf.split(full_df.iloc[:n_train], y):
            tr_data = full_df.iloc[tr_idx]
            tr_y    = y[tr_idx]
            means = tr_data.groupby(col)[TARGET_COL].mean() if TARGET_COL in tr_data.columns else tr_y.groupby(tr_data[col]).mean()
            # workaround since y is separate
            # we build a dataframe for convenience
            tmp = pd.DataFrame({col: tr_data[col].values, "target": tr_y})
            means = tmp.groupby(col)["target"].mean()
            oof[val_idx] = full_df.iloc[val_idx][col].map(means).fillna(y.mean())
        full_df.loc[:n_train-1, te_col] = oof

        # compute encoding for test using full train
        tmp_all = pd.DataFrame({col: full_df.iloc[:n_train][col].values, "target": y})
        means_all = tmp_all.groupby(col)["target"].mean()
        full_df.loc[n_train:, te_col] = full_df.iloc[n_train:][col].map(means_all).fillna(y.mean())

    return full_df, te_cols

# Since our y is separate, we temporarily attach TARGET_COL for convenience
tmp_full = full.copy()
tmp_full[TARGET_COL] = np.concatenate([y, np.full(len(X_test), np.nan)])

tmp_full, te_cols = add_target_encoding(tmp_full, y, cat_cols, n_splits=N_FOLDS_TE, random_state=RANDOM_STATE)
print("Target-encoded columns:", te_cols)

# drop helper TARGET_COL from full feature matrix
tmp_full = tmp_full.drop(columns=[TARGET_COL])

# update full + numeric columns
full = tmp_full
num_cols.extend(te_cols)

# ============================================================
#  7. Final prep: label-encode categoricals (for XGBoost)
# ============================================================

for col in cat_cols:
    full[col] = full[col].astype(str)
    # simple label encoding (consistent across train & test)
    full[col], _ = pd.factorize(full[col])
    num_cols.append(col)

# final feature matrix split back to train / test
X_all = full.iloc[:len(X)].reset_index(drop=True)
X_test_all = full.iloc[len(X):].reset_index(drop=True)

print("Final train features:", X_all.shape)
print("Final test  features:", X_test_all.shape)

# ============================================================
#  8. XGBoost + CatBoost training with Stratified KFold
# ============================================================

xgb_oof = np.zeros(len(X_all))
cb_oof  = np.zeros(len(X_all))

xgb_preds_test = np.zeros(len(X_test_all))
cb_preds_test  = np.zeros(len(X_test_all))

skf = StratifiedKFold(n_splits=N_FOLDS_XGB, shuffle=True, random_state=RANDOM_STATE)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y), 1):
    print(f"\n========== Fold {fold}/{N_FOLDS_XGB} ==========")
    X_tr, X_val = X_all.iloc[tr_idx], X_all.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    # --------------------- XGBoost --------------------------
    xgb_model = XGBClassifier(
        n_estimators=4000,
        learning_rate=0.025,
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=5,
        max_leaves=48,
        subsample=0.9,
        colsample_bytree=0.7,
        min_child_weight=4.0,
        gamma=0.5,
        reg_alpha=0.1,
        reg_lambda=2.0,
        tree_method="gpu_hist",
        gpu_id=0,
        max_bin=256,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1
    )

    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=200
    )

    xgb_val = xgb_model.predict_proba(X_val)[:, 1]
    xgb_oof[val_idx] = xgb_val
    xgb_test_fold = xgb_model.predict_proba(X_test_all)[:, 1]
    xgb_preds_test += xgb_test_fold / N_FOLDS_XGB

    auc_xgb = roc_auc_score(y_val, xgb_val)
    print(f"XGBoost AUC (fold {fold}): {auc_xgb:.6f}")

    # --------------------- CatBoost -------------------------
    # CatBoost uses categorical indices directly
    cb_cat_features = [X_all.columns.get_loc(c) for c in cat_cols if c in X_all.columns]

    cb_model = CatBoostClassifier(
        depth=6,
        iterations=3000,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE + fold,
        l2_leaf_reg=5.0,
        bagging_temperature=0.5,
        border_count=128,
        task_type="GPU",
        verbose=False
    )

    cb_model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        cat_features=cb_cat_features,
        use_best_model=True
    )

    cb_val = cb_model.predict_proba(X_val)[:, 1]
    cb_oof[val_idx] = cb_val
    cb_test_fold = cb_model.predict_proba(X_test_all)[:, 1]
    cb_preds_test += cb_test_fold / N_FOLDS_XGB

    auc_cb = roc_auc_score(y_val, cb_val)
    print(f"CatBoost AUC (fold {fold}): {auc_cb:.6f}")

    # fold-level blend just for monitoring
    blend_val = BLEND_W_XGB * xgb_val + (1 - BLEND_W_XGB) * cb_val
    auc_blend = roc_auc_score(y_val, blend_val)
    print(f"Blended AUC (fold {fold}): {auc_blend:.6f}")

    del xgb_model, cb_model
    gc.collect()

# ============================================================
#  9. Overall OOF metrics
# ============================================================

auc_xgb_oof   = roc_auc_score(y, xgb_oof)
auc_cb_oof    = roc_auc_score(y, cb_oof)
blend_oof     = BLEND_W_XGB * xgb_oof + (1 - BLEND_W_XGB) * cb_oof
auc_blend_oof = roc_auc_score(y, blend_oof)

print("\n" + "="*80)
print("OOF AUC-ROC Scores")
print("="*80)
print(f"XGBoost OOF AUC : {auc_xgb_oof:.6f}")
print(f"CatBoost OOF AUC: {auc_cb_oof:.6f}")
print(f"Blend   OOF AUC : {auc_blend_oof:.6f}")
print("="*80)

# ============================================================
# 10. Final blended test predictions (full-data retrain)
# ============================================================

print("\nRetraining models on FULL training data for final submission...")

# --- XGBoost full-data model ---
xgb_full = XGBClassifier(
    n_estimators=4000,
    learning_rate=0.025,
    objective="binary:logistic",
    eval_metric="auc",
    max_depth=5,
    max_leaves=48,
    subsample=0.9,
    colsample_bytree=0.7,
    min_child_weight=4.0,
    gamma=0.5,
    reg_alpha=0.1,
    reg_lambda=2.0,
    tree_method="gpu_hist",
    gpu_id=0,
    max_bin=256,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# You *can* do a tiny internal split for early stopping if you want,
# but here we just train on all data to use every row.
xgb_full.fit(X_all, y, verbose=False)

xgb_test_full = xgb_full.predict_proba(X_test_all)[:, 1]


# --- CatBoost full-data model ---
cb_cat_features = [X_all.columns.get_loc(c) for c in cat_cols if c in X_all.columns]

cb_full = CatBoostClassifier(
    depth=6,
    iterations=8000,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    l2_leaf_reg=5.0,
    bagging_temperature=0.5,
    border_count=128,
    task_type="GPU",
    verbose=False
)

cb_full.fit(
    X_all, y,
    cat_features=cb_cat_features
)

cb_test_full = cb_full.predict_proba(X_test_all)[:, 1]


# --- Final blend: full-data XGB + full-data CatBoost ---
test_preds_blend_full = BLEND_W_XGB * xgb_test_full + (1 - BLEND_W_XGB) * cb_test_full

submission = pd.DataFrame({
    "id": X_test_ids,
    "loan_paid_back": test_preds_blend_full
})

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv (full-data blended XGB+CatBoost)")





