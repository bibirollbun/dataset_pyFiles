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


import numpy as np
import pandas as pd
import gc

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

RANDOM_STATE = 42
N_FOLDS = 5

pd.set_option("display.max_columns", 200)



train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()



def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-3  # avoid division by zero

    if {"loan_amount", "annual_income"}.issubset(df.columns):
        df["loan_to_annual_income"] = df["loan_amount"] / (df["annual_income"] + eps)

    if {"installment", "monthly_income"}.issubset(df.columns):
        df["loan_to_monthly_income"] = df["installment"] / (df["monthly_income"] + eps)

    if {"current_balance", "total_credit_limit"}.issubset(df.columns):
        df["credit_utilisation"] = df["current_balance"] / (df["total_credit_limit"] + eps)

    if {"installment", "total_credit_limit"}.issubset(df.columns):
        df["installment_to_limit"] = df["installment"] / (df["total_credit_limit"] + eps)

    if {"delinquency_history", "num_of_delinquencies"}.issubset(df.columns):
        df["total_delinquencies"] = df["delinquency_history"] + df["num_of_delinquencies"]

    if {"interest_rate", "loan_term"}.issubset(df.columns):
        df["interest_burden"] = df["interest_rate"] * df["loan_term"]

    return df



TARGET_COL = "loan_paid_back"
ID_COL = "id"

feature_cols = [c for c in train.columns if c not in [TARGET_COL, ID_COL]]

train_features = train[feature_cols].copy()
test_features = test[feature_cols].copy()

full = pd.concat([train_features, test_features], axis=0, ignore_index=True)

full = add_features(full)

# All object columns (gender, marital_status, etc.)
cat_cols = full.select_dtypes(include=["object"]).columns.tolist()
print("Categorical columns:", cat_cols)

# One-hot encode categoricals -> everything is numeric now
full_encoded = pd.get_dummies(full, columns=cat_cols, drop_first=True)

print("Full encoded shape:", full_encoded.shape)
print("Dtypes check:")
print(full_encoded.dtypes.head(20))

# Split back
X = full_encoded.iloc[: len(train)].reset_index(drop=True)
X_test = full_encoded.iloc[len(train):].reset_index(drop=True)
y = train[TARGET_COL].copy()

print("X shape:", X.shape)
print("X_test shape:", X_test.shape)
print("y distribution:")
print(y.value_counts(normalize=True))


from sklearn.model_selection import KFold

TARGET_COL = "loan_paid_back"
ID_COL = "id"

# choose some categoricals that are likely useful + not too low-cardinality
high_card_cols = [
    "grade_subgrade",
    "loan_purpose",
]

# only keep those that actually exist
high_card_cols = [c for c in high_card_cols if c in train.columns]
print("Target-encode columns:", high_card_cols)


def add_target_encoding(train, test, cols, target_col, n_folds=5, seed=42):
    """
    Returns:
      train_te: DataFrame with target-encoded cols for train (out-of-fold)
      test_te:  DataFrame with target-encoded cols for test (fit on full train)
    """
    train_te = pd.DataFrame(index=train.index)
    test_te = pd.DataFrame(index=test.index)
    
    # global prior mean of target
    global_mean = train[target_col].mean()
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    for col in cols:
        print(f"\nTarget encoding for: {col}")
        
        te_col_train = np.zeros(len(train))
        
        # Out-of-fold encoding for train
        for fold, (trn_idx, val_idx) in enumerate(kf.split(train), 1):
            trn_df = train.iloc[trn_idx]
            val_df = train.iloc[val_idx]
            
            # mean target per category using only fold-train
            mapping = trn_df.groupby(col)[target_col].mean()
            
            # map categories in validation set
            te_col_train[val_idx] = val_df[col].map(mapping).fillna(global_mean)
            
            print(f"  Fold {fold} done. Unique cats in train-fold: {len(mapping)}")
        
        train_te[f"{col}_te"] = te_col_train
        
        # For test: fit on FULL train
        full_mapping = train.groupby(col)[target_col].mean()
        test_te[f"{col}_te"] = test[col].map(full_mapping).fillna(global_mean)
    
    return train_te, test_te


train_te, test_te = add_target_encoding(train, test, high_card_cols, TARGET_COL)
print("\nTrain TE shape:", train_te.shape)
print("Test TE shape:", test_te.shape)




# Make sure indices align
X = X.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

train_te = train_te.reset_index(drop=True)
test_te = test_te.reset_index(drop=True)

X_enh = pd.concat([X, train_te], axis=1)
X_test_enh = pd.concat([X_test, test_te], axis=1)

print("Old X shape:", X.shape)
print("New X_enh shape:", X_enh.shape)



from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import gc

TARGET_COL = "loan_paid_back"
ID_COL = "id"

# Features for CatBoost: use original train with string categoricals
cat_features_all = train.select_dtypes(include=["object"]).columns.tolist()
print("CatBoost categorical features:", cat_features_all)

feature_cols_cb = [c for c in train.columns if c not in [TARGET_COL, ID_COL]]
X_cb = train[feature_cols_cb].copy()
X_test_cb = test[feature_cols_cb].copy()
y_cb = train[TARGET_COL].copy()

# indices (positions) of categorical columns
cat_idx = [X_cb.columns.get_loc(c) for c in cat_features_all if c in X_cb.columns]
print("Categorical column indices for CatBoost:", cat_idx)

N_FOLDS = 5
RANDOM_STATE = 123

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

cat_oof = np.zeros(len(X_cb))
cat_test = np.zeros(len(X_test_cb))

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_cb, y_cb), 1):
    print(f"\n[CatBoost] Fold {fold}/{N_FOLDS}")
    
    X_trn, X_val = X_cb.iloc[trn_idx], X_cb.iloc[val_idx]
    y_trn, y_val = y_cb.iloc[trn_idx], y_cb.iloc[val_idx]
    
    train_pool = Pool(X_trn, label=y_trn, cat_features=cat_idx)
    valid_pool = Pool(X_val, label=y_val, cat_features=cat_idx)
    test_pool  = Pool(X_test_cb,            cat_features=cat_idx)
    
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=4000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE + fold,
        od_type="Iter",
        od_wait=200,
        verbose=200,
    )
    
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
    )
    
    val_pred = model.predict_proba(valid_pool)[:, 1]
    test_pred_fold = model.predict_proba(test_pool)[:, 1]
    
    cat_oof[val_idx] = val_pred
    cat_test += test_pred_fold / N_FOLDS
    
    fold_auc = roc_auc_score(y_val, val_pred)
    print(f"[CatBoost] Fold {fold} AUC: {fold_auc:.6f}")
    
    del X_trn, X_val, y_trn, y_val, model, train_pool, valid_pool, test_pool
    gc.collect()

cat_oof_auc = roc_auc_score(y_cb, cat_oof)
print(f"\n[CatBoost] Overall OOF AUC: {cat_oof_auc:.6f}")



from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import gc

N_FOLDS = 5
RANDOM_STATE = 7  # different from LGBM seed to diversify

def train_xgb(X, y, X_test, n_folds=5, seed=7):
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    oof = np.zeros(len(X))
    preds_test = np.zeros(len(X_test))
    
    pos = y.sum()
    neg = len(y) - pos
    scale_pos_weight = neg / pos
    print(f"[XGB] pos_ratio={pos/len(y):.4f}, scale_pos_weight={scale_pos_weight:.2f}")
    
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"\n[XGB] Fold {fold}/{n_folds}")
        
        X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
        
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,      # L1
            reg_lambda=1.0,     # L2
            tree_method="hist", # fast on Kaggle
            scale_pos_weight=scale_pos_weight,
            random_state=seed + fold,
            n_jobs=-1,
        )
        
        model.fit(
            X_trn, y_trn,
            eval_set=[(X_val, y_val)],
            verbose=200,   # prints AUC every 200 rounds
        )
        
        val_pred = model.predict_proba(X_val)[:, 1]
        test_pred_fold = model.predict_proba(X_test)[:, 1]
        
        oof[val_idx] = val_pred
        preds_test += test_pred_fold / n_folds
        
        fold_auc = roc_auc_score(y_val, val_pred)
        print(f"[XGB] Fold {fold} AUC: {fold_auc:.6f}")
        
        del X_trn, X_val, y_trn, y_val, model
        gc.collect()
    
    oof_auc = roc_auc_score(y, oof)
    print(f"\n[XGB] Overall OOF AUC: {oof_auc:.6f}")
    return oof, preds_test

xgb_oof, xgb_test = train_xgb(X, y, X_test, n_folds=N_FOLDS, seed=7)



import numpy as np
import gc
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

N_FOLDS = 5
RANDOM_STATE = 42

kf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

def train_lgbm(X, y, X_test, seed=42):
    oof = np.zeros(len(X))
    preds_test = np.zeros(len(X_test))

    pos = y.sum()
    neg = len(y) - pos
    scale_pos_weight = neg / pos

    print(f"Training LGBM with seed={seed}, pos_ratio={pos/len(y):.4f}")

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"\n[LGBM] Fold {fold}/{N_FOLDS}")

        X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=8000,          # more trees
            learning_rate=0.02,         # smaller LR
            num_leaves=63,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=40,
            reg_alpha=0.2,              # L1
            reg_lambda=1.0,             # L2
            random_state=seed + fold,
            n_jobs=-1,
            class_weight="balanced",
            # boosting_type="gbdt",     # default
        )

        model.fit(
            X_trn,
            y_trn,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=300, verbose=True),
                lgb.log_evaluation(period=200),
            ],
        )

        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is not None:
            val_pred = model.predict_proba(X_val, num_iteration=best_iter)[:, 1]
            test_pred_fold = model.predict_proba(X_test, num_iteration=best_iter)[:, 1]
        else:
            val_pred = model.predict_proba(X_val)[:, 1]
            test_pred_fold = model.predict_proba(X_test)[:, 1]

        oof[val_idx] = val_pred
        preds_test += test_pred_fold / N_FOLDS

        fold_auc = roc_auc_score(y_val, val_pred)
        print(f"[LGBM] Fold {fold} AUC: {fold_auc:.6f}")

        del X_trn, X_val, y_trn, y_val, model
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    print(f"\n[LGBM] Overall OOF AUC: {oof_auc:.6f}")
    return oof, preds_test

# Seed-averaged LightGBM (bagging over seeds)
lgbm_oof = np.zeros(len(X))
lgbm_test = np.zeros(len(X_test))
seeds = [42, 2024]   # you can add more seeds if time allows

for s in seeds:
    oof_s, test_s = train_lgbm(X, y, X_test, seed=s)
    lgbm_oof += oof_s / len(seeds)
    lgbm_test += test_s / len(seeds)

print("\n[LGBM] Bagged OOF AUC:", roc_auc_score(y, lgbm_oof))



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Base model OOF preds: adjust these names to match your notebook
# lgbm_oof, xgb_oof, cat_oof
meta_X = np.vstack([lgbm_oof, xgb_oof, cat_oof]).T  # shape: (n_train, 3)
meta_y = y.values                                   # or y_cb.values

print("Meta feature shape:", meta_X.shape)



kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)

stack_oof = np.zeros(len(meta_y))
stack_coefs = []

for fold, (trn_idx, val_idx) in enumerate(kf.split(meta_X, meta_y), 1):
    print(f"\n[Stacker] Fold {fold}/5")
    X_trn, X_val = meta_X[trn_idx], meta_X[val_idx]
    y_trn, y_val = meta_y[trn_idx], meta_y[val_idx]
    
    stacker = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs"
    )
    stacker.fit(X_trn, y_trn)
    
    val_pred = stacker.predict_proba(X_val)[:, 1]
    stack_oof[val_idx] = val_pred
    
    auc = roc_auc_score(y_val, val_pred)
    print(f"[Stacker] Fold {fold} AUC: {auc:.6f}")
    stack_coefs.append(stacker.coef_[0])

stack_auc = roc_auc_score(meta_y, stack_oof)
print(f"\n[Stacker] Overall OOF AUC: {stack_auc:.6f}")
print("Average stacker coefficients:", np.mean(stack_coefs, axis=0))



meta_X_test = np.vstack([lgbm_test, xgb_test, cat_test]).T  # shape: (n_test, 3)



final_stacker = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver="lbfgs"
)
final_stacker.fit(meta_X, meta_y)

final_test_pred = final_stacker.predict_proba(meta_X_test)[:, 1]



submission = sample_submission.copy()
submission["loan_paid_back"] = final_test_pred
submission.to_csv("submission.csv", index=False)
submission.head()


