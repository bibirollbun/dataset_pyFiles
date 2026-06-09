# ============================================================
# SIMPLE BLENDING — XGBoost + LightGBM + CatBoost (NO K-FOLDS)
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# ------------------------------
# Load data
# ------------------------------
DATA_DIR = "/kaggle/input/playground-series-s5e12"
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")

TARGET = "diagnosed_diabetes"
ID = "id"


# ------------------------------
# Encode categoricals for XGB/LGBM
# ------------------------------
cat_cols = train.select_dtypes(include=["object"]).columns.tolist()
X = train.drop(columns=[TARGET, ID])
y = train[TARGET]
X_test = test.drop(columns=[ID])

X_enc = X.copy()
X_test_enc = X_test.copy()

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X_enc[col], X_test_enc[col]], axis=0).astype(str)
    le.fit(combined)
    X_enc[col] = le.transform(X_enc[col].astype(str))
    X_test_enc[col] = le.transform(X_test_enc[col].astype(str))


# ------------------------------
# Train/validation split
# ------------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_enc = X_enc.iloc[X_train.index]
X_valid_enc = X_enc.iloc[X_valid.index]


# ============================================================
# 1. Train XGBoost
# ============================================================
xgb = XGBClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=42,
    early_stopping_rounds=50,
)

xgb.fit(X_train_enc, y_train, eval_set=[(X_valid_enc, y_valid)], verbose=False)
xgb_val = xgb.predict_proba(X_valid_enc)[:, 1]
xgb_test = xgb.predict_proba(X_test_enc)[:, 1]

print("XGB ROC:", roc_auc_score(y_valid, xgb_val))


# ============================================================
# 2. Train LightGBM
# ============================================================
lgbm = LGBMClassifier(
    n_estimators=1800,
    learning_rate=0.02,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42,
)

lgbm.fit(X_train_enc, y_train)
lgbm_val = lgbm.predict_proba(X_valid_enc)[:, 1]
lgbm_test = lgbm.predict_proba(X_test_enc)[:, 1]

print("LGBM ROC:", roc_auc_score(y_valid, lgbm_val))


# ============================================================
# 3. Train CatBoost (native categoricals)
# ============================================================
cat = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False,
)

cat.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid),
    cat_features=cat_cols,
    verbose=False
)

cat_val = cat.predict_proba(X_valid)[:, 1]
cat_test = cat.predict_proba(X_test)[:, 1]

print("CatBoost ROC:", roc_auc_score(y_valid, cat_val))


# ============================================================
# 4. BLENDING
# ============================================================

# ---- A) Probability average (simple blend)
prob_blend_val = (xgb_val + lgbm_val + cat_val) / 3
prob_auc = roc_auc_score(y_valid, prob_blend_val)
print("Prob Blend ROC:", prob_auc)

# ---- B) Rank average (BEST for ROC)
rank_blend_val = (
    pd.Series(xgb_val).rank(pct=True).values +
    pd.Series(lgbm_val).rank(pct=True).values +
    pd.Series(cat_val).rank(pct=True).values
) / 3

rank_auc = roc_auc_score(y_valid, rank_blend_val)
print("Rank Blend ROC (BEST):", rank_auc)


# ============================================================
# 5. Final TEST predictions (use rank blend)
# ============================================================

# Rank blend test predictions
rank_blend_test = (
    pd.Series(xgb_test).rank(pct=True).values +
    pd.Series(lgbm_test).rank(pct=True).values +
    pd.Series(cat_test).rank(pct=True).values
) / 3

# Create submission
submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": rank_blend_test
})

submission.to_csv("submission.csv", index=False)
print("\nsubmission.csv saved!")


