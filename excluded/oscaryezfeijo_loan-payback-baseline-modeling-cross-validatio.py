# CHUNK 1 — Imports & Setup

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from catboost import CatBoostClassifier, Pool
import lightgbm as lgb

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "/kaggle/input/playground-series-s5e11"
print("Data directory contents:", os.listdir(DATA_DIR))


# CHUNK 2 — Load Train / Test Data

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_sub_path)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

display(train.head())


# CHUNK 3 — Define X, y, and Identify Column Types

TARGET_COL = "loan_paid_back"
ID_COL = "id"

X = train.drop(columns=[TARGET_COL])
y = train[TARGET_COL]

X_test = test.copy()

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

if ID_COL in numeric_features:
    numeric_features.remove(ID_COL)

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)



# CHUNK 4 — Feature Engineering

from itertools import combinations

X_engineered = X.copy()
X_test_engineered = X_test.copy()

# Simple interactions between top numeric features
for col1, col2 in combinations(numeric_features[:5], 2):
    X_engineered[f"{col1}_x_{col2}"] = X_engineered[col1] * X_engineered[col2]
    X_test_engineered[f"{col1}_x_{col2}"] = X_test_engineered[col1] * X_test_engineered[col2]

# Squared terms for non-linearity
for col in numeric_features[:5]:
    X_engineered[f"{col}_sq"] = X_engineered[col] ** 2
    X_test_engineered[f"{col}_sq"] = X_test_engineered[col] ** 2

X = X_engineered
X_test = X_test_engineered

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

if ID_COL in numeric_features:
    numeric_features.remove(ID_COL)

print("After feature engineering:")
print("Total features:", len(X.columns))
print("Numeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))




# CHUNK 5 — CatBoost Model (CV + Full Training + CatBoost-Only Submission)

cat_features = [col for col in X.columns if X[col].dtype == "object"]
print(f"Categorical features detected: {len(cat_features)}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(X))

print("Training CatBoost with 5-Fold CV...")

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    X_tr = X.iloc[train_idx]
    y_tr = y.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    val_pool   = Pool(X_val, y_val, cat_features=cat_features)

    model_cb = CatBoostClassifier(
        iterations=1500,
        depth=8,
        learning_rate=0.03,
        l2_leaf_reg=3,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=RANDOM_STATE + fold,
        verbose=False
    )

    model_cb.fit(train_pool, eval_set=val_pool)

    preds = model_cb.predict_proba(X_val)[:, 1]
    oof[val_idx] = preds

    auc = roc_auc_score(y_val, preds)
    print(f"Fold {fold} ROC AUC: {auc:.5f}")

catboost_oof = oof.copy()
catboost_oof_auc = roc_auc_score(y, catboost_oof)
print(f"CatBoost OOF ROC AUC: {catboost_oof_auc:.5f}")

# ---- Train Final CatBoost Model on Full Data ---- #

final_catboost = CatBoostClassifier(
    iterations=1500,
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=3,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=RANDOM_STATE,
    verbose=False
)

full_pool = Pool(X, y, cat_features=cat_features)
final_catboost.fit(full_pool)

# ---- Predict on Test with CatBoost ---- #

test_pool = Pool(X_test, cat_features=cat_features)
catboost_test_preds = final_catboost.predict_proba(test_pool)[:, 1]

# We will blend later in CHUNK 6, but for now we can also save CatBoost-only submission:
submission_cb = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": catboost_test_preds
})

submission_cb.to_csv("submission_catboost_only.csv", index=False)
print("✓ submission_catboost_only.csv created (CatBoost only).")
display(submission_cb.head())



# CHUNK 6 — LightGBM Model + Blend with CatBoost (Final Submission)

print("\n==== Training LightGBM with 5-Fold CV ====")

# Preprocessor for LightGBM (numeric passthrough + OHE for categoricals)
numeric_transformer = 'passthrough'

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

cv_lgb = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
lgbm_oof = np.zeros(len(X))
test_pred_folds = np.zeros((len(X_test), cv_lgb.n_splits))

for fold, (train_idx, val_idx) in enumerate(cv_lgb.split(X, y), 1):
    X_tr = X.iloc[train_idx]
    y_tr = y.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    lgbm_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", lgb.LGBMClassifier(
                n_estimators=800,
                learning_rate=0.03,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary",
                random_state=RANDOM_STATE + fold,
                n_jobs=-1
            ))
        ]
    )

    lgbm_model.fit(X_tr, y_tr)

    val_pred = lgbm_model.predict_proba(X_val)[:, 1]
    lgbm_oof[val_idx] = val_pred

    fold_auc = roc_auc_score(y_val, val_pred)
    print(f"Fold {fold} ROC AUC (LightGBM): {fold_auc:.5f}")

    test_pred_folds[:, fold - 1] = lgbm_model.predict_proba(X_test)[:, 1]

lgbm_oof_auc = roc_auc_score(y, lgbm_oof)
print(f"LightGBM OOF ROC AUC: {lgbm_oof_auc:.5f}")

lgbm_test_preds = test_pred_folds.mean(axis=1)

# ---- Blend CatBoost + LightGBM ---- #

# CatBoost has shown stronger CV so we weight it a bit more
alpha = 0.65  # weight for CatBoost; 0.35 for LightGBM
blended_oof = alpha * catboost_oof + (1 - alpha) * lgbm_oof
blended_oof_auc = roc_auc_score(y, blended_oof)
print(f"\nBlended OOF ROC AUC (alpha={alpha:.2f} CatBoost): {blended_oof_auc:.5f}")

blended_test_preds = alpha * catboost_test_preds + (1 - alpha) * lgbm_test_preds

submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": blended_test_preds
})

submission.to_csv("submission.csv", index=False)
print("✓ Final blended submission.csv created (CatBoost + LightGBM).")
display(submission.head())


