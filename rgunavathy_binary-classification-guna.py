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


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier


# Path to your dataset
file_path = r"/kaggle/input/playground-series-s5e8/train.csv"
test_path = r"/kaggle/input/playground-series-s5e8/test.csv"
submit_path = r"/kaggle/input/playground-series-s5e8/sample_submission.csv"

# Load the dataset
train_df = pd.read_csv(file_path)
test_df = pd.read_csv(test_path)
submit_df = pd.read_csv(submit_path)

# Make a true copy
train_df_copy = train_df.copy(deep=True)
test_df_copy = test_df.copy(deep=True)
submit_df_copy = submit_df.copy(deep=True)


# Check shape and first few rows
print("Original DataFrame shape:", train_df.shape)
print("Copied DataFrame shape:", train_df_copy.shape)
print(train_df_copy.head())

# Quick checks
print("Test DataFrame shape:", test_df.shape)
print(test_df.head())
print("Submit DataFrame shape:", submit_df.shape)
print(submit_df.head())



# # Separate features & target
# X = train_df.drop(columns=["y"])
# y = train_df["y"]

# # Use X (features only) to detect dtypes
# numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
# categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# print("Numerical Features:", numerical_features)
# print("Categorical Features:", categorical_features)

# # Define test features (keeping id)
# test_features = test_df.copy()

# print("reached here1")

# # Combine train + test for consistent encoding
# combined = pd.concat([X, test_features], axis=0)

# # One-hot encode categoricals
# combined_encoded = pd.get_dummies(combined, drop_first=True)

# # Split back
# X_encoded = combined_encoded.iloc[:len(X), :]
# test_encoded = combined_encoded.iloc[len(X):, :]

# print("reached here2")

# # Train/validation split
# X_train, X_val, y_train, y_val = train_test_split(
#     X_encoded, y, test_size=0.2, random_state=42, stratify=y
# )

# # Train model
# clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
# clf.fit(X_train, y_train)

# print("reached here3")

# # Validation predictions
# val_preds = clf.predict_proba(X_val)[:, 1]
# auc = roc_auc_score(y_val, val_preds)
# print("Validation ROC AUC:", auc)

# print("reached here4")
# # Predictions for test set
# test_preds = clf.predict_proba(test_encoded)[:, 1]

# # Prepare submission
# submission = pd.DataFrame({
#     "id": test_df["id"],  # original IDs
#     "y": test_preds
# })

# print("reached here5")

# submission.to_csv("submission.csv", index=False)
# print("Submission file saved: submission.csv")


# # Features & target
# X = train_df.drop(columns=["y"])
# y = train_df["y"]
# X_test = test_df.copy()

# # Combine train + test for consistent encoding
# combined = pd.concat([X, X_test], axis=0)
# combined_encoded = pd.get_dummies(combined, drop_first=True)

# # Split back
# X_encoded = combined_encoded.iloc[:len(X), :]
# X_test_encoded = combined_encoded.iloc[len(X):, :]

# # Stratified K-Fold CV
# n_splits = 5
# skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# oof_preds = np.zeros(len(X_encoded))
# test_preds = np.zeros(len(X_test_encoded))

# for fold, (train_idx, val_idx) in enumerate(skf.split(X_encoded, y)):
#     print(f"\nFOLD {fold+1}/{n_splits}")
    
#     X_train, X_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
#     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

#     model = LGBMClassifier(
#         n_estimators=5000,
#         learning_rate=0.01,
#         num_leaves=63,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42,
#         n_jobs=-1
#     )

#     from lightgbm import early_stopping, log_evaluation

#     model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         eval_metric="auc",
#         callbacks=[early_stopping(200), log_evaluation(200)]
#     )

#     # Out-of-fold predictions
#     oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

#     # Test predictions (averaged)
#     test_preds += model.predict_proba(X_test_encoded)[:, 1] / n_splits

# # Overall CV ROC AUC
# cv_score = roc_auc_score(y, oof_preds)
# print("\nCV ROC AUC:", cv_score)

# # Submission
# submission = pd.DataFrame({
#     "id": test_df["id"],
#     "y": test_preds
# })

# submission.to_csv("submission.csv", index=False)
# print("Saved submission.csv")


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import roc_auc_score

# # Models
# from lightgbm import LGBMClassifier, early_stopping, log_evaluation
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier

# # # Paths (Kaggle)
# # train_path = "/kaggle/input/playground-series-s5e8/train.csv"
# # test_path  = "/kaggle/input/playground-series-s5e8/test.csv"
# # sub_path   = "/kaggle/input/playground-series-s5e8/sample_submission.csv"

# # # Load
# # train_df = pd.read_csv(train_path)
# # test_df  = pd.read_csv(test_path)
# # submit_df = pd.read_csv(sub_path)

# # Features/target
# X = train_df.drop(columns=["y"])  # keep 'id'
# y = train_df["y"]
# X_test = test_df.copy()           # keep 'id'

# # One-hot encode (consistent columns across train/test)
# combined = pd.concat([X, X_test], axis=0)
# combined_enc = pd.get_dummies(combined, drop_first=True)

# X_enc = combined_enc.iloc[:len(X)].reset_index(drop=True)
# X_test_enc = combined_enc.iloc[len(X):].reset_index(drop=True)
# y = y.reset_index(drop=True)

# # CV setup
# n_splits = 5
# skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# # OOF and test preds per model
# oof_lgb  = np.zeros(len(X_enc))
# oof_xgb  = np.zeros(len(X_enc))
# oof_cat  = np.zeros(len(X_enc))

# test_lgb = np.zeros(len(X_test_enc))
# test_xgb = np.zeros(len(X_test_enc))
# test_cat = np.zeros(len(X_test_enc))

# for fold, (trn_idx, val_idx) in enumerate(skf.split(X_enc, y), 1):
#     print(f"\n========== FOLD {fold}/{n_splits} ==========")
#     X_tr, X_va = X_enc.iloc[trn_idx], X_enc.iloc[val_idx]
#     y_tr, y_va = y.iloc[trn_idx], y.iloc[val_idx]

#     # ---- LightGBM ----
#     lgb = LGBMClassifier(
#         n_estimators=10000,
#         learning_rate=0.01,
#         num_leaves=127,
#         subsample=0.9,
#         colsample_bytree=0.9,
#         max_depth=-1,
#         min_child_samples=20,
#         reg_alpha=1.0,
#         reg_lambda=1.0,
#         random_state=42,
#         n_jobs=-1
#     )
#     lgb.fit(
#         X_tr, y_tr,
#         eval_set=[(X_va, y_va)],
#         eval_metric="auc",
#         callbacks=[early_stopping(200), log_evaluation(200)]
#     )
#     oof_lgb[val_idx] = lgb.predict_proba(X_va)[:, 1]
#     test_lgb += lgb.predict_proba(X_test_enc)[:, 1] / n_splits

#     # ---- XGBoost ----
#     xgb = XGBClassifier(
#         n_estimators=10000,
#         learning_rate=0.01,
#         max_depth=6,
#         subsample=0.9,
#         colsample_bytree=0.9,
#         reg_alpha=1.0,
#         reg_lambda=1.0,
#         objective="binary:logistic",
#         eval_metric="auc",
#         tree_method="hist",
#         random_state=42,
#         n_jobs=-1
#     )
#     xgb.fit(
#         X_tr, y_tr,
#         eval_set=[(X_va, y_va)],
#         early_stopping_rounds=200,
#         verbose=200
#     )
#     oof_xgb[val_idx] = xgb.predict_proba(X_va)[:, 1]
#     test_xgb += xgb.predict_proba(X_test_enc)[:, 1] / n_splits

#     # ---- CatBoost ----
#     cat = CatBoostClassifier(
#         iterations=10000,
#         learning_rate=0.01,
#         depth=6,
#         loss_function="Logloss",
#         eval_metric="AUC",
#         l2_leaf_reg=3.0,
#         random_state=42,
#         allow_writing_files=False,
#         verbose=200
#     )
#     cat.fit(
#         X_tr, y_tr,
#         eval_set=(X_va, y_va),
#         use_best_model=True,
#         early_stopping_rounds=200
#     )
#     oof_cat[val_idx] = cat.predict_proba(X_va)[:, 1]
#     test_cat += cat.predict_proba(X_test_enc)[:, 1] / n_splits

# # OOF AUCs
# auc_lgb = roc_auc_score(y, oof_lgb)
# auc_xgb = roc_auc_score(y, oof_xgb)
# auc_cat = roc_auc_score(y, oof_cat)
# print(f"\nOOF AUCs -> LGBM: {auc_lgb:.6f} | XGB: {auc_xgb:.6f} | CAT: {auc_cat:.6f}")

# # Blend (equal weights; you can weight by OOF AUC if you like)
# oof_blend  = (oof_lgb + oof_xgb + oof_cat) / 3.0
# test_blend = (test_lgb + test_xgb + test_cat) / 3.0

# auc_blend = roc_auc_score(y, oof_blend)
# print(f"OOF AUC (Blend): {auc_blend:.6f}")

# # Submission
# submission = pd.DataFrame({
#     "id": test_df["id"],
#     "y": test_blend
# })
# submission.to_csv("submission.csv", index=False)
# print("Saved submission.csv")


import pandas as pd
import numpy as np
import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

train_path = "/kaggle/input/playground-series-s5e8/train.csv"
test_path  = "/kaggle/input/playground-series-s5e8/test.csv"
sub_path   = "/kaggle/input/playground-series-s5e8/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)
sub_df   = pd.read_csv(sub_path)

print("i am here")
# Feature Engineering
def feature_engineering(df):
    df = df.copy()

    # ---- Balance related
    df["log_balance"] = np.log1p(df["balance"])
    df["is_balance_negative"] = (df["balance"] < 0).astype(int)

    # ---- Call duration related
    df["log_duration"] = np.log1p(df["duration"])
    df["short_call"] = (df["duration"] < 100).astype(int)
    df["long_call"]  = (df["duration"] > 500).astype(int)

    # ---- Contact history
    df["was_contacted_before"] = (df["pdays"] != -1).astype(int)
    df["recent_contact"] = (df["pdays"] < 30).astype(int)

    # ---- Loan features
    df["has_any_loan"] = ((df["housing"] == "yes") | (df["loan"] == "yes")).astype(int)

    # ---- Month encoding
    month_map = {
        'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
        'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12
    }
    df["month_num"] = df["month"].map(month_map)
    df["month_sin"] = np.sin(2*np.pi*df["month_num"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month_num"]/12)

    # ---- Campaign pressure
    df["high_pressure"] = (df["campaign"] > 3).astype(int)
    df["campaign_to_previous"] = df["campaign"] / (df["previous"] + 1)

    # ---- Weekday / Weekend
    df["weekday"] = df.apply(
        lambda row: datetime.date(2008, row["month_num"], row["day"]).weekday(), axis=1
    )
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    return df

print("i am here")
train_fe = feature_engineering(train_df)
test_fe  = feature_engineering(test_df)

X = train_fe.drop(columns=["y"])
y = train_fe["y"]

test_ids = test_fe["id"]

print("i am here")
# Encoding categoricals (One-hot)
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Combine train + test for consistent encoding
combined = pd.concat([X, test_fe], axis=0)
combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

# Split back
X_enc = combined.iloc[:len(X), :].copy()
X_test_enc = combined.iloc[len(X):, :].copy()

# Cross-validation training
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print("i am here")
oof_preds = np.zeros(len(X_enc))
test_preds = np.zeros(len(X_test_enc))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_enc, y)):
    print(f"\nFOLD {fold+1}/{n_splits}")

    X_train, X_val = X_enc.iloc[train_idx], X_enc.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMClassifier(
        n_estimators=5000,
        learning_rate=0.01,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[early_stopping(200), log_evaluation(200)]
    )

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test_enc)[:, 1] / n_splits

print("i am here")
# CV Score
auc = roc_auc_score(y, oof_preds)
print(f"\nOOF ROC AUC = {auc:.5f}")
Print("i am here")

submission = pd.DataFrame({
    "id": test_ids,
    "y": test_preds
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv saved")

