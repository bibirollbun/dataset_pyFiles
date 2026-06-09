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


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


# LOAD DATA
# -------------------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")



# target and ids
y = train["diagnosed_diabetes"]
train_ids = train["id"]
test_ids = test["id"]



# drop id from features
X = train.drop(["id", "diagnosed_diabetes"], axis=1)
test_X = test.drop(["id"], axis=1)


# BASIC SANITY & TYPE FIXES
# -------------------------------
# Convert obvious numeric columns to numeric (coerce errors -> NaN)
numeric_cols = [
    "age", "alcohol_consumption_per_week", "physical_activity_minutes_per_week",
    "diet_score", "sleep_hours_per_day", "screen_time_hours_per_day",
    "bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp", "heart_rate",
    "cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides"
]




for c in numeric_cols:
    if c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        test_X[c] = pd.to_numeric(test_X[c], errors="coerce")



# Categorical columns (based on your list)
cat_cols = [
    "gender", "ethnicity", "education_level", "income_level", "smoking_status",
    "employment_status", "family_history_diabetes", "hypertension_history",
    "cardiovascular_history"
]
cat_cols = [c for c in cat_cols if c in X.columns]



# Fill missing values:
# - numeric: median
# - categorical: 'missing'
for c in numeric_cols:
    if c in X.columns:
        med = X[c].median()
        X[c].fillna(med, inplace=True)
        test_X[c].fillna(med, inplace=True)

for c in cat_cols:
    X[c] = X[c].astype(object).fillna("missing")
    test_X[c] = test_X[c].astype(object).fillna("missing")



# FEATURE ENGINEERING (your columns)
# -------------------------------
# Blood pressure ratio
if {"systolic_bp", "diastolic_bp"}.issubset(X.columns):
    X["bp_ratio"] = X["systolic_bp"] / (X["diastolic_bp"] + 1e-6)
    test_X["bp_ratio"] = test_X["systolic_bp"] / (test_X["diastolic_bp"] + 1e-6)



# Cholesterol ratios
if {"cholesterol_total", "hdl_cholesterol"}.issubset(X.columns):
    X["chol_hdl_ratio"] = X["cholesterol_total"] / (X["hdl_cholesterol"] + 1e-6)
    test_X["chol_hdl_ratio"] = test_X["cholesterol_total"] / (test_X["hdl_cholesterol"] + 1e-6)

if {"ldl_cholesterol", "hdl_cholesterol"}.issubset(X.columns):
    X["ldl_hdl_ratio"] = X["ldl_cholesterol"] / (X["hdl_cholesterol"] + 1e-6)
    test_X["ldl_hdl_ratio"] = test_X["ldl_cholesterol"] / (test_X["hdl_cholesterol"] + 1e-6)



# Lifestyle combined risks
if {"alcohol_consumption_per_week", "sleep_hours_per_day"}.issubset(X.columns):
    X["alcohol_sleep_interaction"] = X["alcohol_consumption_per_week"] * X["sleep_hours_per_day"]
    test_X["alcohol_sleep_interaction"] = test_X["alcohol_consumption_per_week"] * test_X["sleep_hours_per_day"]

if {"screen_time_hours_per_day", "physical_activity_minutes_per_week"}.issubset(X.columns):
    X["screen_activity_ratio"] = X["screen_time_hours_per_day"] / (X["physical_activity_minutes_per_week"] + 1e-6)
    test_X["screen_activity_ratio"] = test_X["screen_time_hours_per_day"] / (test_X["physical_activity_minutes_per_week"] + 1e-6)



# Obesity interactions
if {"bmi", "waist_to_hip_ratio"}.issubset(X.columns):
    X["bmi_waist_interaction"] = X["bmi"] * X["waist_to_hip_ratio"]
    test_X["bmi_waist_interaction"] = test_X["bmi"] * test_X["waist_to_hip_ratio"]



# Age x smoking (encode smoking temporarily to numeric codes)
if "smoking_status" in X.columns:
    X["_smoke_code"] = X["smoking_status"].astype("category").cat.codes
    test_X["_smoke_code"] = test_X["smoking_status"].astype("category").cat.codes
    X["age_smoking_interaction"] = X["age"] * X["_smoke_code"]
    test_X["age_smoking_interaction"] = test_X["age"] * test_X["_smoke_code"]
    X.drop("_smoke_code", axis=1, inplace=True)
    test_X.drop("_smoke_code", axis=1, inplace=True)



# ENCODING: Out-of-fold Target Encoding for selected categorical cols
# (safe: computed only on train-fold, applied to val/test)
# -------------------------------
# Decide which categorical cols to target-encode (higher-cardinality)
te_candidates = []
for c in cat_cols:
    if X[c].nunique() > 8:  # threshold; tuneable
        te_candidates.append(c)

print("Will target-encode:", te_candidates)




# Prepare placeholders
X_te = X.copy()
test_te = test_X.copy()

# We'll create numeric copies to feed to XGBoost
for c in te_candidates:
    X_te[c] = np.nan
    test_te[c] = np.nan


# OOF scheme to compute target-encoding mapping
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr_idx, val_idx in kf.split(X, y):
    X_tr = X.iloc[tr_idx]
    y_tr = y.iloc[tr_idx]
    X_val = X.iloc[val_idx]

    for c in te_candidates:
        # mapping from category -> mean target in train fold
        mapping = X_tr.groupby(c)[y_tr.name].mean() if False else X_tr.groupby(c)[y_tr.name].mean()  # stub to clarify
        # simpler: groupby on X_tr with y_tr values aligned by index
        temp = pd.DataFrame({c: X_tr[c].values, "target": y_tr.values})
        mapping = temp.groupby(c)["target"].mean()
        # map val
        X_te.loc[X_val.index, c] = X_val[c].map(mapping)


# For test: build mapping on full train
for c in te_candidates:
    temp = pd.DataFrame({c: X[c].values, "target": y.values})
    mapping = temp.groupby(c)["target"].mean()
    test_te[c] = test_X[c].map(mapping)

# Replace any remaining NaNs (unseen categories) with global mean
global_mean = y.mean()
for c in te_candidates:
    X_te[c].fillna(global_mean, inplace=True)
    test_te[c].fillna(global_mean, inplace=True)


# For non-TE categorical columns, factorize (ordinal codes)
remaining_cats = [c for c in cat_cols if c not in te_candidates]
for c in remaining_cats:
    # create mapping from combined train+test to ensure consistent codes
    cats = pd.Series(pd.Categorical(pd.concat([X[c], test_X[c]], axis=0)))
    codes = cats.cat.codes[: len(X)].values
    codes_test = cats.cat.codes[len(X):].values
    X_te[c] = codes
    test_te[c] = codes_test


# -------------------------------
# FINAL PREP: ensure there are no NaNs and all numeric
# -------------------------------
# Merge numeric-built features back in case TE columns replaced some dtypes
final_X = X_te.copy()
final_test = test_te.copy()

# Fill any remaining NaNs with median of column
for col in final_X.columns:
    if final_X[col].isnull().any():
        med = final_X[col].median()
        final_X[col].fillna(med, inplace=True)
        if col in final_test.columns:
            final_test[col].fillna(med, inplace=True)


# Convert all to numeric dtype
final_X = final_X.apply(pd.to_numeric, errors="coerce")
final_test = final_test.apply(pd.to_numeric, errors="coerce")

# safety fill
final_X.fillna(final_X.median(), inplace=True)
final_test.fillna(final_X.median(), inplace=True)

print("Final feature count:", final_X.shape[1])


# -------------------------------
# MODELING: Stratified 5-Fold XGBoost with improvements
# -------------------------------
oof_preds = np.zeros(len(final_X))
test_preds = np.zeros(len(final_test))

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_num = 1
for tr_idx, val_idx in kf.split(final_X, y):
    print(f"\nðŸ”µ Training fold {fold_num}...")

    X_tr, X_val = final_X.iloc[tr_idx], final_X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = XGBClassifier(
        n_estimators=4000,               # more trees â†’ higher AUC
        max_depth=6,                    # slightly deeper trees
        learning_rate=0.015,            # smaller LR + more trees = better generalization
        subsample=0.90,
        colsample_bytree=0.80,
        reg_alpha=4,                    # stronger L1
        reg_lambda=6,                   # stronger L2
        min_child_weight=3,
        gamma=0.1,                      # helps avoid overfitting
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42 + fold_num,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=200,      # larger early stopping â†’ better AUC
        verbose=False
    )

    # Predictions
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    test_preds += model.predict_proba(final_test)[:, 1] / 5

    # AUC Score
    fold_auc = roc_auc_score(y_val, val_pred)
    print(f"âœ” Fold {fold_num} AUC: {fold_auc:.5f}")

    fold_num += 1

# -------------------------------
# Final OOF AUC
# -------------------------------
final_auc = roc_auc_score(y, oof_preds)
print(f"\nðŸ”¥ Final OOF AUC: {final_auc:.5f}")




# Final OOF AUC
final_auc = roc_auc_score(y, oof_preds)
print("\n===============================")
print(f"Final 5-Fold OOF AUC: {final_auc:.6f}")
print("===============================")

# -------------------------------
# SUBMISSION
# -------------------------------
submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": np.clip(test_preds, 0, 1)
})
submission.to_csv("submission_xgb_full.csv", index=False)
print("Saved submission_xgb_full.csv")




print(len(submission), len(test))



y = train["diagnosed_diabetes"]





for tr_idx, val_idx in kf.split(X, y):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]



submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": np.clip(test_preds, 0, 1)
})
submission.to_csv("submission.csv", index=False)





