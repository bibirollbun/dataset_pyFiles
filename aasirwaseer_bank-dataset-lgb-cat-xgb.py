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

train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample submission shape:", sample_submission.shape)
print("\nTrain columns:", train.columns.tolist())

# Quick peek
print("\nTrain head:")
print(train.head())

print("\nTarget distribution:")
print(train['y'].value_counts(normalize=True))

print("\nMissing values:")
print(train.isnull().sum())



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import numpy as np

# Separate features and target
X = train.drop(columns=['id', 'y'])
y = train['y']
X_test = test.drop(columns=['id'])

# Combine train and test for consistent encoding
combined = pd.concat([X, X_test], axis=0)

# Simple encoding for categorical features
categorical_cols = combined.select_dtypes(include='object').columns.tolist()
combined_encoded = pd.get_dummies(combined, columns=categorical_cols)

# Split back
X_encoded = combined_encoded.iloc[:len(X), :]
X_test_encoded = combined_encoded.iloc[len(X):, :]

# Baseline LightGBM model
model = LGBMClassifier(n_estimators=100, random_state=42)

# Stratified K-Fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(cv.split(X_encoded, y)):
    X_tr, X_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(X_tr, y_tr)
    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    auc_scores.append(auc)
    
    test_preds += model.predict_proba(X_test_encoded)[:, 1] / cv.n_splits
    print(f"Fold {fold + 1} AUC: {auc:.5f}")

print(f"\nMean AUC: {np.mean(auc_scores):.5f}")

np.save('/kaggle/working/lgb_preds.npy', test_preds)

# Prepare submission
submission = sample_submission.copy()
submission['y'] = test_preds
submission.to_csv("baseline_lgbm_submission.csv", index=False)



X_no_duration = X.drop(columns=['duration'])
X_test_no_duration = X_test.drop(columns=['duration'])

combined_no_duration = pd.concat([X_no_duration, X_test_no_duration], axis=0)
combined_no_duration_encoded = pd.get_dummies(combined_no_duration, columns=categorical_cols)

X_encoded_nd = combined_no_duration_encoded.iloc[:len(X), :]
X_test_encoded_nd = combined_no_duration_encoded.iloc[len(X):, :]

auc_scores_nd = []
test_preds_nd = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(cv.split(X_encoded_nd, y)):
    X_tr, X_val = X_encoded_nd.iloc[train_idx], X_encoded_nd.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_tr, y_tr)
    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    auc_scores_nd.append(auc)

    test_preds_nd += model.predict_proba(X_test_encoded_nd)[:, 1] / cv.n_splits
    print(f"[No Duration] Fold {fold + 1} AUC: {auc:.5f}")


print(f"\n[No Duration] Mean AUC: {np.mean(auc_scores_nd):.5f}")



import matplotlib.pyplot as plt
import lightgbm as lgb

# Fit one full model for feature importance (without duration)
model_full = lgb.LGBMClassifier(n_estimators=100, random_state=42)
model_full.fit(X_encoded_nd, y)

# Plot importance
importances = model_full.feature_importances_
feature_names = X_encoded_nd.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(importance_df['feature'][:30][::-1], importance_df['importance'][:30][::-1])
plt.title("Top 30 Feature Importances (No Duration)")
plt.tight_layout()
plt.show()



# Copy the clean dataset (no duration)
X_fe = X_no_duration.copy()

# 1. Age binning
X_fe['age_bin'] = pd.cut(X_fe['age'], bins=[17, 25, 35, 45, 60, 100], labels=False)

# 2. Balance bins
X_fe['balance_bin'] = pd.qcut(X_fe['balance'], q=5, labels=False, duplicates='drop')

# 3. Contacted before?
X_fe['was_contacted'] = (X_fe['pdays'] != -1).astype(int)

# 4. Interaction: balance per campaign
X_fe['balance_per_campaign'] = X_fe['balance'] / (X_fe['campaign'] + 1)

# 5. Contact preference indicator
X_fe['contact_known'] = (X_fe['contact'] != 'unknown').astype(int)

# 6. Month as numeric (optional)
month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
             'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
X_fe['month_num'] = X_fe['month'].map(month_map)

# 7. Encode categorical again after adding new features
X_test_fe = X_test_no_duration.copy()
X_test_fe['age_bin'] = pd.cut(X_test_fe['age'], bins=[17, 25, 35, 45, 60, 100], labels=False)
X_test_fe['balance_bin'] = pd.qcut(X_test_fe['balance'], q=5, labels=False, duplicates='drop')
X_test_fe['was_contacted'] = (X_test_fe['pdays'] != -1).astype(int)
X_test_fe['balance_per_campaign'] = X_test_fe['balance'] / (X_test_fe['campaign'] + 1)
X_test_fe['contact_known'] = (X_test_fe['contact'] != 'unknown').astype(int)
X_test_fe['month_num'] = X_test_fe['month'].map(month_map)

# Combine and encode
combined_fe = pd.concat([X_fe, X_test_fe], axis=0)
combined_fe_encoded = pd.get_dummies(combined_fe, columns=categorical_cols)

X_fe_encoded = combined_fe_encoded.iloc[:len(X), :]
X_test_fe_encoded = combined_fe_encoded.iloc[len(X):, :]



auc_scores_fe = []
test_preds_fe = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(cv.split(X_fe_encoded, y)):
    X_tr, X_val = X_fe_encoded.iloc[train_idx], X_fe_encoded.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_tr, y_tr)
    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    auc_scores_fe.append(auc)

    test_preds_fe += model.predict_proba(X_test_fe_encoded)[:, 1] / cv.n_splits
    print(f"[FE] Fold {fold + 1} AUC: {auc:.5f}")

print(f"\n[FE] Mean AUC: {np.mean(auc_scores_fe):.5f}")



from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

# Identify categorical features (by name or index)
cat_features = [
    'job', 'marital', 'education', 'default',
    'housing', 'loan', 'contact', 'month', 'poutcome'
]

# Setup
X = train.drop(columns=['id', 'y'])  # keep all features incl. engineered
y = train['y']
test_X = test.drop(columns=['id'])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"\nğŸ”� Fold {fold+1}")
    
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    
    # Create Pool objects
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        eval_metric='AUC',
        random_seed=42,
        early_stopping_rounds=100,
        verbose=100,
        task_type='CPU'  # or 'GPU' if using GPU
    )

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    
    # Validation predictions
    val_preds = model.predict_proba(X_valid)[:, 1]
    oof_preds[valid_idx] = val_preds
    auc = roc_auc_score(y_valid, val_preds)
    print(f"[CATBOOST] Fold {fold+1} AUC: {auc:.5f}")
    scores.append(auc)

    # Test predictions
    test_pool = Pool(test_X, cat_features=cat_features)
    test_preds += model.predict_proba(test_pool)[:, 1] / skf.n_splits

np.save('/kaggle/working/cat_preds.npy', test_preds) 

# Final score
print(f"\nğŸ“Š [CATBOOST] CV AUC Mean: {np.mean(scores):.5f} | Std: {np.std(scores):.5f}")



import numpy as np
import pandas as pd

# === Load Predictions ===
# Replace these with your actual prediction file paths or variables if already in memory
lgb_preds = np.load('/kaggle/working/lgb_preds.npy')       # shape: (250000,)
cat_preds = np.load('/kaggle/working/cat_preds.npy')       # shape: (250000,)

# === Sanity Check ===
assert lgb_preds.shape == cat_preds.shape == (250000,), "Prediction shapes do not match!"

# === Blend Predictions ===
# Option 1: Simple average
blended_preds = 0.5 * lgb_preds + 0.5 * cat_preds

# Optional: Try different weights
# blended_preds = 0.55 * lgb_preds + 0.45 * cat_preds

# === Correlation Check (optional) ===
correlation = np.corrcoef(lgb_preds, cat_preds)[0, 1]
print(f"Correlation between LightGBM and CatBoost predictions: {correlation:.5f}")

# === Create Submission File ===
submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
submission["y"] = blended_preds
submission.to_csv("submission_blend_lgb_cat.csv", index=False)
print("âœ… Blended submission saved as 'submission_blend_lgb_cat.csv'")



import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib

# === Load raw data ===
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

# === Combine for consistent preprocessing ===
test["y"] = -1  # Dummy target
df = pd.concat([train, test], axis=0, ignore_index=True)

# === Drop 'duration' if it's there ===
if "duration" in df.columns:
    df = df.drop(columns=["duration"])

# === Categorical encoding ===
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
df = pd.get_dummies(df, columns=cat_cols)

# === Split back ===
train_fe = df[df["y"] != -1].copy()
test_fe = df[df["y"] == -1].copy()

X = train_fe.drop(columns=["id", "y"])
y = train_fe["y"].astype(int)
X_test = test_fe.drop(columns=["id", "y"])

# === CV Setup ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(X.shape[0])
test_preds = np.zeros(X_test.shape[0])

# === Model config ===
xgb_params = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
    "use_label_encoder": False
}

# === Train XGBoost ===
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"[XGB] Training fold {fold + 1}")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = XGBClassifier(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=100
    )

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

    joblib.dump(model, f"xgb_fold{fold+1}.pkl")

# === AUC ===
auc = roc_auc_score(y, oof_preds)
print(f"[XGB] OOF AUC: {auc:.5f}")

# === Save predictions ===
np.save("oof_xgb.npy", oof_preds)
np.save("preds_test_xgb.npy", test_preds)

# === Create submission ===
submission = sample_submission.copy()
submission["y"] = test_preds
submission.to_csv("submission_xgb.csv", index=False)


