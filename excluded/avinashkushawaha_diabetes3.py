import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb



train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

print(train.shape, test.shape, sample.shape)



TARGET = "diagnosed_diabetes"

X = train.drop(columns=["id", TARGET])
y = train[TARGET]

X_test = test.drop(columns=["id"])

print(X.shape, y.shape, X_test.shape)



all_data = pd.concat([X, X_test], axis=0)

all_data_encoded = pd.get_dummies(all_data, drop_first=True)

X_encoded = all_data_encoded.iloc[:len(X)]
X_test_encoded = all_data_encoded.iloc[len(X):]

print(X_encoded.shape, X_test_encoded.shape)
print("Columns match:", X_encoded.columns.equals(X_test_encoded.columns))



# LightGBM Cross-Validation (FIXED)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgb_auc_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_encoded, y)):
    
    X_train, X_val = X_encoded.iloc[tr_idx], X_encoded.iloc[val_idx]
    y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    # ðŸš« removed verbose argument
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc"
    )
    
    val_preds = lgb_model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    
    lgb_auc_scores.append(auc)
    print(f"Fold {fold+1} AUC: {auc:.5f}")

print("\nMean LGBM CV AUC:", np.mean(lgb_auc_scores))



# FINAL LightGBM model trained on full data

final_lgb = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

final_lgb.fit(X_encoded, y)

# Predict on test set
lgb_test_preds = final_lgb.predict_proba(X_test_encoded)[:, 1]

# Create submission
lgb_submission = sample.copy()
lgb_submission["diagnosed_diabetes"] = lgb_test_preds

# Save CSV (SUBMIT THIS FILE)
lgb_submission.to_csv("lgbm_submission.csv", index=False)

# Final verification
print(lgb_submission.shape)
print(lgb_submission.columns)
print(
    lgb_submission["diagnosed_diabetes"].min(),
    lgb_submission["diagnosed_diabetes"].max()
)

lgb_submission.head()





