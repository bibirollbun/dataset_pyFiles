import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv(
    "/kaggle/input/playground-series-s5e12/train.csv",
    engine="python"
)

test = pd.read_csv(
    "/kaggle/input/playground-series-s5e12/test.csv",
    engine="python"
)


TARGET = "diagnosed_diabetes"
ID_COL = "id"

X = train.drop(columns=[TARGET, ID_COL])
y = train[TARGET]
X_test = test.drop(columns=[ID_COL])
test_id = test[ID_COL]



cat_cols = [
    'gender',
    'ethnicity',
    'education_level',
    'income_level',
    'smoking_status',
    'employment_status',
    'family_history_diabetes',
    'hypertension_history',
    'cardiovascular_history'
]



cat_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'iterations': 8000,
    'learning_rate': 0.03,
    'depth': 7,
    'l2_leaf_reg': 5,
    'random_strength': 1,
    'bagging_temperature': 0.8,
    'od_type': 'Iter',
    'od_wait': 300,
    'random_seed': 42,
    'verbose': 0
}



N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    val_pool = Pool(X_val, y_val, cat_features=cat_cols)

    model = CatBoostClassifier(**cat_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred

    auc = roc_auc_score(y_val, val_pred)
    fold_scores.append(auc)
    print(f"AUC: {auc:.5f}")

    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS



cv_auc = roc_auc_score(y, oof_preds)
print(f"\nFinal CV AUC: {cv_auc:.6f}")



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y, oof_preds)
plt.plot(fpr, tpr, label=f'CatBoost CV (AUC = {cv_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--') # Random chance line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


print(f"Check 1 - Array Max: {test_preds.max()}") 
print(f"Check 2 - Array Sample: {test_preds[:5]}")

submission = pd.DataFrame({
    "id": test_id.values, # Use .values to avoid index alignment issues
    "diagnosed_diabetes": test_preds.astype(float) 
})

submission["diagnosed_diabetes"] = np.clip(submission["diagnosed_diabetes"], 1e-5, 1-1e-5)

submission.to_csv("submission.csv", index=False)
print(submission.head())


import os

if os.path.exists("submission.csv"):
    print("✅ SUCCESS: submission.csv found!")
    # Check if it has the right number of rows
    sub_check = pd.read_csv("submission.csv")
    print(f"✅ Row count: {len(sub_check)}")
    print(sub_check.head())
else:
    print("❌ ERROR: submission.csv NOT found in the working directory.")

