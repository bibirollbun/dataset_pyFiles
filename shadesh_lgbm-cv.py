# =====================================================
# 1. Imports
# =====================================================
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import early_stopping, log_evaluation

import warnings
warnings.filterwarnings("ignore")

# =====================================================
# 2. Load Data
# =====================================================
TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e12/test.csv"

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape)
print("Test shape :", test.shape)

# =====================================================
# 3. Prepare Features
# =====================================================
TARGET = "diagnosed_diabetes"
ID_COL = "id"

X = train.drop(columns=[TARGET, ID_COL])
y = train[TARGET]

X_test = test.drop(columns=[ID_COL])

# -----------------------------------------------------
# ğŸ”‘ FIX: Convert object columns to category
# -----------------------------------------------------
cat_cols = X.select_dtypes(include="object").columns.tolist()
print("Categorical columns:", cat_cols)

for col in cat_cols:
    X[col] = X[col].astype("category")
    X_test[col] = X_test[col].astype("category")

# =====================================================
# 4. Cross-Validation Setup
# =====================================================
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

# =====================================================
# 5. Train LightGBM (Correct Categorical Handling)
# =====================================================
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nğŸ”¹ Fold {fold + 1}/{N_SPLITS}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=64,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        metric="auc",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(200)
        ]
    )

    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds

    fold_auc = roc_auc_score(y_val, val_preds)
    print(f"Fold AUC: {fold_auc:.5f}")

    test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS

# =====================================================
# 6. Overall CV Score
# =====================================================
cv_auc = roc_auc_score(y, oof_preds)
print("\nâœ… Overall CV ROC-AUC:", round(cv_auc, 6))

# =====================================================
# 7. Create Submission
# =====================================================
submission = pd.DataFrame({
    "id": test[ID_COL],
    "diagnosed_diabetes": test_preds
})

submission.to_csv("submission.csv", index=False)

print("\nğŸ“� submission.csv saved successfully")
submission.head()


