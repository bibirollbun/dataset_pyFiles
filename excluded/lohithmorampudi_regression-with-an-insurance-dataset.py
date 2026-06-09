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


# ================================================================
# ğŸš€ Insurance Premium Prediction â€“ RMSE Optimized (Ensemble Model)
# ================================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# ================================================================
# 1ï¸�âƒ£ Load Data
# ================================================================
train = pd.read_csv("/kaggle/input/playground-series-s4e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s4e12/test.csv")

print(f"âœ… Train shape: {train.shape}")
print(f"âœ… Test shape: {test.shape}")

TARGET = "Premium Amount"  # adjust if different in your dataset
y = train[TARGET]
X = train.drop(columns=[TARGET])

# Identify categorical/numeric features
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"ğŸ§  Categorical: {len(cat_cols)} | Numerical: {len(num_cols)}")

# ================================================================
# 2ï¸�âƒ£ Preprocessing
# ================================================================
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_cols] = enc.fit_transform(X[cat_cols].astype(str))
test[cat_cols] = enc.transform(test[cat_cols].astype(str))

imp = SimpleImputer(strategy='median')
X[num_cols] = imp.fit_transform(X[num_cols])
test[num_cols] = imp.transform(test[num_cols])

# ================================================================
# 3ï¸�âƒ£ Models
# ================================================================
models = {
    "LightGBM": LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),
    "CatBoost": CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=10,
        verbose=0,
        random_state=42
    )
}

# ================================================================
# 4ï¸�âƒ£ Cross-Validation + Ensemble
# ================================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nğŸ“‚ Fold {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    fold_preds = np.zeros(len(X_val))
    fold_test = np.zeros(len(test))

    for name, model in models.items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(test)

        fold_preds += val_pred / len(models)
        fold_test += test_pred / len(models)

    oof_preds[val_idx] = fold_preds
    test_preds += fold_test / kf.n_splits

rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\nğŸ�† Final Cross-Validated RMSE: {rmse:.4f}")

# ================================================================
# 5ï¸�âƒ£ Graph: Actual vs Predicted
# ================================================================
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y, y=oof_preds, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Perfect Fit')
plt.xlabel("Actual Premium Amount")
plt.ylabel("Predicted Premium Amount")
plt.title("Actual vs Predicted Premium Amount (Ensemble Model)")
plt.legend()
plt.show()

# ================================================================
# 6ï¸�âƒ£ Create Submission
# ================================================================
submission = pd.DataFrame({
    "id": test["id"],
    "Premium Amount": test_preds
})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("âœ… submission.csv created successfully!")


