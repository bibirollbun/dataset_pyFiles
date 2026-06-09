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


# =========================
# 1. Imports
# =========================
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# =========================
# 2. Load data
# =========================
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# =========================
# 3. Feature Engineering
# =========================
target = "BeatsPerMinute"
X = train.drop(columns=[target])
y = train[target]
X_test = test.copy()

# Log-transform TrackDurationMs
X["log_duration"] = np.log1p(X["TrackDurationMs"])
X_test["log_duration"] = np.log1p(X_test["TrackDurationMs"])

# =========================
# 4. Cross-Validation Setup
# =========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

preds_lgb = np.zeros(len(X_test))
preds_cat = np.zeros(len(X_test))
preds_xgb = np.zeros(len(X_test))

oof_preds = np.zeros(len(X))
oof_rmse = []

# =========================
# 5. Training Loop (CV)
# =========================
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n===== Fold {fold+1} =====")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # --- LightGBM ---
    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[early_stopping(50), log_evaluation(200)]
    )
    val_pred = lgb_model.predict(X_val)
    oof_preds[val_idx] = val_pred
    preds_lgb += lgb_model.predict(X_test) / kf.n_splits

    # --- CatBoost ---
    cat_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        random_seed=42,
        verbose=0
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
    preds_cat += cat_model.predict(X_test) / kf.n_splits

    # --- XGBoost ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric="rmse",
                  early_stopping_rounds=50,
                  verbose=100)
    preds_xgb += xgb_model.predict(X_test) / kf.n_splits

    # Fold RMSE
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    oof_rmse.append(rmse)
    print(f"Fold {fold+1} RMSE: {rmse:.5f}")

print("\n===== CV Results =====")
print("OOF RMSE:", np.sqrt(mean_squared_error(y, oof_preds)))
print("Fold-wise RMSE:", oof_rmse)

# =========================
# 6. Final Ensemble
# =========================
final_preds = (preds_lgb + preds_cat + preds_xgb) / 3

# =========================
# 7. Submission
# =========================
submission = sample.copy()
submission["BeatsPerMinute"] = final_preds
submission.to_csv("submission.csv", index=False)

print("✅ submission.csv saved!", submission.shape)

# =========================
# 8. (Optional) Feature Importance from LightGBM
# =========================
lgb.plot_importance(lgb_model, max_num_features=15, importance_type="gain")
plt.title("Top 15 Feature Importances (Last LGBM Fold)")
plt.show()


