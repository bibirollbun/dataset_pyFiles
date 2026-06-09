# =========================
# Kaggle Playground S5E10: Accident Risk Prediction
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import os, gc, time, warnings
warnings.filterwarnings('ignore')



# =========================
# 1. Load Data
# =========================
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

print(train.shape, test.shape)
print(train.head())


# =========================
# 2. Basic EDA
# =========================
print(train.info())
print(train.describe())

# Target distribution
sns.histplot(train['accident_risk'], bins=50, kde=True)
plt.show()



# =========================
# 3. Preprocessing
# =========================
target = "accident_risk"
features = [col for col in train.columns if col not in ["id", target]]

X = train[features].copy()
y = train[target].copy()
X_test = test[features].copy()

# Encode categorical variables
cat_cols = X.select_dtypes(include="object").columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))


# =========================
# 4. Cross Validation Setup
# =========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb, oof_xgb, oof_cb = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
preds_lgb, preds_xgb, preds_cb = np.zeros(len(X_test)), np.zeros(len(X_test)), np.zeros(len(X_test))



# =========================
# 5. Train Models
# =========================
from lightgbm import early_stopping, log_evaluation

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n===== Fold {fold+1} =====")
    X_tr, y_tr = X.iloc[trn_idx], y.iloc[trn_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # -----------------
    # LightGBM
    # -----------------
    lgb_model = lgb.LGBMRegressor(
        n_estimators=5000, learning_rate=0.03,
        max_depth=-1, subsample=0.8, colsample_bytree=0.8,
        random_state=42
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[early_stopping(200), log_evaluation(500)]
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    preds_lgb += lgb_model.predict(X_test) / kf.n_splits
    
    # -----------------
    # XGBoost
    # -----------------
    xgb_model = xgb.XGBRegressor(
        n_estimators=5000, learning_rate=0.03,
        max_depth=8, subsample=0.8, colsample_bytree=0.8,
        random_state=42, tree_method="hist"
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=200,
        verbose=500
    )
    oof_xgb[val_idx] = xgb_model.predict(X_val)
    preds_xgb += xgb_model.predict(X_test) / kf.n_splits
    
    # -----------------
    # CatBoost
    # -----------------
    cb_model = cb.CatBoostRegressor(
        iterations=5000, learning_rate=0.03, depth=8,
        random_seed=42, loss_function="RMSE", verbose=500
    )
    cb_model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=200
    )
    oof_cb[val_idx] = cb_model.predict(X_val)
    preds_cb += cb_model.predict(X_test) / kf.n_splits


# =========================
# 6. Blend Models with Meta Learner (Stacking)
# =========================
from sklearn.linear_model import Ridge

# Collect OOF predictions as new features
stack_X = np.vstack([oof_lgb, oof_xgb, oof_cb]).T
stack_test = np.vstack([preds_lgb, preds_xgb, preds_cb]).T

# Ridge as meta-learner
meta_model = Ridge(alpha=1.0)
meta_model.fit(stack_X, y)

oof_blend = meta_model.predict(stack_X)
preds_blend = meta_model.predict(stack_test)

rmse = np.sqrt(mean_squared_error(y, oof_blend))
print("\nOOF RMSE (Stacked Ensemble):", rmse)


# =========================
# 7. Submission
# =========================
submission = sample.copy()
submission["accident_risk"] = preds_blend
submission.to_csv("submission.csv", index=False)
print("Submission file saved!")

