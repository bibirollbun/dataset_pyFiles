# ===========================================
# PREDICTING ROAD ACCIDENT RISK - KAGGLE S5E10
# Author: Nishant Dubey
# ===========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# -----------------------
# Load Data
# -----------------------
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

print(train.shape, test.shape)
print(train.columns)

# -----------------------
# Basic preprocessing
# -----------------------
target = 'accident_risk'
y = train[target]
X = train.drop(columns=[target, 'id'])
test_ids = test['id']
X_test = test.drop(columns=['id'])

# Identify categorical columns
cat_cols = X.select_dtypes(include='object').columns.tolist()
bool_cols = X.select_dtypes(include='bool').columns.tolist()
num_cols = X.select_dtypes(include=np.number).columns.tolist()

# Encode categorical features
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])
    le_dict[col] = le

# Convert boolean to int
for col in bool_cols:
    X[col] = X[col].astype(int)
    X_test[col] = X_test[col].astype(int)

# -----------------------
# Model training
# -----------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "LGBM": LGBMRegressor(n_estimators=2000, learning_rate=0.02, random_state=42),
    "CatBoost": CatBoostRegressor(iterations=2000, learning_rate=0.02, verbose=0, random_seed=42),
    "XGB": XGBRegressor(n_estimators=2000, learning_rate=0.02, max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42)
}

preds = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        fold_rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(f"Fold {fold+1} RMSE: {fold_rmse:.5f}")

        oof[val_idx] = y_pred
        test_pred += model.predict(X_test) / kf.n_splits

    rmse = mean_squared_error(y, oof, squared=False)
    print(f"{name} CV RMSE: {rmse:.5f}")
    preds[name] = test_pred

# -----------------------
# Ensemble / Blending
# -----------------------
sub_preds = (
    0.4 * preds["LGBM"] +
    0.35 * preds["CatBoost"] +
    0.25 * preds["XGB"]
)
sub["accident_risk"] = np.clip(sub_preds, 0, 1)
sub.to_csv("submission.csv", index=False)

print("Submission file created successfully!")


