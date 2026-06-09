# --- Import Libraries ---
import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


# --- Load Data ---
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


# --- Feature Engineering ---
def add_feature_cross_terms(df, features):
    df_new = df.copy()
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]
            df_new[f"{f1}_x_{f2}"] = df_new[f1] * df_new[f2]
    return df_new

num_features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
train = add_feature_cross_terms(train, num_features)
test = add_feature_cross_terms(test, num_features)


# Encode categorical
train['Sex'] = train['Sex'].map({'male': 1, 'female': 0}).astype('category')
test['Sex'] = test['Sex'].map({'male': 1, 'female': 0}).astype('category')


# Define target and features
X = train.drop(columns=["id", "Calories"])
y = np.log1p(train["Calories"])
X_test = test.drop(columns=["id"])


# --- KFold Setup ---
FOLDS = 50
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)


# --- OOF & Prediction Containers ---
oof_cb = np.zeros(len(train))
oof_xgb = np.zeros(len(train))
pred_cb = np.zeros(len(test))
pred_xgb = np.zeros(len(test))


# --- CatBoost Training ---
for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"CatBoost Fold {fold+1}")
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model_cb = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.02,
        depth=10,
        l2_leaf_reg=3,
        loss_function='RMSE',
        eval_metric='RMSE',
        early_stopping_rounds=100,
        verbose=0,
        random_state=42,
        task_type="GPU" if os.environ.get("CUDA_VISIBLE_DEVICES") else "CPU",
        cat_features=[X.columns.get_loc("Sex")]
    )
    model_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    oof_cb[val_idx] = model_cb.predict(X_val)
    pred_cb += model_cb.predict(X_test)


# --- XGBoost Training ---
for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"XGBoost Fold {fold+1}")
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model_xgb = XGBRegressor(
        max_depth=10,
        colsample_bytree=0.75,
        subsample=0.9,
        n_estimators=2000,
        learning_rate=0.02,
        gamma=0.01,
        max_delta_step=2,
        early_stopping_rounds=100,
        eval_metric="rmse",
        enable_categorical=True,
        tree_method="gpu_hist" if os.environ.get("CUDA_VISIBLE_DEVICES") else "hist"
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    pred_xgb += model_xgb.predict(X_test)


# Average predictions
pred_cb /= FOLDS
pred_xgb /= FOLDS


# Ensemble
final_pred_log = 0.3 * pred_cb + 0.7 * pred_xgb
final_preds = np.expm1(final_pred_log)
final_preds_clipped = np.clip(final_preds, 1, 314)


# Save submission
submission["Calories"] = final_preds_clipped
submission.to_csv("ensemble_submission.csv", index=False)

print("Final submission saved as ensemble_submission.csv")

