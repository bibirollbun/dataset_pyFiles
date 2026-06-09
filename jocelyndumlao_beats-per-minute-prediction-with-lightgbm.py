import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
import os  # Import the os module


import warnings
warnings.filterwarnings("ignore")


# Load the datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


train.head().style.background_gradient(cmap='plasma')


test.head().style.background_gradient(cmap='plasma')


X = train.drop(["id", "BeatsPerMinute"], axis=1)
y = train["BeatsPerMinute"]
X_test = test.drop("id", axis=1)


# Feature Engineering (polynomial interactions) 
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)
X_test_poly = poly.transform(X_test)

X_poly = pd.DataFrame(X_poly, columns=[f"feature_{i}" for i in range(X_poly.shape[1])])
X_test_poly = pd.DataFrame(X_test_poly, columns=[f"feature_{i}" for i in range(X_test_poly.shape[1])])



# KFold for Blending 
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_poly))
test_preds = np.zeros(len(X_test_poly))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_poly)):
    X_train, X_val = X_poly.iloc[train_idx], X_poly.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # --- Model 1: CatBoost ---
    cat_model = CatBoostRegressor(
        iterations=20000,
        learning_rate=0.01,
        depth=10,
        l2_leaf_reg=30,
        random_seed=42,
        loss_function="RMSE",
        verbose=False,
        early_stopping_rounds=500
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
    cat_val = cat_model.predict(X_val)
    cat_test = cat_model.predict(X_test_poly)

    # --- Model 2: LightGBM ---
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.005,
        'num_leaves': 256,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 2.0,
        'lambda_l2': 30.0,
        'verbose': -1,
        'seed': 42
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    lgb_model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=30000,
        callbacks=[lgb.early_stopping(500, verbose=False)]
    )
    lgb_val = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    lgb_test = lgb_model.predict(X_test_poly, num_iteration=lgb_model.best_iteration)

    # --- Model 3: XGBoost ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=20000,
        learning_rate=0.01,
        max_depth=10,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=30.0,
        random_state=42,
        objective="reg:squarederror",
        tree_method="hist",
    )
    xgb_model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False,
                  early_stopping_rounds=500)
    xgb_val = xgb_model.predict(X_val)
    xgb_test = xgb_model.predict(X_test_poly)

    # --- Blend Predictions (weighted average) ---
    val_pred = 0.4 * cat_val + 0.35 * lgb_val + 0.25 * xgb_val
    test_pred = 0.4 * cat_test + 0.35 * lgb_test + 0.25 * xgb_test

    oof_preds[val_idx] = val_pred
    test_preds += test_pred / kf.n_splits

    fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Fold {fold+1} RMSE: {fold_rmse:.5f}")



# Final CV Score 
cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Final CV RMSE: {cv_rmse:.5f}")


# Submission 
submission = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": test_preds
})
submission.to_csv("submission.csv", index=False)
print("Submission saved.")




