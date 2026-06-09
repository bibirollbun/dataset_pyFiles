import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore")


# ===============================
# Load Data
# ===============================
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")

X = train.drop(["BeatsPerMinute", "id"], axis=1)
y = train["BeatsPerMinute"]
X_test = test.drop(["id"], axis=1)

# ===============================
# Light Feature Engineering
# ===============================
def create_features(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    # Simple polynomials
    for col in num_cols:
        df[f"{col}_squared"] = df[col] ** 2
        df[f"{col}_sqrt"] = np.sqrt(np.abs(df[col]))
    
    # Row stats
    df["row_sum"] = df[num_cols].sum(axis=1)
    df["row_mean"] = df[num_cols].mean(axis=1)
    df["row_std"] = df[num_cols].std(axis=1)
    return df

X = create_features(X)
X_test = create_features(X_test)

# ===============================
# Stacking Setup
# ===============================
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros((len(X), 4))   # 4 base models
test_preds = np.zeros((len(X_test), 4))


# ===============================
# Base Models
# ===============================
# LightGBM
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.01,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42,
}

print("Training LightGBM...")
for fold, (trn_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, y_tr = X.iloc[trn_idx], y.iloc[trn_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**lgb_params, n_estimators=5000)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=100)
        ]
    )
    
    oof_preds[val_idx, 0] = model.predict(X_val)
    test_preds[:, 0] += model.predict(X_test) / N_FOLDS

# XGBoost
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.01,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

print("Training XGBoost...")
for fold, (trn_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, y_tr = X.iloc[trn_idx], y.iloc[trn_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = xgb.XGBRegressor(**xgb_params, n_estimators=5000, tree_method="hist")
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=200,
        verbose=False
    )
    
    oof_preds[val_idx, 1] = model.predict(X_val)
    test_preds[:, 1] += model.predict(X_test) / N_FOLDS

# CatBoost
cb_params = {
    "iterations": 5000,
    "learning_rate": 0.01,
    "depth": 8,
    "l2_leaf_reg": 3,
    "loss_function": "RMSE",
    "random_seed": 42,
    "early_stopping_rounds": 200,
    "verbose": False,
}

print("Training CatBoost...")
for fold, (trn_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, y_tr = X.iloc[trn_idx], y.iloc[trn_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = cb.CatBoostRegressor(**cb_params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
    
    oof_preds[val_idx, 2] = model.predict(X_val)
    test_preds[:, 2] += model.predict(X_test) / N_FOLDS



# Neural Network (MLP)
print("Training MLP...")
for fold, (trn_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, y_tr = X.iloc[trn_idx], y.iloc[trn_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                         activation="relu",
                         solver="adam",
                         alpha=0.001,
                         learning_rate_init=0.001,
                         max_iter=500,
                         early_stopping=True,
                         random_state=42)
    model.fit(X_tr, y_tr)
    
    oof_preds[val_idx, 3] = model.predict(X_val)
    test_preds[:, 3] += model.predict(X_test) / N_FOLDS

# ===============================
# Meta Model (Bayesian Ridge)
# ===============================
meta = BayesianRidge()
meta.fit(oof_preds, y)

final_oof = meta.predict(oof_preds)
final_test = meta.predict(test_preds)

cv_score = mean_squared_error(y, final_oof, squared=False)
print(f"OOF CV RMSE (stacked): {cv_score:.5f}")

# ===============================
# Post-processing
# ===============================
def smooth_predictions(preds, window_size=5):
    return np.convolve(preds, np.ones(window_size)/window_size, mode="same")

final_test_smoothed = smooth_predictions(final_test)
final_test_blended = 0.9 * final_test + 0.1 * final_test_smoothed
final_test_blended = np.clip(final_test_blended, 60, 200)

# ===============================
# Save Submission
# ===============================
sample_sub["BeatsPerMinute"] = final_test_blended
sample_sub.to_csv("submission.csv", index=False)
print("Submission saved!")

# ===============================
# Feature Importance (LightGBM)
# ===============================
lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=500, random_state=42)
lgb_model.fit(X, y)
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": lgb_model.feature_importances_
}).sort_values("importance", ascending=False)

print("Top 10 most important features:")
print(importance_df.head(10))


# ===============================
# Save Submission
# ===============================
sample_sub["BeatsPerMinute"] = final_test_blended
sample_sub.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv!")

# ===============================
# Feature Importance (LightGBM)
# ===============================
lgb_model = lgb.LGBMRegressor(**lgb_params, n_estimators=500, random_state=42)
lgb_model.fit(X, y)
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": lgb_model.feature_importances_
}).sort_values("importance", ascending=False)

print("Top 10 most important features:")
print(importance_df.head(10))

