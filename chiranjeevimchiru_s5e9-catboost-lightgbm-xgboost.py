import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb



train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")

print(train.shape, test.shape)
print(train.head())


# Check target distribution
plt.figure(figsize=(8,4))
sns.histplot(train["BeatsPerMinute"], kde=True, bins=50)
plt.title("Distribution of BeatsPerMinute")
plt.show()


# Correlation heatmap
plt.figure(figsize=(10,8))
corr = train.drop(columns=["id"]).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap")
plt.show()


# scatterplot for selected features vs target
features_to_plot = ["RhythmScore", "AudioLoudness", "Energy"]
for f in features_to_plot:
    plt.figure(figsize=(8,4))
    sns.scatterplot(x=train[f], y=train["BeatsPerMinute"], alpha=0.3)
    plt.title(f"{f} vs BeatsPerMinute")
    plt.show()


X = train.drop(columns=["BeatsPerMinute", "id"])
y = train["BeatsPerMinute"]
X_test = test.drop(columns=["id"])

# Scale features (not necessary for trees, but good for ensemble consistency)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


cat_model = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.03,
    depth=10,
    eval_metric="RMSE",
    random_seed=42,
    od_type="Iter",
    od_wait=100,
    verbose=500
)


lgb_model = lgb.LGBMRegressor(
    objective="regression",
    metric="rmse",
    boosting_type="gbdt",
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=256,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l1=1.0,
    lambda_l2=1.0,
    random_state=42
)

xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=5000,
    learning_rate=0.03,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=1.0,
    random_state=42
)




kf = KFold(n_splits=5, shuffle=True, random_state=42)

cat_oof, lgb_oof, xgb_oof = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
cat_preds, lgb_preds, xgb_preds = np.zeros(len(X_test)), np.zeros(len(X_test)), np.zeros(len(X_test))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_scaled)):
    print(f"\n===== Fold {fold+1} =====")
    
    X_train, X_valid = X_scaled[train_idx], X_scaled[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
    # CatBoost
    cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
    cat_oof[valid_idx] = cat_model.predict(X_valid)
    cat_preds += cat_model.predict(X_test_scaled) / kf.n_splits
    
    # LightGBM
    lgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                  eval_metric="rmse", callbacks=[lgb.early_stopping(100, verbose=False)])
    lgb_oof[valid_idx] = lgb_model.predict(X_valid)
    lgb_preds += lgb_model.predict(X_test_scaled) / kf.n_splits
    
    # XGBoost
    xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                  eval_metric="rmse", early_stopping_rounds=100, verbose=False)
    xgb_oof[valid_idx] = xgb_model.predict(X_valid)
    xgb_preds += xgb_model.predict(X_test_scaled) / kf.n_splits


cat_rmse = mean_squared_error(y, cat_oof, squared=False)
lgb_rmse = mean_squared_error(y, lgb_oof, squared=False)
xgb_rmse = mean_squared_error(y, xgb_oof, squared=False)

print(f"CatBoost RMSE: {cat_rmse:.5f}")
print(f"LightGBM RMSE: {lgb_rmse:.5f}")
print(f"XGBoost RMSE: {xgb_rmse:.5f}")


# Weighted average of models (you can tune weights)

final_preds = (0.4 * cat_preds) + (0.3 * lgb_preds) + (0.3 * xgb_preds)


import matplotlib.pyplot as plt
import seaborn as sns

# RMSE values dictionary
rmse_scores = {
    "CatBoost": cat_rmse,
    "LightGBM": lgb_rmse,
    "XGBoost": xgb_rmse,
    "Ensemble": mean_squared_error(y, 
                   0.4*cat_oof + 0.3*lgb_oof + 0.3*xgb_oof, squared=False)
}

# --- Barplot of RMSE ---
plt.figure(figsize=(8,5))
sns.barplot(x=list(rmse_scores.keys()), y=list(rmse_scores.values()), palette="Set2")
plt.ylabel("RMSE")
plt.title("Model RMSE Comparison")
for i, v in enumerate(rmse_scores.values()):
    plt.text(i, v+0.1, f"{v:.4f}", ha="center", fontweight="bold")
plt.show()

# --- Scatter plots: True vs Predicted ---
models_oof = {
    "CatBoost": cat_oof,
    "LightGBM": lgb_oof,
    "XGBoost": xgb_oof,
    "Ensemble": 0.4*cat_oof + 0.3*lgb_oof + 0.3*xgb_oof
}

fig, axes = plt.subplots(2, 2, figsize=(12,10))
axes = axes.flatten()

for i, (name, preds) in enumerate(models_oof.items()):
    axes[i].scatter(y, preds, alpha=0.3, label=name, s=10)
    axes[i].plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)  # diagonal line
    axes[i].set_title(f"{name}: True vs Predicted")
    axes[i].set_xlabel("True BPM")
    axes[i].set_ylabel("Predicted BPM")
    axes[i].legend()

plt.tight_layout()
plt.show()



submission = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": final_preds
})

submission.to_csv("submission.csv", index=False)

print("✅ Submission file created with Ensemble Predictions!")

