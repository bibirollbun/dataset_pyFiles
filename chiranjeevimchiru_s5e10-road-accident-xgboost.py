
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb

# -------- Load data --------
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
original = pd.read_csv("/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

print("train:", train.shape, "test:", test.shape, "original:", original.shape)
display(train.head())



# -------- Quick EDA --------
plt.figure(figsize=(8,4))
sns.histplot(train["accident_risk"], kde=True, bins=80)
plt.title("Target distribution: accident_risk")
plt.show()

# correlation among numeric features
num_cols = ["num_lanes", "curvature", "speed_limit", "num_reported_accidents", "accident_risk"]
plt.figure(figsize=(8,6))
sns.heatmap(train[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Numeric feature correlations")
plt.show()

# scatter target vs some numeric features
for f in ["curvature", "speed_limit", "num_reported_accidents"]:
    plt.figure(figsize=(7,3.5))
    sns.scatterplot(x=train[f], y=train["accident_risk"], alpha=0.15)
    plt.title(f"{f} vs accident_risk")
    plt.show()



# -------- Preprocessing --------

# Define features
target = "accident_risk"
id_col = "id"
features = [c for c in train.columns if c not in [id_col, target]]
print("features:", features)

# We'll map bools to ints and encode categoricals with OrdinalEncoder for model consistency.
bool_cols = [c for c in features if train[c].dtype == "bool"]
cat_cols = [c for c in features if train[c].dtype == "object"]
num_cols = [c for c in features if c not in bool_cols + cat_cols]

print("bool_cols:", bool_cols)
print("cat_cols:", cat_cols)
print("num_cols:", num_cols)

# Convert bool to int
for c in bool_cols:
    train[c] = train[c].astype(int)
    test[c] = test[c].astype(int)
    original[c] = original[c].astype(int)


# Optional: augment training data with the provided original synthetic dataset
# This can help if distributions match. If you prefer NOT to augment, comment the concat line.
augment = False   # <--- set to False if you don't want augmentation
if augment:
    # Keep only columns matching train; some datasets may differ in column order - align
    original_trim = original[train.columns].copy()
    print("original sample head:")
    display(original_trim.head())
    train_aug = pd.concat([train, original_trim], axis=0).reset_index(drop=True)
    print("After augmentation, train_aug shape:", train_aug.shape)
else:
    train_aug = train.copy()
    print("Augmentation disabled - using original train only.")

# Recompute features sets if needed
X = train_aug[features].copy()
y = train_aug[target].copy()
X_test = test[features].copy()

# Imputer for numeric missing (if any) and ordinal encoding for categoricals
num_imputer = SimpleImputer(strategy="median")
X[num_cols] = num_imputer.fit_transform(X[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

# Ordinal encode categorical columns (works well for tree ensembles)
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
if len(cat_cols) > 0:
    X[cat_cols] = oe.fit_transform(X[cat_cols])
    X_test[cat_cols] = oe.transform(X_test[cat_cols])

# Scale numeric features (not required for tree models but helpful for stacking/consistency)
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Convert to numpy arrays for the sklearn-like fitting flow
X_arr = X.values
X_test_arr = X_test.values
y_arr = y.values

print("Prepared shapes:", X_arr.shape, X_test_arr.shape, y_arr.shape)


# -------- Models (base models / hyperparams) --------
cat_model = CatBoostRegressor(
    iterations=3000,
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
    n_estimators=3000,
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
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=1.0,
    random_state=42
)

# -------- Cross-validation Ensemble --------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

n = X_arr.shape[0]
cat_oof = np.zeros(n); lgb_oof = np.zeros(n); xgb_oof = np.zeros(n)
cat_preds = np.zeros(X_test_arr.shape[0]); lgb_preds = np.zeros(X_test_arr.shape[0]); xgb_preds = np.zeros(X_test_arr.shape[0])

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_arr)):
    print(f"\n===== Fold {fold+1} =====")
    X_train, X_valid = X_arr[train_idx], X_arr[valid_idx]
    y_train, y_valid = y_arr[train_idx], y_arr[valid_idx]

    # ---- CatBoost ----
    cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True, verbose=200)
    cat_oof[valid_idx] = cat_model.predict(X_valid)
    cat_preds += cat_model.predict(X_test_arr) / kf.n_splits

    # ---- LightGBM ----
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=0)
        ]
    )
    lgb_oof[valid_idx] = lgb_model.predict(X_valid)
    lgb_preds += lgb_model.predict(X_test_arr) / kf.n_splits

# -------- Evaluate Out-of-Fold Performance --------
cat_rmse = mean_squared_error(y_arr, cat_oof, squared=False)
lgb_rmse = mean_squared_error(y_arr, lgb_oof, squared=False)
xgb_rmse = mean_squared_error(y_arr, xgb_oof, squared=False)
ensemble_oof = 0.4*cat_oof + 0.3*lgb_oof + 0.3*xgb_oof
ens_rmse = mean_squared_error(y_arr, ensemble_oof, squared=False)

print("\nOOF RMSEs:")
print(f"CatBoost RMSE:  {cat_rmse:.6f}")
print(f"LightGBM RMSE:  {lgb_rmse:.6f}")
print(f"XGBoost RMSE:   {xgb_rmse:.6f}")
print(f"Ensemble RMSE:  {ens_rmse:.6f}")


# -------- Simple blending of test predictions --------
final_preds = 0.4*cat_preds + 0.3*lgb_preds + 0.3*xgb_preds

# If original target scale differs or needs clipping, consider constraints:
# e.g. accident_risk presumably between 0 and 1? If so, clip:
if (train[target].min() >= 0) and (train[target].max() <= 1):
    final_preds = np.clip(final_preds, 0, 1)

# -------- Model comparison plots --------
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
rmse_scores = {"CatBoost":cat_rmse, "LightGBM":lgb_rmse, "XGBoost":xgb_rmse, "Ensemble":ens_rmse}
sns.barplot(x=list(rmse_scores.keys()), y=list(rmse_scores.values()), palette="Set2")
plt.ylabel("RMSE")
plt.title("Model RMSE Comparison")
for i, v in enumerate(rmse_scores.values()):
    plt.text(i, v + 1e-3, f"{v:.6f}", ha="center", fontweight="bold")
plt.show()

# True vs predicted scatter for OOF ensemble
plt.figure(figsize=(6,6))
plt.scatter(y_arr, ensemble_oof, alpha=0.2, s=8)
xmin, xmax = y_arr.min(), y_arr.max()
plt.plot([xmin, xmax], [xmin, xmax], 'r--')
plt.xlabel("True accident_risk")
plt.ylabel("Ensemble OOF Pred")
plt.title("True vs Ensemble OOF Pred")
plt.show()




# -------- Feature importances (LightGBM example) --------
try:
    imp_df = pd.DataFrame({
        "feature": features,
        "importance": lgb_model.feature_importances_
    }).sort_values("importance", ascending=False)
    display(imp_df.head(20))
    plt.figure(figsize=(8,6))
    sns.barplot(x="importance", y="feature", data=imp_df.head(20))
    plt.title("LightGBM feature importances (last fold)")
    plt.show()
except Exception as e:
    print("Could not compute feature importances:", e)



# -------- Submission --------
submission = pd.DataFrame({
    "id": test[id_col],
    "accident_risk": final_preds
})

submission.to_csv("submission.csv", index=False)
display(submission.head())


