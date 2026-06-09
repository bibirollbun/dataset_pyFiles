# Basic libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from lightgbm import early_stopping, log_evaluation
# Ignore warnings for clean output
import warnings
warnings.filterwarnings("ignore")



# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

train.head()


# Target variable analysis
plt.figure(figsize=(7, 5))
sns.histplot(train['accident_risk'], bins=40, kde=True, color="royalblue")
plt.title("Distribution of Accident Risk", fontsize=14, fontweight="bold")
plt.xlabel("Accident Risk")
plt.ylabel("Frequency")
plt.grid(alpha=0.2)
plt.show()

# Display key stats and skewness
print("ğŸ“ˆ Target Variable Summary:")
print(train['accident_risk'].describe())
print(f"\nSkewness: {train['accident_risk'].skew():.3f}")
print(f"Kurtosis: {train['accident_risk'].kurt():.3f}")

# Missing values overview
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) > 0:
    print("\nğŸš§ Columns with Missing Values:")
    display(missing.to_frame(name="Missing Count").head(15))
else:
    print("\nâœ… No missing values detected in the training dataset.")



# Define key columns
TARGET = "accident_risk"
ID = "id"

# Basic validation before splitting
assert TARGET in train.columns, f"â�Œ Target column '{TARGET}' not found in training data!"
assert ID in train.columns, f"â�Œ ID column '{ID}' not found in training data!"

# Split features and target
X = train.drop([ID, TARGET], axis=1)
y = train[TARGET]
X_test = test.drop(ID, axis=1)

# Quick validation for alignment
assert X.shape[1] == X_test.shape[1], "âš ï¸� Mismatch in feature count between train and test!"
assert all(X.columns == X_test.columns), "âš ï¸� Train/Test columns do not match!"

print("âœ… Data successfully separated.")
print(f"Features: {X.shape}, Target: {y.shape}")
print(f"Test set: {X_test.shape}")



# --- Identify categorical and numerical columns ---
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["number"]).columns.tolist()

# --- Quick sanity check ---
print(f"âœ… Total features: {X.shape[1]}")
print(f"ğŸ§© Categorical features: {len(cat_cols)}")
print(f"ğŸ”¢ Numerical features: {len(num_cols)}")

# --- Display examples for clarity ---
if cat_cols:
    print("\nğŸ“¦ Categorical columns:")
    print(cat_cols[:10], "..." if len(cat_cols) > 10 else "")
else:
    print("\nâš ï¸� No categorical columns detected.")

if num_cols:
    print("\nğŸ“Š Numerical columns:")
    print(num_cols[:10], "..." if len(num_cols) > 10 else "")
else:
    print("\nâš ï¸� No numerical columns detected.")



# --- Copy train/test data ---
X_encoded = X.copy()
X_test_encoded = X_test.copy()

# --- Handle Missing Values ---
for col in X_encoded.columns:
    if X_encoded[col].dtype == "object" or str(X_encoded[col].dtype) == "category":
        X_encoded[col] = X_encoded[col].fillna("missing")
        X_test_encoded[col] = X_test_encoded[col].fillna("missing")
    else:
        median_val = X_encoded[col].median()
        X_encoded[col] = X_encoded[col].fillna(median_val)
        X_test_encoded[col] = X_test_encoded[col].fillna(median_val)

# --- Label Encode Categorical Features ---
for col in cat_cols:
    le = LabelEncoder()
    combined_data = pd.concat([X_encoded[col], X_test_encoded[col]], axis=0).astype(str)
    le.fit(combined_data)  # Fit on both to avoid unseen label issues
    
    X_encoded[col] = le.transform(X_encoded[col].astype(str))
    X_test_encoded[col] = le.transform(X_test_encoded[col].astype(str))

# --- Scale Numerical Features ---
scaler = StandardScaler()
X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])
X_test_encoded[num_cols] = scaler.transform(X_test_encoded[num_cols])

print("âœ… Data Encoding, Imputation & Scaling Completed Successfully!")
print(f"ğŸ”¹ X_encoded shape: {X_encoded.shape}")
print(f"ğŸ”¹ X_test_encoded shape: {X_test_encoded.shape}")



# --- Random Forest Baseline ---
rf = RandomForestRegressor(
    n_estimators=500,      # slightly higher for smoother ensemble
    max_depth=None,        # allow full growth (trees decide naturally)
    min_samples_split=5,   # prevent overfitting small splits
    min_samples_leaf=2,    # avoid tiny leaf nodes
    max_features='sqrt',   # better generalization for tabular data
    random_state=42,
    n_jobs=-1
)

# --- Train-validation split ---
X_train, X_valid, y_train, y_valid = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# --- Model training ---
rf.fit(X_train, y_train)

# --- Predictions ---
y_pred = rf.predict(X_valid)

# --- Evaluate RMSE ---
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(f"ğŸŒ² Random Forest RMSE: {rmse:.5f}")

# --- Feature Importance Plot ---
feat_importances = pd.Series(rf.feature_importances_, index=X_encoded.columns)
top_feats = feat_importances.nlargest(20)

plt.figure(figsize=(8, 6))
sns.barplot(x=top_feats.values, y=top_feats.index, palette="viridis")
plt.title("Top 20 Feature Importances â€” Random Forest", fontsize=13)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()



# --- LightGBM with 10-Fold CV ---
kf = KFold(n_splits=10, shuffle=True, random_state=42)
lgb_preds = np.zeros(len(X_test_encoded))
lgb_oof = np.zeros(len(X_encoded))
rmse_scores = []

params = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "seed": 42,
    "learning_rate": 0.015,
    "num_leaves": 128,
    "max_depth": -1,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 1e-3,
    "lambda_l2": 1e-3,
    "min_child_weight": 15,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_encoded, y), 1):
    print(f"\nğŸ”¹ Fold {fold}")
    
    X_train, X_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=30000,
        callbacks=[
            early_stopping(stopping_rounds=400),
            log_evaluation(200)
        ],
    )

    # OOF and Test Predictions
    lgb_oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    lgb_preds += model.predict(X_test_encoded, num_iteration=model.best_iteration) / kf.n_splits

    # RMSE per fold
    fold_rmse = mean_squared_error(y_val, lgb_oof[val_idx], squared=False)
    rmse_scores.append(fold_rmse)
    print(f"âœ… Fold {fold} RMSE: {fold_rmse:.5f}")

# Overall CV Score
cv_rmse = np.mean(rmse_scores)
print(f"\nğŸ�� Overall CV RMSE: {cv_rmse:.5f}")



# --- XGBoost with 10-Fold CV ---
xgb_preds = np.zeros(len(X_test_encoded))
xgb_oof = np.zeros(len(X_encoded))

kf = KFold(n_splits=10, shuffle=True, random_state=42)
rmse_scores = []

xgb_params = {
    "objective": "reg:squarederror",
    "n_estimators": 30000,
    "learning_rate": 0.015,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "colsample_bylevel": 0.9,
    "colsample_bynode": 0.9,
    "reg_alpha": 1e-3,
    "reg_lambda": 1e-3,
    "min_child_weight": 10,
    "gamma": 0.1,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_encoded, y), 1):
    print(f"\nğŸ”¸ Fold {fold}")
    
    X_train, X_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBRegressor(**xgb_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=400,
        verbose=200
    )
    
    # Save predictions
    xgb_oof[val_idx] = model.predict(X_val, iteration_range=(0, model.best_iteration))
    xgb_preds += model.predict(X_test_encoded, iteration_range=(0, model.best_iteration)) / kf.n_splits

    # RMSE per fold
    fold_rmse = mean_squared_error(y_val, xgb_oof[val_idx], squared=False)
    rmse_scores.append(fold_rmse)
    print(f"âœ… Fold {fold} RMSE: {fold_rmse:.5f}")

# Overall CV RMSE
cv_rmse = np.mean(rmse_scores)
print(f"\nğŸ�� Overall XGBoost CV RMSE: {cv_rmse:.5f}")



# --- Ridge Stacking Ensemble ---

# Combine base model OOF predictions
stack_train = np.vstack([lgb_oof, xgb_oof]).T
stack_test = np.vstack([lgb_preds, xgb_preds]).T

# Meta-model (simple linear blend with L2 regularization)
meta_model = Ridge(alpha=1e-3, random_state=42)
meta_model.fit(stack_train, y)

# Predict final blended outputs
final_preds = meta_model.predict(stack_test)

# Evaluate stacking improvement
oof_meta = meta_model.predict(stack_train)
rmse_meta = mean_squared_error(y, oof_meta, squared=False)
print(f"âœ… Ridge Stacking Done | Meta RMSE: {rmse_meta:.5f}")

# Optional visualization for clarity
plt.figure(figsize=(6,4))
sns.scatterplot(x=y, y=oof_meta, alpha=0.6, edgecolor=None)
plt.xlabel("True Target")
plt.ylabel("Stacked Predictions")
plt.title("Ridge Stacking Fit vs True Values")
plt.show()



# --- Optimal Weighted Ensemble (LGB + XGB) ---

best_rmse = float("inf")
best_w = 0

# Search for best blending weight
for w in np.linspace(0, 1, 51):  # finer granularity (0.02 steps)
    blend = w * lgb_oof + (1 - w) * xgb_oof
    rmse = mean_squared_error(y, blend, squared=False)
    if rmse < best_rmse:
        best_rmse = rmse
        best_w = w

print(f"ğŸ”� Best weight for LightGBM: {best_w:.3f}")
print(f"ğŸ�† Best OOF RMSE after blending: {best_rmse:.5f}")

# Apply optimal weight on test predictions
final_preds = best_w * lgb_preds + (1 - best_w) * xgb_preds

# Visualize RMSE vs weight curve
weights = np.linspace(0, 1, 51)
rmses = [mean_squared_error(y, w*lgb_oof + (1-w)*xgb_oof, squared=False) for w in weights]

plt.figure(figsize=(7,5))
plt.plot(weights, rmses, marker="o", color="royalblue")
plt.title("RMSE vs Blending Weight (LGB share)")
plt.xlabel("LightGBM Weight")
plt.ylabel("OOF RMSE")
plt.grid(True, alpha=0.3)
plt.show()



# --- Generate Final Submission File ---

# Clip predictions if necessary to keep within realistic bounds (optional but safer)
submission = pd.DataFrame({
    "id": test[ID],
    "accident_risk": np.clip(final_preds, 0, 1)  # Assuming risk is between 0 and 1
})

# Save for Kaggle submission
submission.to_csv("submission.csv", index=False)

print("âœ… Submission file created successfully!")
print("Saved as: submission.csv")

# Display a quick preview
display(submission.head())


