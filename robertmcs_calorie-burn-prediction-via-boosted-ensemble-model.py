# Cell 1: Library Imports and GPU Setup

# Core libraries
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

# Gradient boosting frameworks
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Model interpretability
import shap

# PyTorch (for residual correction or deep models)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Detect GPU device (NVIDIA CUDA or Apple MPS)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")



# Cell 2: Load training and test data

# Define dataset paths
train_path = "dataset/train.csv"
test_path = "dataset/test.csv"

# Load the datasets
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Basic inspection
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain preview:")
display(train.head())

print("\nTest preview:")
display(test.head())

# Data types and missing values
print("\nTrain info:")
train.info()

print("\nMissing values in train:")
print(train.isnull().sum())

print("\nMissing values in test:")
print(test.isnull().sum())

# Check for duplicate rows
print("\nDuplicate rows in train:", train.duplicated().sum())
print("Duplicate rows in test:", test.duplicated().sum())



# Cell 3: Visualize and assess distribution of the target variable

from scipy.stats import skew, kurtosis

# Plot raw distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(train["Calories"], bins=50, kde=True, color="steelblue")
plt.title("Raw Calories Distribution")
plt.xlabel("Calories")

# Log1p transformation
train["Calories_log"] = np.log1p(train["Calories"])

plt.subplot(1, 2, 2)
sns.histplot(train["Calories_log"], bins=50, kde=True, color="darkorange")
plt.title("Log1p Transformed Calories Distribution")
plt.xlabel("log1p(Calories)")

plt.tight_layout()
plt.show()

# Skewness and kurtosis
raw_skew = skew(train["Calories"])
log_skew = skew(train["Calories_log"])
raw_kurt = kurtosis(train["Calories"])
log_kurt = kurtosis(train["Calories_log"])

print(f"Skewness (Raw): {raw_skew:.4f}")
print(f"Skewness (Log1p): {log_skew:.4f}")
print(f"Kurtosis (Raw): {raw_kurt:.4f}")
print(f"Kurtosis (Log1p): {log_kurt:.4f}")



# Cell 4: Define RMSLE evaluation function

from sklearn.metrics import mean_squared_log_error

def rmsle(y_true, y_pred, clip_negatives=True):
    """
    Compute Root Mean Squared Logarithmic Error (RMSLE).
    
    Parameters:
        y_true (array-like): Actual values (non-transformed)
        y_pred (array-like): Predicted values (non-transformed)
        clip_negatives (bool): If True, clip negative predictions to 0 to avoid domain errors
    
    Returns:
        float: RMSLE score
    """
    if clip_negatives:
        y_pred = np.clip(y_pred, 0, None)
    
    return np.sqrt(mean_squared_log_error(y_true, y_pred))



# Optional: test RMSLE function
y_true_sample = np.array([10, 20, 50])
y_pred_sample = np.array([12, 18, 45])
print("Sample RMSLE:", rmsle(y_true_sample, y_pred_sample))



# Cell 5: Create BMI, MET proxy, and interaction features

def engineer_features(df):
    """
    Engineer physiological and interaction features.
    
    Parameters:
        df (DataFrame): Raw input DataFrame (train or test)
    
    Returns:
        DataFrame: Copy of df with new engineered features
    """
    df = df.copy()
    
    # Convert height from cm to meters
    height_m = df["Height"] / 100

    # BMI: Body Mass Index
    df["BMI"] = df["Weight"] / (height_m ** 2)
    
    # MET proxy: effort based on weight, duration, heart rate
    df["MET_proxy"] = df["Duration"] * df["Heart_Rate"] * df["Weight"]
    
    # Interaction features
    df["HRxDuration"] = df["Heart_Rate"] * df["Duration"]
    df["WeightxDuration"] = df["Weight"] * df["Duration"]
    df["AgexHR"] = df["Age"] * df["Heart_Rate"]

    return df

# Apply to both datasets
train_fe = engineer_features(train)
test_fe = engineer_features(test)

# Confirm new columns
new_cols = ['BMI', 'MET_proxy', 'HRxDuration', 'WeightxDuration', 'AgexHR']
print("New features added:", new_cols)
display(train_fe[new_cols].describe())



# Cell 6: Detect and cap outliers for engineered features

def cap_outliers(df, features, lower_quantile=0.005, upper_quantile=0.995):
    """
    Cap outliers in the given features using quantile thresholds.

    Parameters:
        df (DataFrame): DataFrame to modify
        features (list): List of column names to cap
        lower_quantile (float): Lower threshold (default: 0.5%)
        upper_quantile (float): Upper threshold (default: 99.5%)

    Returns:
        DataFrame: Modified copy with capped features
    """
    df = df.copy()
    
    for col in features:
        lower_bound = df[col].quantile(lower_quantile)
        upper_bound = df[col].quantile(upper_quantile)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        print(f"{col}: capped between {lower_bound:.2f} and {upper_bound:.2f}")
    
    return df

# Features to cap
features_to_cap = ['BMI', 'MET_proxy', 'HRxDuration', 'WeightxDuration', 'AgexHR']

# Apply capping
train_fe = cap_outliers(train_fe, features_to_cap)
test_fe = cap_outliers(test_fe, features_to_cap)



# Cell 7: Leave-One-Out (LOO) Encoding for 'Sex' with K-Fold CV

from sklearn.model_selection import KFold

def loo_encode(train_df, test_df, categorical_col, target_col, n_splits=5):
    """
    Perform Leave-One-Out encoding on a categorical column using K-Fold strategy.
    
    Parameters:
        train_df (DataFrame): Training data including target
        test_df (DataFrame): Test data (no target)
        categorical_col (str): Column to encode
        target_col (str): Target variable name
        n_splits (int): Number of folds for encoding
        
    Returns:
        train_encoded, test_encoded: DataFrames with LOO-encoded column added
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    global_mean = train_df[target_col].mean()

    col_name = f"{categorical_col}_LOO"
    train_df[col_name] = np.nan

    for train_idx, val_idx in kf.split(train_df):
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]
        
        means = fold_train.groupby(categorical_col)[target_col].mean()
        train_df.iloc[val_idx, train_df.columns.get_loc(col_name)] = fold_val[categorical_col].map(means)
    
    # Fill any unseen categories with global mean (shouldn't happen in binary case)
    train_df[col_name].fillna(global_mean, inplace=True)

    # Apply means to test set
    category_means = train_df.groupby(categorical_col)[target_col].mean()
    test_df[col_name] = test_df[categorical_col].map(category_means)
    test_df[col_name].fillna(global_mean, inplace=True)

    # Drop original categorical column
    train_df.drop(columns=[categorical_col], inplace=True)
    test_df.drop(columns=[categorical_col], inplace=True)

    return train_df, test_df

# Apply LOO encoding
train_fe, test_fe = loo_encode(train_fe, test_fe, categorical_col="Sex", target_col="Calories")

# Confirm result
print("LOO Encoding complete. Preview:")
display(train_fe[["Sex_LOO"]].head())



# Cell 8: Feature-target relationship analysis (correlation + scatter plots)

# 1. Correlation heatmap (including Calories)
numeric_cols = train_fe.select_dtypes(include=[np.number]).columns.tolist()
correlation_matrix = train_fe[numeric_cols + ['Calories']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix (including target: Calories)")
plt.tight_layout()
plt.show()

# 2. Scatter plots of engineered features vs Calories
engineered_features = ['BMI', 'MET_proxy', 'HRxDuration', 'WeightxDuration', 'AgexHR', 'Sex_LOO']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, feature in enumerate(engineered_features):
    axes[i].scatter(train_fe[feature], train_fe["Calories"], alpha=0.1, s=10)
    axes[i].set_title(f"{feature} vs. Calories")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Calories")

plt.tight_layout()
plt.show()



# Cell 9: Finalize training and test matrices

# Define target (log-transformed)
y_log = np.log1p(train_fe["Calories"])

# Columns to exclude from features
excluded_cols = ["id", "Calories", "Calories_log"]

# Final feature set
X_train = train_fe.drop(columns=excluded_cols)
X_test = test_fe.drop(columns=["id"])

# Confirm shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_log shape:", y_log.shape)



import os
os.environ["LIGHTGBM_VERBOSE"] = "-1"  # Set to -1 for even less verbosity

import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import numpy as np
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_rmsle = []
test_preds = np.zeros(X_test.shape[0])

# Define a custom callback to completely silence LightGBM's output
def silent_callback(env):
    # This callback does nothing, just prevents default logging
    pass

print("Starting 5-fold cross validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    # Simplified output - just show which fold is training
    print(f"Training Fold {fold + 1}...", end=" ")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
    
    model = LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=1000,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=fold,
        device='gpu',
        verbose=-1  # Set model-specific verbosity to minimum
    )
    
    # Use custom silent callback and disable default logging
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[
            early_stopping(stopping_rounds=50),
            silent_callback  # Use silent callback instead of log_evaluation
        ]
        # Removed the verbose parameter that was causing the error
    )
    
    val_preds_log = model.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    y_val_actual = np.expm1(y_val)
    fold_score = rmsle(y_val_actual, val_preds)
    fold_rmsle.append(fold_score)
    
    # Print just the fold score on the same line
    print(f"RMSLE: {fold_score:.5f}")
    
    test_preds += model.predict(X_test) / kf.n_splits

# Summary
print("\nCross-Validation RMSLE Scores:")
for i, score in enumerate(fold_rmsle, 1):
    print(f"  Fold {i}: {score:.5f}")
print(f"\nAverage CV RMSLE: {np.mean(fold_rmsle):.5f}")


import xgboost as xgb
from sklearn.model_selection import KFold
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=7)
fold_rmsle_xgb = []
test_preds_xgb = np.zeros(X_test.shape[0])

print("Starting 5-fold cross validation (XGBoost)...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Training Fold {fold + 1}...", end=" ")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
    
    # Create DMatrix objects for XGBoost 3.0.0
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Define parameters for XGBoost 3.0.0
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': fold,
        'eval_metric': 'rmse',
        'verbosity': 0
    }
    
    # Train using the XGBoost training API
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dval, 'validation')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # Make predictions
    val_preds_log = model.predict(dval)
    val_preds = np.expm1(val_preds_log)
    y_val_actual = np.expm1(y_val)
    fold_score = rmsle(y_val_actual, val_preds)
    fold_rmsle_xgb.append(fold_score)
    
    print(f"RMSLE: {fold_score:.5f}")
    
    # Create DMatrix for test data
    dtest = xgb.DMatrix(X_test)
    test_preds_xgb += model.predict(dtest) / kf.n_splits

# Summary
print("\nCross-Validation RMSLE Scores (XGBoost):")
for i, score in enumerate(fold_rmsle_xgb, 1):
    print(f"  Fold {i}: {score:.5f}")

# Calculate mean excluding potential outlier in fold 4
mean_rmsle = np.mean(fold_rmsle_xgb)
mean_without_outlier = np.mean([score for i, score in enumerate(fold_rmsle_xgb) if i != 3])  # Exclude fold 4

print(f"\nAverage CV RMSLE: {mean_rmsle:.5f}")
print(f"Average CV RMSLE (excluding fold 4): {mean_without_outlier:.5f}")

# Optional: Investigate if fold 4 is an outlier
if abs(fold_rmsle_xgb[3] - np.median(fold_rmsle_xgb)) > 2 * np.std(fold_rmsle_xgb):
    print("\nNote: Fold 4 appears to be an outlier. Consider investigating or re-running with a different random seed.")


from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
import numpy as np

kf = KFold(n_splits=5, shuffle=True, random_state=7)
fold_rmsle_cb = []
test_preds_cb = np.zeros(X_test.shape[0])

print("Starting 5-fold cross validation (CatBoost)...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Training Fold {fold + 1}...", end=" ")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

    # CatBoost uses Pool objects for training/validation
    train_pool = Pool(data=X_tr, label=y_tr)
    val_pool = Pool(data=X_val, label=y_val)

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        task_type='GPU',
        random_seed=fold,
        verbose=0  # Suppress training logs
    )

    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

    val_preds_log = model.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    y_val_actual = np.expm1(y_val)
    fold_score = rmsle(y_val_actual, val_preds)
    fold_rmsle_cb.append(fold_score)

    print(f"RMSLE: {fold_score:.5f}")

    test_pool = Pool(data=X_test)
    test_preds_cb += model.predict(test_pool) / kf.n_splits

# Summary
print("\nCross-Validation RMSLE Scores (CatBoost):")
for i, score in enumerate(fold_rmsle_cb, 1):
    print(f"  Fold {i}: {score:.5f}")
print(f"\nAverage CV RMSLE: {np.mean(fold_rmsle_cb):.5f}")



# Cell 13: Blend LightGBM, XGBoost, and CatBoost predictions

# Step 1: Simple average
test_preds_avg_simple = (test_preds + test_preds_xgb + test_preds_cb) / 3

# Step 2: Weighted average (based on inverse RMSLE — better models get more weight)
rmsle_scores = {
    'lgb': 0.06013,
    'xgb': 0.06036,
    'cb' : 0.06030
}

# Compute weights (lower RMSLE = higher weight)
inv_errors = {k: 1 / v for k, v in rmsle_scores.items()}
total_weight = sum(inv_errors.values())
weights = {k: v / total_weight for k, v in inv_errors.items()}

# Apply weighted average
test_preds_avg_weighted = (
    test_preds * weights['lgb'] +
    test_preds_xgb * weights['xgb'] +
    test_preds_cb * weights['cb']
)

# Display weights for transparency
print(" Weighted blending coefficients:")
for model, w in weights.items():
    print(f"  {model.upper()}: {w:.4f}")



import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Generate OOF predictions for the best model or ensemble
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(X_train.shape[0])

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

    # Train on fold (CatBoost shown here — replace with best or blended model if needed)
    train_pool = Pool(X_tr, y_tr)
    val_pool = Pool(X_val)

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        task_type='GPU',
        random_seed=42,
        verbose=0
    )
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)

    oof_preds[val_idx] = model.predict(X_val)

# Step 2: Inverse transform and calculate residuals
y_true = np.expm1(y_log)
y_pred = np.expm1(oof_preds)
residuals = y_true - y_pred

# Step 3: Residual plots
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Calories")
plt.ylabel("Residuals")
plt.title("Residuals vs. Predicted Values")
plt.tight_layout()
plt.show()

# Optionally plot residuals vs. a key feature (e.g. Duration)
plt.figure(figsize=(10, 5))
sns.scatterplot(x=X_train['Duration'], y=residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Duration")
plt.ylabel("Residuals")
plt.title("Residuals vs. Duration")
plt.tight_layout()
plt.show()

# Stats summary
print(f"Residual mean: {np.mean(residuals):.4f}")
print(f"Residual std:  {np.std(residuals):.4f}")



from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Step 1: Prepare features and target (residuals)
residual_target = y_true - y_pred  # original-space residuals

# Use the same features as before for correction
X_resid = X_train.copy()

# Optional: scale features for better MLP performance
scaler = StandardScaler()
X_resid_scaled = scaler.fit_transform(X_resid)

# Step 2: Train MLP on residuals
mlp = MLPRegressor(hidden_layer_sizes=(64, 32),
                   activation='relu',
                   solver='adam',
                   max_iter=500,
                   random_state=42,
                   early_stopping=True)

mlp.fit(X_resid_scaled, residual_target)

# Step 3: Apply correction to test predictions
X_test_scaled = scaler.transform(X_test)
residual_corrections = mlp.predict(X_test_scaled)

# Final prediction = blended prediction + residual correction
final_preds = test_preds_avg_weighted + residual_corrections

# Ensure predictions remain non-negative
final_preds = np.maximum(final_preds, 0)

# Display correction stats
print(f"Correction Mean: {np.mean(residual_corrections):.4f}")
print(f"Correction Std : {np.std(residual_corrections):.4f}")



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# Step 1: Split into training and holdout sets (log target)
X_subtrain, X_holdout, y_subtrain_log, y_holdout_log = train_test_split(
    X_train, y_log, test_size=0.2, random_state=42)

y_holdout = np.expm1(y_holdout_log)  # true values in original scale

# Step 2: Train LGB, XGB, CB on subtrain only
def train_model(model, X, y, X_val):
    model.fit(X, y)
    return np.expm1(model.predict(X_val))

# LightGBM
from lightgbm import LGBMRegressor
lgb_model = LGBMRegressor(
    objective='regression',
    n_estimators=1000,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    device='gpu',
    random_state=42
)
lgb_preds = train_model(lgb_model, X_subtrain, y_subtrain_log, X_holdout)

# XGBoost
import xgboost as xgb
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    eval_metric='rmse',
    random_state=42
)
xgb_preds = train_model(xgb_model, X_subtrain, y_subtrain_log, X_holdout)

# CatBoost
from catboost import CatBoostRegressor
cb_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    task_type='GPU',
    loss_function='RMSE',
    random_seed=42,
    verbose=0
)
cb_model.fit(X_subtrain, y_subtrain_log)
cb_preds = np.expm1(cb_model.predict(X_holdout))

# Step 3: Blended prediction (weighted average)
weights = {'lgb': 0.3341, 'xgb': 0.3328, 'cb': 0.3331}
blended_preds = (
    lgb_preds * weights['lgb'] +
    xgb_preds * weights['xgb'] +
    cb_preds  * weights['cb']
)

# Step 4: Residual correction using MLP on subtrain residuals
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Prepare training residuals
residuals_train = np.expm1(y_subtrain_log) - np.expm1(lgb_model.predict(X_subtrain) * weights['lgb'] +
                                                       xgb_model.predict(X_subtrain) * weights['xgb'] +
                                                       cb_model.predict(X_subtrain) * weights['cb'])

scaler = StandardScaler()
X_subtrain_scaled = scaler.fit_transform(X_subtrain)
X_holdout_scaled = scaler.transform(X_holdout)

mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                   max_iter=500, random_state=42, early_stopping=True)
mlp.fit(X_subtrain_scaled, residuals_train)

residual_correction = mlp.predict(X_holdout_scaled)

# Step 5: Final corrected predictions
final_corrected_preds = blended_preds + residual_correction
final_corrected_preds = np.maximum(final_corrected_preds, 0)

# Step 6: RMSLE Comparison
def rmsle_score(y_true, y_pred):
    return mean_squared_log_error(y_true, y_pred) ** 0.5

baseline_rmsle = rmsle_score(y_holdout, blended_preds)
corrected_rmsle = rmsle_score(y_holdout, final_corrected_preds)

print(f"Baseline Ensemble RMSLE on Holdout:       {baseline_rmsle:.5f}")
print(f"Residual-Corrected RMSLE on Holdout:      {corrected_rmsle:.5f}")
print(f"Improvement:                              {baseline_rmsle - corrected_rmsle:.5f}")



import pandas as pd

# Step 1: Load the sample submission
sample_sub = pd.read_csv("dataset/sample_submission.csv")

# Step 2: Insert predictions
sample_sub["Calories"] = test_preds_avg_weighted

# Step 3: Ensure non-negative predictions (safety check)
sample_sub["Calories"] = sample_sub["Calories"].clip(lower=0)

# Step 4: Save submission file
submission_filename = "submission_blended.csv"
sample_sub.to_csv(submission_filename, index=False)

print(f" Submission saved as '{submission_filename}' with {len(sample_sub)} entries.")



# Preview the final submission file
submission_preview = sample_sub.head(10)
submission_preview



# Merge test data with predictions
output_df = X_test.copy()
output_df = output_df.reset_index(drop=True)
output_df["Predicted_Calories"] = test_preds_avg_weighted

# If 'id' exists in original test CSV, merge it too
test_ids = pd.read_csv("dataset/test.csv")[["id", "Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]]
output_df = pd.concat([test_ids.reset_index(drop=True), output_df["Predicted_Calories"]], axis=1)

# Preview the top rows
output_df.head(10)





