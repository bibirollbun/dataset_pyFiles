# ===================================================================================
# BPM Prediction - Final Optimized Python Pipeline
# ===================================================================================
# Version 3: Incorporates insights from both R and Python scripts.
# - Uses external data and advanced feature engineering from the Python script.
# - Adds data standardization (normalization), a key step from the R script,
#   to see if it improves model convergence and final score.
# ===================================================================================

# ---------------
# 1. SETUP
# ---------------
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler # Added for normalization
import gc
import warnings

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

print("Libraries imported successfully.")

# ---------------
# 2. DATA LOADING & MERGING
# ---------------
# Define file paths
PATH_COMPETITION = "/kaggle/input/playground-series-s5e9/"
PATH_ORIGINAL = "/kaggle/input/bpm-prediction-challenge/"

# Load competition data
df_train = pd.read_csv(PATH_COMPETITION + "train.csv")
df_test = pd.read_csv(PATH_COMPETITION + "test.csv")
submission = pd.read_csv(PATH_COMPETITION + "sample_submission.csv")

# Load original data
try:
    df_original = pd.read_csv(PATH_ORIGINAL + "train.csv")
    print(f"Original dataset loaded successfully. Shape: {df_original.shape}")
    
    # Ensure column names match and are in the same order
    df_original = df_original.rename(columns={'Id': 'id'})
    df_original = df_original[df_train.columns]
    
    # Combine the datasets
    df_train = pd.concat([df_train, df_original], ignore_index=True)
    print(f"Combined training data shape: {df_train.shape}")
    
except FileNotFoundError:
    print("Original dataset not found. Proceeding with competition data only.")
    print(f"Training data shape: {df_train.shape}")

# Store test IDs
test_ids = df_test['id']

# Drop 'id'
df_train = df_train.drop('id', axis=1)
df_test = df_test.drop('id', axis=1)

# Separate target variable
X = df_train.drop('BeatsPerMinute', axis=1)
y = df_train['BeatsPerMinute']
X_test = df_test.copy()

# Align columns
X_test = X_test[X.columns]

print("Data loading and merging complete.")
gc.collect()

# ---------------
# 3. FEATURE ENGINEERING
# ---------------
print("Creating interaction and statistical features...")
original_features = list(X.columns)

for i in range(len(original_features)):
    for j in range(i, len(original_features)):
        col1 = original_features[i]
        col2 = original_features[j]
        # Create interaction features
        X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
        X_test[f'{col1}_x_{col2}'] = X_test[col1] * X_test[col2]
        # Create ratio features, handle division by zero
        if col1 != col2:
            X[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-6)
            X_test[f'{col1}_div_{col2}'] = X_test[col1] / (X_test[col2] + 1e-6)

X['feature_mean'] = X[original_features].mean(axis=1)
X['feature_std'] = X[original_features].std(axis=1)
X_test['feature_mean'] = X_test[original_features].mean(axis=1)
X_test['feature_std'] = X_test[original_features].std(axis=1)

print("Feature engineering complete.")
print(f"New shape of training features: {X.shape}")
gc.collect()

# ---------------
# 4. DATA SCALING (Normalization) - Key step from R script
# ---------------
print("Applying StandardScaler to the data...")
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

gc.collect()

# ---------------
# 5. MODEL TRAINING (LightGBM with CV and Seed Ensembling)
# ---------------
# Parameters inspired by high-scoring public notebooks and your R script
lgb_params = {
    'objective': 'regression_l2', # L2 is the default for RMSE
    'metric': 'rmse',
    'n_estimators': 10000, # Increased estimators with early stopping
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}

# Setup for Cross-Validation and Ensembling
N_SPLITS = 10
SEEDS = [1975, 2000, 2503] # Using some seeds from your R script
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
total_models = len(SEEDS) * N_SPLITS

print(f"\nStarting training with {N_SPLITS}-Fold CV and ensembling across {len(SEEDS)} seeds...")

for seed in SEEDS:
    print(f"\n--- Training with Seed: {seed} ---")
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    
    params = lgb_params.copy()
    params['seed'] = seed
    
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        print(f"  -> Fold {fold+1}/{N_SPLITS}")
        
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(500, verbose=False)])

        val_preds = model.predict(X_val)
        fold_test_preds = model.predict(X_test)
        
        oof_preds[val_index] += val_preds / len(SEEDS)
        test_preds += fold_test_preds / total_models
        
        del X_train, X_val, y_train, y_val, model
        gc.collect()

final_oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print("\n-------------------------------------------")
print(f"Training complete.")
print(f"Final OOF RMSE across all seeds: {final_oof_rmse:.5f}")
print("-------------------------------------------")

# ---------------
# 6. SUBMISSION
# ---------------
submission['BeatsPerMinute'] = test_preds
submission.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully.")
display(submission.head())

