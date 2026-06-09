# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Memory-Efficient Baseline for DRW Crypto Market Prediction
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import gc
import warnings
warnings.filterwarnings('ignore')

print("ğŸš€ Starting Memory-Efficient Baseline Model")

# =============================================================================
# Memory optimization function
# =============================================================================
def reduce_mem_usage(df, name="DataFrame"):
    """Reduce memory usage of dataframe by downcasting numeric types"""
    print(f"Optimizing memory for {name}...")
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)  # float16 can be unstable
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem
    print(f"Memory usage: {start_mem:.1f}MB â†’ {end_mem:.1f}MB ({reduction:.1f}% reduction)")
    return df

# =============================================================================
# Step 1: Load and Optimize Data
# =============================================================================
print("\nğŸ“‚ Loading data...")
train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

print(f"Initial shapes - Train: {train.shape}, Test: {test.shape}")

# Optimize memory immediately
train = reduce_mem_usage(train, "train")
test = reduce_mem_usage(test, "test")

# =============================================================================
# Step 2: Feature Selection (Keep only most important to save memory)
# =============================================================================
print("\nğŸ�¯ Feature selection for memory efficiency...")

target = 'label'
y_train = train[target].astype(np.float32)

# Get all feature columns
all_features = [col for col in train.columns if col not in [target, 'ID', 'timestamp']]
print(f"Total features available: {len(all_features)}")

# Quick feature selection based on correlation with target
feature_correlations = []
sample_size = min(50000, len(train))  # Use sample for quick correlation calculation
train_sample = train.sample(sample_size, random_state=42)

print("Calculating feature correlations (using sample)...")
for col in all_features:
    try:
        corr = abs(train_sample[col].corr(train_sample[target]))
        if not np.isnan(corr):
            feature_correlations.append((col, corr))
    except:
        continue

# Sort by correlation and take top features
feature_correlations.sort(key=lambda x: x[1], reverse=True)
top_n_features = min(200, len(feature_correlations))  # Limit to 200 features
selected_features = [feat[0] for feat in feature_correlations[:top_n_features]]

print(f"Selected top {len(selected_features)} features")
print(f"Top 5 features: {[f'{feat}({corr:.4f})' for feat, corr in feature_correlations[:5]]}")

# =============================================================================
# Step 3: Prepare Final Datasets
# =============================================================================
print("\nğŸ”§ Preparing final datasets...")

X_train = train[selected_features].astype(np.float32)
X_test = test[selected_features].astype(np.float32)

# Clean up original dataframes
del train, test
gc.collect()

print(f"Final shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")

# =============================================================================
# Step 4: Data Cleaning (Memory Efficient)
# =============================================================================
print("\nğŸ§¹ Data cleaning...")

# Handle missing values
missing_cols = X_train.columns[X_train.isnull().any()].tolist()
if missing_cols:
    print(f"Filling {len(missing_cols)} columns with missing values...")
    for col in missing_cols:
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)

# Handle infinite values
print("Checking for infinite values...")
inf_mask_train = np.isinf(X_train.values).any(axis=1)
inf_mask_test = np.isinf(X_test.values).any(axis=1)

if inf_mask_train.any() or inf_mask_test.any():
    print("Replacing infinite values...")
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill with median again
    for col in X_train.columns:
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)

print("âœ… Data cleaning completed")

# =============================================================================
# Step 5: Simple XGBoost Model
# =============================================================================
print("\nğŸ�¯ Training lightweight XGBoost model...")

# Memory-efficient XGBoost parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',  # Memory efficient
    'max_depth': 4,         # Reduced depth
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 100,    # Reduced trees
    'random_state': 42,
    'verbosity': 0,
    'n_jobs': 1             # Single thread to save memory
}

# Simple train/validation split instead of full CV to save memory
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"Training set: {X_tr.shape}, Validation set: {X_val.shape}")

# Train model
print("Training XGBoost...")
model = xgb.XGBRegressor(**xgb_params)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,
    verbose=False
)

# =============================================================================
# Step 6: Evaluation
# =============================================================================
print("\nğŸ“Š Model evaluation...")

# Validation predictions
val_pred = model.predict(X_val)

# Calculate metrics
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
val_corr = pearsonr(y_val, val_pred)[0]

print(f"Validation RMSE: {val_rmse:.6f}")
print(f"Validation Correlation: {val_corr:.6f}")

# =============================================================================
# Step 7: Make Predictions and Create Submission
# =============================================================================
print("\nğŸ”® Making test predictions...")

test_predictions = model.predict(X_test)

# Load sample submission
sample_sub = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')

# Create submission
submission = pd.DataFrame({
    sample_sub.columns[0]: sample_sub.iloc[:, 0],
    'prediction': test_predictions.astype(np.float32)
})

# Save submission
submission.to_csv('memory_efficient_baseline.csv', index=False)

print("âœ… Submission saved as 'memory_efficient_baseline.csv'")

# =============================================================================
# Step 8: Summary
# =============================================================================
print(f"\nğŸ�‰ Memory-Efficient Baseline Summary:")
print(f"  ğŸ“Š Validation Correlation: {val_corr:.6f}")
print(f"  ğŸ“ˆ Validation RMSE: {val_rmse:.6f}")
print(f"  ğŸ�¯ Features Used: {len(selected_features)}")
print(f"  ğŸ§  Model: Lightweight XGBoost")
print(f"  ğŸ’¾ Memory: Optimized for Kaggle limits")

print(f"\nğŸ’¡ This baseline should run within memory limits!")
print(f"   Next steps: Gradually add more features/complexity")

# Clean up
del X_train, X_test, X_tr, X_val, y_tr, y_val
gc.collect()

print("ğŸ�¯ Memory cleanup completed")


# Memory-Efficient Baseline for DRW Crypto Market Prediction
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import gc
import warnings
warnings.filterwarnings('ignore')

print("ğŸš€ Starting Memory-Efficient Baseline Model")

# =============================================================================
# Memory optimization function
# =============================================================================
def reduce_mem_usage(df, name="DataFrame"):
    """Reduce memory usage of dataframe by downcasting numeric types"""
    print(f"Optimizing memory for {name}...")
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)  # float16 can be unstable
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem
    print(f"Memory usage: {start_mem:.1f}MB â†’ {end_mem:.1f}MB ({reduction:.1f}% reduction)")
    return df

# =============================================================================
# Step 1: Load and Optimize Data
# =============================================================================
print("\nğŸ“‚ Loading data...")
train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

print(f"Initial shapes - Train: {train.shape}, Test: {test.shape}")

# Optimize memory immediately
train = reduce_mem_usage(train, "train")
test = reduce_mem_usage(test, "test")

# =============================================================================
# Step 2: Feature Selection (Keep only most important to save memory)
# =============================================================================
print("\nğŸ�¯ Feature selection for memory efficiency...")

target = 'label'
y_train = train[target].astype(np.float32)

# Get all feature columns
all_features = [col for col in train.columns if col not in [target, 'ID', 'timestamp']]
print(f"Total features available: {len(all_features)}")

# Quick feature selection based on correlation with target
feature_correlations = []
sample_size = min(50000, len(train))  # Use sample for quick correlation calculation
train_sample = train.sample(sample_size, random_state=42)

print("Calculating feature correlations (using sample)...")
for col in all_features:
    try:
        corr = abs(train_sample[col].corr(train_sample[target]))
        if not np.isnan(corr):
            feature_correlations.append((col, corr))
    except:
        continue

# Sort by correlation and take top features
feature_correlations.sort(key=lambda x: x[1], reverse=True)
top_n_features = min(50, len(feature_correlations))  # Limit to 50 features
selected_features = [feat[0] for feat in feature_correlations[:top_n_features]]

print(f"Selected top {len(selected_features)} features")
print(f"Top 5 features: {[f'{feat}({corr:.4f})' for feat, corr in feature_correlations[:5]]}")

# =============================================================================
# Step 3: Prepare Final Datasets
# =============================================================================
print("\nğŸ”§ Preparing final datasets...")

X_train = train[selected_features].astype(np.float32)
X_test = test[selected_features].astype(np.float32)

# Clean up original dataframes
del train, test
gc.collect()

print(f"Final shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")

# =============================================================================
# Step 4: Data Cleaning (Memory Efficient)
# =============================================================================
print("\nğŸ§¹ Data cleaning...")

# Handle missing values
missing_cols = X_train.columns[X_train.isnull().any()].tolist()
if missing_cols:
    print(f"Filling {len(missing_cols)} columns with missing values...")
    for col in missing_cols:
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)

# Handle infinite values
print("Checking for infinite values...")
inf_mask_train = np.isinf(X_train.values).any(axis=1)
inf_mask_test = np.isinf(X_test.values).any(axis=1)

if inf_mask_train.any() or inf_mask_test.any():
    print("Replacing infinite values...")
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill with median again
    for col in X_train.columns:
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)

print("âœ… Data cleaning completed")

# =============================================================================
# Step 5: Simple XGBoost Model
# =============================================================================
print("\nğŸ�¯ Training lightweight XGBoost model...")

# Memory-efficient XGBoost parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',  # Memory efficient
    'max_depth': 3,         # Reduced depth
    'learning_rate': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'n_estimators': 50,    # Reduced trees
    'random_state': 42,
    'verbosity': 0,
    'n_jobs': 1             # Single thread to save memory
}

# Simple train/validation split instead of full CV to save memory
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"Training set: {X_tr.shape}, Validation set: {X_val.shape}")

# Train model
print("Training XGBoost...")
model = xgb.XGBRegressor(**xgb_params)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,
    verbose=False
)

# =============================================================================
# Step 6: Evaluation
# =============================================================================
print("\nğŸ“Š Model evaluation...")

# Validation predictions
val_pred = model.predict(X_val)

# Calculate metrics
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
val_corr = pearsonr(y_val, val_pred)[0]

print(f"Validation RMSE: {val_rmse:.6f}")
print(f"Validation Correlation: {val_corr:.6f}")

# =============================================================================
# Step 7: Make Predictions and Create Submission
# =============================================================================
print("\nğŸ”® Making test predictions...")

test_predictions = model.predict(X_test)

# Load sample submission
sample_sub = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')

# Create submission
submission = pd.DataFrame({
    sample_sub.columns[0]: sample_sub.iloc[:, 0],
    'prediction': test_predictions.astype(np.float32)
})

# Save submission
submission.to_csv('memory_efficient_baseline.csv', index=False)

print("âœ… Submission saved as 'memory_efficient_baseline.csv'")

# =============================================================================
# Step 8: Summary
# =============================================================================
print(f"\nğŸ�‰ Memory-Efficient Baseline Summary:")
print(f"  ğŸ“Š Validation Correlation: {val_corr:.6f}")
print(f"  ğŸ“ˆ Validation RMSE: {val_rmse:.6f}")
print(f"  ğŸ�¯ Features Used: {len(selected_features)}")
print(f"  ğŸ§  Model: Lightweight XGBoost")
print(f"  ğŸ’¾ Memory: Optimized for Kaggle limits")

print(f"\nğŸ’¡ This baseline should run within memory limits!")
print(f"   Next steps: Gradually add more features/complexity")

# Clean up
del X_train, X_test, X_tr, X_val, y_tr, y_val
gc.collect()

print("ğŸ�¯ Memory cleanup completed")


# Improved Baseline with 4 Key Enhancements
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import gc
import warnings
warnings.filterwarnings('ignore')

print("ğŸš€ Starting Improved Baseline with 4 Key Enhancements")
print("   1. 300 features (was 200)")
print("   2. XGBoost + LightGBM ensemble")
print("   3. Conservative hyperparameters")
print("   4. Time-series validation")

# =============================================================================
# Memory optimization function
# =============================================================================
def reduce_mem_usage(df, name="DataFrame"):
    """Reduce memory usage of dataframe by downcasting numeric types"""
    print(f"Optimizing memory for {name}...")
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem
    print(f"Memory usage: {start_mem:.1f}MB â†’ {end_mem:.1f}MB ({reduction:.1f}% reduction)")
    return df

# =============================================================================
# Step 1: Load and Optimize Data
# =============================================================================
print("\nğŸ“‚ Loading data...")
train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

print(f"Initial shapes - Train: {train.shape}, Test: {test.shape}")

# Optimize memory immediately
train = reduce_mem_usage(train, "train")
test = reduce_mem_usage(test, "test")

# =============================================================================
# Step 2: Enhanced Feature Selection (Improvement #1: 300 features)
# =============================================================================
print("\nğŸ�¯ Enhanced feature selection (300 features)...")

target = 'label'
y_train = train[target].astype(np.float32)

# Get all feature columns
all_features = [col for col in train.columns if col not in [target, 'ID', 'timestamp']]
print(f"Total features available: {len(all_features)}")

# Quick feature selection based on correlation with target
feature_correlations = []
sample_size = min(50000, len(train))
train_sample = train.sample(sample_size, random_state=42)

print("Calculating feature correlations (using sample)...")
for col in all_features:
    try:
        corr = abs(train_sample[col].corr(train_sample[target]))
        if not np.isnan(corr):
            feature_correlations.append((col, corr))
    except:
        continue

# Sort by correlation and take top features
feature_correlations.sort(key=lambda x: x[1], reverse=True)
top_n_features = min(300, len(feature_correlations))  # Increased from 200 to 300
selected_features = [feat[0] for feat in feature_correlations[:top_n_features]]

print(f"Selected top {len(selected_features)} features (improved from 200)")
print(f"Top 5 features: {[f'{feat}({corr:.4f})' for feat, corr in feature_correlations[:5]]}")

# =============================================================================
# Step 3: Prepare Final Datasets
# =============================================================================
print("\nğŸ”§ Preparing final datasets...")

X_train = train[selected_features].astype(np.float32)
X_test = test[selected_features].astype(np.float32)

# Clean up original dataframes
del train, test
gc.collect()

print(f"Final shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")

# =============================================================================
# Step 4: Enhanced Data Cleaning
# =============================================================================
print("\nğŸ§¹ Enhanced data cleaning...")

# Handle missing values more carefully
missing_cols = X_train.columns[X_train.isnull().any()].tolist()
if missing_cols:
    print(f"Filling {len(missing_cols)} columns with missing values...")
    for col in missing_cols:
        # Use median for robustness
        median_val = X_train[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        X_train[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)

# Handle infinite values more robustly
print("Handling infinite values...")
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill any remaining NaN with column median
for col in X_train.columns:
    if X_train[col].isnull().any() or X_test[col].isnull().any():
        median_val = X_train[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        X_train[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)

print("âœ… Enhanced data cleaning completed")

# =============================================================================
# Step 5: Time-Series Validation (Improvement #4)
# =============================================================================
print("\nğŸ“Š Time-series validation (chronological split)...")

# Split data chronologically instead of random split
# Use first 80% for training, last 20% for validation
split_point = int(0.8 * len(X_train))

X_tr = X_train.iloc[:split_point].copy()
X_val = X_train.iloc[split_point:].copy()
y_tr = y_train.iloc[:split_point].copy()
y_val = y_train.iloc[split_point:].copy()

print(f"Time-series split:")
print(f"  Training set: {X_tr.shape} (first 80% chronologically)")
print(f"  Validation set: {X_val.shape} (last 20% chronologically)")

# =============================================================================
# Step 6: Conservative Model Parameters (Improvement #3)
# =============================================================================
print("\nğŸ�¯ Training models with conservative parameters...")

# Conservative XGBoost parameters to prevent overfitting
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'max_depth': 3,            # Reduced from 4 to 3
    'learning_rate': 0.05,     # Reduced from 0.1 to 0.05
    'subsample': 0.7,          # Reduced from 0.8 to 0.7
    'colsample_bytree': 0.7,   # Reduced from 0.8 to 0.7
    'min_child_weight': 5,     # Increased from 1 to 5
    'reg_alpha': 1,            # Increased from 0 to 1
    'reg_lambda': 2,           # Increased from 1 to 2
    'n_estimators': 150,       # Increased for slower learning
    'random_state': 42,
    'verbosity': 0,
    'n_jobs': 1
}

# Conservative LightGBM parameters
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'max_depth': 3,            # Conservative depth
    'learning_rate': 0.05,     # Slow learning
    'subsample': 0.7,          # Conservative sampling
    'colsample_bytree': 0.7,   # Conservative feature sampling
    'min_child_samples': 20,   # Increased for regularization
    'reg_alpha': 1,            # L1 regularization
    'reg_lambda': 2,           # L2 regularization
    'n_estimators': 150,       # More trees with slower learning
    'random_state': 42,
    'verbosity': -1,
    'n_jobs': 1
}

print("Conservative parameters applied:")
print("  - Reduced max_depth (3)")
print("  - Slower learning_rate (0.05)")
print("  - Stronger regularization")
print("  - Conservative sampling rates")

# =============================================================================
# Step 7: Ensemble Training (Improvement #2: XGBoost + LightGBM)
# =============================================================================
print("\nğŸ”„ Training ensemble models...")

# Train XGBoost
print("Training XGBoost...")
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,
    verbose=False
)

# Get XGBoost predictions
xgb_val_pred = xgb_model.predict(X_val)
xgb_test_pred = xgb_model.predict(X_test)

# Calculate XGBoost scores
xgb_val_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
xgb_val_corr = pearsonr(y_val, xgb_val_pred)[0]

print(f"XGBoost - RMSE: {xgb_val_rmse:.6f}, Correlation: {xgb_val_corr:.6f}")

# Train LightGBM
print("Training LightGBM...")
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
)

# Get LightGBM predictions
lgb_val_pred = lgb_model.predict(X_val)
lgb_test_pred = lgb_model.predict(X_test)

# Calculate LightGBM scores
lgb_val_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_pred))
lgb_val_corr = pearsonr(y_val, lgb_val_pred)[0]

print(f"LightGBM - RMSE: {lgb_val_rmse:.6f}, Correlation: {lgb_val_corr:.6f}")

# =============================================================================
# Step 8: Ensemble Combination
# =============================================================================
print("\nğŸ”— Creating ensemble...")

# Simple average ensemble (you could optimize weights here)
ensemble_val_pred = (xgb_val_pred + lgb_val_pred) / 2
ensemble_test_pred = (xgb_test_pred + lgb_test_pred) / 2

# Calculate ensemble scores
ensemble_val_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_pred))
ensemble_val_corr = pearsonr(y_val, ensemble_val_pred)[0]

print(f"Ensemble - RMSE: {ensemble_val_rmse:.6f}, Correlation: {ensemble_val_corr:.6f}")

# =============================================================================
# Step 9: Model Selection (Best performing model)
# =============================================================================
print("\nğŸ�† Model selection...")

models_performance = {
    'XGBoost': xgb_val_corr,
    'LightGBM': lgb_val_corr,
    'Ensemble': ensemble_val_corr
}

best_model = max(models_performance, key=models_performance.get)
print(f"Best model: {best_model} (Correlation: {models_performance[best_model]:.6f})")

# Select predictions from best model
if best_model == 'XGBoost':
    final_predictions = xgb_test_pred
elif best_model == 'LightGBM':
    final_predictions = lgb_test_pred
else:
    final_predictions = ensemble_test_pred

# =============================================================================
# Step 10: Create Enhanced Submission
# =============================================================================
print("\nğŸ’¾ Creating enhanced submission...")

# Load sample submission
sample_sub = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')

# Create submission with best model
submission = pd.DataFrame({
    sample_sub.columns[0]: sample_sub.iloc[:, 0],
    'prediction': final_predictions.astype(np.float32)
})

# Save submission
submission.to_csv('improved_baseline_v2.csv', index=False)

print("âœ… Submission saved as 'improved_baseline_v2.csv'")

# =============================================================================
# Step 11: Enhanced Summary
# =============================================================================
print(f"\nğŸ�‰ Enhanced Baseline Summary:")
print(f"  ğŸ”§ Improvements Applied:")
print(f"     1. Features: 200 â†’ {len(selected_features)}")
print(f"     2. Models: XGBoost + LightGBM ensemble")
print(f"     3. Conservative hyperparameters")
print(f"     4. Time-series validation")
print(f"")
print(f"  ğŸ“Š Individual Model Performance:")
print(f"     XGBoost:  {xgb_val_corr:.6f}")
print(f"     LightGBM: {lgb_val_corr:.6f}")
print(f"     Ensemble: {ensemble_val_corr:.6f}")
print(f"")
print(f"  ğŸ�† Best Model: {best_model}")
print(f"  ğŸ“ˆ Best Correlation: {models_performance[best_model]:.6f}")
print(f"  ğŸ’¾ Memory: Optimized for Kaggle limits")

print(f"\nğŸ�¯ Expected Improvements:")
print(f"  - More stable predictions (time-series validation)")
print(f"  - Better generalization (conservative parameters)")
print(f"  - Higher accuracy (more features + ensemble)")
print(f"  - Reduced overfitting (regularization)")

# Feature importance analysis
print(f"\nğŸ”� Top 10 Most Important Features:")
if best_model == 'XGBoost':
    importance = xgb_model.feature_importances_
elif best_model == 'LightGBM':
    importance = lgb_model.feature_importances_
else:
    # Average importance for ensemble
    importance = (xgb_model.feature_importances_ + lgb_model.feature_importances_) / 2

importance_df = pd.DataFrame({
    'feature': selected_features,
    'importance': importance
}).sort_values('importance', ascending=False)

print(importance_df.head(10).to_string(index=False))

# Clean up
del X_train, X_test, X_tr, X_val, y_tr, y_val
gc.collect()

print("\nğŸ�¯ Enhanced baseline completed!")


