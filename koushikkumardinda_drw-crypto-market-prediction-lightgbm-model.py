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


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import gc
import warnings

warnings.filterwarnings('ignore')

# 1. Load Data
# Assuming train.parquet and test.parquet are in '../input/drw-crypto-market-prediction/'
print("Loading data...")
train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
sample_submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
print("Data loaded.")

# 2. Basic Memory Reduction Function
def reduce_mem_usage(df):
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage of dataframe before reduction: {start_mem:.2f} MB")
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
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else: # floats
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage of dataframe after reduction: {end_mem:.2f} MB")
    return df

train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)

# 3. Feature Engineering (Basic)
def engineer_features(df):
    df['bid_ask_spread'] = df['ask_qty'] - df['bid_qty']
    df['total_qty'] = df['bid_qty'] + df['ask_qty']
    df['buy_sell_diff'] = df['buy_qty'] - df['sell_qty']
    df['volume_per_total_qty'] = df['volume'] / (df['total_qty'] + 1e-6) # Add epsilon to avoid division by zero
    return df

print("Engineering features...")
train_df = engineer_features(train_df)
test_df = engineer_features(test_df)
print("Features engineered.")

# 4. Define Features and Target
TARGET = 'label'
features = [col for col in train_df.columns if col not in ['ID', 'timestamp', TARGET]]

# Drop columns with only one unique value (these provide no predictive power)
# and handle 'label' column in test_df (it's not present during actual test inference)
nunique_cols = [col for col in train_df.columns if train_df[col].nunique() == 1]
train_df.drop(columns=nunique_cols, inplace=True)
features = [f for f in features if f not in nunique_cols]
# Ensure test_df also has these dropped
if TARGET in test_df.columns:
    test_df.drop(columns=[TARGET] + nunique_cols, inplace=True)
else:
    test_df.drop(columns=nunique_cols, inplace=True)

# Align columns - very important for robust predictions
train_cols = set(train_df.columns)
test_cols = set(test_df.columns)
common_features = list(train_cols.intersection(test_cols) - {TARGET, 'ID', 'timestamp'})
features = [f for f in features if f in common_features] # Only use features present in both train and test

X = train_df[features]
y = train_df[TARGET]

del train_df # Free up memory
gc.collect()

# 5. Time Series Split (Crucial for financial data)
# Use a simple time-based split for initial exploration
# The data is already sorted by timestamp.
split_index = int(len(X) * 0.8)
X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

del X, y # Free up memory
gc.collect()

print(f"Train data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# 6. LightGBM Model Training (CPU)
print("Training LightGBM model (CPU)...")
lgb_params = {
    'objective': 'regression_l1', # MAE is often robust for financial predictions
    'metric': 'rmse', # RMSE for internal evaluation
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1, # Use all available CPU cores
    'seed': 42,
    'boosting_type': 'gbdt',
}

model = lgb.LGBMRegressor(**lgb_params)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='rmse', # Use RMSE for early stopping
          callbacks=[lgb.early_stopping(100, verbose=False)], # Early stopping rounds
          )
print("LightGBM training complete.")

# 7. Prediction and Evaluation
val_preds = model.predict(X_val)
correlation, _ = pearsonr(val_preds, y_val)
print(f"Validation Pearson Correlation: {correlation:.4f}")

# 8. Generate Test Predictions
print("Generating test predictions...")
test_preds = model.predict(test_df[features])

# Clip predictions to the expected range if specified in competition
# (e.g., between -5.0 and 5.0 as seen in some discussions)
test_preds = np.clip(test_preds, -5.0, 5.0)

# Create submission file
submission_df = pd.DataFrame({'ID': sample_submission['ID'], 'prediction': test_preds})
submission_df.to_csv('submission1.csv', index=False)
print("Submission file created: submission1.csv")

del X_train, X_val, y_train, y_val, test_df, model, test_preds # Clean up memory
gc.collect()


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import gc
import warnings
import torch # Just to check for CUDA availability if needed for other models

warnings.filterwarnings('ignore')

# 1. Load Data and Memory Reduction (same as Level 1)
print("Loading data...")
train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
sample_submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
print("Data loaded.")

def reduce_mem_usage(df):
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage of dataframe before reduction: {start_mem:.2f} MB")
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
                elif c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max: # Use float16 for floats
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else: # floats
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage of dataframe after reduction: {end_mem:.2f} MB")
    return df

train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)

# 2. Feature Engineering (More Advanced/Time-series focused)
def engineer_features_advanced(df):
    df['bid_ask_spread'] = df['ask_qty'] - df['bid_qty']
    df['total_qty'] = df['bid_qty'] + df['ask_qty']
    df['buy_sell_diff'] = df['buy_qty'] - df['sell_qty']
    df['volume_per_total_qty'] = df['volume'] / (df['total_qty'] + 1e-6)

    # Introduce lag features (common in time series)
    # Be careful with memory here, many lags can explode features
    for col in ['bid_qty', 'ask_qty', 'volume', 'buy_qty', 'sell_qty']:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)
        # Rolling features
        df[f'{col}_rolling_mean_5'] = df[col].rolling(window=5).mean()
        df[f'{col}_rolling_std_5'] = df[col].rolling(window=5).std()
    
    # Interaction features
    df['bid_qty_x_ask_qty'] = df['bid_qty'] * df['ask_qty']

    df.fillna(0, inplace=True) # Or use other imputation strategies like median/mean
    return df

print("Engineering advanced features...")
train_df = engineer_features_advanced(train_df)
test_df = engineer_features_advanced(test_df)
print("Advanced features engineered.")

# 3. Define Features and Target (same as Level 1, ensure alignment)
TARGET = 'label'
features = [col for col in train_df.columns if col not in ['ID', 'timestamp', TARGET]]

nunique_cols = [col for col in train_df.columns if train_df[col].nunique() == 1]
train_df.drop(columns=nunique_cols, inplace=True)
features = [f for f in features if f not in nunique_cols]
if TARGET in test_df.columns:
    test_df.drop(columns=[TARGET] + nunique_cols, inplace=True)
else:
    test_df.drop(columns=nunique_cols, inplace=True)

train_cols = set(train_df.columns)
test_cols = set(test_df.columns)
common_features = list(train_cols.intersection(test_cols) - {TARGET, 'ID', 'timestamp'})
features = [f for f in features if f in common_features]

X = train_df[features]
y = train_df[TARGET]

del train_df # Free up memory
gc.collect()

# 4. Time Series Split
split_index = int(len(X) * 0.8)
X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

del X, y # Free up memory
gc.collect()

print(f"Train data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# 5. LightGBM Model Training (GPU enabled)
print("Training LightGBM model (GPU)...")
lgb_params = {
    'objective': 'regression_l1',
    'metric': 'rmse',
    'n_estimators': 2000, # Increased estimators for potentially better performance
    'learning_rate': 0.02, # Reduced learning rate with more estimators
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 63, # Increased complexity
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
    'device': 'gpu', # <<< Enable GPU
    'gpu_platform_id': 0, # Usually 0 for single GPU
    'gpu_device_id': 0,   # Usually 0 for single GPU
    'max_bin': 63, # Recommended for GPU for speedup
    'gpu_use_dp': False # Use single precision for better performance on most NVIDIA consumer GPUs
}

model = lgb.LGBMRegressor(**lgb_params)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='rmse',
          callbacks=[lgb.early_stopping(200, verbose=False)],
          )
print("LightGBM GPU training complete.")

# 6. Prediction and Evaluation (same as Level 1)
val_preds = model.predict(X_val)
correlation, _ = pearsonr(val_preds, y_val)
print(f"Validation Pearson Correlation: {correlation:.4f}")

# 7. Generate Test Predictions
print("Generating test predictions...")
test_preds = model.predict(test_df[features])
test_preds = np.clip(test_preds, -5.0, 5.0)

submission_df = pd.DataFrame({'ID': sample_submission['ID'], 'prediction': test_preds})
submission_df.to_csv('submission2.csv', index=False)
print("Submission file created: submission2.csv")

del X_train, X_val, y_train, y_val, test_df, model, test_preds # Clean up memory
gc.collect()

