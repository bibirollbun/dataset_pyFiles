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
import gc # For garbage collection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# Define file paths (assuming you are in a Kaggle notebook environment)
TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
SAMPLE_SUB_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"


# Load data
print("Loading training data...")
train_df = pd.read_parquet(TRAIN_PATH, engine='pyarrow')
print("Loading test data...")
test_df = pd.read_parquet(TEST_PATH, engine='pyarrow')
print("Loading sample submission...")
sample_submission = pd.read_csv(SAMPLE_SUB_PATH)


print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage before: {start_mem:.2f} MB')
    for col in df.columns:
        col_type = df[col].dtype
        if 'float' in str(col_type):
            if df[col].min() > np.finfo(np.float16).min and df[col].max() < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif df[col].min() > np.finfo(np.float32).min and df[col].max() < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
        elif 'int' in str(col_type):
            if df[col].min() > np.iinfo(np.int8).min and df[col].max() < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif df[col].min() > np.iinfo(np.int16).min and df[col].max() < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif df[col].min() > np.iinfo(np.int32).min and df[col].max() < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif df[col].min() > np.iinfo(np.int64).min and df[col].max() < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after: {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)


# Replace inf/-inf with NaN, then fill NaNs (e.g., with 0 or median)
for df in [train_df, test_df]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True) # A common strategy for financial data, but consider alternatives like median or forward/backward fill

# Drop columns with only one unique value in the training set
nunique_cols = train_df.nunique()
cols_to_drop = nunique_cols[nunique_cols == 1].index.tolist()
print(f"Dropping {len(cols_to_drop)} columns with only one unique value: {cols_to_drop}")
train_df.drop(columns=cols_to_drop, inplace=True)

# Ensure the same columns are dropped from the test set (excluding 'label' if it was in cols_to_drop from train)
test_cols_to_drop = [col for col in cols_to_drop if col != 'label'] # 'label' won't be in test features
test_df.drop(columns=test_cols_to_drop, inplace=True)


# Feature Engineering Examples (Highly customizable and dataset-specific)
# These are just examples, you'll need to experiment a lot!

def create_features(df):
    # Basic interactions
    df['bid_ask_spread'] = df['ask_qty'] - df['bid_qty']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-6) # Add small epsilon to avoid division by zero
    df['total_qty'] = df['bid_qty'] + df['ask_qty'] + df['buy_qty'] + df['sell_qty']

    # Example: Simple rolling features on 'volume'
    # For actual implementation, consider the 'timestamp' and proper time-series splits
    # and avoiding future leakage. This is a simplified example.
    df['volume_rolling_mean_5'] = df['volume'].rolling(window=5, min_periods=1).mean()
    df['volume_rolling_std_5'] = df['volume'].rolling(window=5, min_periods=1).std()

    # You could also consider more complex features like:
    # - Lagged values of 'label' (if you were doing multi-step forecasting, but here it's predicting future label)
    # - Statistical features (min, max, median, skew, kurtosis) over rolling windows for various X features.
    # - Fourier Transforms or Wavelet Transforms for extracting cyclical patterns.
    # - Technical indicators (if you map X features to known financial concepts).

    return df

# Apply feature engineering
# Be careful with applying rolling features across the entire dataset if you are doing time-series cross-validation
# For the sake of this example, we apply it directly.
# In a real competition, you'd apply these during the split for proper CV.
print("Creating features for training data...")
train_df = create_features(train_df)
print("Creating features for test data...")
test_df = create_features(test_df)

# Drop original timestamp from features as it's not a direct numerical feature for many models
# Keep it as index if you are doing time-series specific operations
if 'timestamp' in train_df.columns:
    train_df.set_index('timestamp', inplace=True)
if 'timestamp' in test_df.columns:
    test_df.set_index('timestamp', inplace=True)

print(f"Train data shape after feature engineering: {train_df.shape}")
print(f"Test data shape after feature engineering: {test_df.shape}")

# Align columns - crucial if feature engineering creates different columns or if some columns were dropped
train_labels = train_df['label']
train_features = train_df.drop(columns=['label'])
test_features = test_df.drop(columns=['label'], errors='ignore') # 'label' might not exist in test_df

# Get common columns after feature engineering and dropping
common_cols = list(set(train_features.columns) & set(test_features.columns))
train_features = train_features[common_cols]
test_features = test_features[common_cols]

print(f"Train features shape after aligning: {train_features.shape}")
print(f"Test features shape after aligning: {test_features.shape}")

# Clean up memory
del train_df, test_df
gc.collect()


# Define features (X) and target (y)
X = train_features
y = train_labels

# Split data into training and validation sets (using a simple split for quick demonstration)
# For time series, a sequential split is often preferred:
# X_train, X_val, y_train, y_val = X.iloc[:-val_size], X.iloc[-val_size:], y.iloc[:-val_size], y.iloc[-val_size:]
# For this example, we'll use a random split for simplicity.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False) # Keep shuffle=False for time series
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# Initialize and train an XGBoost Regressor
# Parameters can be tuned extensively
model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    n_jobs=-1, # Use all available CPU cores
    tree_method='hist', # Faster for large datasets
    # enable_categorical=True # If you had categorical features and wanted to use XGBoost's native handling
)

print("Training model...")
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=50, # Stop if validation error doesn't improve for 50 rounds
          verbose=False) # Set to True for verbose output during training

print("Model training complete.")


# Predict on the validation set
val_preds = model.predict(X_val)
val_corr, _ = pearsonr(y_val, val_preds)
print(f"Validation Pearson Correlation Coefficient: {val_corr:.4f}")

# Predict on the actual test set
print("Generating predictions on test data...")
test_predictions = model.predict(test_features)

# Create submission file
sample_submission['prediction'] = test_predictions
submission_file_name = 'submission.csv'
sample_submission.to_csv(submission_file_name, index=False)

print(f"Submission file '{submission_file_name}' created successfully.")
print(sample_submission.head())

# Optional: Visualize a small sample of predictions vs actuals (from validation set)
plt.figure(figsize=(12, 6))
plt.plot(y_val.values[:200], label='Actual Label')
plt.plot(val_preds[:200], label='Predicted Label')
plt.title('Validation Predictions vs Actuals (First 200 points)')
plt.xlabel('Time Step')
plt.ylabel('Label Value')
plt.legend()
plt.show()

