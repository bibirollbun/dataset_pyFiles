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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import gc # Import garbage collection module

# Function to reduce memory usage by downcasting numerical columns
def reduce_mem_usage(df, verbose=True):
    """
    Iterates through all numerical columns of a dataframe and modifies the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage of dataframe is {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object: # Only process numerical columns
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
            else: # float
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64) # Keep as float64 if it doesn't fit in float32

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage after optimization is: {end_mem:.2f} MB')
        print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    return df

# Load and explore data
# Using dummy paths for demonstration; replace with actual paths if running locally
try:
    train_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
    test_df = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
except FileNotFoundError:
    print("Dataset files not found. Creating dummy dataframes for demonstration.")
    # Create dummy dataframes if files are not found (for local testing without the dataset)
    n_train_rows = 10000 # Increased dummy data size for better memory testing
    n_test_rows = 5000
    num_features = 890

    data_train = {f'X{i}': np.random.rand(n_train_rows) for i in range(1, num_features + 1)}
    data_train.update({
        'timestamp': pd.to_datetime(pd.date_range(start='2023-03-01', periods=n_train_rows, freq='min')),
        'bid_qty': np.random.rand(n_train_rows) * 100,
        'ask_qty': np.random.rand(n_train_rows) * 100,
        'buy_qty': np.random.rand(n_train_rows) * 200,
        'sell_qty': np.random.rand(n_train_rows) * 200,
        'volume': np.random.rand(n_train_rows) * 500,
        'label': np.random.randn(n_train_rows)
    })
    train_df = pd.DataFrame(data_train).set_index('timestamp')

    data_test = {f'X{i}': np.random.rand(n_test_rows) for i in range(1, num_features + 1)}
    data_test.update({
        'timestamp': pd.to_datetime(pd.date_range(start=train_df.index[-1] + pd.Timedelta(minutes=1), periods=n_test_rows, freq='min')),
        'bid_qty': np.random.rand(n_test_rows) * 100,
        'ask_qty': np.random.rand(n_test_rows) * 100,
        'buy_qty': np.random.rand(n_test_rows) * 200,
        'sell_qty': np.random.rand(n_test_rows) * 200,
        'volume': np.random.rand(n_test_rows) * 500,
        'label': np.random.randn(n_test_rows) # Test set might have a 'label' column, but it's usually not used for prediction
    })
    test_df = pd.DataFrame(data_test).set_index('timestamp')

    # Introduce some NaNs and infs for testing the imputation logic
    train_df.iloc[10:20, 5] = np.nan
    train_df.iloc[30, 10] = np.inf
    train_df.iloc[40, 15] = -np.inf
    train_df['all_nan_col'] = np.nan # Introduce an entirely NaN column to simulate the error condition
    test_df['all_nan_col'] = np.nan # Ensure test_df also has this column for consistency


# Apply memory reduction
print("Optimizing train_df memory usage...")
train_df = reduce_mem_usage(train_df)
print("Optimizing test_df memory usage...")
test_df = reduce_mem_usage(test_df)
gc.collect() # Explicitly free up memory after loading and optimizing

# Make column names unique for train_df
if not train_df.columns.is_unique:
    train_df.columns = [f"{col}_{i}" if train_df.columns.duplicated()[i] else col
                        for i, col in enumerate(train_df.columns)]
    print("Duplicate columns in train_df renamed.")

# Make column names unique for test_df
if not test_df.columns.is_unique:
    test_df.columns = [f"{col}_{i}" if test_df.columns.duplicated()[i] else col
                       for i, col in enumerate(test_df.columns)]
    print("Duplicate columns in test_df renamed.")

# Basic data exploration
print("\nData Head:\n", train_df.head())
print("Missing Values (before inf handling):\n", train_df.isnull().sum().head()) # Print head to avoid excessive output
print("Label Distribution:\n", train_df['label'].describe())

# Identify numerical columns for handling inf values and imputation
# This needs to be done carefully to ensure 'timestamp' and 'label' are excluded
# and that the same set of columns is used consistently.
all_numerical_cols = train_df.select_dtypes(include=[np.number]).columns

# Handle infinite values in numerical columns for both train and test
# This step is still necessary even after downcasting, as inf/nan are distinct concepts.
train_df[all_numerical_cols] = train_df[all_numerical_cols].replace([np.inf, -np.inf], np.nan)
test_df[all_numerical_cols] = test_df[all_numerical_cols].replace([np.inf, -np.inf], np.nan)
gc.collect() # Collect garbage after replacing inf values

# Check target 'label' for issues
print("Missing values in label:", train_df['label'].isnull().sum())
print("Infinite values in label:", np.isinf(train_df['label']).sum())

# Define initial feature columns (excluding 'timestamp' and 'label')
# Note: 'timestamp' is the index, so it won't be in columns unless reset_index() is called.
# We explicitly exclude 'label' here.
initial_feature_cols = [col for col in all_numerical_cols if col not in ['label']]

# Identify columns that are entirely NaN after handling inf values in the initial feature set.
# These are the columns that SimpleImputer will drop if strategy is 'mean', 'median', 'most_frequent'.
all_nan_in_features = train_df[initial_feature_cols].columns[train_df[initial_feature_cols].isnull().all()].tolist()
print(f"Columns that are entirely NaN and will be dropped by imputer: {all_nan_in_features}")

# Filter feature_cols to exclude these all-NaN columns.
# This list will be the actual columns processed by the imputer and used for training.
imputer_feature_cols = [col for col in initial_feature_cols if col not in all_nan_in_features]

# Debugging: Check shapes before imputation
print("Number of initial_feature_cols:", len(initial_feature_cols))
print("Number of imputer_feature_cols (after dropping all-NaNs):", len(imputer_feature_cols))
print("Shape of train_df[imputer_feature_cols]:", train_df[imputer_feature_cols].shape)

# Preprocessing: Impute missing values in feature columns
imputer = SimpleImputer(strategy='mean')
imputed_array = imputer.fit_transform(train_df[imputer_feature_cols])

# Debugging: Check shape after imputation
print("Shape of imputed_array:", imputed_array.shape)

# Assign imputed values back to DataFrame using the filtered list of columns
train_df[imputer_feature_cols] = imputed_array
gc.collect() # Collect garbage after imputation

# Feature engineering for train_df
train_df['bid_ask_spread'] = train_df['ask_qty'] - train_df['bid_qty']
train_df['buy_sell_ratio'] = train_df['buy_qty'] / (train_df['sell_qty'] + 1e-6)

# Define the final set of features for training (X)
# This includes the imputed numerical features and the newly engineered features.
final_train_features = imputer_feature_cols + ['bid_ask_spread', 'buy_sell_ratio']

# Split features and target
X = train_df[final_train_features]
y = train_df['label']

# Delete the original train_df to free up memory before training the model
del train_df
gc.collect()

# Training and validation with TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    print(f"\n--- Fold {fold+1} ---")
    print(f"Train data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

    # Train LightGBM with GPU support
    # Added 'device="gpu"' to enable GPU training
    model = LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42, n_jobs=-1, device="gpu")
    model.fit(X_train, y_train)
    
    # Validate
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"Validation MSE: {mse:.4f}")

    # Delete fold-specific data to free memory
    del X_train, X_val, y_train, y_val
    gc.collect()

# Train on full data for final model
print("\nTraining final model on full dataset...")
# Ensure GPU is used for the final training as well
model = LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42, n_jobs=-1, device="gpu")
model.fit(X, y)
print("Final model trained.")

# Delete X and y to free up memory before processing test set
del X, y
gc.collect()

# Process test set - apply the same preprocessing steps as train_df
# Note: all_numerical_cols for test_df was already handled above.

# Apply imputation to test_df using the *fitted* imputer and the same feature columns
# The imputer was fitted on imputer_feature_cols from train_df.
test_df[imputer_feature_cols] = imputer.transform(test_df[imputer_feature_cols])
gc.collect() # Collect garbage after test imputation

# Feature engineering for test_df
test_df['bid_ask_spread'] = test_df['ask_qty'] - test_df['bid_qty']
test_df['buy_sell_ratio'] = test_df['buy_qty'] / (test_df['sell_qty'] + 1e-6)

# Prepare X_test with the exact same columns as X (final_train_features)
X_test = test_df[final_train_features]

# Predict on test set
print("Predicting on test set...")
y_test_pred = model.predict(X_test)
print("Prediction complete.")

# Create submission
# Assuming 'timestamp' is the index in test_df and needs to be a column in submission.
submission = pd.DataFrame({'id': test_df.index, 'label': y_test_pred})
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

# Delete test_df and X_test to free up memory
del test_df, X_test
gc.collect()



################ create changes in new cells  #############




