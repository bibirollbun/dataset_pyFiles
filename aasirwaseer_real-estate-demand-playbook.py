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


# --- Load the data ---
# Adjust the path to where your 'train' folder is located
data_path = '/kaggle/input/china-real-estate-demand-prediction/' # or '/kaggle/input/real-estate-demand-prediction/'

# Target variable data
df_train = pd.read_csv(os.path.join(data_path, 'train/new_house_transactions.csv'))

# Test data
df_test = pd.read_csv(os.path.join(data_path, 'test.csv'))

# Static sector information
df_poi = pd.read_csv(os.path.join(data_path, 'train/sector_POI.csv'))

# --- Initial Inspection ---
print("--- Training Data Info ---")
df_train.info()
print("\n--- Training Data Head ---")
print(df_train.head())

print("\n\n--- Test Data Info ---")
df_test.info()
print("\n--- Test Data Head ---")
print(df_test.head())

print("\n\n--- Sector POI Info ---")
df_poi.info()
print("\n--- Sector POI Head ---")
print(df_poi.head())


def preprocess_time_and_ids(df, is_train=True):
    """Standardizes month and sector columns."""
    if is_train:
        # Convert 'month' from 'YYYY-Mon' string to datetime object
        df['month'] = pd.to_datetime(df['month'], format='%Y-%b')
    else:
        # Split 'id' into 'month' and 'sector'
        df[['month', 'sector']] = df['id'].str.split('_', expand=True)
        # Convert 'month' from 'YYYY Mon' string to datetime object
        df['month'] = pd.to_datetime(df['month'], format='%Y %b')
    
    # Extract integer from 'sector' column (e.g., 'sector 1' -> 1)
    df['sector'] = df['sector'].str.split(' ').str[1].astype(int)
    
    return df

# Apply the preprocessing
df_train = preprocess_time_and_ids(df_train, is_train=True)
df_test = preprocess_time_and_ids(df_test, is_train=False)
df_poi['sector'] = df_poi['sector'].str.split(' ').str[1].astype(int)


# --- Verification ---
print("--- Cleaned Training Data Head ---")
print(df_train.head())
print("\n--- Cleaned Training Data Info ---")
df_train.info()


print("\n\n--- Cleaned Test Data Head ---")
print(df_test.head())
print("\n--- Cleaned Test Data Info ---")
df_test.info()


from itertools import product

# --- Part A: Create the complete data grid for training ---

# Get all unique months from the training data
unique_months = df_train['month'].unique()
# Get all unique sectors from the POI file (most reliable source for all sectors)
unique_sectors = df_poi['sector'].unique()

# Create a Cartesian product to get all month-sector combinations
data_grid = pd.DataFrame(product(unique_months, unique_sectors), columns=['month', 'sector'])

# Merge the original sparse training data onto this complete grid
# We use a left merge to keep all grid entries
train_df = pd.merge(data_grid, df_train, on=['month', 'sector'], how='left')

# **Crucial Step**: Fill missing target values with 0
# The target column is 'amount_new_house_transactions'
train_df['amount_new_house_transactions'] = train_df['amount_new_house_transactions'].fillna(0)

print(f"Original train shape: {df_train.shape}")
print(f"Expanded train grid shape: {train_df.shape}")


# --- Part B: Load and merge all other data sources ---

# List of all data files to merge
data_files = {
    'land': 'train/land_transactions.csv',
    'land_nearby': 'train/land_transactions_nearby_sectors.csv',
    'new_house_nearby': 'train/new_house_transactions_nearby_sectors.csv',
    'pre_owned': 'train/pre_owned_house_transactions.csv',
    'pre_owned_nearby': 'train/pre_owned_house_transactions_nearby_sectors.csv',
}

# Load all dataframes into a dictionary
dfs = {name: pd.read_csv(os.path.join(data_path, path)) for name, path in data_files.items()}
# Add the POI dataframe we already loaded
dfs['poi'] = df_poi

# Preprocess and merge each file
test_df = df_test.copy()

for name, df in dfs.items():
    if 'month' in df.columns:
        df['month'] = pd.to_datetime(df['month'], format='%Y-%b')
        df['sector'] = df['sector'].str.split(' ').str[1].astype(int)
        
        # Merge onto training and test sets
        train_df = pd.merge(train_df, df, on=['month', 'sector'], how='left', suffixes=('', f'_{name}'))
        test_df = pd.merge(test_df, df, on=['month', 'sector'], how='left', suffixes=('', f'_{name}'))
    else: # This is for the static POI data which has no month
        train_df = pd.merge(train_df, df, on='sector', how='left', suffixes=('', f'_{name}'))
        test_df = pd.merge(test_df, df, on='sector', how='left', suffixes=('', f'_{name}'))

# Combine train and test for easier feature engineering later
# We add a 'source' column to keep track
train_df['source'] = 'train'
test_df['source'] = 'test'
full_df = pd.concat([train_df, test_df], ignore_index=True)


# --- Verification ---
print("\n--- Final Combined DataFrame Info ---")
full_df.info(verbose=False, show_counts=True) # Use verbose=False to keep it tidy
print("\n--- Final Combined DataFrame Shape ---")
print(full_df.shape)


# Make sure data is sorted for time-series operations
full_df = full_df.sort_values(by=['sector', 'month']).reset_index(drop=True)

# --- 1. Time-Based Features ---
full_df['year'] = full_df['month'].dt.year
full_df['month_num'] = full_df['month'].dt.month
full_df['quarter'] = full_df['month'].dt.quarter
full_df['week_of_year'] = full_df['month'].dt.isocalendar().week.astype(int)
full_df['day_of_year'] = full_df['month'].dt.dayofyear

# A continuous time index can also be helpful
full_df['time_idx'] = (full_df['month'].dt.year - full_df['month'].dt.year.min()) * 12 + full_df['month'].dt.month

# --- 2. Lag Features ---
# We compute these per sector to avoid data leakage across sectors
# These are lags of our target variable
lags = [1, 2, 3, 6, 12] # Lag by 1, 2, 3, 6, and 12 months
for lag in lags:
    full_df[f'target_lag_{lag}'] = full_df.groupby('sector')['amount_new_house_transactions'].shift(lag)

# --- 3. Rolling Window Features ---
# We'll use a 3-month and 6-month rolling window
windows = [3, 6, 12]
for window in windows:
    # We shift by 1 so that the rolling window only uses data from the past
    rolling_series = full_df.groupby('sector')['amount_new_house_transactions'].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    full_df[f'target_rolling_mean_{window}'] = rolling_series

# --- Verification ---
print("--- DataFrame Shape After Feature Engineering ---")
print(full_df.shape)

print("\n--- Sample of New Features for Sector 1 ---")
# Displaying relevant columns to check our work
cols_to_show = ['month', 'sector', 'amount_new_house_transactions', 
                'target_lag_1', 'target_lag_2', 'target_rolling_mean_3']
print(full_df[full_df['sector'] == 1][cols_to_show].head(10))


import lightgbm as lgb
from sklearn.model_selection import train_test_split

def custom_metric(y_true, y_pred):
    """
    Calculates the custom two-stage evaluation metric for the competition.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle the case where y_true is 0 to avoid division by zero
    # We can replace 0 with a small number like 1, as APE for 0-predictions is handled differently
    # Or more robustly, calculate APE where y_true is not 0
    mask = y_true != 0
    ape = np.zeros_like(y_true, dtype=float)
    ape[mask] = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])

    # Stage 1: Check for excessive high errors
    if np.mean(ape > 1.0) > 0.3:
        return 0.0

    # Stage 2: Calculate scaled MAPE
    valid_ape = ape[ape <= 1.0]
    fraction_valid = len(valid_ape) / len(y_true)
    
    if fraction_valid == 0: # Avoid division by zero if no predictions are valid
        return 0.0

    mape_valid = np.mean(valid_ape)
    scaled_mape = mape_valid / fraction_valid
    
    score = 1 - scaled_mape
    return score

# --- Prepare data for modeling ---

# Split back into the original train and test sets
train_final = full_df[full_df['source'] == 'train'].copy()
test_final = full_df[full_df['source'] == 'test'].copy()

# For our validation, we'll use a time-based split from the training data
# Train on data before 2024, validate on 2024 data
validation_date = '2024-01-01'
train_set = train_final[train_final['month'] < validation_date]
val_set = train_final[train_final['month'] >= validation_date]

# Define features (X) and target (y)
target = 'amount_new_house_transactions'
# Remove non-feature columns
features = [col for col in train_set.columns if col not in [
    'month', 'id', 'source', target, 'new_house_transaction_amount'
]]
categorical_features = ['sector', 'month_num', 'quarter', 'year']

X_train = train_set[features]
y_train = train_set[target]
X_val = val_set[features]
y_val = val_set[target]

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# --- Train LightGBM Model ---

lgb_params = {
    'objective': 'regression_l1', # MAE is often more robust to outliers than MSE
    'metric': 'mae',
    'n_estimators': 2000,
    'learning_rate': 0.02,
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

model = lgb.LGBMRegressor(**lgb_params)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='mae',
          callbacks=[lgb.early_stopping(100, verbose=False)],
          categorical_feature=categorical_features)

# --- Evaluate on Validation Set ---
val_preds = model.predict(X_val)
# Ensure predictions are non-negative
val_preds[val_preds < 0] = 0

# Calculate our custom score
score = custom_metric(y_val, val_preds)

print(f"\nOur local validation score is: {score:.4f}")


# --- Prepare the final datasets ---
# Full training data
X_full = train_final[features]
y_full = train_final[target]

# Final test data
X_test = test_final[features]

print("Retraining model on the full training dataset...")
# --- Retrain the model on all data ---
# We use the same parameters, but train on all available data
final_model = lgb.LGBMRegressor(**lgb_params)
final_model.fit(X_full, y_full, categorical_feature=categorical_features)

print("Predicting on the test set...")
# --- Make predictions on the test set ---
test_predictions = final_model.predict(X_test)

# Ensure predictions are non-negative, as transaction amounts can't be negative
test_predictions[test_predictions < 0] = 0

# --- Create the submission file ---
submission_df = pd.DataFrame({
    'id': test_final['id'], 
    'new_house_transaction_amount': test_predictions
})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file created successfully!")
print("--- Submission File Head ---")
print(submission_df.head())


import numpy as np
import lightgbm as lgb

# --- Assume previous data setup is complete ---
# train_final, features, categorical_features, etc. are already defined.
# The custom_metric function is also defined.

# --- New Strategy: Redefine the feature list ---
# We will remove the short-term, volatile features that are misleading the model.
original_features = [col for col in train_final.columns if col not in [
    'month', 'id', 'source', 'amount_new_house_transactions', 'new_house_transaction_amount'
]]

features_to_remove = [
    'target_lag_1', 
    'target_lag_2', 
    'target_lag_3', 
    'target_rolling_mean_3',
    'target_rolling_mean_6'
]

# The new, robust feature list
features_robust = [f for f in original_features if f not in features_to_remove]

print(f"Original number of features: {len(original_features)}")
print(f"New robust number of features: {len(features_robust)}")

# --- Re-run training with the robust feature set ---
validation_date = '2024-01-01'
train_set = train_final[train_final['month'] < validation_date]
val_set = train_final[train_final['month'] >= validation_date]

target = 'amount_new_house_transactions'
X_train = train_set[features_robust]
y_train_log = np.log1p(train_set[target])
X_val = val_set[features_robust]
y_val = val_set[target]

print("\nTraining ROBUST model on log-transformed target...")

model_robust = lgb.LGBMRegressor(**lgb_params)
model_robust.fit(X_train, y_train_log,
                 eval_set=[(X_val, np.log1p(y_val))],
                 eval_metric='mae',
                 callbacks=[lgb.early_stopping(100, verbose=False)],
                 categorical_feature=categorical_features)

val_preds_log = model_robust.predict(X_val)
val_preds = np.expm1(val_preds_log)
val_preds[val_preds < 0] = 0

score = custom_metric(y_val, val_preds)

print(f"\nOur local validation score with the ROBUST model is: {score:.4f}")


import numpy as np
import lightgbm as lgb
import pandas as pd

# --- Assume all previous data setup is complete ---
# train_final, test_final, lgb_params, categorical_features are already defined.
# The 'features_robust' list from the previous step is what we'll use.

# --- Prepare the final datasets ---
target = 'amount_new_house_transactions'
X_full = train_final[features_robust]
y_full_log = np.log1p(train_final[target])
X_test = test_final[features_robust]

print("Retraining ROBUST model on the full training dataset...")

# --- Retrain the model on all data ---
# We need to ensure the categorical features we pass are present in our robust feature list
valid_cats = [f for f in categorical_features if f in features_robust]

final_model_robust = lgb.LGBMRegressor(**lgb_params)
final_model_robust.fit(X_full, y_full_log, categorical_feature=valid_cats)

print("Predicting on the test set with the robust model...")

# --- Make final predictions ---
test_predictions_log = final_model_robust.predict(X_test)
test_predictions = np.expm1(test_predictions_log)
test_predictions[test_predictions < 0] = 0

# --- Create the submission file ---
submission_df_v3 = pd.DataFrame({
    'id': test_final['id'],
    'new_house_transaction_amount': test_predictions
})

# Save to a new file
submission_df_v3.to_csv('submission_v3_robust.csv', index=False)

print("\nNew submission file 'submission_v3_robust.csv' created successfully!")
print("--- Submission File Head ---")
print(submission_df_v3.head())


import numpy as np
import lightgbm as lgb
import pandas as pd

# --- Step 1: Add new features to the main dataframe ---
print("Adding new lag/rolling features from other variables...")
feature_eng_vars = [
    'price_pre_owned_house_transactions',
    'num_new_house_available_for_sale',
    'amount_new_house_transactions_nearby_sectors',
    'price_new_house_transactions_nearby_sectors'
]
lags = [1, 2, 3]
windows = [3, 6]

for var in feature_eng_vars:
    full_df[var] = full_df.groupby('sector')[var].transform(lambda x: x.ffill().bfill())
    for lag in lags:
        full_df[f'{var}_lag_{lag}'] = full_df.groupby('sector')[var].shift(lag)
    for window in windows:
        rolling_series = full_df.groupby('sector')[var].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        full_df[f'{var}_rolling_mean_{window}'] = rolling_series

print("New features added successfully!")

# --- Step 2: Re-split the data AFTER adding all features ---
train_final_expanded = full_df[full_df['source'] == 'train'].copy()
validation_date = '2024-01-01'
train_set = train_final_expanded[train_final_expanded['month'] < validation_date]
val_set = train_final_expanded[train_final_expanded['month'] >= validation_date]

# --- Step 3: Define the final, expanded feature list ---
features_to_remove = [
    'target_lag_1', 'target_lag_2', 'target_lag_3', 
    'target_rolling_mean_3', 'target_rolling_mean_6'
]
all_features = [col for col in full_df.columns if col not in [
    'month', 'id', 'source', 'amount_new_house_transactions', 'new_house_transaction_amount'
]]
final_features_expanded = [f for f in all_features if f not in features_to_remove]
print(f"\nTotal features for new model: {len(final_features_expanded)}")

# --- Step 4: Define X and y for training/validation ---
target = 'amount_new_house_transactions'
X_train = train_set[final_features_expanded]
y_train_log = np.log1p(train_set[target])
X_val = val_set[final_features_expanded]
y_val = val_set[target]

# --- Step 5: Re-validate with the new feature set ---
print("\nRe-validating with new expanded feature set...")

# Use existing lgb_params
model_expanded = lgb.LGBMRegressor(**lgb_params)
model_expanded.fit(X_train, y_train_log,
                   eval_set=[(X_val, np.log1p(y_val))],
                   eval_metric='mae',
                   callbacks=[lgb.early_stopping(100, verbose=False)],
                   categorical_feature=categorical_features)

val_preds_log = model_expanded.predict(X_val)
val_preds = np.expm1(val_preds_log)
val_preds[val_preds < 0] = 0
score = custom_metric(y_val, val_preds)

print(f"\nOur NEW local validation score with expanded features is: {score:.4f}")


import numpy as np
import lightgbm as lgb
import pandas as pd

# --- Assume all previous data setup is complete ---
# full_df is the latest version with all features.
# lgb_params and categorical_features are defined.
# final_features_expanded is our new, complete feature list.

# --- Prepare the final datasets ---
train_final_expanded = full_df[full_df['source'] == 'train'].copy()
test_final_expanded = full_df[full_df['source'] == 'test'].copy()

target = 'amount_new_house_transactions'
X_full = train_final_expanded[final_features_expanded]
y_full_log = np.log1p(train_final_expanded[target])
X_test = test_final_expanded[final_features_expanded]

print("Retraining EXPANDED model on the full training dataset...")

# --- Retrain the model on all data ---
valid_cats = [f for f in categorical_features if f in final_features_expanded]
final_model_expanded = lgb.LGBMRegressor(**lgb_params)
final_model_expanded.fit(X_full, y_full_log, categorical_feature=valid_cats)

print("Predicting on the test set with the expanded model...")

# --- Make final predictions ---
test_predictions_log = final_model_expanded.predict(X_test)
test_predictions = np.expm1(test_predictions_log)
test_predictions[test_predictions < 0] = 0

# --- Create the submission file ---
submission_df_v4 = pd.DataFrame({
    'id': test_final_expanded['id'],
    'new_house_transaction_amount': test_predictions
})

# Save to a new file
submission_df_v4.to_csv('submission_v4_expanded.csv', index=False)

print("\nNew submission file 'submission_v4_expanded.csv' created successfully!")
print("--- Submission File Head ---")
print(submission_df_v4.head())


import optuna
import lightgbm as lgb
import numpy as np

# This ensures categorical features are handled correctly within Optuna
valid_cats = [f for f in categorical_features if f in final_features_expanded]

# Define the objective function for Optuna to optimize
def objective(trial):
    # 1. Define the hyperparameter search space
    params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'n_estimators': 2000, # Use a fixed large number with early stopping
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 20, 80),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
    }

    # 2. Use the same validation setup
    # (Assuming train_set, val_set, final_features_expanded, etc. are already defined)
    X_train_tune = train_set[final_features_expanded]
    y_train_tune_log = np.log1p(train_set[target])
    X_val_tune = val_set[final_features_expanded]
    y_val_tune = val_set[target]

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train_tune, y_train_tune_log,
              eval_set=[(X_val_tune, np.log1p(y_val_tune))],
              eval_metric='mae',
              callbacks=[lgb.early_stopping(75, verbose=False)],
              categorical_feature=valid_cats)

    preds_log = model.predict(X_val_tune)
    preds = np.expm1(preds_log)
    preds[preds < 0] = 0
    
    score = custom_metric(y_val_tune, preds)
    
    return score

# 3. Run the optimization study
print("Starting hyperparameter optimization with Optuna...")
study = optuna.create_study(direction='maximize')
# We'll run 50 trials to find a good set of parameters
study.optimize(objective, n_trials=50)

# 4. Print the best results
print("\nOptimization finished!")
print(f"Number of finished trials: {len(study.trials)}")
print("\n--- Best Trial ---")
best_trial = study.best_trial
print(f"Validation Score: {best_trial.value:.4f}")
print("Optimal Parameters:")
for key, value in best_trial.params.items():
    print(f"  '{key}': {value},")


import numpy as np
import lightgbm as lgb
import pandas as pd

# --- 1. Define our new, optimized parameters from Optuna ---
best_params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
    # --- Parameters found by Optuna ---
    'learning_rate': 0.010100783568542247,
    'num_leaves': 42,
    'feature_fraction': 0.8177556101935218,
    'bagging_fraction': 0.737487385054064,
    'bagging_freq': 2,
    'lambda_l1': 0.011447615499635019,
    'lambda_l2': 0.0009150667602618044,
    'min_child_samples': 10,
}

# --- 2. Prepare the final datasets (as before) ---
# Assuming full_df, final_features_expanded, categorical_features are all set
train_final_expanded = full_df[full_df['source'] == 'train'].copy()
test_final_expanded = full_df[full_df['source'] == 'test'].copy()

target = 'amount_new_house_transactions'
X_full = train_final_expanded[final_features_expanded]
y_full_log = np.log1p(train_final_expanded[target])
X_test = test_final_expanded[final_features_expanded]

print("Retraining final TUNED model on the full training dataset...")

# --- 3. Retrain the model with the best parameters ---
valid_cats = [f for f in categorical_features if f in final_features_expanded]
# We'll set a high number of estimators for the final training
final_model_tuned = lgb.LGBMRegressor(**best_params, n_estimators=2500)
final_model_tuned.fit(X_full, y_full_log, categorical_feature=valid_cats)

print("Predicting on the test set with the tuned model...")

# --- 4. Make final predictions ---
test_predictions_log = final_model_tuned.predict(X_test)
test_predictions = np.expm1(test_predictions_log)
test_predictions[test_predictions < 0] = 0

# --- 5. Create the submission file ---
submission_df_v5 = pd.DataFrame({
    'id': test_final_expanded['id'],
    'new_house_transaction_amount': test_predictions
})

# Save to a new file
submission_df_v5.to_csv('submission_v5_tuned.csv', index=False)

print("\nFinal submission file 'submission_v5_tuned.csv' created successfully!")
print("--- Submission File Head ---")
print(submission_df_v5.head())


import xgboost as xgb
import numpy as np
import pandas as pd

# --- Ensure you have XGBoost installed ---
# If not, run: pip install xgboost

# --- Use the same robust setup as our best model ---
# Assuming all data (full_df) and feature lists (final_features_expanded) are ready

# 1. Prepare the data
train_final_expanded = full_df[full_df['source'] == 'train'].copy()
validation_date = '2024-01-01'
train_set = train_final_expanded[train_final_expanded['month'] < validation_date]
val_set = train_final_expanded[train_final_expanded['month'] >= validation_date]

target = 'amount_new_house_transactions'
X_train = train_set[final_features_expanded]
y_train_log = np.log1p(train_set[target])
X_val = val_set[final_features_expanded]
y_val = val_set[target]

# 2. Define XGBoost parameters (sensible defaults)
xgb_params = {
    'objective': 'reg:squarederror', # XGBoost's equivalent for regression
    'eval_metric': 'mae',
    'eta': 0.02, # learning_rate
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'n_jobs': -1
}

# 3. Train the XGBoost model
print("Training a baseline XGBoost model...")
model_xgb = xgb.XGBRegressor(**xgb_params, n_estimators=2000)

# XGBoost requires a slightly different early stopping setup
model_xgb.fit(X_train, y_train_log,
              eval_set=[(X_val, np.log1p(y_val))],
              early_stopping_rounds=75,
              verbose=False)

# 4. Evaluate the model
print("Evaluating the XGBoost model...")
val_preds_log_xgb = model_xgb.predict(X_val)
val_preds_xgb = np.expm1(val_preds_log_xgb)
val_preds_xgb[val_preds_xgb < 0] = 0

score_xgb = custom_metric(y_val, val_preds_xgb)

print(f"\nOur XGBoost model's local validation score is: {score_xgb:.4f}")


import lightgbm as lgb
import numpy as np

# --- Assume all data (train_set, val_set, etc.) and val_preds_xgb are ready ---

# 1. Re-train our best tuned LightGBM model to get its validation predictions
print("Generating predictions from the tuned LightGBM model...")
# These are the optimal parameters we found with Optuna
best_lgb_params = {
    'objective': 'regression_l1', 'metric': 'mae', 'verbose': -1,
    'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt',
    'learning_rate': 0.010100783568542247, 'num_leaves': 42,
    'feature_fraction': 0.8177556101935218, 'bagging_fraction': 0.737487385054064,
    'bagging_freq': 2, 'lambda_l1': 0.011447615499635019,
    'lambda_l2': 0.0009150667602618044, 'min_child_samples': 10,
}
valid_cats = [f for f in categorical_features if f in final_features_expanded]
model_lgb = lgb.LGBMRegressor(**best_lgb_params, n_estimators=2000)

model_lgb.fit(X_train, y_train_log,
              eval_set=[(X_val, np.log1p(y_val))],
              eval_metric='mae',
              callbacks=[lgb.early_stopping(75, verbose=False)],
              categorical_feature=valid_cats)

val_preds_log_lgb = model_lgb.predict(X_val)
val_preds_lgb = np.expm1(val_preds_log_lgb)
val_preds_lgb[val_preds_lgb < 0] = 0

# 2. Create the weighted average ensemble
print("\nCreating the ensemble...")
weight_xgb = 0.70
weight_lgb = 0.30

# Combine the predictions from both models
ensemble_preds = (val_preds_xgb * weight_xgb) + (val_preds_lgb * weight_lgb)

# 3. Score the ensemble
ensemble_score = custom_metric(y_val, ensemble_preds)

print("\n--- Validation Score Summary ---")
print(f"Tuned LightGBM Score: {custom_metric(y_val, val_preds_lgb):.4f}")
print(f"Baseline XGBoost Score: {score_xgb:.4f}")
print(f"Ensemble (70/30) Score: {ensemble_score:.4f}")


import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd

# --- Assume all data, feature lists, and parameters are defined from previous steps ---
# (final_features_expanded, best_lgb_params, xgb_params, etc.)

print("Training final models on the full dataset...")

# --- 1. Prepare the full dataset ---
train_final_expanded = full_df[full_df['source'] == 'train'].copy()
test_final_expanded = full_df[full_df['source'] == 'test'].copy()
X_full = train_final_expanded[final_features_expanded]
y_full_log = np.log1p(train_final_expanded[target])
X_test = test_final_expanded[final_features_expanded]

# --- 2. Train the final LightGBM model ---
print("Training final LightGBM...")
valid_cats = [f for f in categorical_features if f in final_features_expanded]
final_model_lgb = lgb.LGBMRegressor(**best_lgb_params, n_estimators=2500)
final_model_lgb.fit(X_full, y_full_log, categorical_feature=valid_cats)
test_preds_log_lgb = final_model_lgb.predict(X_test)
test_preds_lgb = np.expm1(test_preds_log_lgb)
test_preds_lgb[test_preds_lgb < 0] = 0

# --- 3. Train the final XGBoost model ---
print("Training final XGBoost...")
final_model_xgb = xgb.XGBRegressor(**xgb_params, n_estimators=2500)
final_model_xgb.fit(X_full, y_full_log, verbose=False)
test_preds_log_xgb = final_model_xgb.predict(X_test)
test_preds_xgb = np.expm1(test_preds_log_xgb)
test_preds_xgb[test_preds_xgb < 0] = 0

# --- 4. Create the final ensemble prediction for the test set ---
print("Creating the final ensemble predictions...")
weight_xgb = 0.70
weight_lgb = 0.30
final_ensemble_preds = (test_preds_xgb * weight_xgb) + (test_preds_lgb * weight_lgb)

# --- 5. Create the submission file ---
submission_df_final = pd.DataFrame({
    'id': test_final_expanded['id'],
    'new_house_transaction_amount': final_ensemble_preds
})

# Save the final submission file
submission_df_final.to_csv('submission_v6_ensemble.csv', index=False)

print("\nFinal ensemble submission file 'submission_v6_ensemble.csv' created successfully!")
print("--- Submission File Head ---")
print(submission_df_final.head())

