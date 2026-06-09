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
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import numpy as np
import optuna
import datetime




# --- 1. Load Data ---
print(datetime.datetime.now(), "1. Loading data...")
try:
    train_df = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/train.parquet')
    test_df = pd.read_parquet('/kaggle/input/aeroclub-recsys-2025/test.parquet')
except FileNotFoundError:
    print("Error: train.parquet or test.parquet not found. Please download them from Kaggle.")
    exit()

print(f"{datetime.datetime.now()} >Training data shape: {train_df.shape}")
print(f"{datetime.datetime.now()} >Test data shape: {test_df.shape}")

# --- Speed Optimization: Downcast data types to save memory and speed up operations ---
def downcast_dtypes(df):
    """
    Downcasts numerical columns to smaller types (e.g., float64 to float32).
    This can significantly reduce memory usage and speed up computations.
    """
    for col in df.columns:
        # Downcast floats
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        # Downcast integers
        elif df[col].dtype == 'int64':
            # Only downcast if it fits without overflow
            if df[col].max() < np.iinfo(np.int32).max and df[col].min() > np.iinfo(np.int32).min:
                df[col] = df[col].astype('int32')
            elif df[col].max() < np.iinfo(np.int16).max and df[col].min() > np.iinfo(np.int16).min:
                df[col] = df[col].astype('int16')
    return df

print(f"{datetime.datetime.now()} >Optimizing data types for faster processing...")
train_df = downcast_dtypes(train_df)
test_df = downcast_dtypes(test_df)
print(f"{datetime.datetime.now()} >Downcast data types completed...")

# The 'ranker_id' is the crucial grouping variable for this competition.
group_column = 'ranker_id'

# We store the target variable and the groups from the original train_df
# before dropping the 'selected' column for feature engineering.
y_train = train_df['selected']
groups = train_df[group_column]
train_len = len(train_df)

# Drop the target variable from the training DataFrame to prepare for concatenation
train_df = train_df.drop('selected', axis=1)
print(f"{datetime.datetime.now()} >Dropped target variable from training dataframe...")




# --- 2. Feature Engineering & Preprocessing ---
print(f"\n{datetime.datetime.now()} 2. Starting feature engineering...")

# Combine data for consistent feature engineering across train and test sets
all_data = pd.concat([train_df, test_df], axis=0)
print(f"{datetime.datetime.now()} >Combined train and test dataframes...")

# Fill missing numerical values with a placeholder.
numerical_features = [
    'price', 'duration_in_minutes', 'legs0_duration_in_minutes',
    'legs1_duration_in_minutes', 'legs0_segments0_duration',
    'legs0_segments1_duration', 'legs1_segments0_duration',
    'legs1_segments1_duration', 'miniRules0_monetaryAmount',
    'miniRules0_percentage', 'miniRules1_monetaryAmount',
    'miniRules1_percentage'
]

# Filter to only existing columns
numerical_features = [col for col in numerical_features if col in all_data.columns]

# Convert columns to numeric before filling missing values
for col in numerical_features:
    # Coerce errors will turn non-numeric values into NaN
    all_data[col] = pd.to_numeric(all_data[col], errors='coerce')

for col in numerical_features:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(-1)

print(f"{datetime.datetime.now()} >Converted columns to numeric before filling missing values...")

# --- NEW FEATURES (Optimized for speed) ---
# Create a temporary DataFrame to hold all new features before merging
new_features_df = all_data[[group_column]].copy()
print(f"{datetime.datetime.now()} >Created temp copy...")

# Dynamically select columns for group stats
groupby_cols = [col for col in ['price', 'duration_in_minutes'] if col in all_data.columns]
if groupby_cols:
    group_stats = all_data.groupby(group_column)[groupby_cols].agg(['min', 'max', 'mean', 'std']).reset_index()
    group_stats.columns = [f'{col[0]}_{col[1]}_by_group' if col[1] != '' else col[0] for col in group_stats.columns]
    new_features_df = new_features_df.merge(group_stats, on=group_column, how='left')

    # Create relative features only if the base columns exist
    if 'price' in all_data.columns and 'price_mean_by_group' in new_features_df.columns:
        new_features_df['price_relative_to_mean'] = all_data['price'] - new_features_df['price_mean_by_group']
    if 'price' in all_data.columns and 'price_max_by_group' in new_features_df.columns:
        new_features_df['price_normalized_by_max'] = all_data['price'] / new_features_df['price_max_by_group']
    if 'duration_in_minutes' in all_data.columns and 'duration_in_minutes_mean_by_group' in new_features_df.columns:
        new_features_df['duration_relative_to_mean'] = all_data['duration_in_minutes'] - new_features_df['duration_in_minutes_mean_by_group']
    if 'duration_in_minutes' in all_data.columns and 'duration_in_minutes_max_by_group' in new_features_df.columns:
        new_features_df['duration_normalized_by_max'] = all_data['duration_in_minutes'] / new_features_df['duration_in_minutes_max_by_group']

# Dynamically create rank features
if 'price' in all_data.columns:
    new_features_df['price_rank_in_group'] = all_data.groupby(group_column)['price'].rank(ascending=True, method='dense')
if 'duration_in_minutes' in all_data.columns:
    new_features_df['duration_rank_in_group'] = all_data.groupby(group_column)['duration_in_minutes'].rank(ascending=True, method='dense')

# Features from nested data (number of segments/legs)
if 'legs1_duration_in_minutes' in all_data.columns:
    new_features_df['num_legs'] = all_data['legs1_duration_in_minutes'].apply(lambda x: 2 if x > -1 else 1)
if 'legs0_segments1_duration' in all_data.columns:
    new_features_df['num_segments_leg0'] = all_data['legs0_segments1_duration'].apply(lambda x: 2 if x > -1 else 1)
if 'legs1_segments1_duration' in all_data.columns:
    new_features_df['num_segments_leg1'] = all_data['legs1_segments1_duration'].apply(lambda x: 2 if x > -1 else 1)

# Feature: Number of options in the group
group_size = all_data.groupby(group_column).size().reset_index(name='group_size')
new_features_df = new_features_df.merge(group_size, on=group_column, how='left')

# Feature: Time-based features from 'requestDate' (assuming it exists)
if 'requestDate' in all_data.columns:
    all_data['requestDate'] = pd.to_datetime(all_data['requestDate'])
    new_features_df['request_day_of_week'] = all_data['requestDate'].dt.dayofweek
    new_features_df['request_hour'] = all_data['requestDate'].dt.hour
    all_data.drop('requestDate', axis=1, inplace=True)

# Merge all new features into the main DataFrame at once
all_data = pd.concat([all_data, new_features_df.drop(group_column, axis=1)], axis=1)
print(f"{datetime.datetime.now()} >Merged all features into main dataframe...")

# Identify all categorical columns to convert to the 'category' dtype
categorical_features = [
    'origin_city_iata', 'destination_city_iata', 'frequentFlyer',
    'profileId', 'companyID',
    'legs0_segments0_aircraft_code', 'legs0_segments0_arrivalTo_airport_city_iata',
    'legs0_segments0_arrivalTo_airport_iata', 'legs0_segments0_departureFrom_airport_iata',
    'legs0_segments0_flightNumber', 'legs0_segments0_marketingCarrier_code',
    'legs0_segments0_operatingCarrier_code', 'legs0_segments1_aircraft_code',
    'legs0_segments1_arrivalTo_airport_city_iata', 'legs0_segments1_arrivalTo_airport_iata',
    'legs0_segments1_departureFrom_airport_iata', 'legs0_segments1_flightNumber',
    'legs0_segments1_marketingCarrier_code', 'legs0_segments1_operatingCarrier_code',
    'legs0_segments2_aircraft_code', 'legs0_segments2_arrivalTo_airport_city_iata',
    'legs0_segments2_arrivalTo_airport_iata', 'legs0_segments2_departureFrom_airport_iata',
    'legs0_segments2_flightNumber', 'legs0_segments2_marketingCarrier_code',
    'legs0_segments2_operatingCarrier_code', 'legs0_segments3_aircraft_code',
    'legs0_segments3_arrivalTo_airport_city_iata', 'legs0_segments3_arrivalTo_airport_iata',
    'legs0_segments3_departureFrom_airport_iata', 'legs0_segments3_flightNumber',
    'legs0_segments3_marketingCarrier_code', 'legs0_segments3_operatingCarrier_code',
    'legs1_segments0_aircraft_code', 'legs1_segments0_arrivalTo_airport_city_iata',
    'legs1_segments0_arrivalTo_airport_iata', 'legs1_segments0_departureFrom_airport_iata',
    'legs1_segments0_flightNumber', 'legs1_segments0_marketingCarrier_code',
    'legs1_segments0_operatingCarrier_code', 'legs1_segments1_aircraft_code',
    'legs1_segments1_arrivalTo_airport_city_iata', 'legs1_segments1_arrivalTo_airport_iata',
    'legs1_segments1_departureFrom_airport_iata', 'legs1_segments1_flightNumber',
    'legs1_segments1_marketingCarrier_code', 'legs1_segments1_operatingCarrier_code',
    'legs1_segments2_aircraft_code', 'legs1_segments2_arrivalTo_airport_city_iata',
    'legs1_segments2_arrivalTo_airport_iata', 'legs1_segments2_departureFrom_airport_iata',
    'legs1_segments2_flightNumber', 'legs1_segments2_marketingCarrier_code',
    'legs1_segments2_operatingCarrier_code', 'legs1_segments3_aircraft_code',
    'legs1_segments3_arrivalTo_airport_city_iata', 'legs1_segments3_arrivalTo_airport_iata',
    'legs1_segments3_departureFrom_airport_iata', 'legs1_segments3_flightNumber',
    'legs1_segments3_marketingCarrier_code', 'legs1_segments3_operatingCarrier_code',
    'searchRoute'
]

# Fill missing values in categorical columns with a placeholder string.
print(f"{datetime.datetime.now()} >Filling missing values in categorical features...")
for col in categorical_features:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna('missing')
print(f"{datetime.datetime.now()} >Missing values in categorical features filled.")

# Get a list of columns to drop, including non-numeric columns that should not be used as features, and 'Id' and the group column itself.
# We keep only the numerical features and the columns we've explicitly defined as categorical.
# This ensures no stray 'object' columns are passed to the model.
columns_to_keep = set(numerical_features + list(new_features_df.columns) + categorical_features) - set(['ranker_id'])
columns_to_drop = [col for col in all_data.columns if col not in columns_to_keep]

# Convert the categorical columns to the 'category' dtype
for col in categorical_features:
    if col in all_data.columns:
        all_data[col] = all_data[col].astype('category')
print(f"{datetime.datetime.now()} >Converted cat columns to 'category' dtype...")

# Drop the non-feature columns
all_data.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')

# List of all features to use
features = [col for col in all_data.columns if col not in ['Id', 'ranker_id']]

# Separate data back into train and test
X_train = all_data[:train_len][features].copy()
X_test = all_data[train_len:][features].copy()

# Filter categorical_features list to only include columns present in X_train
categorical_features_final = [col for col in categorical_features if col in X_train.columns]

print(f"{datetime.datetime.now()} >Preprocessing complete. Starting hyperparameter tuning with Optuna...\n")




# --- 3. Define the Optuna Objective Function and Metric ---
def hit_rate_at_3(y_true, y_pred, groups):
    """
    Calculates the HitRate@3 metric.
    """
    df = pd.DataFrame({'true': y_true, 'pred': y_pred, 'group': groups})

    # Filter for groups with more than 10 options, as per competition rules
    group_counts = df.groupby('group')['group'].transform('count')
    valid_df = df[group_counts > 10].copy()

    # Get the rank of the true selection within each group
    # We only need the rank for the true selections, not for all predictions.
    true_selections = valid_df[valid_df['true'] == 1].copy()
    
    # Calculate rank for all predictions in valid groups
    valid_df['rank'] = valid_df.groupby('group')['pred'].rank(method='first', ascending=False)
    
    # Check if the true selections' rank is in the top 3
    hits = valid_df[(valid_df['true'] == 1) & (valid_df['rank'] <= 3)]
    
    return len(hits) / len(true_selections) if len(true_selections) > 0 else 0

def objective(trial):
    """Objective function for Optuna to optimize."""
    lgb_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'n_estimators': trial.suggest_int('n_estimators', 50, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 8, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }

    N_FOLDS = 5
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_preds = np.zeros(len(X_train))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups)):
        X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
        groups_train_fold = groups.iloc[train_idx]
        
        X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]
        groups_val_fold = groups.iloc[val_idx]

        group_counts_train = groups_train_fold.value_counts(sort=False).values
        
        model = lgb.LGBMRanker(**lgb_params)
        
        model.fit(X_train_fold, y_train_fold, group=group_counts_train,
                  eval_set=[(X_val_fold, y_val_fold)],
                  eval_group=[groups_val_fold.value_counts(sort=False).values],
                  eval_metric='ndcg',
                  callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
                  categorical_feature=categorical_features_final)
        
        val_preds = model.predict(X_val_fold)
        oof_preds[val_idx] = val_preds

    # Calculate overall HitRate@3 on the full out-of-fold predictions
    overall_hr3 = hit_rate_at_3(y_train, oof_preds, groups)
    return overall_hr3

# --- 4. Run the Optuna Study ---
# Optimize for 50 trials. Can increase this for a more thorough search.
N_TRIALS = 100
print(f"{datetime.datetime.now()} 3.& 4. Starting Optuna search for {N_TRIALS} trials...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS)

print(f"\n{datetime.datetime.now()}--- Optuna Tuning Complete ---")
print(f"Best Trial Score (Overall HitRate@3): {study.best_value:.4f}")
print("Best Hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")



# --- 5. Train the Final Model with Best Parameters ---
print(f"\n{datetime.datetime.now()} 5. Training final model with the best hyperparameters found...")
best_params = study.best_params
best_params['objective'] = 'lambdarank'
best_params['metric'] = 'NDCG'
best_params['verbose'] = -1
best_params['n_jobs'] = -1
best_params['seed'] = 42

# Identify categorical features again for the final model training
categorical_features_final_model = [col for col in categorical_features if col in X_train.columns]

final_model = lgb.LGBMRanker(**best_params)

# Train on the full training set
final_model.fit(X_train, y_train, group=groups.value_counts(sort=False).values, categorical_feature=categorical_features_final_model)
print(f"\n{datetime.datetime.now()} >Training completed...")




# --- 6. Generate Predictions and Submission File ---
print(f"\n{datetime.datetime.now()} 6. Generating final submission file...")
test_df['score'] = final_model.predict(X_test)

# Convert scores to ranks within each 'ranker_id' group
test_df['rank'] = test_df.groupby(group_column)['score'].rank(method='first', ascending=False).astype(int)

# Create the submission file
submission = test_df[['Id', 'ranker_id', 'rank']].copy()
submission = submission[['Id', 'rank']]
submission.rename(columns={'rank': 'selected'}, inplace=True)

# Save the submission file
submission.to_csv('/kaggle/working/aeroclub-recsys-2025/submission.csv', index=False)
print(f"{datetime.datetime.now()} Submission file 'submission.csv' created successfully!")
print("\nFirst 5 rows of the submission file:")
print(submission.head())
# print("\nFirst 5 rows of the test data:")
# print(/kaggle/input/aeroclub-recsys-2025/test_df.head())


