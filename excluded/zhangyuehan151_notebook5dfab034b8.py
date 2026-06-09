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

# Path to the Parquet file
train_parquet_path = '/kaggle/input/drw-crypto-market-prediction/train.parquet'

# Load the Parquet file
train_df = pd.read_parquet(train_parquet_path)

# Now train_df contains the same data as if you loaded train.csv,
# but likely much faster.
print(train_df.head())
print(train_df.info())


print("\nMissing values in 'label':")
print(train_df['label'].isnull().sum())


print("\nDescriptive statistics for 'label':")
print(train_df['label'].describe())


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(train_df['label'], bins=100, kde=True)
plt.title('Distribution of Target Variable (label)')
plt.xlabel('Label Value')
plt.ylabel('Frequency')
plt.show()


print("\nMissing values for core trading features:")
core_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
for col in core_features:
    print(f"{col}: {train_df[col].isnull().sum()}")


print("\nDescriptive statistics for core trading features:")
print(train_df[core_features].describe())


# Example for bid_qty
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(train_df['bid_qty'], bins=50, kde=True)
plt.title('Distribution of Bid Quantity')
plt.xlabel('Bid Quantity')
plt.ylabel('Frequency')
plt.show()

# You can repeat this for other core_features.
# If distributions are heavily skewed (common for volume/quantity),
# consider using `plt.yscale('log')` to see details for smaller values better.





import matplotlib.pyplot as plt
import seaborn as sns

core_features = ['ask_qty', 'buy_qty', 'sell_qty', 'volume']

for col in core_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df[col], bins=50, kde=True)
    plt.title(f'Distribution of {col.replace("_", " ").title()}') # Nicer title
    plt.xlabel(col.replace("_", " ").title())
    plt.ylabel('Frequency')
    # Consider uncommenting the next line if the distribution is very skewed and you want to see low-frequency bins better
    # plt.yscale('log')
    plt.show()


print("\nMissing values for a sample of X features (e.g., first 10):")
x_features_sample = [f'X{i}' for i in range(1, 11) if f'X{i}' in train_df.columns]
for col in x_features_sample:
    print(f"{col}: {train_df[col].isnull().sum()}")


print("\nDescriptive statistics for a sample of X features (e.g., X1-X5):")
print(train_df[['X1', 'X2', 'X3', 'X4', 'X5']].describe())


print(f"\nTotal number of entries: {len(train_df)}")
print(f"Time range: {train_df.index.min()} to {train_df.index.max()}")
time_diffs = train_df.index.to_series().diff().dropna()
print(f"Median time difference between consecutive rows: {time_diffs.median()}")
print(f"Most frequent time difference: {time_diffs.mode()[0]}")
print(f"Number of unique time differences: {len(time_diffs.unique())}")
print(f"Top 5 most frequent time differences:\n{time_diffs.value_counts().head()}")


plt.figure(figsize=(15, 7))
plt.plot(train_df.index, train_df['label'], alpha=0.7, label='Original Label')
plt.title('Target (label) over Time')
plt.xlabel('Timestamp')
plt.ylabel('Label Value')
plt.grid(True)
plt.legend()
plt.show()

# To see broader trends, plot a rolling mean
plt.figure(figsize=(15, 7))
plt.plot(train_df.index, train_df['label'].rolling(window=60*24).mean(), alpha=0.7, label='24-Hour Rolling Mean') # 24 hours of minutes
plt.title('Target (label) 24-Hour Rolling Mean over Time')
plt.xlabel('Timestamp')
plt.ylabel('Rolling Mean Label Value')
plt.grid(True)
plt.legend()
plt.show()


# Select a few features to check correlation
# We already know core_features and some X. Let's make a new list for printing.
features_for_corr_summary = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'X1', 'X2', 'X3', 'X4', 'X5']

print("\nCorrelations with 'label' for selected features:")
print(train_df[features_for_corr_summary + ['label']].corr()['label'].sort_values(ascending=False))


import pandas as pd
import numpy as np

# Let's assume train_df is your loaded DataFrame with timestamp as index

def feature_engineering(df):
    """Creates time-series and interaction features."""
    df_out = df.copy()

    # 1. Log transform skewed features
    # Using np.log1p which is log(1+x) to handle zeros
    skewed_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    for col in skewed_features:
        df_out[f'{col}_log'] = np.log1p(df_out[col])

    # 2. Interaction & Ratio Features (based on EDA insights)
    df_out['order_book_imbalance'] = (df_out['bid_qty'] - df_out['ask_qty']) / (df_out['bid_qty'] + df_out['ask_qty'])
    df_out['trade_imbalance'] = (df_out['buy_qty'] - df_out['sell_qty']) / (df_out['buy_qty'] + df_out['sell_qty'])
    df_out['depth_pressure'] = (df_out['bid_qty'] - df_out['ask_qty']) / (df_out['buy_qty'] + df_out['sell_qty']) # New idea

    # 3. Time-based Features
    df_out['hour'] = df_out.index.hour
    df_out['day_of_week'] = df_out.index.dayofweek # Monday=0, Sunday=6

    # 4. Lagged Features (CRITICAL)
    # Using a few lags for a simple start
    lags = [1, 2, 5, 10] # 1, 2, 5, 10 minutes ago
    features_to_lag = ['volume_log', 'order_book_imbalance', 'trade_imbalance', 'X1', 'X2']
    for lag in lags:
        for feat in features_to_lag:
            df_out[f'{feat}_lag_{lag}'] = df_out[feat].shift(lag)

    # 5. Rolling Window Features (CRITICAL for volatility)
    windows = [5, 10, 30] # 5, 10, 30 minute windows
    features_to_roll = ['volume_log', 'label', 'X1', 'X2'] # Rolling on 'label' is ok for PAST values
    for window in windows:
        for feat in features_to_roll:
            # Shift by 1 to prevent using current value to predict itself, especially for label
            df_out[f'{feat}_roll_std_{window}'] = df_out[feat].shift(1).rolling(window=window).std()
            df_out[f'{feat}_roll_mean_{window}'] = df_out[feat].shift(1).rolling(window=window).mean()

    # Clean up NaNs created by lagging/rolling
    df_out = df_out.replace([np.inf, -np.inf], np.nan) # Replace infs created by division by zero
    df_out = df_out.fillna(0) # Simple strategy: fill NaNs with 0. Forward fill is another option.

    return df_out

# Apply the function
train_featured_df = feature_engineering(train_df)

print("Shape of original df:", train_df.shape)
print("Shape of featured df:", train_featured_df.shape)
print("\nSome new features:")
print(train_featured_df[['order_book_imbalance', 'volume_log', 'X1_lag_5', 'label_roll_std_10']].head(15))


# The target variable is the 'label' column
y_v2 = train_featured_df['label']

# The features are all columns EXCEPT the original label and any other columns we want to exclude
# We should exclude the raw skewed features now that we have the log-transformed versions
excluded_features = ['label', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
features = [col for col in train_featured_df.columns if col not in excluded_features]
X_v2 = train_featured_df[features]

print(f"Number of features: {len(X_v2.columns)}")
print(f"Shape of X: {X_v2.shape}")
print(f"Shape of y: {y_v2.shape}")


import optuna
import lightgbm as lgb
import numpy as np


# def objective(trial):
#     dtrain = lgb.Dataset(
#         X_v2.values.astype('float32'),
#         label=y_v2.values.astype('float32'),
#         free_raw_data=False,
#     )

#     params = {
#         "objective": "regression",
#         "metric": "rmse",
#         "boosting_type": "gbdt",
#         "learning_rate": 0.05,
#         "verbosity": -1,
#         "num_leaves": trial.suggest_int("num_leaves", 20, 80),
#         "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
#         "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
#         "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
#         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
#         "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
#         "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
#         "num_threads": 1,
#     }

#     cv_results = lgb.cv(
#         params,
#         dtrain,
#         nfold=3,                # or 5 if your memory allows
#         num_boost_round=300,    # or your chosen budget
#         shuffle=False,
#         stratified=False,
#         seed=None,
#         callbacks=[
#             lgb.early_stopping(stopping_rounds=50),
#             lgb.log_evaluation(period=0),
#         ],
#     )

#     # pick up the right '-mean' key no matter what prefix
#     mean_key = next(key for key in cv_results if key.endswith("-mean"))
#     result = min(cv_results[mean_key])

#     # clean up
#     del dtrain, cv_results
#     import gc; gc.collect()

#     return result



# # 3️⃣ Create Optuna study
# study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())

# print("--- Starting Hyperparameter Tuning ---")
# study.optimize(objective, n_trials=30)
# print("--- Tuning Complete ---\n")

# # 4️⃣ Print results (just like before)
# print("--- Tuning Results ---")
# print(f"Number of finished trials: {len(study.trials)}")
# print(f"Best trial's Average CV RMSE: {study.best_value:.5f}\n")

# print("Best trial's parameters:")
# for key, value in study.best_params.items():
#     print(f"    {key}: {value}")


 study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())



best_params_from_cv = {
    'num_leaves': 45,
    'lambda_l1': 8.787667104782061,
    'lambda_l2': 0.005771739600459773,
    'feature_fraction': 0.8918692418029897,
    'bagging_fraction': 0.8283591790068191,
    'bagging_freq': 1,
    'min_child_samples': 5,
}



import lightgbm as lgb
import numpy as np


print("--- Training Final Model with OPTUNA‑TUNED Hyperparameters ---")

# 2️⃣ Base params that you always want
final_params = {
    'objective': 'rmse',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'verbose': -1,
    'n_jobs': -1,         # uses all CPU cores
}

# 3️⃣ Merge in your tuned hyperparameters
final_params.update(best_params_from_cv)

# 4️⃣ Decide on how many trees to grow
#    - You tuned up to 2000 in CV, but you can train fewer now.
#    - If you want to leverage early‑stopping, you could:
#         n_estimators=2000, callbacks=[lgb.early_stopping(…)]
#      But here we’ll go with a fixed budget:
final_params['n_estimators'] = 500

# 5️⃣ Instantiate and fit on ALL of your data
final_model = lgb.LGBMRegressor(**final_params)
final_model.fit(X_v2, y_v2)

print("\n--- Final Model is Trained and Ready for Submission ---")



final_model.booster_.save_model('lgbm_final_model.txt')

