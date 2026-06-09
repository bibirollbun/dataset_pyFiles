# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import warnings
import time
from sklearn.model_selection import train_test_split
import optuna
!pip install optuna-integration[xgboost]

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Load in data
df_train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')

# Convert date
df_train['date'] = pd.to_datetime(df_train['date'])

# Sort by each category
df_train.sort_values(by=['country', 'store', 'product', 'date'], inplace=True)


# # Imputation
# # First, let's fill the full missing categories with zeros
# # I am assuming these are sold out or they don't sell this specific type in that specific country
# missing_combos = [('Canada', 'Discount Stickers', 'Holographic Goose'),
#                  ('Kenya', 'Discount Stickers', 'Holographic Goose')]

# for missing_combo in missing_combos:
#     df_train.loc[(df_train['country'] == missing_combo[0]) & 
#                 (df_train['store'] == missing_combo[1]) & 
#                 (df_train['product'] == missing_combo[2]), 'num_sold'] = 0

# # # Lagged value/mean imputation
# df_train.loc[df_train['num_sold'].isnull(), 'num_sold'] = df_train.groupby(['country', 'store', 'product'])['num_sold'].shift(7)
# df_train.loc[df_train['num_sold'].isnull(), 'num_sold'] = df_train.groupby(['country', 'store', 'product'])['num_sold'].shift(14)
# df_train.loc[df_train['num_sold'].isnull(), 'num_sold'] = df_train.groupby(['country', 'store', 'product'])['num_sold'].shift(364)
# df_train.loc[df_train['num_sold'].isnull(), 'num_sold'] = df_train.groupby(['country', 'store', 'product'])['num_sold'].shift(728)
# df_train.loc[df_train['num_sold'].isnull(), 'num_sold'] = df_train.groupby(['country', 'store', 'product'])['num_sold'].transform(lambda x: x.fillna(x.mean()))
# # # I checked the graphs... good enough!!
# # We could add a std multiplied by a random variable but otherwise it's not hugely significant I presume


# Add Test Data
df_testR = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
df_testR['date'] = pd.to_datetime(df_testR['date'])
df_full = pd.concat([df_train, df_testR])


# Feature Creation

# Days of the week, month, year
df_full['year'] = df_full['date'].dt.year
df_full['day_of_week'] = df_full['date'].dt.dayofweek  # 0 = Monday, 6 = Sunday
df_full['day_of_week'] = df_full['day_of_week'].astype('category')
df_full['month'] = df_full['date'].dt.month
df_full['day'] = df_full['date'].dt.day

# Is a weekend
df_full['is_weekend'] = 0
df_full.loc[df_full['day_of_week'].isin([5, 6]), 'is_weekend'] = 1
df_full['is_weekend'] = df_full['is_weekend'].astype('category')

# Determine if each date is in the last week of the year
df_full['is_last_week'] = df_full['date'].apply(lambda x: x.isocalendar()[1] == pd.Timestamp(f"{x.year}-12-31").isocalendar()[1])
df_full['is_last_week'] = df_full['is_last_week'].astype('category')


# Determine whether each date is in a leap year
df_full['is_leap_year'] = (
    (df_full['date'].dt.year % 4 == 0) &
    ((df_full['date'].dt.year % 100 != 0) | (df_full['date'].dt.year % 400 == 0))
)
df_full['is_leap_year'] = df_full['is_leap_year'].astype('category')


# Calculate day of year with leap year adjustment
df_full['days_in_year'] = np.where(df_full['is_leap_year'], 366, 365)
df_full['day_of_year'] = df_full['date'].dt.dayofyear

# Calculate day in 2 year with leap year adjustment
days_in_year_df = df_full[['year', 'days_in_year']].drop_duplicates()
new_row = {'year': 2020, 'days_in_year': 366}

# Add the row using pd.concat
days_in_year_df = pd.concat([days_in_year_df, pd.DataFrame([new_row])], ignore_index=True)

days_in_year_df['two_year_group'] = (days_in_year_df['year'] - days_in_year_df['year'].min()) // 2
# Sum days within each group
two_year_summary = days_in_year_df.groupby('two_year_group')['days_in_year'].sum().reset_index()

# Rename for clarity
two_year_summary.rename(columns={'days_in_year': 'days_in_two_years'}, inplace=True)
days_in_year_df = days_in_year_df.merge(two_year_summary, on='two_year_group', how='left')
days_in_year_df = days_in_year_df.drop(columns=['two_year_group', 'days_in_year'])
df_full = df_full.merge(days_in_year_df, on='year', how='left')
df_full[['year', 'days_in_two_years']]

df_full['day_of_two_year'] = 0
df_full.loc[df_full['year'] % 2 == 1, 'day_of_two_year'] = df_full['days_in_two_years'] - df_full['days_in_year'] + df_full['day_of_year']
df_full.loc[df_full['year'] % 2 == 0, 'day_of_two_year'] = df_full['day_of_year']


# Fourier terms for complex seasonality
# Compute Fourier terms using the adjusted periodicity

# Week Seasonality
for k in range(1, 3):  # Use 2 harmonics as an example
    df_full[f'sin_week_{k}'] = np.sin(2 * np.pi * k * df_full['date'].dt.dayofweek / 7)
    df_full[f'cos_week_{k}'] = np.cos(2 * np.pi * k * df_full['date'].dt.dayofweek / 7)

# Year Seasonality
for k in range(1, 2):  # Use 1 harmonics for the year
    df_full[f'sin_year_{k}'] = np.sin(2 * np.pi * k * df_full['day_of_year'] / df_full['days_in_year'])
    df_full[f'cos_year_{k}'] = np.cos(2 * np.pi * k * df_full['day_of_year'] / df_full['days_in_year'])

# Every other year (bi-annual) Seasonality
for k in range(1, 2):  # Use the first 2 harmonics
    df_full[f'sin_biyear_{k}'] = np.sin(2 * np.pi * k * df_full['day_of_two_year'] / df_full['days_in_two_years'])
    df_full[f'cos_biyear_{k}'] = np.cos(2 * np.pi * k * df_full['day_of_two_year'] / df_full['days_in_two_years'])



# Encode the categorical vars
encoder = LabelEncoder()
for col in ['country', 'store', 'product']:
    df_full[col] = encoder.fit_transform(df_full[col])
    df_full[col] = df_full[col].astype('category')

# Drop unnecessary variables
df_full = df_full.drop(['is_leap_year', 'days_in_year', 'day_of_year', 'days_in_two_years', 'day_of_two_year'], axis=1)


# Train, test, split
train_test_split_date = pd.Timestamp('2017-01-01')

# # Make train and test sets
df_train1 = df_full[(df_full['date'] < train_test_split_date)] 
df_test = df_full[(df_full['date'] >= train_test_split_date)]

# Split Train into train and validation
train_valid_split_date = pd.Timestamp('2015-01-01')
df_valid = df_train1[(df_train1['date'] >= train_valid_split_date)]
df_train2 = df_train1[(df_train1['date'] < train_valid_split_date)]

df_train1.dropna(inplace=True)
df_train2.dropna(inplace=True)
df_valid.dropna(inplace=True)

#X and y
X_train_valid = df_train1.drop(['date', 'id', 'num_sold'], axis=1)
y_train_valid = df_train1['num_sold']

X_train = df_train2.drop(['date', 'id', 'num_sold'], axis=1)
y_train = df_train2['num_sold']

X_valid = df_valid.drop(['date', 'id', 'num_sold'], axis=1)
y_valid = df_valid['num_sold']

X_test = df_test.drop(['date', 'id', 'num_sold'], axis=1)
y_test = df_test[['id', 'num_sold']]

# Convert to matrix type which is good for XGBoost
dtrain = xgb.DMatrix(data=X_train, label=y_train, 
                     enable_categorical=True)
dvalid = xgb.DMatrix(data=X_valid, label=y_valid, 
                     enable_categorical=True)
dtest = xgb.DMatrix(data=X_test, label=y_test['num_sold'].fillna(0), 
                    enable_categorical=True)
dtrainvalid = xgb.DMatrix(data=X_train_valid, 
                          label=y_train_valid, 
                          enable_categorical=True)


base_params = metric = 'mape'
base_params = {
    'objective': 'reg:squarederror',
    'eval_metric': metric,
    'device': 'cuda'
}

learning_rate = 0.3


def objective(trial):
    params = {
        'tree_method': trial.suggest_categorical('tree_method', ['approx', 'hist']),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 250),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 25, log=True),
        'learning_rate': learning_rate,
    }
    num_boost_round = 10000
    params.update(base_params)
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, f'valid-{metric}')
    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round,
                      evals=[(dtrain, 'train'), (dvalid, 'valid')],
                      early_stopping_rounds=50,
                      verbose_eval=0,
                      callbacks=[pruning_callback])
    trial.set_user_attr('best_iteration', model.best_iteration)
    return model.best_score


sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
tic = time.time()
while time.time() - tic < 150:
    study.optimize(objective, n_trials=1)


low_learning_rate = 0.001

params = {}
params.update(base_params)
params.update(study.best_trial.params)
params['learning_rate'] = low_learning_rate
model_stage2 = xgb.train(params=params, dtrain=dtrain, 
                         num_boost_round=10000,
                         evals=[(dtrain, 'train'), (dvalid, 'valid')],
                         early_stopping_rounds=50,
                         verbose_eval=0)


def score_model(model: xgb.core.Booster, dmat: xgb.core.DMatrix) -> float:
    y_true = dmat.get_label() 
    y_pred = model.predict(dmat) 
    return mean_absolute_percentage_error(y_true, y_pred)


print('Stage 2 ==============================')
print(f'best score = {score_model(model_stage2, dvalid)}')
print('boosting params ---------------------------')
print(f'fixed learning rate: {params["learning_rate"]}')
print(f'best boosting round: {model_stage2.best_iteration}')


model_final = xgb.train(params=params, dtrain=dtrainvalid, 
                        num_boost_round=model_stage2.best_iteration,
                        verbose_eval=0)


print('Final Model ==========================')
print('parameters ---------------------------')
for k, v in params.items():
    print(k, ':', v)
print(f'num_boost_round: {model_stage2.best_iteration}')


y_test_predictions = model_final.predict(dtest) 
y_test['num_sold'] = y_test_predictions
y_test


df_test_with_preds = df_testR[['date', 'id', 'country', 'product', 'store']].merge(y_test, on='id', how='left', suffixes=('_true', '_pred'))
df_with_preds = pd.concat([df_train[['date', 'id', 'country', 'product', 'store', 'num_sold']], df_test_with_preds])
df_with_preds = df_with_preds.dropna()
df_with_preds['num_sold'] = df_with_preds['num_sold'].apply(lambda x: round(max(x, 0)))

df_with_preds


date1 = pd.Timestamp('2017-10-01')
date2 = pd.Timestamp('2017-10-15')
i=0
combinations = list(itertools.product(df_with_preds['country'].unique(), df_with_preds['store'].unique(), df_with_preds['product'].unique()))
for cat1, cat2, cat3 in combinations:
    
    df_sample = df_with_preds[(df_with_preds['country'] == cat1) &
                (df_with_preds['store'] == cat2) &
                (df_with_preds['product'] == cat3)] #&
                # (df_with_preds['date'] >= date1) &
                # (df_with_preds['date'] <= date2)]


    plt.figure(figsize=(8, 6))  # Set figure size
    plt.plot(df_sample['date'],df_sample['num_sold'])  # Create the histogram
    plt.title(f'Num Sold over time for {cat1}, {cat2}, {cat3}')  # Title of the plot
    plt.xlabel('Date')  # X-axis label
    plt.ylabel('Number Sold')  # Y-axis label
    plt.grid(True)  # Show grid
    plt.show()

    i = i+1
    # if i ==10:
    #     break


# # Plot feature importance
from xgboost import plot_importance
plot_importance(model_final, importance_type='weight')
plot_importance(model_final, importance_type='cover')
plot_importance(model_final, importance_type='gain')



y_test['num_sold'] = y_test['num_sold'].apply(lambda x: round(max(x, 0)))
y_test.to_csv('submission.csv', index=False)

