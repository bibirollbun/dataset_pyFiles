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


# %% [markdown]
# # Bike Sharing Demand — Kaggle Notebook (Fixed Version)
#
# This notebook performs:
# 1. Data loading
# 2. EDA & visualizations
# 3. Feature engineering
# 4. LightGBM model training (K-Fold CV)
# 5. Submission creation

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# %% [markdown]
# ## Load Data

# %%
TRAIN_PATH = '/kaggle/input/bike-sharing-demand/train.csv'
TEST_PATH = '/kaggle/input/bike-sharing-demand/test.csv'
SAMPLE_SUB_PATH = '/kaggle/input/bike-sharing-demand/sampleSubmission.csv'

train = pd.read_csv(TRAIN_PATH, parse_dates=['datetime'])
test = pd.read_csv(TEST_PATH, parse_dates=['datetime'])
sample = pd.read_csv(SAMPLE_SUB_PATH)

print('train shape:', train.shape)
print('test shape:', test.shape)

# %% [markdown]
# ## Quick EDA

# %%
print('Missing in train:\n', train.isna().sum())
print('\nMissing in test:\n', test.isna().sum())

# %%
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.hist(train['count'], bins=50)
plt.title('Count distribution')
plt.subplot(1,2,2)
plt.hist(np.log1p(train['count']), bins=50)
plt.title('Log1p(count) distribution')
plt.show()

# %%
train['hour'] = train['datetime'].dt.hour
train.groupby('hour')['count'].mean().plot(kind='bar', figsize=(10,4), title='Average count by hour')
plt.show()
train.groupby('season')['count'].mean().plot(kind='bar', figsize=(6,3), title='Average count by season')
plt.show()

# %% [markdown]
# ## Feature Engineering

# %%
def create_features(df):
    df = df.copy()
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['weekday'] = df['datetime'].dt.weekday
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    return df

train_fe = create_features(train)
test_fe = create_features(test)

# %%
cat_cols = ['season', 'holiday', 'workingday', 'weather']
for c in cat_cols:
    train_fe[c] = train_fe[c].astype('int')
    test_fe[c] = test_fe[c].astype('int')

# %%
TARGET = 'count'
train_fe['log_count'] = np.log1p(train_fe[TARGET])

features = [
    'season', 'holiday', 'workingday', 'weather',
    'temp', 'atemp', 'humidity', 'windspeed',
    'hour', 'weekday', 'month', 'year',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
]

X = train_fe[features]
y = train_fe['log_count']
X_test = test_fe[features]

print('Using features:', features)

# %% [markdown]
# ## LightGBM Modeling (K-Fold CV with Fix)

# %%
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbosity': -1,
    'seed': 42
}

NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
sub_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f'Fold {fold+1}')
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    train_set = lgb.Dataset(X_tr, y_tr)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[train_set, val_set],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100)
        ]
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    fold_rmse = mean_squared_error(y_val, val_pred, squared=False)
    print(f'Fold {fold+1} RMSE (log-space):', fold_rmse)

    sub_preds += model.predict(X_test, num_iteration=model.best_iteration) / NFOLD

cv_rmse = mean_squared_error(y, oof_preds, squared=False)
print('CV RMSE (log-space):', cv_rmse)

# %% [markdown]
# ## Submission

# %%
final_preds = np.expm1(sub_preds)
submission = sample.copy()
submission['count'] = final_preds
submission.to_csv('/kaggle/working/submission.csv', index=False)
print('Submission saved to /kaggle/working/submission.csv')

