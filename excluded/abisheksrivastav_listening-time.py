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


!pip install lightgbm xgboost category_encoders



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

# Save test ID
test_ids = test['id']

# Target
target = 'Listening_Time_minutes'

# Combine train and test for preprocessing
train['is_train'] = 1
test['is_train'] = 0
test[target] = -1  # Dummy target for concat
full = pd.concat([train, test], axis=0)

# Handle missing numeric values with mean
num_cols = ['Episode_Length_minutes', 'Guest_Popularity_percentage', 'Number_of_Ads']
for col in num_cols:
    full[col].fillna(full[col].mean(), inplace=True)

# Handle datetime
full['Publication_Hour'] = pd.to_datetime(full['Publication_Time'], errors='coerce').dt.hour
if full['Publication_Hour'].isnull().any():
    mode_val = full['Publication_Hour'].mode()
    if not mode_val.empty:
        full['Publication_Hour'].fillna(mode_val.iloc[0], inplace=True)
full.drop(columns=['Publication_Time'], inplace=True)

# Handle categorical missing values with mode
cat_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Episode_Sentiment']
for col in cat_cols:
    mode_val = full[col].mode()
    if not mode_val.empty:
        full[col].fillna(mode_val.iloc[0], inplace=True)

# Feature Engineering: target encoding on high-cardinality columns
target_encoding_cols = ['Podcast_Name', 'Episode_Title']
for col in target_encoding_cols:
    target_means = full[full['is_train'] == 1].groupby(col)[target].mean()
    full[col + '_target_enc'] = full[col].map(target_means)

# Label encode remaining categoricals
le = LabelEncoder()
for col in ['Genre', 'Publication_Day', 'Episode_Sentiment']:
    full[col] = le.fit_transform(full[col])

# Drop unneeded columns
drop_cols = ['id', 'is_train', 'Podcast_Name', 'Episode_Title']
full.drop(columns=drop_cols, inplace=True)

# Split back
train = full[full[target] != -1].copy()
test = full[full[target] == -1].copy().drop(columns=[target])

X = train.drop(columns=[target])
y = train[target].values
X_test = test

# Stratified KFold based on binned target
y_bins = pd.qcut(y, q=10, labels=False)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42,
    'verbose': -1
}

# Training loop
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_bins)):
    print(f"\n[Fold {fold+1}]")
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    model = lgb.LGBMRegressor(**params, n_estimators=10000)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )

    val_preds = model.predict(X_val, num_iteration=model.best_iteration_)
    oof_preds[val_idx] = val_preds

    test_preds += model.predict(X_test, num_iteration=model.best_iteration_) / kf.n_splits

# Final score
oof_rmse = mean_squared_error(y, oof_preds, squared=False)
print(f"\nðŸŽ¯ Overall OOF RMSE: {oof_rmse:.4f}")

# Save submission
submission = pd.DataFrame({
    'id': test_ids,
    'Listening_Time_minutes': test_preds
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission saved!")


