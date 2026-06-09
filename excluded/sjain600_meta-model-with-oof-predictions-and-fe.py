# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

# Ignore only the specific FutureWarning from pandas option
warnings.filterwarnings(
    action='ignore',
    category=FutureWarning,
    message=r".*use_inf_as_na option is deprecated.*"
)

# Setting matplotlib defaults
plt.rc('figure', figsize=(8, 5), dpi=120)

plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=15, titlepad=10)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/black-jack-smart-effect-of-removal-ml/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/black-jack-smart-effect-of-removal-ml/test.csv', index_col='id')


train.sample(5)


train.info()


num_cols = train.select_dtypes(include=['int64']).columns
train[num_cols] = train[num_cols].astype('int8')
test[num_cols] = test[num_cols].astype('int8')


train.info()


train.shape


test.shape


train.isnull().sum()


test.isnull().sum()


train.describe()


test.describe()


sns.histplot(x='ev', data=train, bins=20, kde=True, color='orange')
plt.show()


sns.heatmap(train.corr(), annot=True, cmap='viridis', fmt='.2f', linewidths=0.5)


train.columns


max_counts = {1:24, 2:24, 3:24, 4:24, 5:24, 6:24, 7:24, 8:24, 9:24, 10:96}
for i in range(1,11):
    train[f'pct_removed_{i}'] = train[str(i)] / max_counts[i]
    test[f'pct_removed_{i}'] = test[str(i)] / max_counts[i]


for i in range(1,11):
    train[f'remaining_{i}'] = max_counts[i] - train[str(i)]
    test[f'remaining_{i}'] = max_counts[i] - test[str(i)]



features = ['2', '3', '4', '5', '6']
for i, col1 in enumerate(features):
    for col2 in (features[i+1:]):
        new_col = f'{col1}_{col2}'
        train[new_col] = train[col1] * train[col2]
        test[new_col] = test[col1] * test[col2]


train['removed_low'] = train[[str(i) for i in range(2, 7)]].sum(axis=1).astype('int8')
test['removed_low'] = test[[str(i) for i in range(2, 7)]].sum(axis=1).astype('int8')

train['removed_low_mean'] = train[[str(i) for i in range(2, 7)]].mean(axis=1).astype('int8')
test['removed_low_mean'] = test[[str(i) for i in range(2, 7)]].mean(axis=1).astype('int8')

train['removed_mid'] = train[[str(i) for i in range(7, 10)]].sum(axis=1).astype('int8')
test['removed_mid'] = test[[str(i) for i in range(7, 10)]].sum(axis=1).astype('int8')

train['removed_mid_std'] = train[[str(i) for i in range(7, 10)]].std(axis=1).astype('int8')
test['removed_mid_std'] = test[[str(i) for i in range(7, 10)]].std(axis=1).astype('int8')

train['total_removed_mean'] = train[[str(i) for i in range(1, 11)]].mean(axis=1).astype('float32')
test['total_removed_mean'] = test[[str(i) for i in range(1, 11)]].mean(axis=1).astype('float32')

train['total_removed_std'] = train[[str(i) for i in range(1, 11)]].std(axis=1).astype('float32')
test['total_removed_std'] = test[[str(i) for i in range(1, 11)]].std(axis=1).astype('float32')

train['total_removed_max'] = train[[str(i) for i in range(1, 11)]].max(axis=1)
test['total_removed_max'] = test[[str(i) for i in range(1, 11)]].max(axis=1)

train['total_removed_min'] = train[[str(i) for i in range(1, 11)]].min(axis=1)
test['total_removed_min'] = test[[str(i) for i in range(1, 11)]].min(axis=1)


train['low_mid_ratios'] = train['removed_low'] / (train['removed_mid'] + 1e-5)
test['low_mid_ratios'] = test['removed_low'] / (test['removed_mid'] + 1e-5)

train['aces_tens_interact'] = train['1'] * train['10'] 
test['aces_tens_interact'] = test['1'] * test['10'] 

train['aces_tens_ratios'] = (train['1'] / (train['10'] + 1e-6)).astype('float32')
test['aces_tens_ratios'] = (test['1'] / (test['10'] + 1e-6)).astype('float32')


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

X = train.copy()
y = X.pop('ev')
X_test = test.copy()

oof_cat = np.zeros(len(y))
oof_hist = np.zeros(len(y))
test_preds_cat = np.zeros(len(X_test))
test_preds_hist = np.zeros(len(X_test))

n_folds=10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=34)


from sklearn.ensemble import HistGradientBoostingRegressor

for fold, (train_index, valid_index) in enumerate(kf.split(X, y)):
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    hist = HistGradientBoostingRegressor(max_iter=10000, learning_rate=0.0878, max_depth=10, warm_start=True, random_state=34).fit(X_train, y_train)

    hist_pred = hist.predict(X_valid)

    fold_mse = mean_squared_error(y_valid, hist_pred)
    print(f"Hist Fold {fold + 1} MSE: {fold_mse:.9f}")
    
    test_preds_hist += hist.predict(X_test) / n_folds


from catboost import CatBoostRegressor

for fold, (train_index, valid_index) in enumerate(kf.split(X, y)):
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    cat = CatBoostRegressor(iterations=17000, learning_rate=0.05, depth=4, rsm=1.0, l2_leaf_reg=3.775,
    random_seed=34).fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=100, verbose=0)

    cat_pred = cat.predict(X_valid)
    oof_cat[valid_index] = cat_pred

    fold_mse = mean_squared_error(y_valid, cat_pred)
    print(f"CAT Fold {fold + 1} MSE: {fold_mse:.9f}")
    
    test_preds_cat += cat.predict(X_test) / n_folds


from sklearn.ensemble import RandomForestRegressor

meta_train = np.column_stack((oof_hist, oof_cat))
meta_test = np.column_stack((test_preds_hist, test_preds_cat))

meta_model = RandomForestRegressor(n_estimators=250, random_state=34, max_depth=4, min_samples_split=4).fit(meta_train, y)

final_preds = meta_model.predict(meta_test)


sub = pd.read_csv('/kaggle/input/black-jack-smart-effect-of-removal-ml/sample_submission.csv')
sub['ev'] = final_preds
sub.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

