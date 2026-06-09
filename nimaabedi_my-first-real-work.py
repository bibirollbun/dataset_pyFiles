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


train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
sub = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')


sample = sub


train


test


sample


def reduce_mem_usage(dataframe, dataset):
    print('Reduce Memory Usage for:', dataset)
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2

    for col in dataframe.columns:
        col_type = dataframe[col].dtype

        c_min = dataframe[col].min()
        c_max = dataframe[col].max()
        if str(col_type)[:3] == 'int':
        
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                dataframe[col] = dataframe[col].astype(np.int8)
        
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                dataframe[col] = dataframe[col].astype(np.int16)

            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                dataframe[col] = dataframe[col].astype(np.int32)

            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                dataframe[col] = dataframe[col].astype(np.int64)

        else:

            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                dataframe[col] = dataframe[col].astype(np.float16)

            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                dataframe[col] = dataframe[col].astype(np.float32)

            else:
                dataframe[col] = dataframe[col].astype(np.float64)

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print('--- Memory usage before: {:.2f} MB'.format(initial_mem_usage))
    print('--- Memory usage Now: {:.2f} MB'.format(final_mem_usage))
    print('--- Memory usage Decreased By: {:.1f}%\n'.format(100 * (initial_mem_usage - final_mem_usage)/initial_mem_usage))

    return dataframe


train = train.reset_index(drop=True)
test = test.reset_index(drop=True)


selected_features = ['X863', 'X856', 'X344', 'X598', 'X862', 'X385', 'X852', 
                    'X603', 'X860', 'X674', 'X415', 'X345', 'X137', 'X855', 
                    'X174', 'X302', 'X178', 'X532', 'X168', 'X612', 'bid_qty', 
                    'ask_qty', 'buy_qty', 'sell_qty', 'volume']


train = train[selected_features + ['label']]
test = test[selected_features]


train.head()


test.head()


train = reduce_mem_usage(train, 'train')
test = reduce_mem_usage(test, 'test')


print(f'Train = {train.shape}')
print(f'Test = {test.shape}')
print(f'Sample = {sample.shape}')


train['liq_imbalance'] = (train['bid_qty'] - train['ask_qty']) / (train['bid_qty'] - train['ask_qty'])
test['liq_imbalance'] = (test['bid_qty'] - test['ask_qty']) / (test['bid_qty'] - test['ask_qty'])

train['buy_sell_ratio'] = train['buy_qty'] / (train['sell_qty'] + 1e-6)
test['buy_sell_ratio'] = test['buy_qty'] / (test['sell_qty'] + 1e-6)


RMV = ['label']
FEATURES = [col for col in train.columns if not col in RMV]
print(f'There are {len(FEATURES)} FEATURES: {FEATURES}')


from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from scipy.stats import pearsonr


Folds = 5
kf = KFold(n_splits = Folds, shuffle = True, random_state = 42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

xgb_params = {
    'tree_method': 'gpu_hist',
    'colsample_bylevel': 0.4778015829774066,
    'colsample_bynode': 0.362764358742407,
    'colsample_bytree': 0.7107423488010493,
    'gamma': 1.7094857725240398,
    'learning_rate': 0.02213323588455387,
    'max_depth': 20,
    'max_leaves': 12,
    'min_child_weight': 16,
    'n_estimators': 1667,
    'n_jobs': -1,
    'random_state': 42,
    'reg_alpha': 39.352415706891264,
    'reg_lambda': 75.44843704068275,
    'subsample': 0.06566669853471274,
    'verbosity': 0
}

for i, (train_idx, valid_idx) in enumerate(kf.split(train)):
    print('#'*25)
    print(f'### Fold {i+1}')
    print('#' * 25)

    X_train = train.iloc[train_idx][FEATURES]
    y_train = train.iloc[train_idx]['label']
    X_valid = train.iloc[valid_idx][FEATURES]
    y_valid = train.iloc[valid_idx]['label']
    X_test = test[FEATURES]

    model = XGBRegressor(**xgb_params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=25,
        verbose=200
    )


    oof_preds[valid_idx] = model.predict(X_valid)
    test_preds += model.predict(X_test)

pearson_score = pearsonr(train['label'], oof_preds)[0]
print('Final Pearson Correlation = ', pearson_score)

test_preds /= Folds


import torch
print(torch.cuda.is_available())  # Should return True


sample['prediction'] = test_preds
sample.to_csv('submission.csv', index = False)


sample




