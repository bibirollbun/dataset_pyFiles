import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


train = pd.read_csv('/kaggle/input/neoai-2025-tricy-table-data/train_tables.csv')


test = pd.read_csv('/kaggle/input/neoai-2025-tricy-table-data/test_tables.csv')


for col in train.columns:
    if col != 'target':
        train[f'{col}_is_nan'] = train[col].isna().astype(int)


train['feat_0_l500'] = (train['feat_0'] < 500).astype(int)
train['feat_5_l60'] = (train['feat_5'] < 60).astype(int)
train['feat_3_l90'] = (train['feat_3'] < 90).astype(int)
train['feat_6_l3k'] = (train['feat_6'] < 3000).astype(int)
train['feat_7_l1200'] = (train['feat_7'] < 1200).astype(int)
train['feat_7_m1000'] = (train['feat_7'] > 1000).astype(int)
train['feat_8_m620'] = (train['feat_8'] > 620).astype(int)


%pip install catboost -qqq


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

mg = pd.concat((train[[x for x in test.columns if x != 'id']], test.drop('id', axis=1)))
mg['target'] = 1
mg['target'].iloc[len(train):] = 0


x_train, x_test, y_train, y_test = train_test_split(mg.drop(['target', 'day', 'hour', 'minute'], axis=1), mg['target'], test_size=0.2)
cb = CatBoostClassifier(2000, verbose=100)
cb.fit(x_train, y_train, eval_set=(x_test, y_test))


cb.get_feature_importance(prettified=True)


train['target'] = np.log(train['target'])


for col in test.columns:
    if 'target' not in col and col != 'id':
        test[f'{col}_is_nan'] = test[col].isna().astype(int)


test['feat_0_l500'] = (test['feat_0'] < 500).astype(int)
test['feat_5_l60'] = (test['feat_5'] < 60).astype(int)
test['feat_3_l90'] = (test['feat_3'] < 90).astype(int)
test['feat_6_l3k'] = (test['feat_6'] < 3000).astype(int)
test['feat_7_l1200'] = (test['feat_7'] < 1200).astype(int)
test['feat_7_m1000'] = (test['feat_7'] > 1000).astype(int)
test['feat_8_m620'] = (test['feat_8'] > 620).astype(int)


def clf_train(train, test, target, weight_col, id_col, name_file = 'sub.csv', func_inv = None):

    param = {
    'learning_rate': 0.1,
    'num_leaves': 48,
    'lambda_l1' : 1,
    'lambda_l2' : 1,
    'min_data_in_leaf' : 100,
    'objective': 'mae',
    'verbosity':-1,
    }
    
    predict_test = np.zeros(len(test))

    tr = lgb.Dataset(train, target, weight=weight_col)
    bst = lgb.train(param, tr, num_boost_round=500)
    predict_test = bst.predict(test)
    if func_inv:
        predict_test = func_inv(predict_test)
    sub = pd.DataFrame()
    sub['id'] = id_col
    sub['target'] = predict_test
    sub.to_csv(name_file, index = None)


def func_inv(x):
    return np.exp(x)  


drop_cols = ['target', 'feat_0']
train_cols = [c for c in train.columns if c not in drop_cols]


for col in train_cols:
    if col.count('_') == 1 and 'feat' in col:
        mn = train[col].median()
        train[col].fillna(mn, inplace=True)
        test[col].fillna(mn, inplace=True)


weight = np.ones(len(train))
test_sub = clf_train(
    train[train_cols], 
    test[train_cols], 
    train['target'], 
    weight, 
    test['id'].tolist(), 
    'submission.csv', 
    func_inv = func_inv
)




