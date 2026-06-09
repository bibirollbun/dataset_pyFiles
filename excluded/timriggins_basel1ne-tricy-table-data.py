import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from tqdm import tqdm


train = pd.read_csv('/kaggle/input/neoai-2025-tricy-table-data//train_tables.csv')
test = pd.read_csv('/kaggle/input/neoai-2025-tricy-table-data/test_tables.csv')


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

ids = test['id'].tolist()

X = train.drop(columns='target')
y = train.target

imputer = KNNImputer(n_neighbors=7)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
test = pd.DataFrame(imputer.transform(test[X.columns]), columns=X.columns)


test = pd.DataFrame(imputer.transform(test[X.columns]), columns=X.columns)

drop_cols = ['target']
train_cols = [c for c in train.columns if c not in drop_cols]
weight = np.ones(len(train))
arg1 = pd.concat([X, np.log(y)], axis=1)[train_cols]
arg2 = test[train_cols]
arg3 = np.log(y)

test_sub = clf_train(arg1, arg2, arg3, weight, ids, func_inv = func_inv)







