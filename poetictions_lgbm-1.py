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


from sklearn.model_selection import train_test_split
import lightgbm
from sklearn.metrics import roc_auc_score

import matplotlib.pylab as plt


input_dir = '/kaggle/input/santander-customer-transaction-prediction'
df_train = pd.read_csv(input_dir + '/train.csv')
df_train


var_columns = [c for c in df_train.columns if c not in ['ID_code', 'target']]

X = df_train.loc[:, var_columns]
y = df_train.loc[:, 'target']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape


train_data = lightgbm.Dataset(X_train, label = y_train)
valid_data = lightgbm.Dataset(X_valid, label=y_valid)


parameters = {'objective' : 'binary',
              'metric': 'auc',
              'is_unbalance': 'true',
              'boosting': 'gbdt',
              'num_leaves':63,
              'feature_fraction' : 0.5,
              'bagging_fraction': 0.5,
              'bagging_freq': 20,
              'learning_rate': 0.01,
              'verbose': -1}


model = lightgbm.train(parameters,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=5000,
    callbacks=[
        lightgbm.early_stopping(stopping_rounds=50),
    ]
)


y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)

print("AUC Train: {:.4f}\nAUC Valid: {:.4f}". format(roc_auc_score(y_train, y_train_pred),
                                                     roc_auc_score(y_valid, y_valid_pred)))



df_test = pd.read_csv(input_dir+ '/test.csv')
df_sample_submission = pd.read_csv(input_dir + '/sample_submission.csv')


X_test = df_test.loc[:,var_columns]
df_sample_submission['target'] = model.predict(X_test)
df_sample_submission


output_dir = '/kaggle/working'
df_sample_submission.to_csv(output_dir + "04_lgbm_scores.csv", index=False)




