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


data_train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
data_train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


data_train = pd.concat([data_train,data_train_extra])
len(data_train)


data_train.head()


data_train['Brand'].isnull().sum()


def missing_checker(data):
    for col in data.columns:
        missing = data[col].isnull().sum()
        print(col + " missing datas: "+ str(missing))


missing_checker(data_train)


def missing_filler(data):
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if str(data[col].dtype) == 'object':
                data.loc[:, col] = data[col].fillna(data[col].mode()[0])
            else:
                data.loc[:, col] = data[col].fillna(data[col].mean())
    return data


data_train.dtypes


col_cat = ['Brand', 'Material', 'Size', 'Laptop Compartment','Waterproof', 'Style', 'Color']


import optuna
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error


y = data_train['Price']
X = data_train.drop(columns=['Price', 'id'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# kf = KFold(n_splits = 5, shuffle = True, random_state = 1)
# def objective(trial):
#     param = {
#         'objective': 'regression',  # Regression problem
#         'metric': 'rmse',  # RMSE as the evaluation metric
#         'boosting_type': 'gbdt',
#         'num_leaves': trial.suggest_int('num_leaves', 20, 100),
#         'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#         'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
#         'max_depth': trial.suggest_int('max_depth', -1, 20),  # -1 means no limit
#         'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
#         'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
#         "verbosity": -1,
#     }
#     cv = []
#     for train_index, valid_index in kf.split(X_train):
#         X_train_cv, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
#         y_train_cv, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
#         #fill the missing values
#         X_train_cv = missing_filler(X_train_cv)
#         X_valid = missing_filler(X_valid)
#         #convert into OHE
#         X_train_cv = pd.get_dummies(X_train_cv, columns=col_cat)
#         X_valid = pd.get_dummies(X_valid, columns=col_cat)
#         #train
#         model = lgb.LGBMRegressor(**param, random_state=1)
#         model.fit(X_train_cv, y_train_cv)
#         preds = model.predict(X_valid)
#         error = np.sqrt(mean_squared_error(y_valid, preds))
#         cv.append(error)
#         print(f"error: {error}")
#     return np.mean(cv)

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)

# print("Best Parameters:", study.best_params)
# print("Best RMSE:", study.best_value)


# Best Parameters: {'num_leaves': 50, 'n_estimators': 739, 'learning_rate': 0.2612404525427742, 'feature_fraction': 0.8401218007802728, 'max_depth': 2, 'lambda_l1': 0.6329572640361554, 'lambda_l2': 0.5871987970278316}
# Best RMSE: 38.90007455148513


best_param = {'num_leaves': 50, 'n_estimators': 739, 'learning_rate': 0.2612404525427742, 'feature_fraction': 0.8401218007802728, 'max_depth': 2, 'lambda_l1': 0.6329572640361554, 'lambda_l2': 0.5871987970278316, "verbosity":-1}
X_train = missing_filler(X_train)
X_test = missing_filler(X_test)
X_train = pd.get_dummies(X_train, columns=col_cat)
X_test = pd.get_dummies(X_test, columns=col_cat)
model = lgb.LGBMRegressor(**best_param, random_state=1)
model.fit(X_train, y_train)
preds = model.predict(X_test)
error = np.sqrt(mean_squared_error(y_test, preds))
print(f"RMSE: {error}")


X = missing_filler(X)
X = pd.get_dummies(X, columns=col_cat)
model = lgb.LGBMRegressor(**best_param, random_state=1)
model.fit(X, y)


data_test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
data_test = missing_filler(data_test)
data_test = pd.get_dummies(data_test, columns=col_cat)
data_test.head()


X_test = data_test.drop(columns=['id'])


preds = model.predict(X_test)
df_res = pd.DataFrame({'id':data_test['id'], 'Price':preds})
df_res.to_csv('/kaggle/working/submission.csv', index=False)
df_res

