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


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from catboost import Pool



train_df = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/train.csv')
test_df = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/test.csv')

sub = pd.read_csv('/kaggle/input/california-homelessness-prediction-challenge/sample_submission.csv')
train_df.head()


# AL, LAなどは市を表し、13、10などはストリートを表現すると思われるので、_でsplitして市と数値に分ける
train_df['city'] = train_df['ID'].str.split('_').str[0]
train_df['num'] = train_df['ID'].str.split('_').str[1]

test_df['city'] = test_df['ID'].str.split('_').str[0]
test_df['num'] = test_df['ID'].str.split('_').str[1]


# 	cityについて、OneHotEncodingする

from sklearn.preprocessing import OneHotEncoder

train = train_df.copy()
test = test_df.copy()

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(train[['city']])

train_onehot_df = pd.DataFrame(enc.transform(train[['city']]).toarray(), columns=enc.get_feature_names_out())
test_onehot_df = pd.DataFrame(enc.transform(test[['city']]).toarray(), columns=enc.get_feature_names_out())

train = pd.concat([train, train_onehot_df], axis=1)
test = pd.concat([test, test_onehot_df], axis=1)

train


# 'num'についてもOneHotEncodingする

enc_2 = OneHotEncoder(handle_unknown='ignore')
enc_2.fit(train[['num']])

train_onehot_df_2 = pd.DataFrame(enc_2.transform(train[['num']]).toarray(), columns=enc_2.get_feature_names_out())
test_onehot_df_2 = pd.DataFrame(enc_2.transform(test[['num']]).toarray(), columns=enc_2.get_feature_names_out())

train = pd.concat([train, train_onehot_df_2], axis=1)
test = pd.concat([test, test_onehot_df_2], axis=1)

train


target = train_df['HOMELESS_RATE']

target



train = train.drop(['ID','HOMELESS_RATE', 'city', 'num'], axis=1)
test = test.drop(['ID', 'city', 'num'], axis=1)



models = (DecisionTreeRegressor(), SVR(), LinearRegression(), Ridge(), Lasso(), ElasticNet(), SGDRegressor(),
          RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor(), LGBMRegressor(), CatBoostRegressor())

kf = KFold(n_splits=5, shuffle=True, random_state=42)

model_name_score_list = []

for model in models:
    rmse_list = []
    for train_index, val_index in kf.split(train):
        X_train, X_val = train.iloc[train_index], train.iloc[val_index]
        y_train, y_val = target.iloc[train_index], target.iloc[val_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse_list.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        
    print(f"{model.__class__.__name__}: {np.mean(rmse_list):.4f}")
    model_name_score_list.append((model.__class__.__name__, np.mean(rmse_list)))

model_name_score_list.sort(key=lambda x: x[1])




print(f"Best model: {model_name_score_list[0][0]}")
print(f"models_params: {model_name_score_list}")


# SGDregressorをつかってパラメーターをGridSearchする

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=42)

param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'l1_ratio': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'eta0': [0.0001, 0.001, 0.01, 0.1, 1.0],
}

model_SGD = SGDRegressor()
grid_search = GridSearchCV(estimator=model_SGD, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

rmse = np.sqrt(mean_squared_error(y_test, grid_search.predict(X_test)))
print("RMSE: ", rmse)

prediction_SGD = grid_search.predict(test)
prediction_SGD


# Lassoを用いてパラメーターをグリッドサーチします。

param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'max_iter': [1000, 2000, 3000, 4000, 5000],
    'tol': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
    'warm_start': [True, False],
}

model_Lasso = Lasso()
grid_search = GridSearchCV(estimator=model_Lasso, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

rmse = np.sqrt(mean_squared_error(y_test, grid_search.predict(X_test)))
print("RMSE: ", rmse)

prediction_Lasso = grid_search.predict(test)
prediction_Lasso




# catboostRegressorのパラメーターをグリッドサーチする
param_grid = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5],
    'border_count': [32, 64, 128],
    'bagging_temperature': [0, 1, 2],
    'random_strength': [0, 1, 2],
    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
}

model_cat = CatBoostRegressor()
grid_search = GridSearchCV(estimator=model_cat, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)




# 最適なパラメータを表示
print("Best parameters found: ", grid_search.best_params_)
print("Best score found: ", grid_search.best_score_)

rmse = np.sqrt(mean_squared_error(y_test, grid_search.predict(X_test)))
print("RMSE: ", rmse)


prediction_cat = grid_search.predict(test)
prediction_cat


prediction = np.mean([prediction_cat, prediction_Lasso, prediction_SGD], axis=0)
prediction


sub['HOMELESS_RATE'] = prediction
sub.to_csv('submission.csv', index = False)

