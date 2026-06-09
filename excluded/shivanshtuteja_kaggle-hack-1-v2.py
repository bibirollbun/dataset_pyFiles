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


data = pd.read_csv(r'/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/train.csv')
data.head()


X = data.drop(columns = ['target'])
Y = data['target']


X_Corr = X.corrwith(Y)
X_Corr

#very low correlation for all the features; set a threshold say 0.05 for dropping features


low_corr = []
for key in X_Corr.keys():
    if abs(X_Corr[key])  < 0.05:
        low_corr.append(key)
low_corr


X_train = X.drop(columns=low_corr)


#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_train,Y, test_size = 0.20)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score


base_models = [RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor()]


results = pd.DataFrame(columns = ['model_name','rmse','r2_score'])


for model in base_models:
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    rmse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    result = pd.DataFrame([[str(model).split('(')[0],rmse, r2]],columns = ['model_name','rmse','r2_score'])
    results = pd.concat([results,result],ignore_index = True)


results


tuned_models = []


model = GradientBoostingRegressor(learning_rate=0.1, max_depth=7, n_estimators=300, 
                                  subsample=0.8, random_state=42)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
rmse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
result = pd.DataFrame([[str(model).split('(')[0] + ' tuned',rmse, r2]],columns = ['model_name','rmse','r2_score'])
results = pd.concat([results,result],ignore_index = True)

tuned_models.append(model)


best_params = {'random_state': 42, 'n_estimators': 300, 'min_samples_split': 5, 
               'min_samples_leaf': 1, 'max_samples': None, 'max_features': 'log2', 
               'max_depth': None, 'bootstrap': False}

model = RandomForestRegressor(**best_params)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
rmse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
result = pd.DataFrame([[str(model).split('(')[0] + ' tuned',rmse, r2]],columns = ['model_name','rmse','r2_score'])
results = pd.concat([results,result],ignore_index = True)

tuned_models.append(model)


results


best_params = {
    'subsample': 0.8, 
    'reg_lambda': 3.0, 
    'reg_alpha': 0, 
    'random_state': 42, 
    'n_estimators': 300, 
    'min_child_weight': 3, 
    'max_depth': 10, 
    'learning_rate': 0.1, 
    'gamma': 0,
    'colsample_bytree': 0.6, 
    'booster': 'dart'
}

model = XGBRegressor(**best_params)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
rmse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
result = pd.DataFrame([[str(model).split('(')[0] + ' tuned',rmse, r2]],columns = ['model_name','rmse','r2_score'])
results = pd.concat([results,result],ignore_index = True)

tuned_models.append(model)


results


base_predictions = []

# for model in base_models:
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     base_predictions.append(y_pred)

for model in tuned_models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    base_predictions.append(y_pred)

base_predictions = np.column_stack(base_predictions)


from sklearn.linear_model import LinearRegression, ElasticNet
meta_model = ElasticNet()
meta_model.fit(base_predictions, y_test)

meta_pred = meta_model.predict(base_predictions)


rmse = mean_squared_error(y_test,meta_pred)
r2 = r2_score(y_test,meta_pred)
result = pd.DataFrame([['mtea model + Elastinet',rmse, r2]],columns = ['model_name','rmse','r2_score'])
results = pd.concat([results,result],ignore_index = True)


results


# from sklearn.model_selection import RandomizedSearchCV

# params = {
#     'fit_intercept': [True, False],  # Whether to calculate the intercept
#     'copy_X': [True, False],  # Whether to copy the data (affects memory usage)
#     'positive': [True, False],  # Restrict coefficients to be positive
#     'n_jobs': [-1]  # Use all available processors (helpful for large datasets)
# }

# random_search = RandomizedSearchCV(estimator=meta_model,param_distributions=params, n_iter=100, scoring='r2', error_score=0, cv=10, n_jobs = -1)

# random_search.fit(base_predictions,y_test)

# best_model = random_search.best_estimator_
# best_params = random_search.best_params_

# print("Best parameters:", best_params)
# print("Test accuracy:", random_search.best_score_)


# from sklearn.model_selection import RandomizedSearchCV

# params = {
#     'alpha': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
#     'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1],  # Ratio of L1 to L2 regularization
#     'max_iter': [1000, 5000, 10000],  # Maximum number of iterations
#     'tol': [1e-4, 1e-3, 1e-2],  # Convergence tolerance
#     'fit_intercept': [True, False],  # Whether to calculate intercept
#     # 'normalize': [True, False]  # Normalize features (for older versions of scikit-learn)
# }

# random_search = RandomizedSearchCV(estimator=meta_model,param_distributions=params, n_iter=100, scoring='r2', error_score=0, cv=10, n_jobs = -1)

# random_search.fit(base_predictions,y_test)

# best_model = random_search.best_estimator_
# best_params = random_search.best_params_

# print("Best parameters:", best_params)
# print("Test accuracy:", random_search.best_score_)


# meta_model_tuned = ElasticNet(**best_params)
meta_model_tuned = ElasticNet()
meta_model_tuned.fit(base_predictions, y_test)


test_data = pd.read_csv(r'/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/test.csv')
x_test_data = test_data.drop(columns = ['id'])
x_test_data = x_test_data.drop(columns = low_corr)


x_test_data = scaler.transform(x_test_data)


test_data_pred = []

# for model in base_models:
#     y_pred = model.predict(x_test_data)
#     test_data_pred.append(y_pred)

for model in tuned_models:
    y_pred = model.predict(x_test_data)
    test_data_pred.append(y_pred)

test_data_pred = np.column_stack(test_data_pred)


y_pred = meta_model_tuned.predict(test_data_pred)


final = pd.DataFrame(test_data['id'])


final['target'] = y_pred


final


final.to_csv('102203513_shivansh_tuteja_2.csv', index = False)




