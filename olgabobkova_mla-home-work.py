!pip install -U lightautoml
import pandas as pd

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import torch

# Библиотеки для AUTOML
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task


df_train = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip')
df_test = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/test.csv.zip')
df_submision = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/sample_submission.csv.zip')


df_train


df_test


df_submision


RANDOM_STATE = 42 # fixed random state for various reasons
TEST_SIZE = 0.15 # Test size for metric check


train, test = train_test_split(df_train, 
                                     test_size=TEST_SIZE,
                                     random_state=RANDOM_STATE)

X_train = train.drop('y', axis=1)
y_train = train['y']

X_test = test.drop('y', axis=1)
y_test = test['y']

print(f'размеры таблиц: тренировочный = {train.shape}, тестовый = {test.shape}')


metrics_df = pd.DataFrame(columns=['model', 'train_R2', 'train_MAE', 'train_RMSE', 'test_R2', 'test_MAE', 'test_RMSE'])
metrics_df


base_y_train = y_train.copy()
base_y_train[:] = y_train.mean()

base_y_test = y_test.copy()
base_y_test[:] = y_train.mean()


r2_test = r2_score(y_test, base_y_test)
mae_test = mean_absolute_error(y_test, base_y_test)
rmse_test = root_mean_squared_error(y_test, base_y_test)

r2_train = r2_score(y_train, base_y_train)
mae_train = mean_absolute_error(y_train, base_y_train)
rmse_train = root_mean_squared_error(y_train, base_y_train)

metrics_df.loc[len(metrics_df)] = ['baseline', r2_train, mae_train, rmse_train, r2_test, mae_test, rmse_test]
metrics_df.round(4)


from catboost import CatBoostRegressor

cat_features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']

model = CatBoostRegressor(
    eval_metric='RMSE',
    early_stopping_rounds=50,
    verbose=100,
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    cat_features = cat_features
)


r2_test = r2_score(y_test, model.predict(X_test))
mae_test = mean_absolute_error(y_test, model.predict(X_test))
rmse_test = root_mean_squared_error(y_test, model.predict(X_test))

r2_train = r2_score(y_train, model.predict(X_train))
mae_train = mean_absolute_error(y_train, model.predict(X_train))
rmse_train = root_mean_squared_error(y_train, model.predict(X_train))

metrics_df.loc[len(metrics_df)] = ['catboost', r2_train, mae_train, rmse_train, r2_test, mae_test, rmse_test]
metrics_df.round(4)


TIMEOUT = 50

np.random.seed(RANDOM_STATE)

task = Task(
    'reg',
    loss = 'mse',
    metric = lambda y_true, y_pred: r2_score(y_true, y_pred)
)

roles = {
    'target': 'y',
    'drop': ['ID'],
}


automl = TabularAutoML(
    task = task, 
    timeout = TIMEOUT,
)

y_train_pred = automl.fit_predict(
    train,
    roles = roles,
    verbose=1
)


r2_test = r2_score(y_test, automl.predict(test).data[:,0])
mae_test = mean_absolute_error(y_test, automl.predict(test).data[:,0])
rmse_test = root_mean_squared_error(y_test, automl.predict(test).data[:,0])

r2_train = r2_score(y_train, automl.predict(train).data[:,0])
mae_train = mean_absolute_error(y_train, automl.predict(train).data[:,0])
rmse_train = root_mean_squared_error(y_train, automl.predict(train).data[:,0])

metrics_df.loc[len(metrics_df)] = ['AutoML', r2_train, mae_train, rmse_train, r2_test, mae_test, rmse_test]

print('Сравнение метрик трех подходов:')
metrics_df.round(4)

