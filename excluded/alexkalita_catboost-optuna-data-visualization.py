# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
sample = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train.head()


train.info()


train.isna().sum()


train = train.apply(lambda x: x.fillna(x.value_counts().index[0]))
train.isna().sum()


del train['id']
train.head()


del test['id']
test.head()


test = test.apply(lambda x: x.fillna(x.value_counts().index[0]))
test.isna().sum()


import matplotlib.pyplot as plt

# aesthetics
default_color_1 = 'blue'
default_color_2 = 'green'
default_color_3 = 'darkred'

# define features and target
features_num = ['Compartments','Weight Capacity (kg)']

features_cat = ['Brand', 'Material','Laptop Compartment','Waterproof', 'Style', 'Color']

target = 'Price'


# plot histograms (train and test)
for f in features_num:
    plt.figure(figsize=(12,3))
    ax1 = plt.subplot(1,2,1)
    train[f].plot(kind='hist', bins=20, color=default_color_1)
    plt.title(f + ' - Train')
    plt.grid()
    ax2 = plt.subplot(1,2,2, sharex=ax1)
    test[f].plot(kind='hist', bins=20, color=default_color_2)
    plt.title(f + ' - Test')
    plt.grid()
    plt.show()


# plot categorical feature distributions (train and test)
for f in features_cat:
    plt.figure(figsize=(14,3))
    ax1 = plt.subplot(1,2,1)
    train[f].value_counts().sort_index().plot(kind='bar', color=default_color_1)
    plt.title(f + ' - Train')
    plt.grid()
    ax2 = plt.subplot(1,2,2)
    test[f].value_counts().sort_index().plot(kind='bar', color=default_color_2)
    plt.title(f + ' - Test')
    plt.grid()
    plt.show()




from catboost import CatBoostRegressor

features = features_num + features_cat




import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from IPython.display import clear_output


X = train[features]
y = train[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



def objective(trial):
    cat_params = dict(
        iterations=trial.suggest_int("iterations", 100, 1000),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        depth=trial.suggest_int("depth", 3, 12),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
        bagging_temperature=trial.suggest_float('bagging_temperature', 0, 2.5),
        random_strength=trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        task_type='GPU',
        early_stopping_rounds=200,
        verbose=False
    )
    
    model = CatBoostRegressor(**cat_params)
    X_train_pool = Pool(X_train, y_train, cat_features = features_cat)
    X_valid_pool = Pool(X_val, y_val, cat_features = features_cat)
    model.fit(X=X_train_pool, eval_set=X_valid_pool)
    
    y_pred = model.predict(X_val)
    score = mean_squared_error(y_val, y_pred)
    
    return score


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=2000)
clear_output()


best_params = study.best_params

final_model = CatBoostRegressor(**best_params)
final_model


X_train_pool = Pool(X, y, cat_features = features_cat)
final_model.fit(X=X_train_pool)

pred_test = final_model.predict(test[features])


# create submission file
df_sub_GLM = sample.copy()
df_sub_GLM[target] = pred_test
df_sub_GLM.to_csv('submission_GLM.csv', index=False)
df_sub_GLM.head(10)

