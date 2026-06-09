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
import sklearn
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor 
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
training_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


y_train = train['Price']
y_extra = training_extra['Price']
X_train_extra = training_extra.drop(columns=['Price'])
X_train = pd.concat([train, X_train_extra], axis=0)
y_train = pd.concat([y_train, y_extra], axis=0)



X_train = X_train.drop(columns=['id', 'Price', 'Color', 'Compartments', 'Waterproof'])


X_train


y_train


X_train.isna().sum()


X_test = test.drop(columns=['id', 'Color', 'Compartments', 'Waterproof']) 


'''
random_array = []
import random
for i in range(X_train.shape[0]):
    random_array.append(random.randint(0, 1))
random = pd.DataFrame({'random': random_array})


importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)
# after using Linear Regression model'''


X_train[['Brand', 'Material', 'Size', 'Style', 'Laptop Compartment']] = \
X_train[['Brand', 'Material', 'Size', 'Style', 'Laptop Compartment']].fillna('Unknown')


X_test[['Brand', 'Material', 'Size', 'Style', 'Laptop Compartment']] = \
X_test[['Brand', 'Material', 'Size', 'Style', 'Laptop Compartment']].fillna('Unknown')


X_train[['Weight Capacity (kg)']] = X_train[['Weight Capacity (kg)']].fillna(X_train[['Weight Capacity (kg)']].sum() / X_train.shape[0])
X_test[['Weight Capacity (kg)']] = X_test[['Weight Capacity (kg)']].fillna(X_test[['Weight Capacity (kg)']].sum() / X_test.shape[0])


cat_column = X_train.columns[X_train.dtypes == 'object']
cat_columns = [x for x in cat_column]


cat_columns


num_column = X_train.columns[X_train.dtypes != 'object']
num_columns = [x for x in num_column]


num_columns[0]




plt.hist(X_train['Weight Capacity (kg)'], bins=500)
plt.show()


cat_train = X_train[cat_columns]
cat_test = X_test[cat_columns]

cat = pd.concat([cat_train, cat_test], axis=0)
cat = pd.get_dummies(cat)

cat_train = cat[:X_train.shape[0]]
cat_test = cat[X_train.shape[0]:]

num_train = X_train[num_columns]
cat_train = pd.get_dummies(cat_train)

X_train = pd.concat([num_train, cat_train], axis=1)

num_test = X_test[num_columns]
cat_test = pd.get_dummies(cat_test)

X_test = pd.concat([num_test, cat_test], axis=1)


X_test


X_train


plt.figure(figsize=(10, 8))
sns.heatmap(X_train.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("corr")
plt.show()


X_train_little = X_train[:30000]
y_train_little = y_train[:30000]


import optuna


sklearn.metrics.get_scorer_names()


'''
import optuna
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 10, 400) 
    max_depth = trial.suggest_int("max_depth", 2, 31) 
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    gamma = trial.suggest_float("gamma", 0, 10)
    reg_alpha = trial.suggest_float("reg_alpha", 0, 10)  
    reg_lambda = trial.suggest_float("reg_lambda", 0, 10)  
    

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        n_jobs=-1
    )
    

    score = cross_val_score(model, X_train_little, y_train_little, cv=5, scoring="neg_root_mean_squared_error").mean()
    
    return score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Ð’Ñ‹Ð²Ð¾Ð´ Ð»ÑƒÑ‡ÑˆÐ¸Ñ… Ð¿Ð°Ñ€Ð°Ð¼ÐµÑ‚Ñ€Ð¾Ð²
print("Best params", study.best_params)
print("best RMSE:", -study.best_value)
'''


"""
import optuna



def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 1000) 
    max_depth = trial.suggest_int("max_depth", 3, 20)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5, log=True)
    num_leaves = trial.suggest_int("num_leaves", 20, 300) 
    min_child_samples = trial.suggest_int("min_child_samples", 5, 50) 
    subsample = trial.suggest_float("subsample", 0.5, 1.0) 
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 10.0) 
    reg_lambda = trial.suggest_float("reg_lambda", 0.0, 10.0)

    model = LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    score = cross_val_score(model, X_train_little, y_train_little, cv=5, scoring="neg_root_mean_squared_error").mean()
    
    return score 


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100)

print("Params", study.best_params)
print("Best RMSE Score", -study.best_value)
"""


'''import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score

def objective(trial):

    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int("max_depth", 3, 16)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5, log=True)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 10)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.5, 1.0) 
    grow_policy = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]) 


    model = CatBoostRegressor(
        iterations=n_estimators,
        depth=max_depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        subsample=subsample,
        colsample_bylevel=colsample_bylevel,
        grow_policy=grow_policy,
        loss_function="RMSE",  
        verbose=0, 
        random_state=42
    )
    score = cross_val_score(model, X_train_little, y_train_little, cv=5, scoring="neg_root_mean_squared_error").mean()
    
    return score 

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50) 

print("Params:", study.best_params)
print("RMSE:", -study.best_value)
'''


model = LGBMRegressor(
    n_estimators=121,
    max_depth=3,
    learning_rate=0.010066896797204919,
    num_leaves=141,
    min_child_samples=18,
    subsample=0.7208561936055725,
    colsample_bytree=0.7659277366009701,
    reg_alpha=9.21259737143864,
    reg_lambda=2.8601819883108437,
    verbosity=-1
)



model.fit(X_train, y_train)


y_pred = model.predict(X_test)


output = pd.DataFrame({'id': test['id'], 'Price': y_pred})
output.to_csv('submission.csv', index=False)

