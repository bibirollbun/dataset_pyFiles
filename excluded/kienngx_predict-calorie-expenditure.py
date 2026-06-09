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


sample = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
train.head()


train.info()


Sex_dummies = pd.get_dummies(train['Sex'], prefix='Sex')
train = pd.concat([train, Sex_dummies], axis=1)
train.drop('Sex', axis=1, inplace=True)
train.info()


import matplotlib.pyplot as plt
import seaborn as sns
for col in train.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(train[col], kde=True)  # use histplot
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


numeric_df = train.drop(columns=['Sex_female', 'Sex_male', 'id'])
corr = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def rmsle(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.maximum(0, np.array(y_pred))
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

X = train.drop(columns='Calories')
y = train['Calories']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=16)
model = LinearRegression()
model.fit(X_train,y_train)
accuracy = model.score(X_test, y_test)
y_pred = model.predict(X_test)
rmsle_score = rmsle(y_test, y_pred)
# Print results
print(f"Test R² Accuracy: {accuracy:.2f}")
print(f"Test RMSLE: {rmsle_score:.4f}")


import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import make_scorer
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

neg_rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

param_dist_lgb = {
    'num_leaves': [31, 50, 70, 100],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [-1, 5, 10, 15, 20],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_samples': [5, 10, 20, 30, 50]
}

lgb_model = LGBMRegressor(random_state=16, device='gpu')

random_search_lgb = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=param_dist_lgb,
    n_iter=50,
    scoring=neg_rmsle_scorer,
    cv=KFold(n_splits=5, shuffle=True, random_state=16),
    random_state=16,
    n_jobs=-1,
    verbose=1
)

random_search_lgb.fit(X, y)

best_lgb = random_search_lgb.best_estimator_
print(f"Best LGB Params: {random_search_lgb.best_params_}")
print(f"Best LGB RMSLE: {-random_search_lgb.best_score_:.4f}")

import optuna

def objective_lgb(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 31, 100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50)
    }
    model = LGBMRegressor(**params, random_state=16, device='gpu')
    kf = KFold(n_splits=5, shuffle=True, random_state=16)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        scores.append(rmsle(y_val, y_pred))
    return np.mean(scores)

study_lgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=16))
study_lgb.optimize(objective_lgb, n_trials=50)

print(f"Optuna Best LGB Params: {study_lgb.best_params}")
print(f"Optuna Best LGB RMSLE: {study_lgb.best_value:.4f}")



from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV

param_dist_cat = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.001, 0.01, 0.05, 0.1],
    'iterations': [100, 200, 500, 1000],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'bagging_temperature': [0.0, 0.2, 0.5, 0.8, 1.0],
    'border_count': [32, 64, 128, 254]
}

cat_model = CatBoostRegressor(task_type='GPU', silent=True, random_state=16)

random_search_cat = RandomizedSearchCV(
    estimator=cat_model,
    param_distributions=param_dist_cat,
    n_iter=50,
    scoring=neg_rmsle_scorer,
    cv=KFold(n_splits=5, shuffle=True, random_state=16),
    random_state=16,
    n_jobs=-1,
    verbose=1
)

random_search_cat.fit(X, y)

best_cat = random_search_cat.best_estimator_
print(f"Best Cat Params: {random_search_cat.best_params_}")
print(f"Best Cat RMSLE: {-random_search_cat.best_score_:.4f}")

import optuna

def objective_cat(trial):
    params = {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 9),
        'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_categorical('border_count', [32, 64, 128, 254])
    }
    model = CatBoostRegressor(**params, task_type='GPU', silent=True, random_state=16)
    kf = KFold(n_splits=5, shuffle=True, random_state=16)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        scores.append(rmsle(y_val, y_pred))
    return np.mean(scores)

study_cat = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=16))
study_cat.optimize(objective_cat, n_trials=50)

print(f"Optuna Best Cat Params: {study_cat.best_params}")
print(f"Optuna Best Cat RMSLE: {study_cat.best_value:.4f}")


