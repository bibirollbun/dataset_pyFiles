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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import optuna
from optuna.samplers import TPESampler

import catboost
from catboost import CatBoostRegressor
from catboost import Pool


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

%load_ext cudf.pandas


import warnings
warnings.filterwarnings("ignore")


train_meow = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_meow = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
del train_meow['id']
del test_meow['id']
train_meow.head()


train_meow.info()


sns.histplot(train_meow['Listening_Time_minutes'], bins=30, kde=True)

plt.xlabel('Listening Time (minutes)')
plt.ylabel('Count')
plt.title('Histogram of Listening Time')
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))

axes = axes.flatten()

cat_cols = train_meow.select_dtypes(include='object')

for i, meow in enumerate(cat_cols):
    sns.histplot(train_meow[meow], ax=axes[i])
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()


print(train_meow['Podcast_Name'].nunique())
print(train_meow['Episode_Title'].nunique())


combined = pd.concat([train_meow, test_meow], axis=0)

day_mapping = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

cat_cols = combined.select_dtypes(include='object').columns

for i, c in enumerate(cat_cols):
    if c == 'Publication_Day':  
        continue
        
    combined[c], _ = pd.factorize(combined[c])
    
    train_meow[c] = combined[c].iloc[:len(train_meow)].astype("float32")
    test_meow[c] = combined[c].iloc[len(train_meow):].astype("float32")

train_meow['Publication_Day'] = train_meow['Publication_Day'].replace(day_mapping).astype('int8')
test_meow['Publication_Day'] = test_meow['Publication_Day'].replace(day_mapping).astype('int8')


def filnal(train_meow):
    for i in train_meow.columns:
        if train_meow[i].isna().any():
            train_meow[i] = train_meow[i].fillna(train_meow[i].median())
    return train_meow
train_meow = filnal(train_meow)
test_meow = filnal(test_meow)


train_meow.head()


from sklearn.feature_selection import mutual_info_regression
cols = [i for i in train_meow if i != 'Listening_Time_minutes']

train_sampe = train_meow.sample(frac = 0.2, random_state = 42)

mutual_info = mutual_info_regression(train_sampe[cols], train_sampe.Listening_Time_minutes, random_state=42)
mutual_info_series = pd.Series(mutual_info)
mutual_info_series.index = cols
mutual_info_df = pd.DataFrame(mutual_info_series.sort_values(ascending=False), columns=["Numerical_Feature_MI"])
styled_mutual_info = mutual_info_df.style.background_gradient("cool")
styled_mutual_info


# later


# x_train, x_test, y_train, y_test = train_test_split(train_meow[cols], train_meow['Listening_Time_minutes'], random_state = 42)

# def objective(trial):
#     # Параметры, которые вы хотите оптимизировать
#     params = {
#         'loss_function': 'RMSE',  # Для регрессии используем RMSE
#         'eval_metric': 'RMSE',
#         'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 0.1),
#         'iterations': trial.suggest_int('iterations', 500, 2000),
#         'depth': trial.suggest_int('depth', 4, 12),
#         'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-5, 10.0),
#         'random_strength': trial.suggest_uniform('random_strength', 0, 10),
#         'task_type': 'GPU',
#         'random_seed': 42,
#         'verbose': False
#     }

#     # Обучение модели
#     model = CatBoostRegressor(**params)
#     model.fit(x_train, y_train, eval_set=(x_test, y_test), verbose=False)

#     # Прогнозирование на валидационном наборе
#     preds = model.predict(x_test)
#     rmse = mean_squared_error(y_test, preds, squared=False)
#     return rmse
    
# sampler = TPESampler()
# study = optuna.create_study(sampler=sampler, direction='minimize')

# study.optimize(objective, n_trials=25)
# print("Лучшие гиперпараметры:", study.best_params)
# print("Лучшая RMSE:", study.best_value)


params = {'learning_rate': 0.04906272040107097,
        'iterations': 1765,
        'depth': 10,
        'l2_leaf_reg': 3.17403601477596,
        'random_strength': 9.20446068819469,
        'loss_function': 'RMSE', 
        'eval_metric': 'RMSE',
        'task_type': 'GPU',
        'random_seed': 42,
        'verbose': False
}


FOLDS = 7

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_meow[cols]))
test_preds = np.zeros(len(test_meow))

for fold, (train_idx, valid_idx) in enumerate(kf.split(train_meow)):
    print(f"Fold {fold + 1}")  
    X_train = train_meow[cols].loc[train_idx].reset_index(drop=True).copy()
    y_train = train_meow['Listening_Time_minutes'].iloc[train_idx].values
        
    X_valid = train_meow[cols].loc[valid_idx].reset_index(drop=True).copy()
    y_valid = train_meow['Listening_Time_minutes'].iloc[valid_idx].values
        
    X_test = test_meow.reset_index(drop=True).copy()

    model = CatBoostRegressor(**params)
    train_pool = Pool(X_train, y_train)
    valid_pool = Pool(X_valid, y_valid)
    X_test_pool = Pool(X_test)
    model.fit(X=train_pool, eval_set=valid_pool, verbose=500, early_stopping_rounds=100)

    oof_preds[valid_idx] = model.predict(X_valid)  
    test_preds += model.predict(test_meow) / FOLDS 


sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
sub.Listening_Time_minutes = test_preds
sub.to_csv("submission.csv", index=False)
!head submission.csv




