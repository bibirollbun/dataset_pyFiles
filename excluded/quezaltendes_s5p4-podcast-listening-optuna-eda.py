# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


X_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv").drop(columns=['Listening_Time_minutes', 'id'])
y_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")['Listening_Time_minutes']
X_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv').drop(columns='id')
ids = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')['id']
ss = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


X_train.describe()


X_train_num = X_train.select_dtypes(include=['float64', 'int64'])
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.ravel()
for i, column in enumerate(X_train_num.columns[:4]):
    sns.boxplot(
        y=X_train_num[column],
        ax=axes[i],
        color="skyblue",
        linewidth=1.5,
        width=0.4
    )
    axes[i].set_title(column, fontsize=12)
plt.show()


X_test_num = X_test.select_dtypes(include=['float64', 'int64'])
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.ravel()
for i, column in enumerate(X_test_num.columns[:4]):
    sns.boxplot(
        y=X_test_num[column],
        ax=axes[i],
        color="skyblue",
        linewidth=1.5,
        width=0.4
    )
    axes[i].set_title(column, fontsize=12)
plt.show()


X_test


full_train = pd.concat([X_train, y_train], axis=1)


Q1_ELm = X_train['Episode_Length_minutes'].quantile(0.25)
Q3_ELm = X_train['Episode_Length_minutes'].quantile(0.75)
IQR_ELm = Q3_ELm - Q1_ELm
lower_bound_ELm = Q1_ELm - 1.5 * IQR_ELm
upper_bound_ELm = Q3_ELm + 1.5 * IQR_ELm



full_train = full_train[
    (full_train['Episode_Length_minutes'] >= lower_bound_ELm) &
    (full_train['Episode_Length_minutes'] <= upper_bound_ELm)
]

median_clean = full_train['Episode_Length_minutes'].median()
X_test['Episode_Length_minutes'] = np.where((X_test['Episode_Length_minutes'] <= lower_bound_ELm) | (X_test['Episode_Length_minutes'] >= upper_bound_ELm), median_clean, X_test['Episode_Length_minutes'])





full_train.isna().sum()


pd.DataFrame(X_test).isna().sum()


Q1_NoA = X_train['Number_of_Ads'].quantile(0.25)
Q3_NoA = X_train['Number_of_Ads'].quantile(0.75)
IQR_NoA = Q3_NoA - Q1_NoA
lower_bound_NoA = Q1_NoA - 1.5 * IQR_NoA
upper_bound_NoA = Q3_NoA + 1.5 * IQR_NoA


full_train = full_train[
    (full_train['Number_of_Ads'] >= lower_bound_NoA) &
    (full_train['Number_of_Ads'] <= upper_bound_NoA)
]
median_NoA = full_train['Number_of_Ads'].median()
X_test['Number_of_Ads'] = np.where((X_test['Number_of_Ads'] <= lower_bound_NoA) | (X_test['Number_of_Ads'] >= upper_bound_NoA), median_NoA, X_test['Number_of_Ads'])


full_train.isna().sum()



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.ravel()
for i, column in enumerate(X_test.select_dtypes(exclude='object').columns[:4]):
    sns.boxplot(
        y=X_test[column],
        ax=axes[i],
        color="skyblue",
        linewidth=1.5,
        width=0.4
    )
    axes[i].set_title(column, fontsize=12)
plt.show()


full_train = full_train.reset_index(drop=True)


import matplotlib.pyplot as plt
for column in full_train.select_dtypes(include=['float64', 'int64']).columns:
    plt.hist(full_train[column], bins=100)
    plt.title(column)
    plt.show()





plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=full_train['Episode_Length_minutes'], 
    y=full_train['Listening_Time_minutes'],
    alpha=0.5,
    color="blue",
)

sns.regplot(
    x=full_train['Episode_Length_minutes'], 
    y=full_train['Listening_Time_minutes'],
    scatter=False, 
    color="red",
    line_kws={"linewidth": 2},
)


plt.legend()
plt.show()


corr_matrix = full_train.select_dtypes(include=['float64', 'int64']).corr()
plt.figure(figsize=(14, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=True, vmax=1, vmin=-1)
plt.show()


full_train


X_train_cat_for_visual = full_train.select_dtypes(include=["object", "category"])
X_train_cat_for_visual_column = X_train_cat_for_visual.columns
plt.figure(figsize=(12, len(X_train_cat_for_visual_column) * 4))
for i, column in enumerate(X_train_cat_for_visual_column, 1):
    plt.subplot(len(X_train_cat_for_visual_column), 1, i)
    ax = sns.countplot(x=full_train[column], order=full_train[column].value_counts().index, palette="Purples_r")
    plt.title(column)
    plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


full_train.isna().sum()


full_train['Guest_Popularity_percentage'] = full_train['Guest_Popularity_percentage'].fillna(full_train['Guest_Popularity_percentage'].median())



full_train.isna().sum()


X_test.isna().sum()


X_test['Guest_Popularity_percentage'] = X_test['Guest_Popularity_percentage'].fillna(full_train['Guest_Popularity_percentage'].median())
X_test['Episode_Length_minutes'] = X_test['Episode_Length_minutes'].fillna(full_train['Episode_Length_minutes'].median())
X_test.isna().sum()


X_train_num_ac = full_train.select_dtypes(exclude='object').drop(columns='Listening_Time_minutes')


combined_cat = pd.concat([
    full_train.select_dtypes(include='object'),
    X_test.select_dtypes(include='object')
], axis=0).reset_index(drop=True)
combined_cat_dummies = pd.get_dummies(combined_cat)


X_train_cat_ac = combined_cat_dummies.iloc[:len(full_train), :]
X_test_cat_ac = combined_cat_dummies.iloc[len(full_train):, :].reset_index(drop=True)


X_train_num_ac = full_train.select_dtypes(exclude='object').drop(columns='Listening_Time_minutes').reset_index(drop=True)
X_test_num = X_test.select_dtypes(exclude='object').reset_index(drop=True)


X_train_final = pd.concat([X_train_num_ac, X_train_cat_ac], axis=1)
X_test_final = pd.concat([X_test_num, X_test_cat_ac], axis=1)


X_test_final = X_test_final[X_train_final.columns]





X_train_final['ad_ratio'] = X_train_final['Episode_Length_minutes'] / np.where(X_train_final['Number_of_Ads'] < 1, 0.5, X_train_final['Number_of_Ads'])
X_test_final['ad_ratio'] = X_test_final['Episode_Length_minutes'] / np.where(X_test_final['Number_of_Ads'] < 1, 0.5, X_test_final['Number_of_Ads'])


X_train_final['delta_popular'] = X_train_final['Host_Popularity_percentage'] - X_train_final['Guest_Popularity_percentage']
X_test_final['delta_popular'] = X_test_final['Host_Popularity_percentage'] - X_test_final['Guest_Popularity_percentage']


X_train_final['is_long'] = np.where(X_train_final['Episode_Length_minutes'] > 60, 1, 0)
X_test_final['is_long'] = np.where(X_test_final['Episode_Length_minutes'] > 60, 1, 0)





X_test_final


import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

scorer = make_scorer(rmse, greater_is_better=False)

def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
        'random_state': 42,
        'n_jobs': -1
    }

    
    model = XGBRegressor(**params)
    
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    
    scores = cross_val_score(
        estimator=model,
        X=X_train_final,
        y=full_train['Listening_Time_minutes'],
        cv=kfold,
        scoring=scorer,
        n_jobs=-1
    )
    
    return np.mean(np.abs(scores))


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20, timeout=3600)

print("Best trial:")
trial = study.best_trial
print(f"  RMSE: {trial.value:.4f}")
print("  Best params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")




best_model = XGBRegressor(**study.best_params, objective='reg:squarederror')
best_model.fit(X_train_final, full_train['Listening_Time_minutes'])
y_pred = best_model.predict(X_test_final)


y_pred


plt.hist(y_pred, bins=100)
plt.show()


pd.DataFrame({'id': ids, 'Listening_Time_minutes': y_pred}).to_csv('submission.csv', index=False)

