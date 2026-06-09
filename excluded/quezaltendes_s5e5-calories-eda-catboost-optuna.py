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
import matplotlib.pyplot as plt
import seaborn as sns
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.filterwarnings("ignore") 


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
ss = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


print(f'NAs: Train: {train.isna().sum()}, Test: {test.isna().sum()}')


y_train = train['Calories']
X_train = train.drop(columns=['id', 'Calories'])
ids = test['id']
X_test = test.drop(columns='id')


X_train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})
X_test['Sex'] = test['Sex'].map({'female': 0, 'male': 1})


train


X_train.describe()


X_test.describe()


X_train_num = X_train.select_dtypes(include=['float64', 'int64'])
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axes = axes.ravel()
for i, column in enumerate(X_train_num.columns[:4]):
    sns.boxplot(
        y=X_train_num[column],
        ax=axes[i],
        color="skyblue",
        linewidth=1.9,
        width=0.5
    )
    axes[i].set_title(column, fontsize=12)
plt.show()


### NOT OUTLIERS!!!



for column in X_train.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(10, 4))
    
    plt.hist(X_train[column], 
             bins=50,
             color='steelblue',
             edgecolor='white',
             alpha=0.8)
    
    mean_val = X_train[column].mean()
    median_val = X_train[column].median()
    
    plt.axvline(mean_val, color='crimson', linestyle='--', linewidth=1.5, label=f'Mean ({mean_val:.1f})')
    plt.axvline(median_val, color='navy', linestyle=':', linewidth=1.5, label=f'Median ({median_val:.1f})')
    
    plt.title(column, fontsize=12, pad=12)
    plt.xlabel('')
    plt.ylabel('Count', fontsize=10)
    
    plt.grid(axis='y', alpha=0.4)
    plt.legend(frameon=False, fontsize=9) 
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.hist(y_train, 
         bins=100,
         color='steelblue',
         edgecolor='white',
         alpha=0.8)

mean_val = y_train.mean()
median_val = y_train.median()

plt.axvline(mean_val, color='crimson', linestyle='--', linewidth=1.5, label=f'Mean ({mean_val:.1f})')
plt.axvline(median_val, color='navy', linestyle=':', linewidth=1.5, label=f'Median ({median_val:.1f})')

plt.title('Original Scale', fontsize=12, pad=12)
plt.xlabel('Value', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.grid(axis='y', alpha=0.4)
plt.legend(frameon=False, fontsize=9)

plt.subplot(1, 2, 2)
plt.hist(np.log1p(y_train), 
         bins=100,
         color='seagreen',
         edgecolor='white',
         alpha=0.8)

log_mean = np.log1p(y_train).mean()
log_median = np.log1p(y_train).median()

plt.axvline(log_mean, color='darkorange', linestyle='--', linewidth=1.5, label=f'Mean ({log_mean:.1f})')
plt.axvline(log_median, color='purple', linestyle=':', linewidth=1.5, label=f'Median ({log_median:.1f})')

plt.title('Log Scale (np.log1p)', fontsize=12, pad=12)
plt.xlabel('Log(Value)', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.grid(axis='y', alpha=0.4)
plt.legend(frameon=False, fontsize=9)

plt.tight_layout()
plt.show()


y_train = np.log1p(y_train)


plt.figure(figsize=(9, 6))
sns.heatmap(X_train.corr(), annot=True,cmap='crest')


for column in X_train.drop(columns=['Sex', 'Age', 'Height', 'Weight']):
    plt.figure(figsize=(6, 3))
    sns.scatterplot(
        x=X_train[column], 
        y=y_train,
        alpha=0.5,
        color="blue",
    )
    
    sns.regplot(
        x=X_train[column], 
        y=y_train,
        scatter=False, 
        color="red",
        line_kws={"linewidth": 2},
    )
    
    
    plt.legend()
    plt.show()


X_train['high_T'] = np.where(X_train['Body_Temp'] > 40, 1, 0)
X_train['long_or_not'] = np.where(X_train['Duration'] > 20, 1, 0)
X_test['high_T'] = np.where(X_test['Body_Temp'] > 40, 1, 0)
X_test['long_or_not'] = np.where(X_test['Duration'] > 20, 1, 0)



X_train


'''import optuna
from catboost import CatBoostRegressor, Pool, cv
import numpy as np

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 2000),
        'depth': trial.suggest_int('depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10),
        'random_seed': 42,
        'verbose': False,
        'loss_function': 'RMSE'
    }

    cv_dataset = Pool(
        data=X_train, 
        label=y_train,
        cat_features=list(X_train.select_dtypes(include=['category', 'object']).columns)
    )
    
    cv_results = cv(
        pool=cv_dataset,
        params=params,
        fold_count=5,
        shuffle=True,
        partition_random_seed=42,
        verbose=False
    )
    
    best_rmse = np.min(cv_results['test-RMSE-mean'])
    
    return best_rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, timeout=7200)

print(f"Best trial:")
trial = study.best_trial
print(f"  Best RMSE: {trial.value:.4f}")
print("  Optimized params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

best_params = study.best_params'''


from xgboost import XGBRegressor


model = XGBRegressor(max_depth=20, colsample_bytree=0.7, subsample=0.9, n_estimators=3000, learning_rate=0.02,
                            gamma=0.01, max_delta_step=2, eval_metric='rmse')
model.fit(X_train, y_train)
pred = np.expm1(model.predict(X_test))


ss


pd.DataFrame({'id': ids, 'Calories': pred}).to_csv('submission.csv', index=False)

