# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
ss = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


train


train.describe()


test


y_train = train['Listening_Time_minutes']
X_train = train.drop(columns=['id', 'Listening_Time_minutes'])


X_train_num_visual = X_train.select_dtypes(include=['int64', 'float64'])


import matplotlib.pyplot as plt
for column in X_train_num_visual.columns:
    plt.hist((X_train_num_visual[column]), bins=100)
    plt.title(column)
    plt.show()


import matplotlib.pyplot as plt
for column in X_train_num_visual.columns:
    plt.hist(np.log1p(X_train_num_visual[column]), bins=100)
    plt.title(column)
    plt.show()



plt.hist((y_train ** 0.5), bins=100)
plt.axvline(x=7.3, color='r', linestyle='--', label='x = 3.99')
plt.title('y_train')
plt.show()


plt.figure(figsize=(14, 8))
plt.hist(np.log1p(y_train), bins=100)
plt.axvline(x=4.05, color='r', linestyle='--', label='x = 3.99')
plt.title('y_train')
plt.show()
plt.figure(figsize=(14, 8))
plt.hist(np.log1p(X_train['Episode_Length_minutes']), bins=100)
plt.axvline(x=4.05, color='r', linestyle='--', label='x = 3.99')
plt.show()


plt.figure(figsize=(14, 8))
plt.hist((y_train), bins=100)
plt.axvline(x=4.05, color='r', linestyle='--', label='x = 3.99')
plt.title('y_train')
plt.show()



X_train_num_visual['added_feat'] = (X_train_num_visual['Episode_Length_minutes'] > 54.5).astype(int)



X_train


X_test = test


features = ['Episode_Length_minutes', 'Number_of_Ads', 'Host_Popularity_percentage', 'Guest_Popularity_percentage']

X_train_clean = X_train.copy()

for feature in features:
    Q1 = X_train[feature].quantile(0.25)
    Q3 = X_train[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    median = X_train[feature].median()
    
    X_train_clean[feature] = np.where(
        (X_train_clean[feature] < lower) | (X_train_clean[feature] > upper) | (X_train_clean[feature].isna() == True),
        median,
        X_train_clean[feature]
    )


X_train_clean.isna().sum()


features = ['Episode_Length_minutes', 'Number_of_Ads', 'Host_Popularity_percentage', 'Guest_Popularity_percentage']

X_test_clean = X_test.copy()

for feature in features:
    Q1 = X_test[feature].quantile(0.25)
    Q3 = X_test[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    median = X_test[feature].median()
    X_test_clean[feature] = np.where(
        (X_test_clean[feature] < lower) | (X_test_clean[feature] > upper) | (X_test_clean[feature].isna() == True),
        median,
        X_test_clean[feature]
    )


X_test_clean.isna().sum()


C = 0.5


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_train['Episode_Length_minutes'], 
    y=y_train,
    alpha=0.5,
    color="blue",
)

sns.regplot(
    x=X_train['Episode_Length_minutes'], 
    y=y_train,
    scatter=False, 
    color="red",
    line_kws={"linewidth": 2},
)


plt.legend()
plt.show()


plt.hist(((X_train_num_visual['Episode_Length_minutes'])), bins=100)
plt.show()
plt.hist(((np.log1p(X_train['Episode_Length_minutes'] + C))), bins=100)
plt.show()
plt.hist(((np.log1p(X_test_clean['Episode_Length_minutes'].dropna()**2))), bins=100)
plt.show()



plt.figure(figsize=(12, 8))
sns.heatmap(
    pd.concat([X_train_num_visual, y_train], axis=1).corr(),
    annot=True,  
    fmt=".4f",      
    cmap="coolwarm", 
    vmin=-1, vmax=1, 
    linewidths=0.5
)
plt.title("Corr Matrix")
plt.show()


X_train_cb = pd.concat([train, pd.Series(np.random.rand(750000))], axis=1).dropna()
X_train_cb


(X_train_num_visual.dropna()['added_feat'])


train_data = Pool(data=(pd.concat([X_train_cb.drop(columns='Listening_Time_minutes'), (X_train_num_visual.dropna()['added_feat'])], axis=1)), label=X_train_cb['Listening_Time_minutes'], cat_features=['Podcast_Name',  'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'])


model = CatBoostRegressor(iterations=100, verbose=False)
model.fit(train_data)


importance = model.get_feature_importance()


importance


X_train_num_visual.dropna()['added_feat']


X_train_num_visual.dropna()['added_feat'].column = 'q'


lsit = {}
for i in range(len(importance)):
    lsit[importance[i]] = (pd.concat([X_train_cb.drop(columns='Listening_Time_minutes'), pd.Series(X_train_num_visual.dropna()['added_feat'])])).columns[i]


lsit


df = pd.DataFrame({
    'Feature': lsit.values(),
    'Importance': lsit.keys()
}).sort_values('Importance', ascending=False)


sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

colors = sns.color_palette("viridis", len(df))

barplot = sns.barplot(
    x='Importance', 
    y='Feature', 
    data=df, 
    palette=colors,
)

plt.title('Feature importances', fontsize=16, pad=20)
plt.xlabel('Importances', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

for p in barplot.patches:
    width = p.get_width()
    plt.text(
        width * 1.02, 
        p.get_y() + p.get_height()/2, 
        f'{width:.2f}', 
        va='center', 
        fontsize=9,
        color='black'
    )

plt.subplots_adjust(left=0.3)

plt.show()





X_train


X_train_ac = X_train_clean.drop(columns=['Episode_Title', 'Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'])


X_train_ac.isna().sum()


X_train_ac.median()


plt.figure(figsize=(14, 8))
plt.hist(X_train_ac['Episode_Length_minutes'], bins=100)
plt.axvline(x=63.84, color='r', linestyle='--')
plt.title('Episode_Length_minutes')
plt.show()


X_train_ac['Ad_ratio'] = X_train_ac['Episode_Length_minutes'] / np.where(X_train_ac['Number_of_Ads'] == 0, 0.5, X_train_ac['Number_of_Ads'])
X_train_ac['Long_or_not'] = np.where(X_train_ac['Episode_Length_minutes'] > 63.84, 1, 0)
X_train_ac['Host_guest'] = X_train_ac['Host_Popularity_percentage'] - X_train_ac['Guest_Popularity_percentage']


test


X_train_ac


ids = test['id']
X_test = X_test_clean.drop(columns=['id', 'Episode_Title', 'Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'])


X_test['Ad_ratio'] = X_test['Episode_Length_minutes'] / np.where(X_test['Number_of_Ads'] == 0, 0.5, X_test['Number_of_Ads'])
X_test['Long_or_not'] = np.where(X_test['Episode_Length_minutes'] > 63.84, 1, 0)
X_test['Host_guest'] = X_test['Host_Popularity_percentage'] - X_test['Guest_Popularity_percentage']





'''
import optuna
from optuna import Trial
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
import numpy as np

cv = KFold(n_splits=5, shuffle=True, random_state=42)
def objective(trial: Trial) -> float:
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
    }
    
    model = xgb.XGBRegressor(**params, random_state=42)
    

    scores = cross_val_score(
        estimator=model,
        X=X_train_ac,
        y=y_train,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    return np.mean(scores)


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
)


study.optimize(objective, n_trials=10, show_progress_bar=True)

print(f"{-study.best_value:.4f}")
for key, value in study.best_params.items():
    print(f"{key}: {value}")
'''


optimized_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'lambda': 0.00012998,
    'alpha': 0.00054868,
    'learning_rate': 0.0234706,
    'max_depth': 9,
    'min_child_weight': 2,
    'subsample': 0.532526,
    'colsample_bytree': 0.974443,
    'gamma': 0.530953,
    'n_estimators': 1636,
    'random_state': 42,
    'n_jobs': -1
}

model = XGBRegressor(**optimized_params)


model.fit(X_train_ac, y_train)
y_test_pred = model.predict(X_test)


plt.figure(figsize=(14, 8))
plt.hist((y_test_pred), bins=100)
plt.title('y_test')
plt.show()





ss


pd.DataFrame({'id': ids, 'Listening_Time_minutes': y_test_pred}).to_csv('submission.csv', index=False)

