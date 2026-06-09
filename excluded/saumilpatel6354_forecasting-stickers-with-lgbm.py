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
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge 

train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')

train.head()

train.info()

train = train.dropna()

def transform_date(df, col):
    df[col] = pd.to_datetime(df[col])
    
    df['year'] = df[col].dt.year.astype('int')
    df['quarter'] = df[col].dt.quarter.astype('int')
    df['month'] = df[col].dt.month.astype('int')
    df['day'] = df[col].dt.day.astype('int')
    df['day_of_week'] = df[col].dt.dayofweek.astype('int')
    df['week_of_year'] = df[col].dt.isocalendar().week.astype('int')
    
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['year_sin'] = np.sin(2 * np.pi * df['year'] / 7)
    df['year_cos'] = np.cos(2 * np.pi * df['year'] / 7)
    
    df['group'] = (df['year'] - 2010) * 48 + df['month'] * 4 + df['day'] // 7
    
    return df

train = transform_date(train, 'date')
test = transform_date(test, 'date')

train = train.drop(columns=['date'], axis=1)
test = test.drop(columns=['date'], axis=1)

cat_cols = ['country','store','product']

label_encoders = {}  
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    label_encoders[col] = le
    
sns.set(style="whitegrid")

plt.figure(figsize=(8, 6))
sns.histplot(train['num_sold'], kde=True, bins=30, color='violet')

plt.title('Distribution of Sticker Sales (num_sold)', fontsize=16)
plt.xlabel('Number of Stickers Sold')
plt.ylabel('Frequency')

plt.show()


train['num_sold'] = np.log1p(train['num_sold'])

sns.set(style="whitegrid")

plt.figure(figsize=(8, 6))
sns.histplot(train['num_sold'], kde=True, bins=30, color='violet')

plt.title('Distribution of Sticker Sales (num_sold)', fontsize=16)
plt.xlabel('Number of Stickers Sold')
plt.ylabel('Frequency')

plt.show()

X = train.drop(columns=['num_sold'])
y = train['num_sold']
X_train, X_valid, y_train, y_valid  = train_test_split(X, y, test_size=0.2, random_state=42)

def tune_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 4000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-6, 1e-2),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-6, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-6, 10.0),
        'random_state': 42,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'eval_metric': 'mape'
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_percentage_error(y_valid, preds)
    
#%%time
#study_xgb = optuna.create_study(direction='minimize')
#study_xgb.optimize(tune_xgb, n_trials=50)

#print("Best XGBoost params:", study_xgb.best_params)
#print("XGBoost MAPE:", study_xgb.best_value)

def tune_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 4000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-6, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-6, 10.0),
        'random_state': 42,
        'device': 'gpu',
        'metric': 'mape'
    }
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_percentage_error(y_valid, preds)
    
#%%time
#study_lgb = optuna.create_study(direction='minimize')
#study_lgb.optimize(tune_lgb, n_trials=50)

#print("Best LightGBM params:", study_lgb.best_params)
#print("LightGBM MAPE:", study_lgb.best_value)

def tune_catboost(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 4000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 3, 15),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-6, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        'loss_function': 'MAPE',
        'eval_metric': 'MAPE',
        'random_state': 42
    }
    model = CatBoostRegressor(**params, verbose=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_percentage_error(y_valid, preds)
    
#%%time
#study_catboost = optuna.create_study(direction='minimize')
#study_catboost.optimize(tune_catboost, n_trials=50)

#print("Best CatBoost params:", study_catboost.best_params)
#print("CatBoost MAPE:", study_catboost.best_value)

xgb_model = XGBRegressor(
    n_estimators=3750,
    learning_rate=0.022928762781800033,
    max_depth=7,
    min_child_weight=6,
    subsample=0.9684855394596099,
    colsample_bytree=0.7414559465626035,
    gamma=0.0002633815873971183,
    reg_alpha=0.033856532345376285,
    reg_lambda=3.848636177756615e-05,
    random_state=42
)
lgb_model = LGBMRegressor(
    n_estimators=3130,
    learning_rate=0.06759246686506182,
    max_depth=13,
    min_child_samples=13,
    colsample_bytree=0.6725055033032713,
    subsample=0.8760367415449087,
    reg_alpha=0.13242379471194435,
    reg_lambda=8.10849734527323e-06,
    random_state=42
)

catboost_model = CatBoostRegressor(
    n_estimators=3671,
    learning_rate=0.042483634105534726,
    depth=8,
    l2_leaf_reg=0.00045721354980041677,
    bagging_temperature=0.6630177442611984,
    random_strength=0.4765511332665974,
    verbose=0,  # To suppress training logs
    random_seed=42
)

#%%time
meta_model = LinearRegression()
stacking_model = StackingRegressor(
    estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('catboost', catboost_model)],
    final_estimator=meta_model,
    n_jobs=-1
)

stacking_model.fit(X,y)

label_encoders = {}  
for col in cat_cols:
    le = LabelEncoder()
    test[col] = le.fit_transform(test[col])
    label_encoders[col] = le
test.head()

submission_ids = test['id']
predictions = stacking_model.predict(test)
predictions = np.expm1(predictions)
submission = pd.DataFrame({
    'id': submission_ids,
    'num_sold': predictions 
})
submission.to_csv('submission.csv', index=False)
print("File Saved!")
print(submission.head())


