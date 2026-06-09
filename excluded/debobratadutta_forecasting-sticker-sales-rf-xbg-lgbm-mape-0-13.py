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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import optuna
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm.callback import early_stopping, log_evaluation 
import lightgbm as lgb

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)


df_training = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
df_training.head()


df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
df_test.head()


df_training.info()


df_training.nunique()


df_training.isnull().sum()


list_of_country = df_training['country'].unique()
for country in list_of_country:
    missing_value_count = len(df_training[(df_training['country'] == country) & (df_training['num_sold'].isnull())]['num_sold'])
    missing_value_percentage = (missing_value_count/len(df_training[df_training['country'] == country]))*100
    print(f"---------------------{country}-------------------")
    print(f"Missing value count : {missing_value_count} and Percentage : {missing_value_percentage}")


list_of_country = df_training['country'].unique()
for country in list_of_country:
    Q3 = np.quantile(df_training.loc[df_training['country'] == country, 'num_sold'], 0.75)
    Q1 = np.quantile(df_training.loc[df_training['country'] == country, 'num_sold'], 0.25)
    IQR = Q3 - Q1
    lower_range = Q1 - 1.5 * IQR
    upper_range = Q3 + 1.5 * IQR
    upper_outlier_count = len(df_training[(df_training['num_sold']>upper_range) & (df_training['country'] == country)])
    lower_outlier_count = len(df_training[(df_training['num_sold']<lower_range) & (df_training['country'] == country)])
    upper_outlier_percentage = (upper_outlier_count/len(df_training[df_training['country'] == country]))*100
    lower_outlier_percentage = (lower_outlier_count/len(df_training[df_training['country'] == country]))*100
    
    print(f"---------------------{country}-------------------")
    print(f"Upper Outlier count : {upper_outlier_count} and Percentage : {upper_outlier_percentage}")
    print(f"Lower Outlier count : {lower_outlier_count}and Percentage : {lower_outlier_percentage}")


(len(df_training[df_training['num_sold'].isnull()])/len(df_training))*100


df_training = df_training.dropna()
df_training.isnull().sum()


df_training[df_training.duplicated()]


df_training['date'] = pd.to_datetime(df_training['date'])
df_training['year'] = df_training['date'].dt.year
df_training['month'] = df_training['date'].dt.month
df_training['day'] = df_training['date'].dt.day
df_training['day_of_week'] = df_training['date'].dt.dayofweek

df_training['day_sin'] = np.sin(2 * np.pi * df_training['day'] / 365.0)
df_training['day_cos'] = np.cos(2 * np.pi * df_training['day'] / 365.0)
df_training['month_sin'] = np.sin(2 * np.pi * df_training['month'] / 12.0)
df_training['month_cos'] = np.cos(2 * np.pi * df_training['month'] / 12.0)
df_training['year_sin'] = np.sin(2 * np.pi * df_training['year'] / 7.0)
df_training['year_cos'] = np.cos(2 * np.pi * df_training['year'] / 7.0)



df_test['date'] = pd.to_datetime(df_test['date'])
df_test['year'] = df_test['date'].dt.year
df_test['month'] = df_test['date'].dt.month
df_test['day'] = df_test['date'].dt.day
df_test['day_of_week'] = df_test['date'].dt.dayofweek

df_test['day_sin'] = np.sin(2 * np.pi * df_test['day'] / 365.0)
df_test['day_cos'] = np.cos(2 * np.pi * df_test['day'] / 365.0)
df_test['month_sin'] = np.sin(2 * np.pi * df_test['month'] / 12.0)
df_test['month_cos'] = np.cos(2 * np.pi * df_test['month'] / 12.0)
df_test['year_sin'] = np.sin(2 * np.pi * df_test['year'] / 7.0)
df_test['year_cos'] = np.cos(2 * np.pi * df_test['year'] / 7.0)


df_training.head()


X = df_training.drop(['num_sold','date','id'],axis=1)
y = np.log1p(df_training['num_sold'])



categorical_columns = X.select_dtypes(exclude='number').columns
numerical_columns = X.select_dtypes(include='number').columns


## Create a copy of test dataset
df_test_copy = df_test.copy()
df_test_copy = df_test_copy.drop(['date','id'],axis=1)
df_test_copy.head()


categorical_columns = X.select_dtypes(exclude='number').columns
numerical_columns = X.select_dtypes(include='number').columns

df_test_copy = df_test.copy()
df_test_copy = df_test_copy.drop(['date','id'],axis=1)
df_test_copy.head()

le = LabelEncoder()

for col in categorical_columns:
    X[col]= le.fit_transform( X[col])
    df_test_copy[col] = le.transform(df_test_copy[col])



kf = KFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    
    params = {
        'n_estimators' : trial.suggest_int('n_estimators',100,1000),
        'learning_rate': trial.suggest_float('learning_rate',0.001,0.1,log=True),
        'num_leaves'   : trial.suggest_int('num_leaves',2,100),
        'subsample':   trial.suggest_float('subsample',0.05,1),
        'colsample_bytree': trial.suggest_float('colsample_bytree',0.05,1),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,100),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 50, log=True),  
        'random_state': 42,
        'metric': 'mae',
        'objective':'regression'
    }

    prediction = np.zeros(len(df_training))
    mape_scores = []
    # Train and validate model using 5-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params,verbose=-1)
        
        # Train using early stopping callback
        model.fit(X_train, y_train)
        # Make predictions
        y_pred = model.predict(X_val)
        prediction[val_idx] = y_pred

        # Calculate MAPE
        mape = mean_absolute_percentage_error(y_val, y_pred)
        mape_scores.append(mape)
        print(f"Fold {fold}: MAPE = {mape:.4f}")
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(mape_scores)
    print(f"Training MAPE score (5-fold average): {mape:.4f}")
    return mape


#study_lgbm = optuna.create_study(direction='minimize')
#study_lgbm.optimize(objective, n_trials=50)
#print("Best LGBM Params:", study_lgbm.best_params)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMRegressor(random_state=42,
                    n_estimators=998,
                    num_leaves=75,
                    min_data_in_leaf=39,
                    learning_rate=0.08286445515603275,
                    subsample=0.23828102118219957,
                    colsample_bytree=0.993203586364454,
                    reg_lambda=0.3000198174172943
                   )

model.fit(X_train, y_train)
y_pred_lgb = model.predict(X_val)


print(f'Mean Aabsolute Percentage Error for LGBMRegressor : {mean_absolute_percentage_error(y_val, y_pred_lgb)}')


y_pred =model.predict(df_test_copy)
submission = pd.DataFrame({'id': df_test['id'],'date':df_test['date'],'country':df_test['country'],
                           'product':df_test['product'],'store':df_test['store'], 'num_sold':np.exp(y_pred)})
submission.head()


plt.figure(figsize=(12, 4))
ax = sns.lineplot(
    data=submission,
    x="date",
    y="num_sold",
    errorbar=None,
    linewidth=0.4
)
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Number Sold", fontsize=10)
ax.tick_params(axis="both", labelsize=8)
plt.title("Number Sold | Years", size=10)
plt.show()


submission[['id','num_sold']].to_csv('submission.csv', index=False)
print("Submission successfully.")




