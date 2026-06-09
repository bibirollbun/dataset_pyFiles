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


os.getcwd()
os.chdir('/kaggle/input/playground-series-s5e4')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from ydata_profiling import ProfileReport

import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import mean_squared_error, accuracy_score


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


df_train.head()


df_train.info()


df_test.head()


df_test.info()


df_train.describe()


df_test.describe()


ProfileReport(df_train)


for col in ['Genre','Publication_Day','Publication_Time']:
    df_train[col] = df_train[col].astype("category")

x_train, y_train, id_train = df_train[['Genre','Host_Popularity_percentage','Publication_Day','Publication_Time']],df_train['Listening_Time_minutes'], df_train['id']
x_tr,x_va,y_tr,y_va = train_test_split(x_train,
                                       y_train,
                                       test_size=0.2,
                                       random_state=123)

params = {
    'boosting_type':'gbdt',
    'objective':'regression',
    'metric':'rmse',
    'learning_rate':0.1,
    'num_leaves':16,
    'n_estimators':100,
    'random_state':123,
    'importance_type':'gain',

}

model = lgb.LGBMRegressor(**params)
model.fit(x_tr,
          y_tr,
          eval_set=[(x_tr, y_tr), (x_va, y_va)],
          callbacks=[lgb.early_stopping(stopping_rounds=100,verbose=True), #early_stoppingのコールバック関数
          lgb.log_evaluation(10)] #コマンドライン用のコールバック関数
         )


#submission
x_test = df_test[['Genre','Host_Popularity_percentage','Publication_Day','Publication_Time']]
for col in [['Genre','Publication_Day','Publication_Time']]:
    x_test[col] = x_test[col].astype("category")
id_test = df_test[['id']]


y_test_pred = model.predict(x_test)


df_submit = pd.DataFrame({"id":id_test['id'],"Listening_Time_minitues":y_test_pred})
display(df_submit.head(5))
df_submit.to_csv('/kaggle/working/lightgbm_ver1.csv',index=False)


# fill with mean

def fill_median(df,col):
    median_col = df[col].median()
    df[col] = df[col].fillna(median_col)
    print(f'{col} :',df[col].isnull().sum())


for col in [["Episode_Length_minutes","Guest_Popularity_percentage","Number_of_Ads"]]:
    fill_median(df_train, col)
    fill_median(df_test, col)


df_train.Number_of_Ads.value_counts()


df_test.Number_of_Ads.value_counts()


# Drop Number_of_Ads > 3 & Nan 
df_train = df_train.loc[~(
    (df_train["Number_of_Ads"] > 3) | (df_train['Number_of_Ads'].isna())
)]
df_train.isnull().sum()


def categorical_astype(df):
    categorical_columns = df.select_dtypes(exclude=['number']).columns
    for col in categorical_columns:
        df[col] = df[col].astype("category")
    df.info()


categorical_astype(df_train)
categorical_astype(df_test)


x = df_train.drop(['Listening_Time_minutes'],axis=1)
y = df_train[['Listening_Time_minutes']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

params = {
    'boosting_type':'gbdt',
    'objective':'regression',
    'metric':'rmse',
    'learning_rate':0.1,
    'num_leaves':16,
    'n_estimators':100,
    'random_state':123,
    'importance_type':'gain',

}

model = lgb.LGBMRegressor(**params)
model.fit(x_train,
          y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          callbacks=[lgb.early_stopping(stopping_rounds=100,verbose=True), #early_stoppingのコールバック関数
          lgb.log_evaluation(10)] #コマンドライン用のコールバック関数
         )


#submission_ver2
x_test = df_test
id_test = df_test[['id']]

y_test_pred = model.predict(x_test)

df_submit = pd.DataFrame({"id":id_test['id'],"Listening_Time_minitues":y_test_pred})
display(df_submit.head(5))
df_submit.to_csv('/kaggle/working/lightgbm_ver2.csv',index=False)


df_train = df_train.reset_index(drop=True)

params = {
    'boosting_type':'gbdt',
    'objective':'regression',
    'metric':'rmse',
    'learning_rate':0.1,
    'num_leaves':16,
    'n_estimators':100,
    'random_state':123,
    'importance_type':'gain',

}

x = df_train.drop(['Listening_Time_minutes'],axis=1)
y = df_train[['Listening_Time_minutes']]

kf = KFold(n_splits=5, shuffle=True, random_state=123)

for fold, (train_index, val_index) in enumerate(kf.split(x)):
    print(f'Fold{fold + 1}')
    x_train, x_val = x.loc[train_index], x.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model = lgb.LGBMRegressor(**params)
    model.fit(x_train,
          y_train,
          eval_set=[(x_train, y_train), (x_val, y_val)],
          callbacks=[lgb.early_stopping(stopping_rounds=100,verbose=True), #early_stoppingのコールバック関数
          lgb.log_evaluation(10)] #コマンドライン用のコールバック関数
         )


def submission(df,num):
    x_test = df
    id_test = df[['id']]
    y_test_pred = model.predict(x_test)
    df_submit = pd.DataFrame({"id":id_test['id'],"Listening_Time_minitues":y_test_pred})
    display(df_submit.head(5))
    df_submit.to_csv(f'/kaggle/working/lightgbm_ver{num}.csv',index=False)


#submission_ver3
submission(df_test,3)

