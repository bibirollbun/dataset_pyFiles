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


from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import optuna
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sns
import holidays


df=pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df.head()


df.info()


df.isnull().mean()


df['country'].unique()


df['date']=pd.to_datetime(df['date'])


df['holiday']=0

ca_holidays=holidays.country_holidays("CA")
fi_holidays=holidays.country_holidays("FI")
it_holidays=holidays.country_holidays("IT")
ke_holidays=holidays.country_holidays("KE")
no_holidays=holidays.country_holidays("NO")
sg_holidays=holidays.country_holidays("SG")


def set_holiday(row):
    value = 1
    if row["country"] == "Canada" and row["date"] in ca_holidays:
        row["holiday"] = value
        
    elif row["country"] == "Finland" and row["date"] in fi_holidays:
        row["holiday"] = value

    elif row["country"] == "Italy" and row["date"] in it_holidays:
        row["holiday"] = value

    elif row["country"] == "Kenya" and row["date"] in ke_holidays:
        row["holiday"] = value


    elif row["country"] == "Norway" and row["date"] in no_holidays:
        row["holiday"] = value

    elif row["country"] == "Singapore" and row["date"] in sg_holidays:
        row["holiday"] = value

    return row

df= df.apply(set_holiday, axis=1)



df['year']=df['date'].dt.year
df['month']=df['date'].dt.month
df['day']=df['date'].dt.day
df['day_of_week']=df['date'].dt.weekday


df=df.dropna()


df = df.drop('date', axis=1)



sns.kdeplot(data=df, x="num_sold")


df['num_sold']=np.log1p(df['num_sold'])
sns.kdeplot(data=df, x=df['num_sold'])


df['num_sold']


df_encoded=pd.get_dummies(df,dtype=int)
df_encoded


ss=StandardScaler()


X=df_encoded.drop(columns=['id','num_sold'])
y=df_encoded['num_sold']


X_scaled=ss.fit_transform(X)


X_train, X_test, y_train, y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)



best_model=LGBMRegressor(colsample_bytree=0.6246864897824295,
              learning_rate=0.09479504608720299, max_bin=353, max_depth=5,
              min_child_samples=67, min_split_gain=0.0018443203818497532,
              n_estimators=394, num_leaves=45, reg_alpha=0.19773510790946916,
              reg_lambda=0.8705390651987932, subsample=0.9519599461874049)


best_model.fit(X_train, y_train)


df_test= pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
df_test


df_test=df_test.dropna()


df_test['holiday']=0
df_test = df_test.apply(set_holiday, axis=1)
df_test['date']=pd.to_datetime(df_test['date'])
df_test['year']=df_test['date'].dt.year
df_test['month']=df_test['date'].dt.month
df_test['day']=df_test['date'].dt.day
df_test['day_of_week']=df_test['date'].dt.weekday
df_test_i = df_test.drop(columns=['date','id'])



df_test_encoded=pd.get_dummies(df_test_i,dtype=int)


df_test_scaled=ss.transform(df_test_encoded)


predictions_log = best_model.predict(df_test_scaled)
predictions = np.expm1(predictions_log)
rounded_predictions = np.ceil(predictions)
submission = pd.DataFrame({'id':df_test['id'],'num_sold':rounded_predictions})
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created.")

