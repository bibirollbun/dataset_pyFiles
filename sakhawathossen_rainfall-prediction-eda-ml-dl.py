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


#Core Libraries 
import pandas as pd
import numpy as np
import random
import warnings
from scipy import stats

#Visualization Libraries 

import matplotlib.pyplot as plt
import seaborn as sns

#machine Learning Libraries 

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder ,FunctionTransformer
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import make_scorer,accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier ,HistGradientBoostingClassifier,RandomForestClassifier,RandomForestRegressor,IsolationForest
from sklearn.compose import ColumnTransformer


df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


df_train


df_test


df_sub


df_train.info()


df_train = df_train.drop(columns=['id'])
df_test = df_test.drop(columns=['id'])


df_train


pip install pydantic-settings


pip install "pydantic==1.*"


from pandas_profiling import ProfileReport

profile = ProfileReport(df_train,explorative=True,config_file="")

# Save the report as an HTML file
profile.to_file("profile_report.html")

# Or display it in a Jupyter Notebook
profile.to_notebook_iframe()


df_train.isnull().sum()


df_train.corr()['rainfall'].sort_values(ascending =False)


# create new features 


# New features

df_train['temp_diff'] = df_train['maxtemp'] - df_train['mintemp']
df_train['cloud_to_sunshine'] = df_train['cloud'] * df_train['sunshine']
df_train['cloud_humidity'] = df_train['cloud'] + df_train['humidity']
df_train['humidity_sunshine'] = df_train['humidity'] * df_train['sunshine']
df_train['dew_point_depression'] = df_train['temparature'] - df_train['dewpoint']



df_train


df_train.corr()['rainfall'].sort_values(ascending =False)


# Lets handle outliers


def remove_outliers_iqr(df, columns):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[~((df[columns] < lower_bound) | (df[columns] > upper_bound)).any(axis=1)]

# Columns to check for outliers
columns_to_check = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed', 'temp_diff', 'cloud_to_sunshine',
'cloud_humidity', 'humidity_sunshine', 'dew_point_depression']

# Remove outliers
df_train = remove_outliers_iqr(df_train, columns_to_check)


df_train





X = df_train.drop(columns = ['rainfall'])


X


y = df_train['rainfall']


y


X_train,X_test ,y_train ,y_test = train_test_split(X,y,test_size = 0.2 ,random_state = 42)


X_train


sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


X_train_scaled


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model = Sequential()
model.add(Dense(64 ,activation = 'relu' , input_dim = 16))
model.add(Dense(48,activation = 'relu'))
model.add(Dense(48,activation = 'relu'))
model.add(Dense(32,activation = 'relu'))
model.add(Dense(16,activation = 'relu'))
model.add(Dense (1,activation = 'sigmoid'))


model.summary()


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


history=model.fit(X_train_scaled,y_train,epochs =200 , validation_split = .2 )


import numpy as np
predict_y = np.where(model.predict(X_test)>.5,1,0)


from sklearn.metrics import accuracy_score
accuracy_score(y_test,predict_y)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])




