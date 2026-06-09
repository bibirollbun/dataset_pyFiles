# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected = True)
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from statistics import mean
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')



train_df.describe()


train_df.info()


num_cols = [colname for colname in train_df.columns if train_df[colname].dtype in ['int64','float64']]
cat_cols = [colname for colname in train_df.columns if (train_df[colname].dtype in ['object'])]
print(num_cols)
print(cat_cols)





fig = plt.figure(figsize = (18,16))
for index,col in enumerate(num_cols[:19]):
    plt.subplot(5,4,index+1)
    sns.distplot(train_df.loc[:,col].dropna())
fig.tight_layout(pad = 1.0)


#Finding out how many distint items there are and filtering them
for i in cat_cols:
    print (f'Value Count for {i}')
    print(train_df[i].value_counts())
    print('_'*20)


#Analyse the categorical data
for i in cat_cols:
    fig , axes = plt.subplots(1,2, figsize=(10,6))
    sns.countplot(train_df , x=i, ax= axes[0])
    sns.boxplot(train_df, x = i, y = train_df['Listening_Time_minutes'], ax= axes[1])
    plt.show()


sns.displot(train_df, x = 'Listening_Time_minutes')


dfnumerical = train_df[num_cols]
correlation = dfnumerical.corr()
print(correlation['Listening_Time_minutes'].sort_values(ascending = False))

fig4 = sns.heatmap(dfnumerical.corr(),cmap="PiYG")
print(fig4)


#Analyse missing values

train_df.isnull().sum().sort_values(ascending= False)





numcols_X = [e for e in num_cols if e not in ('id','Listening_Time_minutes')]
catcols_X = [e for e in cat_cols if e not in ()]
totcols_X = numcols_X + cat_cols


df1 = train_df[totcols_X]
df1_test = test_df[totcols_X]


for i in catcols_X:
    df1[i] = df1[i].astype('category')
    df1_test[i] = df1_test[i].astype('category')








X = df1
y = train_df['Listening_Time_minutes']
test = df1_test


X.info()


model = XGBRegressor(enable_categorical = True)


model.fit(X, y)


predictions = model.predict(test)


submission = pd.DataFrame(data = test_df['id'], index = None, columns = ['id'])
submission['Listening_Time_minute'] = predictions
submission.to_csv("submission.csv", index=False)




