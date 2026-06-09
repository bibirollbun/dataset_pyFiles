# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly as plt
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected = True)
import plotly.express as px
import matplotlib.pyplot as py
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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')



train_df.describe()


train_df.info()


num_cols = [colname for colname in train_df.columns if train_df[colname].dtype in ['int64','float64']]
cat_cols = [colname for colname in train_df.columns if (train_df[colname].dtype in ['object'])]
print(num_cols)
print(cat_cols)


fig = py.figure(figsize = (18,16))
for index,col in enumerate(num_cols[:19]):
    py.subplot(5,4,index+1)
    sns.distplot(train_df.loc[:,col].dropna())
fig.tight_layout(pad = 1.0)


fig = py.figure(figsize = (18,16))
for index,col in enumerate(num_cols[:19]):
    py.subplot(5,4,index+1)
    sns.violinplot(train_df, x = 'rainfall', y = col, palette = 'rocket')
fig.tight_layout(pad = 1.0)


#Analyse missing values

train_df[num_cols].isnull().sum().sort_values(ascending= False)


sns.displot(train_df, x = 'rainfall')


dfnumerical = train_df[num_cols]
correlation = dfnumerical.corr()
print(correlation['rainfall'].sort_values(ascending = False))

fig4 = sns.heatmap(dfnumerical.corr(),cmap="PiYG")
print(fig4)


train_df.info()


numcols_X = [e for e in num_cols if e not in ('id','rainfall')]



y = train_df['rainfall']
X_filter = train_df[numcols_X]
test_filter = test_df[numcols_X]


X_filter.head()


def feature_engineering(df):
    df['high_sun'] = df['sunshine']>6
    df['high_cloud'] = df['cloud']>70
    df['high_humidity'] = df['humidity']>80
    df['cloud_sun_ratio'] = df['cloud']/df['sunshine'].clip(lower = 0.1)
    df['early'] = df['day']<150
    return df

X_feature = feature_engineering(X_filter)
test_feature = feature_engineering(test_filter)


#Remove NaN and replace with zero (change this later?!!!!!!!!)

#X_impute = X_feature.fillna(0)
#test_df_impute = test_feature.fillna(0)



#Impute the median for NaN
def imputemedian(df):
    for col in df.columns:
        df[col] = df[col].fillna(df[col].median())
        return df

X_impute = imputemedian(X_feature)
test_df_impute = imputemedian(test_feature)


#Scale variables
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_impute)
test_scaled = scaler.transform(test_df_impute)



X = X_scaled
X_test = test_scaled


#Split into train and validation datasets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.1, random_state=4)


model = Sequential()
model.add(tf.keras.Input(shape=(X_train.shape[1],)))
model.add(Dense(20, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer='adam',#'sgd'
             loss='binary_crossentropy',
             metrics=['AUC'])


#Fit using train and validation data
model.fit(X_train,y_train,epochs = 200, batch_size = 1, validation_data=(X_valid, y_valid))


#Fit using the whole data
#model.fit(X,y,epochs = 200, batch_size = 1)


predictions = model.predict([X_test])


submission = pd.DataFrame(predictions, columns = ['rainfall'])
submission['id'] = test_df['id']
submission_reord = submission[['id','rainfall']]
submission_reord.to_csv("submission.csv", index=False)


submission_reord


submission_reord[submission_reord['rainfall'].isnull()]

