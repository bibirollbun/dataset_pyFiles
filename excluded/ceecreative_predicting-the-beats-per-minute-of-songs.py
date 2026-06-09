# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

#import models
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')


train_df.head()


train_df.head()


plt.figure(figsize=[10,6])
sb.heatmap(train_df.corr(),annot=True);


#plt.hist(train_df, x='BeatsPerMinute', bins=20);
sb.histplot(train_df['BeatsPerMinute'],kde = True, bins=20);


for col in train_df.drop(['BeatsPerMinute','id'],axis=1).columns:
    sb.histplot(train_df[col], kde = True, bins = 20)
    plt.title(f'distribution of {col}')
    plt.show()


train_df['Is_energetic'] = (train_df['Energy'] >= 0.5).astype(int)


train_df['VocalContent * AcousticQuality'] = train_df[
'VocalContent'] * train_df['AcousticQuality']


train_df['RhythmScore * LivePerformanceLikelihood'] = train_df[
'RhythmScore'] * train_df['LivePerformanceLikelihood']


train_df['Energy per seconds'] = train_df['Energy'] / train_df['TrackDurationMs']
train_df['Loudness-to-Energy ratio'] = train_df['AudioLoudness'] / train_df['Energy']
train_df['Mood per duration'] = train_df['MoodScore'] / train_df['TrackDurationMs']





X = train_df.drop('BeatsPerMinute',axis=1)
y = train_df['BeatsPerMinute'] 


sc = StandardScaler()
X_scaled = sc.fit_transform(X)


X_train,X_val,y_train,y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)


lr = LinearRegression()

lr.fit(X_train,y_train)
lr_model = lr.predict(X_val)


dt = DecisionTreeRegressor(
     max_features =  5,
     min_samples_split = 10,
     splitter = 'random' #or best
        )

dt.fit(X_train,y_train)
dt_model = dt.predict(X_val)


xg = XGBRegressor(
    learning_rate = 0.1, 
    max_depth = 1
    )

xg.fit(X_train,y_train)
xg_model = xg.predict(X_val)


lgbm = LGBMRegressor()
lgbm.fit(X_train,y_train)
lgbm_model = lgbm.predict(X_val)

print('nRMSE for XGB model:',np.sqrt(mean_squared_error(
    y_val,lgbm_model))/y.mean())


print('nRMSE for linear reg model:',np.sqrt(mean_squared_error(
    y_val,lr_model))/y.mean())

print('nRMSE for XGB model:',np.sqrt(mean_squared_error(
    y_val,xg_model))/y.mean())

print('nRMSE for decision Reg model:',np.sqrt(
    mean_squared_error(y_val,dt_model))/y.mean())

print('RMSE for LGBM model:',np.sqrt(
    mean_squared_error(y_val,lgbm_model))/y.mean())


test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


X_test_scaled = sc.transform(test_df)


model = xg.fit(X,y)


predictions = model.predict(X_scaled)


submission_df = test_df[['id']]


submission_df['BeatsPerMinute'] = predictions


submission_df = submission_df.reset_index(drop=True)


submission_df.to_csv('submission.csv', index=False)


submission_df




