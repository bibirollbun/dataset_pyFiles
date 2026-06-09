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


import matplotlib.pyplot as plt 
import seaborn as sns 


DATA_DIR = "/kaggle/input/playground-series-s5e9" if os.path.exists("/kaggle/input") else "."

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

# If running locally, place train.csv and test.csv in working dir.
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
train


train.info()


train.describe()


fig1, axes0 = plt.subplots(1, 1, figsize=(18, 12))  
sns.histplot(train['AudioLoudness'], color='blue',ax=axes0) 
fig2, axes1 = plt.subplots(1, 3, figsize=(18, 6))  
sns.histplot(train['RhythmScore'],color='blue', ax=axes1[0])
sns.histplot(train['VocalContent'],color='blue', ax=axes1[1])
sns.histplot(train['AcousticQuality'],color='blue', ax=axes1[2])
fig1, axes2 = plt.subplots(1, 1, figsize=(18, 12))  
sns.histplot(train['InstrumentalScore'], color='blue',ax=axes2) 
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))  
sns.histplot(train['LivePerformanceLikelihood'],color='blue', ax=axes3[0])
sns.histplot(train['MoodScore'],color='blue', ax=axes3[1])
sns.histplot(train['TrackDurationMs'],color='blue', ax=axes3[2])
fig1, axes4 = plt.subplots(1, 1, figsize=(18, 12))  
sns.histplot(train['Energy'], color='blue',ax=axes4)


sns.heatmap(train.corr(),annot=True,fmt=".2f")


fig2, axes0 = plt.subplots(1, 3, figsize=(18, 6))  
sns.boxplot(train['RhythmScore'],color='blue', ax=axes0[0])
sns.boxplot(train['VocalContent'],color='blue', ax=axes0[1])
sns.boxplot(train['AcousticQuality'],color='blue', ax=axes0[2])
fig2, axes1 = plt.subplots(1, 3, figsize=(18, 6))  
sns.boxplot(train['LivePerformanceLikelihood'],color='blue', ax=axes1[0])
sns.boxplot(train['MoodScore'],color='blue', ax=axes1[1])
sns.boxplot(train['TrackDurationMs'],color='blue', ax=axes1[2])
fig1, axes2 = plt.subplots(1, 3, figsize=(18, 12))  
sns.boxplot(train['Energy'], color='blue',ax=axes2[0])
sns.boxplot(train['AudioLoudness'], color='blue',ax=axes2[1]) 
sns.boxplot(train['InstrumentalScore'], color='blue',ax=axes2[2]) 


train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)
Y=train['BeatsPerMinute']
train.drop('BeatsPerMinute',axis=1,inplace=True)
train


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor ,GradientBoostingRegressor,HistGradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
X_train,X_valid,Y_train,Y_test=train_test_split(train,Y,train_size=0.8,test_size=0.2,random_state=42,shuffle=True)
model0=LGBMRegressor(n_estimators=500,random_state=42)
model1=HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05,random_state=42)
model2=GradientBoostingRegressor(n_estimators=500,random_state=42)
model3=XGBRegressor(n_estimators=500,random_state=42)
model4=CatBoostRegressor(n_estimators=500,random_state=42,learning_rate=0.05)


model0.fit(X_train,Y_train)


model1.fit(X_train,Y_train)


model2.fit(X_train,Y_train)


model3.fit(X_train,Y_train)


model4.fit(X_train,Y_train)


pred0=model0.predict(X_valid)
print(pred0)
print(mean_squared_error(Y_test,pred0))


pred1=model1.predict(X_valid)
print(pred1)
print(mean_squared_error(Y_test,pred1))


pred2=model2.predict(X_valid)
print(pred2)
print(mean_squared_error(Y_test,pred2))


pred3=model3.predict(X_valid)
print(pred3)
print(mean_squared_error(Y_test,pred3))


pred4=model4.predict(X_valid)
print(pred4)
print(mean_squared_error(Y_test,pred4))


testid=pd.read_csv(test_path)
testid=testid['id']


test


df0=pd.DataFrame({"id":testid,"BeatsPerMinute":model0.predict(test)})
df1=pd.DataFrame({"id":testid,"BeatsPerMinute":model1.predict(test)})
df2=pd.DataFrame({"id":testid,"BeatsPerMinute":model2.predict(test)})
df3=pd.DataFrame({"id":testid,"BeatsPerMinute":model3.predict(test)})
df4=pd.DataFrame({"id":testid,"BeatsPerMinute":model4.predict(test)})


df0.to_csv("submission_malek0.csv",index=False)
df1.to_csv("submission_malek1.csv",index=False)
df2.to_csv("submission_malek2.csv",index=False)
df3.to_csv("submission_malek3.csv",index=False)
df4.to_csv("submission_malek4.csv",index=False)


df2

