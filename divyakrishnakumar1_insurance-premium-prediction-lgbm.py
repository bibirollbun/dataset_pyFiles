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
import matplotlib.pyplot as plt
import datetime
from sklearn.impute import SimpleImputer
import seaborn as sns
import statistics as st
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,mean_squared_log_error


train=pd.read_csv('/kaggle/input/playground-series-s4e12/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s4e12/test.csv')


train.head()


train.shape


train.info()


test.head()


test.shape


test.info()


train.isnull().sum()


train.describe()


train['Policy Start Date']=pd.to_datetime(train['Policy Start Date'])
test['Policy Start Date']=pd.to_datetime(test['Policy Start Date'])






train.info()


train['Policy Start_Date']=train['Policy Start Date'].dt.date



test['Policy Start_Date']=test['Policy Start Date'].dt.date


train.head()


test.head()


train.info()


train.head()



train.drop(columns=['Policy Start Date'],inplace=True)
test.drop(columns=['Policy Start Date'],inplace=True)


train.info()


train.head()


test.head()


train['Health Score']=round(train['Health Score'],3)
test['Health Score']=round(test['Health Score'],3)


train.head()


train.isnull().sum()


train.head(50)


train.describe()


train.shape


test.isnull().sum()


test.shape


imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(train.iloc[:,[1,5,8,11]].values)
train.iloc[:,[1,5,8,11]]=imputer.transform(train.iloc[:,[1,5,8,11]].values)


imputertest=SimpleImputer(missing_values=np.nan,strategy='mean')
imputertest.fit(test.iloc[:,[1,5,8,11]].values)
test.iloc[:,[1,5,8,11]]=imputertest.transform(test.iloc[:,[1,5,8,11]].values)





train['Number of Dependents']=round(train['Number of Dependents'],0)
test['Number of Dependents']=round(test['Number of Dependents'],0)


train.head(50)


train['Previous Claims']=round(train['Previous Claims'],0)
test['Previous Claims']=round(test['Previous Claims'],0)



train.head(50)


train.isna().sum()


impute=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
impute.fit(train.iloc[:,[4]].values)
train.iloc[:,[4]]=impute.transform(train.iloc[:,[4]].values)


imputetest=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputetest.fit(test.iloc[:,[4]].values)
test.iloc[:,[4]]=impute.transform(test.iloc[:,[4]].values)


test.isnull().sum()


train['Occupation']=train['Occupation'].fillna('Self-Employed')
test['Occupation']=test['Occupation'].fillna('Self-Employed')


train.head(10)


test.head(10)


train.isna().sum()


test.isna().sum()


train.head(50)


imputer_Income=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer_Income.fit(train.iloc[:,[3,13]].values)
train.iloc[:,[3,13]]=imputer_Income.transform(train.iloc[:,[3,13]].values)


train.isna().sum()


imputer_Income_test=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer_Income_test.fit(test.iloc[:,[3,13]].values)
test.iloc[:,[3,13]]=imputer_Income_test.transform(test.iloc[:,[3,13]].values)


imputer_PC=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer_PC.fit(train.iloc[:,[12,14]].values)
train.iloc[:,[12,14]]=imputer_PC.transform(train.iloc[:,[12,14]].values)


imputer_PCtest=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer_PCtest.fit(test.iloc[:,[12,14]].values)
test.iloc[:,[12,14]]=imputer_PC.transform(test.iloc[:,[12,14]].values)


train.isna().sum()


test.isna().sum()


imputer_Feedback=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputer_Feedback.fit(train.iloc[:,[15]].values)
train.iloc[:,[15]]=imputer_Feedback.transform(train.iloc[:,[15]].values)


imputer_Feedbacktest=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputer_Feedbacktest.fit(test.iloc[:,[15]].values)
test.iloc[:,[15]]=imputer_Feedbacktest.transform(test.iloc[:,[15]].values)


train.isna().sum()


test.isna().sum()


train['Health Score']=round(train['Health Score'],3)
test['Health Score']=round(test['Health Score'],3)


train.head(50)


test.info()


sns.boxplot(test['Age'])


sns.boxplot(test['Annual Income'])


sns.boxplot(test['Number of Dependents'])


sns.boxplot(test['Health Score'])


sns.boxplot(test['Previous Claims'])


sns.boxplot(test['Vehicle Age'])


train['Exercise Frequency'].unique()



Dict={}
lst=['Weekly','Monthly','Daily','Rarely']
for key,val in enumerate(lst):
    Dict[val]=key
print(Dict)


train.head()


train['Exercise_Frequency']=train['Exercise Frequency'].map(Dict)


train.head(50)


test['Exercise_Frequency']=test['Exercise Frequency'].map(Dict)


test.head(50)


gender={'Female':0,'Male':1}

train['Gender_type']=train['Gender'].map(gender)


train.head(10)


test.head()


test['Gender_type']=test['Gender'].map(gender)


marital_status={'Married':0,'Divorced':1,'Single':2}
train['Marital Status']=train['Marital Status'].map(marital_status)


test['Marital Status']=test['Marital Status'].map(marital_status)


train['Education Level'].unique()


edu_level={"Bachelor's":0,"Master's":1,'High School':2,'PhD':3}
train['Education Level']=train['Education Level'].map(edu_level)
test['Education Level']=test['Education Level'].map(edu_level)


train['Occupation'].unique()


occ={'Self-Employed':0,'Employed':1,'Unemployed':2}
train['Occupation']=train['Occupation'].map(occ)
test['Occupation']=test['Occupation'].map(occ)


train['Location'].unique()


location={'Urban':0,'Rural':1,'Suburban':2}
train['Location']=train['Location'].map(location)
test['Location']=test['Location'].map(location)


train['Policy Type'].unique()


poltype={'Premium':0,'Comprehensive':1,'Basic':2}
train['Policy Type']=train['Policy Type'].map(poltype)
test['Policy Type']=test['Policy Type'].map(poltype)


train['Customer Feedback'].unique()


feedback={'Poor':0,'Average':1,'Good':2}
train['Customer Feedback']=train['Customer Feedback'].map(feedback)
test['Customer Feedback']=test['Customer Feedback'].map(feedback)


train['Smoking Status'].unique()


smoke_status={'Yes':0,'No':1}
train['Smoking Status']=train['Smoking Status'].map(smoke_status)
test['Smoking Status']=test['Smoking Status'].map(smoke_status)


train['Property Type'].unique()


prop_type={'House':0,'Apartment':1,'Condo':2}
train['Property Type']=train['Property Type'].map(prop_type)
test['Property Type']=test['Property Type'].map(prop_type)


train=train.drop(columns=['Gender','Exercise Frequency'])
test=test.drop(columns=['Gender','Exercise Frequency'])


train['Credit Score']=round(train['Credit Score'],0)
test['Credit Score']=round(test['Credit Score'],0)



train['Policy Start_Date']=pd.to_datetime(train['Policy Start_Date'])
test['Policy Start_Date']=pd.to_datetime(test['Policy Start_Date'])



today=datetime.datetime.today()
today


train['Time_Elapsed_between_Days']=(today-train['Policy Start_Date']).dt.days
test['Time_Elapsed_between_Days']=(today-train['Policy Start_Date']).dt.days


train=train.drop(columns=['Policy Start_Date'])
test=test.drop(columns=['Policy Start_Date'])



test.head()


test.isnull().sum()


test.info()


train.info()


print(train.shape)
print(test.shape)





X=train.drop(columns=['Premium Amount'])
y=train['Premium Amount']


n_splits=5
lgb_params={
    'objective':'regression',
    'metric':'rmse',
    'boosting_type':'gbdt',
    'learning_rate':0.01,
    'n_estimators':1000,
    'early_stopping_rounds':50,
    'random_state':42
}


xgb_params={
    'objective':'reg:squarederror',
    'eval_metric':'rmse',
    'learning_rate':0.01,
    'n_estimators':1000,
    'early_stopping_rounds':50,
    'random_state':42
}


#Initialise Predictions
lgb_predictions=np.zeros(len(test))
xgb_predictions=np.zeros(len(test))
kf=KFold(n_splits=n_splits,shuffle=True,random_state=42)
#RMSLE function
def acc_score(y_true,y_pred):
    return np.sqrt(mean_squared_log_error(y_true,np.maximum(y_pred,0)))


for fold,(train_idx,val_idx) in enumerate(kf.split(X)):
    X_train,X_val=X.iloc[train_idx],X.iloc[val_idx]
    y_train,y_val=y.iloc[train_idx],y.iloc[val_idx]
    #lightgbm
    lgb_model=lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train,y_train,eval_set=[(X_val,y_val)])
    lgb_val_pred=lgb_model.predict(X_val)
    print(acc_score(y_val,lgb_val_pred))
    lgb_predictions+=lgb_model.predict(test)/n_splits
    
    


lgb_predictions





pred_Targ1=pd.DataFrame()
pred_Targ1['id']=test['id']
pred_Targ1


pred_Targ1['Premium Amount']=lgb_predictions


pred_Targ1.to_csv('//kaggle/working/Submission.csv',header=True,index=False,mode='w')




