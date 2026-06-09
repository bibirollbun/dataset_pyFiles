# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd 
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score# data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest,f_regression
from scipy.stats import randint
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data= pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/train.csv')
test_data = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-1/test.csv')


train_data.head()


test_data.head()


def handle_negative_values(df):
    for column in df.select_dtypes(include=['float64','int64']).columns:
        if(df[column]<0).any():
            df[column] += abs(df[column].min())+1

handle_negative_values(train_data)
handle_negative_values(test_data)            
        


train_data.head()


x = train_data.drop(columns = ['target'])
y = train_data['target']




x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=42)


scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_val_scaled=scaler.transform(x_val)
test_features_scaled =scaler.transform(test_data.drop(columns=['id']))






gbr = GradientBoostingRegressor(random_state=42)
param_dist ={
    'n_estimators':[100,200,300],
    'learning_rate': [0.01,0.05,0.1],
    'max_depth':[3,5,7],
    'subsample':[0.8,1.0],
    'min_samples_leaf':[1,2,4],
    'min_samples_split':[2,5,10],
    

}



random_search =RandomizedSearchCV(estimator=gbr,param_distributions=param_dist,scoring='r2',cv=3,verbose=1,n_jobs=-1,error_score='raise')

random_search.fit(x_train_scaled,y_train)



best_gbr=random_search.best_estimator_
best_gbr.fit(x_train_scaled,y_train)
y_pred_gbr= best_gbr.predict(x_val_scaled)




gbr_r2=r2_score(y_val,y_pred_gbr)




print("gbr:",gbr_r2)


test_data['target']=best_gbr.predict(test_features_scaled)
print(test_data.columns)


test_data[['id','target']].to_csv('submission.csv',index=False)
print("file is saved")

