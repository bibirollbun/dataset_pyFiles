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




data = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


data.head()


data['Height'] = data['Height']/100

data['BMI'] = data['Weight']/((data['Height'])**2)
test['BMI'] = test['Weight']/((test['Height'])**2)

data['age_duration'] = data['Age']*data['Duration']
test['age_duration'] = test['Age']*test['Duration']

data['BMI_duration'] = data['BMI']*data['Duration']
test['BMI_duration'] = test['BMI']*test['Duration']

data['HR_body_temp'] = data['Heart_Rate']*data['Body_Temp']
test['HR_body_temp'] = test['Heart_Rate']*test['Body_Temp']

data['Duration_per_Age'] = data['Duration'] / (data['Age'] + 1)
test['Duration_per_Age'] = test['Duration'] / (test['Age'] + 1)

data['Duration_per_BMI'] = data['Duration'] / (data['BMI'] + 1)
test['Duration_per_BMI'] = test['Duration'] / (test['BMI'] + 1)

data['Effort'] = data['Heart_Rate']*data['Duration']
test['Effort'] = test['Heart_Rate']*test['Duration']

data['HR_BMI'] = data['BMI']*data['Heart_Rate']
test['HR_BMI'] = test['BMI']*test['Heart_Rate']

data['Body_Temp_Age'] = data['Body_Temp']*data['Heart_Rate']
test['Body_Temp_Age'] = test['Body_Temp']*test['Heart_Rate']

data['Age_BMI'] = data['Age']*data['BMI']
test['Age_BMI'] = test['Age']*test['BMI']


test_id = test['id']
data.drop(columns = ['Height','Weight','id','Age','Duration'],inplace =True)
test.drop(columns = ['Height','Weight','id','Age','Duration'],inplace =True)


test.head()


data.isna().mean()*100


data.info()


X= data.drop(columns=['Calories'])
y = data['Calories']


X.head()



from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,VotingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import r2_score, mean_squared_log_error as msl
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler,MaxAbsScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# !pip install lightgbm
import lightgbm as lgb
import seaborn as sns


sns.boxplot(x='Body_Temp',data=data)


X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=42)


from sklearn.metrics import mean_squared_log_error as msl


data.head()


cat_cols = X.select_dtypes(include =('object','category')).columns
num_cols = X.drop(columns='Sex').columns

preprocessing= ColumnTransformer([
    ('ohe',OneHotEncoder(),cat_cols),
    ('scale',StandardScaler(),num_cols)
])

# XGBREGRESSOR 
model_xgb = Pipeline([
    ('preprocess',preprocessing),
    ('algo',XGBRegressor(max_depth=11,learning_rate=0.07,n_estimators=800,n_jobs=-1))
])

# RANDOM FOREST 
model_rf = Pipeline([
    ('preprocess',preprocessing),
    ('algo',RandomForestRegressor(n_estimators = 50,n_jobs=-1))
])

# ADABOOST 
model_ada = Pipeline([
    ('preprocess',preprocessing),
    ('algo',AdaBoostRegressor(learning_rate=0.07,estimator=DecisionTreeRegressor(max_depth=15)))
])

# LIGHT GBM
model_gbm = Pipeline([
    ('preprocess',preprocessing),
    ('algo',lgb.LGBMRegressor(objective='tweedie',learning_rate=0.05,n_estimators=1000,n_jobs=-1))
])

vote = VotingRegressor([
    ('gbm',model_gbm),
    ('xgb',model_xgb)
],n_jobs=-1) 


model_gbm.fit(X_train,y_train)
y_pred_lgb = model_gbm.predict(X_valid)
err_gbm = msl(y_valid,y_pred_lgb)
rmsl = np.sqrt(err_gbm)
rmsl


y_pred_gbm = model_gbm.predict(test)


model_xgb.fit(X_train,y_train)
y_pred_xgb = model_xgb.predict(X_valid)

err_xgb = msl(y_valid,y_pred_xgb)
np.sqrt(err_xgb)



y_pred_xgb = model_xgb.predict(test)


model_rf.fit(X_train,y_train)
y_pred_rf = model_rf.predict(X_valid)
err_rf=msl(y_valid,y_pred_rf)
np.sqrt(err_rf)


y_pred_rf = model_rf.predict(test)


vote.fit(X_train,y_train)
y_pred_vote = vote.predict(X_valid)
err_vote =msl(y_valid,y_pred_vote)
np.sqrt(err_vote)


vote_pred_test = vote.predict(test)


output_lgbm = pd.DataFrame({'id':test_id,'Calories':y_pred_gbm})
output_lgbm.to_csv('submission_lgbm.csv',index=False)

output_vote = pd.DataFrame({'id':test_id,'Calories':vote_pred_test})
output_vote.to_csv('submission_vote.csv',index=False)

output_xgb = pd.DataFrame({'id':test_id,'Calories':y_pred_xgb})
output_xgb.to_csv('submission_xgb.csv',index=False)

output_rf = pd.DataFrame({'id':test_id,'Calories':y_pred_rf})
output_rf.to_csv('submission_rf.csv',index=False)


submit = pd.read_csv('submission_lgbm.csv')
submit = pd.read_csv('submission_vote.csv')
submit = pd.read_csv('submission_xgb.csv')
submit = pd.read_csv('submission_rf.csv')






