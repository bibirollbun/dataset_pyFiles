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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


data_train= pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
data_train.head()


data_train.shape


data_train.info()


data_train.isnull().sum()


data_train= data_train.dropna(subset= ['num_sold'])
data_train.shape


data_train['date']= pd.to_datetime(data_train['date'])
data_train['year']= data_train['date'].dt.year 
data_train['month']= data_train['date'].dt.month
data_train['day']= data_train['date'].dt.day
data_train['day_of_week'] = data_train['date'].dt.weekday
data_train['is_weekend']= data_train['day_of_week'].apply(lambda x: 1 if x>=5 else 0)
data_train= data_train.drop(columns= ['date'], axis= 1)
data_train.head()


fig= plt.figure(figsize=(14, 8))
sns.barplot(data_train, x='year',y='num_sold')
plt.title('Year v/s Sales Plot')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.show()


fig= plt.figure(figsize=(15, 6))
sns.barplot(data_train, x='country', y='num_sold', hue='year')
plt.title('Country v/s Sales')
plt.xlabel('Country')
plt.ylabel('Sales')
plt.show()


fig= plt.figure(figsize=(10, 5))
sns.barplot(data_train, x='product', y='num_sold')
plt.title('Product v/s/ Sales')
plt.xlabel('Product')
plt.ylabel('Sales')
plt.show()


data_train= pd.get_dummies(data_train, columns=['country', 'store', 'product'], drop_first= True, dtype= int)
data_train.shape


data_train.head()


X= data_train.drop(columns='num_sold', axis= 1)
Y= data_train['num_sold']


X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.2, random_state=2)


model_rf= RandomForestRegressor(n_estimators= 100, n_jobs= -1)
model_rf.fit(X_train, Y_train)
y_pred_rf= model_rf.predict(X_test)


model_xgb= XGBRegressor(
    n_estimators= 1000,
    learning_rate= 0.01,
    early_stopping_rounds= 150
)
model_xgb.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], verbose= False)


param_grid= {
    'n_estimators' : [100, 250, 500],
    'learning_rate' : [0.01, 0.05, 0.1],
    'max_depth': [1, 3, 5]
}

grid_search= GridSearchCV(model_xgb, param_grid, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, Y_train, eval_set=[(X_test, Y_test)])


improved_model_xgb= grid_search.best_estimator_


y_pred_xgb= model_xgb.predict(X_test)


rmse_rf= np.sqrt(mean_squared_error(Y_test, y_pred_rf))
rmse_xgb= np.sqrt(mean_squared_error(Y_test, y_pred_xgb))

print(f'Root Mean Squared Error of Random Forest Regressor: {rmse_rf}')
print(f'Root Mean Squared Error of XGBRegressor: {rmse_xgb}')


print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Score: {-grid_search.best_score_}')


data_test= pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
data_test.head()


data_test.shape


data_test= data_test.dropna()
data_test.shape


data_test['date']= pd.to_datetime(data_test['date'])
data_test['year']= data_test['date'].dt.year 
data_test['month']= data_test['date'].dt.month
data_test['day']= data_test['date'].dt.day
data_test['day_of_week'] = data_test['date'].dt.weekday
data_test['is_weekend']= data_test['day_of_week'].apply(lambda x: 1 if x>=5 else 0)
data_test= data_test.drop(columns= ['date'], axis= 1)


data_test= pd.get_dummies(data_test, columns=['country', 'store', 'product'], drop_first= True, dtype= int)


data_result_1= model_rf.predict(data_test)


data_result_2= model_xgb.predict(data_test)


data_result_3= improved_model_xgb.predict(data_test)


submission_file=pd.DataFrame({
    'id': data_test['id'],
    'num_sold': data_result_2
})


submission_file


submission_file.to_csv('submission.csv', index= False)
print('Successful')

