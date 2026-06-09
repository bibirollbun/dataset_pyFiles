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
import seaborn as sns
import warnings
warnings.filterwarnings('ignore',category = FutureWarning)


train_df= pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df= pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


train_df.sample(5)


train_df.info()


test_df.info()


train_df.duplicated().sum()


train_df.describe()


plt.figure(figsize = (20,20))
sns.pairplot(data = train_df)
plt.show()


train_df.drop(columns = ['id'],inplace = True)
ids = test_df['id'].copy()
test_df.drop(columns = ['id'],inplace = True)


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,mean_squared_log_error


le = LabelEncoder()
train_df['Sex'] = le.fit_transform(train_df['Sex'])
test_df['Sex'] = le.fit_transform(test_df['Sex'])


#Feature Engineering
train_df['Height_m'] = train_df['Height'] / 100
train_df['BMI'] = train_df['Weight'] / (train_df['Height_m'] ** 2)


test_df['Height_m'] = test_df['Height'] / 100
test_df['BMI'] = test_df['Weight'] / (test_df['Height_m'] ** 2)


# Basic classification
train_df['Heart_Rate_Zone'] = pd.cut(train_df['Heart_Rate'],
                                bins=[0, 100, 140, 200],
                                labels=['Low', 'Moderate', 'High'])

test_df['Heart_Rate_Zone'] = pd.cut(test_df['Heart_Rate'],
                                bins=[0, 100, 140, 200],
                                labels=['Low', 'Moderate', 'High'])

train_df['Heart_Rate_Zone'] = train_df['Heart_Rate_Zone'].map({'Low': 0, 'Moderate': 1, 'High': 2})
test_df['Heart_Rate_Zone'] = test_df['Heart_Rate_Zone'].map({'Low': 0, 'Moderate': 1, 'High': 2})



train_df['Age_Group'] = pd.cut(train_df['Age'],
                         bins=[0, 18, 35, 50, 100],
                         labels=['Teen', 'Young Adult', 'Middle-aged', 'Senior'])
# Optionally encode it:
train_df['Age_Group'] = train_df['Age_Group'].map({'Teen': 0, 'Young Adult': 1, 'Middle-aged': 2, 'Senior': 3})



test_df['Age_Group'] = pd.cut(test_df['Age'],
                         bins=[0, 18, 35, 50, 100],
                         labels=['Teen', 'Young Adult', 'Middle-aged', 'Senior'])
# Optionally encode it:
test_df['Age_Group'] = test_df['Age_Group'].map({'Teen': 0, 'Young Adult': 1, 'Middle-aged': 2, 'Senior': 3})




train_df.head()


train_df['Heart_Rate_Zone'] = train_df['Heart_Rate_Zone'].astype('int32')
train_df['Age_Group'] = train_df['Age_Group'].astype('int32')
test_df['Heart_Rate_Zone'] = test_df['Heart_Rate_Zone'].astype('int32')
test_df['Age_Group'] = test_df['Age_Group'].astype('int32')




train_df.info()


train_df.skew()


X = train_df.drop(columns = ['Calories'])
y = train_df['Calories']
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


x_train.shape


x_test.shape


import xgboost as xgb

model1 = xgb.XGBRegressor(n_estimators = 3200, learning_rate = 0.04,tree_methods = 'gpu_hist')
model1.fit(x_train,y_train)
y_pred = model1.predict(x_test)
print('r2_score:', r2_score(y_pred,y_test))
print('MSE:', mean_squared_error(y_pred,y_test))
print('MAE:', mean_absolute_error(y_pred,y_test))

epsilon = 1e-10
y_pred = np.maximum(y_pred, 0)  # RMSLE requires non-negative predictions
y_test = np.maximum(y_test, 0)  # RMSLE requires non-negative ground truth

rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred + epsilon))
print('RMSLE:', rmsle)


import lightgbm as lgb
lg = lgb.LGBMRegressor(n_estimators = 3000 ,learning_rate = 0.04)
lg.fit(x_train,y_train)
y_pred = lg.predict(x_test)
print('r2_score:', r2_score(y_pred,y_test))
print('MSE:', mean_squared_error(y_pred,y_test))
print('MAE:', mean_absolute_error(y_pred,y_test))

epsilon = 1e-10
y_pred = np.maximum(y_pred, 0)  # RMSLE requires non-negative predictions
y_test = np.maximum(y_test, 0)  # RMSLE requires non-negative ground truth

rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred + epsilon))
print('RMSLE:', rmsle)



from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5, weights = 'distance')
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
print('r2_score:', r2_score(y_pred,y_test))
print('MSE:', mean_squared_error(y_pred,y_test))
print('MAE:', mean_absolute_error(y_pred,y_test))

epsilon = 1e-10
y_pred = np.maximum(y_pred, 0)  # RMSLE requires non-negative predictions
y_test = np.maximum(y_test, 0)  # RMSLE requires non-negative ground truth

rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred + epsilon))
print('RMSLE:', rmsle)



from sklearn.ensemble import GradientBoostingRegressor,VotingRegressor,BaggingRegressor,StackingRegressor

gbr = GradientBoostingRegressor(n_estimators = 500)
gbr.fit(x_train,y_train)
y_pred = gbr.predict(x_test)
print('r2_score:', r2_score(y_pred,y_test))
print('MSE:', mean_squared_error(y_pred,y_test))
print('MAE:', mean_absolute_error(y_pred,y_test))

epsilon = 1e-10
y_pred = np.maximum(y_pred, 0)  # RMSLE requires non-negative predictions
y_test = np.maximum(y_test, 0)  # RMSLE requires non-negative ground truth

rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred + epsilon))
print('RMSLE:', rmsle)




from sklearn.ensemble import VotingRegressor,BaggingRegressor,StackingRegressor

bg = BaggingRegressor(estimator = model1, n_estimators = 5)
bg.fit(X,y)




y_pred = bg.predict(test_df)


y_pred


submission = pd.DataFrame({
    'id': ids,
    'Calories': y_pred
})


submission.head()


submission.to_csv('sunmission6.csv',index = False)

