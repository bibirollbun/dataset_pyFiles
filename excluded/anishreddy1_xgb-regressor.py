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


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor


traindf=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
testdf=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


traindf.head()


traindf.shape


testdf


traindf.isnull().sum()


traindf.duplicated().mean()


np.isnan(traindf.Calories).sum()


traindf.info()


traindf.describe()


encoder=LabelEncoder()


traindf['Sex']=encoder.fit_transform(traindf['Sex'])
testdf['Sex']=encoder.fit_transform(testdf['Sex'])


traindf.Calories.hist()






sns.boxplot(traindf.Calories)


plt.figure(figsize=(10,10))
sns.heatmap(traindf.corr(),vmin=-1,vmax=1,annot=True,cmap='coolwarm')
plt.show()


traindf.corrwith(traindf.Calories).sort_values(ascending=False)


traindf['BMI']=traindf['Weight']/(np.square(traindf['Height']/100))
traindf['Heartrate_perhour']=traindf['Heart_Rate']*traindf['Duration']
traindf['BodyTemp_perhour']=traindf['Body_Temp']*traindf['Duration']

testdf['BMI']=testdf['Weight']/(np.square(testdf['Height']/100))
testdf['Heartrate_perhour']=testdf['Heart_Rate']*testdf['Duration']
testdf['BodyTemp_perhour']=testdf['Body_Temp']*testdf['Duration']


traindf.describe()


X=traindf.drop('Calories',axis=1)
y=traindf['Calories']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



print("NaNs in y_train:", np.isnan(y_train).sum())
print("Infs in y_train:", np.isinf(y_train).sum())


xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)


lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgb_model.fit(X_train, y_train)

lgb_pred = lgb_model.predict(X_test)


cat_model = CatBoostRegressor(iterations=100, learning_rate=0.1, verbose=0, random_state=42)
cat_model.fit(X_train, y_train)

cat_pred = cat_model.predict(X_test)


y_pred = xgb_model.predict(X_test)
print('MAE', mean_absolute_error(y_test, y_pred))
print('MSE', mean_squared_error(y_test, y_pred))
print('RMSE', np.sqrt(mean_squared_error(y_test, y_pred)))



y_pred = lgb_model.predict(X_test)
print('MAE', mean_absolute_error(y_test, y_pred))
print('MSE', mean_squared_error(y_test, y_pred))
print('RMSE', np.sqrt(mean_squared_error(y_test, y_pred)))


y_pred = cat_model.predict(X_test)
print('MAE', mean_absolute_error(y_test, y_pred))
print('MSE', mean_squared_error(y_test, y_pred))
print('RMSE', np.sqrt(mean_squared_error(y_test, y_pred)))


ensemble = VotingRegressor([
    ('xgb', xgb_model),
    ('lgb', lgb_model),
    ('cat', cat_model),
])


ensemble.fit(X_train, y_train)


ensemble_pred = ensemble.predict(testdf)


# ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
# ensemble_rmse = mean_squared_error(y_test, ensemble_pred, squared=False)

# print("Ensemble MAE:", ensemble_mae)
# print("Ensemble RMSE:", ensemble_rmse)


output=pd.DataFrame({'id':testdf.id,'Calories':ensemble_pred})


output.to_csv('submission.csv',index=False)




