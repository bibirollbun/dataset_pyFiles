import pandas as pd
import numpy as np


test=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
train=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')


print(test.shape)
print(train.shape)
train.head(3)


test.head(3)


print(train.isnull().sum())
print(train.describe())
print(train.info())


print(test.isnull().sum())
print(test.describe())
print(test.info())


from sklearn.preprocessing import LabelEncoder,StandardScaler
le=LabelEncoder()
sc=StandardScaler()

train['Sex']=le.fit_transform(train['Sex'])
test['Sex']=le.transform(test['Sex'])


from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error


model=LGBMRegressor(max_depth=6,n_estimators=1000,learning_rate=0.05,random_state=42)


features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
train['Calories_log'] = np.log1p(train['Calories'])
X = train[features]
y = train['Calories_log']
X_test = test[features]
X_scaled=sc.fit_transform(train[features])
X_test_scaled=sc.transform(test[features])


model.fit(X_scaled,y)


y_pred_log = model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)


submission=pd.DataFrame({'id':test['id'] ,'Calories':y_pred})
submission.to_csv('/kaggle/working/submission.csv',index=False)


y_pred_train_log = model.predict(X_scaled)
y_pred_train = np.expm1(y_pred_train_log)
rmsle = np.sqrt(mean_squared_log_error(np.expm1(y), y_pred_train))
print(f'RMSLE: {rmsle:.5f}')




