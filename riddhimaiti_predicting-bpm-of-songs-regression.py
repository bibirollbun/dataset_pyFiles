import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train=pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv',index_col=0)
train.head()


test=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv',index_col=0)


train.info()


train.describe()


from sklearn.model_selection import train_test_split


X=train.drop('BeatsPerMinute',axis=1)
y=train['BeatsPerMinute']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


from sklearn.preprocessing import StandardScaler


scaler=StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.transform(X_test)


from sklearn.linear_model import LinearRegression


linreg=LinearRegression()
linreg.fit(scaled_X_train,y_train)
pred=linreg.predict(scaled_X_test)


from sklearn.metrics import mean_absolute_error,mean_squared_error


def root_mean_squared_error(y_test,pred):
    return mean_squared_error(y_test,pred)**0.5


print('MAE :',mean_absolute_error(y_test,pred))
print('MSE :',mean_squared_error(y_test,pred))
print('RMSE :',root_mean_squared_error(y_test,pred))


print(root_mean_squared_error(y_test,pred)/np.mean(y_test)*100,'%')


from sklearn.linear_model import Ridge


linreg=Ridge()
linreg.fit(scaled_X_train,y_train)
pred=linreg.predict(scaled_X_test)


print('MAE :',mean_absolute_error(y_test,pred))
print('MSE :',mean_squared_error(y_test,pred))
print('RMSE :',root_mean_squared_error(y_test,pred))


print(root_mean_squared_error(y_test,pred)/np.mean(y_test)*100,'%')


from sklearn.tree import DecisionTreeRegressor


dtree=DecisionTreeRegressor()
dtree.fit(scaled_X_train,y_train)
pred=dtree.predict(scaled_X_test)


print('MAE :',mean_absolute_error(y_test,pred))
print('MSE :',mean_squared_error(y_test,pred))
print('RMSE :',root_mean_squared_error(y_test,pred))


print(root_mean_squared_error(y_test,pred)/np.mean(y_test)*100,'%')


from xgboost import XGBRegressor


xgb=XGBRegressor()
xgb.fit(scaled_X_train,y_train)
pred=xgb.predict(scaled_X_test)


print('MAE :',mean_absolute_error(y_test,pred))
print('MSE :',mean_squared_error(y_test,pred))
print('RMSE :',root_mean_squared_error(y_test,pred))


error=[]
for i in range(1,20):
    xgb=XGBRegressor(n_estimators=i)
    xgb.fit(scaled_X_train,y_train)
    pred=xgb.predict(scaled_X_test)
    error.append(root_mean_squared_error(y_test,pred))
plt.plot(range(1,20),error,'r-o')
plt.xlabel('n_estimators')
plt.ylabel('RMSE')


xgb=XGBRegressor(n_estimators=2)
xgb.fit(scaled_X_train,y_train)
pred=xgb.predict(scaled_X_test)


print('MAE :',mean_absolute_error(y_test,pred))
print('MSE :',mean_squared_error(y_test,pred))
print('RMSE :',root_mean_squared_error(y_test,pred))


print(root_mean_squared_error(y_test,pred)/np.mean(y_test)*100,'%')


from lightgbm import LGBMRegressor


lgbm=LGBMRegressor()
lgbm.fit(scaled_X_train,y_train)
pred=lgbm.predict(scaled_X_test)


print('MAE :',mean_absolute_error(y_test,pred))
print('MSE :',mean_squared_error(y_test,pred))
print('RMSE :',root_mean_squared_error(y_test,pred))


error=[]
for i in range(1,30):
    lgbm=LGBMRegressor(n_estimators=i)
    lgbm.fit(scaled_X_train,y_train)
    pred=lgbm.predict(scaled_X_test)
    error.append(root_mean_squared_error(y_test,pred))


plt.plot(range(1,30),error,'g-o')
plt.xlabel('n_estimators')
plt.ylabel('RMSE')


lgbm=LGBMRegressor(n_estimators=12)
lgbm.fit(scaled_X_train,y_train)
pred=lgbm.predict(scaled_X_test)


print('MAE :',mean_absolute_error(y_test,pred))
print('MSE :',mean_squared_error(y_test,pred))
print('RMSE :',root_mean_squared_error(y_test,pred))


print(root_mean_squared_error(y_test,pred)/np.mean(y_test)*100,'%')


# Final Submission


scaler=StandardScaler()
scaled_X=scaler.fit_transform(X)
scaled_test_X=scaler.transform(test)


lgbm=LGBMRegressor(n_estimators=12)
lgbm.fit(scaled_X,y)
pred=lgbm.predict(scaled_test_X)


final=pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv',index_col=0)


final['BeatsPerMinute']=pred


final.to_csv('final.csv')

