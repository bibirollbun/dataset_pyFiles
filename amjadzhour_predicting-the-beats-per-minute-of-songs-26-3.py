import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns



df_train=pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
submission=pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')


df_train.set_index('id',inplace=True)
df_test.set_index('id',inplace=True)


df_train


df_train.describe()


df_test.describe()


plt.figure(figsize=(12,8))
sns.heatmap(df_train.corr(), annot=True, fmt=".2f")


fig,ax=plt.subplots(5,2,figsize=(16,20))
row=0
col=0
for i in df_train.columns:
    sns.histplot(df_train[i],bins=30,ax=ax[row,col])
    ax[row,col].set_title(f'{i} Distribution')
    col += 1

    if col == 2:
        col = 0
        row += 1
plt.tight_layout(pad=3.0)
plt.show()


fig,ax=plt.subplots(5,2,figsize=(16,20))
row=0
col=0
for i in df_train.columns:
    sns.scatterplot(x=df_train[i], y=df_train['BeatsPerMinute'],ax=ax[row,col])
    ax[row,col].set_title(f'{i} Distribution')
    col += 1

    if col == 2:
        col = 0
        row += 1
plt.tight_layout(pad=3.0)
plt.show()


fig,ax=plt.subplots(5,2,figsize=(16,20))
row=0
col=0
for i in df_train.columns:
    sns.boxplot(x=df_train[i],ax=ax[row,col])
    ax[row,col].set_title(f'{i} Distribution')
    col += 1

    if col == 2:
        col = 0
        row += 1
plt.tight_layout(pad=3.0)
plt.show()


from sklearn.feature_selection import mutual_info_regression

X = df_train.drop(columns=["BeatsPerMinute"])
y = df_train["BeatsPerMinute"]
mi = mutual_info_regression(X, y)
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
print(mi_series)



from sklearn.model_selection import train_test_split
x=df_train.drop('BeatsPerMinute',axis=1)
y=df_train['BeatsPerMinute']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
x_dev,x_test,y_dev,y_test=train_test_split(x_test,y_test,test_size=0.5,random_state=42)


x


from sklearn.linear_model import LinearRegression
import sklearn.metrics as m
m1=LinearRegression()
m1.fit(x_train,y_train)


print("train")
y_hat=m1.predict(x_test) 
print('R^2:',m.r2_score(y_test,y_hat))
print('MAE:',m.mean_absolute_error(y_test,y_hat))
print('RMSE:',m.mean_squared_error(y_test,y_hat))
print('RMSE:',np.sqrt(m.mean_squared_error(y_test,y_hat)))


print('test')
y_hat=m1.predict(x_test) 
print('R^2:',m.r2_score(y_test,y_hat))
print('MAE:',m.mean_absolute_error(y_test,y_hat))
print('RMSE:',m.mean_squared_error(y_test,y_hat))
print('RMSE:',np.sqrt(m.mean_squared_error(y_test,y_hat)))


from catboost import  CatBoostRegressor
from sklearn.model_selection import GridSearchCV
modle=CatBoostRegressor(verbose=False)
modle.fit(x_train,y_train)
print("train")
y_hat=modle.predict(x_train) 
print('R^2:',m.r2_score(y_train,y_hat))
print('MAE:',m.mean_absolute_error(y_train,y_hat))
print('RMSE:',m.mean_squared_error(y_train,y_hat))
print('RMSE:',np.sqrt(m.mean_squared_error(y_train,y_hat)))


print('dev')
y_hat=modle.predict(x_dev) 
print('R^2:',m.r2_score(y_dev,y_hat))
print('MAE:',m.mean_absolute_error(y_dev,y_hat))
print('RMSE:',m.mean_squared_error(y_dev,y_hat))
print('RMSE:',np.sqrt(m.mean_squared_error(y_dev,y_hat)))


print("test")
y_hat=modle.predict(x_test) 
print('R^2:',m.r2_score(y_test,y_hat))
print('MAE:',m.mean_absolute_error(y_test,y_hat))
print('RMSE:',m.mean_squared_error(y_test,y_hat))
print('RMSE:',np.sqrt(m.mean_squared_error(y_test,y_hat)))




from xgboost import XGBRegressor
import sklearn.metrics as m 
modle=XGBRegressor(booster='gbtree',n_estimators=1500,learning_rate=0.1,max_depth=7,gamma=1, reg_alpha=0.5, reg_lambda=1,min_child_weight=5)
modle.fit(x_train,y_train,eval_set=[(x_dev,y_dev)],verbose=100)


print("train")
y_hat=modle.predict(x_train) 
print('R^2:',m.r2_score(y_train,y_hat))
print('MAE:',m.mean_absolute_error(y_train,y_hat))
print('RMSE:',m.mean_squared_error(y_train,y_hat))
print('RMSE:',np.sqrt(m.mean_squared_error(y_train,y_hat)))


print('dev')
y_hat=modle.predict(x_dev) 
print('R^2:',m.r2_score(y_dev,y_hat))
print('MAE:',m.mean_absolute_error(y_dev,y_hat))
print('RMSE:',m.mean_squared_error(y_dev,y_hat))
print('RMSE:',np.sqrt(m.mean_squared_error(y_dev,y_hat)))


print("test")
y_hat=modle.predict(x_test) 
print('R^2:',m.r2_score(y_test,y_hat))
print('MAE:',m.mean_absolute_error(y_test,y_hat))
print('RMSE:',m.mean_squared_error(y_test,y_hat))
print('RMSE:',np.sqrt(m.mean_squared_error(y_test,y_hat)))




r=pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
r['BeatsPerMinute']=modle.predict(df_test)
r.to_csv('submission.csv',index=False)

