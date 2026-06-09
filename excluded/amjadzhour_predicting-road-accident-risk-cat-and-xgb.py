import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns



df_train=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
submission=pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


df_train.set_index('id',inplace=True)
df_test.set_index('id',inplace=True)


df_train.isnull().sum()


df_train.duplicated().sum()


df_train=df_train.drop_duplicates()


df_train


map_road_signs_present={True:1,False:0}
df_train['road_signs_present'] = df_train['road_signs_present'].map(map_road_signs_present)
df_train['public_road'] = df_train['public_road'].map(map_road_signs_present)
df_train['holiday'] = df_train['holiday'].map(map_road_signs_present)
df_train['school_season'] = df_train['school_season'].map(map_road_signs_present)



df_test['road_signs_present'] = df_test['road_signs_present'].map(map_road_signs_present)
df_test['public_road'] = df_test['public_road'].map(map_road_signs_present)
df_test['holiday'] = df_test['holiday'].map(map_road_signs_present)
df_test['school_season'] = df_test['school_season'].map(map_road_signs_present)


df_train


map_lighting={'daylight':0,'dim':1,'night':2}
df_train['lighting'] = df_train['lighting'].map(map_lighting)
df_test['lighting'] = df_test['lighting'].map(map_lighting)

map_weather={'rainy':2,'clear':0,'foggy':1}
df_train['weather'] = df_train['weather'].map(map_weather)
df_test['weather'] = df_test['weather'].map(map_weather)

map_time_of_day={'afternoon':0,'evening':1,'morning':2}
df_train['time_of_day'] = df_train['time_of_day'].map(map_time_of_day)
df_test['time_of_day'] = df_test['time_of_day'].map(map_time_of_day)



map_time_of_day={'urban':0,'rural':1,'highway':2}
df_train['road_type'] = df_train['road_type'].map(map_time_of_day)
df_test['road_type'] = df_test['road_type'].map(map_time_of_day)


df_test


df_train


df_train.describe()


df_test.describe()


plt.figure(figsize=(12,8))
sns.heatmap(df_train.corr(), annot=True, fmt=".2f")


fig,ax=plt.subplots(7,2,figsize=(16,20))
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


fig,ax=plt.subplots(7,2,figsize=(16,20))
row=0
col=0
for i in df_train.columns:
    sns.scatterplot(x=df_train[i], y=df_train['accident_risk'],ax=ax[row,col])
    ax[row,col].set_title(f'{i} Distribution')
    col += 1

    if col == 2:
        col = 0
        row += 1
plt.tight_layout(pad=3.0)
plt.show()


from sklearn.model_selection import train_test_split
x=df_train.drop('accident_risk',axis=1)
y=df_train['accident_risk']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)



x


from sklearn.linear_model import LinearRegression
import sklearn.metrics as m
model=LinearRegression()
model.fit(x_train,y_train)


print("train")
y_hat=model.predict(x_train) 
print('R^2:',m.r2_score(y_train,y_hat))
print('MAE:',m.mean_absolute_error(y_train,y_hat))
print('MSE:',m.mean_squared_error(y_train,y_hat))
print('RMSE:',np.sqrt(m.mean_squared_error(y_train,y_hat)))


print('test')
y_hat=model.predict(x_test) 
print('R^2:',m.r2_score(y_test,y_hat))
print('MAE:',m.mean_absolute_error(y_test,y_hat))
print('MSE:',m.mean_squared_error(y_test,y_hat))
print('RMSE:',np.sqrt(m.mean_squared_error(y_test,y_hat)))


from catboost import  CatBoostRegressor
from sklearn.model_selection import GridSearchCV
model2=CatBoostRegressor(verbose=False)
model2.fit(x_train,y_train)
print("train")
y_hat=model2.predict(x_train) 
print('R^2:',m.r2_score(y_train,y_hat))
print('MAE:',m.mean_absolute_error(y_train,y_hat))
print('MSE:',m.mean_squared_error(y_train,y_hat))
print('RMSE:',np.sqrt(m.mean_squared_error(y_train,y_hat)))




print("test")
y_hat=model2.predict(x_test) 
print('R^2:',m.r2_score(y_test,y_hat))
print('MAE:',m.mean_absolute_error(y_test,y_hat))
print('MSE:',m.mean_squared_error(y_test,y_hat))
print('RMSE:',np.sqrt(m.mean_squared_error(y_test,y_hat)))




from xgboost import XGBRegressor
import sklearn.metrics as m 
model3=XGBRegressor(booster='gbtree',n_estimators=1500,learning_rate=0.1,max_depth=7,gamma=1, reg_alpha=0.5, reg_lambda=1,min_child_weight=5)
model3.fit(x_train,y_train,eval_set=[(x_test,y_test)],verbose=100)


print("train")
y_hat=model3.predict(x_train) 
print('R^2:',m.r2_score(y_train,y_hat))
print('MAE:',m.mean_absolute_error(y_train,y_hat))
print('MSE:',m.mean_squared_error(y_train,y_hat))
print('RMSE:',np.sqrt(m.mean_squared_error(y_train,y_hat)))





print("test")
y_hat=model3.predict(x_test) 
print('R^2:',m.r2_score(y_test,y_hat))
print('MAE:',m.mean_absolute_error(y_test,y_hat))
print('MSE:',m.mean_squared_error(y_test,y_hat))
print('RMSE:',np.sqrt(m.mean_squared_error(y_test,y_hat)))





submission['accident_risk']=model2.predict(df_test)
submission.to_csv('submission(CatBoostRegressor).csv',index=False)




