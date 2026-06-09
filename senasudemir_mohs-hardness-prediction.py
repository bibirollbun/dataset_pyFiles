import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',None)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import normalize, scale


df=pd.read_csv('/kaggle/input/playground-series-s3e25/train.csv')


df.head()


df.shape


df.isnull().sum()


df.info()


df.describe().T


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(numeric_only=True),annot=True);


sns.pairplot(df, diag_kind='kde', hue='Hardness')
plt.show()


sns.histplot(df['Hardness'], bins=20, kde=True, color='blue')
plt.title('Distribution of Hardness')
plt.xlabel('Hardness')
plt.ylabel('Frequency');


plt.figure(figsize=(12, 6))
sns.boxplot(data=df.drop(columns=['id']), orient='h', palette='coolwarm')
plt.title('Boxplot of Feature Distributions')
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['density_Average'], y=df['Hardness'], hue=df['allelectrons_Total'], palette='coolwarm')
plt.title('Density vs. Hardness (Colored by allelectrons_Total)');


abs(df.corr(numeric_only=True)['Hardness'].sort_values(ascending=False))


x=df.drop(['Hardness','id'],axis=1)
y=df[['Hardness']]


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

def regression_algo(x, y, plot=False):
    L = LinearRegression()
    R = Ridge()
    Lass = Lasso()
    E = ElasticNet()
    ETR = ExtraTreeRegressor()
    GBR = GradientBoostingRegressor()
    kn = KNeighborsRegressor()
    dt = DecisionTreeRegressor()
    xgb = XGBRegressor()
    rf = RandomForestRegressor()
    lgbm = LGBMRegressor(verbosity=-1)

    algos = [L, R, Lass, E, ETR, GBR, kn, dt, xgb, rf, lgbm]
    algo_names = ['Linear', 'Ridge', 'Lasso', 'ElasticNet', 'Extra Tree', 'Gradient Boosting',
                  'KNeighborsRegressor', 'DecisionTreeRegressor', 'XGBRegressor',
                  'RandomForestRegressor', 'LGBMRegressor']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    r_squared = []
    rmse = []
    mae = []

    result = pd.DataFrame(columns=['R_Squared', 'RMSE', 'MAE'], index=algo_names)

    for algo in algos:
        p = algo.fit(x_train, y_train).predict(x_test)
        r_squared.append(r2_score(y_test, p))
        rmse.append(mean_squared_error(y_test, p) ** 0.5)
        mae.append(mean_absolute_error(y_test, p))

    result.R_Squared = r_squared
    result.RMSE = rmse
    result.MAE = mae

    r_table = result.sort_values('R_Squared', ascending=False)
    
    if plot:
        best_model = algos[r_squared.index(max(r_squared))]
        y_pred = best_model.predict(x_test)
        
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.reset_index(drop=True), label='Actual', color='green', linestyle='--')
        plt.plot(pd.Series(y_pred), label='Predicted', color='red', linestyle='--')
        plt.xlabel('Predicted Price')
        plt.ylabel('Actual Price')
        plt.title('Actual vs Predicted Price for Best Model')
        plt.legend()
        plt.show()
    
    return r_table


regression_algo(x,y,plot=False)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
lgbm = LGBMRegressor(verbosity=-1)
model= lgbm.fit(x_train, y_train)


import joblib
joblib.dump(model, 'best_regression_model.pkl')


df_test=pd.read_csv('/kaggle/input/playground-series-s3e25/test.csv')


df_test.head()


submission=pd.DataFrame({
    'id':df_test['id']
})


df_test.drop(['id'],axis=1,inplace=True)


df_test['density_Total']=df_test['density_Total']*10


predictions=model.predict(df_test)


submission['Hardness']=predictions


submission.to_csv('submission.csv',index=False)


coefficients = np.abs(lgbm.feature_importances_) 

feature_importance = pd.DataFrame({'Feature': x_train.columns, 'Importance': coefficients})

feature_importance = feature_importance.sort_values(by='Importance', ascending=False).head(30)

plt.figure(figsize=(6, 15))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.gca().invert_yaxis()
plt.title('Top 30 Features by Importance (LGBMRegressor)')
plt.show()


x=df.drop(['Hardness','id'],axis=1)
y=df[['Hardness']]


model = Sequential()
model.add(Dense(120, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(loss='mse', optimizer='adam')


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, verbose=1)


predictions=model.predict(x_test)
r2_score(y_test,predictions)


mean_squared_error(y_test,predictions)**0.5


loss_f=pd.DataFrame(history.history)
loss_f.plot()


predictions=model.predict(df_test)


submission['Hardness']=predictions


submission.to_csv('submission.csv',index=False)

