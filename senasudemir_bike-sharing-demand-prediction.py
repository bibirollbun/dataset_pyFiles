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


df=pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')


df.head()


df.shape


df.isnull().sum()


df.info()


df.describe().T


def feature_engineering(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday  
    df['month'] = df['datetime'].dt.month
    df["season"]=df.season.map({1:"Spring",2:"Summer",3:"Fall",4:"Winter"})
    df["weather"] = df["weather"].map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",\
                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \
                                        3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \
                                        4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })


feature_engineering(df)


df.head()


lower_bound = df['count'].quantile(q=0.03)
upper_bound = df['count'].quantile(q=0.97)
df = df[(df['count'] >= lower_bound) & (df['count'] <= upper_bound)]


df.shape


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(numeric_only=True),annot=True);


plt.figure(figsize=(12, 6))
sns.lineplot(x="hour", y="count", data=df, estimator="mean", ci=None)
plt.title("Average Bike Demand by Hour of the Day")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Count")
plt.grid(True)
plt.show();


plt.figure(figsize=(10, 5))
sns.barplot(x="season", y="count", data=df, estimator="mean", ci=None, palette="coolwarm")
plt.title("Bike Demand Across Seasons")
plt.xlabel("Season (1:Winter, 2:Spring, 3:Summer, 4:Fall)")
plt.ylabel("Average Count")
plt.show()


plt.figure(figsize=(12, 6))
sns.lineplot(x="hour", y="count", hue="season", data=df, estimator="mean", ci=None, palette="coolwarm")
plt.title("Hourly Bike Demand by Season")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Count")
plt.legend(title="Season")
plt.show()


plt.figure(figsize=(10, 5))
sns.boxplot(x="weather", y="count", data=df, palette="coolwarm")
plt.title("Bike Demand by Weather Condition")
plt.xlabel("Weather (1:Clear, 2:Mist, 3:Light Rain, 4:Heavy Rain)")
plt.xticks(rotation=90)
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(10, 5))
sns.scatterplot(x="temp", y="count", data=df, alpha=0.5)
plt.title("Temperature vs. Bike Demand")
plt.xlabel("Temperature (Â°C)")
plt.ylabel("Count")
plt.show()


abs(df.corr(numeric_only=True)['count'].sort_values(ascending=False))


x=df.drop(columns=["count","datetime","registered"],axis=1)
y=df[["count"]]


x.shape,y.shape


x=pd.get_dummies(x,drop_first=True)


def regression_algo(x,y,plot=False):
    L=LinearRegression()
    R=Ridge()
    Lass=Lasso()
    E=ElasticNet()
    ETR=ExtraTreeRegressor()
    GBR=GradientBoostingRegressor()
    kn=KNeighborsRegressor()
    dt=DecisionTreeRegressor()
    xgb=XGBRegressor()
    rf=RandomForestRegressor()

    algos=[L,R,Lass,E,ETR,GBR,kn,dt,xgb,rf]
    algo_names=['Linear','Ridge','Lasso','ElasticNet','Extra Tree','Gradient Bossting','KNeighborRegressor','DecisionTreeRegressor','XGBReggressor','Random Forest Classifier']

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)

    r_squared=[]
    rmse=[]
    mae=[]

    result=pd.DataFrame(columns=['R_Squared','RMSE','MAE'],index=algo_names)

    for algo in algos:
        p=algo.fit(x_train,y_train).predict(x_test)
        r_squared.append(r2_score(y_test,p))
        rmse.append(mean_squared_error(y_test,p)**0.5)
        mae.append(mean_absolute_error(y_test,p))

    result.R_Squared=r_squared
    result.RMSE=rmse
    result.MAE=mae

    r_table=result.sort_values('R_Squared',ascending=False)
    if plot:
        best_model = algos[r_squared.index(max(r_squared))]
        y_pred = best_model.predict(x_test)
        
        plt.figure(figsize=(10,6))
        plt.plot(y_test.reset_index(drop=True),label='Acutal',color='green',linestyle='--')
        plt.plot(pd.Series(y_pred.flatten()), label='Predicted', color='red', linestyle='--')
        plt.xlabel('Predicted Demand')
        plt.ylabel('Actual Demand')
        plt.title('Actual vs Predicted Demand for Best Model')
        plt.legend()
        plt.show()
    return r_table


regression_algo(x,y,plot=False)


x=df.drop(columns=["count","datetime","registered"],axis=1)
y=df[["count"]]
x=pd.get_dummies(x,drop_first=True)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
xgb=XGBRegressor()
model=xgb.fit(x_train,y_train)


import joblib
joblib.dump(model, 'best_regression_model.pkl')


df_test=pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')


feature_engineering(df_test)


df_test.head()


submission=pd.DataFrame({
    'datetime':df_test['datetime']
})



df_test = df_test.drop(columns=["datetime"], axis=1)
df_test = pd.get_dummies(df_test, drop_first=True)
df_test = df_test.reindex(columns=x_train.columns, fill_value=0)





predictions=model.predict(df_test)


submission["count"]=predictions


submission.loc[submission["count"] < 0, "count"] = 0


submission.to_csv("submission.csv",index=False)


importance = xgb.get_booster().get_score(importance_type='weight')
feature_importance = pd.DataFrame({'Feature': list(importance.keys()), 'Importance': list(importance.values())})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False).head(30)
plt.figure(figsize=(6, 15))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.gca().invert_yaxis()
plt.title('Top 30 Features by Importance (XGBoost Model)')
plt.show()


predictions=model.predict(x_test)
residuals = y_test.values.flatten() - predictions.flatten()
sns.histplot(residuals,bins=100,kde=True);


sns.kdeplot(x=residuals,fill=True);


x=df.drop(columns=["count","datetime","registered"],axis=1)
y=df[["count"]]
x=pd.get_dummies(x,drop_first=True)


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


history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, verbose=1)


model.save("model.h5")


predictions=model.predict(x_test)
r2_score(y_test,predictions)


type(x_test)


mean_squared_error(y_test,predictions)**0.5


loss_f=pd.DataFrame(history.history)
loss_f.plot()


df_test=pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
feature_engineering(df_test)
df_test = df_test.drop(columns=["datetime"], axis=1)
df_test = pd.get_dummies(df_test, drop_first=True)
df_test = df_test.reindex(columns=x_train.columns, fill_value=0)


predictions=model.predict(df_test)


submission["count"]=predictions


submission.to_csv("submission.csv",index=False)

