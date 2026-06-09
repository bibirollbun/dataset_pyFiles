import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings

import pickle

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import scale

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)
#全行数の出力
pd.set_option('display.max_rows', 138)


df=pd.read_csv('/kaggle/input/restaurant-revenue-prediction/train.csv.zip')


df.head()


top5_df = df.sort_values(by='revenue', ascending=False).head(15)
print(top5_df)


filtered_df = df[(df['City'] == 'İstanbul')]
row_count = len(filtered_df)
print("フィルタ後の行数：", row_count)
filtered_df


df.shape


# 売却価格のヒストグラム
sns.distplot(df['revenue'])
# 売却価格の概要をみてみる
print(df.describe())
print(f"歪度: {round(df['revenue'].skew(),4)}" )
print(f"尖度: {round(df['revenue'].kurt(),4)}" )


df.info()


null_values = df.isnull().sum()
null_values[null_values>0]


plt.figure(figsize=(25,25))
corr = df.drop('Id',axis=1).corr(numeric_only=True)
mask = corr < 0.9
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.2,mask = mask)
plt.title('Correlation Heatmap')
plt.show()


plt.figure(figsize=(12, 6))
sns.barplot(x='City', y='revenue', data=df, palette='viridis', estimator=sum)
plt.title('Total Revenue by City')
plt.xlabel('City')
plt.ylabel('Total Revenue')
plt.xticks(rotation=60)
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(x='Type', y='revenue', data=df, palette='coolwarm')
plt.title('Revenue Distribution by Type')
plt.xlabel('Type')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 5))
sns.countplot(x='City Group', data=df, palette='pastel')
plt.title('Number of Entities by City Group')
plt.xlabel('City Group')
plt.ylabel('Count of Entities')
plt.show()


plt.figure(figsize=(10, 5))
sns.scatterplot(x='P1', y='revenue', data=df, hue='Type', palette='deep', size='P1', sizes=(20, 200), legend=False)
plt.title('Revenue vs. P1')
plt.xlabel('P1')
plt.ylabel('Revenue')
plt.show()


plt.figure(figsize=(10, 5))
sns.scatterplot(x='P16', y='P36', data=df, hue='Type', palette='deep', size='P1', sizes=(20, 200), legend=False)
plt.title('Revenue vs. P1')
plt.xlabel('P1')
plt.ylabel('Revenue')
plt.show()


plt.figure(figsize=(10, 5))
sns.lineplot(x='P16', y='P36', data=df)
plt.title('P16 vs. P36')
plt.xlabel('P16')
plt.ylabel('P36')
plt.show()


df['Open_date'] = pd.to_datetime(df['Open Date'], dayfirst=False)
df['year'] = df.Open_date.dt.year
df['day'] = df.Open_date.dt.day
df['month'] = df.Open_date.dt.month
#df['P16✕P36'] = (df['P16']+df['P36'])/2



x=df.drop(['Id', 'Open Date','Open_date', 'City', 'Type','revenue','P36'],axis=1)
x=pd.get_dummies(x,drop_first=True)
y=df[['revenue']]


x.head()


def regression_funct(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize models
    L = LinearRegression()
    R = Ridge()
    Lass = Lasso()
    E = ElasticNet()
    ExTree = ExtraTreeRegressor()
    GBR = GradientBoostingRegressor()
    KN = KNeighborsRegressor()
    RF = RandomForestRegressor()  # Random Forest Regressor
    SV = SVR()                   # Support Vector Regressor
    XGB = XGBRegressor()         # XGBoost Regressor

    # List of algorithms and their names
    algos = [L, R, Lass, E, ExTree, GBR, KN, RF, SV, XGB]
    algo_names = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet',
                  'ExtraTreeRegressor', 'GradientBoostingRegressor',
                  'KNeighborsRegressor', 'RandomForestRegressor',
                  'SVR', 'XGBRegressor']

    # Initialize metrics lists
    r_squared = []
    rmse = []
    mae = []

    # Create DataFrame to store results
    result = pd.DataFrame(columns=['R_Squared', 'RMSE', 'MAE'], index=algo_names)

    for item in algos:
        item.fit(x_train, y_train)
        predictions = item.predict(x_test)

        r_squared.append(r2_score(y_test, predictions))
        rmse.append(np.sqrt(mean_squared_error(y_test, predictions)))
        mae.append(mean_absolute_error(y_test, predictions))

    # Fill in the results DataFrame
    result['R_Squared'] = r_squared
    result['RMSE'] = rmse
    result['MAE'] = mae

    return result.sort_values('R_Squared', ascending=False)


regression_funct(x, y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = XGBRegressor()
model.fit(x_train, y_train)

predict = model.predict(x_test)

mse = mean_squared_error(y_test, predict)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predict)
r2 = r2_score(y_test, predict)
print(f"Mean Squared Error: {mse}, \n Root Mean Squared Error: {rmse}, \n Mean Absolute Error: {mae}, \n R^2 Score: {r2}")



with open('restaurant_revenue.pkl', 'wb') as file:
    pickle.dump(model, file)


test_df=pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv.zip')


test_df.head()


test_df.info()


null_values = test_df.isnull().sum()
null_values[null_values>0]


test_df['Open_date'] = pd.to_datetime(test_df['Open Date'], dayfirst=False)
test_df['year'] = test_df.Open_date.dt.year
test_df['day'] = test_df.Open_date.dt.day
test_df['month'] = test_df.Open_date.dt.month
#test_df['P16✕P36'] = (test_df['P16']+test_df['P36'])/2



x=test_df.drop(['Id', 'Open Date','Open_date', 'City', 'Type', 'P36'],axis=1)
pred_x=pd.get_dummies(x,drop_first=True)


predictions = model.predict(pred_x)


submision = pd.DataFrame()
submision['Id'] = test_df['Id']
submision['Prediction'] = predictions


submision.tail()


submision.shape


submision.to_csv('submission.csv', index=False)




