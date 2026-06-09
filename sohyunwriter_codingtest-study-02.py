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



import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
from datetime import datetime

pd.options.display.max_columns = 100

import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import xgboost as xgb
import lightgbm as lgbm
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


import sklearn
sklearn.__version__


df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test_df = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")


df.head()


df.shape


# 데이터 구조 파악
# object 문자형
# int/float 숫자형
df.info()


# 문자형(범주형) 데이터 확인
df.describe(include=['object']).T


# 숫자형 데이터 확인
df.describe()


# 결측치 확인
df.isnull().values.sum()


# parse datetime colum & add new time related columns
df_dt = pd.DatetimeIndex(df['datetime'])
df.set_index(df_dt, inplace=True)


df.head(5)


df['date'] = df_dt.date
df['day'] = df_dt.day
df['month'] = df_dt.month
df['year'] = df_dt.year
df['hour'] = df_dt.hour
df['dow'] = df_dt.dayofweek


# How many columns have null values
df.isnull().sum()


categorizational_columns = ['season','holiday','workingday','weather','month','year','hour','dow']

#categorical하게 변환
for col in categorizational_columns:
    df[col] = df[col].astype('category')


# 온도와 체감온도 비교
plt.figure(figsize=(6, 4))
sns.distplot(df.temp, bins=10, label='real temp.')
sns.distplot(df.atemp, bins=10, label='feels like temp.')
plt.legend()
plt.show()


#습도와 바람속도 비
plt.figure(figsize=(6, 4))
sns.distplot(df.humidity, bins=10, label='humidity')
sns.distplot(df.windspeed, bins=10, label='windspeed')
plt.legend()
plt.show()


# 온도를 제외하면 다른 특성 간에는 명확한 상관관계가 없음
sns.pairplot(df[['temp', 'atemp', 'humidity', 'windspeed']])


# 시간성 정보
plt.figure(figsize=(6, 4))
sns.kdeplot(data=df['count'])
plt.show()


plt.figure(figsize=(8, 4))
sns.pointplot(x=df["hour"], y=df["count"], hue=df["season"])
plt.xlabel("Hour Of The Day")
plt.ylabel("Users Count")
plt.title("Rentals Across Hours")
plt.show()

'''
1 = spring, 2 = summer, 3 = fall, 4 = winter 
season 4인 겨울보다 1인 봄이 자전거 대여를 더 많이 안 함?!?
-> 잘못 라벨링되었다고 의심 & 계절과 날짜 재맵핑
3/4/5 -> 1 (봄) 6/7/8 -> 2 (여름) 9/10/11 -> 3(가을) 12/1/2 -> (겨울)
'''


df.head()


df[['season','month']].value_counts().sort_index()


def badToRight(month):
    if month in [12,1,2]:
        return 4
    elif month in [3,4,5]:
        return 1
    elif month in [6,7,8]:
        return 2
    elif month in [9,10,11]:
        return 3

#apply() 내장함수는 split(),map(),join(),filter()등 과 함꼐 필수적으로 숙지해야 할 함수이다.
df['season'] = df.month.apply(badToRight)
df['season'] = df['season'].astype('category')


plt.figure(figsize=(8, 4))
sns.pointplot(x=df["hour"], y=df["count"], hue=df["season"])
plt.xlabel("Hour Of The Day")
plt.ylabel("Users Count")
plt.title("Rentals Across Hours")
plt.show()


plt.figure(figsize=(8, 6))
sns.lineplot(x="hour", y="count", hue="season", data=df)


plt.figure(figsize=(8,6))
df_hours = pd.DataFrame(
    {"casual" : df.groupby(['hour'])['casual'].mean().values,
    "registered" : df.groupby(['hour'])['registered'].mean().values},
    index = df.groupby(['hour'])['casual'].mean().index)
df_hours.plot.bar(rot=0)
plt.title("Evolution of casual /registered bikers numbers over hours of the day")

plt.show()
     


plt.figure(figsize=(8,6))
df_hours = pd.DataFrame(
    {"casual" : df.groupby(['month'])['casual'].mean().values,
    "registered" : df.groupby(['month'])['registered'].mean().values},
    index = df.groupby(['month'])['casual'].mean().index)
df_hours.plot.bar(rot=0)
plt.title("Evolution of casual /registered bikers numbers over months of the year")

plt.show()


fig, ax = plt.subplots()
fig.set_size_inches(10, 8)
sns.boxplot(data=df, x="month", y="count", orient="v")
ax.set(xlabel="Months" , ylabel="Count", title="Count Across Month");


# working day, holiday 차이 확인

plt.figure(figsize=(10, 5))

bars = ['casual not on working days', 'casual on working days',\
        'registered not on working days', 'registered on working days',\
        'casual not on holidays', 'casual on holidays',\
        'registered not on holidays', 'registered on holidays']

qty = [df.groupby(['workingday'])['casual'].mean()[0], df.groupby(['workingday'])['casual'].mean()[1],\
      df.groupby(['workingday'])['registered'].mean()[0], df.groupby(['workingday'])['registered'].mean()[1],\
      df.groupby(['holiday'])['casual'].mean()[0], df.groupby(['holiday'])['casual'].mean()[1],\
      df.groupby(['holiday'])['registered'].mean()[0], df.groupby(['holiday'])['registered'].mean()[1]]

y_pos = np.arange(len(bars))
plt.barh(y_pos, qty, align='center')

plt.yticks(y_pos, labels=bars)
#plt.invert_yaxis()  # labels read top-to-bottom
plt.xlabel('Mean nb of bikers')
plt.title("Number of bikers on holidays / working days")
plt.show()


df.isnull().sum()


plt.figure(figsize=(8, 4))
sns.pointplot(x=df["hour"], y=df["count"], hue=df["season"])
plt.xlabel("Hour Of The Day")
plt.ylabel("Users Count")
plt.title("Rentals Across Hours")
plt.show()


# feature engineer a new column whether its a peak hour or not
df['peak'] = df[['hour', 'workingday']]\
    .apply(lambda df: 1 if ((df['workingday'] == 1 and (df['hour'] == 8 or 17 <= df['hour'] <= 18))) else 0, axis = 1)


df['peak'] = df['peak'].astype('category')
df.head()


df['time_int'] = df.year.astype('int')*10000 + df.month.astype('int')*100 + df.day.astype('int')


# 더미화 및 scale 데이터 확인
df_x = df[['season','holiday','workingday','weather','temp','atemp','humidity','windspeed','month','year','hour','dow','peak','time_int']]
df_y = df['count']


df_x.info()


#Let's convert all the categorical variables into dummy variables
df_dummies = pd.get_dummies(df_x)
df_dummies.head()


# Scaling all the variables to a range of 0 to 1
from sklearn.preprocessing import MinMaxScaler
features = df_dummies.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(df_dummies)
df_dummies = pd.DataFrame(scaler.transform(df_dummies))
df_dummies.columns = features


df_dummies.head()


# Create Train & Test Data
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(df_dummies, df_y, test_size=0.2, random_state=101)


from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, r2_score
import numpy as np

def evaluate_regression_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    medae = median_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    evs = explained_variance_score(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': mae,
        'MSE': mse,
        'MAPE': mape,
        'Median Absolute Error': medae,
        'RMSE': rmse,
        'Explained Variance Score': evs,
        'R2 Score': r2
    }
     


# -- 선형 회귀
from sklearn.linear_model import LinearRegression

# 모델 생성
linear_model = LinearRegression()

# 모델 훈련
linear_model.fit(X_train, y_train)

# 예측
y_pred = linear_model.predict(X_valid)
evaluate_regression_model(y_valid, y_pred)


# -- 릿지 회귀
from sklearn.linear_model import Ridge

# 모델 생성
ridge_model = Ridge(alpha=1.0)

# 모델 훈련
ridge_model.fit(X_train, y_train)

# 예측
y_pred = ridge_model.predict(X_valid)
evaluate_regression_model(y_valid,y_pred)


from sklearn.linear_model import Lasso

# 모델 생성
lasso_model = Lasso(alpha=0.1)

# 모델 훈련
lasso_model.fit(X_train, y_train)

# 예측
y_pred = lasso_model.predict(X_valid)
evaluate_regression_model(y_valid,y_pred)


from sklearn.linear_model import ElasticNet

# 모델 생성
elasticnet_model = ElasticNet(alpha=0.1, l1_ratio=0.5)

# 모델 훈련
elasticnet_model.fit(X_train, y_train)

# 예측
y_pred = elasticnet_model.predict(X_valid)
evaluate_regression_model(y_valid,y_pred)


from sklearn.svm import SVR

# 모델 생성
svm_model = SVR(kernel='linear', C=1.0)

# 모델 훈련
svm_model.fit(X_train, y_train)

# 예측
y_pred = svm_model.predict(X_valid)
evaluate_regression_model(y_valid,y_pred)


from sklearn.tree import DecisionTreeRegressor

# 모델 생성
decision_tree_model = DecisionTreeRegressor()

# 모델 훈련
decision_tree_model.fit(X_train, y_train)

# 예측
y_pred = decision_tree_model.predict(X_valid)
evaluate_regression_model(y_valid,y_pred)


from sklearn.ensemble import RandomForestRegressor

# 모델 생성
random_forest_model = RandomForestRegressor(n_estimators=100)

# 모델 훈련
random_forest_model.fit(X_train, y_train)

# 예측
y_pred = random_forest_model.predict(X_valid)
evaluate_regression_model(y_valid,y_pred)


from sklearn.ensemble import GradientBoostingRegressor

# 모델 생성
gbm_model = GradientBoostingRegressor(n_estimators=100)

# 모델 훈련
gbm_model.fit(X_train, y_train)

# 예측
y_pred = gbm_model.predict(X_valid)
evaluate_regression_model(y_valid,y_pred)


from xgboost import XGBRegressor

# 모델 생성
xgb_model = XGBRegressor(n_estimators=100)

# 모델 훈련
xgb_model.fit(X_train, y_train)

# 예측
y_pred = xgb_model.predict(X_valid)
evaluate_regression_model(y_valid,y_pred)


from lightgbm import LGBMRegressor

# 모델 생성
lgbm_model = LGBMRegressor(n_estimators=100)

# 모델 훈련
lgbm_model.fit(X_train, y_train)

# 예측
y_pred = lgbm_model.predict(X_valid)
evaluate_regression_model(y_valid,y_pred)


# 모델 생성
gbm_model = GradientBoostingRegressor(n_estimators=100)
xgb_model = XGBRegressor(n_estimators=100)
lgbm_model = LGBMRegressor(n_estimators=100)

# 모델 훈련
gbm_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
lgbm_model.fit(X_train, y_train)

y_pred1 = gbm_model.predict(X_valid)
y_pred2 = xgb_model.predict(X_valid)
y_pred3 = lgbm_model.predict(X_valid)

# 예측값 평균
y_pred_ensemble = (y_pred1 + y_pred2 + y_pred3) / 3
evaluate_regression_model(y_valid,y_pred_ensemble)


!pip install scikit-optimize


from skopt import BayesSearchCV
ridge_model = Ridge()

search = BayesSearchCV(
    ridge_model,
    {
        'alpha': (1e-6, 1e+6, 'log-uniform')
    },
    n_iter=32,
    cv=3
)

search.fit(X_train, y_train)
print(search.best_params_)
y_pred = search.predict(X_valid)
evaluate_regression_model(y_valid,y_pred)


from sklearn.model_selection import RandomizedSearchCV
lasso_model = Lasso()

search = BayesSearchCV(
    lasso_model,
    {
        'alpha': (1e-6, 1e+6, 'log-uniform')
    },
    n_iter=32,
    cv=3
)

search.fit(X_train, y_train)
print(search.best_params_)
y_pred = search.predict(X_valid)
evaluate_regression_model(y_valid,y_pred)


elasticnet_model = ElasticNet()

search = BayesSearchCV(
    elasticnet_model,
    {
        'alpha': (1e-6, 1e+6, 'log-uniform'),
        'l1_ratio': (0, 1, 'uniform')
    },
    n_iter=32,
    cv=3
)

search.fit(X_train, y_train)
print(search.best_params_)
y_pred = search.predict(X_valid)
evaluate_regression_model(y_valid,y_pred)


from sklearn.svm import SVR
svm_model = SVR()

search = BayesSearchCV(
    svm_model,
    {
        'C': (1e-6, 1e+6, 'log-uniform'),
        'epsilon': (1e-6, 1, 'log-uniform')
    },
    n_iter=32,
    cv=3
)

search.fit(X_train, y_train)
print(search.best_params_)
y_pred = search.predict(X_valid)
evaluate_regression_model(y_valid,y_pred)


from sklearn.tree import DecisionTreeRegressor
decision_tree_model = DecisionTreeRegressor()

search = BayesSearchCV(
    decision_tree_model,
    {
        'max_depth': (1, 50),
        'min_samples_split': (2, 100),
        'min_samples_leaf': (1, 100)
    },
    n_iter=32,
    cv=3
)

search.fit(X_train, y_train)
print(search.best_params_)
y_pred = search.predict(X_valid)
evaluate_regression_model(y_valid,y_pred)


from sklearn.ensemble import RandomForestRegressor
random_forest_model = RandomForestRegressor()

search = BayesSearchCV(
    random_forest_model,
    {
        "n_estimators": (50, 300),
        "max_depth": (5, 50),
        "min_samples_split": (2, 20),
        "max_features": (0.1, 0.999),
    },
    n_iter=32,
    cv=3
)

search.fit(X_train, y_train)
print(search.best_params_)
y_pred = search.predict(X_valid)
evaluate_regression_model(y_valid,y_pred)


from sklearn.ensemble import GradientBoostingRegressor
gbm_model = GradientBoostingRegressor()

search = BayesSearchCV(
    gbm_model,
    {
        'n_estimators': (10, 1000),
        'learning_rate': (1e-6, 1.0, 'log-uniform'),
        'max_depth': (1, 50),
        'min_samples_split': (2, 100),
        'min_samples_leaf': (1, 100)
    },
    n_iter=32,
    cv=3
)

search.fit(X_train, y_train)
print(search.best_params_)
y_pred = search.predict(X_valid)
evaluate_regression_model(y_valid,y_pred)



from xgboost import XGBRegressor
xgb_model = XGBRegressor()

search = BayesSearchCV(
    xgb_model,
    {
        "max_depth": (3, 10),
        "learning_rate": (0.01, 0.3),
        "n_estimators": (50, 300),
        "gamma": (0, 5),
        "min_child_weight": (1, 10),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
    },
    n_iter=32,
    cv=3
)

search.fit(X_train, y_train)
print(search.best_params_)
y_pred = search.predict(X_valid)
evaluate_regression_model(y_valid,y_pred)


from lightgbm import LGBMRegressor

lgbm_model = LGBMRegressor()

search = BayesSearchCV(
    lgbm_model,
    {
        'n_estimators': (50, 500),
        'learning_rate': (0.01, 0.3),
        'max_depth': (5, 30),
        'num_leaves': (20, 60),
        "min_child_samples": (10, 200),
        "subsample": (0.7, 1.0),
        "colsample_bytree": (0.7, 1.0),
    },
    n_iter=32,
    cv=3
)

search.fit(X_train, y_train)
print(search.best_params_)
y_pred = search.predict(X_valid)
evaluate_regression_model(y_valid,y_pred)










