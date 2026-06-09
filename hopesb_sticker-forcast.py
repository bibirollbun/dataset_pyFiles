import pandas as pd
import numpy as np

import seaborn as sns
import plotly_express as px
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
from category_encoders import OneHotEncoder, OrdinalEncoder
import optuna


train_filepath = "/kaggle/input/playground-series-s5e1/train.csv"
test_filepath = "/kaggle/input/playground-series-s5e1/test.csv"


def wrangle(filepath):
    df = pd.read_csv(filepath, index_col="id")
    df = df.dropna()

    date = pd.to_datetime(df["date"]).dt

    df["year"] = date.year
    df["day_of_year"] = date.dayofyear
    df["sin_dayofyear"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_dayofyear"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["day_of_week"] = date.dayofweek
    df["day"] = date.day
    df["sin_dayofmonth"] = np.sin(2 * np.pi * df["day"]/ 31)
    df["cos_dayofmonth"] = np.cos(2 * np.pi * df["day"]/ 31)
    df["sin_cos"] = np.sin(2 * np.pi * (df["sin_dayofyear"] + df["cos_dayofyear"])/2)
  
    return df


df = wrangle(train_filepath)
df.head()


df.describe()


df.info()


sns.heatmap(df.select_dtypes("number").corr());


target = "num_sold"
X= df.drop(columns=target)
y= df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = make_pipeline(
    OrdinalEncoder(),
    StandardScaler(),
    LinearRegression()
)
model.fit(X_train, np.log1p(y_train))
y_pred = np.expm1(model.predict(X_train))
score = mean_absolute_percentage_error(y_train, y_pred)

print(score)
y_pred[:5]


xgb_params = {'max_depth': 15, 'learning_rate': 0.09128813028635434, 'n_estimators': 2637}
xgb = make_pipeline(
    OrdinalEncoder(),
    StandardScaler(),
    XGBRegressor(**xgb_params, random_state=42)
)
xgb.fit(X_train, np.log1p(y_train))
y_pred = np.expm1(xgb.predict(X_train))
score = mean_absolute_percentage_error(y_train, y_pred)
print(score)
y_pred[:5]


lgb_params = {'max_depth': 12, 'learning_rate': 0.09943077052196123, 'n_estimators': 2984}
lgb = make_pipeline(
    OrdinalEncoder(),
    StandardScaler(),
    LGBMRegressor(**lgb_params, verbose=0, random_state=42)
)
lgb.fit(X_train, np.log1p(y_train))
y_pred = np.expm1(lgb.predict(X_test))
score = mean_absolute_percentage_error(y_test, y_pred)
print(score)
y_pred[:5]


cat = make_pipeline(
    OrdinalEncoder(),
    StandardScaler(),
    CatBoostRegressor(verbose=0, random_state=42)
)
cat.fit(X_train, np.log1p(y_train))
y_pred = np.expm1(cat.predict(X_test))
score = mean_absolute_percentage_error(y_test, y_pred)
print(score)
y_pred[:5]


estimators = [
    ("xgb", xgb), ("lgb", lgb), ("cat", cat)
]
vote = VotingRegressor(estimators=estimators, weights=[3, 2, 1])
vote.fit(X_train, np.log1p(y_train))
y_pred = np.expm1(vote.predict(X_test))
score = mean_absolute_percentage_error(y_test, y_pred)
print(score)


test = wrangle(test_filepath)
y_pred = np.expm1(vote.predict(test))
pd.DataFrame({"num_sold": y_pred}, index=test.index).to_csv("submission.csv")




