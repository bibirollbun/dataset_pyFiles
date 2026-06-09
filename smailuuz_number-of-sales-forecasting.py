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


df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
df.head()


df.info()


df_test = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
df_test.info()


df = df.dropna()


import datetime as dt

def date_time(x):
    x = x.set_index("id")
    x["date"] = pd.to_datetime(x["date"])
    x["year"] = x["date"].dt.year.astype("int64")
    x["month"] = x["date"].dt.month.astype("int64")
    x["day"] = x["date"].dt.day.astype("int64")
    x["day_ofweek"] = x["date"].dt.weekday.astype("int64")
    x["is_weekend"] = x['date'].dt.weekday >= 5
    x = x.drop("date",axis=1)
    return x


df = date_time(df)
df_test = date_time(df_test)


def time_sin_cos_features(x):
    x["day_sin4"] = np.sin(x["day"] * (8 * np.pi /  365.0))
    x["day_cos4"] = np.cos(x["day"] * (8 * np.pi /  365.0)) 
    x["day_sin_0.5"] = np.sin(x["day"] * (1 * np.pi /  365.0))
    x["day_cos_0.5"] = np.cos(x["day"] * (1 * np.pi /  365.0))    
    x["month_sin"] = np.sin(2 * np.pi * x["month"] / 12.0)
    x["month_cos"] = np.cos(2 * np.pi * x["month"] / 12.0)
    x["year_sin"] = np.sin(2 * np.pi * x["year"] / 7.0)
    x["year_cos"] = np.cos(2 * np.pi * x["year"] / 7.0)
    return x


df = time_sin_cos_features(df)
df_test = time_sin_cos_features(df_test)


from sklearn.preprocessing import OneHotEncoder


cols_to_encode = (["country", "store", "product"])

encoder = OneHotEncoder(sparse_output=False,drop=None)

def encoding_(x):
    x= pd.get_dummies(x)
    
    return x


df = encoding_(df)
df_test = encoding_(df_test)


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

x = df.drop("num_sold",axis=1)
y = df["num_sold"]

x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_scaler.fit(x)
x_scaled = x_scaler.transform(x)

y_scaler.fit(np.array(y).reshape(-1,1))
y_scaled = y_scaler.transform(np.array(y).reshape(-1,1))

test_scaled = x_scaler.transform(df_test)



x_train,x_test,y_train,y_test = train_test_split(x_scaled,y_scaled,test_size=0.3,random_state=42)


import lightgbm as lgb

lgbm = lgb.LGBMRegressor()

lgbm.fit(x_scaled,y_scaled, eval_metric='rmse')





y_pred_lgbm = lgbm.predict(x_test)


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print(f"MSE : {mean_squared_error(y_test,y_pred_lgbm)}")
print(f"R2 Score : {r2_score(y_test,y_pred_lgbm)}")


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(lgbm, x_scaled, y_scaled, cv=5, scoring='neg_mean_squared_error')
print(f"Cross Val Score : {cv_scores * -1}")
print(f"Cross Val Score Mean: {cv_scores.mean() * -1}")
print(f"Cross Val Score Std: {cv_scores.std()}")


import xgboost as xgb

xgbr = xgb.XGBRegressor()

xgbr.fit(x_train, y_train)



y_pred_xgbr = xgbr.predict(x_test)


print(f"MSE : {mean_squared_error(y_test,y_pred_xgbr)}")
print(f"R2 Score : {r2_score(y_test,y_pred_xgbr)}")


cv_scores = cross_val_score(xgbr, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross Val Score : {cv_scores * -1}")
print(f"Cross Val Score Mean: {cv_scores.mean() * -1}")
print(f"Cross Val Score Std: {cv_scores.std()}")


from sklearn.linear_model import ElasticNet

elsnt = ElasticNet()

elsnt.fit(x_train, y_train)



y_pred_elsnt = elsnt.predict(x_test)


print(f"MSE : {mean_squared_error(y_test,y_pred_elsnt)}")
print(f"R2 Score : {r2_score(y_test,y_pred_elsnt)}")


cv_scores = cross_val_score(elsnt, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross Val Score : {cv_scores * -1}")
print(f"Cross Val Score Mean: {cv_scores.mean() * -1}")
print(f"Cross Val Score Std: {cv_scores.std()}")


from catboost import CatBoostRegressor

cbr = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.05, loss_function='RMSE', random_seed=42,verbose=0)

cbr.fit(x_train,y_train)


y_pred_cbr = cbr.predict(x_test)


print(f"MSE : {mean_squared_error(y_test,y_pred_cbr)}")
print(f"R2 Score : {r2_score(y_test,y_pred_cbr)}")


cv_scores = cross_val_score(cbr, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross Val Score : {cv_scores * -1}")
print(f"Cross Val Score Mean: {cv_scores.mean() * -1}")
print(f"Cross Val Score Std: {cv_scores.std()}")


predictions = cbr.predict(test_scaled)


predictions = y_scaler.inverse_transform(np.array(predictions).reshape(-1,1))
predictions 


predictions = pd.DataFrame(predictions,columns=["num_sold"])


output = pd.DataFrame({"id" : df_test.index, "num_sold" :predictions["num_sold"]})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")

