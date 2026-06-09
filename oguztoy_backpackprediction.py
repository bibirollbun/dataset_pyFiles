import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




df1 = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
df1.head()


df2 = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
df2.head()


df1.shape, df2.shape


df = pd.concat([df1,df2])


df.info()


df.isnull().sum()


df.value_counts("Size")


s = {"Small": 1, "Medium": 2, "Large": 3}
df["Size"] = df["Size"].map(s)


df.value_counts("Waterproof")


d = {"Yes": 1, "No":0}
df["Waterproof"] = df["Waterproof"].map(d)


df.value_counts("Laptop Compartment")
df["Laptop Compartment"] = df["Laptop Compartment"].map(d)


df["Brand"].fillna(df["Brand"].mode()[0], inplace=True)
df["Material"].fillna(df["Material"].mode()[0], inplace=True)
df["Size"].fillna(df["Size"].mode()[0], inplace=True)
df["Laptop Compartment"].fillna(df["Laptop Compartment"].mode()[0], inplace=True)
df["Waterproof"].fillna(df["Waterproof"].mode()[0], inplace=True)
df["Style"].fillna(df["Style"].mode()[0], inplace=True)
df["Color"].fillna(df["Color"].mode()[0], inplace=True)
df["Weight Capacity (kg)"].fillna(df["Weight Capacity (kg)"].mean(), inplace=True)


df["Brand_Material"] = df["Brand"] + "_" + df["Material"]
df["Style_Size"] = df["Style"] + "_" + df["Size"].astype(str)

df["Weight_per_Compartment"] = df["Weight Capacity (kg)"] / (df["Compartments"] + 1)


df = pd.get_dummies(df, drop_first=True)


train=df[:300000]
test=df[300000:]


x = train.drop(["id","Price"], axis=1)
y = train[["Price"]]
test = test.drop(["id","Price"], axis=1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=8)


xgb_model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
cat_model = CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05, verbose=0, random_seed=42)
lgb_model = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
rf_model  = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)

xgb_model.fit(x_train, y_train)
cat_model.fit(x_train, y_train)
lgb_model.fit(x_train, y_train)
rf_model.fit(x_train, y_train)


models = {
    "XGBoost": xgb_model,
    "CatBoost": cat_model,
    "LightGBM": lgb_model,
    "Random Forest": rf_model
}

for name, model in models.items():
    pred = model.predict(x_test)

    rmse = mean_squared_error(y_test, pred, squared=False)
    print(f"RMSE: {rmse:.2f}")


model = xgb_model.fit(x,y)
prediction = model.predict(test)


prediction


submission = pd.DataFrame({"id": df2["id"], "Price":prediction})


submission


submission.to_csv("submission.csv", index=False)




