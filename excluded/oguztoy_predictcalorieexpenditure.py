import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import os





for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df1 = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df1.head()


df2 = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
df2.head()


df1.shape, df2.shape


df = pd.concat([df1,df2])


df.sample(7)


#Feature Engineering
df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
df['Heart_Rate_per_Age'] = df['Heart_Rate'] / df['Age']
df['Temp_HR_Interaction'] = df['Body_Temp'] * df['Heart_Rate']
df['HR_above_mean'] = df['Heart_Rate'] - df['Heart_Rate'].mean()
df['Log_Weight'] = np.log1p(df['Weight'])
df['Age_squared'] = df['Age'] ** 2
df['Weight_Height'] = df['Weight'] * df['Height']
df['Duration_HR'] = df['Duration'] * df['Heart_Rate']


df.sample(5)


plt.figure(figsize=(20,12))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Blues");


abs(df.corr(numeric_only=True)["Calories"].sort_values(ascending=False))


df.drop(columns=["Height","Log_Weight","Weight_Height","Weight","Age"], axis=1, inplace=True)


df = pd.get_dummies(df, drop_first=True)


train=df[:750000]
test=df[750000:]


x = train.drop(["id","Calories"], axis=1)
y = train[["Calories"]]
test = test.drop(["id","Calories"], axis=1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=8)


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
    r2 = r2_score(y_test, pred)
    rmse = mean_squared_error(y_test, pred, squared=False)
    print(f"{name} → R²: {r2:.4f} | RMSE: {rmse:.2f}")


model = lgb_model.fit(x,y)
prediction = model.predict(test)


prediction


submission = pd.DataFrame({"id": df2["id"], "Calories":prediction})


submission


submission.to_csv("submission.csv", index=False)




