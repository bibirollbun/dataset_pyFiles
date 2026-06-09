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


df=pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")



df.head()


df.info()


df["Episode_Length_minutes"].value_counts()


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.boxplot(x=df["Episode_Length_minutes"])
plt.title("Outlier Detection via Boxplot")
plt.show()




df.describe()


print(df[df["Episode_Length_minutes"] > 300])



df=df.drop(101637,axis=0)


df.describe()


df["Episode_Length_minutes"]=df["Episode_Length_minutes"].fillna(df["Episode_Length_minutes"].mean())


df.info()


df.head()


df.describe()


df["Guest_Popularity_percentage"]=df["Guest_Popularity_percentage"].fillna(df["Guest_Popularity_percentage"].mean())


df.info()


df.head()


df["Genre"].value_counts()


df["Number_of_Ads"].value_counts()


counts = df["Number_of_Ads"].value_counts()

remove = counts[(counts == 1) | (counts == 2)].index

df = df[~df["Number_of_Ads"].isin(remove)]



df["Number_of_Ads"].value_counts()


df.head()


df.isnull().sum()


df["Number_of_Ads"]=df["Number_of_Ads"].fillna(df["Number_of_Ads"].mean())


df=df.drop(columns=["id","Episode_Title","Podcast_Name"])
df=pd.get_dummies(df,dtype=int)


corr_matrix=df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix)
plt.title("Corr Matrix")
plt.show()


df.head()


test=pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


test.head()


test.info()


test["Episode_Length_minutes"].value_counts()



test.describe()


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.boxplot(x=test["Episode_Length_minutes"])
plt.title("Outlier Detection via Boxplot")
plt.show()



print(test[test["Episode_Length_minutes"] > 300])



mean_value = test[test["Episode_Length_minutes"] <= 300]["Episode_Length_minutes"].mean()
test.loc[test["Episode_Length_minutes"] > 300, "Episode_Length_minutes"] = mean_value

test["Episode_Length_minutes"]=test["Episode_Length_minutes"].fillna(test["Episode_Length_minutes"].mean())



plt.figure(figsize=(10, 6))
sns.boxplot(x=test["Episode_Length_minutes"])
plt.title("Outlier Detection via Boxplot")
plt.show()

test.describe()


test.info()


test["Guest_Popularity_percentage"].value_counts()


test["Guest_Popularity_percentage"]=test["Guest_Popularity_percentage"].fillna(test["Guest_Popularity_percentage"].mean())


test.info()


test.head()


test=test.drop(columns=["Episode_Title","Podcast_Name"])
test=test.set_index("id")


test["Number_of_Ads"].value_counts()


test.loc[test["Number_of_Ads"] == 2063.00, "Number_of_Ads"] = 1.0
test.loc[test["Number_of_Ads"] == 89.12, "Number_of_Ads"] = 0.0

test=pd.get_dummies(test,dtype=int)


df.head()


test.head()


X=df.drop("Listening_Time_minutes",axis=1)
y=df["Listening_Time_minutes"]


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
from sklearn.preprocessing import StandardScaler
from math import sqrt
scaler=StandardScaler()


def compare_Models(X, y, test_size=0.2, random_state=42):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "XGBoost": XGBRegressor(random_state=random_state, verbosity=0)
    }

    results = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)

    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        duration = time.time() - start_time

        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse=sqrt(mse)

        results.append({
            "Model": name,
            "Training Time (s)": round(duration, 4),
            "R2 Score": round(r2, 4),
            "RMSE": round(rmse, 4)
        })

    result_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
    return result_df



compare_Models(X,y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
model=XGBRegressor(subsample=1.0,
    reg_lambda=0.1,
    reg_alpha=1,
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    gamma=0,
    colsample_bytree=0.6,
    random_state=42,
    verbosity=0)


model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)
r2_Score=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=sqrt(mse)
print(f"R2 Score: {r2_Score} and RMSE: {rmse}")


predict=model.predict(test)


predict=pd.DataFrame(predict,index=test.index)


predict.head()


predict.reset_index(inplace=True)



predict.head()


predict.columns=["id","Listening_Time_minutes"]


predict.head()


predict.info()


predict.to_csv("submission.csv", index=False)





