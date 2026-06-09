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


df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
df_train.sample(3)


df_train.isnull().sum()


df_train["Episode_Length_minutes"] = df_train["Episode_Length_minutes"].fillna(df_train["Episode_Length_minutes"].mean())
df_train["Guest_Popularity_percentage"] = df_train["Guest_Popularity_percentage"].fillna(df_train["Guest_Popularity_percentage"].mean())
df_train["Number_of_Ads"] = df_train["Number_of_Ads"].fillna(df_train["Number_of_Ads"].mean())


df_train.isnull().sum()


df_train = df_train.drop(columns = ["id","Podcast_Name","Episode_Title"])


df_train = pd.get_dummies(df_train, columns=['Genre','Publication_Day','Publication_Time','Episode_Sentiment'],drop_first=True).astype(int)


X = df_train.drop(columns=['Listening_Time_minutes'])
y = df_train["Listening_Time_minutes"]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def objective(trial):
     n_estimators = trial.suggest_int("n_estimators", 10, 500)
     max_depth = trial.suggest_int("max_depth", 1, 20)
     min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
     min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
     max_features = trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])

     model = RandomForestRegressor(
         n_estimators = n_estimators,
         max_depth = max_depth,
         min_samples_split = min_samples_split,
         min_samples_leaf=min_samples_leaf,
         max_features = max_features,
         random_state = 42,
         n_jobs=-1
     )
     model.fit(X_train,y_train)
     y_pred = model.predict(X_test)
     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
     return rmse


study = optuna.create_study(direction = "minimize")
study.optimize(objective, n_trials=25)


best_params = study.best_params
final_model = RandomForestRegressor(**best_params,random_state=42)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)


df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
df_test.isnull().sum()


Episode_Length_minutes_mean = df_test["Episode_Length_minutes"].mean()
Guest_Popularity_percentage_mean = df_test["Guest_Popularity_percentage"].mean()
df_test["Episode_Length_minutes"] = df_test["Episode_Length_minutes"].fillna(Episode_Length_minutes_mean)
df_test["Guest_Popularity_percentage"] = df_test["Guest_Popularity_percentage"].fillna(Guest_Popularity_percentage_mean)


df_test.isnull().sum()


test_id = df_test["id"]


df_test = df_test.drop(columns= ["id", "Podcast_Name","Episode_Title"])


df_test = pd.get_dummies(df_test, columns=['Genre','Publication_Day','Publication_Time','Episode_Sentiment'], drop_first=True).astype(int)


pred = final_model.predict(df_test)


submission = pd.DataFrame({
    "id": test_id, 
    "Listening_Time_minutes": pred 
})

# Save submission file
submission.to_csv("Podcast_submission.csv", index=False)




