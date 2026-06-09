# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
train_df.head()


test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
test_df.head()


train_df["date"] = pd.to_datetime(train_df["date"])


train_df["Year"] = train_df["date"].dt.year
train_df["Quarter"] = train_df["date"].dt.quarter
train_df["Month"] = train_df["date"].dt.month
train_df["WeekofYear"] = train_df["date"].dt.isocalendar().week
train_df["Week"] = train_df["date"].dt.weekday
train_df['Holiday'] = train_df['date'].dt.dayofweek >= 5
train_df


train_df['Year_sin'] = np.sin(2 * np.pi * (train_df['Year'] - train_df['Year'].min()) / 2)
train_df['Year_cos'] = np.cos(2 * np.pi * (train_df['Year'] - train_df['Year'].min()) / 2)

train_df['Quarter_sin'] = np.sin(2 * np.pi * train_df['Quarter'] / 4) 
train_df['Quarter_cos'] = np.cos(2 * np.pi * train_df['Quarter'] / 4)

train_df['Month_sin'] = np.sin(2 * np.pi * train_df['Month'] / 12)
train_df['Month_cos'] = np.cos(2 * np.pi * train_df['Month'] / 12)

train_df['WeekofYear_sin'] = np.sin(2 * np.pi * train_df['WeekofYear'] / 52)
train_df['WeekofYear_cos'] = np.cos(2 * np.pi * train_df['WeekofYear'] / 52)

train_df['Week_sin'] = np.sin(2 * np.pi * train_df['Week'] / 7)
train_df['Week_cos'] = np.cos(2 * np.pi * train_df['Week'] / 7)


train_df["num_sold"] = train_df.groupby("country")["num_sold"].transform(lambda x: x.ffill().bfill())
print(train_df.isnull().sum())


remove_col = ["id","date","Month",'Quarter','WeekofYear','Week',"Year"]
train_df = train_df.drop(remove_col, axis=1)


train_df = pd.get_dummies(train_df, columns=["country","store","product"])


train_df["num_sold"] = np.log(train_df["num_sold"] + 1)


X = train_df.drop(["num_sold"],axis =1)
y = train_df["num_sold"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


import xgboost as xgb
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=123)
model.fit(X_train, y_train)
importance = model.get_booster().get_score(importance_type='weight')
importance_df = pd.DataFrame(importance.items(), columns=['Feature', 'Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type='weight', max_num_features=25, title="Feature Importance", height=0.5)
plt.show()


from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
X = train_df.drop(columns=['num_sold'], axis = 1) 
y = train_df['num_sold']


from sklearn.model_selection import GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=123)

param_grid = {
    'max_depth': [6, 7],
    'learning_rate': [0.3],
    'n_estimators': [350]
}

grid_search = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror'),
                           param_grid=param_grid,
                           cv=kf,
                           scoring='neg_mean_squared_error')

grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)


test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")


test_df["date"] = pd.to_datetime(test_df["date"])
test_df["Year"] = test_df["date"].dt.year
test_df["Quarter"] = test_df["date"].dt.quarter
test_df["Month"] = test_df["date"].dt.month
test_df["WeekofYear"] = test_df["date"].dt.isocalendar().week
test_df["Week"] = test_df["date"].dt.weekday
test_df['Holiday'] = test_df['date'].dt.dayofweek >= 5

test_df['Year_sin'] = np.sin(2 * np.pi * (test_df['Year'] - test_df['Year'].min()) / 2)
test_df['Year_cos'] = np.cos(2 * np.pi * (test_df['Year'] - test_df['Year'].min()) / 2)
test_df['Quarter_sin'] = np.sin(2 * np.pi * test_df['Quarter'] / 4) 
test_df['Quarter_cos'] = np.cos(2 * np.pi * test_df['Quarter'] / 4)
test_df['Month_sin'] = np.sin(2 * np.pi * test_df['Month'] / 12)
test_df['Month_cos'] = np.cos(2 * np.pi * test_df['Month'] / 12)
test_df['WeekofYear_sin'] = np.sin(2 * np.pi * test_df['WeekofYear'] / 52)
test_df['WeekofYear_cos'] = np.cos(2 * np.pi * test_df['WeekofYear'] / 52)
test_df['Week_sin'] = np.sin(2 * np.pi * test_df['Week'] / 7)
test_df['Week_cos'] = np.cos(2 * np.pi * test_df['Week'] / 7)

test_df = test_df.drop(remove_col, axis=1)
test_df = pd.get_dummies(test_df, columns=["country","store","product"])
test_df


best_model = grid_search.best_estimator_
y_pred = best_model.predict(test_df)


original_pred = np.exp(y_pred)
original_pred


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")
sample_submission.info()


sample_submission["num_sold"] = original_pred
sample_submission.to_csv("submission.csv", index=False)
sample_submission

