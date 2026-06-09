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


import pandas as pd
import numpy as np


df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


df.shape


df.head()


df.info()


df.corr()


df.isna().sum()


X = df.drop(columns='BeatsPerMinute')
y = df['BeatsPerMinute']


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.preprocessing import StandardScaler


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
neg_rmse_score = -rmse

print("RMSE:", rmse)
print("Negative RMSE Score:", neg_rmse_score)


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import  Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

models = {
    # "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01,random_state=42, max_iter=10000)
    # "Decision Tree": DecisionTreeRegressor(random_state=42),
    # "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    # "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}

for name, model in models.items():
    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = rmse
 
for model_name, score in results.items():
    print(f"{model_name}: Mean CV neg RMSE = {score:.4f}")


# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import Ridge, Lasso
# from sklearn.metrics import mean_squared_error

# ridge_params = {"model__alpha": [0.01, 0.1, 1, 10, 50, 100]}
# lasso_params = {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]}

# # Ridge pipeline
# ridge_pipe = Pipeline([
#     ("scaler", StandardScaler()),
#     ("model", Ridge(random_state=42))
# ])

# ridge_grid = GridSearchCV(
#     ridge_pipe, ridge_params,
#     cv=3, scoring="neg_root_mean_squared_error",
#     n_jobs=-1, verbose=2
# )
# ridge_grid.fit(X_train, y_train)

# # Lasso pipeline
# lasso_pipe = Pipeline([
#     ("scaler", StandardScaler()),
#     ("model", Lasso(random_state=42, max_iter=10000))
# ])

# lasso_grid = GridSearchCV(
#     lasso_pipe, lasso_params,
#     cv=3, scoring="neg_root_mean_squared_error",
#     n_jobs=-1, verbose=2
# )
# lasso_grid.fit(X_train, y_train)

# # Best results
# print("Best Ridge alpha:", ridge_grid.best_params_)
# print("Best Ridge RMSE:", -ridge_grid.best_score_)

# print("Best Lasso alpha:", lasso_grid.best_params_)
# print("Best Lasso RMSE:", -lasso_grid.best_score_)

# # Final test set evaluation
# best_ridge = ridge_grid.best_estimator_
# best_lasso = lasso_grid.best_estimator_

# ridge_rmse_test = mean_squared_error(y_test, best_ridge.predict(X_test), squared=False)
# lasso_rmse_test = mean_squared_error(y_test, best_lasso.predict(X_test), squared=False)

# print("Test RMSE Ridge:", ridge_rmse_test)
# print("Test RMSE Lasso:", lasso_rmse_test)


pipeline.fit(X,y)
ypred = pipeline.predict(test)


submission = pd.DataFrame({"id": test["id"], "BeatsPerMinute": ypred})
submission.to_csv("submission.csv", index=False)
submission.head()

