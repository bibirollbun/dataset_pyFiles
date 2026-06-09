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
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("/kaggle/input/playground-series-s3e25/train.csv")
df


df.info()


df.describe()


plt.figure(figsize = (15, 15))
sns.heatmap(df.corr(), annot = True, cmap = "crest")


sns.scatterplot(x = "density_Average", y = "Hardness", data = df)


sns.boxplot(df["Hardness"])


sns.distplot(df["Hardness"])


df.isna().sum() 
# df.dropna(axis = 0) : to drop rows having null values
# df["column_name"].fillna(df["column_name"].mean()) : to fill null values of a specific column with the mean value for that column


df = df.drop(columns = ["id"])


X = df.drop(columns = "Hardness")
y = df["Hardness"]


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


lr_pred = lr_model.predict(X_val)


mean_squared_error(y_val, lr_pred)


r2_score(y_val, lr_pred)


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf_model.predict(X_val)
print("Random Forest Regressor:")
print(f"RMSE: {mean_squared_error(y_val, y_pred_rf, squared=False):.4f}")
print(f"R2 Score: {r2_score(y_val, y_pred_rf):.4f}")


# XGBoost Regressor
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_xgb = xgb_model.predict(X_val)
print("XGBoost Regressor:")
print(f"RMSE: {mean_squared_error(y_val, y_pred_xgb, squared=False):.4f}")
print(f"R2 Score: {r2_score(y_val, y_pred_xgb):.4f}")



# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_lr = lr_model.predict(X_val)
print("Linear Regression:")
print(f"RMSE: {mean_squared_error(y_val, y_pred_lr, squared=False):.4f}")
print(f"R2 Score: {r2_score(y_val, y_pred_lr):.4f}")



from sklearn.model_selection import GridSearchCV


# Example for Random Forest Hyperparameter Tuning
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30]}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)


print("Best Hyperparameters for Random Forest:", grid_search.best_params_)


# Predict and evaluate
#y_pred_grid_search = grid_search.predict(X_val)
#print("Random Forest Grid Search:")
#print(f"RMSE: {mean_squared_error(y_val, y_pred_grid_search, squared=False):.4f}")
#print(f"R2 Score: {r2_score(y_val, y_pred_grid_search):.4f}")


# Define the parameter grid
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

# GridSearchCV
xgb_grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=xgb_param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
)


xgb_grid_search.fit(X_train, y_train)


# Best parameters and score
print("Best Hyperparameters for XGBoost:", xgb_grid_search.best_params_)
print("Best CV RMSE:", (-xgb_grid_search.best_score_)**0.5)


# Predict and evaluate
y_pred_xgb_grid_search = xgb_grid_search.predict(X_val)
print("XGB Grid Search:")
print(f"RMSE: {mean_squared_error(y_val, y_pred_xgb_grid_search, squared=False):.4f}")
print(f"R2 Score: {r2_score(y_val, y_pred_xgb_grid_search):.4f}")


from sklearn.ensemble import StackingRegressor
stack = StackingRegressor(estimators=[
    ('rf', rf_model), ('xgb', xgb_model), ('lr', lr_model)
], final_estimator=LinearRegression())

stack.fit(X_train, y_train)


# Predict and evaluate
y_pred_stack = stack.predict(X_val)
print("Stack:")
print(f"RMSE: {mean_squared_error(y_val, y_pred_stack, squared=False):.4f}")
print(f"R2 Score: {r2_score(y_val, y_pred_stack):.4f}")




