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


# Importing Libraries to play with the data:)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Libraries for developing the Model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# To remove warnings
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline


data_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
data_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


data_train.head(10)


data_test.head(10)


data_train.info()


data_test.info()


print("Training data: ", data_train.shape)
print("Testing data: ", data_test.shape)


data_train.describe()


data_test.describe()


plt.figure(figsize=(10, 4))
sns.countplot(x = "weather", hue = "num_reported_accidents", data = data_train);


plt.figure(figsize=(10, 4))
sns.countplot(x = "school_season", hue = "num_reported_accidents", data = data_train);


plt.figure(figsize=(12, 8))
sns.heatmap(data_train.select_dtypes(include=['number']).corr(), annot=True)


data_train["curvature_accident_score"] = data_train["curvature"] * data_train["num_reported_accidents"]
data_train["speed_accident_score"] = data_train["speed_limit"] * data_train["num_reported_accidents"]
data_train["speed_accident_score"] = np.log1p(data_train["speed_accident_score"])

data_test["curvature_accident_score"] = data_test["curvature"] * data_test["num_reported_accidents"]
data_test["speed_accident_score"] = data_test["speed_limit"] * data_test["num_reported_accidents"]
data_test["speed_accident_score"] = np.log1p(data_test["speed_accident_score"])


data_train.info()


data_train.describe()


plt.figure(figsize=(12, 8))
sns.heatmap(data_train.select_dtypes(include=['number']).corr(), annot=True)


x_train = data_train.drop(["accident_risk"], axis = 1)
y_train = data_train["accident_risk"]
x_train = pd.get_dummies(x_train, drop_first=True)
x_test = pd.get_dummies(data_test, drop_first=True)


# Dropping id field - Unique for every record, making difficult for the model to build relationship
x_train = x_train.drop(columns=["id"])
x_test = x_test.drop(columns=["id"])


x_train.head()


x_train.info()


import xgboost as xgb
from xgboost import XGBRegressor


# Base model
xgb_base_model = XGBRegressor(objective='reg:squarederror', random_state=42)


param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [8, 9],
    'learning_rate': [0.05],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 0.9],
    'min_child_weight': [4, 5, 6],  # controls overfitting
    'reg_lambda': [1.5, 2],      # L2 regularization
    'reg_alpha': [0, 0.1]      # L1 regularization
}


# GridSearchCV setup
grid_search = GridSearchCV(
    estimator=xgb_base_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # use 'neg_mean_squared_error' for MSE optimization
    cv=3,                  # 3-fold cross validation
    verbose=1,
    n_jobs=-1              # use all CPU cores
)


grid_search.fit(x_train, y_train)


print("Best Parameters:", grid_search.best_params_)
print("Best MSE Score (CV):", round(grid_search.best_score_, 3))


# Using Best parameter
xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    min_child_weight = 6,
    colsample_bytree=0.9,
    reg_lambda = 2,
    reg_alpha = 0.1,
    random_state=42
)


xgb_model.fit(x_train, y_train)


y_train_pred = xgb_model.predict(x_train)
y_test_pred = xgb_model.predict(x_test)


print("RMSE score for Training data: ", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("R2 score for Training data: ", r2_score(y_train, y_train_pred))


res = pd.DataFrame({"id": data_test["id"], "accident_risk": y_test_pred.round(3)})
res.to_csv("submission.csv", index = False)

