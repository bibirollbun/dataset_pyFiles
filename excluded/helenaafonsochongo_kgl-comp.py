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


# Loading the datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")


train.shape


test.shape


train.head()


# Importing the necessary libraries for data preprocessing
# 1. To encode categorical columns 
# 2. To convert boolean columns to 0 or 1
# 3. To keep numeric columns as they are
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline


# Features and target
# The columns id and accident_rsik must be dropped, since the id column is not a feature that
# will affect the risk of accident, and the accident_risk is our target, it will be stored separately so that 
# we can surely cross verify how accurate the model will be by predicting the values in this column
X = train.drop(columns=["id", "accident_risk"])
y = train["accident_risk"]
X_test = test.drop(columns=["id"])


# Identification of the types of the features 
categorical = ['road_type', 'lighting', 'weather', 'time_of_day']
boolean = ['road_signs_present', 'public_road', 'holiday', 'school_season']
numeric = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']


# Preprocessing
preprocessor = ColumnTransformer([
    ('categ', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('bool', 'passthrough', boolean),
    ('num', 'passthrough', numeric)
])


# Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators = 300, random_state = 42))
])


# Spliting the data for local validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)


# Training the model
model.fit(X_train, y_train)


# Validate the model
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Validation RMSE ", rmse)


# Submmission
submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


preds = model.predict(X_test)
submission["accident_risk"] = preds
submission.to_csv("submission.csv", index = False)
print(submission.head())




