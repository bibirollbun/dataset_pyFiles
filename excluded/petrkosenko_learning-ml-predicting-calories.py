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


import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from lightgbm.sklearn import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor


train_data = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv", index_col=0)
test_data = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv", index_col=0)


train_data


test_data


numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OrdinalEncoder())
])

numerical_cols = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
categorical_cols = ["Sex"]

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

X = train_data.drop(['Calories'], axis=1)
y = train_data["Calories"]

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
X_test = test_data


from sklearn.metrics import mean_squared_log_error
modelX = XGBRegressor(n_estimators=800, eta=0.2, min_split_loss=0.05, max_depth=8)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', modelX)])

e = 0.0037734769089671546
# Preprocessing of training data, fit model 
my_pipeline.fit(X_train_full, y_train)
preds = my_pipeline.predict(X_valid_full)
preds[preds < 0] = 0
print(mean_squared_log_error(preds, y_valid))


preds2 = pd.DataFrame({"Calories":  my_pipeline.predict(X_test).tolist()}, index=range(750000, 10**6))




preds2[preds2 < 0] = 0
preds2.to_csv('out.csv', index_label="id")




