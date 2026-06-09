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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error



train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

print(train.head())
print(test.head())



from sklearn.preprocessing import OneHotEncoder

# Fill missing categorical values with "Unknown" and numerical values with median
categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
numerical_features = ["Weight Capacity (kg)"]

# Fill categorical missing values
for col in categorical_features:
    train[col] = train[col].fillna("Unknown")
    test[col] = test[col].fillna("Unknown")

# Fill numerical missing values with median
for col in numerical_features:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(test[col].median())

# One-Hot Encoding for categorical variables
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
train_encoded = pd.DataFrame(encoder.fit_transform(train[categorical_features]))
test_encoded = pd.DataFrame(encoder.transform(test[categorical_features]))

# Ensure encoded column names match
train_encoded.columns = encoder.get_feature_names_out(categorical_features)
test_encoded.columns = encoder.get_feature_names_out(categorical_features)

# Reset index after encoding
train_encoded.index = train.index
test_encoded.index = test.index

# Drop original categorical columns
train = train.drop(columns=categorical_features)
test = test.drop(columns=categorical_features)

# Concatenate the encoded categorical data
train = pd.concat([train, train_encoded], axis=1)
test = pd.concat([test, test_encoded], axis=1)

# Feature Scaling (Standardization)
scaler = StandardScaler()
train["Weight Capacity (kg)"] = scaler.fit_transform(train[["Weight Capacity (kg)"]])
test["Weight Capacity (kg)"] = scaler.transform(test[["Weight Capacity (kg)"]])

print("Preprocessing Complete!")



X = train.drop(columns=["id", "Price"])
y = train["Price"]
X_test = test.drop(columns=["id"])



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42),
    "CatBoost": CatBoostRegressor(iterations=100, verbose=0)
}

# Evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"{name} RMSE: {rmse:.4f}")



from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    "num_leaves": [31, 50, 100],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 500, 1000]
}

lgbm = LGBMRegressor()
grid_search = GridSearchCV(lgbm, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)



best_model = LGBMRegressor(n_estimators=500, learning_rate=0.01, num_leaves=31, random_state=42)
best_model.fit(X, y)
final_preds = best_model.predict(X_test)



submission = pd.DataFrame({"id": test["id"], "Price": final_preds})
submission.to_csv("submission.csv", index=False)





