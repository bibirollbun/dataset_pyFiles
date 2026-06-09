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
import re
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from datetime import datetime


# Load dataset
train_df = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv")
test_df = pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv")


print(train_df.head())  # Display first 5 rows
print(train_df.columns)  # Display column names


print("Train Columns:", train_df.columns)
print("Test Columns:", test_df.columns)


# Feature Engineering
def extract_features(df):
    if "plate" in df.columns:
        df["plate_length"] = df["plate"].apply(len)
        df["has_repeating_digits"] = df["plate"].apply(lambda x: int(len(set(x)) < len(x)))
        
        def extract_numeric_region(plate):
            region_part = plate[-3:]
            numbers = re.findall(r'\d+', region_part)
            return int(numbers[0]) if numbers else 0

        df["region_code"] = df["plate"].apply(extract_numeric_region)
        df["letters_only"] = df["plate"].str.replace(r'\d+', '', regex=True)
        
        le = LabelEncoder()
        df["letters_encoded"] = le.fit_transform(df["letters_only"])
        df.drop(columns=["plate", "letters_only"], inplace=True)
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        df["days_since_start"] = (df["date"] - df["date"].min()).dt.days
        df.drop(columns=["date"], inplace=True)
    
    return df

train_df = extract_features(train_df)
test_df = extract_features(test_df)



# Separate Features & Target
if "price" not in train_df.columns:
    raise KeyError("Column 'price' not found in training dataset!")
X = train_df.drop(columns=["price"])
y = train_df["price"]
X_test = test_df.drop(columns=["price"], errors='ignore')


# Standardize Data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


# Split Data for Hyperparameter Tuning
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# XGBoost Hyperparameter Tuning
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(xgb_model, param_grid, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)


# Best Model Training
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)


# Save Predictions
# Save Predictions with formatted prices
submission = pd.DataFrame({"id": test_df["id"], "price": y_pred.round().astype(int)})
submission.to_csv("submission.csv", index=False)
print("Best Parameters:", grid_search.best_params_)
print("Submission file saved as submission.csv")


submission

