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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
test.head()


train.describe()


train.info()


# Store test IDs before dropping the column
test_ids = test["id"]


# Drop unnecessary columns
train.drop(columns=["id"], inplace=True)
test.drop(columns=["id"], inplace=True)


# Check for missing values
train.fillna(train.median(), inplace=True)
test.fillna(test.median(), inplace=True)


# Feature Engineering: Creating new meaningful features
train["temp_range"] = train["maxtemp"] - train["mintemp"]
train["humidity_ratio"] = train["dewpoint"] / train["humidity"]
train["cloud_sun_ratio"] = train["cloud"] / (train["sunshine"] + 0.1)  # Avoid division by zero


test["temp_range"] = test["maxtemp"] - test["mintemp"]
test["humidity_ratio"] = test["dewpoint"] / test["humidity"]
test["cloud_sun_ratio"] = test["cloud"] / (test["sunshine"] + 0.1)


# Define input features and target variable
X = train.drop(columns=["rainfall"])
y = train["rainfall"]


# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)


# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


# Define XGBoost model
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic", 
    eval_metric="auc",
    use_label_encoder=False
)


# Hyperparameter tuning
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0],
}


grid_search = GridSearchCV(xgb_model, param_grid, scoring="roc_auc", cv=3, verbose=2)
grid_search.fit(X_train, y_train)


# Best model after tuning
best_xgb = grid_search.best_estimator_


# Validate the model
y_valid_pred = best_xgb.predict_proba(X_valid)[:, 1]
auc_score = roc_auc_score(y_valid, y_valid_pred)
print(f"\nValidation AUC-ROC Score: {auc_score:.4f}")


# Make predictions on test set
test_predictions = best_xgb.predict_proba(test_scaled)[:, 1]


# Prepare submission file
submission = pd.DataFrame({"id": test_ids, "rainfall": test_predictions})
submission.to_csv("submission.csv", index=False)

print("\n✅ Submission file created: submission.csv")

