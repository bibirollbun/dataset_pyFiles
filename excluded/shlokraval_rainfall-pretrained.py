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

# Load datasets
train_path = "/kaggle/input/playground-series-s5e3/train.csv"
test_path = "/kaggle/input/playground-series-s5e3/test.csv"
submission_path = "/kaggle/input/playground-series-s5e3/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Display basic info
train_df.info(), test_df.info(), train_df.head(), test_df.head()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Feature Engineering: Add temperature difference
train_df["temp_diff"] = train_df["maxtemp"] - train_df["mintemp"]
test_df["temp_diff"] = test_df["maxtemp"] - test_df["mintemp"]

# Define features and target
X = train_df.drop(columns=["id", "rainfall"])  # Drop ID and target
y = train_df["rainfall"]
X_test = test_df.drop(columns=["id"])

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Model Initialization
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42),
    "LightGBM": lgb.LGBMClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42),
    "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
}

# Train & Evaluate
best_model = None
best_accuracy = 0
model_accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    model_accuracies[name] = acc
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

model_accuracies, best_model



# Remove LightGBM and re-run models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42),
    "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
}

# Train & Evaluate
best_model = None
best_accuracy = 0
model_accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    model_accuracies[name] = acc
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

model_accuracies, best_model



# Reload the dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

# Check column names
train_df.columns, test_df.columns



from sklearn.impute import SimpleImputer
# Feature Engineering: Add temperature difference
train_df["temp_diff"] = train_df["maxtemp"] - train_df["mintemp"]
test_df["temp_diff"] = test_df["maxtemp"] - test_df["mintemp"]

# Define features and target
X = train_df.drop(columns=["id", "rainfall"])  # Drop ID and target
y = train_df["rainfall"]
X_test = test_df.drop(columns=["id"])

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)
X_test = imputer.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Generate predictions for test set
test_predictions = rf_model.predict(X_test)

# Save predictions to CSV
submission_df = pd.DataFrame({"id": test_df["id"], "rainfall": test_predictions})
submission_path = "rainfall4.csv"
submission_df.to_csv(submission_path, index=False)

submission_path





