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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Load datasets
competition_train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")  # Replace with actual file path
competition_test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")  # Replace with actual file path

# Display column names to identify the correct features and target
print("Competition Train Data Columns:", competition_train.columns.tolist())
print("Competition Test Data Columns:", competition_test.columns.tolist())

# Replace 'target_column' with the actual target column name
target_column = "id"  # Replace with actual target column name

# Explore competition dataset
print("\nCompetition Train Data:")
print(competition_train.head())
print(competition_train.info())
print(competition_train.describe())

# Explore test dataset
print("\nCompetition Test Data:")
print(competition_test.head())
print(competition_test.info())
print(competition_test.describe())

# Identify categorical columns in the training set
categorical_columns = competition_train.select_dtypes(include=['object']).columns.tolist()
print("Categorical Columns in Training Data:", categorical_columns)

# Add missing categorical columns to the test set with default values
for col in categorical_columns:
    if col not in competition_test.columns:
        competition_test[col] = "missing"  # Use a default value like 'missing'

# Encode categorical columns using one-hot encoding
competition_train = pd.get_dummies(competition_train, columns=categorical_columns, drop_first=True)
competition_test = pd.get_dummies(competition_test, columns=categorical_columns, drop_first=True)

# Ensure the test set has the same columns as the training set
missing_cols = set(competition_train.columns) - set(competition_test.columns)
for col in missing_cols:
    competition_test[col] = 0  # Add missing columns with default value 0
competition_test = competition_test[competition_train.columns]  # Reorder columns to match training data

# Handle missing values
# Check for missing values
print("\nMissing Values in Training Data:")
print(competition_train.isnull().sum())
print("\nMissing Values in Test Data:")
print(competition_test.isnull().sum())

# Impute missing values (replace NaN with the mean for numerical columns)
imputer = SimpleImputer(strategy='mean')  # Use 'median' or 'most_frequent' if needed
competition_train_imputed = pd.DataFrame(imputer.fit_transform(competition_train), columns=competition_train.columns)
competition_test_imputed = pd.DataFrame(imputer.transform(competition_test), columns=competition_test.columns)

# Split data into features and target
# Exclude 'id' and other non-feature columns from the feature set
X_train = competition_train_imputed.drop(columns=[target_column, 'id'])  # Features
y_train = competition_train_imputed[target_column]  # Target

# Train a model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Generate predictions for the competition test set
# Exclude 'id' and other non-feature columns from the test set
X_test = competition_test_imputed.drop(columns=[target_column, 'id'])  # Features
test_predictions = model.predict(X_test)

# Save predictions for submission
submission = pd.DataFrame({'id': competition_test['id'], 'predicted_price': test_predictions})
submission.to_csv("submission.csv", index=False)

