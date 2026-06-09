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


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


train_data = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/train.csv')
test_data = pd.read_csv('/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/test.csv')


# Verify the first few rows of the data to ensure correct loading
print(train_data.head())
print(test_data.head())


# Check column names to ensure correct processing
print("Columns in training data:", train_data.columns)
print("Columns in test data:", test_data.columns)


# Example: Checking for missing values
print("Missing values in training data:\n", train_data.isnull().sum())
print("Missing values in test data:\n", test_data.isnull().sum())


# Dynamically determine feature columns
if 'id' in train_data.columns:
    X = train_data.drop(columns=["id", "target"])
else:
    X = train_data.drop(columns=["target"])

# Ensure the target column is correct
if 'target' in train_data.columns:
    y = train_data["target"]
else:
    raise KeyError("Target column not found in training data.")

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# Define Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train_scaled, y_train)




# Make predictions on the validation set
y_val_pred = model.predict(X_val_scaled)

# Evaluate the model using R²
r2 = r2_score(y_val, y_val_pred)
print(f"Validation R²: {r2:.4f}")


# Prepare the test data
if 'id' in test_data.columns:
    X_test = test_data.drop(columns=["id"])
    test_ids = test_data['id']
else:
    X_test = test_data
    test_ids = pd.Series(range(len(test_data)))  # Create dummy IDs if not present

X_test_scaled = scaler.transform(X_test)

# Make predictions on the test data
test_predictions = model.predict(X_test_scaled)
# Save predictions to a CSV file in the required format
submission = pd.DataFrame({'id': test_ids, 'target': test_predictions})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Predictions saved to submission.csv")

