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


from sklearn.ensemble import RandomForestClassifier
import pandas as pd


# Load datasets
train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')


# Select numeric columns present in both train and test
numeric_cols = list(set(train.select_dtypes(include=['float64', 'int64']).columns) & set(test.columns))


# Handle missing values
train[numeric_cols] = train[numeric_cols].fillna(train[numeric_cols].median())
test[numeric_cols] = test[numeric_cols].fillna(test[numeric_cols].median())


# Encode categorical columns
from sklearn.preprocessing import LabelEncoder, StandardScaler
cat_cols = train.select_dtypes(include='object').columns
encoder = LabelEncoder()

for col in cat_cols:
    train[col] = encoder.fit_transform(train[col].astype(str))
    test[col] = encoder.transform(test[col].astype(str))


# Scale numeric features
scaler = StandardScaler()
train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
test[numeric_cols] = scaler.transform(test[numeric_cols])


# Define Features (X) and Target (y)
X = train.drop(['ID', 'efs', 'efs_time'], axis=1, errors='ignore')  # Features
y = train['efs']  # Target (Check if 'efs' is the correct target column)


# Model Training
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)


# Make Predictions
X_test = test.drop(['ID'], axis=1, errors='ignore')
predictions = rf_model.predict_proba(X_test)[:, 1]  # Probability of class 1


# Save Submission
submission = pd.DataFrame({'ID': test['ID'].astype(int), 'prediction': predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file saved successfully.")


import pandas as pd

# Load test and submission files
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
submission = pd.read_csv('submission.csv')

# Check if submission has the same number of rows as test.csv
if submission.shape[0] != test.shape[0]:
    print(f"Row count mismatch! Test: {test.shape[0]}, Submission: {submission.shape[0]}")

# Check column names
print("Submission columns:", submission.columns)
print("Expected columns: ['ID', 'prediction']")

# Check for missing values
print("Missing values in submission:\n", submission.isnull().sum())

# Check data types
print("Data types in submission:\n", submission.dtypes)

# Check if predictions are in expected range (0 to 1 for probabilities)
print("Prediction value range:\n", submission['prediction'].describe())



print(f"Test set rows: {test.shape[0]}")
print(f"Submission rows: {submission.shape[0]}")



submission['prediction'] = submission['prediction'].clip(0, 1)



# 1. Ensure submission has the correct number of rows
assert submission.shape[0] == test.shape[0], "Row count mismatch!"

# 2. Clip predictions between 0 and 1
submission['prediction'] = submission['prediction'].clip(0, 1)

# 3. Save the corrected submission file
submission.to_csv('submission.csv', index=False)
print("✅ Final submission file saved. Try submitting again!")


