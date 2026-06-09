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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Load the sample submission data
data = pd.read_csv('/kaggle/input/playground-series-s4e6/sample_submission.csv')

# Display the first few rows of the data
print(data.head())


# Create new features
# Feature 1: Graduate Status (binary)
data['Graduate_Status'] = data['Target'].apply(lambda x: 1 if x == 'Graduate' else 0)

# Feature 2: Graduate Status Count (count of graduates)
data['Graduate_Status_Count'] = data['Graduate_Status'].cumsum()

# Display the updated DataFrame
print(data.head())


# Define features and target variable
X = data[['Graduate_Status', 'Graduate_Status_Count']]  # Features
y = data['Graduate_Status']  # Target variable


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Graduate_Status_Count']),  # Scale numerical features
        ('cat', OneHotEncoder(), ['Graduate_Status'])  # One-hot encode categorical features
    ])

# Create a pipeline that first preprocesses the data and then fits the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])


# Fit the model on the training data
pipeline.fit(X_train, y_train)


# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# Prepare the submission DataFrame
submission = pd.DataFrame({
    'id': data['id'],  # Assuming 'id' is present in the original data
    'Target': pipeline.predict(X)  # Predict on the original data
})

# Save the submission to a CSV file
submission.to_csv('submission.csv', index=False)

