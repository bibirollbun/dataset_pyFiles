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

# Load training and test data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Check the first few rows of the dataset
print(train_df.head())
print(test_df.head())


print(train_df.isnull().sum())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression  # Example regression model
from sklearn.pipeline import Pipeline
import pandas as pd

# Check if 'id' column exists before dropping it
if 'id' in train_df.columns:
    train_df.drop(columns=['id'], inplace=True)
if 'id' in test_df.columns:
    test_ids = test_df['id']  # Save test IDs for submission
    test_df.drop(columns=['id'], inplace=True)

# Identify categorical and numerical features
categorical_features = ['winddirection']  # If categorical
numerical_features = train_df.drop(columns=['rainfall'] + categorical_features).columns

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Split dataset into features (X) and target (y)
X = train_df.drop(columns=['rainfall'])
y = train_df['rainfall']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that first preprocesses the data and then trains a model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())  # Example: Linear Regression model
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = model_pipeline.predict(X_val)

# Print validation predictions and evaluate model performance (optional)
print("Validation Predictions:", y_val_pred)

# Prepare the test set (features) for prediction
X_test = test_df  # Test set without target variable

# Make predictions on the test set
test_predictions = model_pipeline.predict(X_test)

# Prepare the submission file (test_predictions with the corresponding 'id' from the test set)
submission = pd.DataFrame({
    'id': test_ids,
    'rainfall': test_predictions
})

# Save the submission file as a CSV
submission.to_csv('submission.csv', index=False)

print("Submission file saved as 'submission.csv'")


