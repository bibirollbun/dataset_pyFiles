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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Step 1: Load the data
train_data = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')  # Training dataset
test_data = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')    # Test dataset

# Step 2: Prepare the data

# Separate numeric and categorical features
numeric_features = train_data.drop(columns=['id', 'loan_status']).select_dtypes(include=['float64', 'int64']).columns
categorical_features = train_data.drop(columns=['id', 'loan_status']).select_dtypes(include=['object']).columns

# Define preprocessing for numeric and categorical features
numeric_transformer = SimpleImputer(strategy='mean')  # Handle missing values for numeric features
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)  # Encode categorical features

# Combine preprocessing into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Preprocess training data
X = train_data.drop(columns=['id', 'loan_status'])
y = train_data['loan_status']

X = preprocessor.fit_transform(X)  # Apply preprocessing pipeline

# Preprocess test data
X_test = test_data.drop(columns=['id'])
X_test = preprocessor.transform(X_test)  # Apply the same preprocessing to test data

# Convert preprocessed data back to DataFrame for compatibility
# NOTE: OneHotEncoder adds new columns for categorical features—names need reformatting for clarity
encoded_columns = numeric_features.tolist() + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
X = pd.DataFrame(X, columns=encoded_columns)
X_test = pd.DataFrame(X_test, columns=encoded_columns)

# Step 3: Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)  # Adjust parameters for efficiency
model.fit(X_train, y_train)

# Step 5: Evaluate the model (optional step to check performance)
val_proba = model.predict_proba(X_val)[:, 1]  # Probability predictions for validation set
auc = roc_auc_score(y_val, val_proba)
print(f"Validation AUC: {auc:.2f}")

# Step 6: Predict on test set
test_proba = model.predict_proba(X_test)[:, 1]  # Probability predictions for test set

# Step 7: Output predictions in required format
submission = pd.DataFrame({
    'id': test_data['id'],          # Ensure the column name matches 'id'
    'loan_status': test_proba       # Predicted probabilities
})
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")




