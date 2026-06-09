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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



data = pd.read_csv('/kaggle/input/analyze-the-insights-over-mental-health-data/train.csv')


print("Dataset Shape:", data.shape)
print("\nColumn Info:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())

# Handle Missing Values
# Impute 'Profession' based on 'Working Professional or Student'
data['Profession'] = data['Profession'].fillna(data['Working Professional or Student'])

# Correctly identify numerical and categorical columns
numerical_cols = ['Age', 'Work/Study Hours', 'CGPA', 'Financial Stress']  # Strictly numeric columns
categorical_cols = ['Degree', 'Dietary Habits', 'Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Sleep Duration']

# Impute numeric columns with the median
data[numerical_cols] = data[numerical_cols].apply(pd.to_numeric, errors='coerce')  # Ensure numeric
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

# Impute 'Sleep Duration' as categorical
sleep_mapping = {
    'Less than 5 hours': 1,
    '5-6 hours': 2,
    '6-7 hours': 3,
    '7-8 hours': 4,
    'More than 8 hours': 5
}
data['Sleep Duration'] = data['Sleep Duration'].map(sleep_mapping)

# Fill remaining categorical columns with mode
for col in categorical_cols:
    if data[col].isnull().sum() > 0:  # Ensure the column has missing values
        data[col] = data[col].fillna(data[col].mode()[0])

# Feature Engineering
# Create a binary feature for missing values in 'Profession'
data['is_profession_missing'] = data['Profession'].isnull().astype(int)

# Bin ages into categories
data['Age_Group'] = pd.cut(data['Age'], bins=[0, 18, 30, 45, 60], labels=['Teen', 'Young Adult', 'Middle-Aged', 'Senior'])

# Drop redundant or less useful features (if needed)
data = data.drop(columns=['id'])  # Assuming 'id' column exists

# Prepare Data for Training
X = data.drop(columns=['Depression'])
y = data['Depression']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Encode Categorical Variables
categorical_features = ['Gender', 'City', 'Working Professional or Student', 'Profession', 'Degree', 
                        'Dietary Habits', 'Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Age_Group']
numerical_features = ['Age', 'Work/Study Hours', 'Sleep Duration', 'CGPA', 'Financial Stress']

# Define preprocessors for categorical and numerical features
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define Model Pipeline
model = LGBMClassifier(random_state=42)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])

# Hyperparameter Tuning
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [10, 20, 30],
    'classifier__num_leaves': [31, 50, 70]
}

grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluate the Best Model
best_model = grid_search.best_estimator_

y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

print("\nTraining Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
print("\nClassification Report (Test):\n", classification_report(y_test, y_pred_test))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_test))

# Generate Predictions for Submission
# Assuming test.csv exists
test_file_path = '/kaggle/input/analyze-the-insights-over-mental-health-data/test.csv'
test_data = pd.read_csv(test_file_path)
test_data['Profession'] = test_data['Profession'].fillna(test_data['Working Professional or Student'])
test_data[numerical_cols] = test_data[numerical_cols].apply(pd.to_numeric, errors='coerce')
test_data[numerical_cols] = test_data[numerical_cols].fillna(test_data[numerical_cols].median())
test_data['Sleep Duration'] = test_data['Sleep Duration'].map(sleep_mapping)
for col in categorical_cols:
    if col in test_data.columns:
        test_data[col] = test_data[col].fillna(test_data[col].mode()[0])
test_data['is_profession_missing'] = test_data['Profession'].isnull().astype(int)
test_data['Age_Group'] = pd.cut(test_data['Age'], bins=[0, 18, 30, 45, 60], labels=['Teen', 'Young Adult', 'Middle-Aged', 'Senior'])

# Drop redundant columns
if 'id' in test_data.columns:
    ids = test_data['id']
    test_data = test_data.drop(columns=['id'])

predictions = best_model.predict(test_data)

# Create submission file
submission = pd.DataFrame({'id': ids, 'Depression': predictions})
submission.to_csv('submission.csv', index=False)
print("\nSubmission file saved as 'submission.csv'")


