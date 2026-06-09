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


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score



# ======= STEP 1: Load Dataset =======
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')




# ======= STEP 2: Dataset Exploration (Required for Full Points) =======
print("Training Data Overview:")
print(train_data.head())  # Show first few rows

print("\nSummary Statistics:")
print(train_data.describe())  # Summary of numerical data

print("\nMissing Values:")
print(train_data.isnull().sum())  # Check for missing values



# ======= STEP 3: Identify Feature Types =======
numerical_features = list(set(train_data.select_dtypes(include=['int64', 'float64']).columns) &
                           set(test_data.select_dtypes(include=['int64', 'float64']).columns))

categorical_features = list(set(train_data.select_dtypes(include=['object']).columns) &
                            set(test_data.select_dtypes(include=['object']).columns))

# Identify Target Column
target_column = None
for col in train_data.columns:
    if "target" in col.lower():  # Handle variations like 'Target' or 'TARGET'
        target_column = col
        break

if target_column is None:
    raise ValueError("No 'target' column found in train_data. Please check column names.")

# Ensure target is not in numerical features
if target_column in numerical_features:
    numerical_features.remove(target_column)



# Remove target variable if it exists
if 'target' in numerical_features:
    numerical_features.remove('target')


# ======= DEBUG: Print column names to check if 'target' exists =======
print("Columns in train_data:", train_data.columns.tolist())

# Ensure 'target' column exists
target_column = None
for col in train_data.columns:
    if "target" in col.lower():  # Handle variations like 'Target' or 'TARGET'
        target_column = col
        break

if target_column is None:
    raise ValueError("No 'target' column found in train_data. Please check column names.")




# Identify feature types in both train and test
numerical_features = list(set(train_data.select_dtypes(include=['int64', 'float64']).columns) &
                           set(test_data.select_dtypes(include=['int64', 'float64']).columns))

categorical_features = list(set(train_data.select_dtypes(include=['object']).columns) &
                            set(test_data.select_dtypes(include=['object']).columns))



# ======= STEP 4: Feature Engineering=======
# Feature 1: Feature_Range - Measures the range (max - min) of numerical features for each row.
# Why? This helps detect variability in feature values, which might be useful for classification.
train_data["Feature_Range"] = train_data[numerical_features].max(axis=1) - train_data[numerical_features].min(axis=1)
test_data["Feature_Range"] = test_data[numerical_features].max(axis=1) - test_data[numerical_features].min(axis=1)

# Feature 2: Feature_Variance - Measures the variance of numerical features for each row.
# Why? Variance captures how much the features fluctuate, which might be important for identifying different classes.
train_data["Feature_Variance"] = train_data[numerical_features].var(axis=1)
test_data["Feature_Variance"] = test_data[numerical_features].var(axis=1)

# Add new features to numerical_features list
numerical_features.extend(["Feature_Range", "Feature_Variance"])



# Remove target variable if it exists in numerical features
if target_column in numerical_features:
    numerical_features.remove(target_column)



# ======= STEP 5: Preprocessing Pipeline =======
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_features),
    ('cat', cat_pipeline, categorical_features)
])



# ======= STEP 6: Select Model (Random Forest or Decision Tree) =======
model = RandomForestClassifier(n_estimators=100, random_state=42)
# To use Decision Tree instead, replace above with:
# model = DecisionTreeClassifier(random_state=42)




# ======= STEP 7: Create Full Pipeline =======
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', model)
])


# ======= STEP 8: Splitting Data =======
X = train_data.drop(columns=[target_column])
y = train_data[target_column]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ======= STEP 9: Train the Model =======
pipeline.fit(X_train, y_train)

# Evaluate model
y_pred = pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"\nValidation Accuracy: {accuracy:.4f}")

# ======= STEP 10: Make Predictions on Test Data =======
test_predictions = pipeline.predict(test_data)

# ======= STEP 11: Prepare Submission File =======
submission = pd.DataFrame({'id': test_data.index, 'target': test_predictions})
submission.to_csv('submission.csv', index=False)
print("\nSubmission file saved as submission.csv")


