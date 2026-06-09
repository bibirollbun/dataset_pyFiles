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

# Load dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s4e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s4e6/test.csv")

# Check available columns
print(train_df.columns)



# Check for missing values in the entire dataset
print("Missing values per column:\n", train_df.isnull().sum())

# Summary statistics for numerical features
print("Numerical feature summary:\n", train_df.describe())

# Checking unique categories in categorical features
print("Unique values per categorical feature:")
for col in categorical_features:
    print(f"{col}: {train_df[col].unique()[:10]}")



# Avoid division by zero by adding 1 to denominator
train_df["Engagement_Score"] = (
    train_df["Curricular units 1st sem (evaluations)"] + train_df["Curricular units 2nd sem (evaluations)"] +
    train_df["Curricular units 1st sem (approved)"] + train_df["Curricular units 2nd sem (approved)"]
) / (train_df["Curricular units 1st sem (enrolled)"] + train_df["Curricular units 2nd sem (enrolled)"] + 1)

train_df["Dropout_Risk"] = (
    train_df["Curricular units 1st sem (without evaluations)"] + train_df["Curricular units 2nd sem (without evaluations)"]
) / (train_df["Curricular units 1st sem (enrolled)"] + train_df["Curricular units 2nd sem (enrolled)"] + 1)

# Apply same transformation to test dataset
test_df["Engagement_Score"] = (
    test_df["Curricular units 1st sem (evaluations)"] + test_df["Curricular units 2nd sem (evaluations)"] +
    test_df["Curricular units 1st sem (approved)"] + test_df["Curricular units 2nd sem (approved)"]
) / (test_df["Curricular units 1st sem (enrolled)"] + test_df["Curricular units 2nd sem (enrolled)"] + 1)

test_df["Dropout_Risk"] = (
    test_df["Curricular units 1st sem (without evaluations)"] + test_df["Curricular units 2nd sem (without evaluations)"]
) / (test_df["Curricular units 1st sem (enrolled)"] + test_df["Curricular units 2nd sem (enrolled)"] + 1)



# Verify the new features
print(train_df[["Engagement_Score", "Dropout_Risk"]].describe())

# Quick check for missing values
print(train_df[["Engagement_Score", "Dropout_Risk"]].isnull().sum())



# Define categorical and numerical columns
categorical_features = ["Marital status", "Application mode", "Course", "Daytime/evening attendance",
                        "Previous qualification", "Nacionality", "Mother's qualification", "Father's qualification",
                        "Mother's occupation", "Father's occupation", "Gender", "Scholarship holder", "International"]

numerical_features = ["Application order", "Previous qualification (grade)", "Admission grade",
                      "Age at enrollment", "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
                      "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
                      "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
                      "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
                      "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
                      "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)", 
                      "Unemployment rate", "Inflation rate", "GDP", 
                      "Engagement_Score", "Dropout_Risk"]  # Include new features



from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Handle missing values & scale numbers
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  # Fill missing values with median
    ("scaler", StandardScaler())  # Standardize features
])

# Fill missing values & encode categories
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing categorical values
    ("encoder", OneHotEncoder(handle_unknown="ignore"))  # One-hot encode categorical variables
])

# Full ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, numerical_features),
    ("cat", cat_transformer, categorical_features)
])

print("âœ… Preprocessing Pipeline Ready!")



# Define target variable
target_column = "Target"

# Drop unnecessary columns (ID and Target for training)
X = train_df.drop(columns=["id", target_column])
y = train_df[target_column]

# Drop ID column from test set
X_test = test_df.drop(columns=["id"])

print("âœ… Data is preprocessed and ready for modeling!")



from sklearn.model_selection import train_test_split

# Split into training (80%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"âœ… Training size: {X_train.shape}, Validation size: {X_val.shape}")



from sklearn.ensemble import RandomForestClassifier

# I am using fewer trees to speed up training, n_estimators=200 was too slow
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)

# Run training
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
])

pipeline.fit(X_train, y_train)

print("âœ… Model training complete!")



from sklearn.metrics import accuracy_score

# Make predictions on validation data
y_pred = pipeline.predict(X_val)

# Check accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"ğŸ”� Validation Accuracy: {accuracy:.4f}")



# Generate predictions on the test dataset
test_predictions = pipeline.predict(X_test)


# Check first 10 predictions
print(test_predictions[:10])

# Check unique prediction categories
print(set(test_predictions))



# Create a DataFrame for submission
submission = pd.DataFrame({"id": test_df["id"], "Target": test_predictions})

# Save to CSV (required format)
submission.to_csv("submission.csv", index=False)

print("âœ… Submission file saved as 'submission.csv'")


