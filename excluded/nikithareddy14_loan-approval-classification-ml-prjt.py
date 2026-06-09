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


import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)
    for filename in filenames:
        print("  -", filename)



# === LOAN DATA ANALYSIS (Train & Test only) ===

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("âœ… Starting Loan Data Analysis using TRAIN and TEST files...\n")

# === STEP 1: Locate CSV files ===
train_path, test_path = None, None

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if "train" in filename and filename.endswith('.csv'):
            train_path = os.path.join(dirname, filename)
        elif "test" in filename and filename.endswith('.csv'):
            test_path = os.path.join(dirname, filename)

if not train_path or not test_path:
    raise FileNotFoundError("â�Œ train.csv or test.csv not found!")

print(f"ğŸ“‚ Train file: {train_path}")
print(f"ğŸ“‚ Test file: {test_path}\n")

# === STEP 2: Load datasets ===
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("âœ… Files Loaded Successfully!\n")

# === STEP 3: Basic Info ===
print("ğŸ“Š TRAIN DATA OVERVIEW ===")
print("Shape:", train_df.shape)
print("Columns:", train_df.columns.tolist())
print("\nPreview:\n", train_df.head())

print("\nğŸ“Š TEST DATA OVERVIEW ===")
print("Shape:", test_df.shape)
print("Columns:", test_df.columns.tolist())
print("\nPreview:\n", test_df.head())

# === STEP 4: Missing Values ===
print("\nğŸ§© Missing Values in Train:\n", train_df.isnull().sum())
print("\nğŸ§© Missing Values in Test:\n", test_df.isnull().sum())

# === STEP 5: Target Analysis (loan_status) ===
if "loan_status" in train_df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x="loan_status", data=train_df, palette="pastel", edgecolor="black")
    plt.title("Loan Status Distribution in Train Data")
    plt.show()
else:
    print("\nâš ï¸� 'loan_status' not found in train dataset.")

# === STEP 6: Numeric Feature Visualization ===
numeric_cols = train_df.select_dtypes(include=['int64','float64']).columns
if len(numeric_cols) > 1:
    train_df[numeric_cols].hist(bins=20, figsize=(12, 8), color='skyblue', edgecolor='black')
    plt.suptitle("Numeric Feature Distributions", fontsize=14)
    plt.show()



# === LOAN DATA ANALYSIS (Train & Test only) ===

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("âœ… Starting Loan Data Analysis using TRAIN and TEST files...\n")

# === STEP 1: Locate CSV files ===
train_path, test_path = None, None

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if "train" in filename.lower() and filename.endswith('.csv'):
            train_path = os.path.join(dirname, filename)
        elif "test" in filename.lower() and filename.endswith('.csv'):
            test_path = os.path.join(dirname, filename)

if not train_path or not test_path:
    raise FileNotFoundError("â�Œ train.csv or test.csv not found!")

print(f"ğŸ“‚ Train file: {train_path}")
print(f"ğŸ“‚ Test file: {test_path}\n")

# === STEP 2: Load datasets ===
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("âœ… Files Loaded Successfully!\n")

# === STEP 3: Basic Info ===
print("ğŸ“Š TRAIN DATA OVERVIEW ===")
print("Shape:", train_df.shape)
print("Columns:", train_df.columns.tolist())
print("\nPreview:\n", train_df.head())

print("\nğŸ“Š TEST DATA OVERVIEW ===")
print("Shape:", test_df.shape)
print("Columns:", test_df.columns.tolist())
print("\nPreview:\n", test_df.head())

# === STEP 4: Missing Values ===
print("\nğŸ§© Missing Values in Train:\n", train_df.isnull().sum())
print("\nğŸ§© Missing Values in Test:\n", test_df.isnull().sum())

# === STEP 5: Target Analysis (loan_status) ===
target_candidates = [c for c in train_df.columns if 'status' in c.lower() or 'target' in c.lower() or 'approval' in c.lower()]
if target_candidates:
    target_col = target_candidates[0]
    print(f"\nğŸ�¯ Target column detected: '{target_col}'\n")
    plt.figure(figsize=(6,4))
    sns.countplot(x=target_col, data=train_df, palette="pastel", edgecolor="black")
    plt.title("Loan Status Distribution in Train Data")
    plt.show()
else:
    raise KeyError("â�Œ Target column not found. Please check your dataset.")

# === STEP 6: Numeric Feature Visualization ===
numeric_cols = train_df.select_dtypes(include=['int64','float64']).columns
if len(numeric_cols) > 1:
    train_df[numeric_cols].hist(bins=20, figsize=(12, 8), color='skyblue', edgecolor='black')
    plt.suptitle("Numeric Feature Distributions", fontsize=14)
    plt.show()

# === STEP 7: Data Preprocessing ===
train = train_df.copy()
test = test_df.copy()

# Encode categorical features
label_enc = LabelEncoder()
for col in train.select_dtypes(include=['object']).columns:
    train[col] = label_enc.fit_transform(train[col].astype(str))
    if col in test.columns:
        test[col] = label_enc.transform(test[col].astype(str))

# Fill missing values
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# === STEP 8: Split Features and Target ===
X = train.drop(columns=[target_col])
y = train[target_col]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === STEP 9: Train Model ===
print("\nğŸ§  Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# === STEP 10: Evaluate Accuracy ===
val_preds = model.predict(X_val)
accuracy = accuracy_score(y_val, val_preds)
print(f"âœ… Model Training Completed! Validation Accuracy: {accuracy * 100:.2f}%\n")

# === STEP 11: Predictions on Test Data ===
test_preds = model.predict(test)

# === STEP 12: Create Submission File ===
id_col = None
for col in test.columns:
    if "id" in col.lower() or "loan" in col.lower():
        id_col = col
        break

if id_col:
    submission = pd.DataFrame({id_col: test[id_col], target_col: test_preds})
else:
    submission = pd.DataFrame({'ID': range(len(test_preds)), target_col: test_preds})

submission.to_csv('submission.csv', index=False)
print("ğŸ“� submission.csv created successfully!")
print(submission.head())


