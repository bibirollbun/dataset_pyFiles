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


# ğŸ�¦ Bank Churn Prediction - Data Loading & Analysis
# =====================================

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("ğŸ�¦ Starting Bank Churn Analysis...\n")

# =====================================
# STEP 1: Check Input Files
# =====================================
print("ğŸ“‚ Checking Kaggle input folders:\n")
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)
    for filename in filenames:
        print("  -", filename)
print("\n")

# Automatically find train and test files (ignore sample_submission)
train_path, test_path = None, None

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if "train" in filename.lower() and filename.endswith(".csv"):
            train_path = os.path.join(dirname, filename)
        elif "test" in filename.lower() and filename.endswith(".csv"):
            test_path = os.path.join(dirname, filename)

if not train_path or not test_path:
    raise FileNotFoundError("â�Œ train.csv or test.csv not found in /kaggle/input folder!")

print(f"âœ… Train file located at: {train_path}")
print(f"âœ… Test file located at:  {test_path}\n")

# =====================================
# STEP 2: Load the Datasets
# =====================================
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("âœ… Datasets Loaded Successfully!\n")

# =====================================
# STEP 3: Basic Dataset Information
# =====================================
print("ğŸ“Š TRAIN DATA OVERVIEW ===")
print("Shape:", train_df.shape)
print("Columns:", list(train_df.columns))
print("\nPreview:\n", train_df.head())
print("\nInfo:\n")
print(train_df.info())

print("\nğŸ“Š TEST DATA OVERVIEW ===")
print("Shape:", test_df.shape)
print("Columns:", list(test_df.columns))
print("\nPreview:\n", test_df.head())
print("\nInfo:\n")
print(test_df.info())

# =====================================
# STEP 4: Missing Values Check
# =====================================
print("\nğŸ§© Missing Values in Train Data:\n", train_df.isnull().sum())
print("\nğŸ§© Missing Values in Test Data:\n", test_df.isnull().sum())

# =====================================
# STEP 5: Target Column (Churn) Distribution
# =====================================
if "churn" in train_df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x="churn", data=train_df, palette="pastel", edgecolor="black")
    plt.title("Customer Churn Distribution")
    plt.show()
else:
    print("\nâš ï¸� 'churn' column not found in train data.")

# =====================================
# STEP 6: Quick Visualization - Numeric Columns
# =====================================
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns

if len(numeric_cols) > 1:
    train_df[numeric_cols].hist(bins=20, figsize=(12, 8), color='skyblue', edgecolor='black')
    plt.suptitle("Numeric Feature Distributions", fontsize=14)
    plt.show()
else:
    print("\nâš ï¸� No numeric columns found for histogram visualization.")



# =====================================
# ğŸ�¦ BANK CHURN PREDICTION (KAGGLE READY)
# =====================================

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("ğŸ�¦ Starting Bank Churn Prediction...\n")

# =====================================
# STEP 1: Locate Train & Test Files
# =====================================
train_path, test_path = None, None

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if "train" in filename.lower() and filename.endswith(".csv"):
            train_path = os.path.join(dirname, filename)
        elif "test" in filename.lower() and filename.endswith(".csv"):
            test_path = os.path.join(dirname, filename)

if not train_path or not test_path:
    raise FileNotFoundError("â�Œ train.csv or test.csv not found!")

print(f"âœ… Train file: {train_path}")
print(f"âœ… Test file: {test_path}\n")

# =====================================
# STEP 2: Load Data
# =====================================
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("âœ… Data Loaded Successfully!")
print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}\n")

# =====================================
# STEP 3: Basic Info
# =====================================
print("ğŸ“Š TRAIN DATA PREVIEW ===")
print(train_df.head(), "\n")

# Target column is 'Exited'
target_col = 'Exited'

# =====================================
# STEP 4: Encode Categorical Columns
# =====================================
cat_cols = ['Geography', 'Gender']

le = LabelEncoder()
for col in cat_cols:
    if col in train_df.columns:
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

# =====================================
# STEP 5: Select Features
# =====================================
drop_cols = ['id', 'CustomerId', 'Surname']
X = train_df.drop(columns=drop_cols + [target_col])
y = train_df[target_col]

# =====================================
# STEP 6: Split for Accuracy Evaluation
# =====================================
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =====================================
# STEP 7: Train Model
# =====================================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# =====================================
# STEP 8: Evaluate Accuracy
# =====================================
y_pred = model.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred)
print(f"ğŸ�¯ Model Accuracy on Validation Data: {accuracy * 100:.2f}%")

# =====================================
# STEP 9: Predict on Test Data
# =====================================
X_test = test_df.drop(columns=drop_cols, errors='ignore')
test_preds = model.predict(X_test)

# =====================================
# STEP 10: Create Submission File
# =====================================
submission = pd.DataFrame({
    'id': test_df['id'],
    'Exited': test_preds
})

submission.to_csv('/kaggle/working/submission.csv', index=False)
print("\nâœ… submission.csv file created successfully at /kaggle/working/submission.csv")

# =====================================
# STEP 11: Visualization (Optional)
# =====================================
plt.figure(figsize=(6,4))
sns.countplot(x='Exited', data=train_df, palette='pastel', edgecolor='black')
plt.title('Customer Churn Distribution (Exited)')
plt.show()

print("\nğŸ�‰ Done! Upload 'submission.csv' to Kaggle for evaluation.")


