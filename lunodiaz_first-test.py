# ============================================
# Project: Predicting Loan Payback
# Author: Alioune Sarr
# Role: Junior Data Analyst
#
# Workflow:
# 1. Load and inspect the dataset
# 2. Clean and prepare the data
# 3. Exploratory Data Analysis (EDA)
# 4. Feature preparation
# 5. Build a baseline model
# 6. Evaluate performance
# ============================================


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Display settings
pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")


# Load dataset
df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")

# Display first rows
df.head()


# Shape of the dataset
df.shape


# General information
df.info()


# Statistical summary
df.describe()


df.columns


# Target distribution
df["loan_paid_back"].value_counts()


# Target distribution (percentage)
df["loan_paid_back"].value_counts(normalize=True) * 100


plt.figure(figsize=(6,4))
sns.countplot(x="loan_paid_back", data=df)
plt.title("Distribution of Loan Payback")
plt.show()


plt.figure(figsize=(6,4))
sns.boxplot(x="loan_paid_back", y="annual_income", data=df)
plt.title("Annual Income vs Loan Payback")
plt.show()


plt.figure(figsize=(6,4))
sns.boxplot(x="loan_paid_back", y="credit_score", data=df)
plt.title("Credit Score vs Loan Payback")
plt.show()


plt.figure(figsize=(6,4))
sns.boxplot(x="loan_paid_back", y="loan_amount", data=df)
plt.title("Loan Amount vs Loan Payback")
plt.show()


plt.figure(figsize=(7,4))
sns.countplot(x="employment_status", hue="loan_paid_back", data=df)
plt.title("Employment Status vs Loan Payback")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(7,4))
sns.countplot(x="education_level", hue="loan_paid_back", data=df)
plt.title("Education Level vs Loan Payback")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8,4))
sns.countplot(x="loan_purpose", hue="loan_paid_back", data=df)
plt.title("Loan Purpose vs Loan Payback")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8,4))
sns.countplot(x="grade_subgrade", hue="loan_paid_back", data=df)
plt.title("Loan Grade vs Loan Payback")
plt.xticks(rotation=45)
plt.show()


# Separate features and target
X = df.drop(columns=["loan_paid_back", "id"])
y = df["loan_paid_back"]


categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

categorical_cols, numerical_cols


le = LabelEncoder()

for col in categorical_cols:
    X[col] = le.fit_transform(X[col])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))




