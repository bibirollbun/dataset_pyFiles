# Step 1: Load and Explore the Data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/kaggle/input/creditcard/creditcard.csv')

# Overview of the dataset
print("Dataset Info:")
print(df.info())

# Check the class distribution
print("\nClass Distribution:")
print(df['Class'].value_counts())

# Plot class distribution
plt.figure(figsize=(8, 4))
sns.countplot(x='Class', data=df)
plt.title("Class Distribution")
plt.show()



from sklearn.preprocessing import StandardScaler

# Scale 'Amount' column
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# Drop the 'Time' column
df = df.drop(columns=['Time'])

# Check the dataset after preprocessing
print("\nData after preprocessing:")
print(df.head())


from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = df.drop(columns=['Class'])
y = df['Class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Check the class distribution in training and testing sets
print("\nClass distribution in training set:")
print(y_train.value_counts())

print("\nClass distribution in testing set:")
print(y_test.value_counts())


pip install --upgrade pip


pip install imblearn


from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Check class distribution after SMOTE
print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts())


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Train logistic regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_smote, y_train_smote)

# Predict on test set
y_pred = log_reg.predict(X_test)

# Evaluate performance
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Logistic Regression)")
plt.show()


import numpy as np


from sklearn.ensemble import RandomForestClassifier

# Train Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_smote, y_train_smote)

# Predict on test set
y_pred_rf = rf_clf.predict(X_test)

# Evaluate performance
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# Feature Importance
importances = rf_clf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X_train.shape[1]), importances[sorted_indices], align="center")
plt.xticks(range(X_train.shape[1]), X.columns[sorted_indices], rotation=90)
plt.show()


from xgboost import XGBClassifier

# Train XGBoost
xgb_clf = XGBClassifier(scale_pos_weight=284315/492, random_state=42)
xgb_clf.fit(X_train, y_train)

# Predict on test set
y_pred_xgb = xgb_clf.predict(X_test)

# Evaluate performance
print("\nClassification Report (XGBoost):")
print(classification_report(y_test, y_pred_xgb))


from sklearn.metrics import roc_auc_score

# Calculate ROC-AUC scores
roc_log_reg = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
roc_rf = roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:, 1])
roc_xgb = roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1])

print(f"ROC-AUC Scores:")
print(f"Logistic Regression: {roc_log_reg}")
print(f"Random Forest: {roc_rf}")
print(f"XGBoost: {roc_xgb}")

