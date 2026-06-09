# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load Data
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings like the one you saw

# Quick View
train.head()


# Check for missing values and data types
train.info()
train.isnull().sum()
train['Personality'].value_counts()


# Plot personality distribution
sns.countplot(data=train, x='Personality', palette='pastel')
plt.title('Target Class Distribution')
plt.show()


# Histograms of numerical features
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
train[num_cols].hist(bins=30, figsize=(15, 8), color='skyblue', edgecolor='black')
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()


# Categorical vs Target
cat_cols = ['Stage_fear', 'Drained_after_socializing']

for col in cat_cols:
    sns.countplot(data=train, x=col, hue='Personality', palette='Set2')
    plt.title(f'{col} vs Personality')
    plt.show()


# Fill missing numerical columns with median
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
for col in num_cols:
    train[col].fillna(train[col].median(), inplace=True)

# Fill missing categorical columns with mode
cat_cols = ['Stage_fear', 'Drained_after_socializing']
for col in cat_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)


# Binary encoding
binary_map = {'Yes': 1, 'No': 0}
train['Stage_fear'] = train['Stage_fear'].map(binary_map)
train['Drained_after_socializing'] = train['Drained_after_socializing'].map(binary_map)

# Encode target
label_map = {'Introvert': 0, 'Extrovert': 1}
train['Personality'] = train['Personality'].map(label_map)


# Drop ID column
X = train.drop(['id', 'Personality'], axis=1)
y = train['Personality']

# Train-validation split (for local testing)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
model.fit(X_train, y_train)


# Predict on validation set
y_pred = model.predict(X_val)

# Accuracy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

acc = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", acc)

# Detailed report
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# Copy test data for safety
test_copy = test.copy()

# Fill missing values
for col in num_cols:
    test_copy[col].fillna(train[col].median(), inplace=True)

for col in cat_cols:
    test_copy[col].fillna(train[col].mode()[0], inplace=True)

# Encode binary features
test_copy['Stage_fear'] = test_copy['Stage_fear'].map(binary_map)
test_copy['Drained_after_socializing'] = test_copy['Drained_after_socializing'].map(binary_map)


# Make a fresh copy
test_copy = test.copy()

# Impute numerical columns with training set medians
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
for col in num_cols:
    test_copy[col] = test_copy[col].fillna(train[col].median())

# Impute categorical columns with training set mode
cat_cols = ['Stage_fear', 'Drained_after_socializing']
for col in cat_cols:
    test_copy[col] = test_copy[col].fillna(train[col].mode()[0])

# Map binary categorical columns
binary_map = {'Yes': 1, 'No': 0}
test_copy['Stage_fear'] = test_copy['Stage_fear'].map(binary_map)
test_copy['Drained_after_socializing'] = test_copy['Drained_after_socializing'].map(binary_map)

# Final check and fill any remaining NaNs (just in case)
test_copy.fillna(0, inplace=True)

# Drop 'id' column
X_test = test_copy.drop('id', axis=1)

# Ensure columns match training set exactly
X_test = X_test[X_train.columns]

# Predict
test_preds = model.predict(X_test)


# Map predictions back to label names
reverse_label_map = {0: 'Introvert', 1: 'Extrovert'}
test_preds_labels = [reverse_label_map[p] for p in test_preds]

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_copy['id'],
    'Personality': test_preds_labels
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

# Display first few rows
submission.head()

