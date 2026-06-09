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


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

import warnings
warnings.filterwarnings('ignore')


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


# Display basic info
print("Training Data Info:")
print(train.info())
print("\nTest Data Info:")
print(test.info())


# Summary statistics for numerical features
print("\nTraining Data Summary Statistics:")
print(train.describe())


# Check for missing values
print("\nMissing Values in Training Data:")
print(train.isnull().sum())
print("\nMissing Values in Test Data:")
print(test.isnull().sum())


# Check for duplicated rows
print("Duplicates in Train:", train.duplicated().sum())
print("Duplicates in test:", test.duplicated().sum())


# Target variable distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Personality', data=train)
plt.title('Distribution of Personality Types')
plt.show()


# Numerical features distribution
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.ravel()
for i, col in enumerate(numerical_cols):
    sns.histplot(train[col], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


# Categorical features distribution
categorical_cols = ['Stage_fear', 'Drained_after_socializing']
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, col in enumerate(categorical_cols):
    sns.countplot(x=col, hue='Personality', data=train, ax=axes[i])
    axes[i].set_title(f'{col} vs Personality')
plt.tight_layout()
plt.show()


# Correlation matrix for numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(train[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# Check for outliers using boxplots
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.ravel()
for i, col in enumerate(numerical_cols):
    sns.boxplot(y=col, x='Personality', data=train, ax=axes[i])
    axes[i].set_title(f'Boxplot of {col} by Personality')
plt.tight_layout()
plt.show()


# Copy the data for later use
train_original = train.copy()


# Handle missing values
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing']

for col in numerical_cols:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(test[col].median())

for col in categorical_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])


# Encode categorical variables
label_encoders = {}
for col in categorical_cols + ['Personality']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    if col != 'Personality':  # Avoid encoding test target
        test[col] = le.transform(test[col])
    label_encoders[col] = le


# Scale numerical features
scaler = StandardScaler()
train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])


# Separate features and target
X = train.drop(columns=['id', 'Personality'])
y = train['Personality']
X_test = test.drop(columns=['id'])


# Create a social engagement score
X['social_engagement_score'] = (X['Social_event_attendance'] + X['Going_outside'] + 
                               X['Friends_circle_size'] + X['Post_frequency']) / 4

X_test['social_engagement_score'] = (X_test['Social_event_attendance'] + X_test['Going_outside'] + 
                                    X_test['Friends_circle_size'] + X_test['Post_frequency']) / 4


# Add interaction term between Time_spent_Alone and Stage_fear
X['time_stage_interaction'] = X['Time_spent_Alone'] * X['Stage_fear']
X_test['time_stage_interaction'] = X_test['Time_spent_Alone'] * X_test['Stage_fear']


# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True)
}


# Perform cross-validation
cv_scores = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_scores[name] = scores.mean()
    print(f"{name} CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")


# Train and evaluate best model (highest CV score) on full training set
best_model_name = max(cv_scores, key=cv_scores.get)
best_model = models[best_model_name]
best_model.fit(X, y)
y_pred = best_model.predict(X)
train_accuracy = accuracy_score(y, y_pred)
print(f"\nBest Model: {best_model_name}")
print(f"Training Accuracy: {train_accuracy:.4f}")


# Store best model for later use
joblib.dump(best_model, 'best_model.pkl')


# Load original preprocessed data 
train = train_original.copy()
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


# Reapply preprocessing 
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing']

for col in numerical_cols:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(test[col].median())

for col in categorical_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])

label_encoders = {}
for col in categorical_cols + ['Personality']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    if col != 'Personality':
        test[col] = le.transform(test[col])
    label_encoders[col] = le

scaler = StandardScaler()
train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])

X = train.drop(columns=['id', 'Personality'])
y = train['Personality']
X_test = test.drop(columns=['id'])


# Apply feature engineering (same as Section 3)
X['social_engagement_score'] = (X['Social_event_attendance'] + X['Going_outside'] + 
                               X['Friends_circle_size'] + X['Post_frequency']) / 4
X_test['social_engagement_score'] = (X_test['Social_event_attendance'] + X_test['Going_outside'] + 
                                    X_test['Friends_circle_size'] + X_test['Post_frequency']) / 4
X['time_stage_interaction'] = X['Time_spent_Alone'] * X['Stage_fear']
X_test['time_stage_interaction'] = X_test['Time_spent_Alone'] * X_test['Stage_fear']


# Define parameter distribution for RandomizedSearchCV
param_dist = {
    'C': np.logspace(-1, 1, 10),
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto'] + list(np.logspace(-2, 0, 5))
}


# Initialize and run RandomizedSearchCV
svm = SVC(random_state=42, probability=True)
random_search = RandomizedSearchCV(svm, param_distributions=param_dist, n_iter=20, cv=5, 
                                  scoring='accuracy', n_jobs=-1, random_state=42)
random_search.fit(X, y)

# Best parameters and score
print("Best Parameters:", random_search.best_params_)
print("Best Cross-Validation Score:", random_search.best_score_)


# Train final model with best parameters
best_svm = random_search.best_estimator_
best_svm.fit(X, y)
train_accuracy = best_svm.score(X, y)
print("Training Accuracy with Best Parameters:", train_accuracy)


# Save the tuned model
joblib.dump(best_svm, 'best_tuned_model.pkl')


# Load the tuned model
best_svm = joblib.load('best_tuned_model.pkl')


# Load preprocessed data 
train = train_original.copy()
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


# Reapply preprocessing and feature engineering (same as before)
numerical_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
categorical_cols = ['Stage_fear', 'Drained_after_socializing']

for col in numerical_cols:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(test[col].median())

for col in categorical_cols:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])

label_encoders = {}
for col in categorical_cols + ['Personality']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    if col != 'Personality':
        test[col] = le.transform(test[col])
    label_encoders[col] = le

scaler = StandardScaler()
train[numerical_cols] = scaler.fit_transform(train[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])

X = train.drop(columns=['id', 'Personality'])
y = train['Personality']
X_test = test.drop(columns=['id'])

X['social_engagement_score'] = (X['Social_event_attendance'] + X['Going_outside'] + 
                               X['Friends_circle_size'] + X['Post_frequency']) / 4
X_test['social_engagement_score'] = (X_test['Social_event_attendance'] + X_test['Going_outside'] + 
                                    X_test['Friends_circle_size'] + X_test['Post_frequency']) / 4
X['time_stage_interaction'] = X['Time_spent_Alone'] * X['Stage_fear']
X_test['time_stage_interaction'] = X_test['Time_spent_Alone'] * X_test['Stage_fear']

# Perform cross-validation on tuned model
cv_scores = cross_val_score(best_svm, X, y, cv=5, scoring='accuracy')
print("Final CV Accuracy: {:.4f} (+/- {:.4f})".format(cv_scores.mean(), cv_scores.std() * 2))


# Generate predictions on test set
test_predictions = best_svm.predict(X_test)
test_predictions = label_encoders['Personality'].inverse_transform(test_predictions)


# Prepare submission file
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': test_predictions
})


# Save submission file
submission.to_csv('submission.csv', index=False)

