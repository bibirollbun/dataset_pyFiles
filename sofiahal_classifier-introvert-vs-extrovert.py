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
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression


# Load data
train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


# Basic EDA
print(train_df.info())
print(train_df.describe())
print(train_df['Personality'].value_counts())


# Visualizations
# 1. Correlation heatmap (for numerical features)
plt.figure(figsize=(10,6))
sns.heatmap(train_df.corr(numeric_only=True), annot=True)
plt.title('Feature Correlation')


# 2. Distribution of key features by personality type
fig, axes = plt.subplots(2, 3, figsize=(15,10))
sns.boxplot(x='Personality', y='Time_spent_Alone', data=train_df, ax=axes[0,0])
sns.boxplot(x='Personality', y='Social_event_attendance', data=train_df, ax=axes[0,1])
sns.boxplot(x='Personality', y='Friends_circle_size', data=train_df, ax=axes[0,2])
sns.countplot(x='Stage_fear', hue='Personality', data=train_df, ax=axes[1,0])
sns.countplot(x='Going_outside', hue='Personality', data=train_df, ax=axes[1,1])
sns.countplot(x='Drained_after_socializing', hue='Personality', data=train_df, ax=axes[1,2])
plt.tight_layout()


# Split data FIRST (before any feature preprocessing)
X = train_df.drop(['id', 'Personality'], axis=1)
y = train_df['Personality']  # Will encode later

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define column types
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Friends_circle_size', 'Post_frequency']
cat_cols = ['Stage_fear', 'Going_outside', 'Drained_after_socializing']


# Define numeric and categorical transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


# For categorical data:
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle nulls
    ('onehot', OneHotEncoder(drop='if_binary'))            # Auto-convert binary to 0/1
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])


# Initialize and fit LabelEncoder on target
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)  # Use same encoder


# Random Forest
rf_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(       
        n_estimators=200,  # Number of trees in the forest
        max_depth=10,       # Maximum depth of each tree
        min_samples_split=5,  # Minimum samples required to split a node
        random_state=42,    # For reproducibility
        verbose=1))
])

# Gradient Boosting
gb_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

# SVM (requires dense matrix)
svm_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='linear', probability=True))
])

# Logistic Regression
lr_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])


def evaluate_model(model, X, y, set_name):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    
    print(f"\n{set_name} Evaluation:")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred))
    print("Recall:", recall_score(y, y_pred))
    print("F1 Score:", f1_score(y, y_pred))
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Introvert', 'Extrovert'],
                yticklabels=['Introvert', 'Extrovert'])
    plt.title(f'{set_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return y_pred, y_proba


models = {
    'Random Forest': rf_pipe,
    'Gradient Boosting': gb_pipe,
    'SVM': svm_pipe,
    'Logistic Regression': lr_pipe
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train_encoded)
    val_acc = model.score(X_val, y_val_encoded)
    print(f"{name} Validation Accuracy: {val_acc:.4f}")
    
    evaluate_model(model, X_train, y_train_encoded, "Train")


# Get best model
best_model = max(models.values(), key=lambda x: x.score(X_val, y_val_encoded))

# Preprocess test data through the full pipeline
test_preds = best_model.predict(test_df.drop('id', axis=1))
test_pred_labels = le.inverse_transform(test_preds)  # Convert back to original labels

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Personality': test_pred_labels
})
submission.to_csv('predictions.csv', index=False)

