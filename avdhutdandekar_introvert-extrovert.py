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

train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')



# Data manipulation
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully! ✅")





# Load real Kaggle competition data
print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())
print(f"\nTrain shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTrain head:")
train.head()





# Define feature columns (all except 'id' and 'Personality')
feature_cols = [col for col in train.columns if col not in ['id', 'Personality']]
print("Feature columns:", feature_cols)

# Create copies to avoid modifying original data
train_data = train[feature_cols + ['Personality']].copy()
test_data = test[feature_cols].copy()

# Convert categorical Yes/No columns to numeric (LabelEncoding)
for col in ['Stage_fear', 'Going_outside', 'Drained_after_socializing']:
    train_data[col] = train_data[col].astype(str).str.strip().str.lower()
    test_data[col] = test_data[col].astype(str).str.strip().str.lower()
    train_data[col] = train_data[col].replace({'yes': 1, 'no': 0, 'nan': np.nan})
    test_data[col] = test_data[col].replace({'yes': 1, 'no': 0, 'nan': np.nan})

# Fill missing values with median
for col in feature_cols:
    train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
    test_data[col] = pd.to_numeric(test_data[col], errors='coerce')
    if train_data[col].isnull().any():
        median_val = train_data[col].median()
        train_data[col].fillna(median_val, inplace=True)
        test_data[col].fillna(median_val, inplace=True)

print(f"\nTrain data shape after preprocessing: {train_data.shape}")
print(f"Test data shape after preprocessing: {test_data.shape}")

# EDA - Class Distribution
print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("="*50)
print("\nClass Distribution:")
print(train_data['Personality'].value_counts())
print(f"\nClass Percentages:")
print(train_data['Personality'].value_counts(normalize=True) * 100)

# Train-validation split with stratification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

X_train, X_val, y_train, y_val = train_test_split(
    train_data[feature_cols], 
    train_data['Personality'], 
    test_size=0.2, 
    random_state=42, 
    stratify=train_data['Personality']
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")

# Apply StandardScaler for numerical features
print("\n" + "="*50)
print("APPLYING STANDARDSCALER FOR FEATURE SCALING")
print("="*50)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_data[feature_cols])

print(f"\nFeature scaling completed!")
print(f"Mean of scaled features (should be ~0): {X_train_scaled.mean(axis=0).round(2)}")
print(f"Std of scaled features (should be ~1): {X_train_scaled.std(axis=0).round(2)}")

# Label encode the target
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)
print(f"\nLabel encoding mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

# Train baseline model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train_encoded)

y_val_pred = model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val_encoded, y_val_pred)
print(f"\nBaseline Validation Accuracy (with scaling): {val_accuracy:.4f}")


# HYPERPARAMETER TUNING WITH RANDOMSEARCHCV
print("\n" + "="*50)
print("HYPERPARAMETER TUNING FOR RANDOM FOREST")
print("="*50)

from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Define hyperparameter grid for RandomForest
param_distributions = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Initialize RandomForest
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

# RandomizedSearchCV with 3-fold cross-validation
random_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_distributions,
    n_iter=20,  # Number of parameter settings sampled
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1,
    scoring='accuracy'
)

# Fit on scaled data
print("\nStarting hyperparameter tuning...")
random_search.fit(X_train_scaled, y_train_encoded)

# Best parameters
print("\nBest Hyperparameters:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

# Best model performance
best_rf_model = random_search.best_estimator_
y_val_pred_tuned = best_rf_model.predict(X_val_scaled)
tuned_accuracy = accuracy_score(y_val_encoded, y_val_pred_tuned)

print(f"\nBaseline RF Accuracy: {val_accuracy:.4f}")
print(f"Tuned RF Accuracy: {tuned_accuracy:.4f}")
print(f"Improvement: {(tuned_accuracy - val_accuracy):.4f}")

# Best cross-validation score
print(f"\nBest CV Score: {random_search.best_score_:.4f}")
print(f"\nHyperparameter tuning completed!")


# Use best model for test predictions
from sklearn.ensemble import RandomForestClassifier

features = ['Time_spent_Alone','Stage_fear','Social_event_attendance','Going_outside','Drained_after_socializing','Friends_circle_size','Post_frequency']

# Train new Random Forest model with correct features
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(train_data[features], train_data['Personality'])

test_preds = rf_model.predict(test_data[features])

# Prepare submission
submission = sample_submission.copy()
submission['Personality'] = test_preds
submission.to_csv('submission.csv', index=False)
print('Submission file created: submission.csv')

