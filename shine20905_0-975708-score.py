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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import RandomizedSearchCV

# Load data
print("Loading data...")
train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train columns: {train.columns.tolist()}")

# Handle categorical features
cat_cols = ['Stage_fear', 'Drained_after_socializing']

# Combine train and test to get all possible categories
for col in cat_cols:
    combined = pd.concat([train[col], test[col]])
    le = LabelEncoder()
    le.fit(combined)
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# Encode target
train['target'] = (train['Personality'] == 'Extrovert').astype(int)

# Prepare features
X = train.drop(['id', 'Personality', 'target'], axis=1)
y = train['target']

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# Initialize models
xgb = XGBClassifier(random_state=42, eval_metric='logloss', verbose=0)
catboost = CatBoostClassifier(random_state=42, verbose=0)
lgbm = LGBMClassifier(random_state=42, verbose=-1)

# Train individual models
print("\nTraining XGBoost...")
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_val)
xgb_acc = accuracy_score(y_val, xgb_pred)
print(f"XGBoost Accuracy: {xgb_acc:.4f}")

print("Training CatBoost...")
catboost.fit(X_train, y_train)
catboost_pred = catboost.predict(X_val)
catboost_acc = accuracy_score(y_val, catboost_pred)
print(f"CatBoost Accuracy: {catboost_acc:.4f}")

print("Training LightGBM...")
lgbm.fit(X_train, y_train)
lgbm_pred = lgbm.predict(X_val)
lgbm_acc = accuracy_score(y_val, lgbm_pred)
print(f"LightGBM Accuracy: {lgbm_acc:.4f}")

# Create ensemble model
print("\nTraining Ensemble Model...")
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('catboost', catboost),
        ('lgbm', lgbm)
    ],
    voting='soft'
)

ensemble.fit(X_train, y_train)
ensemble_pred = ensemble.predict(X_val)
ensemble_acc = accuracy_score(y_val, ensemble_pred)
print(f"Ensemble Accuracy: {ensemble_acc:.4f}")

# Hyperparameter grid for XGBoost
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

print("\nTuning XGBoost hyperparameters...")
xgb_base = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False, verbosity=0)
xgb_search = RandomizedSearchCV(
    xgb_base,
    param_distributions=param_dist,
    n_iter=10,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42
)
xgb_search.fit(X_train, y_train)
print("Best XGBoost params:", xgb_search.best_params_)

# Use the best XGBoost estimator
xgb = xgb_search.best_estimator_

# Re-train CatBoost and LightGBM as before
catboost = CatBoostClassifier(random_state=42, verbose=0)
lgbm = LGBMClassifier(random_state=42, verbose=-1)

print("\nTraining CatBoost...")
catboost.fit(X_train, y_train)
catboost_pred = catboost.predict(X_val)
catboost_acc = accuracy_score(y_val, catboost_pred)
print(f"CatBoost Accuracy: {catboost_acc:.4f}")

print("Training LightGBM...")
lgbm.fit(X_train, y_train)
lgbm_pred = lgbm.predict(X_val)
lgbm_acc = accuracy_score(y_val, lgbm_pred)
print(f"LightGBM Accuracy: {lgbm_acc:.4f}")

# Ensemble with tuned XGBoost
print("\nTraining Ensemble Model with Tuned XGBoost...")
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('catboost', catboost),
        ('lgbm', lgbm)
    ],
    voting='soft'
)

ensemble.fit(X_train, y_train)
ensemble_pred = ensemble.predict(X_val)
ensemble_acc = accuracy_score(y_val, ensemble_pred)
print(f"Ensemble Accuracy: {ensemble_acc:.4f}")

# Detailed evaluation
print("\n=== ENSEMBLE MODEL EVALUATION ===")
print(f"Accuracy: {ensemble_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, ensemble_pred, target_names=['Introvert', 'Extrovert']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, ensemble_pred))

# Feature importance (using XGBoost as example)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Prepare test data for predictions
X_test = test.drop(['id'], axis=1)
print(f"\nTest features shape: {X_test.shape}")

# Make predictions on test set
test_predictions = ensemble.predict(X_test)
test_probabilities = ensemble.predict_proba(X_test)[:, 1]  # Probability of being Extrovert

# Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'Personality': ['Extrovert' if pred == 1 else 'Introvert' for pred in test_predictions],
    'probability_extrovert': test_probabilities
})

submission.to_csv('submission.csv', index=False)
print(f"\nSubmission file saved with {len(submission)} predictions")
print(f"Predicted Extroverts: {(test_predictions == 1).sum()}")
print(f"Predicted Introverts: {(test_predictions == 0).sum()}") 


