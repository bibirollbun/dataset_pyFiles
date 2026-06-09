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


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PySphere Movie Review Sentiment Challenge - Complete Solution
This script provides an end-to-end solution for the sentiment analysis challenge.
"""

# ==============================================================================
# 1. IMPORTS AND SETUP
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# ==============================================================================
# 2. DATA LOADING
# ==============================================================================

print("Loading data...")
train = pd.read_csv('/kaggle/input/py-sphere-movie-review-sentiment-challenge/train.csv')
test = pd.read_csv('/kaggle/input/py-sphere-movie-review-sentiment-challenge/test.csv')
sample_submission = pd.read_csv('/kaggle/input/py-sphere-movie-review-sentiment-challenge/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Sample submission shape: {sample_submission.shape}")

# ==============================================================================
# 3. DATA EXPLORATION
# ==============================================================================

print("\n" + "="*50)
print("DATA EXPLORATION")
print("="*50)

# Display first few rows
print("\nTrain data head:")
print(train.head())
print("\nTest data head:")
print(test.head())

# Check for missing values
print("\nMissing values in train:", train.isnull().sum().sum())
print("Missing values in test:", test.isnull().sum().sum())

# Sentiment distribution
print("\nSentiment distribution in training data:")
print(train['sentiment'].value_counts())
print(f"Percentage positive: {train['sentiment'].mean()*100:.2f}%")

# Review length statistics
train['review_length'] = train['review'].str.len()
test['review_length'] = test['review'].str.len()

print("\nReview length statistics (training):")
print(train['review_length'].describe())

# ==============================================================================
# 4. TEXT PREPROCESSING FUNCTIONS
# ==============================================================================

def basic_preprocess(text):
    """Basic preprocessing: lowercase only"""
    return text.lower()

def advanced_preprocess(text):
    """Advanced preprocessing with cleaning"""
    # Remove special characters, numbers, and extra spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    return text

def enhanced_preprocess(text):
    """Enhanced preprocessing with additional features"""
    # Remove HTML tags if any
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove special characters but keep some punctuation for emphasis
    text = re.sub(r'[^a-zA-Z\s!?]', ' ', text)
    # Handle repeated characters (e.g., 'sooooo' -> 'so')
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    return text

# ==============================================================================
# 5. APPLY PREPROCESSING
# ==============================================================================

print("\n" + "="*50)
print("PREPROCESSING TEXT")
print("="*50)

# We'll use multiple preprocessing strategies
train['text_basic'] = train['review'].apply(basic_preprocess)
train['text_advanced'] = train['review'].apply(advanced_preprocess)
train['text_enhanced'] = train['review'].apply(enhanced_preprocess)

test['text_basic'] = test['review'].apply(basic_preprocess)
test['text_advanced'] = test['review'].apply(advanced_preprocess)
test['text_enhanced'] = test['review'].apply(enhanced_preprocess)

print("Preprocessing complete!")

# ==============================================================================
# 6. FEATURE ENGINEERING
# ==============================================================================

print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

# Split data for training and validation
X_train_basic, X_valid_basic, y_train, y_valid = train_test_split(
    train['text_basic'], train['sentiment'], test_size=0.2, random_state=42, stratify=train['sentiment']
)

X_train_advanced, X_valid_advanced, _, _ = train_test_split(
    train['text_advanced'], train['sentiment'], test_size=0.2, random_state=42, stratify=train['sentiment']
)

X_train_enhanced, X_valid_enhanced, _, _ = train_test_split(
    train['text_enhanced'], train['sentiment'], test_size=0.2, random_state=42, stratify=train['sentiment']
)

# TF-IDF Vectorizers with different configurations
tfidf_basic = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95)
tfidf_advanced = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), min_df=2, max_df=0.95)
tfidf_enhanced = TfidfVectorizer(max_features=15000, ngram_range=(1, 3), min_df=1, max_df=0.95, 
                                 sublinear_tf=True, use_idf=True)

# Fit and transform
X_train_tfidf_basic = tfidf_basic.fit_transform(X_train_basic)
X_valid_tfidf_basic = tfidf_basic.transform(X_valid_basic)
X_test_tfidf_basic = tfidf_basic.transform(test['text_basic'])

X_train_tfidf_advanced = tfidf_advanced.fit_transform(X_train_advanced)
X_valid_tfidf_advanced = tfidf_advanced.transform(X_valid_advanced)
X_test_tfidf_advanced = tfidf_advanced.transform(test['text_advanced'])

X_train_tfidf_enhanced = tfidf_enhanced.fit_transform(X_train_enhanced)
X_valid_tfidf_enhanced = tfidf_enhanced.transform(X_valid_enhanced)
X_test_tfidf_enhanced = tfidf_enhanced.transform(test['text_enhanced'])

print(f"Basic TF-IDF shape: {X_train_tfidf_basic.shape}")
print(f"Advanced TF-IDF shape: {X_train_tfidf_advanced.shape}")
print(f"Enhanced TF-IDF shape: {X_train_tfidf_enhanced.shape}")

# ==============================================================================
# 7. MODEL TRAINING - MULTIPLE MODELS
# ==============================================================================

print("\n" + "="*50)
print("TRAINING MODELS")
print("="*50)

# Dictionary to store results
results = {}

# Model 1: Logistic Regression with basic preprocessing
print("\n1. Logistic Regression (Basic)")
lr_basic = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr_basic.fit(X_train_tfidf_basic, y_train)
y_pred_lr_basic = lr_basic.predict(X_valid_tfidf_basic)
acc_lr_basic = accuracy_score(y_valid, y_pred_lr_basic)
print(f"Validation Accuracy: {acc_lr_basic:.4f}")
results['LR_Basic'] = acc_lr_basic

# Model 2: Logistic Regression with advanced preprocessing
print("\n2. Logistic Regression (Advanced)")
lr_advanced = LogisticRegression(max_iter=1000, random_state=42, C=0.5, solver='liblinear')
lr_advanced.fit(X_train_tfidf_advanced, y_train)
y_pred_lr_advanced = lr_advanced.predict(X_valid_tfidf_advanced)
acc_lr_advanced = accuracy_score(y_valid, y_pred_lr_advanced)
print(f"Validation Accuracy: {acc_lr_advanced:.4f}")
results['LR_Advanced'] = acc_lr_advanced

# Model 3: Linear SVC
print("\n3. Linear SVC (Enhanced)")
svc = LinearSVC(max_iter=2000, random_state=42, C=0.1)
svc.fit(X_train_tfidf_enhanced, y_train)
y_pred_svc = svc.predict(X_valid_tfidf_enhanced)
acc_svc = accuracy_score(y_valid, y_pred_svc)
print(f"Validation Accuracy: {acc_svc:.4f}")
results['SVC_Enhanced'] = acc_svc

# Model 4: Naive Bayes
print("\n4. Multinomial Naive Bayes (Advanced)")
nb = MultinomialNB(alpha=0.1)
nb.fit(X_train_tfidf_advanced, y_train)
y_pred_nb = nb.predict(X_valid_tfidf_advanced)
acc_nb = accuracy_score(y_valid, y_pred_nb)
print(f"Validation Accuracy: {acc_nb:.4f}")
results['NB_Advanced'] = acc_nb

# Model 5: Random Forest (on reduced features for speed)
print("\n5. Random Forest (Basic - reduced features)")
rf = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=42, n_jobs=-1)
rf.fit(X_train_tfidf_basic, y_train)
y_pred_rf = rf.predict(X_valid_tfidf_basic)
acc_rf = accuracy_score(y_valid, y_pred_rf)
print(f"Validation Accuracy: {acc_rf:.4f}")
results['RF_Basic'] = acc_rf

# ==============================================================================
# 8. CROSS-VALIDATION FOR BEST MODEL
# ==============================================================================

print("\n" + "="*50)
print("CROSS-VALIDATION")
print("="*50)

# Find best model
best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]
print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")

# Perform cross-validation on best configuration
if 'Basic' in best_model_name:
    X_cv = tfidf_basic.fit_transform(train['text_basic'])
    if 'LR' in best_model_name:
        model_cv = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
elif 'Advanced' in best_model_name:
    X_cv = tfidf_advanced.fit_transform(train['text_advanced'])
    if 'LR' in best_model_name:
        model_cv = LogisticRegression(max_iter=1000, random_state=42, C=0.5, solver='liblinear')
    elif 'NB' in best_model_name:
        model_cv = MultinomialNB(alpha=0.1)
else:  # Enhanced
    X_cv = tfidf_enhanced.fit_transform(train['text_enhanced'])
    model_cv = LinearSVC(max_iter=2000, random_state=42, C=0.1)

cv_scores = cross_val_score(model_cv, X_cv, train['sentiment'], cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ==============================================================================
# 9. ENSEMBLE MODEL
# ==============================================================================

print("\n" + "="*50)
print("ENSEMBLE MODEL")
print("="*50)

# Train final models on full training data for ensemble
print("Training ensemble components on full training data...")

# Prepare full training data
X_full_basic = tfidf_basic.fit_transform(train['text_basic'])
X_full_advanced = tfidf_advanced.fit_transform(train['text_advanced'])
X_full_enhanced = tfidf_enhanced.fit_transform(train['text_enhanced'])

# Train individual models
lr_final = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr_final.fit(X_full_basic, train['sentiment'])

svc_final = LinearSVC(max_iter=2000, random_state=42, C=0.1)
svc_final.fit(X_full_enhanced, train['sentiment'])

nb_final = MultinomialNB(alpha=0.1)
nb_final.fit(X_full_advanced, train['sentiment'])

# Get predictions from each model
test_pred_lr = lr_final.predict(tfidf_basic.transform(test['text_basic']))
test_pred_svc = svc_final.predict(tfidf_enhanced.transform(test['text_enhanced']))
test_pred_nb = nb_final.predict(tfidf_advanced.transform(test['text_advanced']))

# Ensemble voting (majority vote)
test_pred_ensemble = np.round((test_pred_lr + test_pred_svc + test_pred_nb) / 3).astype(int)

print("Ensemble training complete!")

# ==============================================================================
# 10. FINAL PREDICTION AND SUBMISSION
# ==============================================================================

print("\n" + "="*50)
print("GENERATING SUBMISSIONS")
print("="*50)

# Create multiple submissions
submissions = {}

# Submission 1: Best single model
if best_accuracy >= 0.99:  # If we have near-perfect accuracy, use that model
    print(f"Using {best_model_name} for primary submission")
    if 'LR_Basic' in best_model_name:
        final_predictions = lr_final.predict(tfidf_basic.transform(test['text_basic']))
    elif 'SVC' in best_model_name:
        final_predictions = svc_final.predict(tfidf_enhanced.transform(test['text_enhanced']))
    else:
        final_predictions = test_pred_ensemble
else:
    final_predictions = test_pred_ensemble

# Create submission dataframe
submission = pd.DataFrame({
    'id': test['id'],
    'sentiment': final_predictions.astype(int)
})

# Save primary submission
submission.to_csv('submission.csv', index=False)
print(f"Primary submission saved to 'submission.csv'")
print(f"Submission shape: {submission.shape}")
print("\nFirst 10 predictions:")
print(submission.head(10))

# Alternative submission with ensemble
submission_ensemble = pd.DataFrame({
    'id': test['id'],
    'sentiment': test_pred_ensemble.astype(int)
})
submission_ensemble.to_csv('submission_ensemble.csv', index=False)
print(f"\nEnsemble submission saved to 'submission_ensemble.csv'")

# ==============================================================================
# 11. RESULTS SUMMARY
# ==============================================================================

print("\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)

print("\nModel Performance Summary:")
for model, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model:20s}: {acc:.4f}")

print(f"\nBest Single Model: {best_model_name} ({best_accuracy:.4f})")
print(f"Cross-Validation Score: {cv_scores.mean():.4f}")

# Check prediction distribution
print("\nPrediction Distribution:")
print(f"Negative (0): {(final_predictions == 0).sum()} ({(final_predictions == 0).sum()/len(final_predictions)*100:.2f}%)")
print(f"Positive (1): {(final_predictions == 1).sum()} ({(final_predictions == 1).sum()/len(final_predictions)*100:.2f}%)")

print("\n" + "="*50)
print("PROCESS COMPLETE!")
print("="*50)
print("\nSubmission files created:")
print("1. submission.csv (primary)")
print("2. submission_ensemble.csv (ensemble backup)")
print("\nGood luck with the competition!")

