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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import re

# Load data
train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

print(f"Training data: {train.shape}, Test data: {test.shape}")
print(f"Target mean: {train['rule_violation'].mean():.3f}")

# SIMPLE but effective preprocessing
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s.!?]', ' ', text)  # Keep basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train['clean_text'] = train['body'].apply(clean_text)
test['clean_text'] = test['body'].apply(clean_text)

# ONLY the most important features
def create_simple_features(df):
    df['text_len'] = df['clean_text'].str.len()
    df['word_count'] = df['clean_text'].str.split().str.len()
    df['excl_count'] = df['body'].str.count('!')
    df['quest_count'] = df['body'].str.count('\?')
    df['has_question'] = (df['quest_count'] > 0).astype(int)
    df['has_exclamation'] = (df['excl_count'] > 0).astype(int)
    return df

train = create_simple_features(train)
test = create_simple_features(test)

# Simple features only
simple_features = ['text_len', 'word_count', 'has_question', 'has_exclamation']

# SINGLE VECTORIZER - optimized for generalization
vectorizer = TfidfVectorizer(
    max_features=4000,  # Reduced to prevent overfitting
    ngram_range=(1, 2),
    min_df=5,           # Focus on common patterns
    max_df=0.7,
    stop_words='english',
    sublinear_tf=True
)

X_text = train['clean_text']
y = train['rule_violation']

# Transform text
X_tfidf = vectorizer.fit_transform(X_text)
X_test_tfidf = vectorizer.transform(test['clean_text'])

# Add simple numeric features
X_numeric = train[simple_features].fillna(0).values
X_test_numeric = test[simple_features].fillna(0).values

# Combine features
X_combined = hstack([X_tfidf, X_numeric])
X_test_combined = hstack([X_test_tfidf, X_test_numeric])

print(f"Feature dimensions: {X_combined.shape}")

# SINGLE MODEL - Well-tuned Logistic Regression
print("\n=== OPTIMIZED SINGLE MODEL ===")
model = LogisticRegression(
    C=0.08,           # Carefully tuned regularization
    max_iter=2000,
    class_weight='balanced',
    random_state=42,
    solver='liblinear',
    penalty='l2'
)

# Cross-validation
cv_scores = cross_val_score(model, X_combined, y, cv=5, scoring='roc_auc')
print(f"CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train final model
model.fit(X_combined, y)

# Predict
test_preds = model.predict_proba(X_test_combined)[:, 1]

# CRITICAL: Adjust predictions to match training distribution
target_mean = y.mean()
current_mean = test_preds.mean()

print(f"\nPrediction adjustment:")
print(f"Before - Min: {test_preds.min():.3f}, Max: {test_preds.max():.3f}, Mean: {test_preds.mean():.3f}")
print(f"Target mean: {target_mean:.3f}")

# Simple adjustment to match target distribution
if current_mean < target_mean:
    # Increase predictions slightly
    adjustment = min(0.1, (target_mean - current_mean) * 1.5)
    test_preds = test_preds + adjustment
else:
    # Decrease predictions slightly
    adjustment = min(0.1, (current_mean - target_mean) * 1.5)
    test_preds = test_preds - adjustment

# Ensure bounds
test_preds = np.clip(test_preds, 0.05, 0.95)

print(f"After  - Min: {test_preds.min():.3f}, Max: {test_preds.max():.3f}, Mean: {test_preds.mean():.3f}")

# Create submission
submission = pd.DataFrame({
    "row_id": test["row_id"],
    "rule_violation": test_preds
})

submission.to_csv("submission.csv", index=False)
print("\n=== FINAL SUBMISSION ===")
print(submission)

