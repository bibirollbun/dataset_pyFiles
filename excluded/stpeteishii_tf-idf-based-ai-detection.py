


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt

# Load data
train_df = pd.read_csv('/kaggle/input/mercor-ai-detection/train.csv')
test_df = pd.read_csv('/kaggle/input/mercor-ai-detection/test.csv')

print(f"Training set: {len(train_df)} samples")
print(f"Test set: {len(test_df)} samples")
print(f"Class distribution:\n{train_df['is_cheating'].value_counts()}\n")

# Simple text preprocessing
def preprocess_text(text):
    """Basic text preprocessing"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    return text

train_df['answer'] = train_df['answer'].apply(preprocess_text)
test_df['answer'] = test_df['answer'].apply(preprocess_text)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=5000,      # Maximum number of features
    min_df=5,               # Minimum document frequency
    max_df=0.8,             # Maximum document frequency
    ngram_range=(1, 2),     # Use unigrams and bigrams
    stop_words='english'    # Remove English stop words
)

# Vectorize training data
X_train_tfidf = tfidf.fit_transform(train_df['answer'])
y_train = train_df['is_cheating'].values

print(f"Number of TF-IDF features: {X_train_tfidf.shape[1]}\n")
print(type(X_train_tfidf))
print(type(y_train))


# Split into training and validation sets
X_train, X_val, y_train_split, y_val = train_test_split(
    X_train_tfidf, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples\n")

# Train Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'  # Handle class imbalance
)

model.fit(X_train, y_train_split)

# Evaluate on validation set
y_val_pred = model.predict(X_val)
y_val_pred_proba = model.predict_proba(X_val)[:, 1]

val_auc = roc_auc_score(y_val, y_val_pred_proba)
print(f"Validation ROC-AUC: {val_auc:.4f}")
print(f"\nClassification Report (Validation):")
print(classification_report(y_val, y_val_pred, target_names=['Authentic', 'Inauthentic']))

# Cross-validation on full training set
cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='roc_auc')
print(f"\nCross-validation ROC-AUC scores: {cv_scores}")
print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")

# Retrain model on entire training set for predictions
model_final = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)
model_final.fit(X_train_tfidf, y_train)

# Generate predictions for test set
X_test_tfidf = tfidf.transform(test_df['answer'])
test_pred_proba = model_final.predict_proba(X_test_tfidf)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'is_cheating': test_pred_proba
})

submission.to_csv('submission.csv', index=False)
print(f"\nPrediction statistics:")
print(f"Mean probability: {test_pred_proba.mean():.4f}")
print(f"Std probability: {test_pred_proba.std():.4f}")
print(f"Min probability: {test_pred_proba.min():.4f}")
print(f"Max probability: {test_pred_proba.max():.4f}")




