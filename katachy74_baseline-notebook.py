import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import os, warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET_COLS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

train_path = "/kaggle/input/ioai-2026-sf-r-comments-classification/train.csv"
test_path  = "/kaggle/input/ioai-2026-sf-r-comments-classification/test.csv"

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

print("Train:", train.shape, " Test:", test.shape)

train["comment_text"] = train["comment_text"].fillna(" ")
test["comment_text"]  = test["comment_text"].fillna(" ")

X = train["comment_text"].values
y = train[TARGET_COLS].values.astype(int)
X_test = test["comment_text"].values

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=(y.sum(axis=1) > 0).astype(int)
)

print(f"Train/Valid sizes: {len(X_train)}/{len(X_valid)}")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 1),
    max_features=50_000,
    min_df=5,
    max_df=0.9
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_valid_tfidf = vectorizer.transform(X_valid)
X_test_tfidf  = vectorizer.transform(X_test)

print("TF-IDF shapes:", X_train_tfidf.shape, X_valid_tfidf.shape)

models = {}
valid_pred_proba = np.zeros((len(X_valid), len(TARGET_COLS)))

for i, col in enumerate(TARGET_COLS):
    print(f"Training for {col}")
    clf = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=100
    )
    clf.fit(X_train_tfidf, y_train[:, i])
    models[col] = clf
    valid_pred_proba[:, i] = clf.predict_proba(X_valid_tfidf)[:, 1]

valid_pred_bin = (valid_pred_proba >= 0.5).astype(int)

macro_f1 = f1_score(y_valid, valid_pred_bin, average="macro", zero_division=0)
micro_f1 = f1_score(y_valid, valid_pred_bin, average="micro", zero_division=0)

print(f"\nSIMPLE BASELINE F1:")
print(f"  macro F1: {macro_f1:.4f}")
print(f"  micro F1: {micro_f1:.4f}")

X_full_tfidf = vectorizer.fit_transform(X)
X_test_tfidf = vectorizer.transform(X_test)

final_pred_proba = np.zeros((len(X_test), len(TARGET_COLS)))
for i, col in enumerate(TARGET_COLS):
    clf = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=100
    )
    clf.fit(X_full_tfidf, y[:, i])
    final_pred_proba[:, i] = clf.predict_proba(X_test_tfidf)[:, 1]

final_pred_bin = (final_pred_proba >= 0.5).astype(int)

submission = pd.DataFrame({"id": test["id"]})
for i, col in enumerate(TARGET_COLS):
    submission[col] = final_pred_bin[:, i].astype(int)

submission.to_csv("simple_submission.csv", index=False)
print("\nSaved simple_submission.csv:", submission.shape)
print(submission.head())


