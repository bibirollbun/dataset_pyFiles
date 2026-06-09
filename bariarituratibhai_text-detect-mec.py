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


# mercor_pipeline_verified.py
import os
import sys
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "/kaggle/input/mercor-ai-detection" 
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_submission.csv")  # optional but used to validate IDs

REQUIRE_TEST_ROWS = 264   # Condition B: required number of test rows
TRUNCATE_IF_LARGER = True # if test has >264 rows, truncate to first 264 (preserves order)
N_SPLITS = 5              # Stratified KFold for stability check (Condition C)
RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 20000

# -----------------------------
# Helper utilities
# -----------------------------
def simple_clean(text):
    if pd.isna(text):
        return ""
    s = str(text)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def exit_err(msg):
    print("ERROR:", msg)
    sys.exit(1)

# -----------------------------
# Load data
# -----------------------------
if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
    exit_err(f"train.csv and/or test.csv not found in '{DATA_DIR}'. Put your files there and re-run.")

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print(f"Loaded train: {train.shape}, test: {test.shape}")

# optional sample_submission for ID verification
sample_exists = os.path.exists(SAMPLE_PATH)
sample = pd.read_csv(SAMPLE_PATH) if sample_exists else None
if sample_exists:
    print(f"Loaded sample_submission: {sample.shape}")

# -----------------------------
# Condition B: enforce 264 rows
# -----------------------------
if test.shape[0] != REQUIRE_TEST_ROWS:
    msg = f"Test rows = {test.shape[0]}, required = {REQUIRE_TEST_ROWS}."
    if test.shape[0] > REQUIRE_TEST_ROWS and TRUNCATE_IF_LARGER:
        warnings.warn(msg + f" Truncating to first {REQUIRE_TEST_ROWS} rows (preserve order).")
        test = test.iloc[:REQUIRE_TEST_ROWS].reset_index(drop=True)
    else:
        exit_err(msg + " Set TRUNCATE_IF_LARGER=True to allow truncation, or supply correct test.csv.")

print("Test rows after enforcement:", test.shape[0])

# -----------------------------
# Condition A: ID matching
# - If sample_submission provided, ensure submission IDs match sample exactly (values and order).
# - Else we will preserve test['id'] as-is and print a reminder to compare to Kaggle sample_submission.
# -----------------------------
if sample_exists:
    # ensure sample length matches enforced test length
    if sample.shape[0] != test.shape[0]:
        exit_err(f"sample_submission has {sample.shape[0]} rows but enforced test has {test.shape[0]} rows.")
    # Align test to sample order by sample['id'] if sample ids are present elsewhere
    # If the test ids are already the same set as sample ids, reorder test to sample order
    sample_ids = sample['id'].astype(str).tolist()
    test_ids = test['id'].astype(str).tolist()
    if set(sample_ids) == set(test_ids):
        # reorder test to match sample order precisely
        test = test.set_index(test['id'].astype(str)).loc[sample_ids].reset_index(drop=True)
        # replace test['id'] with sample ids to enforce identical dtype/format
        test['id'] = sample['id'].values
        print("Aligned test rows to match sample_submission IDs and order.")
    else:
        exit_err("IDs in sample_submission and test.csv do not match (different sets). Please provide the correct test.csv matching sample_submission.")
else:
    print("No sample_submission.csv found — submission will use test['id'] in its current order. Make sure these IDs match Kaggle's sample_submission IDs.")

# -----------------------------
# Basic cleaning
# -----------------------------
for df in (train, test):
    # ensure columns exist and clean answers
    if 'answer' not in df.columns:
        exit_err("Missing 'answer' column in train/test.")
    df['answer'] = df['answer'].astype(str).apply(simple_clean)

# -----------------------------
# Quick check: train label exists
# -----------------------------
if 'is_cheating' not in train.columns:
    exit_err("Missing 'is_cheating' column in train.csv.")

# -----------------------------
# Feature engineering: simple TF-IDF on topic + answer
# -----------------------------
train_text = train['topic'].astype(str) + " " + train['answer'].astype(str)
test_text  = test['topic'].astype(str) + " " + test['answer'].astype(str)

tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1,2), min_df=2)
X_all = tfidf.fit_transform(pd.concat([train_text, test_text], ignore_index=True))
X_train = X_all[:len(train_text)]
X_test  = X_all[len(train_text):]

y = train['is_cheating'].values

# -----------------------------
# Quick baseline: Stratified K-Fold CV (Condition C: evaluate stability)
# We compute OOF ROC-AUC on train using a fast model (LogisticRegression)
# -----------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(train))
fold_scores = []
models = []
fold = 0

print(f"Running {N_SPLITS}-fold Stratified CV to estimate stability (imitates public/private split behavior)...")

for tr_idx, val_idx in skf.split(X_train, y):
    fold += 1
    X_tr = X_train[tr_idx]
    X_val = X_train[val_idx]
    y_tr = y[tr_idx]
    y_val = y[val_idx]

    # pipeline with scaler and LR (works with sparse)
    clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(solver='saga', C=1.0, max_iter=2000, random_state=RANDOM_STATE))
    clf.fit(X_tr, y_tr)

    proba_val = clf.predict_proba(X_val)[:, 1]
    oof[val_idx] = proba_val
    score = roc_auc_score(y_val, proba_val)
    fold_scores.append(score)
    models.append(clf)
    print(f" Fold {fold} ROC-AUC: {score:.4f}")

mean_auc = np.mean(fold_scores)
std_auc = np.std(fold_scores)
oof_auc = roc_auc_score(y, oof)
print(f"\nCV mean ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")
print(f"OOF ROC-AUC (full): {oof_auc:.4f}")

# This gives you an idea how your model may behave between public (30%) and private (70%) splits.
print("Interpretation tip: small std across folds indicates stable generalization (less leaderboard variance).")

# -----------------------------
# Fit final model on full train (using same model as folds' base)
# -----------------------------
final_clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(solver='saga', C=1.0, max_iter=2000, random_state=RANDOM_STATE))
final_clf.fit(X_train, y)

# -----------------------------
# Final predictions on enforced test set
# -----------------------------
test_pred = final_clf.predict_proba(X_test)[:, 1]

# -----------------------------
# Build submission ensuring IDs match sample (Condition A) and row count = 264 (Condition B)
# -----------------------------
submission = pd.DataFrame({
    "id": test['id'].values,
    "is_cheating": test_pred
})

# final safety checks
if submission.shape[0] != REQUIRE_TEST_ROWS:
    exit_err(f"Submission rows = {submission.shape[0]} but expected {REQUIRE_TEST_ROWS}.")
if sample_exists:
    # verify exact match to sample_submission IDs/order
    if not np.array_equal(submission['id'].astype(str).values, sample['id'].astype(str).values):
        exit_err("Final submission IDs do NOT exactly match sample_submission IDs after alignment. Aborting to avoid Kaggle mismatch.")
print("\nAll checks passed: submission IDs match and row count is correct.")

# -----------------------------
# Save submission
# -----------------------------
OUT_PATH = "submission.csv"
submission.to_csv(OUT_PATH, index=False)
print(f"Saved submission to {OUT_PATH}")
print(submission.head(10))





