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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ===== 1. Load Data =====
train = pd.read_csv('/kaggle/input/mercor-ai-detection/train.csv')
test = pd.read_csv('/kaggle/input/mercor-ai-detection/test.csv')

# Fill missing text
train['answer'] = train['answer'].fillna('')
test['answer'] = test['answer'].fillna('')


# ===== 2. Train/Validation Split =====
X_train, X_val, y_train, y_val = train_test_split(
    train['answer'],
    train['is_cheating'],
    test_size=0.2,
    random_state=42,
    stratify=train['is_cheating']
)


# ===== 3. TF-IDF Vectorization (word and bigram) =====
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,2),
    stop_words='english'
)

X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val)


# ===== 4. Logistic Regression Classifier =====
clf = LogisticRegression(max_iter=500)
clf.fit(X_train_vec, y_train)


# ===== 5. Validation Score =====
val_pred = clf.predict_proba(X_val_vec)[:, 1]
roc = roc_auc_score(y_val, val_pred)
print("Validation ROC-AUC:", roc)


# ===== 6. Train on Full Data + Predict Test =====
full_tfidf = tfidf.fit_transform(train['answer'])
test_tfidf = tfidf.transform(test['answer'])

clf_full = LogisticRegression(max_iter=500)
clf_full.fit(full_tfidf, train['is_cheating'])

test_pred = clf_full.predict_proba(test_tfidf)[:, 1]


# ===== 7. Create Submission File =====
submission = pd.DataFrame({
    'id': test['id'],
    'is_cheating': test_pred
})

submission.to_csv('/kaggle/working/submission_baseline.csv', index=False)
print("Saved submission file: submission_baseline.csv")


subs = pd.read_csv('/kaggle/working/submission_baseline.csv')
subs.head()


# Load data
train = pd.read_csv('/kaggle/input/mercor-ai-detection/train.csv')
test = pd.read_csv('/kaggle/input/mercor-ai-detection/test.csv')
train['answer'] = train['answer'].fillna('')
test['answer'] = test['answer'].fillna('')


# Split
X_train, X_val, y_train, y_val = train_test_split(
    train['answer'], train['is_cheating'], 
    test_size=0.2, random_state=42, stratify=train['is_cheating']
)


# Character TF-IDF — chars capture punctuation, misspellings, structure
char_tfidf = TfidfVectorizer(
    analyzer='char',
    ngram_range=(1,5), #character 1–5 grams
    max_features=100_000
)

X_train_vec = char_tfidf.fit_transform(X_train)
X_val_vec = char_tfidf.transform(X_val)


# Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

val_pred = clf.predict_proba(X_val_vec)[:,1]
print("Char TF-IDF ROC-AUC:", roc_auc_score(y_val, val_pred))


# Train full + predict test
full_vec = char_tfidf.fit_transform(train['answer'])
test_vec = char_tfidf.transform(test['answer'])

clf_full = LogisticRegression(max_iter=1000)
clf_full.fit(full_vec, train['is_cheating'])

test_pred = clf_full.predict_proba(test_vec)[:,1]

submission = pd.DataFrame({'id': test['id'], 'is_cheating': test_pred})
submission.to_csv('/kaggle/working/submission_char_tfidf.csv', index=False)
print("Saved: submission_char_tfidf.csv")


model1 = pd.read_csv('/kaggle/working/submission_char_tfidf.csv')
model1.head()

