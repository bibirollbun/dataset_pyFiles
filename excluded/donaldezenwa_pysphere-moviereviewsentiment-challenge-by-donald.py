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


# Helper: find dataset files regardless of exact competition folder name
import os

def find_file(root="/kaggle/input", name="train.csv"):
    for d, _, files in os.walk(root):
        if name in files:
            return os.path.join(d, name)
    raise FileNotFoundError(f"{name} not found under {root}")

train_path  = find_file(name="train.csv")
test_path   = find_file(name="test.csv")
sample_path = find_file(name="sample_submission.csv")

train_path, test_path, sample_path



import pandas as pd

train_df = pd.read_csv(train_path)   # columns: id, review, sentiment
test_df  = pd.read_csv(test_path)    # columns: id, review
sample   = pd.read_csv(sample_path)

train_df.head(), test_df.head(), train_df.shape, test_df.shape



print("Columns:", train_df.columns.tolist())
print("Train label distribution:\n", train_df['sentiment'].value_counts(normalize=True))
print("Missing in train:", train_df.isna().sum().to_dict())
print("Missing in test:", test_df.isna().sum().to_dict())

# Simple length analysis
train_df['char_len'] = train_df['review'].astype(str).str.len()
print("Char length (train):", train_df['char_len'].describe())



import matplotlib.pyplot as plt

plt.figure()
train_df['sentiment'].value_counts().sort_index().plot(kind='bar')
plt.title("Label counts (0=neg, 1=pos)")
plt.xlabel("sentiment"); plt.ylabel("count")
plt.show()

plt.figure()
train_df['char_len'].plot(kind='hist', bins=30)
plt.title("Review length distribution (chars)")
plt.xlabel("chars"); plt.ylabel("count")
plt.show()



import re

def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)     # remove punctuation/symbols
    s = re.sub(r"\s+", " ", s).strip()     # normalize spaces
    return s

train_df['clean_review'] = train_df['review'].apply(clean_text)
test_df['clean_review']  = test_df['review'].apply(clean_text)



from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB

# Common vectorizer settings
tfidf_common = dict(
    max_features=10000,       # expressive but still fast for ~2k texts
    ngram_range=(1,2),        # unigrams + bigrams
    stop_words='english',     # light stopwording
    sublinear_tf=True         # helps with very frequent words
)

X_text = train_df['clean_review'].values
y      = train_df['sentiment'].values
X_test_text = test_df['clean_review'].values

pipelines = {
    "LogReg": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_common)),
        ("clf", LogisticRegression(max_iter=2000, C=2.0, class_weight='balanced', n_jobs=None))
    ]),
    "LinearSVC": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_common)),
        ("clf", LinearSVC(C=1.0))
    ]),
    "CompNB": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_common)),
        ("clf", ComplementNB(alpha=0.5))
    ])
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = {}
for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X_text, y, scoring="accuracy", cv=cv, n_jobs=-1)
    cv_scores[name] = (scores.mean(), scores.std())
    print(f"{name}: CV Acc = {scores.mean():.4f} Â± {scores.std():.4f}")

best_name = max(cv_scores, key=lambda k: cv_scores[k][0])
best_name, cv_scores[best_name]



best_pipe = pipelines[best_name]
best_pipe.fit(X_text, y)

test_preds = best_pipe.predict(X_test_text)
pd.Series(test_preds).value_counts()



submission = pd.DataFrame({
    "id": test_df["id"],
    "sentiment": test_preds.astype(int)
})
submission.head()



out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print(f"âœ… Saved: {out_path}")



from sklearn.ensemble import VotingClassifier

# Build fresh independent pipelines (so they don't share fitted state)
lr_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(**tfidf_common)),
    ("clf", LogisticRegression(max_iter=2000, C=2.0, class_weight='balanced'))
])

svm_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(**tfidf_common)),
    ("clf", LinearSVC(C=1.0))
])

nb_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(**tfidf_common)),
    ("clf", ComplementNB(alpha=0.5))
])

# VotingClassifier expects estimators, not Pipelines that will be cloned; that's OK.
voter = VotingClassifier(
    estimators=[("lr", lr_pipe), ("svm", svm_pipe), ("nb", nb_pipe)],
    voting="hard"  # LinearSVC has no predict_proba; use hard voting
)

voter_scores = cross_val_score(voter, X_text, y, scoring="accuracy", cv=cv, n_jobs=-1)
print(f"Voting (hard): CV Acc = {voter_scores.mean():.4f} Â± {voter_scores.std():.4f}")

# If ensemble is better, use it; else keep the single best
use_ensemble = voter_scores.mean() > cv_scores[best_name][0]
final_model_name = "Ensemble" if use_ensemble else best_name
print("Using:", final_model_name)

final_model = voter if use_ensemble else best_pipe
final_model.fit(X_text, y)
final_preds = final_model.predict(X_test_text)

final_sub = pd.DataFrame({"id": test_df["id"], "sentiment": final_preds.astype(int)})
final_sub_path = "/kaggle/working/submission.csv"
final_sub.to_csv(final_sub_path, index=False)
print(f"âœ… Final submission saved: {final_sub_path} (model: {final_model_name})")


