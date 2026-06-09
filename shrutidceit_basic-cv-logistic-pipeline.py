import os, gc, re, textwrap
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42


train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test  = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

print(train.head(2))
print("Train shape:", train.shape)
print("Distinct rules in TRAIN:", train['rule'].nunique())
print(train['rule'].value_counts())


skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=RANDOM_STATE)


pipe = Pipeline([
    ("tfidf", TfidfVectorizer(      # Converts body into numerical TF-IDF vectors
        ngram_range=(1,3),          # unigrams + bigrams + trigrams
        min_df=2,                   # ignore words that appear in fewer than 2 doc
        max_features=250_000,       # cap features for speed/memory
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(     # Trains a Logistic Regression for binary classification on the TF-IDF features
        C=4.0,                      # Controls regularization strength - bit stronger than default (1.0)
        max_iter=300,               # Logistic regression uses an iterative solver - (optimization algorithm)max iterations allowed for convergence
        n_jobs=-1,                   # Controls parallelization - use all CPU cores
        random_state=RANDOM_STATE
    ))
])


oof = np.zeros(len(train))


for fold, (tr_idx, va_idx) in enumerate(skf.split(train['body'], train['rule_violation']), 1):
    tr_x = train.loc[tr_idx, 'body']
    tr_y = train.loc[tr_idx, 'rule_violation']
    va_x = train.loc[va_idx, 'body']
    va_y = train.loc[va_idx, 'rule_violation']

    pipe.fit(tr_x, tr_y) # Trains the pipeline (TF-IDF + Logistic Regression) on the training split
    oof[va_idx] = pipe.predict_proba(va_x)[:,1]  # Makes probability predictions (predict_proba) on the validation split
    auc = roc_auc_score(va_y, oof[va_idx])       # Computes AUC score for this fold
    print(f"Fold {fold} AUC: {auc:.4f}")


print("OOF AUC:", roc_auc_score(train['rule_violation'], oof))


final_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1,3),
        min_df=2,
        max_features=350_000,   # slightly higher capacity for final fit
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        C=4.0,
        max_iter=500,
        n_jobs=-1,
        random_state=RANDOM_STATE
    ))
])

final_pipe.fit(train['body'], train['rule_violation'])
test_pred = final_pipe.predict_proba(test['body'])[:, 1]  


# Cell 5: write submission
sub = pd.DataFrame({
    "row_id": test["row_id"],
    "rule_violation": test_pred
})
sub.to_csv("submission.csv", index=False)
sub.head()


