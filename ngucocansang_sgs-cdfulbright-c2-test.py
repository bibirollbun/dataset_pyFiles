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


import os, re, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load
train = pd.read_csv("/kaggle/input/rmit-hackathon-2025/train.csv")
test = pd.read_csv("/kaggle/input/rmit-hackathon-2025/test.csv")
train.columns = train.columns.str.lower()
test.columns  = test.columns.str.lower()
train["text"] = train["text"].astype(str)
test["text"]  = test["text"].astype(str)

# Clean text
def clean_text(s):
    s = s.lower()
    s = re.sub(r"http\S+|www\S+", " url ", s)
    s = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", " ipaddr ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

train["clean"] = train["text"].apply(clean_text)
test["clean"]  = test["text"].apply(clean_text)

# Encode label
y = train["label"].map({"benign":0, "jailbreak":1}).values

# Basic features
def build_feats(df):
    s = df["clean"]
    return pd.DataFrame({
        "len": s.str.len(),
        "words": s.str.split().apply(len),
        "exc": s.str.count("!"),
        "quest": s.str.count(r"\?"),
        "has_url": s.str.contains("url").astype(int)
    })

feat_train = build_feats(train)
feat_test  = build_feats(test)
scaler = StandardScaler()
feat_train_s = scaler.fit_transform(feat_train)
feat_test_s  = scaler.transform(feat_test)

feat_train_sp = csr_matrix(feat_train_s)
feat_test_sp  = csr_matrix(feat_test_s)

# TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=20000,
    min_df=3,
    stop_words="english"
)
X_tfidf = vectorizer.fit_transform(train["clean"])
X_test_tfidf = vectorizer.transform(test["clean"])

# Combine features
X = hstack([X_tfidf, feat_train_sp])
X_test = hstack([X_test_tfidf, feat_test_sp])

# Train with cross-val
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(X.shape[0])
test_preds = np.zeros(X_test.shape[0])

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\nFold {fold}")
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    clf = LogisticRegression(C=3.0, solver="saga", max_iter=2000, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    
    oof[val_idx] = clf.predict_proba(X_val)[:,1]
    test_preds += clf.predict_proba(X_test)[:,1] / kf.n_splits
    
    print("Fold AUC:", roc_auc_score(y_val, oof[val_idx]))

# Final AUC
print("\nOOF AUC:", roc_auc_score(y, oof))

# Final train
final_clf = LogisticRegression(C=3.0, solver="saga", max_iter=2000, n_jobs=-1)
final_clf.fit(X, y)
final_preds = final_clf.predict_proba(X_test)[:,1]

# Submission
submission = pd.DataFrame({
    "Id": test["id"] if "id" in test.columns else test.index,
    "TARGET": final_preds
})
submission.to_csv("submission.csv", index=False)
print("Saved: submission_fast.csv")


