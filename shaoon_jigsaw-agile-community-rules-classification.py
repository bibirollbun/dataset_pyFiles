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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack
from collections import Counter

# Load data
train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
sample_sub = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")

# Fill NA
for col in ['body', 'rule']:
    train_df[col] = train_df[col].fillna("none")
    test_df[col] = test_df[col].fillna("none")

y = train_df['rule_violation']

# Shared TF-IDF for classification
tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=10000)
tfidf.fit(pd.concat([train_df['body'], train_df['rule'], test_df['body'], test_df['rule']]))

X_body = tfidf.transform(train_df['body'])
X_rule = tfidf.transform(train_df['rule'])
X_combined = hstack([X_body, X_rule])

# PMI trace feature
def compute_pmi_trace(rule, body, n=4):
    def get_ngrams(text, n):
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    
    rule_ngrams = get_ngrams(rule, n)
    body_ngrams = get_ngrams(body, n)

    rule_freq = Counter(rule_ngrams)
    body_freq = Counter(body_ngrams)

    all_ngrams = set(rule_freq.keys()) | set(body_freq.keys())
    total_rule = sum(rule_freq.values())
    total_body = sum(body_freq.values())
    
    score = 0
    for ng in rule_freq:
        if ng in body_freq:
            p_rule = rule_freq[ng] / total_rule
            p_body = body_freq[ng] / total_body
            p_joint = (rule_freq[ng] + body_freq[ng]) / (total_rule + total_body)
            pmi = np.log2(p_joint / (p_rule * p_body + 1e-9) + 1e-9)

            # Check relative position
            pos_rule = rule.find(ng) / max(len(rule) - n + 1, 1)
            pos_body = body.find(ng) / max(len(body) - n + 1, 1)
            position_match = 1 - abs(pos_rule - pos_body)  # 1 if aligned
            score += pmi * position_match
    return score

# Train PMI features
pmi_features = np.array([
    compute_pmi_trace(train_df['rule'][i], train_df['body'][i])
    for i in range(len(train_df))
]).reshape(-1, 1)

scaler = MinMaxScaler()
pmi_scaled = scaler.fit_transform(pmi_features)
X_train_final = hstack([X_combined, pmi_scaled])

# Test
X_body_test = tfidf.transform(test_df['body'])
X_rule_test = tfidf.transform(test_df['rule'])
X_test_combined = hstack([X_body_test, X_rule_test])

pmi_test = np.array([
    compute_pmi_trace(test_df['rule'][i], test_df['body'][i])
    for i in range(len(test_df))
]).reshape(-1, 1)
pmi_test_scaled = scaler.transform(pmi_test)
X_test_final = hstack([X_test_combined, pmi_test_scaled])

# Train/test split
X_tr, X_val, y_tr, y_val = train_test_split(X_train_final, y, test_size=0.2, random_state=42)

clf = SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=1000, tol=1e-3)
clf.fit(X_tr, y_tr)

val_preds = clf.predict_proba(X_val)[:, 1]
val_score = roc_auc_score(y_val, val_preds)
print(f"âœ… Validation ROC AUC (PMI Trace): {val_score:.4f}")

test_preds = clf.predict_proba(X_test_final)[:, 1]
sample_sub['rule_violation'] = test_preds
sample_sub.to_csv("submission.csv", index=False)
print("ğŸ“� submission.csv saved.")





