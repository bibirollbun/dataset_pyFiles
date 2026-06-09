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


import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
import re

# --- Load Data ---
train = pd.read_csv("/kaggle/input/rmit-hackathon-2025/train.csv")
test  = pd.read_csv("/kaggle/input/rmit-hackathon-2025/test.csv")

y = train['label'].map({'benign': 0, 'jailbreak': 1}).astype(int)

# --- Clean Text ---
def clean_text(s):
    if pd.isna(s): 
        return ""
    s = str(s)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

train['text'] = train['text'].apply(clean_text)
test['text']  = test['text'].apply(clean_text)

# --- Hand-Crafted + Keyword Features ---
def add_text_features(df):
    df['len_chars'] = df['text'].str.len()
    df['len_words'] = df['text'].str.split().map(len)
    df['count_digits'] = df['text'].apply(lambda x: sum(ch.isdigit() for ch in x))
    df['count_punct']  = df['text'].apply(lambda x: sum(1 for ch in x if ch in '.,;:!?()[]{}"\''))
    df['upper_ratio']  = df['text'].apply(lambda x: sum(1 for ch in x if ch.isupper()) / max(1, len(x)))
    return df

def keyword_flag(s):
    keywords = ['jailbreak', 'prompt', 'system', 'ignore', 'bypass', 
                'instruction', 'secret', 'admin', 'override']
    return sum(kw in s.lower() for kw in keywords)

train = add_text_features(train)
test  = add_text_features(test)
train['keyword_hits'] = train['text'].apply(keyword_flag)
test['keyword_hits']  = test['text'].apply(keyword_flag)

num_feat_cols = ['len_chars', 'len_words', 'count_digits', 
                 'count_punct', 'upper_ratio', 'keyword_hits']

# scale numeric features
scaler = MinMaxScaler()
scaler.fit(pd.concat([train[num_feat_cols], test[num_feat_cols]], axis=0))
train_num = scaler.transform(train[num_feat_cols])
test_num  = scaler.transform(test[num_feat_cols])

# --- TF-IDF Features (Word + Char) ---
word_vec = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=70000,
    min_df=2,
    strip_accents='unicode',
    sublinear_tf=True
)
X_word = word_vec.fit_transform(train['text'])
X_word_test = word_vec.transform(test['text'])

char_vec = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3,6),
    max_features=30000,
    sublinear_tf=True
)
X_char = char_vec.fit_transform(train['text'])
X_char_test = char_vec.transform(test['text'])

# combine TF-IDF + numeric
X_sparse = sps.hstack([X_word, X_char], format='csr')
X_sparse_test = sps.hstack([X_word_test, X_char_test], format='csr')
X = sps.hstack([X_sparse, sps.csr_matrix(train_num)], format='csr')
X_test = sps.hstack([X_sparse_test, sps.csr_matrix(test_num)], format='csr')

# --- Dimensionality Reduction ---
n_components = 500
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_reduced = svd.fit_transform(X)
X_reduced_test = svd.transform(X_test)

X_final = X_reduced
X_final_test = X_reduced_test

# --- Stratified K-Fold + Models ---
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train))
preds_lgb = np.zeros(len(test))
oof_lr = np.zeros(len(train))
preds_lr = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_final, y)):
    print(f"\n========== Fold {fold+1} ==========")
    X_tr, X_val = X_final[tr_idx], X_final[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    # LightGBM params (tuned for text)
    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.015,
        "num_leaves": 128,
        "max_depth": -1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 4,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "min_data_in_leaf": 25,
        "min_gain_to_split": 0.01,
        "verbose": -1,
        "scale_pos_weight": len(y[y==0]) / len(y[y==1]),
        "seed": 42 + fold
    }

    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    model = lgb.train(
        params=lgb_params,
        train_set=lgb_train,
        num_boost_round=6000,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=300),
            lgb.log_evaluation(200)
        ]
    )

    oof_lgb[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    preds_lgb += model.predict(X_final_test, num_iteration=model.best_iteration) / kf.n_splits

    # Logistic Regression
    lr = LogisticRegression(C=0.7, max_iter=3000, class_weight='balanced', solver='saga')
    lr.fit(X_tr, y_tr)
    oof_lr[val_idx] = lr.predict_proba(X_val)[:,1]
    preds_lr += lr.predict_proba(X_final_test)[:,1] / kf.n_splits

# --- Ensemble ---
oof_ensemble = 0.7 * oof_lgb + 0.3 * oof_lr
preds_ensemble = 0.7 * preds_lgb + 0.3 * preds_lr

# --- CV Scores ---
cv_lgb = roc_auc_score(y, oof_lgb)
cv_lr = roc_auc_score(y, oof_lr)
cv_ens = roc_auc_score(y, oof_ensemble)

print(f"\n CV AUC (LightGBM): {cv_lgb:.5f}")
print(f" CV AUC (LogReg):   {cv_lr:.5f}")
print(f" CV AUC (Ensemble): {cv_ens:.5f}")

# --- Save Submission ---
submission = pd.DataFrame({
    "Id": test["Id"],
    "target": preds_ensemble
})
submission.to_csv("submission.csv", index=False)
print("\n submission.csv saved — ready for Kaggle upload!")
print(submission.head(10).to_csv(index=False))


