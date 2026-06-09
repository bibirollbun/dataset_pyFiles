import numpy as np
import pandas as pd
import re
import string
import gc

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

SEED = 42
np.random.seed(SEED)


train = pd.read_csv('/kaggle/input/c/jigsaw-agile-community-rules/train.csv')
test = pd.read_csv('/kaggle/input/c/jigsaw-agile-community-rules/test.csv')
sample = pd.read_csv('/kaggle/input/c/jigsaw-agile-community-rules/sample_submission.csv')

print(train.shape, test.shape)
train.head()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return text

for col in ['body','rule','positive_example_1','positive_example_2','negative_example_1','negative_example_2']:
    train[col] = train[col].fillna("").apply(clean_text)
    test[col] = test[col].fillna("").apply(clean_text)

def build_context(df):
    return (
        df['rule'] + ' [SEP] ' + df['subreddit'] + 
        ' [SEP] pos_ex1: ' + df['positive_example_1'] + 
        ' [SEP] neg_ex1: ' + df['negative_example_1']
    )

train['rule_context'] = build_context(train)
test['rule_context'] = build_context(test)


tfidf_context = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.95, max_features=80000)
tfidf_body = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.95, max_features=80000)

X_rule = tfidf_context.fit_transform(train['rule_context'])
X_rule_test = tfidf_context.transform(test['rule_context'])

X_body = tfidf_body.fit_transform(train['body'])
X_body_test = tfidf_body.transform(test['body'])

X = hstack([X_rule, X_body]).tocsr()
X_test = hstack([X_rule_test, X_body_test]).tocsr()
y = train['rule_violation'].values

print("TF-IDF feature matrix shape:", X.shape)


# Robust cosine-features cell (replace the old cosine block with this)
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import numpy as np
import gc

gc.collect()

# Build one TF-IDF vectorizer for ALL relevant text so vocab (n_features) matches
combined = pd.concat([
    train['body'].astype(str), test['body'].astype(str),
    train['rule_context'].astype(str), test['rule_context'].astype(str),
    train['positive_example_1'].astype(str), test['positive_example_1'].astype(str)
], ignore_index=True)

vec_all = TfidfVectorizer(max_features=30000)
vec_all.fit(combined)

# transform columns using the SAME vectorizer
body_train = vec_all.transform(train['body'].astype(str))
body_test  = vec_all.transform(test['body'].astype(str))

rule_train = vec_all.transform(train['rule_context'].astype(str))
rule_test  = vec_all.transform(test['rule_context'].astype(str))

pos_train  = vec_all.transform(train['positive_example_1'].astype(str))
pos_test   = vec_all.transform(test['positive_example_1'].astype(str))

print("Feature dims (should be identical):",
      body_train.shape[1], rule_train.shape[1], pos_train.shape[1])

# efficient row-wise cosine diagonal for sparse CSR matrices
def row_cosine_diag(A, B):
    # A, B: csr_matrix with same n_features
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"feature dim mismatch: {A.shape[1]} vs {B.shape[1]}")
    # numerator: elementwise row dot (sparse)
    num = A.multiply(B).sum(axis=1).A1  # shape (n_samples,)
    # norms
    normA = np.sqrt(A.multiply(A).sum(axis=1).A1)
    normB = np.sqrt(B.multiply(B).sum(axis=1).A1)
    denom = normA * normB
    denom[denom == 0] = 1e-9
    return (num / denom).reshape(-1, 1)

# compute diagonals (cosine between each comment and its corresponding rule / pos example)
cos_rule_train = row_cosine_diag(body_train, rule_train)
cos_rule_test  = row_cosine_diag(body_test, rule_test)

cos_pos_train  = row_cosine_diag(body_train, pos_train)
cos_pos_test   = row_cosine_diag(body_test, pos_test)

# stack into feature arrays and append to X / X_test
cos_features_train = np.hstack([cos_rule_train, cos_pos_train])
cos_features_test  = np.hstack([cos_rule_test, cos_pos_test])

X = hstack([X, csr_matrix(cos_features_train)]).tocsr()
X_test = hstack([X_test, csr_matrix(cos_features_test)]).tocsr()

print("Added cosine features. New shapes:")
print("X:", X.shape)
print("X_test:", X_test.shape)


scaler = StandardScaler()
rule_meta_train = pd.DataFrame({
    "rule_len_chars": train["rule"].str.len(),
    "rule_len_words": train["rule"].str.split().map(len)
})
rule_meta_test = pd.DataFrame({
    "rule_len_chars": test["rule"].str.len(),
    "rule_len_words": test["rule"].str.split().map(len)
})

rule_meta_scaled = scaler.fit_transform(rule_meta_train)
rule_meta_test_scaled = scaler.transform(rule_meta_test)

X = hstack([X, csr_matrix(rule_meta_scaled)]).tocsr()
X_test = hstack([X_test, csr_matrix(rule_meta_test_scaled)]).tocsr()

print("After adding meta features:", X.shape)


from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

SEED = 42
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof = np.zeros(len(train))
preds = np.zeros(len(test))

lgb_params = {
    'n_estimators': 3000,
    'learning_rate': 0.02,
    'num_leaves': 128,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'random_state': SEED,
    'n_jobs': -1
}

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print(f"FOLD {fold}")
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model = LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric='auc',
        callbacks=[early_stopping(100), log_evaluation(0)]  
    )
    oof[va_idx] = model.predict_proba(X_va)[:, 1]
    preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

print("Baseline CV AUC:", roc_auc_score(y, oof))


from xgboost import XGBClassifier

# XGBoost params
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'auc',
    'random_state': SEED,
    'n_jobs': -1
}

oof_lgb, oof_xgb = np.zeros(len(train)), np.zeros(len(train))
preds_lgb, preds_xgb = np.zeros(len(test)), np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print(f"FOLD {fold}")

    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # LightGBM
    lgb = LGBMClassifier(**lgb_params)
    lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc', callbacks=[early_stopping(100), log_evaluation(0)])
    oof_lgb[va_idx] = lgb.predict_proba(X_va)[:, 1]
    preds_lgb += lgb.predict_proba(X_test)[:, 1] / skf.n_splits

    # XGBoost
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=100, verbose=False)
    oof_xgb[va_idx] = xgb.predict_proba(X_va)[:, 1]
    preds_xgb += xgb.predict_proba(X_test)[:, 1] / skf.n_splits

# Blended predictions 
oof_blend = (oof_lgb + oof_xgb) / 2
preds_blend = (preds_lgb + preds_xgb) / 2

print("Blended CV AUC:", roc_auc_score(y, oof_blend))


# Create submission DataFrame
submission = pd.DataFrame({
    'row_id': test['row_id'],
    'rule_violation': preds_blend  
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")
submission.head()

