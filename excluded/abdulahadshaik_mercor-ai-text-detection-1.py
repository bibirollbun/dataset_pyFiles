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


# ============================================================
# Mercor AI Text Detection - Multi-model Stack:
# LightGBM + CatBoost + RandomForest + LogisticRegression
# Topic-grouped CV, embeddings, TF-IDF+SVD, rank-based NNLS blending
# Outputs only: submission.csv (probabilities)
# ============================================================

import os, gc, time, random, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from scipy.stats import rankdata
from scipy.optimize import nnls
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ----------------------------
# 1) Load data
# ----------------------------
def find_file(name):
    for root in [".", "/kaggle/input/mercor-ai-detection", "/kaggle/input"]:
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found")

train = pd.read_csv(find_file("train.csv"))
test  = pd.read_csv(find_file("test.csv"))
sample = pd.read_csv(find_file("sample_submission.csv"))
print("Shapes:", train.shape, test.shape)

# ----------------------------
# 2) Text feature helpers
# ----------------------------
def add_text_stats(df, text_col='answer'):
    s = df[text_col].fillna("").astype(str)
    df['char_len'] = s.str.len()
    df['word_len'] = s.str.split().apply(len)
    df['avg_word_len'] = df['char_len'] / (df['word_len'] + 1e-9)
    df['sentence_count'] = s.str.count(r'[.!?]') + 1
    df['comma_count'] = s.str.count(',')
    df['excl_count'] = s.str.count('!')
    df['question_count'] = s.str.count(r'\?')
    df['punct_ratio'] = (df['comma_count'] + df['excl_count'] + df['question_count']) / (df['char_len'] + 1e-9)
    df['upper_ratio'] = s.apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1e-9))
    df['digit_ratio'] = s.apply(lambda x: sum(1 for c in x if c.isdigit()) / (len(x) + 1e-9))
    return df

train = add_text_stats(train)
test  = add_text_stats(test)
num_features = ['char_len','word_len','avg_word_len','sentence_count','punct_ratio','upper_ratio','digit_ratio']

# ----------------------------
# 3) Topic target encode + safe label encode
# ----------------------------
from sklearn.model_selection import StratifiedKFold

def target_encode_oof(train_df, test_df, col, target, n_splits=5):
    oof = pd.Series(np.nan, index=train_df.index)
    test_te = np.zeros(len(test_df))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    groups = train_df['topic']
    for tr_idx, val_idx in skf.split(train_df, train_df[target], groups):
        tr = train_df.iloc[tr_idx]
        val = train_df.iloc[val_idx]
        m = 10
        global_mean = tr[target].mean()
        means = tr.groupby(col)[target].agg(['mean','count'])
        means['smt'] = (means['mean'] * means['count'] + global_mean * m) / (means['count'] + m)
        oof.iloc[val_idx] = val[col].map(means['smt']).fillna(global_mean)
    agg = train_df.groupby(col)[target].agg(['mean','count'])
    m = 10
    global_mean = train_df[target].mean()
    agg['smt'] = (agg['mean'] * agg['count'] + global_mean * m) / (agg['count'] + m)
    test_te = test_df[col].map(agg['smt']).fillna(global_mean).values
    return oof.astype(float), test_te

train['topic'] = train['topic'].astype(str)
test['topic']  = test['topic'].astype(str)
te_oof, te_test = target_encode_oof(train, test, col='topic', target='is_cheating')
train['topic_te'] = te_oof
test['topic_te'] = te_test

le = LabelEncoder()
all_topics = pd.concat([train['topic'], test['topic']]).astype(str)
le.fit(all_topics)
train['topic_le'] = le.transform(train['topic'].astype(str))
test['topic_le'] = le.transform(test['topic'].astype(str))

# ----------------------------
# 4) TF-IDF + SVD
# ----------------------------
print("Building TF-IDF...")
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=20000,
                        sublinear_tf=True, strip_accents='unicode', min_df=2, max_df=0.95)
tfidf.fit(pd.concat([train['answer'], test['answer']]))
train_tfidf = tfidf.transform(train['answer'])
test_tfidf  = tfidf.transform(test['answer'])

print("SVD compress...")
svd = TruncatedSVD(n_components=150, random_state=SEED)
svd.fit(train_tfidf)
train_svd = svd.transform(train_tfidf)
test_svd  = svd.transform(test_tfidf)

# ----------------------------
# 5) SentenceTransformer embeddings
# ----------------------------
!pip install -q sentence-transformers
from sentence_transformers import SentenceTransformer
print("Generating embeddings...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
train_emb = embed_model.encode(train['answer'].tolist(), batch_size=16, show_progress_bar=True)
test_emb  = embed_model.encode(test['answer'].tolist(), batch_size=16, show_progress_bar=True)

# small gaussian noise for regularization
train_emb += np.random.normal(0, 0.01, train_emb.shape)
test_emb  += np.random.normal(0, 0.01, test_emb.shape)

# ----------------------------
# 6) Combine & scale numeric + embeddings
# ----------------------------
from sklearn.preprocessing import StandardScaler
X_num_train = train[num_features + ['topic_te','topic_le']].fillna(0).values
X_num_test  = test[num_features + ['topic_te','topic_le']].fillna(0).values

scaler_num = StandardScaler().fit(np.vstack([X_num_train, X_num_test]))
X_num_train = scaler_num.transform(X_num_train)
X_num_test  = scaler_num.transform(X_num_test)

scaler_emb = StandardScaler().fit(np.vstack([train_emb, test_emb]))
train_emb_s = scaler_emb.transform(train_emb)
test_emb_s  = scaler_emb.transform(test_emb)

X_train_feat = np.hstack([train_svd, train_emb_s, X_num_train])
X_test_feat  = np.hstack([test_svd, test_emb_s, X_num_test])
y = train['is_cheating'].values.astype(int)
print("Final feature shape:", X_train_feat.shape)

# ----------------------------
# 7) CV setup (topic-grouped)
# ----------------------------
n_splits = 5
skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
groups = train['topic']

# containers for OOF and test predictions
oof_preds = {}
test_preds = {}

# Utility: rank-normalize predictions (for blending later)
def rank_normalize(arr):
    return rankdata(arr) / len(arr)

# ----------------------------
# 8) Train LightGBM
# ----------------------------
print("\nTraining LightGBM...")
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.0829037851263818,
    'num_leaves': 255,
    'feature_fraction': 0.8236629867393129,
    'bagging_fraction': 0.6191282775851725,
    'bagging_freq': 1,
    'lambda_l1': 0.29442545939152276,
    'lambda_l2': 0.04810589805993497,
    'min_data_in_leaf': 38,
    'verbosity': -1,
    'seed': SEED,
    'n_jobs': -1
}
oof_lgb = np.zeros(len(y))
pred_lgb = np.zeros(X_test_feat.shape[0])
for fold, (tr, va) in enumerate(skf.split(X_train_feat, y, groups), 1):
    dtrain = lgb.Dataset(X_train_feat[tr], label=y[tr])
    dvalid = lgb.Dataset(X_train_feat[va], label=y[va])
    model = lgb.train(lgb_params, dtrain, 10000,
                      valid_sets=[dtrain, dvalid],
                      callbacks=[lgb.early_stopping(stopping_rounds=100),
                                 lgb.log_evaluation(0)])
    oof_lgb[va] = model.predict(X_train_feat[va], num_iteration=model.best_iteration)
    pred_lgb += model.predict(X_test_feat, num_iteration=model.best_iteration) / n_splits
    print(f" LGB fold {fold} AUC:", roc_auc_score(y[va], oof_lgb[va]))
oof_preds['lgb'] = oof_lgb
test_preds['lgb'] = pred_lgb
print(" LGB OOF AUC:", roc_auc_score(y, oof_lgb))

# ----------------------------
# 9) Train CatBoost
# ----------------------------
print("\nTraining CatBoost...")
!pip install -q catboost
from catboost import CatBoostClassifier
oof_cat = np.zeros(len(y))
pred_cat = np.zeros(X_test_feat.shape[0])
for fold, (tr, va) in enumerate(skf.split(X_train_feat, y, groups), 1):
    X_tr, X_va = X_train_feat[tr], X_train_feat[va]
    y_tr, y_va = y[tr], y[va]
    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.06260985940204772,
        depth=6,
        l2_leaf_reg=8.902962171518904,
        border_count = 159,
        random_seed=SEED,
        eval_metric='AUC',
        verbose=False,
        use_best_model=True
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100, verbose=False)
    oof_cat[va] = model.predict_proba(X_va)[:,1]
    pred_cat += model.predict_proba(X_test_feat)[:,1] / n_splits
    print(f" CAT fold {fold} AUC:", roc_auc_score(y_va, oof_cat[va]))
oof_preds['cat'] = oof_cat
test_preds['cat'] = pred_cat
print(" CAT OOF AUC:", roc_auc_score(y, oof_cat))

# ----------------------------
# 10) Train RandomForest
# ----------------------------
print("\nTraining RandomForest...")
oof_rf = np.zeros(len(y))
pred_rf = np.zeros(X_test_feat.shape[0])
rf = RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=SEED)
for fold, (tr, va) in enumerate(skf.split(X_train_feat, y, groups), 1):
    X_tr, X_va = X_train_feat[tr], X_train_feat[va]
    y_tr, y_va = y[tr], y[va]
    clf = RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=SEED+fold)
    clf.fit(X_tr, y_tr)
    oof_rf[va] = clf.predict_proba(X_va)[:,1]
    pred_rf += clf.predict_proba(X_test_feat)[:,1] / n_splits
    print(f" RF fold {fold} AUC:", roc_auc_score(y_va, oof_rf[va]))
oof_preds['rf'] = oof_rf
test_preds['rf'] = pred_rf
print(" RF OOF AUC:", roc_auc_score(y, oof_rf))

# ----------------------------
# 11) Train Logistic Regression on TF-IDF (CV)
# ----------------------------
print("\nTraining LogisticRegression (TF-IDF) with topic-grouped CV...")
oof_lr = np.zeros(len(y))
pred_lr = np.zeros(test_tfidf.shape[0])
lr_model = LogisticRegression(max_iter=5000, solver='saga', C=2.0, random_state=SEED, n_jobs=-1)
for fold, (tr, va) in enumerate(skf.split(train_tfidf, y, groups), 1):
    X_tr = train_tfidf[tr]
    X_va = train_tfidf[va]
    y_tr = y[tr]
    lr = LogisticRegression(max_iter=5000, solver='saga', C=2.0, random_state=SEED+fold, n_jobs=-1)
    lr.fit(X_tr, y_tr)
    oof_lr[va] = lr.predict_proba(X_va)[:,1]
    pred_lr += lr.predict_proba(test_tfidf)[:,1] / n_splits
    print(f" LR fold {fold} AUC:", roc_auc_score(y[va], oof_lr[va]))
oof_preds['lr'] = oof_lr
test_preds['lr'] = pred_lr
print(" LR OOF AUC:", roc_auc_score(y, oof_lr))

# ----------------------------
# 12) Rank-normalize OOF & test preds and stack
# ----------------------------
print("\nRank-normalizing and stacking OOF preds for NNLS blending...")
model_keys = ['lgb','cat','rf','lr']
oof_stack = np.vstack([rank_normalize(oof_preds[k]) for k in model_keys]).T  # shape (n_samples, n_models)
test_stack = np.vstack([rank_normalize(test_preds[k]) for k in model_keys]).T  # (n_test, n_models)

# ----------------------------
# 13) Fit non-negative least squares (NNLS) on OOF to get blending weights
# ----------------------------
print("Fitting NNLS for non-negative blending weights...")
weights, _ = nnls(oof_stack, y)
if weights.sum() > 0:
    weights = weights / weights.sum()
else:
    # fallback to equal weights
    weights = np.ones(len(weights)) / len(weights)
blend_weights = dict(zip(model_keys, weights))
print("Blend weights (nnls, sum->1):", blend_weights)

# ----------------------------
# 14) Compute final blended predictions (rank-space -> use test_stack)
# ----------------------------
final_rank = test_stack.dot(weights)
# convert rank-space to probability-space (min-max normalize)
final_prob = (final_rank - final_rank.min()) / (final_rank.max() - final_rank.min() + 1e-9)

# Optional: compute blended OOF AUC for diagnostics
oof_rank_blend = oof_stack.dot(weights)
oof_prob_like = (oof_rank_blend - oof_rank_blend.min()) / (oof_rank_blend.max() - oof_rank_blend.min() + 1e-9)
print("Blended OOF ROC-AUC:", roc_auc_score(y, oof_prob_like))
print("Individual OOF AUCs:")
for k in model_keys:
    print(f"  {k}: {roc_auc_score(y, oof_preds[k]):.6f}")

# ----------------------------
# 15) Save probability-only submission
# ----------------------------
submission = pd.DataFrame({'id': test['id'], 'is_cheating': final_prob})
submission.to_csv('submission.csv', index=False)
print("\n✅ submission.csv saved (probabilities only).")
print("Blend weights used:", blend_weights)


