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




import os
import gc
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# Configuration
# ----------------------------
SEED = 42
NFOLDS = 5
WORD_TFIDF_MAX_FEATURES = 20000
CHAR_TFIDF_MAX_FEATURES = 20000
SVD_COMP_WORD = 100
SVD_COMP_CHAR = 50

LGB_PARAMS = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'seed': SEED,
}

CAT_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'eval_metric': 'AUC',
    'random_seed': SEED,
    'verbose': False,
    'early_stopping_rounds': 100
}

# Paths
TRAIN_PATH = "/kaggle/input/mercor-ai-detection/train.csv"
TEST_PATH = "/kaggle/input/mercor-ai-detection/test.csv"
SAMPLE_PATH = "/kaggle/input/mercor-ai-detection/sample_submission.csv"
OUTPUT_PATH = "submission.csv"

# ----------------------------
# Utility functions
# ----------------------------
def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_basic_feats(df, col="answer"):
    txt = df[col].fillna("").astype(str)
    d = {}
    d['len_chars'] = txt.str.len().astype(int)
    d['len_words'] = txt.str.split().apply(len).astype(int)
    d['avg_word_len'] = d['len_chars'] / (d['len_words'] + 1)
    d['q_marks'] = txt.str.count(r"\?")
    d['exclam'] = txt.str.count(r"!")
    d['commas'] = txt.str.count(r",")
    d['dots'] = txt.str.count(r"\.")
    d['digits'] = txt.str.count(r"\d")
    d['upper_count'] = txt.apply(lambda s: sum(1 for c in s if c.isupper()))
    d['upper_ratio'] = d['upper_count'] / (d['len_chars'] + 1)
    return pd.DataFrame(d)

def target_encode_kfold(train_col, target, test_col=None, n_splits=5, seed=SEED, smooth=20):
    df = pd.DataFrame({'col': train_col, 'target': target})
    global_mean = target.mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = pd.Series(index=df.index, dtype=float)
    for tr, val in skf.split(df, df['target']):
        tmp = df.iloc[tr].groupby('col')['target'].agg(['count','mean'])
        cnt = tmp['count']
        mu = tmp['mean']
        # smoothing
        weight = 1 / (1 + np.exp(-(cnt - smooth)))
        enc = global_mean * (1 - weight) + mu * weight
        mapping = enc.to_dict()
        out.iloc[val] = df['col'].iloc[val].map(mapping).fillna(global_mean)
    if test_col is None:
        return out.values, None
    # build full mapping
    full = df.groupby('col')['target'].agg(['count','mean'])
    cnt = full['count']
    mu = full['mean']
    weight = 1 / (1 + np.exp(-(cnt - smooth)))
    enc = global_mean * (1 - weight) + mu * weight
    mapping_full = enc.to_dict()
    return out.values, test_col.map(mapping_full).fillna(global_mean).values

# ----------------------------
# Main pipeline
# ----------------------------
def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sub0 = pd.read_csv(SAMPLE_PATH)

    # Preprocess
    for df in (train, test):
        df['answer'] = df['answer'].fillna("").astype(str)
        df['answer_clean'] = df['answer'].apply(clean_text)
        df['topic'] = df.get('topic', "").fillna("unknown").astype(str)

    # Basic features
    Xb_train = get_basic_feats(train, 'answer_clean')
    Xb_test = get_basic_feats(test, 'answer_clean')

    # Topic target-encoding
    te_tr, te_te = target_encode_kfold(train['topic'], train['is_cheating'], test['topic'], n_splits=NFOLDS, seed=SEED, smooth=50)
    Xb_train['topic_te'] = te_tr
    Xb_test['topic_te'] = te_te
    Xb_train['topic_len'] = train['topic'].str.len().fillna(0)
    Xb_test['topic_len'] = test['topic'].str.len().fillna(0)

    # TF-IDF + SVD
    all_text = pd.concat([train['answer_clean'], test['answer_clean']], axis=0).reset_index(drop=True)
    # Word-level
    wtf = TfidfVectorizer(ngram_range=(1,2), max_features=WORD_TFIDF_MAX_FEATURES, analyzer='word')
    W = wtf.fit_transform(all_text)
    # Char-level
    ctf = TfidfVectorizer(ngram_range=(3,6), max_features=CHAR_TFIDF_MAX_FEATURES, analyzer='char')
    C = ctf.fit_transform(all_text)

    n_tr = train.shape[0]
    # SVD
    svd_w = TruncatedSVD(n_components=SVD_COMP_WORD, random_state=SEED)
    W_svd = svd_w.fit_transform(W)
    svd_c = TruncatedSVD(n_components=SVD_COMP_CHAR, random_state=SEED)
    C_svd = svd_c.fit_transform(C)
    W_tr, W_te = W_svd[:n_tr, :], W_svd[n_tr:, :]
    C_tr, C_te = C_svd[:n_tr, :], C_svd[n_tr:, :]
    del W, C, W_svd, C_svd
    gc.collect()

    # Final matrices
    X_num_tr = Xb_train.reset_index(drop=True).values
    X_num_te = Xb_test.reset_index(drop=True).values
    # For tree models (dense)
    X_tr_tree = np.hstack([X_num_tr, W_tr, C_tr])
    X_te_tree = np.hstack([X_num_te, W_te, C_te])
    # For linear/logistic (scaled)
    scaler = StandardScaler()
    X_num_tr_s = scaler.fit_transform(X_num_tr)
    X_num_te_s = scaler.transform(X_num_te)
    X_tr_lin = np.hstack([X_num_tr_s, W_tr, C_tr])
    X_te_lin = np.hstack([X_num_te_s, W_te, C_te])

    y = train['is_cheating'].values

    # Prepare out-of-fold arrays
    oof_lgb = np.zeros(len(train))
    oof_cat = np.zeros(len(train))
    oof_log = np.zeros(len(train))
    pred_lgb = np.zeros(len(test))
    pred_cat = np.zeros(len(test))
    pred_log = np.zeros(len(test))

    skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr_tree, y), start=1):
        print(f"Fold {fold}/{NFOLDS}")
        Xtr_t, Xvl_t = X_tr_tree[tr_idx], X_tr_tree[val_idx]
        Xtr_l, Xvl_l = X_tr_lin[tr_idx], X_tr_lin[val_idx]
        ytr, yvl = y[tr_idx], y[val_idx]

        # LightGBM with callback early stopping
        dtrain = lgb.Dataset(Xtr_t, label=ytr)
        dvalid = lgb.Dataset(Xvl_t, label=yvl)
        bst = lgb.train(
            LGB_PARAMS,
            train_set=dtrain,
            valid_sets=[dvalid],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)],
        )
        oof_lgb[val_idx] = bst.predict(Xvl_t, num_iteration=bst.best_iteration)
        pred_lgb += bst.predict(X_te_tree, num_iteration=bst.best_iteration) / NFOLDS

        # CatBoost
        cb = CatBoostClassifier(**CAT_PARAMS)
        cb.fit(Xtr_t, ytr, eval_set=(Xvl_t, yvl), use_best_model=True)
        oof_cat[val_idx] = cb.predict_proba(Xvl_t)[:,1]
        pred_cat += cb.predict_proba(X_te_tree)[:,1] / NFOLDS

        # Logistic regression
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        clf.fit(Xtr_l, ytr)
        oof_log[val_idx] = clf.predict_proba(Xvl_l)[:,1]
        pred_log += clf.predict_proba(X_te_lin)[:,1] / NFOLDS

        # cleanup
        del dtrain, dvalid, bst, cb, clf
        gc.collect()

    # Evaluate
    auc_l = roc_auc_score(y, oof_lgb)
    auc_c = roc_auc_score(y, oof_cat)
    auc_r = roc_auc_score(y, oof_log)
    oof_ens = (oof_lgb + oof_cat + oof_log) / 3.0
    auc_ens = roc_auc_score(y, oof_ens)
    print("OOF AUC — LGB:", auc_l, "Cat:", auc_c, "Log:", auc_r, "Ensemble:", auc_ens)

    # Final predictions
    final_pred = (pred_lgb + pred_cat + pred_log) / 3.0
    final_pred = np.clip(final_pred, 1e-6, 1 - 1e-6)

    sub = pd.DataFrame({
        'id': test['id'],
        'is_cheating': final_pred
    })
    sub.to_csv(OUTPUT_PATH, index=False)
    print("Saved submission:", OUTPUT_PATH)

if __name__ == "__main__":
    main()


