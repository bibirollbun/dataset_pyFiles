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
import re
import string
import gc
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import nltk


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


data_folder = '/kaggle/input/jigsaw-agile-community-rules/'
train = pd.read_csv(os.path.join(data_folder,"train.csv"))
test = pd.read_csv(os.path.join(data_folder,"test.csv"))


test_ids = test['row_id'].values

# 2. Feature Engineer
def create_features(df, train_df=None):
    df['body'] = df['body'].fillna('')
    
    
    df['body_len'] = df['body'].apply(len)
    df['word_count'] = df['body'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['body'].apply(lambda x: len(str(x).replace(" ", "")))
    df['avg_word_length'] = df['char_count'] / (df['word_count'] + 1e-5)
    df['has_url'] = df['body'].str.contains(r'http[s]?://|www\.').astype(int)
    
    
    df['exclamation_count'] = df['body'].apply(lambda x: x.count('!'))
    df['question_count'] = df['body'].apply(lambda x: x.count('?'))
    df['uppercase_ratio'] = df['body'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )
    df['digit_count'] = df['body'].apply(lambda x: sum(1 for c in x if c.isdigit()))
    
    
    if train_df is not None:
        subreddit_counts = train_df['subreddit'].value_counts(normalize=True).to_dict()
        rule_counts = train_df['rule'].value_counts(normalize=True).to_dict()
    else:
        subreddit_counts = df['subreddit'].value_counts(normalize=True).to_dict()
        rule_counts = df['rule'].value_counts(normalize=True).to_dict()
    
    df['subreddit_freq'] = df['subreddit'].map(subreddit_counts).fillna(0)
    df['rule_freq'] = df['rule'].map(rule_counts).fillna(0)
    
    
    spam_words = ['buy', 'sell', 'discount', 'offer', 'click', 'promo', 'code', 'deal', 'win', 'free']
    df['contains_spam_word'] = df['body'].apply(
        lambda x: 1 if any(word in str(x).lower() for word in spam_words) else 0
    )
    
    
    rule_keywords = ['spam', 'advertis', 'solicit', 'promot', 'market', 'sell']
    df['matches_rule_keywords'] = df.apply(
        lambda row: int(any(kw in str(row['body']).lower() for kw in rule_keywords)), 
        axis=1
    )
    
    return df


train = create_features(train)
test = create_features(test, train)


stop_words = set(nltk.corpus.stopwords.words('english'))
ps = SnowballStemmer('english')

def advanced_preprocessing(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' URL_TOKEN ', text)  # 保留URL模式
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
   
    tokens = word_tokenize(text)
    filtered_tokens = [ps.stem(token) for token in tokens if token not in stop_words and len(token) > 1]
    
    return ' '.join(filtered_tokens)

train['processed_body'] = train['body'].apply(advanced_preprocessing)
test['processed_body'] = test['body'].apply(advanced_preprocessing)

#  TF-IDF
tfidf_vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 3),
    min_df=5,
    max_df=0.85
)
tfidf_train = tfidf_vectorizer.fit_transform(train['processed_body'])
tfidf_test = tfidf_vectorizer.transform(test['processed_body'])

# Count
count_vectorizer = CountVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=10,
    max_df=0.9
)
count_train = count_vectorizer.fit_transform(train['processed_body'])
count_test = count_vectorizer.transform(test['processed_body'])


rule_vectorizer = CountVectorizer(max_features=100)
rule_train = rule_vectorizer.fit_transform(train['rule'])
rule_test = rule_vectorizer.transform(test['rule'])

subreddit_vectorizer = CountVectorizer(max_features=200)
subreddit_train = subreddit_vectorizer.fit_transform(train['subreddit'])
subreddit_test = subreddit_vectorizer.transform(test['subreddit'])

def create_bow_features(texts, vectorizer=None):
    if vectorizer is None:
        vectorizer = CountVectorizer(max_features=1000)
        return vectorizer.fit_transform(texts), vectorizer
    else:
        return vectorizer.transform(texts), vectorizer

bow_train, bow_vectorizer = create_bow_features(train['processed_body'])
bow_test, _ = create_bow_features(test['processed_body'], bow_vectorizer)


meta_features = [
    'body_len', 'word_count', 'has_url', 'exclamation_count', 
    'question_count', 'subreddit_freq', 'rule_freq',
    'contains_spam_word', 'matches_rule_keywords'
]

train_meta = csr_matrix(train[meta_features].astype(float).values)
test_meta = csr_matrix(test[meta_features].astype(float).values)

X_train = hstack([
    tfidf_train, 
    count_train,
    rule_train,
    subreddit_train,
    train_meta,
    bow_train
], format='csr')

X_test = hstack([
    tfidf_test, 
    count_test,
    rule_test,
    subreddit_test,
    test_meta,
    bow_test
], format='csr')

y_train = train['rule_violation'].values

del tfidf_train, tfidf_test, count_train, count_test
del rule_train, rule_test, subreddit_train, subreddit_test
del train_meta, test_meta, bow_train, bow_test
gc.collect()

#  SVD
svd = TruncatedSVD(n_components=500, random_state=42)
X_train = svd.fit_transform(X_train)
X_test = svd.transform(X_test)



test_preds_lgb = []
test_preds_lr = []
oof_preds_lgb = np.zeros(X_train.shape[0])
oof_preds_lr = np.zeros(X_train.shape[0])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n====== Fold {fold+1} ======")
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    
    #LightGBM
    params_lgb = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': 5,
        'min_child_samples': 50,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.7,
        'reg_lambda': 0.7,
        'random_state': 42 + fold,
        'n_jobs': -1,
        'verbose': -1
    }
    
    lgb_model = lgb.LGBMClassifier(**params_lgb, n_estimators=5000)
    
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
        ]
    )
    
    
    val_preds_lgb = lgb_model.predict_proba(X_val)[:, 1]
    oof_preds_lgb[val_idx] = val_preds_lgb
    fold_auc_lgb = roc_auc_score(y_val, val_preds_lgb)
    print(f"LightGBM Fold {fold+1} AUC: {fold_auc_lgb:.5f}")
    

    test_preds_lgb.append(lgb_model.predict_proba(X_test)[:, 1])
    
    # LR
    lr_model = LogisticRegression(
        C=0.01,
        penalty='l1',
        solver='liblinear',
        max_iter=1000,
        random_state=42 + fold
    )
    
    calibrated_lr = CalibratedClassifierCV(lr_model, method='isotonic', cv=3)
    calibrated_lr.fit(X_tr, y_tr)
    
    val_preds_lr = calibrated_lr.predict_proba(X_val)[:, 1]
    oof_preds_lr[val_idx] = val_preds_lr
    fold_auc_lr = roc_auc_score(y_val, val_preds_lr)
    print(f"LogisticRegression Fold {fold+1} AUC: {fold_auc_lr:.5f}")
    
    test_preds_lr.append(calibrated_lr.predict_proba(X_test)[:, 1])

lgb_oof_auc = roc_auc_score(y_train, oof_preds_lgb)
lr_oof_auc = roc_auc_score(y_train, oof_preds_lr)
print(f"\nLightGBM OOF AUC: {lgb_oof_auc:.5f}")
print(f"LogisticRegression OOF AUC: {lr_oof_auc:.5f}")

final_test_preds_lgb = np.mean(test_preds_lgb, axis=0)
final_test_preds_lr = np.mean(test_preds_lr, axis=0)

# OOF AUC
lgb_weight = lgb_oof_auc / (lgb_oof_auc + lr_oof_auc)
lr_weight = lr_oof_auc / (lgb_oof_auc + lr_oof_auc)
final_preds = (lgb_weight * final_test_preds_lgb + lr_weight * final_test_preds_lr)

def apply_post_processing(row, pred):
    body_len = row['body_len']
    exclamation_count = row['exclamation_count']
    has_url = row['has_url']
    contains_spam_word = row['contains_spam_word']
    
    if body_len < 30 and exclamation_count >= 2:
        pred = min(pred * 1.5, 1.0)
    
    if body_len > 150 and has_url and contains_spam_word:
        pred = min(pred * 1.3, 1.0)
    
    if row['subreddit_freq'] > 0.1:
        pred = max(pred * 0.9, 0.0)
    
    return pred

test['final_pred'] = final_preds
test['final_pred_adjusted'] = test.apply(
    lambda row: apply_post_processing(row, row['final_pred']), axis=1
)

submission = pd.DataFrame({
    "row_id": test['row_id'],
    "rule_violation": test['final_pred_adjusted']
})
submission.to_csv("submission.csv", index=False)
print("\nSaved: submission.csv")

# 12. 输出特征重要性
if hasattr(lgb_model, 'feature_importances_'):
    # 创建特征名称列表
    feature_names = (
        [f'tfidf_{i}' for i in range(15000)] + 
        [f'count_{i}' for i in range(10000)] + 
        [f'rule_{i}' for i in range(100)] + 
        [f'subreddit_{i}' for i in range(200)] + 
        meta_features +
        [f'bow_{i}' for i in range(1000)]
    )
    
    # 只获取LightGBM模型的特征重要性
    importances = lgb_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    print("\nTop 30 Features:")
    for i in range(30):
        feat_name = feature_names[sorted_idx[i]] if sorted_idx[i] < len(feature_names) else f'feat_{sorted_idx[i]}'
        print(f"{feat_name}: {importances[sorted_idx[i]]}")




