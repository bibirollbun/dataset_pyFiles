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



!pip install -q sentence-transformers lightgbm xgboost catboost optuna



# %% [code]
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

print("âœ… All libraries imported successfully!")



# %% [code]
train = pd.read_csv("/kaggle/input/mercor-ai-detection/train.csv")
test = pd.read_csv("/kaggle/input/mercor-ai-detection/test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Target mean:", train['is_cheating'].mean())

# BoÅŸ deÄŸerleri dolduralÄ±m
train['answer'] = train['answer'].fillna("")
test['answer'] = test['answer'].fillna("")



# %% [code]
def extract_features(df):
    feats = pd.DataFrame()
    feats['text_length'] = df['answer'].str.len()
    feats['word_count'] = df['answer'].str.split().str.len()
    feats['avg_word_length'] = feats['text_length'] / (feats['word_count'] + 1)
    feats['sentence_count'] = df['answer'].str.count(r'[.!?]+')
    feats['avg_sentence_length'] = feats['word_count'] / (feats['sentence_count'] + 1)
    feats['comma_count'] = df['answer'].str.count(',')
    feats['period_count'] = df['answer'].str.count(r'\.')
    feats['punct_ratio'] = (feats['comma_count'] + feats['period_count']) / (feats['text_length'] + 1)
    feats['unique_words'] = df['answer'].apply(lambda x: len(set(x.split())))
    feats['ttr'] = feats['unique_words'] / (feats['word_count'] + 1)
    return feats

train_feats = extract_features(train)
test_feats = extract_features(test)
print("Feature shape:", train_feats.shape)



# %% [code]
# Fit sadece train'de yapÄ±lmalÄ±
tfidf_word = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,
    stop_words='english'
)

tfidf_char = TfidfVectorizer(
    max_features=1500,
    analyzer='char_wb',
    ngram_range=(3, 5),
    sublinear_tf=True
)

# Fit & Transform
tfidf_word.fit(train['answer'])
X_word_tr = tfidf_word.transform(train['answer'])
X_word_te = tfidf_word.transform(test['answer'])

tfidf_char.fit(train['answer'])
X_char_tr = tfidf_char.transform(train['answer'])
X_char_te = tfidf_char.transform(test['answer'])

print("TF-IDF word shape:", X_word_tr.shape)
print("TF-IDF char shape:", X_char_tr.shape)



# %% [code]
X_train = np.hstack([
    X_word_tr.toarray(),
    X_char_tr.toarray(),
    train_feats.values
])

X_test = np.hstack([
    X_word_te.toarray(),
    X_char_te.toarray(),
    test_feats.values
])

y_train = train['is_cheating'].values

print("âœ… Final feature dimensions:")
print("Train:", X_train.shape)
print("Test :", X_test.shape)



# %% [code]
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros((len(test), 4))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nðŸŒ€ Fold {fold+1}")
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    
    # LightGBM
    lgbm = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.02, max_depth=6,
        subsample=0.8, colsample_bytree=0.7, random_state=42
    )
    lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100, verbose=False)])
    val_pred = lgbm.predict_proba(X_val)[:, 1]
    test_fold_pred = lgbm.predict_proba(X_test)[:, 1]
    oof_preds[val_idx] = val_pred
    test_preds[:, 0] += test_fold_pred / skf.n_splits
    print("  LGB AUC:", roc_auc_score(y_val, val_pred))

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.02, max_depth=6,
        subsample=0.8, colsample_bytree=0.7, eval_metric='auc',
        tree_method='hist', random_state=42
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    val_pred = xgb_model.predict_proba(X_val)[:, 1]
    test_fold_pred = xgb_model.predict_proba(X_test)[:, 1]
    oof_preds[val_idx] += val_pred
    test_preds[:, 1] += test_fold_pred / skf.n_splits
    print("  XGB AUC:", roc_auc_score(y_val, val_pred))
    
    # CatBoost
    cat = CatBoostClassifier(iterations=800, learning_rate=0.03, depth=6, verbose=0)
    cat.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    val_pred = cat.predict_proba(X_val)[:, 1]
    test_fold_pred = cat.predict_proba(X_test)[:, 1]
    oof_preds[val_idx] += val_pred
    test_preds[:, 2] += test_fold_pred / skf.n_splits
    print("  CAT AUC:", roc_auc_score(y_val, val_pred))

    # Logistic Regression
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, C=0.5)
    lr.fit(X_tr_scaled, y_tr)
    val_pred = lr.predict_proba(X_val_scaled)[:, 1]
    test_fold_pred = lr.predict_proba(X_test_scaled)[:, 1]
    oof_preds[val_idx] += val_pred
    test_preds[:, 3] += test_fold_pred / skf.n_splits
    print("  LR AUC:", roc_auc_score(y_val, val_pred))



# %% [code]
# Ortalama veya aÄŸÄ±rlÄ±klÄ± ortalama
final_test_pred = test_preds.mean(axis=1)

submission = pd.DataFrame({
    "id": test["id"],
    "is_cheating": final_test_pred
})

submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv saved!")














