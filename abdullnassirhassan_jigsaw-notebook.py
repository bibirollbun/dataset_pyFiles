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
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import lightgbm as lgb


# Load data
train_data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')


import emoji

def clean_text_adv(text):
    text = str(text).lower()
    text = emoji.demojize(text)  # convert emojis to text
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    text = re.sub(r'[^a-z0-9\s!?]', ' ', text)  # keep ! and ?
    text = re.sub(r'\s+', ' ', text).strip()
    return text

for df in [train_data, test_data]:
    df['body_clean'] = df['body'].apply(clean_text)
    df['rule_clean'] = df['rule'].apply(clean_text)
    df['combined_clean'] = df['body_clean'] + ' [SEP] ' + df['rule_clean']


from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

text = " ".join(train_data['rule_clean'])
wordcloud = WordCloud(width=800, height=400, background_color='white',
                      stopwords=ENGLISH_STOP_WORDS).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Rule Descriptions')
plt.show()


for df in [train_data, test_data]:
    df['body_length'] = df['body'].str.len()
    df['num_words'] = df['body'].str.split().apply(len)
    df['num_exclamations'] = df['body'].str.count('!')
    df['num_questions'] = df['body'].str.count('\?')
    df['num_upper'] = df['body'].str.count(r'[A-Z]')
    df['caps_ratio'] = df['num_upper'] / (df['num_words'] + 1e-5)
    df['rule_length'] = df['rule'].str.len()
    df['rule_words'] = df['rule'].str.split().apply(len)

meta_features = ['body_length', 'num_words', 'num_exclamations', 'num_questions', 'caps_ratio', 'rule_length', 'rule_words']
scaler = StandardScaler()
X_meta = scaler.fit_transform(train_data[meta_features])
X_meta_test = scaler.transform(test_data[meta_features])


# TF-IDF (word + char)
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), stop_words='english', analyzer='word')
X_tfidf = tfidf.fit_transform(train_data['combined_clean'])
X_tfidf_test = tfidf.transform(test_data['combined_clean'])

# Character-level TF-IDF
tfidf_char = TfidfVectorizer(max_features=20000, ngram_range=(2,5), analyzer='char')
X_char = tfidf_char.fit_transform(train_data['combined_clean'])
X_char_test = tfidf_char.transform(test_data['combined_clean'])

# Cosine similarity between body and rule
tfidf_sep = TfidfVectorizer(max_features=30000, ngram_range=(1,2), stop_words='english')
combined_texts = train_data['body_clean'].tolist() + train_data['rule_clean'].tolist() + \
                 test_data['body_clean'].tolist() + test_data['rule_clean'].tolist()
tfidf_sep.fit(combined_texts)

train_body_vecs = tfidf_sep.transform(train_data['body_clean'])
train_rule_vecs = tfidf_sep.transform(train_data['rule_clean'])
test_body_vecs = tfidf_sep.transform(test_data['body_clean'])
test_rule_vecs = tfidf_sep.transform(test_data['rule_clean'])

train_similarity = cosine_similarity(train_body_vecs, train_rule_vecs).diagonal().reshape(-1,1)
test_similarity = cosine_similarity(test_body_vecs, test_rule_vecs).diagonal().reshape(-1,1)


# Final Feature Matrix
X_train = hstack([X_tfidf, X_char, X_meta, train_similarity]).tocsr()
X_test = hstack([X_tfidf_test, X_char_test, X_meta_test, test_similarity]).tocsr()
y = train_data['rule_violation']


# LightGBM Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(X_train.shape[0])
test_preds = np.zeros(X_test.shape[0])

params = {
    'objective':'binary',
    'metric':'auc',
    'boosting_type':'gbdt',
    'learning_rate':0.01,
    'num_leaves':256,
    'feature_fraction':0.8,
    'bagging_fraction':0.8,
    'bagging_freq':5,
    'lambda_l1':1.0,
    'lambda_l2':1.0,
    'verbose':-1,
    'random_state':42}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    print(f"Fold {fold+1}")
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    train_set = lgb.Dataset(X_tr, y_tr)
    val_set = lgb.Dataset(X_val, y_val)
    
    model = lgb.train(params,train_set,
    valid_sets=[val_set],num_boost_round=5000,
    callbacks=[lgb.early_stopping(stopping_rounds=100),
              lgb.log_evaluation(period=100)])
    
    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits


auc_score = roc_auc_score(y, oof_preds)
print(f"\nOOF AUC = {auc_score:.5f}")


submission = pd.DataFrame({"row_id": test_data["row_id"], "rule_violation": test_preds})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved")

