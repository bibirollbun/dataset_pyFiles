import os
import gc
import re
import json
import time
from pathlib import Path


import numpy as np
import pandas as pd


from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


import lightgbm as lgb


# Text / NLP helpers
import nltk
nltk.download('punkt')


# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


train = pd.read_csv('/content/train.csv')
test = pd.read_csv('/content/test.csv')
sub = pd.read_csv('/content/sample_submission.csv')


print('\nTrain shape:', train.shape)
print('Test shape:', test.shape)
print('\nTrain columns:', train.columns.tolist())


# Quick head
train.head()


def basic_text_stats(df, text_col='answer'):
    df = df.copy()
    df['char_count'] = df[text_col].astype(str).apply(len)
    df['word_count'] = df[text_col].astype(str).apply(lambda x: len(x.split()))
    df['sentence_count'] = df[text_col].astype(str).apply(lambda x: len(nltk.tokenize.sent_tokenize(x)))
    return df[['char_count','word_count','sentence_count']]


if 'is_cheating' in train.columns:
  print('\nLabel distribution:')
  print(train['is_cheating'].value_counts(normalize=True))


import nltk
import os

# Set NLTK data path explicitly
nltk_data_path = '/root/nltk_data'
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)

print('\nTrain text length stats:')
print(basic_text_stats(train).describe())


if 'topic' in train.columns:
  print('\nTop topics (train):')
  print(train['topic'].value_counts().head(10))


print('\nSample negative (is_cheating=0):')
print(train[train['is_cheating']==0]['answer'].head(2).tolist())
print('\nSample positive (is_cheating=1):')
print(train[train['is_cheating']==1]['answer'].head(2).tolist())


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Separate cheating and non-cheating texts
cheating_texts = ' '.join(train[train['is_cheating'] == 1]['answer'])
non_cheating_texts = ' '.join(train[train['is_cheating'] == 0]['answer'])

# Generate word clouds
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Non-cheating word cloud
wordcloud1 = WordCloud(width=800, height=400, background_color='white',
                       max_words=100, colormap='viridis').generate(non_cheating_texts)
ax1.imshow(wordcloud1, interpolation='bilinear')
ax1.set_title('Most Common Words in Human Writing', fontsize=14)
ax1.axis('off')

# Cheating word cloud
wordcloud2 = WordCloud(width=800, height=400, background_color='white',
                       max_words=100, colormap='plasma').generate(cheating_texts)
ax2.imshow(wordcloud2, interpolation='bilinear')
ax2.set_title('Most Common Words in AI/Copied Writing', fontsize=14)
ax2.axis('off')

plt.tight_layout()
plt.show()


from textstat import flesch_reading_ease, flesch_kincaid_grade
import seaborn as sns

# Calculate readability scores
train['reading_ease'] = train['answer'].apply(flesch_reading_ease)
train['kincaid_grade'] = train['answer'].apply(flesch_kincaid_grade)

# Create comprehensive readability visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Reading Ease Distribution
sns.histplot(data=train, x='reading_ease', hue='is_cheating',
             bins=30, kde=True, ax=ax1, element='step')
ax1.set_title('Reading Ease Score Distribution')
ax1.set_xlabel('Flesch Reading Ease (Higher = Easier)')
ax1.legend(['Cheating', 'Human'])

# Grade Level Distribution
sns.boxplot(data=train, x='is_cheating', y='kincaid_grade', ax=ax2)
ax2.set_title('Grade Level by Cheating Status')
ax2.set_xlabel('Is Cheating (0=Human, 1=AI/Copied)')
ax2.set_ylabel('Kincaid Grade Level')

# Reading Ease vs Grade Level Scatter
sns.scatterplot(data=train, x='reading_ease', y='kincaid_grade',
                hue='is_cheating', alpha=0.6, ax=ax3)
ax3.set_title('Reading Ease vs Grade Level')
ax3.set_xlabel('Reading Ease')
ax3.set_ylabel('Grade Level')

# Combined readability score
train['readability_ratio'] = train['reading_ease'] / train['kincaid_grade']
sns.violinplot(data=train, x='is_cheating', y='readability_ratio', ax=ax4)
ax4.set_title('Readability Ratio (Ease/Grade)')
ax4.set_xlabel('Is Cheating (0=Human, 1=AI/Copied)')

plt.tight_layout()
plt.show()


# Calculate personalization features
first_person = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours']
positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'like', 'enjoy']
negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'poor']
concrete_words = ['table', 'chair', 'car', 'house', 'book', 'phone', 'computer', 'food', 'water']

train['first_person_freq'] = train['answer'].apply(
    lambda x: sum([x.lower().count(pronoun) for pronoun in first_person]) / len(x.split()) if len(x.split()) > 0 else 0
)
train['positive_freq'] = train['answer'].apply(
    lambda x: sum([x.lower().count(word) for word in positive_words]) / len(x.split()) if len(x.split()) > 0 else 0
)
train['negative_freq'] = train['answer'].apply(
    lambda x: sum([x.lower().count(word) for word in negative_words]) / len(x.split()) if len(x.split()) > 0 else 0
)
train['concrete_freq'] = train['answer'].apply(
    lambda x: sum([x.lower().count(word) for word in concrete_words]) / len(x.split()) if len(x.split()) > 0 else 0
)

# Create personalization dashboard
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# First Person Pronoun Usage
sns.boxplot(data=train, x='is_cheating', y='first_person_freq', ax=ax1)
ax1.set_title('First Person Pronoun Usage')
ax1.set_ylabel('Frequency')
ax1.set_xlabel('Is Cheating (0=Human, 1=AI/Copied)')

# Emotional Words
train['emotional_freq'] = train['positive_freq'] + train['negative_freq']
sns.violinplot(data=train, x='is_cheating', y='emotional_freq', ax=ax2)
ax2.set_title('Emotional Words Usage')
ax2.set_ylabel('Frequency')
ax2.set_xlabel('Is Cheating (0=Human, 1=AI/Copied)')

# Concrete vs Abstract Language
sns.scatterplot(data=train, x='concrete_freq', y='first_person_freq',
                hue='is_cheating', alpha=0.6, ax=ax3)
ax3.set_title('Concrete Words vs Personal Pronouns')
ax3.set_xlabel('Concrete Words Frequency')
ax3.set_ylabel('First Person Frequency')

# Personalization Score
train['personalization_score'] = (train['first_person_freq'] +
                                    train['emotional_freq'] +
                                    train['concrete_freq'])
sns.kdeplot(data=train, x='personalization_score', hue='is_cheating',
            fill=True, ax=ax4)
ax4.set_title('Personalization Score Distribution')
ax4.set_xlabel('Personalization Score')
ax4.set_ylabel('Density')

plt.tight_layout()
plt.show()


def minimal_clean(text):
    if pd.isna(text):
        return ''
    text = str(text)
    # normalize unicode and whitespace
    text = text.replace('\r',' ').replace('\n',' ').strip()
    text = re.sub(r'\s+', ' ', text)
    return text


train['answer'] = train['answer'].apply(minimal_clean)
test['answer'] = test['answer'].apply(minimal_clean)


# Drop exact duplicate rows in training (optional; keep track)
before = len(train)
train = train.drop_duplicates(subset=['answer']).reset_index(drop=True)
after = len(train)
print(f'\nDropped {before-after} exact duplicate answers from train')


import math
from collections import Counter


# Simple stylometry & doc stats
def doc_features(text):
    words = text.split()
    n_words = len(words)
    chars = len(text)
    sentences = nltk.tokenize.sent_tokenize(text)
    n_sents = max(1, len(sentences))
    avg_word_len = np.mean([len(w) for w in words]) if n_words>0 else 0
    unique_words = len(set(words))
    ttr = unique_words / max(1, n_words)
    punct_counts = Counter(ch for ch in text if ch in '.,;:!?"\'')
    uppercase_words = sum(1 for w in words if w.isupper())
    return {
    'char_count': chars,
    'word_count': n_words,
    'sent_count': n_sents,
    'avg_word_len': avg_word_len,
    'type_token_ratio': ttr,
    'upper_frac': uppercase_words / max(1, n_words),
    'comma_frac': text.count(',') / max(1, chars)
    }


# Apply to datasets
train_feats = pd.DataFrame(train['answer'].apply(doc_features).tolist())
test_feats = pd.DataFrame(test['answer'].apply(doc_features).tolist())


# Merge back
train = pd.concat([train.reset_index(drop=True), train_feats.reset_index(drop=True)], axis=1)
test = pd.concat([test.reset_index(drop=True), test_feats.reset_index(drop=True)], axis=1)


print('\nFeature sample:')
train[['char_count','word_count','type_token_ratio']].describe()


# Parameters - tune for your data/compute
TF_MAX_FEATURES = 100000
TF_NGRAMS = (1,3)
MAX_FOLDS = 5


tfidf = TfidfVectorizer(ngram_range=TF_NGRAMS, max_features=TF_MAX_FEATURES, analyzer='word', min_df=2)
X_tfidf = tfidf.fit_transform(train['answer'])
X_test_tfidf = tfidf.transform(test['answer'])


y = train['is_cheating'].values


skf = StratifiedKFold(n_splits=MAX_FOLDS, shuffle=True, random_state=RANDOM_SEED)


oof_preds_tfidf = np.zeros(len(train))
test_preds_tfidf = np.zeros(len(test))


for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tfidf, y)):
    print(f'Fold {fold+1}/{MAX_FOLDS}')
    X_tr = X_tfidf[tr_idx]
    X_val = X_tfidf[val_idx]
    y_tr = y[tr_idx]
    y_val = y[val_idx]


    clf = LogisticRegression(max_iter=2000, C=3.0, class_weight='balanced', solver='saga', random_state=RANDOM_SEED)
    clf.fit(X_tr, y_tr)
    oof_preds_tfidf[val_idx] = clf.predict_proba(X_val)[:,1]
    test_preds_tfidf += clf.predict_proba(X_test_tfidf)[:,1] / MAX_FOLDS


print('\nTF-IDF OOF AUC:', roc_auc_score(y, oof_preds_tfidf))


# Save OOF preds for stacking
train['oof_tfidf'] = oof_preds_tfidf
test['pred_tfidf'] = test_preds_tfidf


feature_cols = ['char_count','word_count','sent_count','avg_word_len','type_token_ratio','upper_frac','comma_frac']


# Optionally reduce TF-IDF with TruncatedSVD and include as features (commented out for speed)
# from sklearn.decomposition import TruncatedSVD
# svd = TruncatedSVD(n_components=100, random_state=RANDOM_SEED)
# X_svd = svd.fit_transform(X_tfidf)
# for i in range(X_svd.shape[1]):
# train[f'svd_{i}'] = X_svd[:,i]


X = train[feature_cols].values
X_test = test[feature_cols].values


oof_preds_lgb = np.zeros(len(train))
test_preds_lgb = np.zeros(len(test))


for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f'LGB Fold {fold+1}/{MAX_FOLDS}')
    X_tr, X_val_fold = X[tr_idx], X[val_idx]
    y_tr, y_val_fold = y[tr_idx], y[val_idx]


    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val_fold, label=y_val_fold, reference=dtrain)


    params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'seed': RANDOM_SEED,
    'verbosity': -1
    }
    model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=1000,
                      callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True),
                                 lgb.log_evaluation(period=100)])
    oof_preds_lgb[val_idx] = model.predict(X_val_fold)
    test_preds_lgb += model.predict(X_test) / MAX_FOLDS

# Store the validation data from the last fold
X_val = X_val_fold
y_val = y_val_fold


print('\nLGB OOF AUC on engineered features:', roc_auc_score(y, oof_preds_lgb))


train['oof_lgb'] = oof_preds_lgb
test['pred_lgb'] = test_preds_lgb


from sklearn.metrics import roc_auc_score

# Predict probabilities for validation data
y_pred = model.predict(X_val)

# Compute ROC-AUC
auc = roc_auc_score(y_val, y_pred)
print(f"Validation ROC-AUC: {auc:.4f}")


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    # Create a new LightGBM Dataset for training and validation in each fold
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

    # Define LightGBM parameters (can be the same as in the previous cell)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'seed': RANDOM_SEED,
        'verbosity': -1
    }

    # Train a new model instance for this fold
    model_fold = lgb.train(params, dtrain, valid_sets=[dvalid], num_boost_round=1000,
                           callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]) # verbose=False to reduce output during CV

    # Predict probabilities on the validation set for this fold
    y_pred = model_fold.predict(X_valid) # Use .predict for probabilities

    # Compute ROC-AUC for this fold
    auc = roc_auc_score(y_valid, y_pred)
    auc_scores.append(auc)
    print(f"Fold {fold} AUC: {auc:.4f}")

print(f"\nMean ROC-AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")


meta_features = ['oof_tfidf','oof_lgb']
meta_X = train[meta_features].values
meta_y = train['is_cheating'].values


from sklearn.linear_model import LogisticRegression


meta_oof = np.zeros(len(train))
meta_test = np.zeros(len(test))


for fold, (tr_idx, val_idx) in enumerate(skf.split(meta_X, meta_y)):
    X_tr, X_val = meta_X[tr_idx], meta_X[val_idx]
    y_tr, y_val = meta_y[tr_idx], meta_y[val_idx]
    meta_clf = LogisticRegression(max_iter=1000)
    meta_clf.fit(X_tr, y_tr)
    meta_oof[val_idx] = meta_clf.predict_proba(X_val)[:,1]
    meta_test += meta_clf.predict_proba(test[ ['pred_tfidf','pred_lgb'] ].values)[:,1] / MAX_FOLDS


print('\nMeta OOF AUC:', roc_auc_score(meta_y, meta_oof))


train['oof_meta'] = meta_oof
test['pred_meta'] = meta_test


if 'topic' in train.columns:
    topic_aucs = []
    for t, grp in train.groupby('topic'):
        if len(grp) < 20:
            continue
        try:
            auc = roc_auc_score(grp['is_cheating'], grp['oof_meta'])
            topic_aucs.append((t, auc, len(grp)))
        except Exception:
            continue
    topic_aucs = sorted(topic_aucs, key=lambda x: x[1])
    print('\nSample per-topic AUC (low->high):')
    print(topic_aucs[:10])


# Adversarial validation (train vs test) using tfidf quick check
from sklearn.model_selection import train_test_split
adv_X = np.vstack([X_tfidf.toarray(), X_test_tfidf.toarray()]) if X_tfidf.shape[0] < 5000 else None
# If dataset too large, skip or use sampled rows for adv validation
if adv_X is not None:
    adv_y = np.concatenate([np.zeros(X_tfidf.shape[0]), np.ones(X_test_tfidf.shape[0])])
    adv_clf = LogisticRegression(max_iter=500)
    adv_clf.fit(adv_X, adv_y)
    adv_pred = adv_clf.predict_proba(adv_X)[:,1]
    try:
        print('\nAdversarial AUC (train vs test):', roc_auc_score(adv_y, adv_pred))
    except Exception:
        pass
else:
    print('\nAdv validation skipped due to size - sample your data if you want this check')


submission = test[['id', 'pred_meta']].copy()
submission.rename(columns={'pred_meta': 'is_cheating'}, inplace=True)
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")

