import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import re
from scipy.sparse import hstack


train = pd.read_csv('/kaggle/input/mercor-ai-detection/train.csv')
test = pd.read_csv('/kaggle/input/mercor-ai-detection/test.csv')


def extract_features(df):
    features = pd.DataFrame()

    features['len_answer'] = df['answer'].str.len()
    features['word_count'] = df['answer'].str.split().str.len()
    features['avg_word_len'] = features['len_answer'] / (features['word_count'] + 1)

    features['sentence_count'] = df['answer'].str.count(r'[.!?]+')
    features['avg_sentence_len'] = features['word_count'] / (features['sentence_count'] + 1)

    features['comma_count'] = df['answer'].str.count(',')
    features['quote_count'] = df['answer'].str.count('"')
    features['exclamation_count'] = df['answer'].str.count('!')
    features['question_count'] = df['answer'].str.count(r'\?')

    features['capital_ratio'] = df['answer'].apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))
    features['title_case_count'] = df['answer'].apply(lambda x: sum(1 for word in x.split() if word and word[0].isupper()))
    
    features['paragraph_count'] = df['answer'].str.count('\n\n') + 1

    features['unique_word_ratio'] = df['answer'].apply(
        lambda x: len(set(x.lower().split())) / (len(x.split()) + 1)
    )

    features['special_char_count'] = df['answer'].apply(lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', x)))
    features['digit_count'] = df['answer'].str.count(r'\d')
    
    features['newline_count'] = df['answer'].str.count('\n')
    features['double_space_count'] = df['answer'].str.count('  ')

    features['lexical_diversity'] = df['answer'].apply(
        lambda x: len(set(x.lower().split())) / (len(x.lower().split()) + 1) if len(x.split()) > 0 else 0
    )
    
    features['has_bullet_points'] = df['answer'].str.contains(r'^\s*[-*â€¢]', regex=True).astype(int)
    features['has_numbered_list'] = df['answer'].str.contains(r'^\s*\d+\.', regex=True).astype(int)

    features['emoji_count'] = df['answer'].apply(lambda x: len(re.findall(r'[ğŸ˜€-ğŸ™�ğŸŒ€-ğŸ—¿ğŸš€-ğŸ›¿]', x)))

    features['word_repetition'] = df['answer'].apply(
        lambda x: len(x.split()) - len(set(x.lower().split()))
    )
    return features


train_features = extract_features(train)
test_features = extract_features(test)


# TF - IDF
tfidf_word = TfidfVectorizer(
    max_features = 3000,
    ngram_range = (1, 3),
    min_df = 2,
    max_df = 0.95,
    sublinear_tf = True
)

tfidf_char = TfidfVectorizer(
    analyzer = 'char',
    ngram_range = (2, 5),
    max_features = 2000,
    min_df = 2,
    max_df = 0.95
)


train_tfidf_word = tfidf_word.fit_transform(train['answer'])
test_tfidf_word = tfidf_word.transform(test['answer'])


train_tfidf_char = tfidf_char.fit_transform(train['answer'])
test_tfidf_char = tfidf_char.transform(test['answer'])


X_train = hstack([
    train_features.values,
    train_tfidf_word,
    train_tfidf_char
]).tocsr()


X_test = hstack([
    test_features.values,
    test_tfidf_word,
    test_tfidf_char
]).tocsr()


y_train = train['is_cheating'].values


skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)


oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))


for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"Fold {fold + 1} / 5")
    
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X_tr, label = y_tr)
    val_data = lgb.Dataset(X_val, label = y_val, reference = train_data)
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round = 1000,
        valid_sets = [val_data],
        callbacks = [lgb.early_stopping(stopping_rounds = 50), lgb.log_evaluation(100)]
    )
    
    oof_preds[val_idx] = model.predict(X_val, num_iteration = model.best_iteration)
    test_preds += model.predict(X_test, num_iteration = model.best_iteration) / 5
    
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold + 1} AUC: {fold_auc:.6f}")


cv_auc = roc_auc_score(y_train, oof_preds)
print(f"\nOverall CV AUC: {cv_auc:.6f}")


submission = pd.DataFrame({
    'id': test['id'],
    'is_cheating': test_preds
})

submission.to_csv('submission.csv', index = False)




