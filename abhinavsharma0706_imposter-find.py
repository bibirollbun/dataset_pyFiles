# Install the libraries (including spaCy)
!pip install textstat -q
!pip install spacy -q
!pip install lightgbm -q

# Download spaCy English model (only need to do once)
!python -m spacy download en_core_web_sm -q


# Imports
import pandas as pd
import numpy as np
import os
import glob
from tqdm.auto import tqdm
import textstat
import spacy
import string
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import urllib.error

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


class Config:
    BASE_PATH = "/kaggle/input/fake-or-real-the-impostor-hunt/data/"
    TRAIN_PATH = os.path.join(BASE_PATH, "train")
    TEST_PATH = os.path.join(BASE_PATH, "test")
    TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
    SUBMISSION_FILE = "submission.csv"
    N_SPLITS = 10
    RANDOM_STATE = 42
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 3000,
        'learning_rate': 0.01,
        'num_leaves': 20,
        'max_depth': 5,
        'seed': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
        'colsample_bytree': 0.7,
        'subsample': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }


def read_text_files_robust(df, path):
    texts_1, texts_2 = [], []
    all_dirs = glob.glob(os.path.join(path, 'article_*'))
    dir_map = {int(os.path.basename(p).replace('article_', '')): p for p in all_dirs}

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Reading files from {os.path.basename(path)}"):
        article_id = row['id']
        dir_path = dir_map.get(article_id)

        if dir_path:
            try:
                with open(os.path.join(dir_path, 'file_1.txt'), 'r', encoding='utf-8') as f:
                    texts_1.append(f.read())
            except FileNotFoundError:
                texts_1.append("")

            try:
                with open(os.path.join(dir_path, 'file_2.txt'), 'r', encoding='utf-8') as f:
                    texts_2.append(f.read())
            except FileNotFoundError:
                texts_2.append("")
        else:
            texts_1.append("")
            texts_2.append("")

    df['text_1'] = texts_1
    df['text_2'] = texts_2
    return df


def load_data(config):
    train_df = pd.read_csv(config.TRAIN_CSV)
    test_dirs = glob.glob(os.path.join(config.TEST_PATH, 'article_*'))
    if not test_dirs:
        raise FileNotFoundError(f"No 'article_*' directories found in {config.TEST_PATH}")
    test_ids = [int(os.path.basename(p).replace('article_', '')) for p in test_dirs]
    test_df = pd.DataFrame(sorted(test_ids), columns=['id'])

    train_df = read_text_files_robust(train_df, config.TRAIN_PATH)
    test_df = read_text_files_robust(test_df, config.TEST_PATH)

    return train_df, test_df

config = Config()
train_df, test_df = load_data(config)


# --- Updated get_text_features using spaCy instead of NLTK ---
def get_text_features(text):
    # Return zeroed features if text is invalid or empty
    if not isinstance(text, str) or not text.strip():
        return {feat: 0 for feat in [
            'char_count', 'word_count', 'sentence_count', 'avg_word_length',
            'avg_sentence_length', 'unique_word_count', 'ttr', 'stopword_count',
            'stopword_ratio', 'punctuation_count', 'flesch_reading_ease',
            'flesch_kincaid_grade', 'gunning_fog', 'smog_index', 'coleman_liau_index',
            'automated_readability_index', 'dale_chall_readability_score', 'linsear_write_formula'
        ]}

    doc = nlp(text)

    words = [token.text for token in doc if token.is_alpha]
    word_count = len(words)

    if word_count == 0:
        return {feat: 0 for feat in [
            'char_count', 'word_count', 'sentence_count', 'avg_word_length',
            'avg_sentence_length', 'unique_word_count', 'ttr', 'stopword_count',
            'stopword_ratio', 'punctuation_count', 'flesch_reading_ease',
            'flesch_kincaid_grade', 'gunning_fog', 'smog_index', 'coleman_liau_index',
            'automated_readability_index', 'dale_chall_readability_score', 'linsear_write_formula'
        ]}

    stopword_count = sum(1 for token in doc if token.is_stop)
    punctuation_count = sum(1 for token in doc if token.is_punct)
    sentence_count = len(list(doc.sents))

    features = {
        'char_count': len(text),
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'avg_sentence_length': word_count / sentence_count if sentence_count > 0 else 0,
        'unique_word_count': len(set(w.lower() for w in words)),
        'ttr': len(set(w.lower() for w in words)) / word_count,
        'stopword_count': stopword_count,
        'stopword_ratio': stopword_count / word_count,
        'punctuation_count': punctuation_count,
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'gunning_fog': textstat.gunning_fog(text),
        'smog_index': textstat.smog_index(text),
        'coleman_liau_index': textstat.coleman_liau_index(text),
        'automated_readability_index': textstat.automated_readability_index(text),
        'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
        'linsear_write_formula': textstat.linsear_write_formula(text)
    }

    return features


def create_features(df):
    features_1 = df['text_1'].apply(get_text_features).apply(pd.Series)
    features_2 = df['text_2'].apply(get_text_features).apply(pd.Series)

    feature_cols = list(features_1.columns)

    for col in tqdm(feature_cols, desc="Creating comparison features"):
        df[f'{col}_diff'] = features_1[col].astype(float) - features_2[col].astype(float)
        df[f'{col}_ratio'] = features_1[col].astype(float) / (features_2[col].astype(float) + 1e-9)

    final_feature_cols = [f'{col}_diff' for col in feature_cols] + [f'{col}_ratio' for col in feature_cols]
    return df, final_feature_cols

train_df, feature_cols = create_features(train_df)
test_df, _ = create_features(test_df)

print(f"\n Created {len(feature_cols)} features.")
print("ðŸ”Ž Feature columns example:", feature_cols[:5])


def train_and_predict(train_df, test_df, feature_cols, config):
    X = train_df[feature_cols]
    y = train_df['real_text_id'].apply(lambda x: 0 if x == 1 else 1)
    X_test = test_df[feature_cols]

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"===== Fold {fold+1} =====")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**config.LGBM_PARAMS)
        callbacks = [lgb.early_stopping(100, verbose=False)]

        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='logloss',
                  callbacks=callbacks)

        val_preds_proba = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds_proba
        test_preds += model.predict_proba(X_test)[:, 1] / config.N_SPLITS

    oof_accuracy = accuracy_score(y, np.round(oof_preds))
    print(f"\nOverall CV Accuracy: {oof_accuracy:.5f}")

    return test_preds

test_predictions_proba = train_and_predict(train_df, test_df, feature_cols, config)
final_predictions_class = (test_predictions_proba > 0.5).astype(int)

submission_df = pd.DataFrame({'id': test_df['id']})
submission_df['real_text_id'] = final_predictions_class + 1
submission_df.to_csv(config.SUBMISSION_FILE, index=False)

print(f"\nSubmission file created successfully: {config.SUBMISSION_FILE}")
print("Submission file head:")
print(submission_df.head())

