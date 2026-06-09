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


df=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
dt=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
samp=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")


df.info()


dt.info()


df.head(1)


dt.head(1)


# -*- coding: utf-8 -*-
"""
map3_offline_xgb_lgbm.py
Offline-optimized solution for the MAP student misunderstanding competition
"""

import numpy as np
import pandas as pd
import re
import nltk
import warnings
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from scipy import sparse
import xgboost as xgb
from lightgbm import LGBMClassifier

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
warnings.filterwarnings('ignore')

# MAP@3 metric
def map3(target_list, pred_list):
    score = 0.
    for t, p in zip(target_list, pred_list):
        if t == p[0]: score += 1.
        elif len(p) > 1 and t == p[1]: score += 1/2
        elif len(p) > 2 and t == p[2]: score += 1/3
    return score / len(target_list)

def advanced_clean(text):
    text = re.sub(r'(\d+)\s*/\s*(\d+)', r'FRAC_\1_\2', text)
    text = re.sub(r'\\frac\{([^\}]+)\}\{([^\}]+)\}', r'FRAC_\1_\2', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s_]', '', text)
    return text.strip().lower()

def extract_math_features(text):
    features = {
        'frac_count': len(re.findall(r'FRAC_\d+_\d+|\\frac', text)),
        'number_count': len(re.findall(r'\b\d+\b', text)),
        'operator_count': len(re.findall(r'[\+\-\*\/\=]', text))
    }
    return features

def fast_lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def create_features(df):
    df['mc_answer_len'] = df['MC_Answer'].astype(str).str.len()
    df['explanation_len'] = df['StudentExplanation'].astype(str).str.len()
    df['question_len'] = df['QuestionText'].astype(str).str.len()
    df['explanation_to_question_ratio'] = df['explanation_len'] / (df['question_len'] + 1)

    for col in ['QuestionText', 'MC_Answer']:
        math_features = df[col].apply(extract_math_features).apply(pd.Series)
        prefix = 'mc_' if col == 'MC_Answer' else ''
        math_features.columns = [f'{prefix}{c}' for c in math_features.columns]
        df = pd.concat([df, math_features], axis=1)
    return df

# Load data
train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
sample = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")

train['Misconception'] = train['Misconception'].fillna('NA')
train['target_cat'] = train['Category'] + ':' + train['Misconception']
train = train.sort_values('target_cat').reset_index(drop=True)

le = LabelEncoder()
train['target_encoded'] = le.fit_transform(train['target_cat'])
target_classes = le.classes_
n_classes = len(target_classes)

train = create_features(train)
test = create_features(test)

train['combined_text'] = "Question: " + train['QuestionText'] + " Answer: " + train['MC_Answer'] + " Explanation: " + train['StudentExplanation']
test['combined_text'] = "Question: " + test['QuestionText'] + " Answer: " + test['MC_Answer'] + " Explanation: " + test['StudentExplanation']

train['cleaned_text'] = train['combined_text'].apply(advanced_clean).apply(fast_lemmatize)
test['cleaned_text'] = test['combined_text'].apply(advanced_clean).apply(fast_lemmatize)

# TF-IDF
tfidf_word = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=5000)
tfidf_word.fit(pd.concat([train['cleaned_text'], test['cleaned_text']]))

train_tfidf_word = tfidf_word.transform(train['cleaned_text'])
test_tfidf_word = tfidf_word.transform(test['cleaned_text'])

tfidf_expl = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=3000)
tfidf_expl.fit(pd.concat([train['StudentExplanation'], test['StudentExplanation']]))
train_tfidf_expl = tfidf_expl.transform(train['StudentExplanation'])
test_tfidf_expl = tfidf_expl.transform(test['StudentExplanation'])

char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=3000)
char_tfidf.fit(pd.concat([train['cleaned_text'], test['cleaned_text']]))
train_char = char_tfidf.transform(train['cleaned_text'])
test_char = char_tfidf.transform(test['cleaned_text'])

# Numeric features
numeric_cols = [
    'mc_answer_len', 'explanation_len', 'question_len',
    'explanation_to_question_ratio', 'frac_count', 'number_count',
    'operator_count', 'mc_frac_count', 'mc_number_count', 'mc_operator_count']
X_numeric = sparse.csr_matrix(train[numeric_cols].fillna(0).values)
X_numeric_test = sparse.csr_matrix(test[numeric_cols].fillna(0).values)

X_train = sparse.hstack([train_tfidf_word, train_tfidf_expl, train_char, X_numeric])
X_test = sparse.hstack([test_tfidf_word, test_tfidf_expl, test_char, X_numeric_test])

y = train['target_encoded'].values

# XGBoost KFold
oof_preds = np.zeros((len(train), n_classes))
test_preds = np.zeros((len(test), n_classes))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

params = {
    'objective': 'multi:softprob',
    'num_class': n_classes,
    'eval_metric': 'mlogloss',
    'max_depth': 10,
    'learning_rate': 0.05,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'random_state': 42
}

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y)):
    print(f"Fold {fold+1}")
    dtrain = xgb.DMatrix(X_train[trn_idx], label=y[trn_idx])
    dvalid = xgb.DMatrix(X_train[val_idx], label=y[val_idx])

    model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dvalid, 'valid')], early_stopping_rounds=50, verbose_eval=50)
    oof_preds[val_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))
    test_preds += model.predict(xgb.DMatrix(X_test), iteration_range=(0, model.best_iteration)) / skf.n_splits

# MAP@3
oof_top3 = np.argsort(-oof_preds, axis=1)[:, :3]
oof_labels = [[le.inverse_transform([i])[0] for i in row] for row in oof_top3]
y_true = train['target_cat'].tolist()
map_score = map3(y_true, oof_labels)
print(f"\nValidation MAP@3: {map_score:.4f}")

# Prepare submission
top3_test = np.argsort(-test_preds, axis=1)[:, :3]
preds = [' '.join([le.inverse_transform([i])[0] for i in row]) for row in top3_test]
sample['Category:Misconception'] = preds
sample.to_csv("submission.csv", index=False)
print("Saved submission.csv")



sample




