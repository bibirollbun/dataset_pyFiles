%%time

import numpy as np
import pandas as pd
import cudf
import cuml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics
import re
from nltk.stem import WordNetLemmatizer
import nltk
from scipy import sparse
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')


%%time

train = pd.read_csv(
    "/kaggle/input/map-charting-student-math-misunderstandings/train.csv"
)
test = pd.read_csv(
    "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"
)


re_frac_slash = re.compile(r'(\d+)\s*/\s*(\d+)')
re_frac_latex = re.compile(r'\\frac\{([^\}]+)\}\{([^\}]+)\}')
re_newlines = re.compile(r'\n+')
re_spaces = re.compile(r'\s+')
re_punct = re.compile(r'[^a-zA-Z0-9\s_]')

def txt_clean(text):
    text = re_frac_slash.sub(r'FRAC_\1_\2', text)
    text = re_frac_latex.sub(r'FRAC_\1_\2', text)
    text = re_newlines.sub(' ', text)
    text = re_spaces.sub(' ', text)
    text = re_punct.sub('', text)
    return text.strip().lower()

def extract_math_features(text):
    text = text.lower()

    features = {}
    features['frac_count'] = len(re.findall(r'FRAC_\d+_\d+|\\frac', text))
    features['number_count'] = len(re.findall(r'\b\d+\b', text))
    features['operator_count'] = len(re.findall(r'[\+\-\*\/\=]', text))
    features['multiply_sign_count'] = len(re.findall(r'[\*×·]|times', text))
    features['power_count'] = len(re.findall(r'\^|\*\*|\b[sS]quared\b|\b[cC]ubed\b', text))

    return features

def fast_lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def create_features(df, is_train=True):
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

train['Misconception'] = train['Misconception'].fillna('NA').astype(str)
train['target_cat'] = train.apply(
    lambda x: x['Category'] + ":" + x['Misconception'], axis=1
)
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

train = create_features(train, is_train=True)
test = create_features(test, is_train=False)

train['combined_text'] = (
    "Question: " + train['QuestionText'].astype(str) +
    " Answer: " + train['MC_Answer'].astype(str) +
    " Explanation: " + train['StudentExplanation'].astype(str)
)
test['combined_text'] = (
    "Question: " + test['QuestionText'].astype(str) +
    " Answer: " + test['MC_Answer'].astype(str) +
    " Explanation: " + test['StudentExplanation'].astype(str)
)

train['cleaned_text'] = train['combined_text'].apply(txt_clean).apply(fast_lemmatize)
test['cleaned_text'] = test['combined_text'].apply(txt_clean).apply(fast_lemmatize)

print(f"Train shape After New Features: {train.shape}, Test shape After New Features: {test.shape}")


%%time

le = LabelEncoder()
train['target_encoded'] = le.fit_transform(train['target_cat'])
target_classes = le.classes_
n_classes = len(target_classes)
print(f"Number of target classes: {n_classes}")


%%time

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 4), max_df=0.95, min_df=2, max_features=2000, sublinear_tf=True)

total_embedding = pd.concat([train['cleaned_text'], test['cleaned_text']])
vectorizer.fit(total_embedding)

train_embed = vectorizer.transform(train['cleaned_text'])
test_embed = vectorizer.transform(test['cleaned_text'])

num_cols = ['mc_answer_len', 'explanation_len', 'question_len','explanation_to_question_ratio', 'frac_count',
            'number_count','operator_count', 'mc_frac_count', 'mc_number_count','mc_operator_count']

num_fe = [f for f in num_cols if f in train.columns]

train_num = train[num_fe].fillna(0).values
test_num = test[num_fe].fillna(0).values

train_ = sparse.hstack([train_embed, sparse.csr_matrix(train_num)])
test_ = sparse.hstack([test_embed, sparse.csr_matrix(test_num)])

print(f"Train Final Shape: {train_.shape}")
print(f"Test Final Shape: {test_.shape}")


%%time

SPLITS = 10
SEED = 0

def map3(target_list, pred_list):
    score = 0.
    for t, p in zip(target_list, pred_list):
        if t == p[0]:
            score += 1.
        elif len(p) > 1 and t == p[1]:
            score += 1/2
        elif len(p) > 2 and t == p[2]:
            score += 1/3
    return score / len(target_list)

def TRAIN(params):
    
    kFold = StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=SEED) 
    
    oof = np.zeros((len(train), n_classes))
    preds = np.zeros((len(test), n_classes))
    
    for fold, (train_idx, valid_idx) in enumerate(kFold.split(train_, train['target_encoded'])):
        print(f"Fold {fold + 1}")
    
        X_train_fold = train_[train_idx]
        y_train_fold = train['target_encoded'].iloc[train_idx]
        X_valid_fold = train_[valid_idx]
        y_valid_fold = train['target_encoded'].iloc[valid_idx]
    
        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold)
    
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000, 
            evals=[(dvalid, 'valid')],
            early_stopping_rounds=50, 
            verbose_eval=False
        )
    
        oof[valid_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))
        preds += model.predict(xgb.DMatrix(test_), iteration_range=(0, model.best_iteration)) / SPLITS
    
    oof_pred = np.argmax(oof, axis=1)
    accuracy = np.mean(train['target_encoded'] == oof_pred)
    f1 = sklearn.metrics.f1_score(train['target_encoded'], oof_pred, average='weighted')
    
    print(f"\nValidation Accuracy: {accuracy:.4f}")
    print(f"Validation F1-score: {f1:.4f}")

    return oof, preds

X_PARAMS = {'max_depth': 10, 'learning_rate': 0.0010974684673250828, 'min_child_weight': 1,
            'subsample': 0.9800969106980701, 'colsample_bytree': 0.6305990340548516, 'gamma': 2.4140963859555216,
            'gpu_id': 0, 'random_state' : SEED, 'tree_method': 'gpu_hist','objective': 'multi:softprob',
            'num_class': n_classes, 'eval_metric': 'mlogloss',
           }

xgb_oof, xgb_pred = TRAIN(X_PARAMS) # ACC : 0.7856, MAPE@3 : 0.8732  


%%time

train_top3 = np.argsort(-xgb_oof, axis=1)[:, :3]

predictions = []
for indices in train_top3:
    predictions.append([target_classes[i] for i in indices])

map_score = map3(train['target_cat'].tolist(), predictions)
print(f"Validation MAP@3: {map_score:.4f}")

test_top3 = np.argsort(-xgb_pred, axis=1)[:, :3]

test_preds = []
for indices in test_top3:
    pred = [target_classes[i] for i in indices]
    test_preds.append(' '.join(pred))

submission = pd.read_csv(
    "/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv"
)
submission['Category:Misconception'] = test_preds
submission.to_csv("submission.csv", index=False)
submission

