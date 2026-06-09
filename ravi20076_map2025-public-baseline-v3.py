


%%writefile -a imports.py

import torch
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

from tqdm.notebook import tqdm

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
warnings.filterwarnings('ignore')


%%writefile -a myscript.py 

def map3(target_list, pred_list):
    score = 0.0
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


%%writefile -a myscript.py 

print()
train  = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test   = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
print(f"---> Shapes = {train.shape} {test.shape}")

idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
tmp = train.loc[idx].copy()
tmp['c'] = tmp.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
tmp = tmp.sort_values('c',ascending=False)
tmp = tmp.drop_duplicates(['QuestionId'])
tmp = tmp[['QuestionId','MC_Answer']]
tmp['is_correct'] = 1

train = train.merge(tmp, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)

test = test.merge(tmp, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

train = train.sort_values(["row_id"])
test  = test.sort_values(["row_id"])

print(f"---> Shapes = {train.shape} {test.shape}")


%%writefile -a myscript.py 

train['Misconception'] = train['Misconception'].fillna('NA')
train['target_cat']    = train['Category'] + ':' + train['Misconception']

le = LabelEncoder()
train['target_encoded'] = le.fit_transform(train['target_cat'])
target_classes          = le.classes_
n_classes               = len(target_classes)

train = create_features(train)
test  = create_features(test)

train['combined_text'] = \
"Question: " + train['QuestionText'] + " Answer: " + train['MC_Answer'] + " Explanation: " + train['StudentExplanation']

test['combined_text'] = \
"Question: " + test['QuestionText'] + " Answer: " + test['MC_Answer'] + " Explanation: " + test['StudentExplanation']

train['cleaned_text'] = train['combined_text'].apply(advanced_clean).apply(fast_lemmatize)
test['cleaned_text']  = test['combined_text'].apply(advanced_clean).apply(fast_lemmatize)

train = train.sort_values(["row_id"])
test  = test.sort_values(["row_id"])

tfidf_word = TfidfVectorizer(
    ngram_range  = (1, 4), 
    stop_words   = 'english',
    max_features = 4000
)
tfidf_word.fit(pd.concat([train['cleaned_text'], test['cleaned_text']]))

train_tfidf_word = tfidf_word.transform(train['cleaned_text'])
test_tfidf_word = tfidf_word.transform(test['cleaned_text'])

tfidf_expl = TfidfVectorizer(
    ngram_range  = (1, 4), 
    stop_words   = 'english',
    max_features = 3000,
)
tfidf_expl.fit(pd.concat([train['StudentExplanation'], test['StudentExplanation']]))
train_tfidf_expl = tfidf_expl.transform(train['StudentExplanation'])
test_tfidf_expl = tfidf_expl.transform(test['StudentExplanation'])

char_tfidf = TfidfVectorizer(
    analyzer='char_wb', 
    ngram_range  = (2, 5),
    max_features = 3000,
)
char_tfidf.fit(pd.concat([train['cleaned_text'], test['cleaned_text']]))
train_char = char_tfidf.transform(train['cleaned_text'])
test_char = char_tfidf.transform(test['cleaned_text'])

numeric_cols = [
    'mc_answer_len',
    'explanation_len', 
    'question_len',
    'explanation_to_question_ratio', 
    'frac_count', 
    'number_count',
    'operator_count', 
    'mc_frac_count', 
    'mc_number_count',
    'mc_operator_count',
    'is_correct',
]

X_numeric = sparse.csr_matrix(train[numeric_cols].fillna(0).values)
X_numeric_test = sparse.csr_matrix(test[numeric_cols].fillna(0).values)

X_train = sparse.hstack([train_tfidf_word, train_tfidf_expl, train_char, X_numeric])
X_test  = sparse.hstack([test_tfidf_word, test_tfidf_expl, test_char, X_numeric_test])
y       = train['target_encoded'].values



%%writefile -a myscript.py 

print()
oof_preds  = np.zeros((len(train), n_classes))
test_preds = np.zeros((len(test), n_classes))
cv         = StratifiedKFold(n_splits = n_splits , shuffle=True, random_state=42)

for fold, (trn_idx, val_idx) in tqdm( enumerate(cv.split(X_train, y)) ):
    
    dtrain = xgb.DMatrix(X_train[trn_idx], label=y[trn_idx])
    dvalid = xgb.DMatrix(X_train[val_idx], label=y[val_idx])

    model = xgb.train(
        {
            'objective'       : 'multi:softprob',
            'num_class'       : n_classes,
            'eval_metric'     : 'mlogloss',
            'max_depth'       : 10 if test_req == False else 3,
            'learning_rate'   : 0.04,
            'subsample'       : 0.275,
            'colsample_bytree': 0.275,
            'device'          : "cuda:0" if torch.cuda.is_available() else "cpu",
            'random_state'    : 42,
        }, 
        dtrain, 
        num_boost_round        = 5000 if test_req == False else 10, 
        evals                  = [(dvalid, 'valid')], 
        early_stopping_rounds  = 100 if test_req == False else 5, 
        verbose_eval           = 0,
    )
    
    oof_preds[val_idx] = \
    model.predict(
        dvalid, iteration_range=(0, model.best_iteration)
    )
    
    test_preds += \
    model.predict(
        xgb.DMatrix(X_test), iteration_range=(0, model.best_iteration)
    ) / n_splits

    print(f"Fold {fold + 1} complete")



%%writefile -a myscript.py 

oof_top3   = np.argsort(-oof_preds, axis=1)[:, :3]
oof_labels = [[le.inverse_transform([i])[0] for i in row] for row in oof_top3]
y_true     = train['target_cat'].tolist()
score      = map3(y_true, oof_labels)
print(f"\n---> Final CV score = {score:.8f}")

top3_test = np.argsort(-test_preds, axis=1)[:, :3]
preds     = [' '.join([le.inverse_transform([i])[0] for i in row]) for row in top3_test]
sample['Category:Misconception'] = preds
sample.to_csv("submission.csv", index=False)


import pandas as pd

test_req = False
n_splits = 5
cutoff   = 10


%%time 

sample = pd.read_csv(
    "/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv"
)

if len(sample) <= cutoff :
    print(f"---> Exiting code and submitting the sample file")
    sample.to_csv("submission.csv", index=False)

else:
    exec(open(f"imports.py", "r").read())
    exec(open(f"myscript.py", "r").read())

!ls
print()
!head submission.csv
print()

