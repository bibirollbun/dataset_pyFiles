








import numpy as np
import pandas as pd
import re
import nltk
import warnings
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import xgboost as xgb

# Setup
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
warnings.filterwarnings('ignore')

def map3(target_list, pred_list):
    score = 0.
    for t, p in zip(target_list, pred_list):
        if t == p[0]: score += 1.
        elif len(p) > 1 and t == p[1]: score += 1/2
        elif len(p) > 2 and t == p[2]: score += 1/3
    return score / len(target_list)

def advanced_clean(text):
    text = re.sub(r'(\d+)\s*/\s*(\d+)', r'FRAC_\1_\2', text)
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'FRAC_\1_\2', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s_]', '', text)
    return text.strip().lower()

def extract_math_features(text):
    return {
        'frac_count': len(re.findall(r'FRAC_\d+_\d+|\\frac', text)),
        'number_count': len(re.findall(r'\b\d+\b', text)),
        'operator_count': len(re.findall(r'[\+\-\*/=]', text)),
        'starts_with_number': int(bool(re.match(r'^\d+', text))),
        'has_frac_token': int('FRAC_' in text)
    }

def fast_lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def more_math_feats(text):
    return {
        'starts_with_frac': int(text.startswith('FRAC_')),
        'contains_eq': int('=' in text)
    }

def create_features(df):
    df['mc_answer_len'] = df['MC_Answer'].astype(str).str.len()
    df['explanation_len'] = df['StudentExplanation'].astype(str).str.len()
    df['question_len'] = df['QuestionText'].astype(str).str.len()
    df['explanation_to_question_ratio'] = df['explanation_len'] / (df['question_len'] + 1)

    for col in ['QuestionText', 'MC_Answer', 'StudentExplanation']:
        tokens = df[col].astype(str).apply(advanced_clean).apply(fast_lemmatize).str.split()
        df[f'{col}_tok_count'] = tokens.apply(len)
        df[f'{col}_uniq_ratio'] = tokens.apply(lambda x: len(set(x)) / (len(x) + 1))

        feats = pd.DataFrame(df[col].astype(str).apply(more_math_feats).tolist())
        feats.columns = [f'{col.lower()}_{c}' for c in feats.columns]
        df = pd.concat([df, feats], axis=1)

    for col in ['QuestionText', 'MC_Answer']:
        features = df[col].astype(str).apply(advanced_clean).apply(extract_math_features).apply(pd.Series)
        prefix = 'mc_' if col == 'MC_Answer' else ''
        features.columns = [f'{prefix}{c}' for c in features.columns]
        df = pd.concat([df, features], axis=1)

    df['answer_to_question_ratio'] = df['mc_answer_len'] / (df['question_len'] + 1)
    df['explanation_to_answer_ratio'] = df['explanation_len'] / (df['mc_answer_len'] + 1)
    return df

train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
sample = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")

train['Misconception'] = train['Misconception'].fillna('NA')
train['target_cat'] = train['Category'] + ':' + train['Misconception']
train = train.sort_values('target_cat').reset_index(drop=True)

le = LabelEncoder()
train['target_encoded'] = le.fit_transform(train['target_cat'])
n_classes = len(le.classes_)

train = create_features(train)
test = create_features(test)

train['combined_text'] = "Question: " + train['QuestionText'] + " Answer: " + train['MC_Answer'] + " Explanation: " + train['StudentExplanation']
test['combined_text'] = "Question: " + test['QuestionText'] + " Answer: " + test['MC_Answer'] + " Explanation: " + test['StudentExplanation']

train['cleaned_text'] = train['combined_text'].apply(advanced_clean).apply(fast_lemmatize)
test['cleaned_text'] = test['combined_text'].apply(advanced_clean).apply(fast_lemmatize)

train['mc_cleaned'] = train['MC_Answer'].astype(str).apply(advanced_clean).apply(fast_lemmatize)
test['mc_cleaned'] = test['MC_Answer'].astype(str).apply(advanced_clean).apply(fast_lemmatize)

train['q_cleaned'] = train['QuestionText'].astype(str).apply(advanced_clean).apply(fast_lemmatize)
test['q_cleaned'] = test['QuestionText'].astype(str).apply(advanced_clean).apply(fast_lemmatize)

tfidf_word = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=5000)
tfidf_word.fit(pd.concat([train['cleaned_text'], test['cleaned_text']]))
train_word = tfidf_word.transform(train['cleaned_text'])
test_word = tfidf_word.transform(test['cleaned_text'])

tfidf_expl = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=3000)
tfidf_expl.fit(pd.concat([train['StudentExplanation'], test['StudentExplanation']]))
train_expl = tfidf_expl.transform(train['StudentExplanation'])
test_expl = tfidf_expl.transform(test['StudentExplanation'])

tfidf_mc = TfidfVectorizer(ngram_range=(1, 3), max_features=3000)
tfidf_mc.fit(pd.concat([train['mc_cleaned'], test['mc_cleaned']]))
train_mc = tfidf_mc.transform(train['mc_cleaned'])
test_mc = tfidf_mc.transform(test['mc_cleaned'])

# Cosine similarity between question and MC_Answer
q_vecs = tfidf_mc.transform(train['q_cleaned'])
a_vecs = tfidf_mc.transform(train['mc_cleaned'])
train['qa_cosine'] = cosine_similarity(q_vecs, a_vecs).diagonal()

q_vecs_test = tfidf_mc.transform(test['q_cleaned'])
a_vecs_test = tfidf_mc.transform(test['mc_cleaned'])
test['qa_cosine'] = cosine_similarity(q_vecs_test, a_vecs_test).diagonal()

char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=3000)
char_tfidf.fit(pd.concat([train['cleaned_text'], test['cleaned_text']]))
train_char = char_tfidf.transform(train['cleaned_text'])
test_char = char_tfidf.transform(test['cleaned_text'])

numeric_cols = [col for col in train.columns if col.endswith(('_len', '_ratio', '_count', '_tok_count', '_uniq_ratio', '_number', '_operator', '_starts_with_number', '_has_frac_token', '_starts_with_frac', '_contains_eq', 'qa_cosine'))]
X_numeric = sparse.csr_matrix(train[numeric_cols].fillna(0).values)
X_numeric_test = sparse.csr_matrix(test[numeric_cols].fillna(0).values)

X_train = sparse.hstack([train_word, train_expl, train_char, train_mc, X_numeric])
X_test = sparse.hstack([test_word, test_expl, test_char, test_mc, X_numeric_test])
y = train['target_encoded'].values

oof_preds = np.zeros((len(train), n_classes))
test_preds = np.zeros((len(test), n_classes))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

params = {
    'objective': 'multi:softprob',
    'num_class': n_classes,
    'eval_metric': 'mlogloss',
    'max_depth': 12,
    'learning_rate': 0.05,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'tree_method': 'gpu_hist',
    'random_state': 42
}

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y)):
    print(f"\nFold {fold+1}")
    dtrain = xgb.DMatrix(X_train[trn_idx], label=y[trn_idx])
    dvalid = xgb.DMatrix(X_train[val_idx], label=y[val_idx])

    model = xgb.train(params, dtrain, num_boost_round=1000,
                      evals=[(dvalid, 'valid')],
                      early_stopping_rounds=50,
                      verbose_eval=50)

    oof_preds[val_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))
    test_preds += model.predict(xgb.DMatrix(X_test), iteration_range=(0, model.best_iteration)) / skf.n_splits

oof_top3 = np.argsort(-oof_preds, axis=1)[:, :3]
oof_labels = [[le.inverse_transform([i])[0] for i in row] for row in oof_top3]
map_score = map3(train['target_cat'].tolist(), oof_labels)
print(f"\nValidation MAP@3: {map_score:.4f}")

top3_test = np.argsort(-test_preds, axis=1)[:, :3]
preds = [' '.join([le.inverse_transform([i])[0] for i in row]) for row in top3_test]
sample['Category:Misconception'] = preds
sample.to_csv("submission.csv", index=False)
print("Saved submission.csv")








