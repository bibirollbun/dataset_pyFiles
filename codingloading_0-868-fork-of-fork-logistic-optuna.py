# === SETUP ===
#!pip install optuna --quiet








# === SETUP ===
%load_ext cudf.pandas

import numpy as np
import cudf
import cuml
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from cuml.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
# import optuna  # Commented out since Optuna is no longer used

import re
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

import warnings
warnings.filterwarnings('ignore')

print("RAPIDS Version:", cuml.__version__)

# === LOAD DATA ===
train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")

train['Misconception'] = train['Misconception'].fillna('NA').astype(str)
train['target_cat'] = train['Category'] + ":" + train['Misconception']

# === ENCODE TARGET LABELS ===
map_target1 = {k: i for i, k in enumerate(train['Category'].value_counts().index)}
map_target2 = {k: i for i, k in enumerate(train['Misconception'].value_counts().index)}

train['target1'] = train['Category'].map(map_target1)
train['target2'] = train['Misconception'].map(map_target2)

# === CREATE SENTENCES ===
train['sentence'] = "Question: " + train['QuestionText'].astype(str) + \
                    " Answer: " + train['MC_Answer'].astype(str) + \
                    " Explanation: " + train['StudentExplanation'].astype(str)

test['sentence'] = "Question: " + test['QuestionText'].astype(str) + \
                   " Answer: " + test['MC_Answer'].astype(str) + \
                   " Explanation: " + test['StudentExplanation'].astype(str)

# === CLEAN TEXT ===
clean_newlines = re.compile(r'\n+')
clean_spaces = re.compile(r'\s+')
clean_punct = re.compile(r'[^a-zA-Z0-9\s]')
lemmatizer = WordNetLemmatizer()

def fast_clean(text):
    text = clean_newlines.sub(' ', text)
    text = clean_spaces.sub(' ', text)
    text = clean_punct.sub('', text)
    return text.strip().lower()

def fast_lemmatize(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

train['sentence'] = train['sentence'].apply(fast_clean).apply(fast_lemmatize)
test['sentence'] = test['sentence'].apply(fast_clean).apply(fast_lemmatize)

# === TF-IDF ===
# Remove more common terms
vectorizer = TfidfVectorizer(ngram_range=(1, 5), max_df=0.95, min_df=2)
vectorizer.fit(pd.concat([train['sentence'], test['sentence']]))

train_embeddings = vectorizer.transform(train['sentence'])
test_embeddings = vectorizer.transform(test['sentence'])

# === OPTUNA TUNING FUNCTION (COMMENTED OUT) ===
# def objective(trial, target, labels):
#     C = trial.suggest_float("C", 0.01, 10.0)
#     acc_scores = []
#     skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#     for train_idx, val_idx in skf.split(target, labels):
#         model = LogisticRegression(C=C)
#         model.fit(target[train_idx], labels.iloc[train_idx])
#         preds = model.predict(target[val_idx])
#         acc = (preds == labels.iloc[val_idx].values).mean()
#         acc_scores.append(acc)
#     return np.mean(acc_scores)

# === FIXED C VALUES FROM PREVIOUS OPTUNA RUN ===
best_C1 = 5.9462366951543295
best_C2 = 8.968620263623801

print("Using fixed C for Category:", best_C1)
print("Using fixed C for Misconception:", best_C2)

# === FINAL TRAINING ===
ytrain1 = np.zeros((len(train), len(map_target1)))
ytest1 = np.zeros((len(test), len(map_target1)))

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
for i, (train_index, valid_index) in enumerate(skf.split(train_embeddings, train['target1'])):
    print(f"Category Fold {i}")
    model = LogisticRegression(C=best_C1)
    model.fit(train_embeddings[train_index], train['target1'].iloc[train_index])
    ytrain1[valid_index] = model.predict_proba(train_embeddings[valid_index])
    ytest1 += model.predict_proba(test_embeddings) / 10.

print("Category ACC:", np.mean(train['target1'] == np.argmax(ytrain1, 1)))
print("Category F1:", sklearn.metrics.f1_score(train['target1'], np.argmax(ytrain1, 1), average='weighted'))

# === MISCONCEPTION MODEL ===
ytrain2 = np.zeros((len(train), len(map_target2)))
ytest2 = np.zeros((len(test), len(map_target2)))

for i, (train_index, valid_index) in enumerate(skf.split(train_embeddings, train['target2'])):
    print(f"Misconception Fold {i}")
    model = LogisticRegression(C=best_C2)
    model.fit(train_embeddings[train_index], train['target2'].iloc[train_index])
    ytrain2[valid_index] = model.predict_proba(train_embeddings[valid_index])
    ytest2 += model.predict_proba(test_embeddings) / 10.

print("Misconception ACC:", np.mean(train['target2'] == np.argmax(ytrain2, 1)))
print("Misconception F1:", sklearn.metrics.f1_score(train['target2'], np.argmax(ytrain2, 1), average='weighted'))

# === PREDICTION PREP ===
map_inverse1 = {v: k for k, v in map_target1.items()}
map_inverse2 = {v: k for k, v in map_target2.items()}

ytrain2[:, 0] = 0  # ensure NA is deprioritized
predicted1 = np.argsort(-ytrain1, axis=1)[:, :3]
predicted2 = np.argsort(-ytrain2, axis=1)[:, :3]

predict = []
for i in range(len(predicted1)):
    pred = []
    for j in range(3):
        p1 = map_inverse1[predicted1[i, j]]
        p2 = map_inverse2[predicted2[i, 0]]
        if 'Misconception' in p1:
            pred.append(p1 + ":" + p2)
        else:
            pred.append(p1 + ":NA")
    predict.append(pred)

def map3(target_list, pred_list):
    score = 0.
    for t, p in zip(target_list, pred_list):
        if t == p[0]:
            score += 1.
        elif t == p[1]:
            score += 1 / 2
        elif t == p[2]:
            score += 1 / 3
    return score / len(target_list)

print("MAP@3:", map3(train['target_cat'].tolist(), predict))

# === FINAL TEST SUBMISSION ===
ytest2[:, 0] = 0
predicted1 = np.argsort(-ytest1, axis=1)[:, :3]
predicted2 = np.argsort(-ytest2, axis=1)[:, :3]

final_predict = []
for i in range(len(predicted1)):
    pred = []
    for j in range(3):
        p1 = map_inverse1[predicted1[i, j]]
        p2 = map_inverse2[predicted2[i, 0]]
        if 'Misconception' in p1:
            pred.append(p1 + ":" + p2)
        else:
            pred.append(p1 + ":NA")
    final_predict.append(" ".join(pred))

sub = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")
sub['Category:Misconception'] = final_predict
sub.to_csv("submission.csv", index=False)
sub.head()














