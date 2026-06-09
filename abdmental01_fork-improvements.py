%%time

%load_ext cudf.pandas

import numpy as np
from IPython.display import clear_output
import cudf
import cuml
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import sklearn.metrics
from scipy import sparse
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
from sklearn.metrics import *

import re
from nltk.stem import WordNetLemmatizer
import nltk

import cupy as cp
import cupyx

import warnings
warnings.filterwarnings('ignore')

print('RAPIDS',cuml.__version__)


%%time

train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")

train['Misconception'] = train['Misconception'].fillna('NA')
train['Misconception'] = train['Misconception'].map(str)
train['target_cat'] = train.apply(lambda x: x['Category'] + ":" + x['Misconception'], axis=1)

print(train.shape, test.shape)

map_target1 = train['Category'].value_counts().to_frame()
map_target1['count'] = np.arange(len(map_target1))
map_target1 = map_target1.to_dict()['count']

map_target2 = train['Misconception'].value_counts().to_frame()
map_target2['count'] = np.arange(len(map_target2))
map_target2 = map_target2.to_dict()['count']

train['target1'] = train['Category'].map(map_target1)
train['target2'] = train['Misconception'].map(map_target2)


%%time

train['sentence'] = "Question: " + train['QuestionText'].astype(str) + \
                    " Answer: " + train['MC_Answer'].astype(str) + \
                    " Explanation: " + train['StudentExplanation'].astype(str)

test['sentence'] = "Question: " + test['QuestionText'].astype(str) + \
                   " Answer: " + test['MC_Answer'].astype(str) + \
                   " Explanation: " + test['StudentExplanation'].astype(str)

clean_newlines = re.compile(r'\n+')
clean_spaces = re.compile(r'\s+')
clean_punct = re.compile(r'[^a-zA-Z0-9\s_]')  

def fast_clean(text):
    text = clean_newlines.sub(' ', text)
    text = clean_spaces.sub(' ', text)
    text = clean_punct.sub('', text)
    return text.strip().lower()

train['sentence'] = train['sentence'].apply(fast_clean)
test['sentence'] = test['sentence'].apply(fast_clean)

lemmatizer = WordNetLemmatizer()

def fast_lemmatize(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

train['sentence'] = train['sentence'].apply(fast_lemmatize)
test['sentence'] = test['sentence'].apply(fast_lemmatize)

%time

print("# \n ---------------------------------------------------------------")
print("# CHAR VECTORIZER StudentExplanation")
print("# ---------------------------------------------------------------\n")

vec_student = TfidfVectorizer(ngram_range=(1, 4), analyzer='char', max_df=0.95, min_df=2)
vec_student.fit(pd.concat([train['StudentExplanation'], test['StudentExplanation']]))

train_stu = vec_student.transform(train['StudentExplanation'])
test_stu = vec_student.transform(test['StudentExplanation'])

print("# \n---------------------------------------------------------------")
print("# CHAR VECTORIZER ")
print("# ---------------------------------------------------------------")

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 4), max_df=0.95, min_df=2)

total_embedding = pd.concat([train['sentence'], test['sentence']])
vectorizer.fit(total_embedding)

train_embed = vectorizer.transform(train['sentence'])
test_embed = vectorizer.transform(test['sentence'])

train_embeddings = sparse.hstack([train_stu, train_embed])
test_embeddings = sparse.hstack([test_stu, test_embed])

train_embeddings = cupyx.scipy.sparse.csr_matrix(train_embeddings)
test_embeddings = cupyx.scipy.sparse.csr_matrix(test_embeddings)

print(f"Train Final Shape: {train_embeddings.shape}")
print(f"Test Final Shape: {test_embeddings.shape}")


%%time

p = {'C': 21.50207632521388, 'max_iter': 9700, 'tol': 1.2363026579561568e-06}

SEED = 0

oof_1 = np.zeros((len(train), len(map_target1)))
pred_1 = np.zeros((len(test), len(map_target1)))

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
for i, (train_index, valid_index) in enumerate(skf.split(train_embeddings, train['target1'])):
    print(f"Fold {i}, {len(train_index)}, {len(valid_index)}:")
    model = cuml.LogisticRegression(**p,class_weight='balanced', fit_intercept = True, penalty = 'l2', solver ='qn')
    model.fit(train_embeddings[train_index], train['target1'].iloc[train_index])
    oof_1[valid_index] = model.predict_proba(train_embeddings[valid_index]).get()
    pred_1 += (model.predict_proba(test_embeddings).get() / 10.)

print("ACC:", np.mean( train['target1'] == np.argmax(oof_1, 1) ) )
print("F1:", sklearn.metrics.f1_score(train['target1'] , np.argmax(oof_1, 1), average='weighted')) # 0.8338


%time

print("# \n ---------------------------------------------------------------")
print("# TARGET Misconception CHAR VECTORIZER StudentExplanation")
print("# ---------------------------------------------------------------\n")

vec_student = TfidfVectorizer(ngram_range=(1, 4), analyzer='char', max_df=0.95, min_df=2)
vec_student.fit(pd.concat([train['StudentExplanation'], test['StudentExplanation']]))

train_stu = vec_student.transform(train['StudentExplanation'])
test_stu = vec_student.transform(test['StudentExplanation'])

print("# \n---------------------------------------------------------------")
print("# TARGET Misconception CHAR VECTORIZER ")
print("# ---------------------------------------------------------------")

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 4), max_df=0.95, min_df=2)

total_embedding = pd.concat([train['sentence'], test['sentence']])
vectorizer.fit(total_embedding)

train_embed = vectorizer.transform(train['sentence'])
test_embed = vectorizer.transform(test['sentence'])

train_embeddings = sparse.hstack([train_stu, train_embed])
test_embeddings = sparse.hstack([test_stu, test_embed])

train_embeddings = cupyx.scipy.sparse.csr_matrix(train_embeddings)
test_embeddings = cupyx.scipy.sparse.csr_matrix(test_embeddings)

print(f"Train Final Shape: {train_embeddings.shape}")
print(f"Test Final Shape: {test_embeddings.shape}")


%%time

p2 = {'C': 9.845314136706442, 'max_iter': 1500, 'tol': 0.00013445181991934277}

oof_2 = np.zeros((len(train), len(map_target2)))
pred_2 = np.zeros((len(test), len(map_target2)))

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
for i, (train_index, valid_index) in enumerate(skf.split(train_embeddings, train['target2'])):
    print(f"Fold {i}, {len(train_index)}, {len(valid_index)}:")
    model = cuml.LogisticRegression(**p2, class_weight='balanced', fit_intercept = True, penalty = 'l2', solver ='qn')
    model.fit(train_embeddings[train_index], train['target2'].iloc[train_index])
    oof_2[valid_index] = model.predict_proba(train_embeddings[valid_index]).get()
    pred_2 += (model.predict_proba(test_embeddings).get() / 10.)

print("ACC:", np.mean( train['target2'] == np.argmax(oof_2, 1) ) )
print("F1:", sklearn.metrics.f1_score(train['target2'] , np.argmax(oof_2, 1), average='weighted')) # 0.9253


%%time

map_inverse1 = {map_target1[k]:k for k in map_target1}
map_inverse2 = {map_target2[k]:k for k in map_target2}


%%time

oof_2[:, 0] = 0
predicted1 = np.argsort(-oof_1, 1)[:,:3]
predicted2 = np.argsort(-oof_2, 1)[:,:3]


%%time

predict = []
for i in range(len(predicted1)):
    pred = []
    for j in range(3):
        p1 = map_inverse1[predicted1[i, j]]
        p2 = map_inverse2[predicted2[i, 0]]
        if 'Misconception' in p1:
            pred.append(p1 + ":" + p2 )
        else:
            pred.append(p1 + ":NA")
    predict.append(pred)

print('ACCURACY_1')
print( np.mean(train['target_cat'] == [p[0] for p in predict]) )
print('ACCURACY_2')
print( np.mean(train['target_cat'] == [p[1] for p in predict]) )
print('ACCURACY_3')
print( np.mean(train['target_cat'] == [p[2] for p in predict]) )


%%time

def map3(target_list, pred_list):
    score = 0.
    for t, p in zip(target_list, pred_list):
        if t == p[0]:
            score+=1.
        elif t == p[1]:
            score+=1/2
        elif t == p[2]:
            score+=1/3
    return score / len(target_list)
        
print(f"MAP@3: {map3(train['target_cat'].tolist(), predict)}") # 0.9036


%%time

pred_2[:, 0] = 0
predicted1 = np.argsort(-pred_1, 1)[:,:3]
predicted2 = np.argsort(-pred_2, 1)[:,:3]

predict = []
for i in range(len(predicted1)):
    pred = []
    for j in range(3):
        p1 = map_inverse1[predicted1[i, j]]
        p2 = map_inverse2[predicted2[i, 0]]        
        if 'Misconception' in p1:
            pred.append(p1 + ":" + p2 )
        else:
            pred.append(p1 + ":NA")
    predict.append(" ".join(pred))

sub = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")
sub['Category:Misconception'] = predict
sub.to_csv("submission.csv", index=False)
sub

