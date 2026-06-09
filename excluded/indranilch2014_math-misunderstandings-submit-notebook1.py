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


train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train.tail()


test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
test.tail()


test['QuestionText'][0]


test['StudentExplanation'][0]


test['QuestionText'][1]


test['StudentExplanation'][2]


test['StudentExplanation'][1]


test['QuestionText'][2]


sample_submission = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv')
sample_submission.tail()


train.info()


test.info()


sample_submission.info()


import seaborn as sns
import matplotlib.pyplot as plt


#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

plt.figure(figsize=(10,4))
sns.countplot(data=train, x='Category', order=train['Category'].value_counts().index, color='b')
plt.xticks(rotation=45)
plt.title("Misconception Categories Distribution")
plt.show()


plt.figure(figsize=(10,4))
sns.countplot(data=train, x='StudentExplanation', order=train['StudentExplanation'].value_counts().head().index, color='r')
plt.xticks(rotation=45)
plt.title("Student Explanations Distribution")
plt.show()


train['QuestionText'][36692]


train['StudentExplanation'][36692]


plt.figure(figsize=(10,4))
sns.countplot(data=train, x='Misconception', order=train['Misconception'].value_counts().head(20).index, color='purple')
plt.xticks(rotation=45)
plt.title("Misconceptions top 20 Distribution")
plt.show()


plt.figure(figsize=(10,4))
sns.countplot(data=train, x='Misconception', order=train['Misconception'].value_counts().tail(15).index, color='g')
plt.xticks(rotation=60)
plt.title("Misconceptions (15 bottom) Distribution")
plt.show()


import keras
import keras_hub
import numpy as np



import torch
print(torch.cuda.is_available())
print(torch.version.cuda)


%%time

import numpy as np
from IPython.display import clear_output
import pandas as pd
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
import warnings
warnings.filterwarnings('ignore')

# Print versions
print('NumPy:', np.__version__)
print('Pandas:', pd.__version__)
print('Scikit-learn:', sklearn.__version__)
print('XGBoost:', xgb.__version__)


%%time

train['Misconception'] = train['Misconception'].fillna('NA')
train['Misconception'] = train['Misconception'].map(str)
train['target_cat'] = train.apply(lambda x: x['Category'] + ":" + x['Misconception'], axis=1)

print(train.shape, test.shape)
train.head()


%%time

train['target_cat'].value_counts()


%%time

map_target1 = train['Category'].value_counts().to_frame()
map_target1['count'] = np.arange(len(map_target1))
map_target1 = map_target1.to_dict()['count']

map_target2 = train['Misconception'].value_counts().to_frame()
map_target2['count'] = np.arange(len(map_target2))
map_target2 = map_target2.to_dict()['count']

map_target1


%%time

train['target1'] = train['Category'].map(map_target1)
train['target2'] = train['Misconception'].map(map_target2)

train['Category'].value_counts()


%%time

train['Misconception'].value_counts()


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

model = TfidfVectorizer(ngram_range=(1, 4), analyzer='char', max_df=0.95, min_df=2)

model.fit(pd.concat([train['sentence'], test['sentence']]))

train_embeddings = model.transform(train['sentence'])
test_embeddings = model.transform(test['sentence'])
clear_output(wait=True)

print('Train sparse shape is', train_embeddings.shape)
print('Test sparse shape is', test_embeddings.shape)


%%time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

p = {'C': 5.340371428482327, 
     'max_iter': 5500, 
     'tol': 3.596680030801802e-06, 
     'penalty': 'l2', 
     'solver': 'lbfgs'}  # Changed solver to CPU-compatible option

SEED = 0

oof_1 = np.zeros((len(train), len(map_target1)))
pred_1 = np.zeros((len(test), len(map_target1)))

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
for i, (train_index, valid_index) in enumerate(skf.split(train_embeddings, train['target1'])):
    print(f"Fold {i}, {len(train_index)}, {len(valid_index)}:")
    model = LogisticRegression(**p, class_weight='balanced', verbose=0)
    model.fit(train_embeddings[train_index], train['target1'].iloc[train_index])
    oof_1[valid_index] = model.predict_proba(train_embeddings[valid_index])  # Removed .get()
    pred_1 += (model.predict_proba(test_embeddings) / 10.)  # Removed .get()

print("ACC:", np.mean(train['target1'] == np.argmax(oof_1, 1)))
print("F1:", f1_score(train['target1'], np.argmax(oof_1, 1), average='weighted'))


%%time

map_inverse1 = {map_target1[k]:k for k in map_target1}
map_inverse2 = {map_target2[k]:k for k in map_target2}


%%time

model = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), analyzer='char', max_df=0.95, min_df=2)

model.fit(pd.concat([train, test]).sentence)

train_embeddings = model.transform(train.sentence)
test_embeddings = model.transform(test.sentence)
clear_output(wait=True)

print('Train sparse shape is',train_embeddings.shape)
print('Test sparse shape is',test_embeddings.shape)


%%time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

p = {'C': 5.340371428482327, 
     'max_iter': 5500, 
     'tol': 3.596680030801802e-06, 
     'penalty': 'l2', 
     'solver': 'lbfgs'}  # Changed solver to CPU-compatible option

SEED = 0

oof_2 = np.zeros((len(train), len(map_target2)))
pred_2 = np.zeros((len(test), len(map_target2)))

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
for i, (train_index, valid_index) in enumerate(skf.split(train_embeddings, train['target2'])):
    print(f"Fold {i}, {len(train_index)}, {len(valid_index)}:")
    model = LogisticRegression(**p, class_weight='balanced', verbose=0)
    model.fit(train_embeddings[train_index], train['target2'].iloc[train_index])
    oof_2[valid_index] = model.predict_proba(train_embeddings[valid_index])   # removed get()
    pred_2 += (model.predict_proba(test_embeddings) / 10.)  # Removed get()

print("ACC:", np.mean( train['target2'] == np.argmax(oof_2, 1) ) )
print("F1:", sklearn.metrics.f1_score(train['target2'] , np.argmax(oof_2, 1), average='weighted') )


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
        
print(f"MAP@3: {map3(train['target_cat'].tolist(), predict)}") # 0.884 CV (approx)


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

submission_file = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")
submission_file['Category:Misconception'] = predict
submission_file.to_csv("submission.csv", index=False)
submission_file

