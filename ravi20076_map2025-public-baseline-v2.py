


%load_ext cudf.pandas

from IPython.display import clear_output
import numpy as np
from scipy import sparse
import pandas as pd
from tqdm.notebook import tqdm

import cudf, cuml
from cuml.feature_extraction.text import TfidfVectorizer as TFIDF
from cuml import LogisticRegression

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import *
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import *

import re
from nltk.stem import WordNetLemmatizer
import nltk

from warnings import filterwarnings
filterwarnings('ignore')


test_req = False
n_splits = 10
state    = 42


def advanced_clean(text):
    text = re.sub(r'(\d+)\s*/\s*(\d+)', r'FRAC_\1_\2', text)
    text = re.sub(r'\\frac\{([^\}]+)\}\{([^\}]+)\}', r'FRAC_\1_\2', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s_]', '', text)

    return text.strip().lower()

def extract_math_features(text):
    features = {}
    features['frac_count'] = len(re.findall(r'FRAC_\d+_\d+|\\frac', text))
    features['number_count'] = len(re.findall(r'\b\d+\b', text))
    features['operator_count'] = len(re.findall(r'[\+\-\*\/\=]', text))
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


%%time 

train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test  = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")

train['Misconception'] = train['Misconception'].fillna('NA')
train['Misconception'] = train['Misconception'].map(str)
train['target_cat'] = train.apply(lambda x: x['Category'] + ":" + x['Misconception'], axis=1)

print(f"\n\n---> Shapes = {train.shape} {test.shape}")

map_target1 = train['Category'].value_counts().to_frame()
map_target1['count'] = np.arange(len(map_target1))
map_target1 = map_target1.to_dict()['count']

map_target2 = train['Misconception'].value_counts().to_frame()
map_target2['count'] = np.arange(len(map_target2))
map_target2 = map_target2.to_dict()['count']

train['target1'] = train['Category'].map(map_target1)
train['target2'] = train['Misconception'].map(map_target2)

train = create_features(train, is_train=True)
test  = create_features(test, is_train=False)
print(f"---> Shapes = {train.shape} {test.shape} | Extra features")

train['sentence'] = "Question: " + train['QuestionText'].astype(str) + \
                    " Answer: " + train['MC_Answer'].astype(str) + \
                    " Explanation: " + train['StudentExplanation'].astype(str)

test['sentence'] = "Question: " + test['QuestionText'].astype(str) + \
                   " Answer: " + test['MC_Answer'].astype(str) + \
                   " Explanation: " + test['StudentExplanation'].astype(str)

train['sentence'] = train['sentence'].apply(advanced_clean).apply(fast_lemmatize)
test['sentence']  = test['sentence'].apply(advanced_clean).apply(fast_lemmatize)

print(f"---> Shapes = {train.shape} {test.shape} | Data cleaning")
print()




%%time 

numeric_features = [
    'mc_answer_len', 'explanation_len', 'question_len',
    'explanation_to_question_ratio', 'frac_count', 'number_count',
    'operator_count', 'mc_frac_count', 'mc_number_count',
    'mc_operator_count'
]
numeric_features = [f for f in numeric_features if f in train.columns]

X_numeric      = train[numeric_features].fillna(0).values
X_numeric_test = test[numeric_features].fillna(0).values

scl            = StandardScaler()
X_numeric      = scl.fit_transform(X_numeric)
X_numeric_test = scl.transform(X_numeric_test)

print(f"---> Shapes = {X_numeric.shape} {X_numeric_test.shape} | Data scaling - num-cols")


%%time 

class VotingModelMaker(ClassifierMixin, BaseEstimator):
    "Creates a simple average-voting model similar to scikit-learn VotingClassifier using CUML base models"

    def __init__(self, estimators : list,):
        self.estimators = estimators

    def fit(self, X , y , **params):
        self.fitted_models = []

        for model in self.estimators :
            model.fit(X, y)
            self.fitted_models.append(model)

        self.is_fitted = True
        return self

    def predict_proba(self, X, y = None, **params):

        preds = 0
        n_models = len(self.fitted_models)
        
        for model in self.fitted_models :
            try:
                p = model.predict_proba(X).get()
            except:
                p = model.predict_proba(X)
            preds += p

        preds = preds / n_models
        return preds

    def fit_predict(self, X, y, **params):
        return self.fit(X,y).predict_proba(X)  


estimators = []
for my_c in [0.05, 1, 5, 10, 15] :
    estimators.append(
        LogisticRegression(**{'C'           : my_c, 
                             'max_iter'     : 10_000, 
                             'tol'          : 3.60e-06, 
                             'penalty'      : 'l2', 
                             'solver'       : 'qn',
                             'class_weight' : "balanced",
                             }
                           ),
    )
  
mymodel = VotingModelMaker(estimators)


%%time 

target = "target1"

if test_req :
    char_vec  = TFIDF(
        ngram_range=(1, 4), 
        analyzer='char', 
        max_df=0.95, 
        min_df=2, 
        max_features = 10
    )
    
    word_vec = TFIDF(
        analyzer="word", 
        ngram_range=(1,3), 
        min_df = 3, 
        max_df = 0.90,   
        stop_words="english", 
        dtype=np.float32,
        max_features = 10
    )
    
else:
    char_vec  = TFIDF(
        ngram_range=(1, 4), 
        analyzer='char', 
        max_df=0.95,
        min_df=2, 
        max_features = 10000
    )

    word_vec = TFIDF(
        analyzer="word", 
        ngram_range=(1,3), 
        min_df = 3, 
        max_df = 0.90,   
        stop_words="english", 
        dtype=np.float32,
    )

char_vec.fit(cudf.Series(pd.concat([train.sentence, test.sentence])))
word_vec.fit(cudf.Series(pd.concat([train.sentence, test.sentence])))

train_char = char_vec.transform(cudf.Series(train.sentence))
train_word = word_vec.transform(cudf.Series(train.sentence))
test_char  = char_vec.transform(cudf.Series(test.sentence))
test_word  = word_vec.transform(cudf.Series(test.sentence))

Xtrain = sparse.hstack([
    train_char.get(), 
    train_word.get(), 
    sparse.csr_matrix(X_numeric)
]
).tocsr()

Xtest  = sparse.hstack([
    test_char.get() , 
    test_word.get() ,
    sparse.csr_matrix(X_numeric_test),
]
).tocsr()
del train_char, train_word, test_char, test_word

OOF_Preds1 = []
Mdl_Preds1 = []
cv         = StratifiedKFold(n_splits= n_splits, shuffle=True, random_state= state)
ytrain     = train[target].values

print(f"\n---> Shapes = {Xtrain.shape} {ytrain.shape} {Xtest.shape}\n")

for fold_nb, (train_idx, dev_idx) in tqdm( 
    enumerate(cv.split(Xtrain, ytrain), start = 1), target
):

    Xtr  = Xtrain[train_idx]
    Xdev = Xtrain[dev_idx]
    ytr  = ytrain[train_idx]
    ydev = ytrain[dev_idx]
    
    model = clone(mymodel)
    model.fit(Xtr, ytr)

    dev_preds  = model.predict_proba(Xdev)
    test_preds = model.predict_proba(Xtest)
    
    OOF_Preds1.append(pd.DataFrame(dev_preds, index = dev_idx))
    Mdl_Preds1.append(pd.DataFrame(test_preds))
    
    del (test_preds, dev_preds, Xtr, Xdev, ytr, ydev)

    if fold_nb <= 9 :
        print(f"---> Fold{fold_nb}  training complete")
    else:
        print(f"---> Fold{fold_nb} training complete")

print()
del Xtrain, Xtest


%%time 

target = "target2"

if test_req :
    char_vec  = TFIDF(
        ngram_range=(1, 3), 
        analyzer='char', 
        max_df=0.95, 
        min_df=2, 
        max_features = 10
    )
    
    
    word_vec = TFIDF(
        analyzer="word", 
        ngram_range=(1,2), 
        min_df = 3, 
        max_df = 0.90,   
        stop_words="english", 
        dtype=np.float32,
        max_features = 10
    )
    
else:
    char_vec  = TFIDF(
        ngram_range=(1, 3), 
        analyzer='char', 
        max_df=0.95,
        min_df=2, 
        max_features = 5000
    )

    word_vec = TFIDF(
        analyzer="word", 
        ngram_range=(1,2), 
        min_df = 3, 
        max_df = 0.90,   
        stop_words="english", 
        dtype=np.float32,
    )
    
char_vec.fit(cudf.Series(pd.concat([train.sentence, test.sentence])))
word_vec.fit(cudf.Series(pd.concat([train.sentence, test.sentence])))

train_char = char_vec.transform(cudf.Series(train.sentence))
train_word = word_vec.transform(cudf.Series(train.sentence))
test_char  = char_vec.transform(cudf.Series(test.sentence))
test_word  = word_vec.transform(cudf.Series(test.sentence))

Xtrain = sparse.hstack([
    train_char.get(), 
    train_word.get(), 
    sparse.csr_matrix(X_numeric)
]
).tocsr()

Xtest  = sparse.hstack([
    test_char.get() , 
    test_word.get() ,
    sparse.csr_matrix(X_numeric_test),
]
).tocsr()
del train_char, train_word, test_char, test_word

OOF_Preds2 = []
Mdl_Preds2 = []
cv         = StratifiedKFold(n_splits= n_splits, shuffle=True, random_state= state)
ytrain     = train[target].values

print(f"\n---> Shapes = {Xtrain.shape} {ytrain.shape} {Xtest.shape}\n")

for fold_nb, (train_index, dev_idx) in tqdm(
    enumerate(cv.split(Xtrain, ytrain), start = 1), target
):

    Xtr  = Xtrain[train_idx]
    Xdev = Xtrain[dev_idx]
    ytr  = ytrain[train_idx]
    ydev = ytrain[dev_idx]
    
    model = clone(mymodel)
    model.fit(Xtr, ytr)

    dev_preds  = model.predict_proba(Xdev)
    test_preds = model.predict_proba(Xtest)
    
    OOF_Preds2.append(pd.DataFrame(dev_preds, index = dev_idx))
    Mdl_Preds2.append(pd.DataFrame(test_preds))
    
    del (test_preds, dev_preds, Xtr, Xdev, ytr, ydev)
    if fold_nb <= 9 :
        print(f"---> Fold{fold_nb}  training complete")
    else:
        print(f"---> Fold{fold_nb} training complete")

print()
del Xtrain, Xtest


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

oof_1   = pd.concat(OOF_Preds1, axis=0).sort_index(ascending = True).to_numpy()
oof_2   = pd.concat(OOF_Preds2, axis=0).sort_index(ascending = True).to_numpy()

map_inverse1 = {map_target1[k]:k for k in map_target1}
map_inverse2 = {map_target2[k]:k for k in map_target2}

oof_2[:, 0] = 0
predicted1  = np.argsort(-oof_1, 1)[:,:3]
predicted2  = np.argsort(-oof_2, 1)[:,:3]

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

print(
    f"\n---> Final OOF score = {map3(train['target_cat'].tolist(), predict) :,.8f}\n"
)


%%time 

pred_1 = pd.concat(Mdl_Preds1, axis=0).sort_index(ascending = True).groupby(level=0).mean().to_numpy()
pred_2 = pd.concat(Mdl_Preds2, axis=0).sort_index(ascending = True).groupby(level=0).mean().to_numpy()

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

sub_fl = pd.read_csv(
    "/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv",
)

sub_fl['Category:Misconception'] = predict
sub_fl.to_csv("submission.csv", index = None)

print()
!ls
print()
!head submission.csv

