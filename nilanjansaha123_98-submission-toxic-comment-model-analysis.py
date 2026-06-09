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


import optuna
import re,string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from transformers import AutoTokenizer


import zipfile

# Path to your zip file
zip_path = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip'

# Open the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # List contents (optional)
    print(zip_ref.namelist())

    # Extract and read the CSV directly
    with zip_ref.open('train.csv') as file:
        df_train = pd.read_csv(file)

# Preview the DataFrame
df_train.head()


import zipfile

# Path to your zip file
zip_path = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip'

# Open the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # List contents (optional)
    print(zip_ref.namelist())

    # Extract and read the CSV directly
    with zip_ref.open('test.csv') as file:
        df_test = pd.read_csv(file)

# Preview the DataFrame
df_test.head()


import zipfile

# Path to your zip file
zip_path = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip'

# Open the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # List contents (optional)
    print(zip_ref.namelist())

    # Extract and read the CSV directly
    with zip_ref.open('sample_submission.csv') as file:
        df_sample = pd.read_csv(file)

# Preview the DataFrame
df_sample.head()


y_labels=['toxic','severe_toxic','obscene','threat','insult','identity_hate']


def preprocessor(content):
    url_remover=re.compile(r'https?:[^\s]+')
    hash_remover=re.compile(r'#[^\s]+')
    gmail_remover=re.compile(r'[^\s]+@gmail.com')
    username_remover=re.compile(r'@[^\s]+')
    line_remover=re.compile(r'\n')
    number_remover=re.compile(r'[0-9]{3}-[0-9]{3}-[0-9]{4}')
    #processing
    content=url_remover.sub('',content)
    content=hash_remover.sub('',content)
    content=gmail_remover.sub('',content)
    content=username_remover.sub('',content)
    content=line_remover.sub('',content)
    content=number_remover.sub('',content)
    return content
df_train['clean_text']=df_train['comment_text'].apply(preprocessor)
df_test['clean_text']=df_test['comment_text'].apply(preprocessor)


#This custom tokennizer isnt getting used now. Previously experimented and BERT tokenizer performed better
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenizer(s): return re_tok.sub(r' \1 ', s).split()


tokenizer3=AutoTokenizer.from_pretrained('bert-base-uncased')


import gc
import numpy as np
from scipy import sparse
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- stable r computation with smoothing ---
def compute_r(X_csr, y, eps=1e-9):
    """
    X_csr: scipy sparse matrix (n_samples, n_features)
    y: 1D numpy array of 0/1 labels
    returns: r as 1D numpy float32 array (n_features,)
    """
    Xc = X_csr.tocsr()
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    # sum returns 1xF sparse -> convert to dense 1D
    p_pos = Xc[y==1].sum(axis=0).A1.astype(np.float64)
    p_neg = Xc[y==0].sum(axis=0).A1.astype(np.float64)

    # Laplace smoothing (same idea as original)
    pos_freq = (p_pos + 1.0) / (n_pos + 1.0)
    neg_freq = (p_neg + 1.0) / (n_neg + 1.0)

    # safe log ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.log(pos_freq / neg_freq)

    # replace inf/nan with 0 (no effect)
    r = np.nan_to_num(r, posinf=0.0, neginf=0.0, nan=0.0)
    return r.astype(np.float32)

# --- main training function ---
def models_and_score(feature_name='clean_text', y_labels=y_labels,
                     tokenizer=tokenizer3.tokenize, ngram_range=(1,2),
                     stratify=True, avg_per_class_weights=True,
                     tfidf_kwargs=None, xgb_base_params=None):
    models = {}
    tfidf_kwargs = {} if tfidf_kwargs is None else tfidf_kwargs
    xgb_base_params = {} if xgb_base_params is None else xgb_base_params

    vectorizer = TfidfVectorizer(ngram_range=ngram_range, tokenizer=tokenizer,
                                 strip_accents='unicode', use_idf=1, smooth_idf=1,
                                 sublinear_tf=1, min_df=3, max_df=0.9, **tfidf_kwargs)

    X_text = df_train[feature_name].values
    print(f'\nVectorizing input text({feature_name})')
    X_base = vectorizer.fit_transform(X_text)        # sparse csr (n_samples, n_features)
    print('Vectorizing input text done')
    print("Vocabulary size:", len(vectorizer.vocabulary_))

    # Ensure float32 (sparse) to reduce memory
    X_base = X_base.astype(np.float32)

    # default XGBoost params corrected for GPU
    default_xgb = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'gpu_id': 0,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        # you can override n_estimators, max_depth, etc. via xgb_base_params
    }
    default_xgb.update(xgb_base_params)

    for label in y_labels:
        print('\nFinding Best Model for', label)
        y = df_train[label].values.astype(int)

        # make a copy per label so we don't mutate the base matrix
        X = X_base.copy()

        r = None
        if avg_per_class_weights:
            r = compute_r(X, y)            # numpy 1D float32
            # multiply: sparse_matrix.multiply accepts 1D array (broadcasts on columns)
            X = X.multiply(r)              # returns new sparse matrix
            # keep float32
            X = X.astype(np.float32)
            print('Weights computed and multiplied')

        # ensure X exists even if avg_per_class_weights False
        # split
        if stratify:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression() #XGBClassifier(**model_params)

        # IMPORTANT: use eval_set + early stopping to avoid huge training times
        model.fit(X, y)
        #y_pred=model.predict_proba(X_val)
        #score=roc_auc_score(y_val,y_pred[:,1])
        #print(f'\nThe ROC-AUC score for label-{label} is {score*100}%')
        models[label] = (model, r)
        print(f'{label} training done')
        print('-'*80)

        # free memory between labels
        del X, X_train, X_val
        gc.collect()

    return models, vectorizer

# --- inference function (unchanged but robust) ---
def result_dataframe(df_test, models, vectorizer, feature_name='clean_text'):
    y_final = []
    X_test_text = df_test[feature_name].values
    ids = df_test['id'].values.reshape(-1, 1)
    y_final.append(ids)
    labels = ['id']

    X_test = vectorizer.transform(X_test_text).astype(np.float32)  # sparse csr float32

    for label, (model_curr, r) in models.items():
        X_test_temp = X_test.copy()
        if r is not None:
            # multiply must be assigned
            X_test_temp = X_test_temp.multiply(r)
        # predict_proba should accept csr
        y_pred = model_curr.predict_proba(X_test_temp)[:, 1].reshape(-1, 1)
        y_final.append(y_pred)
        labels.append(label)

    y_pred_final = np.concatenate(y_final, axis=1)
    res = pd.DataFrame(y_pred_final, columns=labels)
    return res



#tokenizer3=AutoTokenizer.from_pretrained('bert-base-uncased')
models4,vectorizer4=models_and_score(tokenizer=tokenizer3.tokenize,ngram_range=(1,2))


res_final=result_dataframe(df_test, models4, vectorizer4, feature_name='clean_text')


res_final.to_csv('toxic_output_1-2ngram_bert_full.csv',index=False)




