# Upgrade SKLearn
!pip install -qq scikit-learn==1.6.1
!pip install --upgrade xgboost -qq


#
# Libraries
#

# General
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, string, re, random, gc, pickle, math,warnings
import json
from itertools import *
from datetime import date
from tqdm.keras import TqdmCallback
from tqdm import tqdm
import nltk

# Optuna
import optuna

# Smote
from imblearn.over_sampling import *

# Spacy
import spacy

# XGBoost
import xgboost as xgb
from xgboost import XGBClassifier 

# NLTK & Txt
import spacy
from nltk.corpus import stopwords
nlp = spacy.load('en_core_web_sm')



# Sklearn
from sklearn.model_selection import *
from sklearn.feature_extraction import *
from sklearn.metrics import *
from sklearn.metrics import pairwise
from sklearn.preprocessing import *
from sklearn.utils import *
from sklearn.pipeline import *
from sklearn.compose import *
from sklearn.ensemble import *

# Stats
import scipy
from scipy.stats import *
from scipy.sparse import csr_matrix, hstack

# Setting
pd.set_option('max_colwidth',None)
seed = 805
warnings.simplefilter('ignore')
stopw = pd.read_json('/kaggle/input/english-stopwords/stop_words_english.json')
stopw = stopw.Stopwords.tolist()
nlp = spacy.load('en_core_web_sm')

data_path = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith('csv'):
            data_path.append(os.path.join(dirname, filename))


#
# Load Data
#

# Training Data
train_ds = pd.read_csv(data_path[1])

# Test Data
test_ds = pd.read_csv(data_path[2])

# Sample Submission Data
subm_ds = pd.read_csv(data_path[0])

# View
print(f"Training Data Shape: {train_ds.shape} \nTest Data Shape: {test_ds.shape} \n\n")
train_ds.head()


#
# Custom Function - Math Text Cleanse
#

def clean_string(text, stem="None"):
    """
    Text cleansing.

    Args:
        text: The input text.
        stem: 
            Stem = Stemming
            Lem = Lemmatization
            Spacy = Lemmatization using Spacy

    Returns:
        cleaned text
    """

    # final string
    final_string = ""

    # convert to string
    text = str(text)

    # mathematical symbols
    math_symbols = {
    r'\$': ' dollar ',
    r'\=': ' equals ',
    r'\<': ' less than ',
    r'\>': ' greater than ',
    r'\+': ' plus ',
    r'\-': ' minus ',
    r'\*': ' times ',
    r'\/': ' divided by ',
    r'\^': ' to the power of ',
    r'\√': ' square root ',
    r'\π': ' pi ',
    r'\∑': ' sum ',
    r'\∫': ' integral ',
    r'\∞': ' infinity ' }

    # replace math symbols
    for pattern, replacement in math_symbols.items():
            text = re.sub(pattern, replacement, text)

    # Remove weird chars
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    text = re.sub(r'\{([^}]*)\}', r' \1 ', text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\?\!]", " ", text)
    text = re.sub(r'(\d+)([a-zA-Z])', r'\1 \2', text)  
    text = re.sub(r'([a-zA-Z])(\d+)', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()

    # Remove line breaks
    text = re.sub(r'\n', '', text)
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Remove stop words
    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english")
    useless_words = useless_words + ['hi', 'im']

    text_filtered = [word for word in text if not word in useless_words]

    # Remove special chars - any residual !
    text_filtered = [re.sub(r'[^a-zA-Z0-9\s]', '', w) for w in text_filtered]
    
    # Remove Whitespaces
    text_filtered = [w.strip() for w in text_filtered]

    # Stem or Lemmatize
    if stem == 'Stem':
        stemmer = PorterStemmer() 
        text_stemmed = [stemmer.stem(y) for y in text_filtered]
    elif stem == 'Lem':
        lem = WordNetLemmatizer()
        text_stemmed = [lem.lemmatize(y) for y in text_filtered]
    elif stem == 'Spacy':
        text_filtered = nlp(' '.join(text_filtered))
        text_stemmed = [y.lemma_ for y in text_filtered]
    else:
        text_stemmed = text_filtered

    # Word > 3 letters only
    text_stemmed = [w for w in text_stemmed if len(w) >= 3]
	
    final_string = ' '.join(text_stemmed)
    
    return final_string


#
# Text Cleansing
#

# create copy of OG column
train_ds['Question_cpy'] = train_ds['Question'].copy()
test_ds['Question_cpy'] = test_ds['Question'].copy()

# Cleaning
train_ds['Question'] = train_ds['Question'].apply(lambda x: clean_string(x,stem='Spacy'))
test_ds['Question'] = train_ds['Question'].apply(lambda x: clean_string(x,stem='Spacy'))

# View
train_ds.head()


#
# Custom Function - Extract Features
#

def extract_features(txt):
    """
    Feature Extraction.

    Args:
        txt: The input text.

    Returns:
        series of features
    """    
    features = {
        'num_count': len(re.findall(r'\d+', txt)),
        'equation_count': len(re.findall(r'equals|equation|formula|solve for', txt)),
        'function_count': len(re.findall(r'function|f\(x\)|derivative|integral', txt)),
        'geometry_count': len(re.findall(r'angle|triangle|circle|area|volume', txt)),
        'algebra_count': len(re.findall(r'variable|polynomial|matrix|vector', txt)),
        'calculus_count': len(re.findall(r'derivative|integral|limit|differentiation', txt)),
        'char_count': len(txt),
        'word_count': len(txt.split()),
        'is_proof': int('prove' in txt or 'show that' in txt),
        'is_compute': int('compute' in txt or 'calculate' in txt),
        'is_find': int('find' in txt or 'determine' in txt),
    }
    
    return pd.Series(features)


#
# Feature Extraction
#

feats_train = train_ds['Question'].apply(extract_features)
feats_test = test_ds['Question'].apply(extract_features)


# Feature Vectors
tfidf = text.TfidfVectorizer(max_features=15000,  ngram_range=(1,2))

x_train_txt = tfidf.fit_transform(train_ds['Question'])
x_test_txt = tfidf.transform(test_ds['Question'])


# Stack 'em up !

x_train = hstack([x_train_txt, csr_matrix(feats_train.values)])
x_test = hstack([x_test_txt, csr_matrix(feats_test.values)])

y = train_ds['label']

# view
print(f"Train: {x_train.shape} \nTest: {x_test.shape} ")


#
# SMOTE (Synthetic Minority Over-sampling Technique)
#

# instantiate
smote = BorderlineSMOTE(random_state=seed)

# Apply
x_train_smote = x_train.toarray()
x_resampled, y_resampled = smote.fit_resample(x_train_smote, y)
x_resampled_sparse = csr_matrix(x_resampled)

# view
print(f"Before SMOTE: {dict(pd.Series(y).value_counts())}")
print(f"After SMOTE:  {dict(pd.Series(y_resampled).value_counts())}")


#
# Label Encoding & Train-Test Split
#

# encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y_resampled)

# train-test Split
x_train, x_val, y_train, y_val = train_test_split(x_resampled, y_resampled, test_size=0.05, random_state=seed)

# view
print(f"Training size: {len(x_train)} \nValidation size: {len(x_val)}")


#
# Training Model - Baseline
#

# baseline model
base_clf = RandomForestClassifier(random_state=seed)

# model fit
base_clf.fit(x_train,y_train)

# prediction
preds = base_clf.predict(x_val)

# evaluation
f1 = f1_score(y_val,preds,average='micro')
print(f"Baseline F1 Score (micro): {f1:.4f}")


#
# Custom Function - Optimization using StratifiedKFold
#

def kfold_rfc(X, y, n_splits=10):
    
    oof_preds = np.zeros(len(y), dtype=int)
    models = []
    f1_micro_scores = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = RandomForestClassifier(random_state=seed)

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        f1_m = f1_score(y_val, preds, average='micro')
        print(f"F1 Micro: {f1_m:.4f}")

        f1_micro_scores.append(f1_m)
        models.append(model)

    overall_f1_m = f1_score(y, oof_preds, average='micro')
    print(f"\nOverall F1 Micro: {overall_f1_m:.4f}")

    return {
        'models': models,
        'oof_preds': oof_preds,
        'fold_f1_micro': f1_micro_scores,
        'overall_f1_micro': overall_f1_m
    }


# Run Trials
results = kfold_rfc(x_resampled_sparse, y_encoded)


y_true = np.array(y_encoded)
y_pred = np.array(results['oof_preds'])

unique_label_indices = np.unique(np.concatenate((y_true, y_pred)))
label_names = [str(cls) for cls in le.inverse_transform(unique_label_indices)]

print("\nClassification Report (RandomForest):")
print(classification_report(
    y_true,
    y_pred,
    labels=unique_label_indices,
    target_names=label_names
))


#
# Submission
#

test_features_combined = hstack([
    tfidf.transform(test_ds['Question']),
    feats_test.values
])

test_preds_encoded = results['models'][0].predict(test_features_combined)
test_preds = le.inverse_transform(test_preds_encoded)

submission = pd.DataFrame({
    'id': test_ds['id'],
    'label': test_preds
})

submission.to_csv('submission.csv', index=False)

