!pip install -q sentence-transformers



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import Pipeline
import optuna



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Reading the data
train_data=pd.read_csv("/kaggle/input/mercor-ai-detection/train.csv")
test_data=pd.read_csv("/kaggle/input/mercor-ai-detection/test.csv")
sample_sub_data=pd.read_csv("/kaggle/input/mercor-ai-detection/sample_submission.csv")


#Creating a copy of data
train=train_data.copy()
test=test_data.copy()


def get_cleaned_string(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)   # remove punctuation/symbols
    text = ' '.join(text.split())  
    return text

def get_cleaned_text_for_transformer(text):
    text=' '.join(text.split())
    text=text.lower()
    return text
    
def add_stopword_counts(df):
    stop_words = set(stopwords.words('english'))   # Loaded once per function call

    def count_stopwords(text):
        text = ' '.join(text.split()).lower()
        words = text.split()
        return sum(1 for word in words if word in stop_words)

    df['topic_stopword_count'] = df['topic'].apply(count_stopwords)
    df['answer_stopword_count'] = df['answer'].apply(count_stopwords)
    return df

def get_punctuation_count(text):
    text=' '.join(text.split())
    count=len(re.findall(r'[.,!?;:\'\"()\[\]{}\-\—…]', text))
    return count
    
def get_avg_word_length(text):
    text = ' '.join(text.split())
    text_clean = re.sub(r'[^\w\s]', '', text).lower()
    word_list = text_clean.split()
    if not word_list:
        return 0.0
    total_chars = len(''.join(word_list))
    total_words = len(word_list)    
    avg_word_len = total_chars / total_words
    return avg_word_len


def get_capital_words_count(text):
    text = ' '.join(text.split())
    text=re.sub(r'[^A-Za-z\s]', '', text)
    text_list=text.split()
    uppercase_words_count=len([word for word in text_list if word.isupper()])
    return uppercase_words_count

def get_number_count(text):
    text = ' '.join(text.split())
    number_count=len(re.findall(r'\d+',text))
    return number_count

def get_symbols_count(text):
    text=' '.join(text.split())
    count=len(re.findall(r'[@#$%^&*+=|\\/<>~_]', text))
    return count


#characters count function
def get_characters_count(text):
    return len(text)

#Words count
def get_words_count(text):
    return len(text.split())

#Unique words count
def get_unique_words_count(text):
    return len(set(text.split()))

def get_tfidf_vectorizer(topic, answer):
    vectorizer=TfidfVectorizer(stop_words='english')
    corpus=topic.tolist()+answer.tolist()
    vectorizer.fit(corpus)
    return vectorizer


#Cosine similarity score
def get_similarity_score(df,vectorizer):
    topic_tfidf=vectorizer.transform(df['cleaned_topics'])
    answer_tfidf=vectorizer.transform(df['cleaned_answers'])
    scores=[]
    for a,b in zip(topic_tfidf, answer_tfidf):
        scores.append(cosine_similarity(a,b)[0][0])

    df['topic_answer_similarity_score']=scores
    return df



def add_text_features(df):
    # Cleaned text
    df['cleaned_topics'] = df['topic'].apply(get_cleaned_string)
    df['cleaned_answers'] = df['answer'].apply(get_cleaned_string)


    # Stopwords count
    df=add_stopword_counts(df)
    
    # Characters count
    df['topic_character_count'] = df['topic'].apply(get_characters_count)
    df['answer_character_count'] = df['answer'].apply(get_characters_count)

    # Words count
    df['topic_word_count'] = df['topic'].apply(get_words_count)
    df['answer_word_count'] = df['answer'].apply(get_words_count)

    # Unique words count
    df['topic_unique_word_count'] = df['topic'].apply(get_unique_words_count)
    df['answer_unique_word_count'] = df['answer'].apply(get_unique_words_count)


    # Punctuation count
    df['topic_punctuation_count'] = df['topic'].apply(get_punctuation_count)
    df['answer_punctuation_count'] = df['answer'].apply(get_punctuation_count)

    # Average word length
    df['topic_avg_word_length'] = df['topic'].apply(get_avg_word_length)
    df['answer_avg_word_length'] = df['answer'].apply(get_avg_word_length)

    # Capital words count
    df['topic_capital_words_count'] = df['topic'].apply(get_capital_words_count)
    df['answer_capital_words_count'] = df['answer'].apply(get_capital_words_count)

    # Number count
    df['topic_number_count'] = df['topic'].apply(get_number_count)
    df['answer_number_count'] = df['answer'].apply(get_number_count)

    # Symbol count
    df['topic_symbol_count'] = df['topic'].apply(get_symbols_count)
    df['answer_symbol_count'] = df['answer'].apply(get_symbols_count)
    

    return df



#Getting static feature
X,y=train.drop(columns=['is_cheating']),train['is_cheating']
X=add_text_features(X)



Best_params_xgb = {
    'n_estimators': 193,
    'max_depth': 5,
    'learning_rate': 0.281507973368965,
    'subsample': 0.6518902295271632,
    'colsample_bytree': 0.9690796836434973,
    'reg_lambda': 0.3005743734576379,
    'reg_alpha': 0.08956184517935871,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}



# --- Setup and Initialization ---
N_FOLDS = 5
oof_preds_xgb = np.zeros(X.shape[0]) # Initialize OOF array with size of X
SEED = 42
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for i, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"--- Fold {i} started ---")

    X_train,X_val=X.iloc[train_index],X.iloc[val_index]
    y_train,y_val=y[train_index],y[val_index]

    
    vectorizer=get_tfidf_vectorizer(X_train['cleaned_topics'],X_train['cleaned_answers'])
    X_train=get_similarity_score(X_train,vectorizer)
    X_val=get_similarity_score(X_val,vectorizer)

    train_topic_tfidf = vectorizer.transform(X_train['cleaned_topics'])
    train_answer_tfidf = vectorizer.transform(X_train['cleaned_answers'])

    val_topic_tfidf = vectorizer.transform(X_val['cleaned_topics'])
    val_answer_tfidf = vectorizer.transform(X_val['cleaned_answers'])

    X_train=X_train.drop(columns=['id','topic','answer','cleaned_topics','cleaned_answers'],axis=1)
    X_train=np.concatenate((np.array(X_train),train_topic_tfidf.toarray(),train_answer_tfidf.toarray()),axis=1)

    X_val=X_val.drop(columns=['id','topic','answer','cleaned_topics','cleaned_answers'],axis=1)
    X_val=np.concatenate((np.array(X_val),val_topic_tfidf.toarray(),val_answer_tfidf.toarray()),axis=1)

    xgb=XGBClassifier(**Best_params_xgb)
    xgb.fit(X_train,y_train)
    
    y_pred_proba = xgb.predict_proba(X_val)[:, 1]
    oof_preds_xgb[val_index] = y_pred_proba # Store results in the main OOF array using the correct indices
    
    auc_score = roc_auc_score(y_val, y_pred_proba)
    print(f"Fold {i} AUC score is {auc_score:.4f}")

# --- Final Evaluation ---
overall_oof_auc = roc_auc_score(y, oof_preds_xgb)
print(f"\nOverall OOF AUC: {overall_oof_auc:.4f}")
    



oof_preds_xgb


train['cleaned_topic_for_transformer']=train['topic'].apply(get_cleaned_text_for_transformer)
train['cleaned_answer_for_transformer']=train['answer'].apply(get_cleaned_text_for_transformer)


model = SentenceTransformer('all-MiniLM-L6-v2')
topic_embeddings = model.encode(train['cleaned_topic_for_transformer'].tolist(), show_progress_bar=True)
answer_embeddings = model.encode(train['cleaned_answer_for_transformer'].tolist(), show_progress_bar=True)


semantic_diff = topic_embeddings - answer_embeddings 
semantic_product = topic_embeddings * answer_embeddings


X=np.hstack((topic_embeddings,answer_embeddings,semantic_diff,semantic_product))
y=train['is_cheating']


# def objective(trial):
#     C = trial.suggest_loguniform('C', 1e-4, 1e2)
#     penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
    
#     # Choose valid solver based on penalty
#     solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'saga'])
#     if penalty == 'l1' and solver not in ['liblinear', 'saga']:
#         raise optuna.TrialPruned()  # invalid combo
#     if penalty == 'elasticnet' and solver != 'saga':
#         raise optuna.TrialPruned()
    
#     l1_ratio = None
#     if penalty == 'elasticnet':
#         l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
    
#     class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
    
#     model = Pipeline([
#         ('scaler', StandardScaler()),
#         ('clf', LogisticRegression(
#             C=C,
#             penalty=penalty,
#             solver=solver,
#             l1_ratio=l1_ratio,
#             class_weight=class_weight,
#             max_iter=500,
#             random_state=42
#         ))
#     ])
    
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc').mean()
#     return auc

# # -------------------------------
# # 3️⃣ Run Optuna study
# # -------------------------------
# study = optuna.create_study(direction='maximize')  # maximize AUC
# study.optimize(objective, n_trials=30)

# # -------------------------------
# # 4️⃣ Display best results
# # -------------------------------
# print("Best trial:")
# print(f"  ROC-AUC Value: {study.best_trial.value:.4f}")
# print("  Best Parameters:")
# for key, value in study.best_trial.params.items():
#     print(f"    {key}: {value}")



best_params_lr={'C': 0.011904536214543474,
 'penalty': 'l2',
 'solver': 'saga',
 'class_weight': None}


skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_lr = np.zeros(X.shape[0]) # Initialize OOF array with size of X

for i,(train_idx,val_idx) in enumerate(skf.split(X, y)):
    print(f"--- Fold {i} started ---")

    X_train,X_val=X[train_idx],X[val_idx]
    y_train,y_val=y[train_idx],y[val_idx]

    ss=StandardScaler()
    X_train=ss.fit_transform(X_train)
    X_val=ss.transform(X_val)
    
    lr=LogisticRegression(**best_params_lr)
    lr.fit(X_train,y_train)

    y_pred_proba = lr.predict_proba(X_val)[:, 1]
    oof_preds_lr[val_idx] = y_pred_proba # Store results in the main OOF array using the correct indices
    
    auc_score = roc_auc_score(y_val, y_pred_proba)
    print(f"Fold {i} AUC score is {auc_score:.4f}")

# --- Final Evaluation ---
overall_oof_auc = roc_auc_score(y, oof_preds_lr)
print(f"\nOverall OOF AUC: {overall_oof_auc:.4f}")


X_meta = np.column_stack([
    oof_preds_xgb, # The structural/behavioral signal
    oof_preds_lr   # The semantic/linear signal
])

# Use the true labels for training
y_meta = train_data['is_cheating']


scaler = StandardScaler()
X_meta_scaled = scaler.fit_transform(X_meta)
final_meta_learner = LogisticRegression(
    solver='liblinear', 
    C=0.01,              # Small regularization is generally good
    random_state=42
)

final_meta_learner.fit(X_meta_scaled, y_meta)

final_oof_proba = final_meta_learner.predict_proba(X_meta_scaled)[:, 1]
final_auc = roc_auc_score(y_meta, final_oof_proba)


final_auc


X,y=train_data.drop(columns=['is_cheating']),train['is_cheating']



X_train=add_text_features(X)
y_train=y
X_test=add_text_features(test)

vectorizer=get_tfidf_vectorizer(X_train['cleaned_topics'],X_train['cleaned_answers'])
X_train=get_similarity_score(X_train,vectorizer)
X_test=get_similarity_score(X_test,vectorizer)

train_topic_tfidf = vectorizer.transform(X_train['cleaned_topics'])
train_answer_tfidf = vectorizer.transform(X_train['cleaned_answers'])

test_topic_tfidf = vectorizer.transform(X_test['cleaned_topics'])
test_answer_tfidf = vectorizer.transform(X_test['cleaned_answers'])

X_train=X_train.drop(columns=['id','topic','answer','cleaned_topics','cleaned_answers'],axis=1)
X_train=np.concatenate((np.array(X_train),train_topic_tfidf.toarray(),train_answer_tfidf.toarray()),axis=1)

X_test=X_test.drop(columns=['id','topic','answer','cleaned_topics','cleaned_answers'],axis=1)
X_test=np.concatenate((np.array(X_test),test_topic_tfidf.toarray(),test_answer_tfidf.toarray()),axis=1)

xgb=XGBClassifier(**Best_params_xgb)
xgb.fit(X_train,y_train)

oof_test_xgb = xgb.predict_proba(X_test)[:, 1]




X_train=X
y_train=y
X_test=test

X_train['cleaned_topic_for_transformer']=X_train['topic'].apply(get_cleaned_text_for_transformer)
X_train['cleaned_answer_for_transformer']=X_train['answer'].apply(get_cleaned_text_for_transformer)

X_test['cleaned_topic_for_transformer']=X_test['topic'].apply(get_cleaned_text_for_transformer)
X_test['cleaned_answer_for_transformer']=X_test['answer'].apply(get_cleaned_text_for_transformer)

model = SentenceTransformer('all-MiniLM-L6-v2')
topic_embeddings_train = model.encode(X_train['cleaned_topic_for_transformer'].tolist(), show_progress_bar=True)
answer_embeddings_train = model.encode(X_train['cleaned_answer_for_transformer'].tolist(), show_progress_bar=True)

topic_embeddings_test = model.encode(X_test['cleaned_topic_for_transformer'].tolist(), show_progress_bar=True)
answer_embeddings_test= model.encode(X_test['cleaned_answer_for_transformer'].tolist(), show_progress_bar=True)

semantic_diff_train = topic_embeddings_train - answer_embeddings_train 
semantic_product_train = topic_embeddings_train * answer_embeddings_train

semantic_diff_test = topic_embeddings_test - answer_embeddings_test 
semantic_product_test = topic_embeddings_test * answer_embeddings_test

X_train=np.hstack((topic_embeddings_train,answer_embeddings_train,semantic_diff,semantic_product))
X_test=np.hstack((topic_embeddings_test,answer_embeddings_test,semantic_diff_test,semantic_product_test))

ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

lr=LogisticRegression(**best_params_lr)
lr.fit(X_train,y_train)

oof_test_lr = lr.predict_proba(X_test)[:, 1]



oof_test_lr.shape


## Final Stacking and Prediction Block (Corrected)

# 1. Create Meta-Feature Matrices (X_meta_train and X_meta_test)
X_meta_train = np.column_stack([
    oof_preds_xgb, # P_XGBoost
    oof_preds_lr   # P_LLM
])

X_meta_test = np.column_stack([
    oof_test_xgb, # P_XGBoost for test set
    oof_test_lr   # P_LLM for test set
])

y_meta = train_data['is_cheating']

# 2. Scaling (CRITICAL STEP)
# Fit scaler ONLY on the TRAIN meta-features
scaler = StandardScaler()
X_meta_train_scaled = scaler.fit_transform(X_meta_train)

# Transform the TEST meta-features using the SAME fitted scaler
X_meta_test_scaled = scaler.transform(X_meta_test) 

# 3. Train Final Meta-Learner
# Note: You used C=0.01. I'll keep that, but tuning this is recommended!
final_meta_learner = LogisticRegression(
    solver='liblinear', 
    C=0.01,
    random_state=42
)

final_meta_learner.fit(X_meta_train_scaled, y_meta)

# 4. Generate Final Predictions (using the scaled test features)
# Predict probabilities for final submission
final_y_pred_proba = final_meta_learner.predict_proba(X_meta_test_scaled)[:, 1]

# You can also get binary predictions if needed:
# final_y_pred = final_meta_learner.predict(X_meta_test_scaled)


submission=test[['id']]
submission['is_cheating']=final_y_pred_proba
submission.to_csv('submission.csv', index=False)







