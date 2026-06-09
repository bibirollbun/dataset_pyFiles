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


df_train = pd.read_csv('/kaggle/input/nlp-2025-experiment-task-4-delay/train.csv',encoding='utf-8',encoding_errors='ignore',index_col='Id')
df_test = pd.read_csv('/kaggle/input/nlp-2025-experiment-task-4-delay/test.csv',encoding='utf-8',encoding_errors='ignore',index_col='Id')



df_train.head()


df_train['Category'].value_counts().to_dict()


label_map = {'PlayMusic': 0,
 'RateBook': 1,
 'SearchCreativeWork': 2,
 'AddToPlaylist': 3,
 'GetWeather': 4,
 'BookRestaurant': 5,
 'SearchScreeningEvent': 6}


df_train['Category']=df_train['Category'].map(label_map)


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    pattern = r'[^\w\s]'
    text = re.sub(pattern,' ',text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


df_train['Sentence'] = df_train['Sentence'].apply(clean_text)
df_test['Sentence'] = df_test['Sentence'].apply(clean_text)


df_train.head()


from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
X_train,X_val,y_train,y_val = train_test_split(df_train['Sentence'],df_train['Category'],test_size=0.2,random_state=42,stratify =df_train['Category'] )


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(df_test['Sentence'])


!pip install optuna
import optuna
from xgboost import XGBClassifier
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "tree_method":"hist",
        "device":"cuda",
        "random_state": 42,
    }
    model = XGBClassifier(**params,n_jobs=-1)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_tfidf,y_train, scoring='accuracy',cv=skf)
    print(scores)
    return scores.mean()


study = optuna.create_study(direction='maximize',sampler = optuna.samplers.TPESampler())
study.optimize(objective,n_trials=50)
print('Best Trial:',study.best_trial)


best_params_optuna = study.best_trial.params
best_params_optuna


xgb_bayes = XGBClassifier(**best_params_optuna,tree_method = "hist", device = "cuda",random_state=42)
xgb_bayes.fit(X_train_tfidf,y_train)


pred_xgb = xgb_bayes.predict(X_val_tfidf)
pred_xgb


from sklearn.metrics import accuracy_score



accuracy = accuracy_score(y_val,pred_xgb)
print(accuracy)


reverse_map = {value:key for key,value in label_map.items()}
reverse_map


y_test_pred = xgb_bayes.predict(X_test_tfidf)
y_test_pred=[reverse_map[i] for i in y_test_pred]
y_test_pred


sample_sub=pd.read_csv('/kaggle/input/nlp-2025-experiment-task-4-delay/text_classfication_submission_sample.csv')


sample_sub['Category'] = y_test_pred


sample_sub.head()


sample_sub.to_csv('samplesubmission1.csv',index=False)

