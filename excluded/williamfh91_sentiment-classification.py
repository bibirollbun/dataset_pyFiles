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


test_path              = "/kaggle/input/ml-olympiad-tfugsurabaya-2024/test.tsv"
train_path             = "/kaggle/input/ml-olympiad-tfugsurabaya-2024/train.tsv"
sample_submission_path = "/kaggle/input/ml-olympiad-tfugsurabaya-2024/sample_submission.csv"


!pip install nltk
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


df_train = pd.read_table(train_path, sep='\t')
df_test = pd.read_table(test_path, sep='\t')
df_submission = pd.read_csv(sample_submission_path)


df_train.head()



df_train.shape, df_test.shape


from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re




stop_words_indonesian = stopwords.words('indonesian')
stop_words_english = stopwords.words('english')
stop_words_combined = stop_words_indonesian + stop_words_english


def convert_to_lower(text):
    return text.lower()
def remove_numbers(text):
    number_pattern = r'\d+'
    without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
    return without_number

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    removed = []
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        if tokens[i] not in stop_words_combined:
            removed.append(tokens[i])
    return " ".join(removed)
def remove_extra_white_spaces(text):
    single_char_pattern = r'\s+[a-zA-Z]\s+'
    without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
    return without_sc
def lemmatizing(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        lemma_word = lemmatizer.lemmatize(tokens[i])
        tokens[i] = lemma_word
    return " ".join(tokens)


df = df_train.rename({'LABEL':'label','REVIEW':'text'},axis=1)
df.drop(['ID'],axis=1,inplace=True)
df['text_clean'] = df['text'].apply(lambda x: convert_to_lower(x))
df['text_clean'] = df['text_clean'].apply(lambda x: remove_numbers(x))
df['text_clean'] = df['text_clean'].apply(lambda x: remove_punctuation(x))
df['text_clean'] = df['text_clean'].apply(lambda x: remove_extra_white_spaces(x))
df['text_clean'] = df['text_clean'].apply(lambda x: remove_stopwords(x))
df['length_after_cleaning'] = df['text_clean'].apply(lambda x: len(x))
df.head()


X = df['text_clean']
y = df['label'].values


from sklearn.pipeline  import Pipeline 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# pakai metode unigram hingga trigram
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42, stratify=y)

def uni_tri_model(model_name,X_train,y_train,X_test,y_test):
    pipeline=Pipeline([
      ('cv', CountVectorizer(tokenizer=word_tokenize, ngram_range=(1,3), binary=True, min_df = 2)),
      ('model', model_name),
    ])
    print(model_name)
    pipeline.fit(X_train,y_train)

    preds=pipeline.predict(X_test)

    print (classification_report(y_test,preds))
    print (confusion_matrix(y_test,preds))
    print('Accuracy:', pipeline.score(X_test, y_test)*100)
    print("Training Score:",pipeline.score(X_train,y_train)*100)
    score = accuracy_score(y_test,preds)
    return score

mnb = uni_tri_model(MultinomialNB(),X_train,y_train,X_test,y_test)
maxent = uni_tri_model(LogisticRegression(),X_train,y_train,X_test,y_test)
svc = uni_tri_model(SVC(),X_train,y_train,X_test,y_test)
uni_tri_models = pd.DataFrame({
    'Model':['MNB','MaxEnt', 'SVM'],
    'Accuracy_score' :[mnb ,maxent, svc]
})
uni_tri_models.sort_values(by='Accuracy_score', ascending=False)


#unigram sampai bigram
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42, stratify=y)

def uni_bi_model(model_name,X_train,y_train,X_test,y_test):
    pipeline=Pipeline([
      ('cv', CountVectorizer(tokenizer=word_tokenize, ngram_range=(1,2), binary=True, min_df = 1)),
      ('model', model_name),
    ])
    print(model_name)
    pipeline.fit(X_train,y_train)

    preds=pipeline.predict(X_test)

    print (classification_report(y_test,preds))
    print (confusion_matrix(y_test,preds))
    print('Accuracy:', pipeline.score(X_test, y_test)*100)
    print("Training Score:",pipeline.score(X_train,y_train)*100)
    score = accuracy_score(y_test,preds)
    return score

mnb = uni_bi_model(MultinomialNB(),X_train,y_train,X_test,y_test)
maxent = uni_bi_model(LogisticRegression(),X_train,y_train,X_test,y_test)
svc = uni_bi_model(SVC(),X_train,y_train,X_test,y_test)
uni_bi_models = pd.DataFrame({
    'Model':['MNB','MaxEnt', 'SVM'],
    'Accuracy_score' :[mnb ,maxent, svc]
})
uni_bi_models.sort_values(by='Accuracy_score', ascending=False)


#unigram
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42, stratify=y)

def uni_model(model_name,X_train,y_train,X_test,y_test):
    pipeline=Pipeline([
      ('cv', CountVectorizer(tokenizer=word_tokenize, ngram_range=(1,1), binary=True, min_df = 2)),
      ('model', model_name),
    ])
    print(model_name)
    pipeline.fit(X_train,y_train)

    preds=pipeline.predict(X_test)

    print (classification_report(y_test,preds))
    print (confusion_matrix(y_test,preds))
    print('Accuracy:', pipeline.score(X_test, y_test)*100)
    print("Training Score:",pipeline.score(X_train,y_train)*100)
    score = accuracy_score(y_test,preds)
    return score

mnb = uni_model(MultinomialNB(),X_train,y_train,X_test,y_test)
maxent = uni_model(LogisticRegression(),X_train,y_train,X_test,y_test)
svc = uni_model(SVC(),X_train,y_train,X_test,y_test)
uni_models = pd.DataFrame({
    'Model':['MNB','MaxEnt', 'SVM'],
    'Accuracy_score' :[mnb ,maxent, svc]
})
uni_models.sort_values(by='Accuracy_score', ascending=False)


#bigram
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42, stratify=y)

def bi_model(model_name,X_train,y_train,X_test,y_test):
    pipeline=Pipeline([
      ('cv', CountVectorizer(tokenizer=word_tokenize, ngram_range=(2,2), binary=False, min_df = 1)),
      ('model', model_name),
    ])
    print(model_name)
    pipeline.fit(X_train,y_train)

    preds=pipeline.predict(X_test)

    print (classification_report(y_test,preds))
    print (confusion_matrix(y_test,preds))
    print('Accuracy:', pipeline.score(X_test, y_test)*100)
    print("Training Score:",pipeline.score(X_train,y_train)*100)
    score = accuracy_score(y_test,preds)
    return score

mnb = bi_model(MultinomialNB(),X_train,y_train,X_test,y_test)
maxent = bi_model(LogisticRegression(),X_train,y_train,X_test,y_test)
svc = bi_model(SVC(),X_train,y_train,X_test,y_test)
bi_models = pd.DataFrame({
    'Model':['MNB','MaxEnt', 'SVM'],
    'Accuracy_score' :[mnb ,maxent, svc]
})
bi_models.sort_values(by='Accuracy_score', ascending=False)


#cari yg tertinggi
pipeline=Pipeline([
  ('cv', CountVectorizer(tokenizer=word_tokenize, ngram_range=(1,3), binary=True, min_df = 1)),
  ('model', LogisticRegression()),
])
pipeline.fit(X,y)

preds=pipeline.predict(X)

print (classification_report(y,preds))
print (confusion_matrix(y,preds))
print('Accuracy:', pipeline.score(X, y)*100)
score = accuracy_score(y,preds)
score


df_test['text_clean'] = df_test['REVIEW'].apply(lambda x: convert_to_lower(x))
df_test['text_clean'] = df_test['text_clean'].apply(lambda x: remove_numbers(x))
df_test['text_clean'] = df_test['text_clean'].apply(lambda x: remove_punctuation(x))
df_test['text_clean'] = df_test['text_clean'].apply(lambda x: remove_extra_white_spaces(x))
df_test['text_clean'] = df_test['text_clean'].apply(lambda x: remove_stopwords(x))
X_test=df_test['text_clean']
y_test_pred = pipeline.predict(X_test)
y_test_pred


submission_df = pd.DataFrame({'ID': df_test.ID, 'LABEL': y_test_pred})
submission_df.to_csv('submission.csv', index=False)
submission_df

