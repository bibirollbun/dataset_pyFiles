
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


rules = pd.read_csv('/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv')
rules.head()


def path(text_id,article_id):
    path = f"/kaggle/input/fake-or-real-the-impostor-hunt/data/train/article_{article_id}/file_{text_id}.txt"
    return path
def file_read(path):
    with open(path,'r',encoding='utf-8') as f:
        text = f.read().strip()
    return text
# /article_0000/file_1.txt


rules['article'] = rules['id'].apply(lambda x: str(x).zfill(4))
rules['fake_text_id'] = rules['real_text_id'].apply(lambda x: 2 if x==1 else 1)
rules['real_text_file'] = rules[['real_text_id','article']].apply(lambda x: path(x['real_text_id'], x['article']), axis=1)
rules['fake_text_file'] = rules[['fake_text_id','article']].apply(lambda x: path(x['fake_text_id'], x['article']), axis=1)
rules['real_text'] = rules['real_text_file'].apply(file_read)
rules['fake_text'] = rules['fake_text_file'].apply(file_read)
rules.head()


df_real = rules[['article','real_text']].copy()
df_real.columns = ['article', 'text']
df_real['label'] =1
df_fake = rules[['article','fake_text']].copy()
df_fake.columns = ['article', 'text']
df_fake['label'] =0

df = pd.concat([df_real,df_fake],ignore_index =True)


df['text'][3]


df.info()


import string
exclude = string.punctuation

def lower(text):
    return text.lower()
example = 'The importance for understanding how stars evolve has led researchers to focus on their multiplicity (the presence or absence multiple star partners). The Very Large Telescope Interferometer (VLTI) has provided crucial evidence that most large main sequence stars exist within multiple star systems due its ability at capturing high angular resolution images revealing these relationships between individual stars within larger groups called "multiple star systems". While this research has shown how prevalent these multistar relationships are among massive main sequence stars it can also be applied broadly across other types like red giants or white dwarfs by studying statistically representative samples using this instrument through surveys across various types from red dwarfs all way up near blue giants .\nThe VLTI excels at studying individual components within those multistar relationships by providing detailed information about them through observations like measuring orbital parameters or measuring mass differences between members within an interacting system .'
# print(lower(example))
def rem_punc(text):
    return text.translate(str.maketrans("","",exclude))
# print(rem_punc(example))

from nltk.corpus import stopwords

def rem_stopwords(text):
    new_text = []
    
    # print("new ---------",text)
   
    for word in text.split():
        if word not in stopwords.words('english'):
            new_text.append(word)

    return " ".join(new_text)
# print(rem_stopwords(example))
import nltk
from nltk.tokenize import TweetTokenizer, sent_tokenize
def tokenize(text):
    tokenizer_words = TweetTokenizer()
    tokens_sentences = [tokenizer_words.tokenize(t) for t in nltk.sent_tokenize(text)]
    return tokens_sentences

# print(tokenize(example))    
# example = tokenize(example)
from nltk import WordNetLemmatizer
wordNet = WordNetLemmatizer()
def lemm(text):
    
    new_text = []
    for sent in text:
         for word in sent:
            new_text.append(wordNet.lemmatize(word,'v'))
    return " ".join(new_text)
# print(lemm(example))


def preprocess(text):
    text = lower(text)
    
    text = rem_punc(text)
    text = rem_stopwords(text)
    text = tokenize(text)
    text = lemm(text)
    
    return text
# print(example)
print(preprocess(example))
df['processed'] = df['text'].astype(str).apply(preprocess)


df.head()


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=10000,)
x = tfidf.fit_transform(df['processed'])


y = df['label']


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , confusion_matrix
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , confusion_matrix
lr = RandomForestClassifier( n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced_subsample',
            random_state=42)
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))



from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

model = GaussianNB()
model.fit(X_train.toarray(), y_train)

y_pred = model.predict(X_test.toarray())

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Negative","Positive"]))



from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

model = SVC(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Negative","Positive"]))



from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Negative","Positive"]))

# print("2")

# y_pred2 = model.predict(X_train)
# print(confusion_matrix(y_train, y_pred2))
# print(classification_report(y_train, y_pred2, target_names=["Negative","Positive"]))





from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
param_grid = {
    'C': [0.1, 1, 10],                  # Regularization strength
    'gamma': ['scale', 'auto', 0.1, 1], # Kernel coefficient for 'rbf'
    'kernel': ['rbf', 'linear', 'poly'] # Kernel types
}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)


print("Best Parameters:", grid.best_params_)

y_pred = grid.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))





from gensim.models import FastText

ft_model = FastText(sentences=df['processed'], vector_size=100, window=4, min_count=1)

X = np.array([
    np.mean([ft_model.wv[word] for word in doc if word in ft_model.wv] or [np.zeros(100)], axis=0)
    for doc in df['processed']
])


y = df['label']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

model = SVC(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Negative","Positive"]))


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
param_grid = {
    'C': [0.1, 1, 10],                  
    'gamma': ['scale', 'auto', 0.1, 1], 
    'kernel': ['rbf', 'linear', 'poly'] 
}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)


print("Best Parameters:", grid.best_params_)

y_pred = grid.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

# Initialize the model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Set up GridSearchCV
grid = GridSearchCV(estimator=xgb,
                    param_grid=param_grid,
                    scoring='f1_weighted',
                    cv=5,
                    verbose=1,
                    n_jobs=-1)

# Fit on training data
grid.fit(X_train, y_train)

# Best model from grid search
best_model = grid.best_estimator_

# Predict on test data
y_pred = best_model.predict(X_test)

# Print results
print("Best Hyperparameters:", grid.best_params_)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))



from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

model = SVC(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Negative","Positive"]))


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , confusion_matrix
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))


df_test = pd.DataFrame(columns=[['id','text_1','text_2']])


# import os
# path = '/kaggle/input/competition-fake-or-real-news/data/test'
# i=0
# for item in os.listdir(path):
#     if i==8:
#         break
#     id   = item[8:]
#     path_art = f"{path}/{item}"
#     path_file_1 = f"{path_art}/file_1.txt"
#     path_file_2 = f"{path_art}/file_2.txt"
#     with open(path_file_1,'r',encoding='utf-8') as f:
#         text_1 = f.read().strip()
#     with open(path_file_2,'r',encoding='utf-8') as f:
#         text_2 = f.read().strip()
#     text_1 = preprocess(text_1)
    
#     tf_text = tfidf.transform([text_1])
#     y_text_1 = model.predict(tf_text)
#     text_2 = preprocess(text_2)
#     tf_text = tfidf.transform([text_2])
#     y_text_2 = model.predict(tf_text)
#     print("path - ",path_art)
#     print("file 1 - ",y_text_1)
#     print("file 2 - ",y_text_2)
#     i+=1
#     print(id)
#     output = 1 if y_text_1==1 else 2
#     df_test[-1] = [id,]


from pathlib import Path
data_path = Path("/kaggle/input/fake-or-real-the-impostor-hunt/data/test")
folders = sorted([f for f in data_path.iterdir() if f.is_dir()])

submission_rows = []

for idx, folder in enumerate(folders):  
    texts = []
    for fname in ["file_1.txt", "file_2.txt"]:
        fp = folder / fname
        texts.append(file_read(fp))
    # print(folder)
    path_file_1 = f"{folder}/file_1.txt"
    path_file_2 = f"{folder}/file_2.txt"
    with open(path_file_1,'r',encoding='utf-8') as f:
        text_1 = f.read().strip()
    with open(path_file_2,'r',encoding='utf-8') as f:
        text_2 = f.read().strip()
    text_1 = preprocess(text_1)
    
    tf_text = tfidf.transform([text_1])
    y_text_1 = model.predict(tf_text)
    text_2 = preprocess(text_2)
    tf_text = tfidf.transform([text_2])
    y_text_2 = model.predict(tf_text)
    
    output = 1 if y_text_1==1 else 2 
    # print(f"id: {idx} real_text_id: {output}")
    submission_rows.append({"id": idx, "real_text_id": output})  

submission = pd.DataFrame(submission_rows)
submission.to_csv("submission.csv", index=False)
# print(submission.head())




