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


train=pd.read_csv("/kaggle/input/TweetSentimentBR/Train.csv")
test=pd.read_csv('/kaggle/input/TweetSentimentBR/Test.csv')
sample=pd.read_csv("/kaggle/input/TweetSentimentBR/zSample_Submission.csv")


train.head()


test.head()


sample.head()


train.shape


test.shape


train.isnull().sum()


df=pd.concat([train,test])


df.head()


df=df[["tweet_text","sentiment"]]


df.head()


import matplotlib.pyplot as plt

value_counts = df['sentiment'].value_counts()
plt.figure(figsize=(8, 6))
value_counts.plot(kind="bar",color=["blue","yellow"])

plt.xticks(rotation=0)  
plt.show()


pip install langdetect


from langdetect import detect
from collections import Counter
import pandas as pd

# Dil tespit fonksiyonu (hataları yakalar)
def detect_language_safe(text):
    try:
        return detect(text)
    except:
        return "error"

# Dil sütununu oluştur
test['lang'] = test['tweet_text'].astype(str).apply(detect_language_safe)

# Dilleri say
language_counts = test['lang'].value_counts()

# Sonuçları yazdır
print(language_counts)



df['text']=df['tweet_text'].str.lower() 


df['text']=df['text'].str.replace('[^\w\s]','')
df['text']=df['text'].str.replace('\n','')
df['text']=df['text'].str.replace('\d+','')
df['text']=df['text'].str.replace('\r',' ')


import re
import string
import nltk

def tokenization(text):
    text = re.split('\W+', text)
    return text

df['tokenized'] = df['text'].apply(lambda x: tokenization(x.lower()))
df.head()


import nltk
nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('portuguese')


def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
    
df['nonstop'] = df['tokenized'].apply(lambda x: remove_stopwords(x))
df.head(10)


ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

df['stemmed'] = df['nonstop'].apply(lambda x: stemming(x))
df.head()


def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text


df['stemmedtext'] = df['stemmed'].apply(lambda x: ' '.join(x))


df.head()


from sklearn.linear_model import LogisticRegression


from nltk.stem import PorterStemmer
pr=PorterStemmer()


def lemmafn(text):
    words=TextBlob(text).words
    return[pr.stem(word) for word in words]


from sklearn.feature_extraction.text import CountVectorizer
vect=CountVectorizer(stop_words='english',ngram_range=(1,2),max_features=10000,analyzer=lemmafn)


vect


train = df.iloc[:50000]
test = df.iloc[50000:]

x_train=train.stemmedtext
y_train=train['sentiment']

x_test=test.stemmedtext
y_test=test['sentiment']

from textblob import TextBlob
x_train=vect.fit_transform(x_train)


def fnc_classification_all_model(x,y):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    from sklearn.metrics import confusion_matrix,classification_report
              
    g=GaussianNB()
    b=BernoulliNB()
    D=DecisionTreeClassifier()
    R=RandomForestClassifier()
    Log=LogisticRegression()
    XGB=XGBClassifier()
    G=GradientBoostingClassifier()
      
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
    
    
    algos=[b,D,R,Log,XGB,G]
    algo_names=['BernoulliNB','DecisionTreeClassifier','RandomForestClassifier','LogisticRegression','XGBClassifier','GradientBoostingClassifier']
    
    accuracy_scored=[]
    precision_scored=[]
    recall_scored=[]
    f1_scored=[]
       
    
    for item in algos:
        print(item)

        predict=item.fit(x_train,y_train).predict(x_test)
        
        
        accuracy_scored.append(accuracy_score(y_test,predict))
        precision_scored.append(precision_score(y_test,predict,average='macro'))
        recall_scored.append(recall_score(y_test,predict,average='macro'))
        f1_scored.append(f1_score(y_test,predict,average='macro'))

    result=pd.DataFrame(columns=['accuracy_score','f1_score','recall_score','precision_score'],index=algo_names)
    result['accuracy_score']=accuracy_scored
    result['f1_score']=f1_scored
    result['recall_score']=recall_scored
    result['precision_score']=precision_scored
    
    return result.sort_values('accuracy_score',ascending=False)


fnc_classification_all_model(x_train,y_train)


from sklearn.linear_model import LogisticRegression
Log=LogisticRegression()


Log.fit(x_train,y_train)


x_test=vect.transform(x_test) #x_testi de vektörize ettik
pred=Log.predict(x_test)


pred


submission=pd.DataFrame()


test1=pd.read_csv('/kaggle/input/TweetSentimentBR/Test.csv')


submission["id"]=test1["id"]


submission["sentiment"]=pred.astype(int)


submission.head()


submission.to_csv("submission.csv", index=False)




