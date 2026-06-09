import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',None)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

import nltk
nltk.download("stopwords")
import re
from nltk.corpus import stopwords
import string


df=pd.read_csv("/kaggle/input/TweetSentimentBR/Train.csv")


df.head()


df.shape


df.isnull().sum()


df['sentiment'].value_counts()


sentiment_counts = df['sentiment'].value_counts()

plt.figure(figsize=(6, 4))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=["red", "green"])
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1], labels=["Negative", "Positive"])
plt.title("Sentiment Distribution")
plt.show()


df['tweet_date'] = pd.to_datetime(df['tweet_date'])  # Convert to datetime
df_time = df.groupby(df['tweet_date'].dt.date)['sentiment'].count()  # Count tweets per day

plt.figure(figsize=(12, 5))
df_time.plot()
plt.xlabel("Date")
plt.ylabel("Tweet Count")
plt.title("Tweet Activity Over Time")
plt.grid()
plt.show()


df['tweet_date'] = pd.to_datetime(df['tweet_date'])
df_time_sentiment = df.groupby([df['tweet_date'].dt.date, 'sentiment']).size().unstack()

df_time_sentiment.plot(figsize=(12, 5), marker='o')
plt.xlabel("Date")
plt.ylabel("Tweet Count")
plt.title("Sentiment Trends Over Time")
plt.legend(["Negative", "Positive"])
plt.grid()
plt.show()


stemmer=nltk.SnowballStemmer("portuguese")
stopword=set(stopwords.words("portuguese"))
def clean_text(df,text_column):
    # Convert to lowercase
    df[text_column] = df[text_column].str.lower()
    
    # Remove unwanted characters
    df[text_column] = df[text_column].str.replace('[^\w\s]', '', regex=True)
    df[text_column] = df[text_column].str.replace('\w*\d\w*', '', regex=True)
    df[text_column] = df[text_column].str.replace('\n', '', regex=True)
    df[text_column] = df[text_column].str.replace('\r', '', regex=True)
    df[text_column] = df[text_column].str.replace('https?://\S+|www\.\S+', '', regex=True)
    df[text_column] = df[text_column].str.replace('<.*?>+', '', regex=True)
    df[text_column] = df[text_column].str.replace('\[.*?\]', '', regex=True)
    
    # Remove stopwords and apply stemming
    df[text_column] = df[text_column].apply(lambda x: ' '.join(
        stemmer.stem(word) 
        for word in x.split() 
        if word not in stopword
    ))
    return df[text_column]


df.iloc[0]["tweet_text"]


df["tweet_text"]=clean_text(df,"tweet_text")


df.iloc[0]["tweet_text"]


from wordcloud import WordCloud
from wordcloud import STOPWORDS
from PIL import Image
def wc(data,color):
    plt.figure(figsize=(10,10))
   # mask=np.array(Image.open('cloud.png'))
    wc=WordCloud(background_color=color,stopwords=STOPWORDS)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis('off')


wc(df[df["sentiment"]==0]["tweet_text"],"white")


wc(df[df["sentiment"]==1]["tweet_text"],"white")


from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')

portuguese_stopwords = stopwords.words("portuguese") 

def get_top_ngrams(texts, ngram_range=(2, 3), top_n=10):
    vect = CountVectorizer(ngram_range=ngram_range, stop_words=portuguese_stopwords) 
    dtm = vect.fit_transform(texts)
    ngram_counts = dtm.sum(axis=0)
    ngrams_freq = [(word, ngram_counts[0, idx]) for word, idx in vect.vocabulary_.items()]
    return sorted(ngrams_freq, key=lambda x: x[1], reverse=True)[:top_n]


negative_text = df[df['sentiment'] == 0]['tweet_text']
negative_text_ngrams = get_top_ngrams(negative_text, ngram_range=(2, 2), top_n=20)
negative_text_ngrams


positive_text = df[df['sentiment'] == 1]['tweet_text']
positive_text_ngrams = get_top_ngrams(positive_text, ngram_range=(2, 2), top_n=20)
positive_text_ngrams


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',100)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier  
from sklearn.naive_bayes import MultinomialNB    


def classification_test(x,y,vect,confusion_mtr=False):
    b=BernoulliNB()
    l=LogisticRegression()
    d=DecisionTreeClassifier()
    rf=RandomForestClassifier()
    h=GradientBoostingClassifier()
    a=AdaBoostClassifier()
    m=MultinomialNB()
    algos=[b,l,d,rf,h,a,b]

    algo_names=['Bernoulli NB','Logistic Regression','Decision Tree Classifier','Random Forest Classifier','Gradient Boosting Classifier','Ada Boost Classifier','Multinomial NB']

    accuracy=[]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    result=pd.DataFrame(columns=['Accuracy Score'],index=algo_names)

    for i, algo in enumerate(algos):
        x_train_dtm = vect.fit_transform(x_train)
        x_test_dtm = vect.transform(x_test)

        if hasattr(algo, 'fit'):
            x_train_dtm = x_train_dtm  
            x_test_dtm = x_test_dtm   

        p = algo.fit(x_train_dtm, y_train).predict(x_test_dtm)
        accuracy.append(accuracy_score(y_test, p))
        if confusion_mtr:
            cm = confusion_matrix(y_test, p)
            plt.figure(figsize=(5, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=algo.classes_, yticklabels=algo.classes_,cbar=None)
            plt.title(f"Confusion Matrix - {algo_names[i]}")
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()

    result['Accuracy Score']=accuracy

    r_table=result.sort_values('Accuracy Score',ascending=False)
    
        
        
    return r_table[['Accuracy Score']]


x=df["tweet_text"]
y=df["sentiment"]


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(ngram_range=(1,2))
classification_test(x,y,vect,confusion_mtr=True)


vect = CountVectorizer(ngram_range=(1,2))
l=LogisticRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train_dtm = vect.fit_transform(x_train)
x_test_dtm = vect.transform(x_test)
model=l.fit(x_train_dtm, y_train)


import joblib  

joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(vect, 'vectorizer.pkl')


df_test=pd.read_csv('/kaggle/input/TweetSentimentBR/Test.csv')


df_test.head()


df_test["tweet_text"]=clean_text(df_test,"tweet_text")


test_data= vect.transform(df_test["tweet_text"])
predictions=model.predict(test_data)


df_test["predictions"]=predictions


df_test.head()


submission=pd.DataFrame({
    "id":df_test["id"],
    "sentiment":df_test["predictions"]
})


submission.to_csv("submission.csv",index=False)

