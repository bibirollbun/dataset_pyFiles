# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv')
df


df.shape


df.dtypes


df.info()


df.isnull().sum()


df.duplicated().sum()


df.drop('id', axis=1, inplace=True)


df['sentiment'].value_counts()


df.head()


df.tail()


sns.countplot(x=df['sentiment'], data=df)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Count of Positive, Negative & Neutral')
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width() / 2,  # x-coordinate
             p.get_height() + 1,            # y-coordinate
             int(p.get_height()),           # The count value
             ha='center') 
plt.show()


label_status = df['sentiment'].value_counts()
pie_df = pd.DataFrame({
    'transactions': label_status.index,
    'quantity': label_status.values
})
figure = px.pie(pie_df, names='transactions', values='quantity', title='percent of sentiment')
figure.show()


df['text'][0]


df['text'][188]


punctuation = set(string.punctuation)
print(punctuation)


len(punctuation)


bng_stopword = set(stopwords.words('bengali'))
#bng_stopword


def preprocess_text(text):
    token = word_tokenize(text)
    filter_token = [word for word in token if word.lower() not in bng_stopword and word not in punctuation]
    return filter_token


df['text'] = df['text'].apply(preprocess_text)
df['text']


lemmatizer = WordNetLemmatizer()


def text_lemmatize(text):
    lemmatized_text = [lemmatizer.lemmatize(word) for word in text]
    clean_text = ' '.join(lemmatized_text)
    return clean_text


!unzip /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora/


df['text'] = df['text'].apply(text_lemmatize)
df['text']


df['text'][0]


df['text'][10]


df.shape


vectorizer = TfidfVectorizer()


text_column = 'text'
label_column = 'sentiment'


x = vectorizer.fit_transform(df[text_column])


x.toarray()


pd.DataFrame(x.toarray())


vectorizer.get_feature_names_out()


pd.DataFrame(x.toarray(), index=df['text'], columns=vectorizer.get_feature_names_out())


y = df[label_column]


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=42)


xtrain.shape


xtest.shape


model1 = LogisticRegression()


model1.fit(xtrain, ytrain)


y_pred1 = model1.predict(xtest)
y_pred1


model1.score(xtrain, ytrain)


model1.score(xtest, ytest)


model2 = MultinomialNB()


model2.fit(xtrain, ytrain)


model2.score(xtrain, ytrain)


model2.score(xtest, ytest)


model3 = BernoulliNB()


model3.fit(xtrain, ytrain)


model3.score(xtrain, ytrain)


model3.score(xtest, ytest)




