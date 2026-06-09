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




train_df=pd.read_csv("/kaggle/input/TweetSentimentBR/Train.csv")


train_df.head()


train_df.shape


train_df.isnull().sum()


train_df['tweet_date'] = pd.to_datetime(train_df['tweet_date'])


train_df.drop(columns=['id', 'query_used'], inplace=True)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # linkleri kaldır
    text = re.sub(r'@\w+', '', text)     # mentionları kaldır
    text = re.sub(r'#\w+', '', text)     # hashtagleri kaldır
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # noktalama işaretleri
    text = re.sub(r'\d+', '', text)      # sayılar
    text = re.sub(r'\s{2,}', ' ', text)  # fazla boşluk
    return text.strip()

# Temizlenmiş tweet kolonunu oluştur
train_df['clean_text'] = train_df['tweet_text'].apply(clean_text)


# Tweet uzunluğu
train_df['text_length'] = train_df['clean_text'].apply(len)

# Kelime sayısı
train_df['word_count'] = train_df['clean_text'].apply(lambda x: len(x.split()))

# Saat bilgisi (gündüz-gece ayrımı yapabiliriz)
train_df['hour'] = train_df['tweet_date'].dt.hour

# Gün bilgisi
train_df['weekday'] = train_df['tweet_date'].dt.day_name()

# Hafta sonu mu?
train_df['is_weekend'] = train_df['tweet_date'].dt.weekday >= 5



# Sentiment distribution
sns.countplot(data=train_df, x='sentiment')
plt.title("Positive vs Negative Tweet Count")
plt.xticks([0, 1], ["Negative", "Positive"])
plt.show()


# Tweet length histogram
plt.figure(figsize=(10,5))
sns.histplot(data=train_df, x='text_length', hue='sentiment', bins=30, kde=True)
plt.title("Tweet Length and Sentiment Distribution")
plt.show()


# Convert 'tweet_date' to datetime format
train_df['tweet_date'] = pd.to_datetime(train_df['tweet_date'])

# Group by date and sentiment, and count the number of tweets
df_time_sentiment = train_df.groupby([train_df['tweet_date'].dt.date, 'sentiment']).size().unstack()

# Plot the sentiment trends over time
df_time_sentiment.plot(figsize=(12, 5), marker='o')
plt.xlabel("Date")
plt.ylabel("Tweet Count")
plt.title("Sentiment Trends Over Time")
plt.legend(["Negative", "Positive"])
plt.grid(True)
plt.show()



# Distribution by Days
plt.figure(figsize=(10,5))
sns.countplot(data=train_df, x='weekday', hue='sentiment',
              order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
plt.title("Tweet Sentiments by Day of the Week")
plt.xticks(rotation=45)
plt.show()


# Hourly analysis
plt.figure(figsize=(10,5))
sns.boxplot(data=train_df, x='hour', y='text_length', hue='sentiment')
plt.title("Tweet Lengths by Hour and Sentiment")
plt.xlabel("Hour")
plt.ylabel("Text Length")
plt.show()



train_df['tweet_date'] = pd.to_datetime(train_df['tweet_date'])  # Convert to datetime
train_df_time = train_df.groupby(train_df['tweet_date'].dt.date)['sentiment'].count()  # Count tweets per day

plt.figure(figsize=(12, 5))
train_df_time.plot()
plt.xlabel("Date")
plt.ylabel("Tweet Count")
plt.title("Tweet Activity Over Time")
plt.grid()
plt.show()




