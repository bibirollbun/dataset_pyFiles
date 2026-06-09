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


train = pd.read_csv("/kaggle/input/toxic-comments-classification-2023/train_data.csv")
test = pd.read_csv("/kaggle/input/toxic-comments-classification-2023/test_data.csv")
sample_submission = pd.read_csv("/kaggle/input/toxic-comments-classification-2023/sample_submission.csv")


train.head(), test.head()


# Cleaning
from nltk.corpus import stopwords
from nltk import SnowballStemmer
import re, string

stop_words = stopwords.words('russian')

def remove_stopwords(text):
    return ' '.join(word for word in text.split(' ') if word not in stop_words)

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

stemmer = SnowballStemmer('russian')

def stem_text(text):
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text

train["comment"] = train["comment"].apply(remove_stopwords)
train["comment"] = train["comment"].apply(clean_text)
train["comment"] = train["comment"].apply(stem_text)

test["comment"] = test["comment"].apply(remove_stopwords)
test["comment"] = test["comment"].apply(clean_text)
test["comment"] = test["comment"].apply(stem_text)

train.head(), test.head()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

X = train["comment"].to_numpy()
y = np.array(train["toxic"], dtype=int)

X_train, X_test, y_train, y_test = train_test_split(X, y)

encoder = TfidfVectorizer()
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

X_train.shape, X_test.shape


from sklearn.svm import SVC
from xgboost import XGBClassifier

clf = XGBClassifier()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
preds.shape


sample = ["Я радостный", "Заткнись урод"]

for i, s in enumerate(sample):
    sample[i] = stem_text(clean_text(remove_stopwords(s)))

sample = np.array(sample)

sample = encoder.transform(sample)

clf.predict(sample)


accuracy = 0.

match = (y_test == preds).astype(int).sum()
total = len(y_test)
print(f"Eval accuracy: {(match/total)*100:.2f}%")


X_sub = test["comment"]
X_sub = X_sub.to_numpy()
X_sub = encoder.transform(X_sub)

sub = clf.predict(X_sub)
sub


test["toxic"] = sub
test[["comment_id", "toxic"]].to_csv("submission.csv", index=None)




