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


train = pd.read_csv('/kaggle/input/natural-language-processing-with-disaster-tweets/train.csv')
test = pd.read_csv('/kaggle/input/natural-language-processing-with-disaster-tweets/test.csv')


train.head()
train.info()
train['target'].value_counts()  # 0: not disaster, 1: disaster


import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def preprocess(text):
    text = re.sub(r'http\S+|www\S+', '', text)      # URL除去
    text = re.sub(r'[^a-zA-Z\s]', '', text)         # 記号除去
    text = re.sub(r'\s+', ' ', text).strip()        # 連続空白を1つに、前後の空白除去
    text = word_tokenize(text.lower())              # トークン化、小文字化
    text = [token for token in text if not token in stop_words]
    text = [stemmer.stem(token) for token in text]
    return " ".join(text)

train['clean_text'] = train['text'].apply(preprocess)
test['clean_text'] = test['text'].apply(preprocess)


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# 欠損のないデータで学習
train_with_keyword = train[train['keyword'].notna()].copy()
train_missing_keyword = train[train['keyword'].isna()].copy()

# textを説明変数、keywordを目的変数とする
X_train_texts = train_with_keyword['text']
y_train_keywords = train_with_keyword['keyword']

# ラベルを数値にエンコード
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_keywords)

# TF-IDFベクトル化
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train_texts)

# ロジスティック回帰モデルで学習
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train_encoded)

# 欠損データのtextをベクトル化して予測
X_missing_vec = vectorizer.transform(train_missing_keyword['text'])
y_missing_pred = model.predict(X_missing_vec)
y_missing_pred_label = le.inverse_transform(y_missing_pred)

# 補完した keyword を train に戻す
train.loc[train['keyword'].isna(), 'keyword'] = y_missing_pred_label

# 欠損のないデータで学習
test_with_keyword = test[test['keyword'].notna()].copy()
test_missing_keyword = test[test['keyword'].isna()].copy()

# テキストとラベルの準備
X_test_texts = test_with_keyword['text']
y_test_keywords = test_with_keyword['keyword']

# ラベルを数値にエンコード
le = LabelEncoder()
y_test_encoded = le.fit_transform(y_test_keywords)

# TF-IDFベクトル化
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_test_vec = vectorizer.fit_transform(X_test_texts)

# ロジスティック回帰モデルで学習
model = LogisticRegression(max_iter=1000)
model.fit(X_test_vec, y_test_encoded)

# 欠損データのtextをベクトル化して予測
X_missing_vec = vectorizer.transform(test_missing_keyword['text'])
y_missing_pred = model.predict(X_missing_vec)
y_missing_pred_label = le.inverse_transform(y_missing_pred)

# 補完した keyword を test に戻す
test.loc[test['keyword'].isna(), 'keyword'] = y_missing_pred_label


print(train.isnull().sum())
print(test.isnull().sum())


from sklearn.feature_extraction.text import TfidfVectorizer

train['combined'] = train['clean_text'] + ' ' + train['keyword']
test['combined'] = test['clean_text'] + ' ' + test['keyword']

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(train['combined'])
X_test = tfidf.transform(test['combined'])
y = train['target']


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

model = LogisticRegression()
model.fit(X_train, y_train)

preds = model.predict(X_val)
print(classification_report(y_val, preds))


test_predicted = model.predict(X_test)


test_predicted


sub = pd.read_csv('/kaggle/input/natural-language-processing-with-disaster-tweets/sample_submission.csv')
sub['target'] = list(map(int, test_predicted))
sub.to_csv('submission.csv', index = False)

