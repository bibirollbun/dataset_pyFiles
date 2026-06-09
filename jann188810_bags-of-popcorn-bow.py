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


labeled_train_data= pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip', delimiter ='\t', quoting=3)
test_data= pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip', delimiter ='\t', quoting=3)
unlabeled_train_data= pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip', delimiter ='\t', quoting=3)


unlabeled_train_data.head()


from bs4 import BeautifulSoup
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))

def preprocess (review):
    text = BeautifulSoup(review, "html.parser").get_text()
    text = re.sub("[^a-zA-Z]"," ", text).lower()
    return " ".join([word for word in text.split() if word not in stop_words])


labeled_train_data['cleaned_review'] = labeled_train_data['review'].map(preprocess)
unlabeled_train_data['cleaned_review'] = unlabeled_train_data['review'].map(preprocess)
test_data['cleaned_review'] = test_data['review'].map(preprocess)


combined_data = pd.concat([labeled_train_data[['cleaned_review']], unlabeled_train_data[['cleaned_review']]])


combined_data.head()


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=5000)
combined_features = vectorizer.fit_transform(combined_data['cleaned_review'])


combined_features


X_train_combined = combined_features[:len(labeled_train_data)]
y_train_combined = labeled_train_data['sentiment']


from sklearn.model_selection import train_test_split


X_train, X_val, y_train, y_val = train_test_split(X_train_combined, y_train_combined, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier (n_estimators=100, random_state=42)
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification report:", classification_report(y_val,y_pred))


X_test = vectorizer.transform(test_data['cleaned_review'])
test_data['sentiment'] = model.predict(X_test)


test_data.head()


import csv


submission=test_data[['id','sentiment']]
submission.to_csv('submission.csv', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')

