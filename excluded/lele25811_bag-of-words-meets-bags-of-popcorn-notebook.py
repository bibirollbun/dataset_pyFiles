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


!unzip /kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip -d /kaggle/working/
!unzip /kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip -d /kaggle/working/
!unzip /kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip -d /kaggle/working/


train_data = pd.read_csv('/kaggle/working/labeledTrainData.tsv', sep='\t')
test_data = pd.read_csv('/kaggle/working/testData.tsv', sep='\t')


train_data.columns, test_data.columns


train_data, test_data


train_data.isnull().sum()


test_data.isnull().sum()


import re
def clean_text(text):
    text = text.lower()  # set lower case
    text = re.sub(r"<.*?>", "", text)  # remove HTML
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation marks
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = text.strip()  # remove extra spaces
    return text


train_data['review'] = train_data['review'].apply(clean_text)
test_data['reveiw'] = test_data['review'].apply(clean_text)


train_data, test_data


train_X = train_data['review']
train_y = train_data['sentiment']

train_X, train_y


test_X = test_data['review']
test_X


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
train_X_vectorized = vectorizer.fit_transform(train_X)
test_X_vectorized = vectorizer.transform(test_X)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(train_X_vectorized, train_y)


prediction = model.predict(test_X_vectorized)
prediction


output = pd.DataFrame({'id': test_data['id'], 'sentiment': prediction})
output.to_csv('/kaggle/working/submissionBagofWordsMeetsBagsofPopcorn.csv', index=False)

