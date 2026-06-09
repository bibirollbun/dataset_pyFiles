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


!unzip /kaggle/input/quora-question-pairs/train.csv.zip -d /kaggle/working
!unzip /kaggle/input/quora-question-pairs/test.csv.zip -d /kaggle/working


import warnings
warnings.simplefilter("ignore")


train = pd.read_csv('/kaggle/working/train.csv')
test = pd.read_csv('/kaggle/working/test.csv')


train.head()


train.isna().sum()


train.drop(columns='id', inplace=True)


train.drop(columns=['qid1', 'qid2'], inplace=True)


train.dropna(inplace=True)


train.info()


test.info()


train.duplicated().sum()


import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text: str) -> str:
  text = text.lower()
  text = re.sub(r'http\S+|www\S+', '', text)
  text = re.sub(r'@\w+', '', text)
  text = re.sub(r'#', '', text)
  text = re.sub(r'[^a-z\s]', '', text)

  tokens = text.split()
  stop_words = set(stopwords.words('english'))
  tokens = [w for w in tokens if w not in stop_words]

  lemmatizer = WordNetLemmatizer()
  tokens = [lemmatizer.lemmatize(w) for w in tokens]

  return ' '.join(tokens)


for col in ['question1', 'question2']:
  train[col] = train[col].astype(str)
  test[col] = test[col].astype(str)

train.info()


for col in ['question1', 'question2']:
  train[col] = train[col].apply(clean_text)
  test[col] = test[col].apply(clean_text)

train.head()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = train.drop(columns=['is_duplicate'])
y = train['is_duplicate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(transformers=[
    ('question1', TfidfVectorizer(max_features=5000, ngram_range=(1,2)), 'question1'),
    ('question2', TfidfVectorizer(max_features=5000, ngram_range=(1,2)), 'question2')
], remainder='drop')

model = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression(max_iter=300))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))



test.head()


model.fit(X, y)
prediction = model.predict(test.drop(columns=['test_id']))

submission = pd.DataFrame({
  'test_id': test['test_id'],
  'is_duplicate': prediction
})
submission.to_csv('submission.csv', index=False)


submission.head()


duplicate_count = submission['test_id'].duplicated().sum()
print(duplicate_count)


submission = submission.drop_duplicates(subset=['test_id'], keep='first')

