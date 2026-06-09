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


!unzip /kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip
!unzip /kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip


train_df = pd.read_csv('/kaggle/working/train.tsv', sep = '\t')
test_df = pd.read_csv("/kaggle/working/test.tsv", sep = '\t')


train_df.head()


test_df.head()


print(len(train_df))


for i in range(10):
    print(train_df.loc[i, 'Phrase'])


for i in range(1, 11):
    df_len  = len(train_df)
    print(train_df.loc[df_len - i, "Phrase"])


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

vader_lexicon = sia.lexicon

positive_words = {word for word, score in vader_lexicon.items() if score > 0}
negative_words = {word for word, score in vader_lexicon.items() if score < 0}


import re

INTENSIFIERS = {
    'very', 'extremely', 'really', 'so', 'too', 'quite', 'highly', 'deeply', 'absolutely', 'completely', 'totally'
}
COMPARATIVES = {
    'better', 'worse', 'more', 'less', 'faster', 'slower', 'stronger', 'weaker', 'higher', 'lower'
}
NEGATIONS = {
    'not', 'no', 'never', "don't", "doesn't", "didn't", "can't", "couldn't", "won't", "wouldn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't"
}


def is_exclam(text):
    if "!" in text:
        return 1
    
    return 0 

def is_question(text):
    if "?" in text:
        return 1
    
    return 0 

def length(text):
    return len(text)

def pleasant_amount(text):
    words = re.findall(r'\w+', text.lower())
    good_count = sum(word in positive_words for word in words)
    return good_count

def negative_amount(text):
    words = re.findall(r'\w+', text.lower())
    bad_count = sum(word in negative_words for word in words)
    return bad_count

def average_word_len(text):
    text = text.replace(",", '').replace("?", '').replace("!", '').replace('.', '')
    return sum([len(word) for word in text.split(" ")]) / len(text.split(" "))

def intensifiers_amount(text):
    words = re.findall(r"\b\w+'\w+|\w+\b", text.lower())  # tokenize including contractions
    intensifier_count = sum(word in INTENSIFIERS for word in words)
    return intensifier_count

def comparatives_amount(text):
    words = re.findall(r"\b\w+'\w+|\w+\b", text.lower())  # tokenize including contractions
    comparative_count = sum(word in COMPARATIVES for word in words)
    return comparative_count

def upper_case_amount(text):
    return len([char for char in text if char.isupper()])



from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class AddFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X = X.copy()
        X.loc[:, 'is_exclam'] = X.loc[:, "Phrase"].apply(lambda x: is_exclam(str(x)))
        X.loc[:, 'is_question'] = X.loc[:, "Phrase"].apply(lambda x: is_question(str(x)))
        X.loc[:, 'len'] = X.loc[:, "Phrase"].apply(lambda x: length(str(x)))
        X.loc[:, 'pleasant_amount'] = X.loc[:, "Phrase"].apply(lambda x: pleasant_amount(str(x)))
        X.loc[:, 'negative_amount'] = X.loc[:, "Phrase"].apply(lambda x: negative_amount(str(x)))
        X.loc[:, 'average_word_len'] = X.loc[:, "Phrase"].apply(lambda x: average_word_len(str(x)))
        X.loc[:, 'upper_case_amount'] = X.loc[:, "Phrase"].apply(lambda x: upper_case_amount(str(x)))
        X.loc[:, 'intensifiers_amount'] = X.loc[:, "Phrase"].apply(lambda x: intensifiers_amount(str(x)))
        X.loc[:, 'comparatives_amount'] = X.loc[:, "Phrase"].apply(lambda x: comparatives_amount(str(x)))

        return X

class DropColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        columns_to_drop = ['PhraseId', 'SentenceId', 'Phrase']
        return X.drop(columns=columns_to_drop)


col_transformer = Pipeline([
    ('add_features', AddFeatures()),
    ('drop_columns', DropColumns())
])

# Применяем трансформации
result_df = col_transformer.fit_transform(train_df)
result_df.head()


result_df = result_df.astype({"is_exclam": 'object', "is_question": "object"})


result_df['len'].describe()


result_df["average_word_len"].describe()


result_df["is_exclam"].describe()


result_df["is_question"].describe()


result_df[result_df["average_word_len"] == 0]


train_df.loc[result_df[result_df["average_word_len"] == 0].index, "Phrase"]


result_df = result_df[result_df["average_word_len"] != 0]


result_df["pleasant_amount"].describe()


result_df["negative_amount"].describe()


result_df["pleasant_amount"].value_counts(normalize=True) * 100


result_df["negative_amount"].value_counts(normalize=True) * 100


result_df["intensifiers_amount"].value_counts(normalize=True) * 100


result_df['comparatives_amount'].value_counts(normalize=True) * 100


from sklearn.linear_model import LogisticRegression

X = result_df.drop(['Sentiment'], axis = 1)
y = result_df["Sentiment"]

model = LogisticRegression(max_iter = 500, random_state = 42)

model.fit(X, y)


test_df_transformed = col_transformer.fit_transform(test_df)


test_df_transformed.head()


preds = model.predict(test_df_transformed)


from sklearn.metrics import classification_report

y_test = pd.read_csv("/kaggle/input/sentiment-analysis-on-movie-reviews/sampleSubmission.csv")

test_df.loc[:, "predicted"] = preds
result = test_df.merge(y_test, left_on="PhraseId", right_on="PhraseId")
predicted = result.loc[:, "predicted"].to_list()
sentiment = result.loc[:, "Sentiment"].to_list()
target_names = list(map(str, result.loc[:, "predicted"].unique()))
print(classification_report(sentiment, 
                            predicted, 
                            target_names=target_names))

