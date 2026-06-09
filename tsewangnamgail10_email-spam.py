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


traning_df=pd.read_csv('/kaggle/input/urfu-infosec-2024-competition-1/Train.csv')
traning_df.head()


target_df=pd.read_csv('/kaggle/input/urfu-infosec-2024-competition-1/Target.csv')
target_df.head()


data=pd.merge(traning_df,target_df,on="row ID")
data.head()


print("Data shape:", data.shape)
print(data.columns)
print(data.head())


# 3. Handle Missing Values
data['email'] = data['email'].fillna("")


# 4. Extra Feature Engineering
import re

from sklearn.base import BaseEstimator, TransformerMixin

class ExtraFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    
    def transform(self, X):
        # X is a pandas Series of text
        out = []
        url_pattern = re.compile(r"http[s]?://\S+")
        for text in X:
            text = str(text)
            length = len(text)
            n_words = len(text.split())
            n_exclaim = text.count("!")
            n_upper = sum(1 for c in text if c.isupper())
            url_count = len(url_pattern.findall(text))
            out.append([length, n_words, n_exclaim, n_upper, url_count])
        return np.array(out)


# 5. Features & Target
from sklearn.model_selection import train_test_split

X = data['email']      # email text
y = data['label']      # target

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 6. Pipeline (CountVectorizer + Extra Features + Naive Bayes)
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer

# Pipeline for text
text_features = Pipeline([
    ('vect', CountVectorizer(max_features=10000, ngram_range=(1,2)))  # bag-of-words + bigrams
])

# Pipeline for numeric features
numeric_features = Pipeline([
    ('extra', ExtraFeatures())
])

# Combine them
combined_features = FeatureUnion([
    ('text', text_features),
    ('extra', numeric_features)
])

clf = Pipeline([
    ('features', combined_features),
    ('nb', MultinomialNB())
])



# 7. Train
clf.fit(x_train, y_train)


from sklearn.metrics import classification_report, confusion_matrix
y_pred = clf.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#  Try on sample
sample = ["Congratulations! You've won a free prize. Click here to claim."]
print("Prediction for sample:", clf.predict(sample))


# Sample emails for testing
samples = [
    "Congratulations! You've won a free iPhone. Click here to claim now!",   # spam
    "Dear John, your meeting is scheduled for tomorrow at 10 AM.",             # ham
    "Limited time offer!!! Buy one get one free on all products!",             # spam
    "Hi team, please find attached the weekly report.",                        # ham
    "URGENT: Your account has been compromised. Reset your password immediately.", # spam
    "Can we reschedule our lunch meeting to next week?",                       # ham
    "You have been selected to receive a $1000 gift card!",                     # spam
    "Please review the updated project plan and share your feedback."          # ham
]

# Predict using the trained pipeline
predictions = clf.predict(samples)

# Show results
for email, pred in zip(samples, predictions):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"Email: {email}\nPrediction: {label}\n{'-'*60}")


