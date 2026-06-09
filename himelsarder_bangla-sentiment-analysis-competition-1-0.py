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
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv")


df.head()


# Preprocessing function
def clean_text(text):
    text = re.sub(r'[^\u0980-\u09FF\s]', '', text)  # Keep only Bengali characters
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df['cleaned_text'] = df['text'].apply(clean_text)


df.head()


# Encode sentiment labels
sentiment_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}  # Change -1 to 0
df['sentiment_label'] = df['sentiment'].map(sentiment_mapping)


df.head()


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['sentiment_label'], test_size=0.2, random_state=42)


X_train.shape


# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# Train ML Models
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}


for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))


print(f"Training data shape: {df.shape}")
print(f"Class distribution:")
print(df['sentiment'].value_counts())


# Train SVM Model
svc_model = SVC()
svc_model.fit(X_train_tfidf, y_train)
y_pred = svc_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))


# Make predictions on all training data
predictions = svc_model.predict(tfidf.transform(df['cleaned_text']))


# Create the submission file
submission = pd.DataFrame({
    'id': df['id'],
    'sentiment': predictions
})


# Save the submission file
submission.to_csv('submission.csv', index=False)
print(f"\nCreated submission.csv with {len(submission)} rows")


print("\nDistribution of predictions:")
print(submission['sentiment'].value_counts())


print("\nFirst 5 rows of submission.csv:")
print(submission.head())

