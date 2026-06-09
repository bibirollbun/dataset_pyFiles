#import libraries
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load Data
df_train = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")
df_test = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")



# Clean Text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_train['cleaned'] = df_train['Question'].apply(clean_text)
df_test['cleaned'] = df_test['Question'].apply(clean_text)


# Encode Labels
encoder = LabelEncoder()
df_train['label_encoded'] = encoder.fit_transform(df_train['label'])


# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_df=0.95, min_df=2)
X_train = vectorizer.fit_transform(df_train['cleaned'])
X_test = vectorizer.transform(df_test['cleaned'])

y_train = df_train['label_encoded']


# Train SVM Classifier
model = LinearSVC()
model.fit(X_train, y_train)


# Predict
preds = model.predict(X_test)
labels = encoder.inverse_transform(preds)


# Prepare submission
df_submission = pd.DataFrame({
    'id': df_test['id'],
    'label': labels
})

df_submission.to_csv("submission.csv", index=False)


