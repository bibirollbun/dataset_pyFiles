import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_df = pd.read_csv("/kaggle/input/kcvanguard-deep-learning-assignment/train-reviews-gmaps.csv")
test_df = pd.read_csv("/kaggle/input/kcvanguard-deep-learning-assignment/test-review-gmaps-new.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print(train_df.head())


train_df['label'] = train_df['label'].map({'Positive': 1, 'Negative': 0})

print(train_df.columns)


X_train, X_val, y_train, y_val = train_test_split(
    train_df['reviews'], train_df['label'], test_size=0.2, random_state=42
)


tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(test_df['reviews'])


model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_val_tfidf)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))


test_pred = model.predict(X_test_tfidf)
test_df['label'] = np.where(test_pred == 1, 'Positive', 'Negative')


submission = test_df[['id', 'label']]
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("✅ Submission file saved as submission.csv")

