#import libraries
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv')
test_df = pd.read_csv('/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv')


vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(train_df['Question'])
y = train_df['label']



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))


X_test = vectorizer.transform(test_df['Question'])
test_predictions = model.predict(X_test)


submission = pd.DataFrame({
    'id': test_df['id'],
    'label': test_predictions
})
submission.to_csv('submission1.csv', index=False)

