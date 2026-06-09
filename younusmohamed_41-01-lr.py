# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset
train_data = pd.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/train.parquet')


# Preprocess data
X = train_data['prompt'] + ' ' + train_data['response_a'] + ' ' + train_data['response_b']
y = train_data['winner']

# Split data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)


# Train a logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_tfidf, y_train)


# Evaluate on validation set
y_pred = model.predict(X_val_tfidf)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2%}")


# Generate submission file
test_data = pd.read_parquet('/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet')
X_test = test_data['prompt'] + ' ' + test_data['response_a'] + ' ' + test_data['response_b']
X_test_tfidf = vectorizer.transform(X_test)
test_data['winner'] = model.predict(X_test_tfidf)


# Save submission
submission = test_data[['id', 'winner']]
submission.to_csv('submission.csv', index=False)
print("Submission file created.")




