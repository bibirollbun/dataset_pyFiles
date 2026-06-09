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





# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
import re
import seaborn as sns
import matplotlib.pyplot as plt

# Load Spacy model
nlp = spacy.load('en_core_web_sm')

# Load data
train_data = pd.read_csv('/kaggle/input/depi-r-2-competition-1/xy_train.csv')
test_data = pd.read_csv('/kaggle/input/depi-r-2-competition-1/x_test.csv')

# Inspect data
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)
print(train_data.info())
print("\n\n\n\n\n")
print(test_data.info())

# Target column values
train_data['label'].unique()

# Drop rows with label = 2
train_data = train_data[train_data['label'] != 2]

# Preprocessing Data
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove all except digits, characters, and spaces
    text = text.lower()
    doc = nlp(text)
    words = [token.lemma_ for token in doc if not token.is_stop]  # Lemmatization and stopword removal
    return ' '.join(words)

# Apply preprocessing
train_data['processed_text'] = train_data['text'].apply(preprocess_text)
test_data['processed_text'] = test_data['text'].apply(preprocess_text)

# Feature and target
X = train_data['processed_text']
y = train_data['label']
X_test = test_data['processed_text']

# Submission DataFrame
submission = pd.DataFrame({'ID': test_data['ID'], 'label': None})

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# Create and train the Logistic Regression model
model = LogisticRegression(C=0.1, max_iter=1000)
model.fit(X_train_vec, y_train)

# Make predictions
y_val_pred = model.predict(X_val_vec)
accuracy = accuracy_score(y_val, y_val_pred)
roc_auc = roc_auc_score(y_val, model.predict_proba(X_val_vec)[:, 1])
print("Accuracy:", accuracy)
print("ROC AUC Score:", roc_auc)

# Random Forest Model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train_vec, y_train)
y_val_pred_rf = random_forest_model.predict(X_val_vec)
print("Random Forest Accuracy:", accuracy_score(y_val, y_val_pred_rf))

# XGBoost Model
xgboost_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgboost_model.fit(X_train_vec, y_train)
y_val_pred_xgb = xgboost_model.predict(X_val_vec)
print("XGBoost Accuracy:", accuracy_score(y_val, y_val_pred_xgb))

# Naive Bayes Model
X_train_dense = X_train_vec.toarray()
X_val_dense = X_val_vec.toarray()
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train_dense, y_train)
y_val_pred_nb = naive_bayes_model.predict(X_val_dense)
print("Naive Bayes Accuracy:", accuracy_score(y_val, y_val_pred_nb))

# Parameters
max_words = 10000
max_len = 100

# Build the Dense Neural Network Model
dnn_model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile the model
dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# Train the model
dnn_model.fit(X_train_vec, y_train, validation_data=(X_val_vec, y_val), epochs=5, batch_size=32)

# Make predictions on the test set
# y_test_pred = dnn_model.predict(X_test_vec)
# y_test_pred_labels = (y_test_pred > 0.5).astype(int)

# Validation accuracy
y_val_pred = dnn_model.predict(X_val_vec)
y_val_pred_labels = (y_val_pred > 0.5).astype(int)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred_labels))

# Save submission file
# submission['label'] = y_test_pred_labels
# submission.to_csv('submission.csv', index=False)
# print("Submission file saved as 'submission.csv'")






y_test_pred = model.predict(X_test_vec)


# Submission DataFrame
submission['label'] = y_test_pred
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")

