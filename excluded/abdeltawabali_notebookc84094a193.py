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
from tensorflow.keras.layers import Flatten
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,accuracy_score,roc_auc_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer

# download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


# Load data
train_data = pd.read_csv('/kaggle/input/depi-r-2-competition-1/xy_train.csv')
test_data = pd.read_csv('/kaggle/input/depi-r-2-competition-1/x_test.csv')


# Inspect data
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)


# Inspect data
print( train_data.info())
print("\n\n\n\n\n")
print( test_data.info())


# Target column values
train_data['label'].unique()


# no. of rows with taget column = 2
train_data[train_data['label']== 2]


# drop rows with label = 2
train_data = train_data[train_data['label'] != 2]


# Preprocessing Data
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # remove all except digits,characters,and spaces
    text = text.lower()
    # Apply tokenization, stop words removal, and stemming
    words = word_tokenize(text) # cutting the text to words
    stop_words = set(stopwords.words('english')) #load the stopwords dictionary
    words = [word for word in words if word not in stop_words] # remove stopwords
    ps = PorterStemmer() #load the stemmer to remove the extra from the words
    words = [ps.stem(word) for word in words]
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
X_val_vec = vectorizer.fit_transform(X_val)
X_test_vec = vectorizer.fit_transform(X_test)


# Create and train the Logistic Regression model
model = LogisticRegression(C=0.1, max_iter=1000)  # Use the best parameters found earlier
model.fit(X_train_vec, y_train)

# Make predictions
y_val_pred = model.predict(X_val_vec)



# Calculate accuracy
accuracy = accuracy_score(y_val, y_val_pred)

# Calculate AUC-ROC score
y_val_prob = model.predict_proba(X_val_vec)[:, 1]  # Probability estimates for the positive class
roc_auc = roc_auc_score(y_val, y_val_prob)

# Output the results
print("Accuracy:", accuracy)
print("ROC AUC Score:", roc_auc)


# 2. Random Forest Model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train_vec, y_train)
y_val_pred_rf = random_forest_model.predict(X_val_vec)
rf_accuracy = accuracy_score(y_val, y_val_pred_rf)
print("Random Forest Accuracy:", rf_accuracy)

# 3. XGBoost Model
xgboost_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgboost_model.fit(X_train_vec, y_train)
y_val_pred_xgb = xgboost_model.predict(X_val_vec)
xgb_accuracy = accuracy_score(y_val, y_val_pred_xgb)
print("XGBoost Accuracy:", xgb_accuracy)


# تحويل Sparse Matrix إلى Dense Array
X_train_vec = X_train_vec.toarray()
X_val_vec = X_val_vec.toarray()

# 3. XGBoost Model
xgboost_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgboost_model.fit(X_train_vec, y_train)
y_val_pred_xgb = xgboost_model.predict(X_val_vec)
xgb_accuracy = accuracy_score(y_val, y_val_pred_xgb)
print("XGBoost Accuracy:", xgb_accuracy)



# Parameters
max_words = 10000
max_len = 100

# Build the Dense Neural Network Model
dnn_model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    Flatten(),  # Flatten the output from the embedding layer
    Dense(128, activation='relu'),  # First Dense layer
    Dropout(0.3),  # Dropout for regularization
    Dense(64, activation='relu'),  # Second Dense layer
    Dropout(0.3),  # Another dropout
    Dense(1, activation='sigmoid')  # Output layer for binary classification (0 or 1)
])

# Compile the model
dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# Train the model
dnn_model.fit(X_train_vec, y_train, validation_data=(X_val_vec, y_val), epochs=5, batch_size=32)

# Make predictions on the test set
y_test_pred = dnn_model.predict(X_test_vec)



y_test_pred_labels = (y_test_pred > 0.5).astype(int)



# If you want to calculate accuracy on validation set
y_val_pred = dnn_model.predict(X_val_vec)
y_val_pred_labels = (y_val_pred > 0.5).astype(int)
accuracy = accuracy_score(y_val, y_val_pred_labels)

# Output the accuracy
print("Validation Accuracy:", accuracy)


submission['label'] = y_test_pred_labels
submission.head()


submission.to_csv('submission.csv', index=False)

print("Submission file saved as 'submission.csv'")

