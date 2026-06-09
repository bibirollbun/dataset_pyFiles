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

# Load the datasets
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')

# Display the first few rows of each dataset
train_df.head(), test_df.head(), sample_submission_df.head()


# Check for missing values in the train and test datasets
train_missing = train_df.isnull().sum()
test_missing = test_df.isnull().sum()

train_missing, test_missing


import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download('stopwords', quiet=True)

# Function to clean text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply cleaning to the 'body' column in both datasets
train_df['cleaned_body'] = train_df['body'].apply(clean_text)
test_df['cleaned_body'] = test_df['body'].apply(clean_text)

# Display a sample of cleaned text
train_df[['body', 'cleaned_body']].head()


# Simplified function to clean text (without stopwords)
def clean_text_simple(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Apply simplified cleaning to the 'body' column in both datasets
train_df['cleaned_body'] = train_df['body'].apply(clean_text_simple)
test_df['cleaned_body'] = test_df['body'].apply(clean_text_simple)

# Display a sample of cleaned text
train_df[['body', 'cleaned_body']].head()


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

# TF-IDF Vectorization for the 'cleaned_body' text
tfidf = TfidfVectorizer(max_features=5000)
X_text = tfidf.fit_transform(train_df['cleaned_body'])

# One-Hot Encoding for the 'subreddit' column
encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
X_subreddit = encoder.fit_transform(train_df[['subreddit']])

# Combine text and subreddit features
X_train = pd.concat([
    pd.DataFrame(X_text.toarray()),
    pd.DataFrame(X_subreddit.toarray())
], axis=1)

# Target variable
y_train = train_df['rule_violation']

# Display shapes
X_train.shape, y_train.shape


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Train a baseline logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_split, y_train_split)

# Predict probabilities on the validation set
y_pred_proba = model.predict_proba(X_val_split)[:, 1]

# Calculate AUC score
auc_score = roc_auc_score(y_val_split, y_pred_proba)

auc_score


# Get the one-hot encoded subreddit features as a DataFrame
subreddit_encoded_df = pd.DataFrame(X_subreddit.toarray(), columns=encoder.get_feature_names_out(['subreddit']))

# Display a sample of the one-hot encoded subreddit features
subreddit_encoded_sample = subreddit_encoded_df.head(10)

# Display the first few rows of the sample submission file
submission_sample = sample_submission_df.head()

subreddit_encoded_sample, submission_sample


import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of the target variable 'rule_violation'
plt.figure(figsize=(8, 5))
sns.countplot(x='rule_violation', data=train_df)
plt.title('Distribution of Rule Violation in Training Data')
plt.xlabel('Rule Violation')
plt.ylabel('Count')
plt.xticks([0, 1])
plt.show()


# Create a sample submission file with 100 rows
sample_submission_100 = sample_submission_df.head(100)

# Save the sample submission file to a CSV
sample_submission_path = 'sample_submission_100.csv'
sample_submission_100.to_csv(sample_submission_path, index=False)

sample_submission_path


# Preprocess the test data (TF-IDF and one-hot encoding)
X_test_text = tfidf.transform(test_df['cleaned_body'])
X_test_subreddit = encoder.transform(test_df[['subreddit']])

# Combine text and subreddit features for the test set
X_test = pd.concat([
    pd.DataFrame(X_test_text.toarray()),
    pd.DataFrame(X_test_subreddit.toarray())
], axis=1)

# Predict probabilities for the test set
y_test_pred_proba = model.predict_proba(X_test)[:, 1]

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': y_test_pred_proba
})

# Save the submission file
submission_path = 'submission.csv'
submission_df.to_csv(submission_path, index=False)

submission_path


# Redefine and train the logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict probabilities on the training data for ROC curve
y_pred_proba_train = model.predict_proba(X_train)[:, 1]

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_train, y_pred_proba_train)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

