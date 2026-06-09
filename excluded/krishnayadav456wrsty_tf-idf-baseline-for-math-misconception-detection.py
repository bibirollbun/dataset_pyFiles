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
import numpy as np
from pathlib import Path
import os
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Check available files
input_dir = Path('/kaggle/input/map-charting-student-math-misunderstandings')
print("Available files:")
for file in input_dir.glob('*'):
    print(file.name)

# Load data
train_df = pd.read_csv(input_dir / 'train.csv')
test_df = pd.read_csv(input_dir / 'test.csv')
sample_submission = pd.read_csv(input_dir / 'sample_submission.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove special characters but keep mathematical symbols
    text = re.sub(r'[^\w\s\.\-\+\*\=\>\<\(\)]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Create combined text features
def create_features(df):
    df = df.copy()
    for col in ['QuestionText', 'MC_Answer', 'StudentExplanation']:
        df[col] = df[col].apply(preprocess_text)
    
    df['text'] = (
        "Question: " + df['QuestionText'] + 
        " | Answer: " + df['MC_Answer'] + 
        " | Explanation: " + df['StudentExplanation']
    )
    return df

train_df = create_features(train_df)
test_df = create_features(test_df)

# Create label for training - handle NaN values
train_df['Misconception'] = train_df['Misconception'].fillna('NA')
train_df['label'] = train_df['Category'] + ':' + train_df['Misconception']

# Encode labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_df['label'])
num_labels = len(label_encoder.classes_)

print(f"Number of unique labels: {num_labels}")

# Handle the case where some classes have only one sample
indices = np.arange(len(train_df))
labels = train_labels_encoded

train_indices = []
val_indices = []

for class_id in np.unique(labels):
    class_indices = indices[labels == class_id]
    if len(class_indices) == 1:
        # For classes with only one sample, put it in training
        train_indices.extend(class_indices)
    else:
        # For classes with multiple samples, do a stratified split
        class_train_indices, class_val_indices = train_test_split(
            class_indices, 
            test_size=0.1, 
            random_state=42
        )
        train_indices.extend(class_train_indices)
        val_indices.extend(class_val_indices)

train_texts = [train_df['text'].iloc[i] for i in train_indices]
val_texts = [train_df['text'].iloc[i] for i in val_indices]
train_labels = [train_labels_encoded[i] for i in train_indices]
val_labels = [train_labels_encoded[i] for i in val_indices]

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# TF-IDF Vectorization
print("Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2,
    max_df=0.8
)

X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_df['text'])

print(f"TF-IDF matrix shape: {X_train.shape}")

# Train a Logistic Regression model
print("Training Logistic Regression model...")
model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    ),
    n_jobs=-1
)

model.fit(X_train, train_labels)

# Evaluate on validation set
val_preds = model.predict(X_val)
val_accuracy = accuracy_score(val_labels, val_preds)
print(f"Validation accuracy: {val_accuracy:.4f}")

# Get predicted probabilities for test set
test_probs = model.predict_proba(X_test)

# Get top 3 predictions for each sample
top3_indices = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]

# Convert indices to labels
top3_labels = []
for row in top3_indices:
    row_labels = []
    for idx in row:
        if idx < len(label_encoder.classes_):
            row_labels.append(label_encoder.classes_[idx])
        else:
            # Fallback to most common label if index is out of bounds
            row_labels.append(label_encoder.classes_[0])
    top3_labels.append(row_labels)

# Format predictions for submission
def format_predictions(labels_row):
    return " ".join(str(label) for label in labels_row)

test_df['Category:Misconception'] = [format_predictions(row) for row in top3_labels]

# Create submission file
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'Category:Misconception': test_df['Category:Misconception']
})

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission file created!")
print(submission.head())

# Show some examples of predictions
print("\nSample predictions:")
for i in range(min(3, len(test_df))):
    print(f"Question: {test_df['QuestionText'].iloc[i][:100]}...")
    print(f"Prediction: {submission['Category:Misconception'].iloc[i]}")
    print()

