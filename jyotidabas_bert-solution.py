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


# ===========================================
# MAP Competition: BERT + GPU + High Accuracy
# Author: Jyoti Dabass
# ===========================================

import os
import re
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("ðŸš€ GPU-ENABLED SOLUTION FOR MAP COMPETITION USING BERT")
print("=" * 60)

# Load BERT tokenizer and model from Kaggle dataset
bert_dir = '/kaggle/input/bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_dir)
bert_model = BertModel.from_pretrained(bert_dir).to(device)

def preprocess_text(text):
    """Advanced text preprocessing"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\b(\d+\.\d+)\b', ' decimal_number ', text)
    text = re.sub(r'\b(\d+/\d+)\b', ' fraction_number ', text)
    text = re.sub(r'\b(\d+%)\b', ' percent_number ', text)
    text = re.sub(r'[+\-*/=<>()[\]{}]', ' math_symbol ', text)
    text = re.sub(r'\b\d+\b', ' number ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_bert_embeddings(texts, max_len=128):
    """Generate CLS embeddings using BERT pooler output"""
    embeddings = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=max_len, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        cls_embeddings = outputs.pooler_output.cpu().numpy()
        embeddings.append(cls_embeddings)
    return np.vstack(embeddings)

def train_and_predict(X_train, X_test, y_train, label_encoder, model_name):
    """Train and evaluate logistic regression"""
    print(f"\nTraining model for {model_name}...")
    clf = LogisticRegression(max_iter=1000)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        clf.fit(X_train[train_idx], y_train[train_idx])
        preds = clf.predict(X_train[val_idx])
        acc = accuracy_score(y_train[val_idx], preds)
        scores.append(acc)
        print(f"Fold {fold + 1} - Accuracy: {acc:.4f}")
    print(f"Average CV Accuracy for {model_name}: {np.mean(scores):.4f}")
    clf.fit(X_train, y_train)
    y_probs = clf.predict_proba(X_test)
    return clf, y_probs

def create_submission(test_df, category_probs, misconception_probs, category_encoder, misconception_encoder):
    """Prepare submission with top-3 joint predictions"""
    submissions = []
    for idx, row_id in enumerate(test_df['row_id']):
        top_categories = np.argsort(category_probs[idx])[::-1][:3]
        top_misconceptions = np.argsort(misconception_probs[idx])[::-1][:3]
        predictions, scores = [], []
        for cat_idx in top_categories:
            for mis_idx in top_misconceptions:
                category = category_encoder.inverse_transform([cat_idx])[0]
                misconception = misconception_encoder.inverse_transform([mis_idx])[0]
                score = category_probs[idx][cat_idx] * misconception_probs[idx][mis_idx]
                predictions.append(f"{category}:{misconception}")
                scores.append(score)
        sorted_indices = np.argsort(scores)[::-1][:3]
        top_preds = [predictions[i] for i in sorted_indices]
        while len(top_preds) < 3:
            top_preds.append("True_Correct:NA")
        submissions.append({'row_id': row_id, 'Category:Misconception': " ".join(top_preds[:3])})
    return pd.DataFrame(submissions)

# Load data
print("Loading data...")
train_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Combine and preprocess
train_df['combined_text'] = train_df['QuestionText'].fillna('') + ' [SEP] ' + train_df['StudentExplanation'].fillna('')
test_df['combined_text'] = test_df['QuestionText'].fillna('') + ' [SEP] ' + test_df['StudentExplanation'].fillna('')

train_df['processed_text'] = train_df['combined_text'].apply(preprocess_text)
test_df['processed_text'] = test_df['combined_text'].apply(preprocess_text)

# Generate embeddings
print("\nGenerating BERT embeddings...")
X_train = get_bert_embeddings(train_df['processed_text'].tolist())
X_test = get_bert_embeddings(test_df['processed_text'].tolist())

# Encode labels
category_encoder = LabelEncoder()
y_category = category_encoder.fit_transform(train_df['Category'])

misconception_encoder = LabelEncoder()
y_misconception = misconception_encoder.fit_transform(train_df['Misconception'])

# Train models
category_model, category_probs = train_and_predict(X_train, X_test, y_category, category_encoder, "Category")
misconception_model, misconception_probs = train_and_predict(X_train, X_test, y_misconception, misconception_encoder, "Misconception")

# Create submission
submission_df = create_submission(test_df, category_probs, misconception_probs, category_encoder, misconception_encoder)
submission_df.to_csv('submission.csv', index=False)
print("\nâœ… Submission saved as 'submission.csv'")
print("Sample predictions:")
print(submission_df.head())


