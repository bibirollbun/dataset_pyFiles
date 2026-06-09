import pandas as pd
import os
import pandas as pd
import os
import re
import string
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset

# Path to your data
data_dir = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"  # each folder like 123/, 124/...
train_csv_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"

# Read the train CSV
train_df = pd.read_csv(train_csv_path)

# Prepare the final dataset
data = []
    
for _, row in train_df.iterrows():
    folder_id = str(row[0])  # first column: folder name
    real_text_id =str(row[1])  
    if len(folder_id)==1:
        folder_name = f"article_{'000'+folder_id}" 
    else:
         folder_name = f"article_{'00'+folder_id}" 
    
    folder_path = os.path.join(data_dir, folder_name)
      # Loop through both files
    for file_id in ["1", '2']:
        file_name = f"file_{file_id}.txt"
        file_path = os.path.join(folder_path, file_name)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            continue

        # Assign label: 1 = real, 0 = fake
        label = 1 if file_id == real_text_id else 0

        data.append({'text': text, 'real': label})

# Save to DataFrame
df = pd.DataFrame(data)
df.to_csv("train_individual_texts.csv", index=False)

print("✅ Done! Saved as train_individual_texts.csv")
print(df.head())


df


print(df['text'].isna().sum())



df = df.dropna(subset=['text'])


import os

# Test folder path
test_path = '/kaggle/input/fake-or-real-the-impostor-hunt/data/test'

# Initialize a list to store the test data
test_data = []

# Loop over each folder in the test directory
for folder in os.listdir(test_path):
    folder_path = os.path.join(test_path, folder)
    if os.path.isdir(folder_path):
        # Read file_1
        file_1_path = os.path.join(folder_path, 'file_1.txt')
        if os.path.exists(file_1_path):
            with open(file_1_path, 'r', encoding='utf-8') as f:
                text1 = f.read()
            test_data.append({'folder': folder, 'file_id': 'file_1', 'text': text1})

        # Read file_2
        file_2_path = os.path.join(folder_path, 'file_2.txt')
        if os.path.exists(file_2_path):
            with open(file_2_path, 'r', encoding='utf-8') as f:
                text2 = f.read()
            test_data.append({'folder': folder, 'file_id': 'file_2', 'text': text2})

# Convert to DataFrame
test_df = pd.DataFrame(test_data)



df['real'] =df['real'].astype(int)


print(df['text'].isna().sum())


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# ------------------ 1. Feature Extraction (Unigrams + Bigrams) ------------------
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['real']
X_test = vectorizer.transform(test_df['text'])

# ------------------ 2. Train-Validation Split ------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ 3. Train Models ------------------
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'GradientBoosting': GradientBoostingClassifier(),
    'SVM': SVC(probability=True)
}

trained_models = {}
val_scores = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model

    val_preds = model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    val_scores[name] = acc
    print(f"{name} Validation Accuracy: {acc:.4f}")

# ------------------ 4. Predict on Test ------------------
for name, model in trained_models.items():
    test_df[f'prediction_{name}'] = model.predict(X_test)

# ------------------ 5. Final Predictions ------------------

# A. Best Model
best_model_name = max(val_scores, key=val_scores.get)
test_df['final_prediction_best_model'] = test_df[f'prediction_{best_model_name}']

# B. Majority Vote
pred_cols = [col for col in test_df.columns if col.startswith("prediction_")]
test_df['final_prediction_majority_vote'] = test_df[pred_cols].mode(axis=1)[0]

test_df['article_id'] = test_df['folder'].str.extract(r'article_(\d+)').astype(int)
test_df['file_number'] = test_df['file_id'].str.extract(r'file_(\d+)').astype(int)
final_selection = []

for article_id in test_df['article_id'].unique():
    article_files = test_df[test_df['article_id'] == article_id]

    # 1. Try both models say it's real
    match = article_files[
        (article_files['final_prediction_best_model'] == 1) &
        (article_files['final_prediction_majority_vote'] == 1)
    ]
    if match.empty:
        # 2. Fallback to best model
        match = article_files[article_files['final_prediction_best_model'] == 1]
    if match.empty:
        # 3. Fallback to majority vote
        match = article_files[article_files['final_prediction_majority_vote'] == 1]
    if match.empty:
        # 4. Fallback to first file
        match = article_files.iloc[[0]]

    # Pick the top file (or highest prob if available)
    match = match.sort_values(['file_number'])  # Add score sorting if needed
    selected = match.iloc[0]
    final_selection.append({'id': selected['article_id'], 'real_text_id': selected['file_number']})

# Create submission
submission_df = pd.DataFrame(final_selection)
submission_df = submission_df.sort_values('id')
submission_df.to_csv("submission.csv", index=False)
print("✅ Fallback-safe submission created.")



submission_df




