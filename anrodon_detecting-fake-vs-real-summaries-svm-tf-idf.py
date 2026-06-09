import os
import re
import string
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from scipy.sparse import hstack, csr_matrix

RANDOM_STATE = 42


nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')

# Set of English stopwords for later calculations
stopwords_set = set(stopwords.words('english'))


train_rules = pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv")
train_rules.head()


def create_article_path(text_id, article):
    # Function to construct the path for a given article and text file
    path = f"/kaggle/input/fake-or-real-the-impostor-hunt/data/train/article_{article}/file_{text_id}.txt"
    return path

def read_file(file_path):
    # Function to read the content of the file at the given path
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    return text


# Create the fake_text_id column: if real is 1, fake is 2; otherwise fake is 1
train_rules['fake_text_id'] = train_rules['real_text_id'].apply(lambda x: 2 if x == 1 else 1)

# Format article ID with leading zeros (e.g., '0001')
train_rules['article'] = train_rules['id'].apply(lambda x: str(x).zfill(4))

# Construct file paths for real and fake texts
train_rules['real_text_file'] = train_rules[['real_text_id', 'article']].apply(
    lambda x: create_article_path(x['real_text_id'], x['article']), axis=1)
train_rules['fake_text_file'] = train_rules[['fake_text_id', 'article']].apply(
    lambda x: create_article_path(x['fake_text_id'], x['article']), axis=1)

# Load text content from each file
train_rules['real_text'] = train_rules['real_text_file'].apply(read_file)
train_rules['fake_text'] = train_rules['fake_text_file'].apply(read_file)


# DataFrame with only real texts
df_real = train_rules[['article', 'real_text']].copy()
df_real.columns = ['article', 'text']  # Standardize column names
df_real['label'] = 1

# DataFrame with only fake texts
df_fake = train_rules[['article', 'fake_text']].copy()
df_fake.columns = ['article', 'text']
df_fake['label'] = 0

# Concatenate real and fake data into one DataFrame
df_full = pd.concat([df_real, df_fake], ignore_index=True)

# Show result
df_full.head()


sample_article = '0090'
print(f"Label: {df_full[df_full['article'] == sample_article].label.iloc[0]}")
df_full[df_full['article'] == sample_article].text.iloc[0]


def entropy(text):
    """Calculate the Shannon entropy of the text."""
    if len(text) == 0:
        return 0
    probs = [v / len(text) for v in Counter(text).values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def extract_features(text):
    words = word_tokenize(text.lower())
    word_count = len(words)
    non_latin = len(re.findall(r'[^\x00-\x7F]', text))
    punct_count = sum(1 for c in text if c in string.punctuation)
    num_lines = text.count('\n')
    stop_count = sum(1 for w in words if w in stopwords_set)
    long_words = sum(1 for w in words if len(w) > 15)
    ent = entropy(text)
    num_sentences = text.count('.') + text.count('!') + text.count('?') + text.count('-')
    avg_sent_len = word_count / (num_sentences + 1)

    return {
        'text_length': word_count,
        'non_latin_chars': non_latin,
        'punctuation_count': punct_count,
        'line_count': num_lines,
        'stopword_ratio': stop_count / word_count if word_count > 0 else 0,
        'long_words': long_words,
        'entropy': ent,
        'num_sentences': num_sentences,
        'avg_sentence_len': avg_sent_len
    }

# Extract features for every text
feature_df = df_full['text'].apply(extract_features).apply(pd.Series)
df_full = pd.concat([df_full, feature_df], axis=1)
df_full.head()


features = ['text_length', 'non_latin_chars', 'punctuation_count', 'line_count',
            'stopword_ratio', 'long_words', 'entropy', 'num_sentences', 'avg_sentence_len']

# 3x3 grid of boxplots
fig, axes = plt.subplots(3, 3, figsize=(18, 12))  # (rows, columns)
fig.suptitle('Feature Distributions by Class (0 = fake, 1 = real)', fontsize=16)

for i, feature in enumerate(features):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    sns.boxplot(x='label', y=feature, data=df_full, ax=ax)
    ax.set_title(feature)
    ax.set_xlabel("")
    ax.set_ylabel("")

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for main title
plt.show()



X = df_full[features]
y = df_full['label']

# Compute ANOVA F-values and p-values for each feature
f_vals, p_vals = f_classif(X, y)

# Compute Mutual Information scores for each feature
mi_scores = mutual_info_classif(X, y, random_state=42)

# Summarize and rank features
ranking = pd.DataFrame({
    'feature': X.columns,
    'anova_f': f_vals,
    'anova_p': p_vals,
    'mutual_info': mi_scores
}).sort_values(by='mutual_info', ascending=False)

ranking



# Select only the feature columns
df_corr = df_full[features]

# Calculate the correlation matrix
correlation_matrix = df_corr.corr()

# Plot the heatmap of feature correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


# Select only the columns useful for modeling
useful_columns = ['article', 'text', 'text_length', 'stopword_ratio', 'entropy', 'label', 'avg_sentence_len']
df_final = df_full[useful_columns].copy()
df_final.head()


columns_x = ['text_length', 'stopword_ratio', 'entropy', 'avg_sentence_len']
columns_y = 'label'

X = df_final[columns_x]
y = df_final[columns_y]
X.head()


# Shuffle the dataset
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# TF-IDF vectorization of the text
tfidf = TfidfVectorizer(max_features=500)
X_tfidf = tfidf.fit_transform(df_final['text'])

# Standardization of handcrafted features
scaler = StandardScaler()
X_extra = scaler.fit_transform(df_final[columns_x])

# Combine TF-IDF and handcrafted features
X_combined = hstack([X_tfidf, X_extra])
X_combined = csr_matrix(X_combined)  # Allows sparse matrix indexing

# Target variable
y = df_final['label'].values


# Define models to compare
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# 5-fold stratified cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    accuracies, precisions, recalls, f1s = [], [], [], []

    for train_index, test_index in kf.split(X_combined, y):
        X_train, X_test = X_combined[train_index], X_combined[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

        accuracies.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    print(f"\nğŸ“Œ Model: {name}")
    print(f"Accuracy:  {np.mean(accuracies):.4f}")
    print(f"Precision: {np.mean(precisions):.4f}")
    print(f"Recall:    {np.mean(recalls):.4f}")
    print(f"F1-score:  {np.mean(f1s):.4f}")


# 1. Model and pipeline definition
svm = SVC(probability=True)

pipeline = Pipeline([
    ('clf', svm)
])

# 2. Hyperparameters to search
param_grid = {
    'clf__kernel': ['linear', 'rbf'],
    'clf__C': [0.1, 1, 10],
    'clf__gamma': ['scale', 'auto']  # Only affects RBF kernel
}

# 3. Stratified 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=cv,
    n_jobs=-1,  # Use all CPU cores
    verbose=2
)

# 4. Run Grid Search
grid.fit(X_combined, y)

# 5. Results
print("ğŸ”� Best parameter combination:")
print(grid.best_params_)
print("\nğŸ“ˆ Best mean F1-score (cross-validation):")
print(grid.best_score_)


# Retrieve the best model from the grid search
best_model = grid.best_estimator_

# Retrain the best model using the full dataset
best_model.fit(X_combined, y)


# 1. Read test folders in sorted order
data_path = Path("/kaggle/input/fake-or-real-the-impostor-hunt/data/test")
folders = sorted([f for f in data_path.iterdir() if f.is_dir()])

submission_rows = []

for idx, folder in enumerate(folders):  
    texts = []
    for fname in ["file_1.txt", "file_2.txt"]:
        fp = folder / fname
        texts.append(read_file(fp))
    
    feat_rows = [extract_features(t) for t in texts]
    X_text = tfidf.transform(texts)
    X_manual = scaler.transform(pd.DataFrame(feat_rows)[columns_x])
    X_pair = hstack([X_text, X_manual])
    
    scores = best_model.predict_proba(X_pair)[:, 1]
    chosen = 1 if scores[0] > scores[1] else 2
    
    submission_rows.append({"id": idx, "real_text_id": chosen})  

submission = pd.DataFrame(submission_rows)
submission.to_csv("submission.csv", index=False)
print(submission.head())

