# ============================================
# Fake vs Real Classification - Random Forest Only
# ============================================

import os
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# -----------------------------
# Helper: Read text pairs
# -----------------------------
def read_texts_from_dir(dir_path):
    data = []
    for folder_name in sorted(os.listdir(dir_path)):
        folder_path = os.path.join(dir_path, folder_name)
        if os.path.isdir(folder_path):
            try:
                with open(os.path.join(folder_path, 'file_1.txt'), 'r', encoding='utf-8') as f1:
                    text1 = f1.read().strip()
                with open(os.path.join(folder_path, 'file_2.txt'), 'r', encoding='utf-8') as f2:
                    text2 = f2.read().strip()
                index = int(folder_name[-4:])
                data.append((index, text1, text2))
            except Exception as e:
                print(f"Error reading {folder_name}: {e}")
    return pd.DataFrame(data, columns=['id', 'file_1', 'file_2']).set_index('id')

# -----------------------------
# Enhanced Text Preprocessing
# -----------------------------
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    
    stemmer = PorterStemmer()
    stemmed_text = [stemmer.stem(word) for word in filtered_text]
    
    return " ".join(stemmed_text)

# -----------------------------
# Text Feature Extractor
# -----------------------------
class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        features = []
        for text in X:
            if not isinstance(text, str):
                features.append([0, 0, 0, 0, 0])
                continue
            length = len(text)
            words = text.split()
            word_count = len(words)
            avg_word_len = np.mean([len(word) for word in words]) if word_count > 0 else 0
            sentence_count = text.count('.') + text.count('!') + text.count('?')
            unique_ratio = len(set(words)) / word_count if word_count > 0 else 0
            features.append([length, word_count, avg_word_len, sentence_count, unique_ratio])
        return np.array(features)

# -----------------------------
# Load Data
# -----------------------------
train_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
test_path  = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"
df_train = read_texts_from_dir(train_path)
df_test  = read_texts_from_dir(test_path)

labels = pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv").set_index("id")
df_train = df_train.join(labels)

print("Training samples:", len(df_train))
print(df_train.head())

# -----------------------------
# Preprocessing
# -----------------------------
print("Preprocessing texts...")
for col in ["file_1", "file_2"]:
    df_train[col] = df_train[col].apply(preprocess_text)
    df_test[col]  = df_test[col].apply(preprocess_text)

# -----------------------------
# Build training set
# -----------------------------
texts, y = [], []
for _, row in df_train.iterrows():
    texts.append(row["file_1"])
    y.append(1 if row["real_text_id"] == 1 else 0)
    texts.append(row["file_2"])
    y.append(1 if row["real_text_id"] == 2 else 0)
y = np.array(y)

# -----------------------------
# Feature pipeline
# -----------------------------
feature_union = FeatureUnion([
    ('tfidf', TfidfVectorizer(
        max_features=30000, ngram_range=(1, 3),
        min_df=2, max_df=0.95, sublinear_tf=True
    )),
    ('count', CountVectorizer(
        max_features=10000, ngram_range=(1, 2), binary=True
    )),
    ('stats', Pipeline([
        ('extractor', TextFeatureExtractor()),
        ('scaler', StandardScaler())
    ]))
])

print("Extracting features...")
X = feature_union.fit_transform(texts)

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")

# -----------------------------
# Random Forest Training
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)
val_preds = rf_model.predict(X_val)
score = accuracy_score(y_val, val_preds)
print(f"Random Forest Validation Accuracy: {score:.4f}")
print(classification_report(y_val, val_preds))

# -----------------------------
# Train Final Model on Full Data
# -----------------------------
print("Training Random Forest on full data...")
rf_model.fit(X, y)

# -----------------------------
# Predict Test Set
# -----------------------------
print("Preparing test data...")
test_texts = []
for _, row in df_test.iterrows():
    test_texts.append(row["file_1"])
    test_texts.append(row["file_2"])

print("Extracting test features...")
X_test = feature_union.transform(test_texts)

print("Making predictions...")
test_probs = rf_model.predict_proba(X_test)

preds = []
for i in range(0, len(test_probs), 2):
    score1 = test_probs[i][1]  # Probability that text1 is real
    score2 = test_probs[i+1][1]  # Probability that text2 is real
    real = 1 if score1 > score2 else 2
    preds.append((df_test.index[i//2], real))

df_sub = pd.DataFrame(preds, columns=["id", "real_text_id"])
df_sub.to_csv("submission.csv", index=False)

print("Final submission.csv saved using Random Forest!")
print(df_sub.head())


