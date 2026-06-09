#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Проверка доступности GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Загрузка данных
print("Loading data...")
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target distribution:\n{train_df['rule_violation'].value_counts(normalize=True)}")

# Загрузка модели Stella
print("Loading Stella model...")
try:
    model_name = "infgrad/stella-en-400M-v5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    stella_model = AutoModel.from_pretrained(model_name).to(device)
    print("Stella model loaded successfully!")
except Exception as e:
    print(f"Error loading Stella model: {e}")
    print("Using alternative model...")
    model_name = "sentence-transformers/all-mpnet-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    stella_model = AutoModel.from_pretrained(model_name).to(device)

# Предобработка текста
def preprocess_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip()

print("Preprocessing text...")
train_df['cleaned_body'] = train_df['body'].apply(preprocess_text)
test_df['cleaned_body'] = test_df['body'].apply(preprocess_text)

# Создание комбинированного текста
train_df['combined_text'] = "Subreddit: " + train_df['subreddit'] + ". Rule: " + train_df['rule'] + ". Comment: " + train_df['cleaned_body']
test_df['combined_text'] = "Subreddit: " + test_df['subreddit'] + ". Rule: " + test_df['rule'] + ". Comment: " + test_df['cleaned_body']

# Функция для получения эмбеддингов
def get_embeddings(texts, batch_size=16, max_length=256):
    """Получение эмбеддингов с помощью модели Stella"""
    embeddings = []
    
    stella_model.eval()
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Токенизация
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors='pt'
        ).to(device)
        
        # Получение эмбеддингов
        with torch.no_grad():
            outputs = stella_model(**inputs)
            # Используем усреднение по последнему hidden state
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.vstack(embeddings)

# Получение эмбеддингов для тренировочных данных
print("Generating embeddings for training data...")
train_texts = train_df['combined_text'].tolist()
train_embeddings = get_embeddings(train_texts)
train_labels = train_df['rule_violation'].values

print(f"Train embeddings shape: {train_embeddings.shape}")

# Получение эмбеддингов для тестовых данных
print("Generating embeddings for test data...")
test_texts = test_df['combined_text'].tolist()
test_embeddings = get_embeddings(test_texts)

print(f"Test embeddings shape: {test_embeddings.shape}")

# Нормализация эмбеддингов
scaler = StandardScaler()
train_embeddings_scaled = scaler.fit_transform(train_embeddings)
test_embeddings_scaled = scaler.transform(test_embeddings)

# Обучение классификатора на эмбеддингах
print("Training classifier...")

# Попробуем несколько классификаторов
classifiers = {
    'LogisticRegression': LogisticRegression(
        n_jobs=-1,
        random_state=42,
        max_iter=1000,
        class_weight='balanced',
        C=0.1
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
}

best_auc = 0
best_classifier = None
best_predictions = None

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    
    # Кросс-валидация
    cv_scores = cross_val_score(clf, train_embeddings_scaled, train_labels, 
                              cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"{name} CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Обучение на всех данных
    clf.fit(train_embeddings_scaled, train_labels)
    
    # Предсказание на тренировочных данных
    train_preds = clf.predict_proba(train_embeddings_scaled)[:, 1]
    train_auc = roc_auc_score(train_labels, train_preds)
    print(f"{name} Train AUC: {train_auc:.4f}")
    
    if train_auc > best_auc:
        best_auc = train_auc
        best_classifier = clf
        best_predictions = clf.predict_proba(test_embeddings_scaled)[:, 1]

# Если ни один классификатор не был выбран, используем LogisticRegression
if best_classifier is None:
    print("Using LogisticRegression as default...")
    best_classifier = LogisticRegression(
        n_jobs=-1, random_state=42, max_iter=1000, class_weight='balanced'
    )
    best_classifier.fit(train_embeddings_scaled, train_labels)
    best_predictions = best_classifier.predict_proba(test_embeddings_scaled)[:, 1]

print(f"\nBest train AUC: {best_auc:.4f}")

# Создание submission файла
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'rule_violation': best_predictions
})

# Проверка распределения предсказаний
print(f"\nPrediction distribution:")
print(f"Min: {submission['rule_violation'].min():.4f}")
print(f"Max: {submission['rule_violation'].max():.4f}")
print(f"Mean: {submission['rule_violation'].mean():.4f}")
print(f"Std: {submission['rule_violation'].std():.4f}")

# Сохранение submission файла
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")

# Дополнительная информация
print(f"\nModel used: {model_name}")
print(f"Embedding dimension: {train_embeddings.shape[1]}")
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Очистка памяти
del stella_model, tokenizer
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("Script completed successfully!")




