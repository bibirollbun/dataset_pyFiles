# Celda 1 de tu notebook
!pip uninstall -y tensorflow
!pip install -q sentence-transformers
!pip install -q sentence-transformers emoji langdetect


import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import emoji
from langdetect import detect, LangDetectException

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from transformers import pipeline
from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings('ignore')

# Semilla para langdetect
from langdetect import DetectorFactory
DetectorFactory.seed = 0

print("✅ Librerías importadas")


# Cargar datos
train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv').set_index("PetID")
test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv').set_index("PetID")

print(f"Train: {train.shape}, Test: {test.shape}")


# Preparar textos
train_texts = train['Description'].fillna('').astype(str).tolist()
test_texts = test['Description'].fillna('').astype(str).tolist()

train_texts_clean = [t if len(t.strip()) > 0 else "no description" for t in train_texts]
test_texts_clean = [t if len(t.strip()) > 0 else "no description" for t in test_texts]


def safe_is_english(text):
    try:
        # Si es muy corto o solo números, langdetect suele fallar o dar erróneos
        if len(text) < 3: return 0 
        return 1 if detect(text) == 'en' else 0
    except LangDetectException:
        return 0


def extract_linguistic_features(texts):
    features = []
    for text in tqdm(texts, desc="Features lingüísticas"):
        text = str(text) if text else ""
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = max(1, len(re.split(r'[.!?]+', text)))
        
        features.append({
            'txt_char_count': char_count,
            'txt_word_count': word_count,
            'txt_sentence_count': sentence_count,
            'txt_avg_word_len': np.mean([len(w) for w in text.split()]) if word_count > 0 else 0,
            'txt_exclamation': text.count('!'),
            'txt_question': text.count('?'),
            'txt_uppercase_ratio': sum(1 for c in text if c.isupper()) / max(1, char_count),
            'txt_is_empty': 1 if char_count < 5 else 0,
            # --- NUEVAS FEATURES ---
            'txt_emoji_count': emoji.emoji_count(text),
            'txt_is_english': safe_is_english(text),
            # Keywords importantes (de EDA de competidores)
            'txt_has_adopt': 1 if 'adopt' in text.lower() else 0,
            'txt_has_rescue': 1 if 'rescue' in text.lower() else 0,
            'txt_has_urgent': 1 if 'urgent' in text.lower() else 0,
            'txt_has_free': 1 if 'free' in text.lower() else 0,
            'txt_has_loving': 1 if 'loving' in text.lower() else 0,
            'txt_has_friendly': 1 if 'friendly' in text.lower() else 0,
            'txt_has_playful': 1 if 'playful' in text.lower() else 0,
        })
    return pd.DataFrame(features)

train_ling = extract_linguistic_features(train_texts)
test_ling = extract_linguistic_features(test_texts)
print(f"Lingüísticas: {train_ling.shape[1]} features")


# TF-IDF sobre descripción
N_SVD_COMPONENTS = 64

print("Calculando TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True
)

# Fit en train+test (esto es válido para TF-IDF ya que no usa el target)
all_texts = train_texts_clean + test_texts_clean
tfidf.fit(all_texts)

train_tfidf = tfidf.transform(train_texts_clean)
test_tfidf = tfidf.transform(test_texts_clean)

print(f"TF-IDF shape: {train_tfidf.shape}")

# SVD para reducir dimensionalidad
print(f"Aplicando SVD ({N_SVD_COMPONENTS} componentes)...")
svd = TruncatedSVD(n_components=N_SVD_COMPONENTS, random_state=42)
svd.fit(train_tfidf)  # Fit solo en train

train_svd = svd.transform(train_tfidf)
test_svd = svd.transform(test_tfidf)

train_svd_df = pd.DataFrame(train_svd, columns=[f'tfidf_svd_{i}' for i in range(N_SVD_COMPONENTS)])
test_svd_df = pd.DataFrame(test_svd, columns=[f'tfidf_svd_{i}' for i in range(N_SVD_COMPONENTS)])

print(f"SVD varianza explicada: {svd.explained_variance_ratio_.sum():.2%}")


# Sentence embeddings con modelo moderno
#print("Generando sentence embeddings...")
#sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

print("Generando sentence embeddings (Multilingual)...")
sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda')

train_emb = sentence_model.encode(train_texts_clean, show_progress_bar=True, batch_size=64)
test_emb = sentence_model.encode(test_texts_clean, show_progress_bar=True, batch_size=64)

# Reducir con SVD para evitar demasiadas features
N_EMB_SVD = 64
svd_emb = TruncatedSVD(n_components=N_EMB_SVD, random_state=42)
train_emb_svd = svd_emb.fit_transform(train_emb)
test_emb_svd = svd_emb.transform(test_emb)

train_emb_df = pd.DataFrame(train_emb_svd, columns=[f'emb_svd_{i}' for i in range(N_EMB_SVD)])
test_emb_df = pd.DataFrame(test_emb_svd, columns=[f'emb_svd_{i}' for i in range(N_EMB_SVD)])

print(f"Embeddings SVD: {train_emb_df.shape[1]} features")


# Sentimiento
print("Analizando sentimiento...")
sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", 
                     device=0, batch_size=32)

train_sent = sentiment(train_texts_clean, truncation=True, max_length=512)
test_sent = sentiment(test_texts_clean, truncation=True, max_length=512)

train_sent_df = pd.DataFrame([{
    'sent_positive': 1 if r['label'] == 'POSITIVE' else 0,
    'sent_score': r['score'] if r['label'] == 'POSITIVE' else 1 - r['score']
} for r in train_sent])

test_sent_df = pd.DataFrame([{
    'sent_positive': 1 if r['label'] == 'POSITIVE' else 0,
    'sent_score': r['score'] if r['label'] == 'POSITIVE' else 1 - r['score']
} for r in test_sent])

print(f"Sentimiento: {train_sent_df.shape[1]} features")


# Combinar todas las features
train_features = pd.concat([
    train_ling.reset_index(drop=True),
    train_svd_df.reset_index(drop=True),
    train_emb_df.reset_index(drop=True),
    train_sent_df.reset_index(drop=True),
], axis=1)
train_features.index = train.index

test_features = pd.concat([
    test_ling.reset_index(drop=True),
    test_svd_df.reset_index(drop=True),
    test_emb_df.reset_index(drop=True),
    test_sent_df.reset_index(drop=True),
], axis=1)
test_features.index = test.index

print(f"\n✅ Total features de texto: {train_features.shape[1]}")


# Guardar
train_features.to_parquet("train.parquet")
test_features.to_parquet("test.parquet")

print(f"Guardado: train.parquet {train_features.shape}")
print(f"Guardado: test.parquet {test_features.shape}")







