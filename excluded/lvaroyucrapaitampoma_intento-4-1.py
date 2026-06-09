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


# ==============================================================================
# COMPETENCIA: Fake or Real: The Impostor Hunt in Texts
# SOLUCIÃ“N DE ALTO RENDIMIENTO (V3): N-Grams de CarÃ¡cter + EstilometrÃ­a Pura
# ==============================================================================

# ==========================
# 1. Imports
# ==========================
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

# LibrerÃ­as ESTÃ�NDARES (sin instalaciÃ³n necesaria)
import re
import string

# Lista bÃ¡sica de palabras funcionales (detenciÃ³n) en espaÃ±ol e inglÃ©s
# El texto cientÃ­fico suele ser en inglÃ©s, pero incluimos una base robusta.
FUNCTION_WORDS = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'by', 
    'for', 'to', 'in', 'of', 'on', 'at', 'with', 'from', 'as', 'it', 'he', 
    'she', 'they', 'we', 'you', 'i', 'this', 'that', 'these', 'those', 'can', 
    'will', 'would', 'should', 'could', 'may', 'might', 'do', 'did', 'does', 
    'have', 'has', 'had', 'es', 'son', 'fue', 'fueron', 'y', 'o', 'pero', 'por',
    'para', 'en', 'de', 'con', 'a', 'la', 'el', 'un', 'una'
])

# ==========================
# 2. Paths (Mismo que el original)
# ...
# ==========================
BASE_DIR = "/kaggle/input/fake-or-real-the-impostor-hunt/data"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# ==========================
# 3. FunciÃ³n de Carga de Texto
# ...
# ==========================
def load_article_text(article_id, file_id, folder):
    article_folder = f"article_{article_id:04d}"
    file_path = os.path.join(folder, article_folder, f"file_{file_id}.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# ==========================
# 4. Carga de Datos de Entrenamiento
# ... (SecciÃ³n idÃ©ntica a la original) ...
# ==========================
train_df = pd.read_csv(TRAIN_CSV)
pairs = []
for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
    art_id = row["id"]
    real_id = row["real_text_id"]
    text1 = load_article_text(art_id, 1, folder=TRAIN_DIR)
    text2 = load_article_text(art_id, 2, folder=TRAIN_DIR)
    pairs.append((art_id, text1, text2, real_id))
train_texts = pd.DataFrame(pairs, columns=["id", "text1", "text2", "real_text_id"])
print(f"âœ… Pares de entrenamiento cargados: {train_texts.shape[0]}")


# ========================================================================
# 5. INGENIERÃ�A DE CARACTERÃ�STICAS (EstilometrÃ­a Pura y N-Grams)
# ========================================================================

# --- 5.1. FunciÃ³n de EstilometrÃ­a Pura (NUEVA) ---
def calculate_pure_stylometry(text):
    """Calcula caracterÃ­sticas estilÃ­sticas y de complejidad usando solo Python estÃ¡ndar."""
    
    if not text:
        return {k: 0 for k in ['TTR', 'avg_word_len', 'avg_sentence_len', 'punc_ratio', 'func_word_ratio']}
        
    # TokenizaciÃ³n
    words = re.findall(r'\b\w+\b', text.lower())
    sentences = re.split(r'[.!?\n]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    punc_count = sum(text.count(p) for p in string.punctuation)

    word_count = len(words)
    unique_word_count = len(set(words))
    
    # MÃ©tricas
    TTR = unique_word_count / word_count if word_count > 0 else 0 # Type-Token Ratio (Diversidad lÃ©xica)
    avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0
    avg_sentence_len = word_count / len(sentences) if len(sentences) > 0 else 0
    punc_ratio = punc_count / len(text) if len(text) > 0 else 0 # Ratio de PuntuaciÃ³n
    
    # Palabras Funcionales
    func_words_count = sum(1 for w in words if w in FUNCTION_WORDS)
    func_word_ratio = func_words_count / word_count if word_count > 0 else 0

    return {
        'TTR': TTR,
        'avg_word_len': avg_word_len,
        'avg_sentence_len': avg_sentence_len,
        'punc_ratio': punc_ratio,
        'func_word_ratio': func_word_ratio
    }

# --- 5.2. VectorizaciÃ³n AVANZADA (N-Grams de CarÃ¡cter) ---
all_texts = list(train_texts["text1"]) + list(train_texts["text2"])

# Usar N-grams de caracteres (mÃ¡s sensible a errores de sintaxis y patrones)
vectorizer_char = TfidfVectorizer(
    analyzer='char',       # Analizar caracteres
    ngram_range=(2, 5),    # Usar n-grams de 2, 3, 4 y 5 caracteres
    max_features=25000     # Alto nÃºmero de features
)
tfidf_char = vectorizer_char.fit_transform(all_texts)

X1_train_char = tfidf_char[:len(train_texts)]
X2_train_char = tfidf_char[len(train_texts):]

# --- 5.3. Calcular TODAS las CaracterÃ­sticas ---
features_data = []

print("\nâš™ï¸� Calculando caracterÃ­sticas avanzadas (N-Grams y EstilometrÃ­a Pura)...")
for i in tqdm(range(len(train_texts))):
    text1 = train_texts.loc[i, "text1"]
    text2 = train_texts.loc[i, "text2"]

    # 1. Similitud (usando N-Grams de CarÃ¡cter)
    cos_sim_char = cosine_similarity(X1_train_char[i], X2_train_char[i])[0][0]

    # 2. CaracterÃ­sticas de EstilometrÃ­a
    style1 = calculate_pure_stylometry(text1)
    style2 = calculate_pure_stylometry(text2)

    features_data.append({
        'cos_sim_char': cos_sim_char, 
        'len_diff_char': abs(len(text1) - len(text2)), 
        
        # Diferencias de Estilo (NUEVAS CLAVES)
        'TTR_diff': abs(style1['TTR'] - style2['TTR']),
        'avg_word_len_diff': abs(style1['avg_word_len'] - style2['avg_word_len']),
        'avg_sentence_len_diff': abs(style1['avg_sentence_len'] - style2['avg_sentence_len']),
        'punc_ratio_diff': abs(style1['punc_ratio'] - style2['punc_ratio']),
        'func_word_ratio_diff': abs(style1['func_word_ratio'] - style2['func_word_ratio']),
    })

features = pd.DataFrame(features_data)
print(f"âœ… Features shape: {features.shape} (Ahora con {features.shape[1]} features)")
print(features.head())

# ==========================
# 6. Labels y 7. Split
# (Se mantienen iguales)
# ==========================
labels = (train_texts["real_text_id"] == 1).astype(int)
X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels)
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

# ==========================
# 8. Train LightGBM
# (Mismos parÃ¡metros optimizados)
# ==========================
params = {
    "objective": "binary", "metric": "binary_error", "verbosity": -1, 
    "boosting_type": "gbdt", "seed": 42, "n_estimators": 500,
    'num_leaves': 31, 'learning_rate': 0.05, 
}

print("\nğŸš€ Iniciando entrenamiento de LightGBM con EstilometrÃ­a Pura...")
gbm = lgb.train(
    params, lgb_train, valid_sets=[lgb_val], num_boost_round=500,
    callbacks=[early_stopping(50, verbose=False), log_evaluation(50)]
)
print(f"âœ… Entrenamiento finalizado en {gbm.best_iteration} iteraciones.")


# ========================================================================
# 9. Predict Test Set (ADAPTADO A LAS NUEVAS FEATURES)
# ========================================================================
print("\nâš™ï¸� Preparando el conjunto de prueba y generando predicciones...")

test_articles = sorted(os.listdir(TEST_DIR))
predictions = []
feature_columns = features.columns.tolist() # Lista de las 7 columnas

for art in tqdm(test_articles):
    try:
        art_id = int(art.split("_")[1])
    except:
        continue 

    t1 = load_article_text(art_id, 1, folder=TEST_DIR)
    t2 = load_article_text(art_id, 2, folder=TEST_DIR)

    # 9.1. Transformar a TF-IDF (N-Grams de CarÃ¡cter)
    tfidf_test_char = vectorizer_char.transform([t1, t2])
    
    # 9.2. Calcular Features (Similitud y EstilometrÃ­a Pura)
    cos_sim_char = cosine_similarity(tfidf_test_char[0], tfidf_test_char[1])[0][0]
    len_diff_char = abs(len(t1) - len(t2))

    # EstilometrÃ­a Pura
    style1 = calculate_pure_stylometry(t1)
    style2 = calculate_pure_stylometry(t2)
    
    # Crear la fila de features de prueba
    feat = pd.DataFrame([[
        cos_sim_char, 
        len_diff_char, 
        abs(style1['TTR'] - style2['TTR']),
        abs(style1['avg_word_len'] - style2['avg_word_len']),
        abs(style1['avg_sentence_len'] - style2['avg_sentence_len']),
        abs(style1['punc_ratio'] - style2['punc_ratio']),
        abs(style1['func_word_ratio'] - style2['func_word_ratio']),
    ]], columns=feature_columns)
    
    # 9.3. Predecir Probabilidad
    prob = gbm.predict(feat)[0] 

    # 9.4. Determinar la ID real para el envÃ­o
    pred = 1 if prob > 0.5 else 2
    predictions.append((art_id, pred))

submission = pd.DataFrame(predictions, columns=["id", "real_text_id"])
print("\nğŸš€ Predicciones de envÃ­o listas (primeras filas):")
print(submission.head())

# ==========================
# 10. Save Submission
# (El cÃ³digo de guardado es el mismo, pero lo ponemos aquÃ­ completo)
# ==========================
SUBMISSION_FILE_NAME = "submission.csv"
submission.to_csv(SUBMISSION_FILE_NAME, index=False)
print(f"âœ… {SUBMISSION_FILE_NAME} guardado: {submission.shape}. Â¡Listo para subir a Kaggle!")


# ==========================
# 10. Save and Download Submission 4
# ==========================
SUBMISSION_FILE_NAME = "submission_4.csv"

# 1. Guarda el DataFrame 'submission' en el archivo CSV del entorno
# Es crucial usar index=False para el formato de Kaggle
submission.to_csv(SUBMISSION_FILE_NAME, index=False)
print(f"âœ… Archivo {SUBMISSION_FILE_NAME} guardado en el entorno remoto.")

# 2. Intenta forzar la descarga a tu mÃ¡quina local
try:
    # Este comando funciona en Google Colab y algunos entornos Jupyter
    from google.colab import files
    files.download(SUBMISSION_FILE_NAME)
    print("â¬‡ï¸� DESCARGA INICIADA: Verifica tu carpeta de descargas.")
except ImportError:
    # Si la descarga automÃ¡tica falla (como en Kaggle Notebooks)
    print("\nâš ï¸� ADVERTENCIA: La descarga automÃ¡tica no estÃ¡ disponible o fallÃ³.")
    print(f"   Por favor, localiza y descarga manualmente el archivo '{SUBMISSION_FILE_NAME}' desde la pestaÃ±a 'Output' o el directorio de trabajo.")
    
print(f"   Formato de envÃ­o: {submission.shape}")

