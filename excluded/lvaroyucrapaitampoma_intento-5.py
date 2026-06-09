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
# SOLUCIÃ“N SÃšPER-RENDIMIENTO (V4): STACKING DE MODELOS
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
from sklearn.model_selection import train_test_split, KFold # KFold para Stacking
from sklearn.preprocessing import StandardScaler # Escalador para modelos sensibles

# Modelos base para Stacking
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

# LibrerÃ­as ESTÃ�NDARES para Features
import re
import string

# Lista bÃ¡sica de palabras funcionales (detenciÃ³n)
FUNCTION_WORDS = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'by', 
    'for', 'to', 'in', 'of', 'on', 'at', 'with', 'from', 'as', 'it', 'he', 
    'she', 'they', 'we', 'you', 'i', 'this', 'that', 'these', 'those', 'can', 
    'will', 'would', 'should', 'could', 'may', 'might', 'do', 'did', 'does', 
    'have', 'has', 'had', 'es', 'son', 'fue', 'fueron', 'y', 'o', 'pero', 'por',
    'para', 'en', 'de', 'con', 'a', 'la', 'el', 'un', 'una'
])

# ==========================
# 2-4. Carga de Datos y Funciones (Igual que VersiÃ³n 3)
# ... [CÃ³digo de Carga de Paths, load_article_text, y train_texts] ...
# ==========================
BASE_DIR = "/kaggle/input/fake-or-real-the-impostor-hunt/data"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

def load_article_text(article_id, file_id, folder):
    article_folder = f"article_{article_id:04d}"
    file_path = os.path.join(folder, article_folder, f"file_{file_id}.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

train_df = pd.read_csv(TRAIN_CSV)
pairs = []
for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
    art_id = row["id"]
    real_id = row["real_text_id"]
    text1 = load_article_text(art_id, 1, folder=TRAIN_DIR)
    text2 = load_article_text(art_id, 2, folder=TRAIN_DIR)
    pairs.append((art_id, text1, text2, real_id))
train_texts = pd.DataFrame(pairs, columns=["id", "text1", "text2", "real_text_id"])

# ========================================================================
# 5. INGENIERÃ�A DE CARACTERÃ�STICAS (Igual que VersiÃ³n 3)
# ========================================================================

# --- 5.1. FunciÃ³n de EstilometrÃ­a Pura ---
def calculate_pure_stylometry(text):
    """Calcula caracterÃ­sticas estilÃ­sticas y de complejidad usando solo Python estÃ¡ndar."""
    if not text:
        return {k: 0 for k in ['TTR', 'avg_word_len', 'avg_sentence_len', 'punc_ratio', 'func_word_ratio']}
        
    words = re.findall(r'\b\w+\b', text.lower())
    sentences = re.split(r'[.!?\n]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    punc_count = sum(text.count(p) for p in string.punctuation)

    word_count = len(words)
    unique_word_count = len(set(words))
    
    TTR = unique_word_count / word_count if word_count > 0 else 0
    avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0
    avg_sentence_len = word_count / len(sentences) if len(sentences) > 0 else 0
    punc_ratio = punc_count / len(text) if len(text) > 0 else 0
    
    func_words_count = sum(1 for w in words if w in FUNCTION_WORDS)
    func_word_ratio = func_words_count / word_count if word_count > 0 else 0

    return {
        'TTR': TTR, 'avg_word_len': avg_word_len, 'avg_sentence_len': avg_sentence_len,
        'punc_ratio': punc_ratio, 'func_word_ratio': func_word_ratio
    }

# --- 5.2. VectorizaciÃ³n AVANZADA (N-Grams de CarÃ¡cter) ---
all_texts = list(train_texts["text1"]) + list(train_texts["text2"])
vectorizer_char = TfidfVectorizer(
    analyzer='char', ngram_range=(2, 5), max_features=25000 
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

    cos_sim_char = cosine_similarity(X1_train_char[i], X2_train_char[i])[0][0]
    style1 = calculate_pure_stylometry(text1)
    style2 = calculate_pure_stylometry(text2)

    features_data.append({
        'cos_sim_char': cos_sim_char, 
        'len_diff_char': abs(len(text1) - len(text2)), 
        'TTR_diff': abs(style1['TTR'] - style2['TTR']),
        'avg_word_len_diff': abs(style1['avg_word_len'] - style2['avg_word_len']),
        'avg_sentence_len_diff': abs(style1['avg_sentence_len'] - style2['avg_sentence_len']),
        'punc_ratio_diff': abs(style1['punc_ratio'] - style2['punc_ratio']),
        'func_word_ratio_diff': abs(style1['func_word_ratio'] - style2['func_word_ratio']),
    })

features = pd.DataFrame(features_data)
labels = (train_texts["real_text_id"] == 1).astype(int)
print(f"âœ… Features shape: {features.shape}")

# ========================================================================
# 6. STACKING AVANZADO (MEJORA CLAVE para > 0.7)
# ========================================================================

# Escalado de Datos (Necesario para K-Vecinos y RegresiÃ³n LogÃ­stica)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
y = labels.values # Etiquetas como array numpy

NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# --- DefiniciÃ³n de Modelos Base ---
models = [
    ('lgbm', lgb.LGBMClassifier(objective='binary', n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)),
    ('knn', KNeighborsClassifier(n_neighbors=5)), # Modelo de Vecinos Cercanos
    ('logreg', LogisticRegression(random_state=42, solver='liblinear')) # Modelo de RegresiÃ³n LogÃ­stica
]

# Matriz para almacenar las predicciones de Nivel 1 (Meta-Features)
S_train = np.zeros((X_scaled.shape[0], len(models)))

print("\nğŸ§  Iniciando Entrenamiento de Nivel 1 (Base Models)...")

# --- Entrenamiento de Modelos Base (Nivel 1) ---
for i, (name, model) in enumerate(models):
    print(f"   -> Entrenando {name}...")
    # Usamos Cross-Validation (KFold) para generar las predicciones fuera de muestra
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y)):
        X_train_f, X_val_f = X_scaled[train_idx], X_scaled[val_idx]
        y_train_f, y_val_f = y[train_idx], y[val_idx]
        
        # ParÃ¡metros especÃ­ficos para LightGBM
        if name == 'lgbm':
            model.fit(X_train_f, y_train_f,
                      eval_set=[(X_val_f, y_val_f)],
                      callbacks=[early_stopping(stopping_rounds=50, verbose=False)])
        else:
            model.fit(X_train_f, y_train_f)
        
        # Almacenar las probabilidades del modelo base
        S_train[val_idx, i] = model.predict_proba(X_val_f)[:, 1]

# --- Modelo Final (Meta-Learner) Nivel 2 ---
# Usamos un clasificador simple y robusto (RegresiÃ³n LogÃ­stica)
# para aprender a ponderar las predicciones de los modelos base.
meta_learner = LogisticRegression(random_state=42, solver='liblinear') 

print("\nğŸ�¯ Entrenando Meta-Learner (RegresiÃ³n LogÃ­stica Nivel 2)...")
meta_learner.fit(S_train, y)

print("âœ… Stacking finalizado. Meta-Learner entrenado.")

# ========================================================================
# 7. PredicciÃ³n del Conjunto de Prueba
# ========================================================================

# ========================================================================
# 7. PredicciÃ³n del Conjunto de Prueba
# ========================================================================

# --- 7.1. PreparaciÃ³n de Features de Prueba (CORRECCIÃ“N FINAL: Incluye y separa 'id') ---
test_articles = sorted(os.listdir(TEST_DIR))
test_features_data = []

print("\nâš™ï¸� Calculando features de prueba...")
# Es fundamental que el bucle de tqdm tenga la misma lista que en el bucle de IDs
for art in tqdm(test_articles):
    try:
        art_id = int(art.split("_")[1])
    except:
        continue 
    
    t1 = load_article_text(art_id, 1, folder=TEST_DIR)
    t2 = load_article_text(art_id, 2, folder=TEST_DIR)

    # Transformar a TF-IDF (N-Grams de CarÃ¡cter)
    tfidf_test_char = vectorizer_char.transform([t1, t2])
    cos_sim_char = cosine_similarity(tfidf_test_char[0], tfidf_test_char[1])[0][0]

    # EstilometrÃ­a Pura
    style1 = calculate_pure_stylometry(t1)
    style2 = calculate_pure_stylometry(t2)
    
    test_features_data.append({
        'id': art_id, # <--- Se incluye el ID en el diccionario
        'cos_sim_char': cos_sim_char, 
        'len_diff_char': abs(len(t1) - len(t2)), 
        'TTR_diff': abs(style1['TTR'] - style2['TTR']),
        'avg_word_len_diff': abs(style1['avg_word_len'] - style2['avg_word_len']),
        'avg_sentence_len_diff': abs(style1['avg_sentence_len'] - style2['avg_sentence_len']),
        'punc_ratio_diff': abs(style1['punc_ratio'] - style2['punc_ratio']),
        'func_word_ratio_diff': abs(style1['func_word_ratio'] - style2['func_word_ratio']),
    })

# DataFrame que contiene TODAS las columnas (incluido 'id')
X_test_full = pd.DataFrame(test_features_data) 

# DataFrame solo de features numÃ©ricas para el modelo
X_test = X_test_full.drop(columns=['id'])

# Escalado de Datos de Prueba (Usando el scaler ajustado en el entrenamiento)
X_test_scaled = scaler.transform(X_test)


# --- 7.2. PredicciÃ³n de Nivel 1 (Meta-Features de Prueba) ---
S_test = np.zeros((X_test_scaled.shape[0], len(models)))
for i, (name, model) in enumerate(models):
    S_test[:, i] = model.predict_proba(X_test_scaled)[:, 1]

# --- 7.3. PredicciÃ³n de Nivel 2 (Meta-Learner) ---
final_probabilities = meta_learner.predict_proba(S_test)[:, 1]


# --- 7.4. GeneraciÃ³n de EnvÃ­o (CORREGIDO: Usa X_test_full para los IDs) ---
predictions = []
# Usamos el array de IDs del DataFrame completo (X_test_full)
for art_id, prob in zip(X_test_full['id'].values, final_probabilities): 
    # Si la prob final > 0.5, predecimos que file_1 es el real (pred=1). Sino, file_2 (pred=2).
    pred = 1 if prob > 0.5 else 2
    predictions.append((art_id, pred))

submission = pd.DataFrame(predictions, columns=["id", "real_text_id"])
print("\nğŸš€ Predicciones de envÃ­o listas (primeras filas):")
print(submission.head())

# ==========================
# 8. Save Submission
# ==========================
SUBMISSION_FILE_NAME = "submission_4_stacking.csv"
submission.to_csv(SUBMISSION_FILE_NAME, index=False)
print(f"âœ… {SUBMISSION_FILE_NAME} guardado: {submission.shape}. Â¡Alto potencial para > 0.7!")


# ==========================
# 8. Save Submission
# ==========================
# Â¡AquÃ­ cambiamos el nombre del archivo!
SUBMISSION_FILE_NAME = "submission_5.csv" 

# 1. Guarda el DataFrame 'submission' con el nuevo nombre
submission.to_csv(SUBMISSION_FILE_NAME, index=False)
print(f"âœ… Archivo {SUBMISSION_FILE_NAME} guardado en el entorno remoto.")

# 2. Intento de descarga automÃ¡tica (Para Colab)
try:
    from google.colab import files
    files.download(SUBMISSION_FILE_NAME)
    print("â¬‡ï¸� DESCARGA INICIADA: Verifica tu carpeta de descargas (si estÃ¡s en Colab).")
except Exception:
    # Este es el mensaje si estÃ¡s en Kaggle o si el comando falla
    print("\nâš ï¸� FallÃ³ el cÃ³digo de descarga automÃ¡tica (files.download).")
    print(f"   Por favor, localiza y descarga manualmente el archivo '{SUBMISSION_FILE_NAME}' desde la pestaÃ±a 'Output' o el directorio de trabajo.")
    
print(f"   Formato de envÃ­o: {submission.shape}")

