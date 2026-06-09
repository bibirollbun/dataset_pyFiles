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
# SOLUCIÃ“N: LightGBM con Similitud de Coseno y Diferencia de Longitud (Baseline)
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

# ==========================
# 2. Paths
# ==========================
# NOTA: Estas rutas asumen que estÃ¡s en un entorno de Kaggle Notebook
BASE_DIR = "/kaggle/input/fake-or-real-the-impostor-hunt/data"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

print(f"Train CSV: {TRAIN_CSV}\nTrain DIR: {TRAIN_DIR}\nTest DIR : {TEST_DIR}")

# ==========================
# 3. FunciÃ³n de Carga de Texto
# ==========================
def load_article_text(article_id, file_id, folder):
    """FunciÃ³n auxiliar para leer el contenido de un archivo de texto especÃ­fico."""
    # Asegura que el ID del artÃ­culo tenga formato de 4 dÃ­gitos (e.g., 'article_0001')
    article_folder = f"article_{article_id:04d}"
    file_path = os.path.join(folder, article_folder, f"file_{file_id}.txt")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# ==========================
# 4. Carga de Datos de Entrenamiento
# ==========================
train_df = pd.read_csv(TRAIN_CSV)
print(f"\nâœ… Train shape: {train_df.shape}")

pairs = []
print("ğŸ“� Cargando pares de textos de entrenamiento...")
for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
    art_id = row["id"]
    real_id = row["real_text_id"]

    text1 = load_article_text(art_id, 1, folder=TRAIN_DIR)
    text2 = load_article_text(art_id, 2, folder=TRAIN_DIR)

    pairs.append((art_id, text1, text2, real_id))

train_texts = pd.DataFrame(pairs, columns=["id", "text1", "text2", "real_text_id"])
print(f"âœ… Pares de entrenamiento cargados: {train_texts.shape[0]}")

# ==========================
# 5. IngenierÃ­a de CaracterÃ­sticas (Feature Engineering)
# ==========================

# 5.1. Inicializar TF-IDF y ajustar con todos los textos de entrenamiento
all_texts = list(train_texts["text1"]) + list(train_texts["text2"])

# Limitar el vocabulario a 20000 para balancear rendimiento y complejidad
vectorizer = TfidfVectorizer(max_features=20000)
tfidf = vectorizer.fit_transform(all_texts)

# Separar vectores de nuevo
X1_train = tfidf[:len(train_texts)]
X2_train = tfidf[len(train_texts):]

# 5.2. Calcular CaracterÃ­sticas
cos_sims = []
len_diffs = []

print("\nâš™ï¸� Calculando caracterÃ­sticas de similitud para entrenamiento...")
for i in tqdm(range(len(train_texts))):
    # Similitud de Coseno: QuÃ© tan cerca estÃ¡n los temas
    cos_sim = cosine_similarity(X1_train[i], X2_train[i])[0][0]
    
    # Diferencia de Longitud: Una seÃ±al simple de manipulaciÃ³n/ediciÃ³n
    len_diff = abs(len(train_texts.loc[i, "text1"]) - len(train_texts.loc[i, "text2"]))
    
    cos_sims.append(cos_sim)
    len_diffs.append(len_diff)

features = pd.DataFrame({
    "cos_sim": cos_sims,
    "len_diff": len_diffs
})

# 5.3. Preparar Etiquetas (Labels)
# Target = 1 si file_1 es Real, Target = 0 si file_2 es Real
labels = (train_texts["real_text_id"] == 1).astype(int)

print(f"âœ… Features shape: {features.shape}. Labels shape: {labels.shape}")

# ==========================
# 6. Train / Validation split
# ==========================
# Se utiliza stratify=labels para asegurar que la proporciÃ³n de 1s y 0s sea similar
# en los conjuntos de entrenamiento y validaciÃ³n.
X_train, X_val, y_train, y_val = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

# ==========================
# 7. Entrenamiento del Modelo LightGBM
# ==========================
params = {
    "objective": "binary",
    "metric": "binary_error",
    "verbosity": -1, 
    "boosting_type": "gbdt",
    "seed": 42,
    "n_estimators": 500
}

print("\nğŸš€ Iniciando entrenamiento de LightGBM...")
gbm = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_val], 
    num_boost_round=500,
    callbacks=[early_stopping(50, verbose=False), log_evaluation(50)]
)

print(f"âœ… Entrenamiento finalizado en {gbm.best_iteration} iteraciones.")

# ==========================
# 8. PredicciÃ³n en el Conjunto de Prueba
# ==========================
print("\nâš™ï¸� Preparando el conjunto de prueba y generando predicciones...")

# Obtiene la lista de artÃ­culos de prueba (carpetas) y los ordena
test_articles = sorted(os.listdir(TEST_DIR))
predictions = []

for art in tqdm(test_articles):
    # Extrae el ID numÃ©rico
    try:
        art_id = int(art.split("_")[1])
    except:
        # Ignorar archivos que no son carpetas de artÃ­culos (ej. .DS_Store)
        continue 

    # Carga los textos de prueba
    t1 = load_article_text(art_id, 1, folder=TEST_DIR)
    t2 = load_article_text(art_id, 2, folder=TEST_DIR)

    # 8.1. Transformar a TF-IDF (usando el vectorizador ajustado en el entrenamiento)
    tfidf_test = vectorizer.transform([t1, t2])
    
    # 8.2. Calcular CaracterÃ­sticas (la misma lÃ³gica de sim. de coseno y diff. de longitud)
    cos_sim = cosine_similarity(tfidf_test[0], tfidf_test[1])[0][0]
    len_diff = abs(len(t1) - len(t2))

    # Crear DataFrame de caracterÃ­sticas de una sola fila
    feat = pd.DataFrame([[cos_sim, len_diff]], columns=["cos_sim", "len_diff"])
    
    # 8.3. Predecir Probabilidad (probabilidad de que 'text1' sea REAL)
    prob = gbm.predict(feat)[0] 

    # 8.4. Determinar la ID real para el envÃ­o
    # Si prob > 0.5, predecimos que file_1 es el real (pred=1). Sino, file_2 es el real (pred=2).
    pred = 1 if prob > 0.5 else 2
    predictions.append((art_id, pred))

submission = pd.DataFrame(predictions, columns=["id", "real_text_id"])
print("\nğŸš€ Predicciones de envÃ­o listas (primeras filas):")
print(submission.head())

# ==========================
# 9. Guardar EnvÃ­o
# ==========================
SUBMISSION_FILE_NAME = "submission.csv"
submission.to_csv(SUBMISSION_FILE_NAME, index=False)
print(f"âœ… {SUBMISSION_FILE_NAME} guardado: {submission.shape}. Â¡Listo para subir a Kaggle!")

