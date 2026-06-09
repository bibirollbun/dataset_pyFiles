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
# SOLUCIÃ“N FINAL V11.4: BI-LSTM HÃ�BRIDO (ANTI-CRASH EDITION)
# ==============================================================================

import os
# PARCHE DE SEGURIDAD: Previene errores de Protobuf/Keras
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
import numpy as np
import re
import string
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import (Input, Embedding, Bidirectional, LSTM, 
                                     GlobalMaxPool1D, Dense, Dropout, Concatenate, BatchNormalization)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler

# --- CONFIGURACIÃ“N ---
MAX_WORDS = 15000
MAX_LEN = 300
EMBEDDING_DIM = 100
DATA_PATH = "/kaggle/input/fake-or-real-the-impostor-hunt/data"

# 1. FUNCIÃ“N DE ESTILOMETRÃ�A BLINDADA
def get_advanced_style(text):
    if not text or len(text.strip()) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    
    words = re.findall(r'\b\w+\b', text.lower())
    sentences = [s for s in re.split(r'[.!?\n]', text) if s.strip()]
    stop_words_count = len([w for w in words if w in ['the', 'is', 'at', 'which', 'on', 'and', 'a']])
    
    ttr = len(set(words)) / len(words) if words else 0
    avg_sent = len(words) / len(sentences) if sentences else 0
    punc_ratio = sum(text.count(p) for p in string.punctuation) / len(text) if len(text) > 0 else 0
    stop_ratio = stop_words_count / len(words) if words else 0
    
    return [ttr, avg_sent, punc_ratio, stop_ratio]

# 2. CARGA DE DATOS DE ENTRENAMIENTO
print("â�³ Cargando datos de entrenamiento...")
train_df = pd.read_csv(f"{DATA_PATH}/train.csv")
texts_train, styles_train, labels_train = [], [], []

for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    path = f"{DATA_PATH}/train/article_{row['id']:04d}/file_{row['real_text_id']}.txt"
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            t = f.read()
        texts_train.append(t)
        styles_train.append(get_advanced_style(t))
        labels_train.append(1 if row['real_text_id'] == 1 else 0)

# Procesamiento de Texto
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts_train)
X_seq = pad_sequences(tokenizer.texts_to_sequences(texts_train), maxlen=MAX_LEN)

# Escalador para Estilo
scaler = StandardScaler()
X_style = scaler.fit_transform(np.array(styles_train))
y = np.array(labels_train)

# 3. CONSTRUCCIÃ“N DEL MODELO BI-LSTM
def build_bilstm(style_dim):
    # Rama de Secuencia
    text_in = Input(shape=(MAX_LEN,))
    x = Embedding(MAX_WORDS, EMBEDDING_DIM)(text_in)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x) # Dropout alto para evitar memorizaciÃ³n (Overfitting)
    
    # Rama de Estilo
    style_in = Input(shape=(style_dim,))
    s = Dense(16, activation='relu')(style_in)
    
    # FusiÃ³n
    merged = Concatenate()([x, s])
    merged = Dense(64, activation='relu')(merged)
    merged = BatchNormalization()(merged)
    out = Dense(1, activation='sigmoid')(merged)
    
    model = tf.keras.Model(inputs=[text_in, style_in], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_bilstm(X_style.shape[1])

# 4. ENTRENAMIENTO CONTROLADO (Pocas Ã©pocas para no memorizar)
print("ğŸš€ Entrenando Modelo...")
model.fit([X_seq, X_style], y, epochs=5, batch_size=16, validation_split=0.1, verbose=1)

# 5. PREDICCIÃ“N COMPARATIVA (ELIMINA SESGOS)
print("â�³ Generando Predicciones para Test...")
test_dir = f"{DATA_PATH}/test"
results = []

for art_folder in tqdm(os.listdir(test_dir)):
    try:
        aid = int(art_folder.split('_')[1])
        
        def get_prob(fid):
            f_path = f"{test_dir}/{art_folder}/file_{fid}.txt"
            with open(f_path, 'r', encoding='utf-8') as f:
                content = f.read()
            s_vec = np.array([get_advanced_style(content)])
            s_scl = scaler.transform(s_vec)
            sq = pad_sequences(tokenizer.texts_to_sequences([content]), maxlen=MAX_LEN)
            return model.predict([sq, s_scl], verbose=0)[0][0]

        p1 = get_prob(1)
        p2 = get_prob(2)
        
        # El archivo con mayor probabilidad de ser humano (1) gana
        results.append({"id": aid, "real_text_id": 1 if p1 > p2 else 2})
    except:
        continue

# 6. GUARDADO FINAL
submission = pd.DataFrame(results).sort_values("id")
submission.to_csv("submission_V11_final.csv", index=False)
print("âœ… Â¡TODO LISTO! Descarga 'submission_V11_final.csv' y sÃºbelo.")

