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


import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. ESTILOMETR��A DE ALTA PRECISI��N ---
def get_pro_style(text):
    if not text or len(text) < 10: return [0]*6
    words = re.findall(r'\b\w+\b', text.lower())
    sentences = [s for s in re.split(r'[.!?\n]', text) if s.strip()]
    
    # Rasgos que las IAs NO pueden imitar bien:
    ttr = len(set(words)) / len(words) if words else 0
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    # Entrop铆a de longitud de frases (IA es muy constante, humanos variamos mucho)
    sent_len_std = np.std([len(s.split()) for s in sentences]) if sentences else 0
    # Uso de stop words espec铆ficas (IA abusa de conectores l贸gicos)
    ai_markers = len([w for w in words if w in ['moreover', 'consequently', 'furthermore', 'therefore']]) / len(words) if words else 0
    punc = sum(text.count(p) for p in '.,;:!?') / len(text)
    
    return [ttr, avg_word_len, sent_len_std, ai_markers, punc, len(words)]

# --- 2. CARGA Y PREPARACI��N ---
DATA_PATH = "/kaggle/input/fake-or-real-the-impostor-hunt/data"
train_df = pd.read_csv(f"{DATA_PATH}/train.csv")

texts, styles, labels = [], [], []
for _, row in train_df.iterrows():
    with open(f"{DATA_PATH}/train/article_{row['id']:04d}/file_{row['real_text_id']}.txt", 'r') as f:
        t = f.read()
    texts.append(t)
    styles.append(get_pro_style(t))
    labels.append(1 if row['real_text_id'] == 1 else 0)

# Tokenizaci贸n para la Red Neuronal
tok = Tokenizer(num_words=10000)
tok.fit_on_texts(texts)
X_seq = pad_sequences(tok.texts_to_sequences(texts), maxlen=300)
X_style = StandardScaler().fit_transform(styles)
y = np.array(labels)

# --- 3. ENTRENAMIENTO DE LOS "DOS JUECES" ---

# Juez 1: Bi-LSTM (Para el contexto)
input_seq = tf.keras.layers.Input(shape=(300,))
emb = tf.keras.layers.Embedding(10000, 100)(input_seq)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(emb)
x = tf.keras.layers.Dropout(0.5)(x)
out_nn = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model_nn = tf.keras.Model(input_seq, out_nn)
model_nn.compile(optimizer='adam', loss='binary_crossentropy')
model_nn.fit(X_seq, y, epochs=8, batch_size=16, verbose=0)

# Juez 2: Random Forest (Para las estad铆sticas puras)
model_rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model_rf.fit(X_style, y)

# --- 4. INFERENCIA DE ENSAMBLE ---
print("���� Ejecutando Ensamble de votaci贸n...")
test_dir = f"{DATA_PATH}/test"
results = []
scaler = StandardScaler().fit(styles)

for art in tqdm(os.listdir(test_dir)):
    aid = int(art.split('_')[1])
    def get_score(fid):
        with open(f"{test_dir}/{art}/file_{fid}.txt", 'r') as f:
            t = f.read()
        # Predicci贸n NN
        seq = pad_sequences(tok.texts_to_sequences([t]), maxlen=300)
        p_nn = model_nn.predict(seq, verbose=0)[0][0]
        # Predicci贸n RF
        st = scaler.transform([get_pro_style(t)])
        p_rf = model_rf.predict_proba(st)[0][1]
        # Promedio ponderado (Damos un poco m谩s de peso al estilo en sets peque帽os)
        return (p_nn * 0.4) + (p_rf * 0.6)

    results.append({"id": aid, "real_text_id": 1 if get_score(1) > get_score(2) else 2})

pd.DataFrame(results).to_csv("submission_V12.csv", index=False)

