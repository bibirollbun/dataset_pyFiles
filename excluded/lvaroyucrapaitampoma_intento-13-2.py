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
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_nlp
from tqdm import tqdm

# --- 1. CONFIGURACIÃ“N Y PARCHES ---
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
# Forzamos a que las etiquetas tengan la dimensiÃ³n correcta para BERT
DATA_PATH = "/kaggle/input/fake-or-real-the-impostor-hunt/data"

# --- 2. CARGA DE DATOS ---
train_df = pd.read_csv(f"{DATA_PATH}/train.csv")
texts, labels = [], []

for _, row in train_df.iterrows():
    path = f"{DATA_PATH}/train/article_{row['id']:04d}/file_{row['real_text_id']}.txt"
    with open(path, 'r', encoding='utf-8') as f:
        texts.append(f.read())
    # IMPORTANTE: Para BERT con 2 clases, las etiquetas deben ser enteros
    labels.append(1 if row['real_text_id'] == 1 else 0)

X = np.array(texts)
y = np.array(labels) # Forma (95,)

# --- 3. MODELO BERT (DistilBERT) ---
# Usamos preprocessor automÃ¡tico de keras_nlp
classifier = keras_nlp.models.DistilBertClassifier.from_preset(
    "distil_bert_base_en_uncased",
    num_classes=2, # Esto genera salida (None, 2)
)

# --- 4. SOLUCIÃ“N AL VALUE ERROR ---
# Usamos SparseCategoricalCrossentropy para que acepte etiquetas [0, 1, 0...] 
# contra una salida de modelo de [prob_clase0, prob_clase1]
classifier.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    metrics=["accuracy"]
)

print("ğŸš€ Entrenando BERT...")
# Entrenamos con pocas Ã©pocas porque el dataset es muy pequeÃ±o
classifier.fit(X, y, batch_size=4, epochs=4)

# --- 5. INFERENCIA COMPARATIVA BERT ---
print("â�³ Generando Predicciones Comparativas...")
test_dir = f"{DATA_PATH}/test"
results = []

for art_folder in tqdm(os.listdir(test_dir)):
    try:
        aid = int(art_folder.split('_')[1])
        
        def get_bert_prob(fid):
            with open(f"{test_dir}/{art_folder}/file_{fid}.txt", 'r') as f:
                t = [f.read()]
            # predict devuelve [[prob_c0, prob_c1]] -> tomamos prob_c1 (humanidad)
            preds = classifier.predict(np.array(t), verbose=0)
            return preds[0][1] 

        p1 = get_bert_prob(1)
        p2 = get_bert_prob(2)
        
        results.append({"id": aid, "real_text_id": 1 if p1 > p2 else 2})
    except:
        continue

# Guardar
submission = pd.DataFrame(results).sort_values("id")
submission.to_csv("submission_BERT_final.csv", index=False)
print("âœ… Â¡Finalizado con BERT!")

