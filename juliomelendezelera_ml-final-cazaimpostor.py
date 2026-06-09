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


import pandas as pd
import os

# --- PASO 1: DEFINICIÃ“N DE RUTAS ---
# Rutas basadas en la estructura estÃ¡ndar de esta competencia
ruta_base_data = '/kaggle/input/fake-or-real-the-impostor-hunt/data/'
ruta_train_csv = '/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv'

# --- PASO 2: CARGA DE LA HOJA DE RESPUESTAS ---
print("ğŸ“‚ Cargando hoja de respuestas (train.csv)...")
try:
    df_indices = pd.read_csv(ruta_train_csv)
    print(f"âœ… Hoja de respuestas cargada. Contiene {len(df_indices)} casos.")
except FileNotFoundError:
    print("â�Œ ERROR: No encontrÃ© el archivo train.csv. Verifica las rutas.")

# --- PASO 3: LECTURA DE TEXTOS Y ETIQUETADO ---
print("\nğŸš€ Iniciando la lectura de los archivos de texto... (Paciencia, esto toma unos segundos)")

datos_procesados = []

# Recorremos cada fila de la hoja de respuestas
for index, fila in df_indices.iterrows():
    try:
        # 1. Identificar carpeta y archivos
        id_caso = fila['id']
        id_real = fila['real_text_id'] # Dice si el real es el 1 o el 2
        
        # El nombre de la carpeta tiene ceros a la izquierda (ej: article_0026)
        nombre_carpeta = f"article_{str(id_caso).zfill(4)}"
        ruta_carpeta = os.path.join(ruta_base_data, 'train', nombre_carpeta)
        
        # 2. Leer los dos archivos de esa carpeta
        path_file_1 = os.path.join(ruta_carpeta, 'file_1.txt')
        path_file_2 = os.path.join(ruta_carpeta, 'file_2.txt')
        
        with open(path_file_1, 'r', encoding='utf-8', errors='ignore') as f:
            texto_1 = f.read()
        
        with open(path_file_2, 'r', encoding='utf-8', errors='ignore') as f:
            texto_2 = f.read()
            
        # 3. Asignar etiquetas (1 = Real, 0 = Fake)
        if id_real == 1:
            datos_procesados.append({'texto': texto_1, 'etiqueta': 1}) # File 1 es Real
            datos_procesados.append({'texto': texto_2, 'etiqueta': 0}) # File 2 es Fake
        else: # id_real == 2
            datos_procesados.append({'texto': texto_1, 'etiqueta': 0}) # File 1 es Fake
            datos_procesados.append({'texto': texto_2, 'etiqueta': 1}) # File 2 es Real
            
    except Exception as e:
        print(f"âš ï¸� Error en el caso {id_caso}: {e}")

# --- PASO 4: CREACIÃ“N DEL DATAFRAME FINAL ---
df_train = pd.DataFrame(datos_procesados)

print("\nâœ¨ Â¡Proceso completado!")
print(f"ğŸ“Š Total de noticias procesadas: {len(df_train)}")
print("--- Muestra de los datos ---")
display(df_train.head())

# VerificaciÃ³n de balance (DeberÃ­a ser 50% y 50%)
print("\n--- Balance de Clases ---")
print(df_train['etiqueta'].value_counts())


import re
import string

# Definimos nuestra funciÃ³n "Lavadora de Texto"
def limpiar_texto(texto):
    # 1. Convertir todo a minÃºsculas
    texto = str(texto).lower()
    
    # 2. Eliminar corchetes, urls o caracteres especiales usando Expresiones Regulares (Regex)
    texto = re.sub(r'\[.*?\]', '', texto) # Quita cosas entre corchetes
    texto = re.sub(r'https?://\S+|www\.\S+', '', texto) # Quita URLs web
    texto = re.sub(r'<.*?>+', '', texto) # Quita cÃ³digo HTML si lo hubiera
    
    # 3. Eliminar puntuaciÃ³n (Signos como ! ? , .)
    # Esta lÃ­nea mÃ¡gica dice: "Reemplaza cualquier signo de puntuaciÃ³n por un espacio"
    texto = re.sub(f'[{re.escape(string.punctuation)}]', '', texto)
    
    # 4. Eliminar saltos de lÃ­nea y espacios extra
    texto = re.sub(r'\n', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

print("ğŸ§¹ Iniciando limpieza de las 190 noticias...")

# Aplicamos la funciÃ³n a la columna 'texto' y creamos una nueva llamada 'texto_limpio'
df_train['texto_limpio'] = df_train['texto'].apply(limpiar_texto)

print("âœ… Â¡Limpieza terminada!")

# Vamos a ver la diferencia
print("\n--- Comparativa Antes vs DespuÃ©s ---")
display(df_train[['texto', 'texto_limpio']].head())


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("ğŸ§® Dividiendo datos en Entrenamiento y Examen...")

# 1. Separamos los datos (X = Texto, y = Etiqueta)
X = df_train['texto_limpio']
y = df_train['etiqueta']

# Usamos el 20% para testear (test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"   -> Entrenamiento: {len(X_train)} noticias")
print(f"   -> ValidaciÃ³n: {len(X_val)} noticias")

# 2. VectorizaciÃ³n (Convertir palabras a nÃºmeros)
print("\nğŸ”¢ Convirtiendo texto a vectores TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000) # Usamos las 5000 palabras mÃ¡s comunes

# "Fit" aprende las palabras del entrenamiento. "Transform" las convierte.
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val) # OJO: AquÃ­ solo transformamos, no aprendemos de nuevo

# 3. Entrenar el Modelo
print("\nğŸ¤– Entrenando RegresiÃ³n LogÃ­stica...")
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 4. Evaluar
print("ğŸ“� Evaluando el modelo...")
y_pred = model.predict(X_val_vec)

acc = accuracy_score(y_val, y_pred)
print(f"\nğŸ�† PRECISIÃ“N (Accuracy): {acc:.4f}")
print("\n--- Reporte Detallado ---")
print(classification_report(y_val, y_pred, target_names=['Fake', 'Real']))


# --- PASO 4: GENERACIÃ“N DE PREDICCIONES (TEST) ---
print("ğŸš€ Iniciando procesamiento de datos de PRUEBA (Test)...")

ruta_test = os.path.join(ruta_base_data, 'test')
ids_test = []
preds_test = []

# Listamos todas las carpetas en 'test' (ej: article_0026, article_0192...)
carpetas_test = sorted(os.listdir(ruta_test))

print(f"ğŸ“‚ Se encontraron {len(carpetas_test)} casos para predecir.")

count = 0
for carpeta in carpetas_test:
    # Extraemos el ID numÃ©rico del nombre de la carpeta 'article_0192' -> 192
    try:
        id_caso = int(carpeta.split('_')[1])
    except:
        continue # Si hay algÃºn archivo raro, lo saltamos
        
    ruta_carpeta_actual = os.path.join(ruta_test, carpeta)
    
    # Leemos los dos archivos
    path_f1 = os.path.join(ruta_carpeta_actual, 'file_1.txt')
    path_f2 = os.path.join(ruta_carpeta_actual, 'file_2.txt')
    
    try:
        with open(path_f1, 'r', encoding='utf-8', errors='ignore') as f: txt1 = f.read()
        with open(path_f2, 'r', encoding='utf-8', errors='ignore') as f: txt2 = f.read()
        
        # 1. Limpiamos (Usando la MISMA funciÃ³n que definimos antes)
        txt1_clean = limpiar_texto(txt1)
        txt2_clean = limpiar_texto(txt2)
        
        # 2. Vectorizamos (Usando el MISMO vectorizer ya entrenado)
        # OJO: Usamos .transform(), JAMÃ�S .fit_transform() aquÃ­
        vecs = vectorizer.transform([txt1_clean, txt2_clean])
        
        # 3. Predecimos la PROBABILIDAD de ser "Real" (clase 1)
        # model.predict_proba devuelve [[prob_Fake, prob_Real]]
        probs = model.predict_proba(vecs)
        prob_real_f1 = probs[0][1] # Probabilidad de que file_1 sea Real
        prob_real_f2 = probs[1][1] # Probabilidad de que file_2 sea Real
        
        # 4. DecisiÃ³n: Â¿CuÃ¡l es el verdadero?
        # Elegimos el que tenga MAYOR probabilidad de ser Real
        if prob_real_f1 > prob_real_f2:
            prediccion = 1 # El file_1 es el real
        else:
            prediccion = 2 # El file_2 es el real
            
        ids_test.append(id_caso)
        preds_test.append(prediccion)
        
        count += 1
        if count % 50 == 0: print(f"   ... procesados {count} casos")
            
    except Exception as e:
        print(f"âš ï¸� Error en {carpeta}: {e}")

# --- CREAR ARCHIVO DE ENVÃ�O ---
df_submission = pd.DataFrame({
    'id': ids_test,
    'real_text_id': preds_test
})

# Guardamos el CSV
nombre_archivo = 'submission_modelo_base.csv'
df_submission.to_csv(nombre_archivo, index=False)

print(f"\nâœ… Â¡Archivo '{nombre_archivo}' generado con Ã©xito!")
print(f"ğŸ“� Contiene {len(df_submission)} predicciones.")
print("--- Primeras filas del envÃ­o ---")
display(df_submission.head())


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

print("âš™ï¸� Configurando parÃ¡metros de la Red Neuronal...")

# HIPERPARÃ�METROS (Ajustes de la mÃ¡quina)
MAX_PALABRAS = 10000       # Solo recordaremos las 10,000 palabras mÃ¡s comunes
LONGITUD_MAXIMA = 300      # Cortaremos las noticias a 300 palabras (o rellenaremos)

# 1. Crear el Tokenizer (El diccionario de la mÃ¡quina)
tokenizer = Tokenizer(num_words=MAX_PALABRAS, oov_token="<OOV>")
# oov_token significa "Out Of Vocabulary". Si llega una palabra rara, la reemplaza por esto.

# 2. Entrenar el diccionario con nuestros textos
tokenizer.fit_on_texts(df_train['texto_limpio'])
word_index = tokenizer.word_index
print(f"ğŸ“š El diccionario aprendiÃ³ {len(word_index)} palabras Ãºnicas.")

# 3. Convertir TEXTO a SECUENCIAS numÃ©ricas
# Ejemplo: "El gato come" -> [1, 45, 12]
secuencias_train = tokenizer.texts_to_sequences(df_train['texto_limpio'])

# 4. PADDING (Relleno)
# Ejemplo: [1, 45, 12] -> [1, 45, 12, 0, 0, 0...] hasta llegar a 300
X_train_padded = pad_sequences(secuencias_train, maxlen=LONGITUD_MAXIMA, padding='post', truncating='post')

# Convertimos las etiquetas a formato numpy (necesario para TensorFlow)
y_train_np = np.array(df_train['etiqueta'])

print("\nâœ… Datos listos para la Red Neuronal.")
print(f"Dimensiones de entrada: {X_train_padded.shape}")
print("\n--- Ejemplo de transformaciÃ³n ---")
print(f"Texto original: {df_train['texto_limpio'][0][:50]}...")
print(f"Secuencia numÃ©rica: {secuencias_train[0][:10]}...")


# --- DEFINICIÃ“N DE LA RED NEURONAL ---
print("ğŸ�—ï¸� Construyendo el cerebro de la IA...")

model = tf.keras.Sequential([
    # 1. Capa de Embedding: Transforma nÃºmeros en "conceptos" (vectores)
    tf.keras.layers.Embedding(input_dim=MAX_PALABRAS, output_dim=64, input_length=LONGITUD_MAXIMA),
    
    # 2. Capa LSTM Bidireccional: Lee el texto en ambas direcciones
    # Dropout ayuda a que el modelo no memorice (overfitting), sino que aprenda
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
    
    # 3. Capas Densas: Para procesar lo que entendiÃ³ la LSTM
    tf.keras.layers.Dropout(0.5), # Apaga el 50% de neuronas al azar para obligarlo a aprender rutas alternas
    tf.keras.layers.Dense(32, activation='relu'),
    
    # 4. Capa de Salida: Una sola neurona (0 = Fake, 1 = Real)
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilamos el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# --- ENTRENAMIENTO ---
print("\nğŸ�‹ï¸� Iniciando entrenamiento intensivo...")
# Epochs = CuÃ¡ntas veces repasarÃ¡ todos los datos (5 veces suele bastar para empezar)
history = model.fit(
    X_train_padded, y_train_np, 
    epochs=10,  # Le damos 10 vueltas al estudio
    validation_split=0.2, # Usamos 20% para validar en cada vuelta
    verbose=1
)

print("âœ… Â¡Entrenamiento finalizado!")


# --- GENERACIÃ“N DE PREDICCIONES CON LSTM (ENVÃ�O #2) ---
print("ğŸš€ Generando predicciones con la Red Neuronal...")

ids_test = []
preds_test = []

ruta_test = os.path.join(ruta_base_data, 'test')
carpetas_test = sorted(os.listdir(ruta_test))

# Reutilizamos el tokenizer que ya entrenamos
for carpeta in carpetas_test:
    try:
        id_caso = int(carpeta.split('_')[1])
        ruta_carpeta = os.path.join(ruta_test, carpeta)
        
        # Leer archivos
        with open(os.path.join(ruta_carpeta, 'file_1.txt'), 'r', errors='ignore') as f: t1 = limpiar_texto(f.read())
        with open(os.path.join(ruta_carpeta, 'file_2.txt'), 'r', errors='ignore') as f: t2 = limpiar_texto(f.read())
        
        # 1. Convertir a secuencias (nÃºmeros)
        seqs = tokenizer.texts_to_sequences([t1, t2])
        
        # 2. Padding (Rellenar con ceros hasta 300)
        seqs_padded = pad_sequences(seqs, maxlen=LONGITUD_MAXIMA, padding='post', truncating='post')
        
        # 3. Predecir
        predicciones = model.predict(seqs_padded, verbose=0)
        prob_f1 = predicciones[0][0]
        prob_f2 = predicciones[1][0]
        
        # 4. Decidir quiÃ©n gana (quiÃ©n tiene mÃ¡s probabilidad de ser Real/1)
        if prob_f1 > prob_f2:
            ganador = 1
        else:
            ganador = 2
            
        ids_test.append(id_caso)
        preds_test.append(ganador)
        
    except:
        continue

# Guardar
df_sub_lstm = pd.DataFrame({'id': ids_test, 'real_text_id': preds_test})
df_sub_lstm.to_csv('submission_lstm_v2.csv', index=False)
print("âœ… Archivo 'submission_lstm_v2.csv' generado.")


# --- INSTALACIÃ“N Y CARGA DE BERT ---
# Kaggle suele tener transformers instalado, pero por si acaso silenciamos la instalaciÃ³n
!pip install -q transformers

import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split

print("ğŸ¤– Cargando el cerebro de DistilBERT (esto puede tardar un poco)...")

# 1. Cargamos el Tokenizer (El traductor de BERT)
# Usamos 'distilbert-base-uncased' (versiÃ³n ligera y en minÃºsculas)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# 2. Cargamos el Modelo Pre-entrenado
model_bert = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1)

print("âœ… Cerebro cargado exitosamente.")


# --- PREPARACIÃ“N DE DATOS PARA BERT ---
def convertir_para_bert(textos, tokenizer, max_len=300):
    input_ids = []
    attention_masks = []
    
    for texto in textos:
        encoded = tokenizer.encode_plus(
            texto,
            add_special_tokens=True, # AÃ±ade tokens de inicio [CLS] y fin [SEP]
            max_length=max_len,
            padding='max_length',    # Rellena hasta 300
            truncation=True,         # Corta si se pasa de 300
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(encoded['input_ids'][0])
        attention_masks.append(encoded['attention_mask'][0])
        
    return np.array(input_ids), np.array(attention_masks)

print("âš™ï¸� Traduciendo noticias al idioma de BERT...")

# Preparamos los datos de entrenamiento
X_ids, X_masks = convertir_para_bert(df_train['texto_limpio'], tokenizer)
y_labels = np.array(df_train['etiqueta'])

# Dividimos en Train y ValidaciÃ³n (80/20)
X_train_ids, X_val_ids, X_train_masks, X_val_masks, y_train, y_val = train_test_split(
    X_ids, X_masks, y_labels, test_size=0.2, random_state=42
)

print(f"âœ… Datos listos. Entrenamiento: {len(X_train_ids)} | ValidaciÃ³n: {len(X_val_ids)}")


import tensorflow as tf
from transformers import TFDistilBertModel
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda
from tensorflow.keras.models import Model

print("ğŸ�—ï¸� Construyendo modelo HÃ­brido (Keras + BERT) con Parche de Compatibilidad...")

# 1. Cargamos el BERT "Base"
bert_base = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Opcional: Congelamos BERT para que no se "rompa" al principio (recomendado)
bert_base.trainable = False 

# 2. Definimos las Entradas
input_ids = Input(shape=(300,), dtype=tf.int32, name='input_ids')
input_masks = Input(shape=(300,), dtype=tf.int32, name='attention_mask')

# --- PARCHE DE COMPATIBILIDAD (LAMBDA) ---
# Creamos una funciÃ³n que envuelve a BERT para que Keras 3 no se queje
def envoltorio_bert(args):
    ids, masks = args
    # BERT devuelve una tupla, tomamos el primer elemento (last_hidden_state)
    return bert_base(ids, attention_mask=masks)[0]

# Usamos Lambda para aplicar esa funciÃ³n de forma segura
bert_output = Lambda(envoltorio_bert, output_shape=(300, 768))([input_ids, input_masks])
# -----------------------------------------

# 4. Extraemos el concepto global (Token [CLS], Ã­ndice 0)
cls_token = bert_output[:, 0, :]

# 5. Capas de ClasificaciÃ³n
x = Dense(64, activation='relu')(cls_token)
x = Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)

# 6. Ensamblamos el Modelo
model_final = Model(inputs=[input_ids, input_masks], outputs=output)

# 7. Compilamos
# Usamos un learning rate un poco mayor porque congelamos BERT
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3) 
model_final.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

print("âœ… Modelo construido y compilado (Â¡Ahora sÃ­!).")
model_final.summary()

# --- ENTRENAMIENTO ---
print("\nğŸš€ Iniciando entrenamiento final...")
history_bert = model_final.fit(
    x=[X_train_ids, X_train_masks],
    y=y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=5, # Subimos a 5 Ã©pocas ya que BERT estÃ¡ congelado
    batch_size=16, # Subimos un poco el batch size para ir mÃ¡s rÃ¡pido
    verbose=1
)


# --- GENERACIÃ“N DE PREDICCIONES CON BERT (ENVÃ�O #3) ---
print("ğŸš€ Generando predicciones con BERT HÃ­brido...")

ids_test = []
preds_test = []

ruta_test = os.path.join(ruta_base_data, 'test')
carpetas_test = sorted(os.listdir(ruta_test))

# Contador para ver el progreso
contador = 0

for carpeta in carpetas_test:
    try:
        # Extraer ID
        id_caso = int(carpeta.split('_')[1])
        ruta_carpeta = os.path.join(ruta_test, carpeta)
        
        # Leer textos
        with open(os.path.join(ruta_carpeta, 'file_1.txt'), 'r', errors='ignore') as f: t1 = limpiar_texto(f.read())
        with open(os.path.join(ruta_carpeta, 'file_2.txt'), 'r', errors='ignore') as f: t2 = limpiar_texto(f.read())
        
        # --- PREPARAR PARA BERT (TOKENIZER) ---
        # FunciÃ³n auxiliar rÃ¡pida para tokenizar
        def preparar(texto):
            enc = tokenizer.encode_plus(
                texto,
                max_length=300,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='tf'
            )
            return enc['input_ids'], enc['attention_mask']

        id1, mask1 = preparar(t1)
        id2, mask2 = preparar(t2)
        
        # --- PREDECIR ---
        # El modelo espera una lista: [input_ids, attention_mask]
        pred1 = model_final.predict([id1, mask1], verbose=0)[0][0]
        pred2 = model_final.predict([id2, mask2], verbose=0)[0][0]
        
        # --- DECISIÃ“N ---
        # Quien tenga mayor probabilidad de ser 1 (Real) gana
        if pred1 > pred2:
            ganador = 1
        else:
            ganador = 2
            
        ids_test.append(id_caso)
        preds_test.append(ganador)
        
        contador += 1
        if contador % 100 == 0: print(f"   ... {contador} noticias procesadas")
            
    except Exception as e:
        print(f"Error en {carpeta}: {e}")
        continue

# Guardar CSV
df_sub_bert = pd.DataFrame({'id': ids_test, 'real_text_id': preds_test})
df_sub_bert.to_csv('submission_bert_v3.csv', index=False)
print("\nâœ… Â¡Archivo 'submission_bert_v3.csv' generado exitosamente!")


# --- FASE 4: FINE-TUNING TOTAL (DESCONGELAR BERT) ---
print("ğŸ”“ Descongelando el cerebro de BERT para ajuste fino...")

# 1. Descongelamos el modelo base
# Esto permite que BERT modifique sus neuronas internas para adaptarse a TUS noticias
bert_base.trainable = True

# 2. Re-compilamos el modelo completo
# OJO: Usamos una tasa de aprendizaje MUY BAJA (1e-5)
# Es vital que sea baja. Si ponemos una alta, el modelo "olvidarÃ¡" el inglÃ©s.
optimizer_fine = tf.keras.optimizers.Adam(learning_rate=1e-5)

model_final.compile(optimizer=optimizer_fine, loss='binary_crossentropy', metrics=['accuracy'])

# Mostramos cuÃ¡ntos parÃ¡metros se van a entrenar (Â¡SerÃ¡n millones!)
model_final.summary()

print("\nğŸš€ Iniciando entrenamiento delicado (Fine-Tuning)...")

# 3. Entrenamos de nuevo
# Solo 2 Ã©pocas. Descongelado aprende rapidÃ­simo y no queremos overfitting.
history_fine = model_final.fit(
    x=[X_train_ids, X_train_masks],
    y=y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=2, 
    batch_size=8, # Mantenemos batch pequeÃ±o para cuidar la memoria GPU
    verbose=1
)


# --- GENERACIÃ“N DE PREDICCIONES FINALES ---
print("ğŸ�� Generando el envÃ­o final...")

ids_test = []
preds_test = []

ruta_test = os.path.join(ruta_base_data, 'test')
carpetas_test = sorted(os.listdir(ruta_test))

for carpeta in carpetas_test:
    try:
        # ID y Rutas
        id_caso = int(carpeta.split('_')[1])
        ruta_carpeta = os.path.join(ruta_test, carpeta)
        
        # Lectura
        with open(os.path.join(ruta_carpeta, 'file_1.txt'), 'r', errors='ignore') as f: t1 = limpiar_texto(f.read())
        with open(os.path.join(ruta_carpeta, 'file_2.txt'), 'r', errors='ignore') as f: t2 = limpiar_texto(f.read())
        
        # Tokenizer (FunciÃ³n auxiliar)
        def preparar(texto):
            enc = tokenizer.encode_plus(
                texto, max_length=300, padding='max_length', truncation=True,
                return_attention_mask=True, return_tensors='tf'
            )
            return enc['input_ids'], enc['attention_mask']

        id1, mask1 = preparar(t1)
        id2, mask2 = preparar(t2)
        
        # PredicciÃ³n
        pred1 = model_final.predict([id1, mask1], verbose=0)[0][0]
        pred2 = model_final.predict([id2, mask2], verbose=0)[0][0]
        
        # Ganador
        ganador = 1 if pred1 > pred2 else 2
            
        ids_test.append(id_caso)
        preds_test.append(ganador)
        
    except Exception as e:
        continue

# Guardar CSV Final
df_sub_final = pd.DataFrame({'id': ids_test, 'real_text_id': preds_test})
df_sub_final.to_csv('submission_bert_final_v4.csv', index=False)
print("âœ… Â¡MisiÃ³n Cumplida! Archivo final generado.")


# --- CONFIGURACIÃ“N VERSIÃ“N 5 ---
VERSION = "v5"

# 1. CAMBIO CLAVE: Subimos un poco la velocidad de aprendizaje
# Antes usamos 1e-5. Ahora probamos 2e-5.
NUEVO_LEARNING_RATE = 2e-5 

# Mantenemos las Ã©pocas en 2 para comparar "peras con peras" (solo cambiamos LR)
NUEVAS_EPOCAS = 2

print(f"ğŸ§ª Iniciando Experimento {VERSION}")
print(f"âš™ï¸� Ajuste: Learning Rate aumentado a {NUEVO_LEARNING_RATE}")

# 1. Limpiamos la sesiÃ³n anterior para empezar de cero
tf.keras.backend.clear_session()

# 2. Re-construimos el modelo (Igual que antes)
bert_base = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
bert_base.trainable = True # Descongelado

input_ids = Input(shape=(300,), dtype=tf.int32, name='input_ids')
input_masks = Input(shape=(300,), dtype=tf.int32, name='attention_mask')

def envoltorio_bert(args):
    ids, masks = args
    return bert_base(ids, attention_mask=masks)[0]

bert_output = Lambda(envoltorio_bert, output_shape=(300, 768))([input_ids, input_masks])
cls_token = bert_output[:, 0, :]
x = Dense(64, activation='relu')(cls_token)
x = Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)

model_v5 = Model(inputs=[input_ids, input_masks], outputs=output)

# 3. Compilamos con el NUEVO Learning Rate
optimizer = tf.keras.optimizers.Adam(learning_rate=NUEVO_LEARNING_RATE)
model_v5.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 4. Entrenamos
history_v5 = model_v5.fit(
    x=[X_train_ids, X_train_masks],
    y=y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=NUEVAS_EPOCAS,
    batch_size=8,
    verbose=1
)


# --- GENERAR CSV PARA EL EXPERIMENTO V5 ---
print("ğŸ’¾ Generando archivo de envÃ­o para v5 (El experimento fallido)...")

ids_test = []
preds_test = []
ruta_test = os.path.join(ruta_base_data, 'test')
carpetas_test = sorted(os.listdir(ruta_test))

# FunciÃ³n auxiliar (necesaria para asegurar que usa el tokenizer global)
def preparar(texto):
    enc = tokenizer.encode_plus(
        texto, max_length=300, padding='max_length', truncation=True,
        return_attention_mask=True, return_tensors='tf'
    )
    return enc['input_ids'], enc['attention_mask']

for carpeta in carpetas_test:
    try:
        id_caso = int(carpeta.split('_')[1])
        ruta_carpeta = os.path.join(ruta_test, carpeta)
        with open(os.path.join(ruta_carpeta, 'file_1.txt'), 'r', errors='ignore') as f: t1 = limpiar_texto(f.read())
        with open(os.path.join(ruta_carpeta, 'file_2.txt'), 'r', errors='ignore') as f: t2 = limpiar_texto(f.read())
        
        id1, mask1 = preparar(t1)
        id2, mask2 = preparar(t2)
        
        pred1 = model_v5.predict([id1, mask1], verbose=0)[0][0]
        pred2 = model_v5.predict([id2, mask2], verbose=0)[0][0]
        
        ids_test.append(id_caso)
        preds_test.append(1 if pred1 > pred2 else 2)
            
    except:
        continue

df_sub = pd.DataFrame({'id': ids_test, 'real_text_id': preds_test})
df_sub.to_csv('submission_bert_v5.csv', index=False)
print("âœ… Archivo 'submission_bert_v5.csv' listo. Â¡SÃºbelo para registrar el experimento!")


# --- CONFIGURACIÃ“N MANUAL VERSIÃ“N 6 ---
VERSION = "v6"

# 1. CAMBIO CLAVE: Modo "Microcirujano" (Muy lento y preciso)
NUEVO_LEARNING_RATE = 5e-6  # 0.000005
NUEVAS_EPOCAS = 4           # MÃ¡s tiempo para aprender despacio

print(f"ğŸ§ª Iniciando Experimento {VERSION}")
print(f"âš™ï¸� Ajuste: LR reducido a {NUEVO_LEARNING_RATE} | Ã‰pocas aumentadas a {NUEVAS_EPOCAS}")

# Limpieza
tf.keras.backend.clear_session()

# ReconstrucciÃ³n (BERT Descongelado)
bert_base = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
bert_base.trainable = True 

input_ids = Input(shape=(300,), dtype=tf.int32, name='input_ids')
input_masks = Input(shape=(300,), dtype=tf.int32, name='attention_mask')

def envoltorio_bert(args):
    ids, masks = args
    return bert_base(ids, attention_mask=masks)[0]

bert_output = Lambda(envoltorio_bert, output_shape=(300, 768))([input_ids, input_masks])
cls_token = bert_output[:, 0, :]
x = Dense(64, activation='relu')(cls_token)
x = Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)

model_v6 = Model(inputs=[input_ids, input_masks], outputs=output)

# CompilaciÃ³n
optimizer = tf.keras.optimizers.Adam(learning_rate=NUEVO_LEARNING_RATE)
model_v6.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento
history_v6 = model_v6.fit(
    x=[X_train_ids, X_train_masks],
    y=y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=NUEVAS_EPOCAS,
    batch_size=8,
    verbose=1
)


# --- 1. GENERAR CSV DEL V6 (EL LENTO) ---
print("ğŸ’¾ Guardando el intento fallido v6...")
# (AquÃ­ usamos el cÃ³digo de generaciÃ³n rÃ¡pido que ya tienes, pero resumido)
ids_test, preds_test = [], []
ruta_test = os.path.join(ruta_base_data, 'test')
carpetas_test = sorted(os.listdir(ruta_test))

def preparar(texto):
    enc = tokenizer.encode_plus(texto, max_length=300, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='tf')
    return enc['input_ids'], enc['attention_mask']

for carpeta in carpetas_test:
    try:
        id_caso = int(carpeta.split('_')[1])
        ruta_carpeta = os.path.join(ruta_test, carpeta)
        with open(os.path.join(ruta_carpeta, 'file_1.txt'), 'r', errors='ignore') as f: t1 = limpiar_texto(f.read())
        with open(os.path.join(ruta_carpeta, 'file_2.txt'), 'r', errors='ignore') as f: t2 = limpiar_texto(f.read())
        id1, mask1 = preparar(t1); id2, mask2 = preparar(t2)
        pred1 = model_v6.predict([id1, mask1], verbose=0)[0][0]
        pred2 = model_v6.predict([id2, mask2], verbose=0)[0][0]
        ids_test.append(id_caso); preds_test.append(1 if pred1 > pred2 else 2)
    except: continue
pd.DataFrame({'id': ids_test, 'real_text_id': preds_test}).to_csv('submission_bert_v6.csv', index=False)
print("âœ… v6 Guardado.")

# --- 2. ENTRENAR V7 (EL PUNTO DULCE) ---
VERSION = "v7"
NUEVO_LEARNING_RATE = 1e-5  # Volvemos al valor ganador
NUEVAS_EPOCAS = 3           # Pero probamos con 3 vueltas en vez de 2

print(f"\nğŸ§ª Iniciando Experimento {VERSION} (Intento de RÃ©cord)")
tf.keras.backend.clear_session()

bert_base = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
bert_base.trainable = True 

input_ids = Input(shape=(300,), dtype=tf.int32, name='input_ids')
input_masks = Input(shape=(300,), dtype=tf.int32, name='attention_mask')
def envoltorio_bert(args): return bert_base(args[0], attention_mask=args[1])[0]
bert_output = Lambda(envoltorio_bert, output_shape=(300, 768))([input_ids, input_masks])
x = Dense(64, activation='relu')(bert_output[:, 0, :])
x = Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)

model_v7 = Model(inputs=[input_ids, input_masks], outputs=output)
optimizer = tf.keras.optimizers.Adam(learning_rate=NUEVO_LEARNING_RATE)
model_v7.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

history_v7 = model_v7.fit(
    x=[X_train_ids, X_train_masks], y=y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=NUEVAS_EPOCAS, batch_size=8, verbose=1
)


# --- 1. GUARDAR EL FALLO V7 ---
print("ğŸ’¾ Guardando v7 (El modelo colapsado)...")
# (CÃ³digo resumido de guardado)
ids_test, preds_test = [], []
ruta_test = os.path.join(ruta_base_data, 'test')
carpetas_test = sorted(os.listdir(ruta_test))
def preparar(texto):
    enc = tokenizer.encode_plus(texto, max_length=300, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='tf')
    return enc['input_ids'], enc['attention_mask']

for carpeta in carpetas_test:
    try:
        id_caso = int(carpeta.split('_')[1])
        ruta_carpeta = os.path.join(ruta_test, carpeta)
        with open(os.path.join(ruta_carpeta, 'file_1.txt'), 'r', errors='ignore') as f: t1 = limpiar_texto(f.read())
        with open(os.path.join(ruta_carpeta, 'file_2.txt'), 'r', errors='ignore') as f: t2 = limpiar_texto(f.read())
        id1, mask1 = preparar(t1); id2, mask2 = preparar(t2)
        pred1 = model_v7.predict([id1, mask1], verbose=0)[0][0]
        pred2 = model_v7.predict([id2, mask2], verbose=0)[0][0]
        ids_test.append(id_caso); preds_test.append(1 if pred1 > pred2 else 2)
    except: continue
pd.DataFrame({'id': ids_test, 'real_text_id': preds_test}).to_csv('submission_bert_v7.csv', index=False)
print("âœ… v7 Guardado.")

# --- 2. INICIAR V8 (ESTRATEGIA DOS PASOS) ---
VERSION = "v8"
print(f"\nğŸ§ª Iniciando Experimento {VERSION} (Estrategia: Congelar -> Descongelar)")

tf.keras.backend.clear_session()

# PASO 1: CARGAR CONGELADO
bert_base = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
bert_base.trainable = False # ğŸ”’ CANDADO PUESTO

input_ids = Input(shape=(300,), dtype=tf.int32, name='input_ids')
input_masks = Input(shape=(300,), dtype=tf.int32, name='attention_mask')
def envoltorio_bert(args): return bert_base(args[0], attention_mask=args[1])[0]
bert_output = Lambda(envoltorio_bert, output_shape=(300, 768))([input_ids, input_masks])
x = Dense(64, activation='relu')(bert_output[:, 0, :])
x = Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)

model_v8 = Model(inputs=[input_ids, input_masks], outputs=output)

print("ğŸ§Š FASE 1: Entrenamiento con BERT Congelado (Calentamiento)...")
model_v8.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_v8.fit(
    x=[X_train_ids, X_train_masks], y=y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=2, batch_size=8, verbose=1
)

# PASO 2: DESCONGELAR Y REFINAR
print("\nğŸ”“ FASE 2: Descongelando BERT para Fine-Tuning (El toque maestro)...")
bert_base.trainable = True # ğŸ”“ CANDADO QUITADO
optimizer_fine = tf.keras.optimizers.Adam(learning_rate=1e-5) # Lento y seguro
model_v8.compile(optimizer=optimizer_fine, loss='binary_crossentropy', metrics=['accuracy'])

history_v8 = model_v8.fit(
    x=[X_train_ids, X_train_masks], y=y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=3, # Probamos 3 Ã©pocas aquÃ­ (una mÃ¡s que en la v4)
    batch_size=8, verbose=1
)


# --- GENERAR CSV PARA v8 ---
print("ğŸ’¾ Generando archivo de envÃ­o para v8 (Modelo Estabilizado)...")

ids_test, preds_test = [], []
ruta_test = os.path.join(ruta_base_data, 'test')
carpetas_test = sorted(os.listdir(ruta_test))

# FunciÃ³n auxiliar
def preparar(texto):
    enc = tokenizer.encode_plus(texto, max_length=300, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='tf')
    return enc['input_ids'], enc['attention_mask']

for carpeta in carpetas_test:
    try:
        id_caso = int(carpeta.split('_')[1])
        ruta_carpeta = os.path.join(ruta_test, carpeta)
        with open(os.path.join(ruta_carpeta, 'file_1.txt'), 'r', errors='ignore') as f: t1 = limpiar_texto(f.read())
        with open(os.path.join(ruta_carpeta, 'file_2.txt'), 'r', errors='ignore') as f: t2 = limpiar_texto(f.read())
        id1, mask1 = preparar(t1); id2, mask2 = preparar(t2)
        
        pred1 = model_v8.predict([id1, mask1], verbose=0)[0][0]
        pred2 = model_v8.predict([id2, mask2], verbose=0)[0][0]
        ids_test.append(id_caso); preds_test.append(1 if pred1 > pred2 else 2)
    except: continue

pd.DataFrame({'id': ids_test, 'real_text_id': preds_test}).to_csv('submission_bert_v8.csv', index=False)
print("âœ… Archivo 'submission_bert_v8.csv' listo. Â¡A ver cuÃ¡nto saca!")


# --- INICIAR EXPERIMENTO V9 (WARM-UP + MAYOR VELOCIDAD) ---
VERSION = "v9"
print(f"\nğŸ§ª Iniciando Experimento {VERSION} (HipÃ³tesis: El Warm-up permite mayor LR)")

tf.keras.backend.clear_session()

# PASO 1: CONGELADO (CALENTAMIENTO)
bert_base = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
bert_base.trainable = False 

input_ids = Input(shape=(300,), dtype=tf.int32, name='input_ids')
input_masks = Input(shape=(300,), dtype=tf.int32, name='attention_mask')
def envoltorio_bert(args): return bert_base(args[0], attention_mask=args[1])[0]
bert_output = Lambda(envoltorio_bert, output_shape=(300, 768))([input_ids, input_masks])
x = Dense(64, activation='relu')(bert_output[:, 0, :])
x = Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)

model_v9 = Model(inputs=[input_ids, input_masks], outputs=output)

print("ğŸ§Š FASE 1: Calentamiento (2 Ã‰pocas)...")
model_v9.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_v9.fit(
    x=[X_train_ids, X_train_masks], y=y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=2, batch_size=8, verbose=1
)

# PASO 2: DESCONGELADO + MAYOR VELOCIDAD
print("\nğŸ”“ FASE 2: Fine-Tuning Acelerado (LR 2e-5)...")
bert_base.trainable = True 
# AQUÃ� ESTÃ� LA CLAVE: Usamos 2e-5 en vez de 1e-5
optimizer_bold = tf.keras.optimizers.Adam(learning_rate=2e-5) 

model_v9.compile(optimizer=optimizer_bold, loss='binary_crossentropy', metrics=['accuracy'])

history_v9 = model_v9.fit(
    x=[X_train_ids, X_train_masks], y=y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=3, 
    batch_size=8, verbose=1
)


# --- GENERAR CSV PARA v9 ---
print("ğŸ’¾ Generando archivo de envÃ­o para v9 (Warm-up + Velocidad)...")

ids_test, preds_test = [], []
ruta_test = os.path.join(ruta_base_data, 'test')
carpetas_test = sorted(os.listdir(ruta_test))

def preparar(texto):
    enc = tokenizer.encode_plus(texto, max_length=300, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='tf')
    return enc['input_ids'], enc['attention_mask']

for carpeta in carpetas_test:
    try:
        id_caso = int(carpeta.split('_')[1])
        ruta_carpeta = os.path.join(ruta_test, carpeta)
        with open(os.path.join(ruta_carpeta, 'file_1.txt'), 'r', errors='ignore') as f: t1 = limpiar_texto(f.read())
        with open(os.path.join(ruta_carpeta, 'file_2.txt'), 'r', errors='ignore') as f: t2 = limpiar_texto(f.read())
        id1, mask1 = preparar(t1); id2, mask2 = preparar(t2)
        
        pred1 = model_v9.predict([id1, mask1], verbose=0)[0][0]
        pred2 = model_v9.predict([id2, mask2], verbose=0)[0][0]
        ids_test.append(id_caso); preds_test.append(1 if pred1 > pred2 else 2)
    except: continue

pd.DataFrame({'id': ids_test, 'real_text_id': preds_test}).to_csv('submission_bert_v9.csv', index=False)
print("âœ… Archivo 'submission_bert_v9.csv' listo. Â¡Este promete!")


# --- INICIAR EXPERIMENTO V10 (MAYOR REGULARIZACIÃ“N / DROPOUT) ---
VERSION = "v10"
NUEVO_DROPOUT = 0.4 # Apagamos el 40% de neuronas (MÃ¡s difÃ­cil para el modelo)

print(f"\nğŸ§ª Iniciando Experimento {VERSION} (HipÃ³tesis: MÃ¡s Dropout reduce el overfitting)")

tf.keras.backend.clear_session()

# PASO 1: CONGELADO 
bert_base = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
bert_base.trainable = False 

input_ids = Input(shape=(300,), dtype=tf.int32, name='input_ids')
input_masks = Input(shape=(300,), dtype=tf.int32, name='attention_mask')
def envoltorio_bert(args): return bert_base(args[0], attention_mask=args[1])[0]
bert_output = Lambda(envoltorio_bert, output_shape=(300, 768))([input_ids, input_masks])
cls_token = bert_output[:, 0, :]

# --- AQUÃ� ESTÃ� EL CAMBIO ---
x = Dense(64, activation='relu')(cls_token)
x = Dropout(NUEVO_DROPOUT)(x) # Usamos 0.4
output = Dense(1, activation='sigmoid')(x)
# ---------------------------

model_v10 = Model(inputs=[input_ids, input_masks], outputs=output)

print("ğŸ§Š FASE 1: Calentamiento con Dropout alto...")
model_v10.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_v10.fit(
    x=[X_train_ids, X_train_masks], y=y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=2, batch_size=8, verbose=1
)

# PASO 2: DESCONGELADO
print("\nğŸ”“ FASE 2: Fine-Tuning EstÃ¡ndar...")
bert_base.trainable = True 
optimizer_safe = tf.keras.optimizers.Adam(learning_rate=1e-5) 

model_v10.compile(optimizer=optimizer_safe, loss='binary_crossentropy', metrics=['accuracy'])

history_v10 = model_v10.fit(
    x=[X_train_ids, X_train_masks], y=y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=3, 
    batch_size=8, verbose=1
)


# --- GENERAR CSV PARA v10 ---
print("ğŸ’¾ Generando archivo de envÃ­o para v10 (Alta RegularizaciÃ³n)...")

ids_test, preds_test = [], []
ruta_test = os.path.join(ruta_base_data, 'test')
carpetas_test = sorted(os.listdir(ruta_test))

def preparar(texto):
    enc = tokenizer.encode_plus(texto, max_length=300, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='tf')
    return enc['input_ids'], enc['attention_mask']

for carpeta in carpetas_test:
    try:
        id_caso = int(carpeta.split('_')[1])
        ruta_carpeta = os.path.join(ruta_test, carpeta)
        with open(os.path.join(ruta_carpeta, 'file_1.txt'), 'r', errors='ignore') as f: t1 = limpiar_texto(f.read())
        with open(os.path.join(ruta_carpeta, 'file_2.txt'), 'r', errors='ignore') as f: t2 = limpiar_texto(f.read())
        id1, mask1 = preparar(t1); id2, mask2 = preparar(t2)
        
        pred1 = model_v10.predict([id1, mask1], verbose=0)[0][0]
        pred2 = model_v10.predict([id2, mask2], verbose=0)[0][0]
        ids_test.append(id_caso); preds_test.append(1 if pred1 > pred2 else 2)
    except: continue

pd.DataFrame({'id': ids_test, 'real_text_id': preds_test}).to_csv('submission_bert_v10.csv', index=False)
print("âœ… Archivo 'submission_bert_v10.csv' listo.")


# --- INICIAR EXPERIMENTO V11 (ALL-IN: ENTRENAR CON TODO) ---
VERSION = "v11"
print(f"\nğŸš€ Iniciando Experimento {VERSION} (Estrategia: Usar 100% de los datos)")

# 1. JUNTAR LOS DATOS (FusiÃ³n)
import numpy as np
print("ğŸ“š Fusionando Train + Validation...")
X_total_ids = np.concatenate([X_train_ids, X_val_ids])
X_total_masks = np.concatenate([X_train_masks, X_val_masks])
y_total = np.concatenate([y_train, y_val])

print(f"   Total de noticias para entrenar: {len(y_total)}")

# 2. LIMPIEZA
tf.keras.backend.clear_session()

# 3. MODELO (ConfiguraciÃ³n Ganadora v9: Warmup + LR 2e-5)
bert_base = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
bert_base.trainable = False  # ğŸ”’ FASE 1: CONGELADO

input_ids = Input(shape=(300,), dtype=tf.int32, name='input_ids')
input_masks = Input(shape=(300,), dtype=tf.int32, name='attention_mask')
def envoltorio_bert(args): return bert_base(args[0], attention_mask=args[1])[0]
bert_output = Lambda(envoltorio_bert, output_shape=(300, 768))([input_ids, input_masks])
cls_token = bert_output[:, 0, :]
x = Dense(64, activation='relu')(cls_token)
x = Dropout(0.2)(x) # Volvemos al Dropout estÃ¡ndar
output = Dense(1, activation='sigmoid')(x)

model_v11 = Model(inputs=[input_ids, input_masks], outputs=output)

print("ğŸ§Š FASE 1: Calentamiento con TODOS los datos...")
model_v11.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_v11.fit(
    x=[X_total_ids, X_total_masks], y=y_total,
    epochs=2, batch_size=8, verbose=1
    # Â¡OJO! No hay validation_data aquÃ­
)

print("\nğŸ”“ FASE 2: Fine-Tuning con TODOS los datos (LR 2e-5)...")
bert_base.trainable = True # ğŸ”“ DESCONGELADO
optimizer_bold = tf.keras.optimizers.Adam(learning_rate=2e-5) 

model_v11.compile(optimizer=optimizer_bold, loss='binary_crossentropy', metrics=['accuracy'])

history_v11 = model_v11.fit(
    x=[X_total_ids, X_total_masks], y=y_total,
    epochs=3, batch_size=8, verbose=1
)

# --- GENERAR EL CSV INMEDIATAMENTE ---
print(f"\nğŸ’¾ Generando envÃ­o {VERSION} (ALL-IN)...")
# (Usamos el mismo cÃ³digo de predicciÃ³n, resumido)
ids_test, preds_test = [], []
for carpeta in carpetas_test:
    try:
        id_caso = int(carpeta.split('_')[1])
        ruta_carpeta = os.path.join(ruta_test, carpeta)
        with open(os.path.join(ruta_carpeta, 'file_1.txt'), 'r', errors='ignore') as f: t1 = limpiar_texto(f.read())
        with open(os.path.join(ruta_carpeta, 'file_2.txt'), 'r', errors='ignore') as f: t2 = limpiar_texto(f.read())
        id1, mask1 = preparar(t1); id2, mask2 = preparar(t2)
        pred1 = model_v11.predict([id1, mask1], verbose=0)[0][0]
        pred2 = model_v11.predict([id2, mask2], verbose=0)[0][0]
        ids_test.append(id_caso); preds_test.append(1 if pred1 > pred2 else 2)
    except: continue
pd.DataFrame({'id': ids_test, 'real_text_id': preds_test}).to_csv('submission_bert_v11.csv', index=False)
print("âœ… Â¡v11 LISTO! Este lleva toda la potencia posible.")


import pandas as pd
import numpy as np
from scipy import stats

print("ğŸ—³ï¸� INICIANDO VOTACIÃ“N DEL DREAM TEAM...")

# 1. Cargar las predicciones de ayer
# AsegÃºrate de haber subido estos archivos a la sesiÃ³n actual
try:
    df_v9 = pd.read_csv('submission_bert_v9.csv')   # El Audaz (0.863)
    df_v11 = pd.read_csv('submission_bert_v11.csv') # El All-In (0.863)
    df_v10 = pd.read_csv('submission_bert_v10.csv') # El Duro (0.848) - Desempate

    print("âœ… Archivos cargados correctamente.")

    # 2. VotaciÃ³n (Hard Voting)
    # Convertimos a arrays
    preds_9 = df_v9['real_text_id'].values
    preds_11 = df_v11['real_text_id'].values
    preds_10 = df_v10['real_text_id'].values

    # Usamos la moda (el valor que mÃ¡s se repite entre los 3)
    # Ej: Si v9 dice 1, v11 dice 1, v10 dice 2 -> Gana 1.
    final_preds, count = stats.mode([preds_9, preds_11, preds_10], axis=0, keepdims=True)
    final_preds = final_preds[0]

    # 3. Guardar el Resultado Final
    df_ensemble = pd.DataFrame({
        'id': df_v9['id'],
        'real_text_id': final_preds
    })
    
    nombre_archivo = 'submission_ENSEMBLE_V12.csv'
    df_ensemble.to_csv(nombre_archivo, index=False)
    
    print(f"ğŸ�† Â¡VotaciÃ³n terminada! Archivo '{nombre_archivo}' generado.")
    

except FileNotFoundError:
    print("âš ï¸� ERROR: No encuentro los archivos .csv.")
    print("   Por favor, sube 'submission_bert_v9.csv', 'v11.csv' y 'v10.csv' a la carpeta de archivos.")


# --- INICIAR EXPERIMENTO V13 (REFINAMIENTO DE LR) ---
VERSION = "v13"
print(f"\nğŸ§ª Iniciando Experimento {VERSION} (HipÃ³tesis: LR mÃ¡s bajo en Fase 2 mejora la convergencia fina)")

tf.keras.backend.clear_session()

# PASO 1: CONGELADO (CALENTAMIENTO) - SE MANTIENE IGUAL QUE V9
bert_base = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
bert_base.trainable = False 

input_ids = Input(shape=(300,), dtype=tf.int32, name='input_ids')
input_masks = Input(shape=(300,), dtype=tf.int32, name='attention_mask')
def envoltorio_bert(args): return bert_base(args[0], attention_mask=args[1])[0]
bert_output = Lambda(envoltorio_bert, output_shape=(300, 768))([input_ids, input_masks])

# Mismo "Head" que v9
x = Dense(64, activation='relu')(bert_output[:, 0, :])
x = Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)

model_v13 = Model(inputs=[input_ids, input_masks], outputs=output)

print("ğŸ§Š FASE 1: Calentamiento (Igual a v9)...")
model_v13.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_v13.fit(
    x=[X_train_ids, X_train_masks], y=y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=2, batch_size=8, verbose=1
)

# PASO 2: DESCONGELADO + MENOR VELOCIDAD (AquÃ­ estÃ¡ el cambio)
print("\nğŸ”“ FASE 2: Fine-Tuning de PrecisiÃ³n (LR 1e-5)...")
bert_base.trainable = True 

# CAMBIO CRÃ�TICO: Bajamos de 2e-5 a 1e-5
optimizer_refinado = tf.keras.optimizers.Adam(learning_rate=1e-5) 

model_v13.compile(optimizer=optimizer_refinado, loss='binary_crossentropy', metrics=['accuracy'])

history_v13 = model_v13.fit(
    x=[X_train_ids, X_train_masks], y=y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=3, # Mantenemos las mismas Ã©pocas para comparar manzanas con manzanas
    batch_size=8, verbose=1
)


from tensorflow.keras.callbacks import EarlyStopping

# --- INICIAR EXPERIMENTO V14 (EXTENSIÃ“N DE V13) ---
VERSION = "v14"
print(f"\nğŸ§ª Iniciando Experimento {VERSION} (HipÃ³tesis: V13 necesitaba mÃ¡s Ã©pocas para converger)")

tf.keras.backend.clear_session()

# PASO 1: CONGELADO (Igual que antes)
bert_base = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
bert_base.trainable = False 

input_ids = Input(shape=(300,), dtype=tf.int32, name='input_ids')
input_masks = Input(shape=(300,), dtype=tf.int32, name='attention_mask')
def envoltorio_bert(args): return bert_base(args[0], attention_mask=args[1])[0]
bert_output = Lambda(envoltorio_bert, output_shape=(300, 768))([input_ids, input_masks])

x = Dense(64, activation='relu')(bert_output[:, 0, :])
x = Dropout(0.2)(x)
output = Dense(1, activation='sigmoid')(x)

model_v14 = Model(inputs=[input_ids, input_masks], outputs=output)

print("ğŸ§Š FASE 1: Calentamiento...")
model_v14.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_v14.fit(
    x=[X_train_ids, X_train_masks], y=y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=2, batch_size=8, verbose=1
)

# PASO 2: DESCONGELADO EXTENDIDO
print("\nğŸ”“ FASE 2: Fine-Tuning Largo (LR 1e-5 + Early Stopping)...")
bert_base.trainable = True 
optimizer_refinado = tf.keras.optimizers.Adam(learning_rate=1e-5) 

model_v14.compile(optimizer=optimizer_refinado, loss='binary_crossentropy', metrics=['accuracy'])

# Definimos el freno de emergencia
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=2,           # Espera 2 Ã©pocas antes de detenerse si no mejora
    restore_best_weights=True, # Al final, regresa al mejor momento (no al Ãºltimo)
    verbose=1
)

history_v14 = model_v14.fit(
    x=[X_train_ids, X_train_masks], y=y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=10,            # Le damos espacio para correr
    batch_size=8, 
    callbacks=[early_stop], # Activamos el freno
    verbose=1
)


import pandas as pd
import os
import numpy as np
import tensorflow as tf

print("ğŸ’¾ Generando archivo de envÃ­o para v14 (Mejor Loss: 0.4509)...")

ids_test = []
preds_test = []

# Rutas basadas en tu notebook
ruta_base_data = '/kaggle/input/fake-or-real-the-impostor-hunt/data/'
ruta_test = os.path.join(ruta_base_data, 'test')
carpetas_test = sorted(os.listdir(ruta_test))

# FunciÃ³n de preparaciÃ³n (usa el tokenizer que ya tienes en memoria)
def preparar(texto):
    enc = tokenizer.encode_plus(
        texto, 
        max_length=300, 
        padding='max_length', 
        truncation=True, 
        return_attention_mask=True, 
        return_tensors='tf'
    )
    return enc['input_ids'], enc['attention_mask']

print(f"ğŸ“‚ Procesando {len(carpetas_test)} casos de prueba...")

# Bucle de predicciÃ³n
count = 0
for carpeta in carpetas_test:
    try:
        # Extraer ID del nombre de la carpeta
        id_caso = int(carpeta.split('_')[1])
        ruta_carpeta = os.path.join(ruta_test, carpeta)
        
        # Leer textos y limpiar
        with open(os.path.join(ruta_carpeta, 'file_1.txt'), 'r', errors='ignore') as f: 
            t1 = limpiar_texto(f.read())
        with open(os.path.join(ruta_carpeta, 'file_2.txt'), 'r', errors='ignore') as f: 
            t2 = limpiar_texto(f.read())
        
        # Preparar tensores
        id1, mask1 = preparar(t1)
        id2, mask2 = preparar(t2)
        
        # PREDECIR CON EL MODELO V14
        # El modelo devuelve la probabilidad de ser REAL (Clase 1)
        pred1 = model_v14.predict([id1, mask1], verbose=0)[0][0]
        pred2 = model_v14.predict([id2, mask2], verbose=0)[0][0]
        
        # LÃ³gica del Impostor:
        # Si pred1 > pred2, entonces el archivo 1 es el Real (1)
        # Si no, el archivo 2 es el Real (2)
        if pred1 > pred2:
            ganador = 1
        else:
            ganador = 2
            
        ids_test.append(id_caso)
        preds_test.append(ganador)
        
        count += 1
        if count % 100 == 0: print(f"   ... {count} procesados")
            
    except Exception as e:
        print(f"â�Œ Error en {carpeta}: {e}")
        continue

# Crear DataFrame y Guardar
df_sub = pd.DataFrame({'id': ids_test, 'real_text_id': preds_test})
nombre_archivo = 'submission_bert_v13.csv'
df_sub.to_csv(nombre_archivo, index=False)

print(f"âœ… Â¡Ã‰XITO! Archivo '{nombre_archivo}' generado.")
print(df_sub.head())


from tensorflow.keras.callbacks import EarlyStopping

# --- INICIAR EXPERIMENTO V15 (REGULARIZACIÃ“N FUERTE) ---
VERSION = "v15"
print(f"\nğŸ§ª Iniciando Experimento {VERSION} (HipÃ³tesis: Mayor Dropout (0.4) mejora generalizaciÃ³n)")

tf.keras.backend.clear_session()

# PASO 1: CONGELADO (CALENTAMIENTO)
bert_base = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
bert_base.trainable = False 

input_ids = Input(shape=(300,), dtype=tf.int32, name='input_ids')
input_masks = Input(shape=(300,), dtype=tf.int32, name='attention_mask')
def envoltorio_bert(args): return bert_base(args[0], attention_mask=args[1])[0]
bert_output = Lambda(envoltorio_bert, output_shape=(300, 768))([input_ids, input_masks])

# CAMBIO AQUÃ�: Aumentamos Dropout de 0.2 a 0.4
x = Dense(64, activation='relu')(bert_output[:, 0, :])
x = Dropout(0.4)(x)  # <--- MÃ�S AGRESIVO
output = Dense(1, activation='sigmoid')(x)

model_v15 = Model(inputs=[input_ids, input_masks], outputs=output)

print("ğŸ§Š FASE 1: Calentamiento...")
model_v15.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_v15.fit(
    x=[X_train_ids, X_train_masks], y=y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=2, batch_size=8, verbose=1
)

# PASO 2: DESCONGELADO (Misma configuraciÃ³n exitosa de v14)
print("\nğŸ”“ FASE 2: Fine-Tuning (LR 1e-5 + Dropout Alto)...")
bert_base.trainable = True 
optimizer_refinado = tf.keras.optimizers.Adam(learning_rate=1e-5) 

model_v15.compile(optimizer=optimizer_refinado, loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=2, 
    restore_best_weights=True, 
    verbose=1
)

history_v15 = model_v15.fit(
    x=[X_train_ids, X_train_masks], y=y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=10, 
    batch_size=8, 
    callbacks=[early_stop], 
    verbose=1
)


# --- INICIAR EXPERIMENTO V16 (EL GEMELO DE V15) ---
VERSION = "v16"
print(f"\nğŸ§ª Iniciando Experimento {VERSION} (HipÃ³tesis: Misma config v15, distinta semilla)")

tf.keras.backend.clear_session()

# CAMBIO DE SEMILLA: Para buscar otro mÃ­nimo local
tf.random.set_seed(1234) # Antes era por defecto (42 en muchos casos)
np.random.seed(1234)

# PASO 1: CONGELADO 
bert_base = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
bert_base.trainable = False 

input_ids = Input(shape=(300,), dtype=tf.int32, name='input_ids')
input_masks = Input(shape=(300,), dtype=tf.int32, name='attention_mask')
def envoltorio_bert(args): return bert_base(args[0], attention_mask=args[1])[0]
bert_output = Lambda(envoltorio_bert, output_shape=(300, 768))([input_ids, input_masks])

# MISMOS HIPERPARÃ�METROS QUE V15 (Dropout 0.4)
x = Dense(64, activation='relu')(bert_output[:, 0, :])
x = Dropout(0.4)(x) 
output = Dense(1, activation='sigmoid')(x)

model_v16 = Model(inputs=[input_ids, input_masks], outputs=output)

print("ğŸ§Š FASE 1: Calentamiento...")
model_v16.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_v16.fit(
    x=[X_train_ids, X_train_masks], y=y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=2, batch_size=8, verbose=1
)

# PASO 2: DESCONGELADO 
print("\nğŸ”“ FASE 2: Fine-Tuning (LR 1e-5 + Dropout 0.4)...")
bert_base.trainable = True 
optimizer_refinado = tf.keras.optimizers.Adam(learning_rate=1e-5) 

model_v16.compile(optimizer=optimizer_refinado, loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=2, 
    restore_best_weights=True, 
    verbose=1
)

history_v16 = model_v16.fit(
    x=[X_train_ids, X_train_masks], y=y_train,
    validation_data=([X_val_ids, X_val_masks], y_val),
    epochs=10, 
    batch_size=8, 
    callbacks=[early_stop], 
    verbose=1
)


import pandas as pd
import os
import numpy as np
import tensorflow as tf

# Ajusta el nombre si decidiste llamarlo diferente
print("ğŸ’¾ Generando archivo para Carga 15 (Model v16)...")

ids_test = []
preds_test = []

ruta_base_data = '/kaggle/input/fake-or-real-the-impostor-hunt/data/'
ruta_test = os.path.join(ruta_base_data, 'test')
carpetas_test = sorted(os.listdir(ruta_test))

# Reusamos la funciÃ³n 'preparar' y el 'tokenizer' que ya estÃ¡n en memoria
# Si se borraron, avÃ­same para pasarte el bloque de reinicio.

count = 0
for carpeta in carpetas_test:
    try:
        id_caso = int(carpeta.split('_')[1])
        ruta_carpeta = os.path.join(ruta_test, carpeta)
        
        with open(os.path.join(ruta_carpeta, 'file_1.txt'), 'r', errors='ignore') as f: 
            t1 = limpiar_texto(f.read())
        with open(os.path.join(ruta_carpeta, 'file_2.txt'), 'r', errors='ignore') as f: 
            t2 = limpiar_texto(f.read())
        
        id1, mask1 = preparar(t1)
        id2, mask2 = preparar(t2)
        
        # PREDICCIÃ“N CON MODEL_V16 (El Gemelo)
        pred1 = model_v16.predict([id1, mask1], verbose=0)[0][0]
        pred2 = model_v16.predict([id2, mask2], verbose=0)[0][0]
        
        if pred1 > pred2:
            ganador = 1
        else:
            ganador = 2
            
        ids_test.append(id_caso)
        preds_test.append(ganador)
        
        count += 1
        if count % 100 == 0: print(f"   ... {count} procesados")
            
    except Exception as e:
        print(f"â�Œ Error en {carpeta}: {e}")
        continue

df_sub = pd.DataFrame({'id': ids_test, 'real_text_id': preds_test})

# NOMBRE CORREGIDO SEGÃšN TU INDICACIÃ“N: CARGA 15 -> V15
nombre_archivo = 'submission_bert_v15.csv'
df_sub.to_csv(nombre_archivo, index=False)

print(f"âœ… Â¡Ã‰XITO! Archivo '{nombre_archivo}' generado.")
print(df_sub.head())


import pandas as pd
import scipy.stats as stats

print("ğŸ—³ï¸� Iniciando VotaciÃ³n por MayorÃ­a (Ensemble)...")

# 1. CARGA TUS 3 MEJORES ARCHIVOS
# AsegÃºrate de que los nombres coincidan con lo que tienes en la carpeta output
file1 = 'submission_bert_v13.csv'  # El del LR bajo
file2 = 'submission_bert_v15.csv'  # El del Dropout alto
file3 = 'submission_bert_v11.csv'  # <--- CAMBIA ESTE por tu 3er mejor archivo (ej. v16 o v11)

# Si no tienes un 3ro a mano, avÃ­same y cambiamos la estrategia, 
# pero idealmente necesitamos 3 impares para desempatar.

try:
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    print(f"âœ… Archivos cargados: {file1}, {file2}, {file3}")

    # 2. VERIFICAR CORRELACIÃ“N (Opcional, para ver si son distintos)
    # Si son muy iguales (1.0), el ensemble no sirve mucho. Si es 0.9, es perfecto.
    corr12 = (df1['real_text_id'] == df2['real_text_id']).mean()
    print(f"Coincidencia entre {file1} y {file2}: {corr12:.2%}")

    # 3. VOTACIÃ“N (La Moda EstadÃ­stica)
    # Juntamos las columnas de predicciÃ³n
    votes = pd.DataFrame({
        'v1': df1['real_text_id'],
        'v2': df2['real_text_id'],
        'v3': df3['real_text_id']
    })

    # Calculamos la moda (el valor que mÃ¡s se repite en cada fila)
    # axis=1 es por fila
    final_vote = votes.mode(axis=1)[0].astype(int)

    # 4. GUARDAR
    submission = pd.DataFrame({
        'id': df1['id'],
        'real_text_id': final_vote
    })

    filename = 'submission_ensemble_mayority_V16.csv'
    submission.to_csv(filename, index=False)

    print(f"ğŸ�† Â¡Ensemble generado! Archivo: {filename}")
    print(submission.head())

except FileNotFoundError as e:
    print(f"â�Œ Error: No encuentro alguno de los archivos. Revisa los nombres.\n{e}")


# --- 1. IMPORTACIONES Y CONFIGURACIÃ“N ---
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification
from sklearn.model_selection import train_test_split

# ConfiguraciÃ³n Anti-Errores
# RoBERTa es grande: usamos batch_size pequeÃ±o (4 u 8) para evitar "OOM" (Out Of Memory)
BATCH_SIZE = 8  
MAX_LEN = 256   # 256 es suficiente para detectar patrones, 512 podrÃ­a ser muy pesado
LR = 1e-5       # RoBERTa necesita una tasa de aprendizaje MUY baja

print("âœ… LibrerÃ­as cargadas. ConfiguraciÃ³n lista.")


# --- 2. CARGA DE DATOS ---
BASE_PATH = "/kaggle/input/fake-or-real-the-impostor-hunt/data"

def cargar_dataset_entrenamiento():
    # 1. Cargar etiquetas
    df_labels = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
    
    datos = []
    print("ğŸ“‚ Leyendo archivos de entrenamiento...")
    
    for _, row in df_labels.iterrows():
        id_articulo = row['id']
        real_id = row['real_text_id'] # 1 o 2
        
        # Ruta de la carpeta del artÃ­culo (ajustar ceros segÃºn el nombre real de la carpeta)
        # Nota: Asumo que las carpetas no tienen prefijos raros, si falla, verificamos nombres.
        # En el snippet vi nombres tipo "article_0192". Intentamos construir la ruta:
        
        # Buscamos la carpeta que contiene el ID
        # (Esto es mÃ¡s seguro que intentar adivinar los ceros a la izquierda)
        folder_path = None
        for d in os.listdir(os.path.join(BASE_PATH, "train")):
            if str(id_articulo) in d:
                folder_path = os.path.join(BASE_PATH, "train", d)
                break
        
        if not folder_path: continue

        try:
            # Leemos file_1 y file_2
            with open(os.path.join(folder_path, "file_1.txt"), "r", encoding="utf-8", errors="ignore") as f:
                text_1 = f.read()
            with open(os.path.join(folder_path, "file_2.txt"), "r", encoding="utf-8", errors="ignore") as f:
                text_2 = f.read()
            
            # Asignamos etiquetas: Si real_id es 1, text_1 es Real(1) y text_2 es Fake(0)
            if real_id == 1:
                datos.append({'text': text_1, 'label': 1})
                datos.append({'text': text_2, 'label': 0})
            else:
                datos.append({'text': text_1, 'label': 0})
                datos.append({'text': text_2, 'label': 1})
                
        except Exception as e:
            print(f"âš ï¸� Error en artÃ­culo {id_articulo}: {e}")

    return pd.DataFrame(datos)

df_train = cargar_dataset_entrenamiento()
print(f"âœ… Datos cargados: {len(df_train)} filas (debe ser el doble de artÃ­culos).")
display(df_train.head())


# --- 3. TOKENIZACIÃ“N ---
print("ğŸ¤– Iniciando Tokenizer de RoBERTa...")
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

def codificar_datos(textos, max_len=MAX_LEN):
    encodings = tokenizer(
        textos.tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='tf'
    )
    # Convertimos a formato diccionario para TF
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask']
    }

# Separamos X e y
X = df_train['text']
y = df_train['label'].values

# Split (80% Train, 20% Val)
X_train_txt, X_val_txt, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenizamos
print("âš™ï¸� Codificando textos (esto toma unos segundos)...")
train_encodings = codificar_datos(X_train_txt)
val_encodings = codificar_datos(X_val_txt)

# Convertimos a Dataset de TensorFlow (MÃ¡s eficiente para memoria)
train_dataset = tf.data.Dataset.from_tensor_slices((train_encodings, y_train)).shuffle(1000).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_encodings, y_val)).batch(BATCH_SIZE)

print("âœ… Datos listos para entrar al modelo.")


# --- 4. MODELO Y ENTRENAMIENTO (SOLUCIÃ“N DEFINITIVA VERSIÃ“N) ---
import tf_keras  # Esta librerÃ­a puente soluciona el conflicto

tf.keras.backend.clear_session()

print("ğŸ�—ï¸� Cargando modelo RoBERTa...")
# Cargar modelo
model = TFRobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=1)

# --- CORRECCIÃ“N CLAVE ---
# Usamos el optimizador y la loss de 'tf_keras', no de 'tf.keras'
# Esto satisface a la librerÃ­a transformers
optimizer = tf_keras.optimizers.Adam(learning_rate=LR)
loss = tf_keras.losses.BinaryCrossentropy(from_logits=True)

# Compilamos usando los componentes compatibles
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

print("ğŸš€ Iniciando entrenamiento...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3 
)


# --- 5. GENERACIÃ“N DE SUBMISSION ---
print("ğŸ”® Generando predicciones para Test...")

test_folder = os.path.join(BASE_PATH, "test")
submission_rows = []

# Iteramos sobre las carpetas de test
carpetas = [f for f in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, f))]

for folder_name in carpetas:
    try:
        # Extraer ID del nombre de la carpeta (ej: article_0192 -> 192)
        article_id = int(''.join(filter(str.isdigit, folder_name)))
        folder_path = os.path.join(test_folder, folder_name)
        
        # Leer textos
        with open(os.path.join(folder_path, "file_1.txt"), "r", errors="ignore") as f: t1 = f.read()
        with open(os.path.join(folder_path, "file_2.txt"), "r", errors="ignore") as f: t2 = f.read()
        
        # Tokenizar par
        # Ojo: batch de 2 (file 1 y file 2)
        enc = tokenizer([t1, t2], truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='tf')
        dataset_input = {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask']}
        
        # Predecir (logits)
        preds = model.predict(dataset_input, verbose=0).logits
        
        # Convertir logits a probabilidad (sigmoide)
        probs = tf.nn.sigmoid(preds).numpy().flatten()
        
        # LÃ³gica: Â¿CuÃ¡l tiene mayor probabilidad de ser 'Real' (clase 1)?
        prob_f1 = probs[0]
        prob_f2 = probs[1]
        
        if prob_f1 > prob_f2:
            vote = 1
        else:
            vote = 2
            
        submission_rows.append({'id': article_id, 'real_text_id': vote})
        
    except Exception as e:
        print(f"Error en {folder_name}: {e}")

# Guardar
df_sub = pd.DataFrame(submission_rows)
df_sub = df_sub.sort_values('id') # Kaggle suele pedir orden
filename = 'submission_roberta_v17.csv'
df_sub.to_csv(filename, index=False)

print(f"ğŸ�† Â¡Listo! Archivo generado: {filename}")
display(df_sub.head())


# --- ENSEMBLE: MEZCLA DE V16 y V17 ---
import pandas as pd

# 1. Cargar tus dos mejores versiones
# AsegÃºrate de que los nombres de archivo sean correctos
sub_v16 = pd.read_csv('submission_ensemble_mayority_V16.csv') # Tu mejor anterior
sub_v17 = pd.read_csv('submission_roberta_v17.csv')           # Tu nuevo RoBERTa

print("Longitud v16:", len(sub_v16))
print("Longitud v17:", len(sub_v17))

# 2. Unir (Merge) para comparar fila por fila
merged = pd.merge(sub_v16, sub_v17, on='id', suffixes=('_v16', '_v17'))

# 3. Ver cuÃ¡ntas diferencias hay
merged['diff'] = merged['real_text_id_v16'] != merged['real_text_id_v17']
num_diff = merged['diff'].sum()

print(f"\nâš¡ CONFLICTOS DETECTADOS: {num_diff} de {len(merged)} filas.")
print(f"Porcentaje de desacuerdo: {(num_diff / len(merged)) * 100:.2f}%")

# 4. RESOLUCIÃ“N DE CONFLICTOS (La Estrategia)
# Si tienes un tercer archivo (ej. v15), Ãºsalo para desempatar (Voto por MayorÃ­a).
# Si NO tienes tercero, confiaremos en v17 (RoBERTa) porque suele ser mÃ¡s robusto,
# PERO si v16 era muy bueno, podrÃ­amos revisar esos casos.

# --- OPCIÃ“N: VOTO PONDERADO / JERÃ�RQUICO ---
# AquÃ­ creamos la columna final. 
# LÃ³gica: Si son iguales, genial. Si son distintos, gana v17 (RoBERTa).
# (Si quieres cambiarlo para que gane v16, cambia la lÃ³gica en el 'else')

def resolver_conflicto(row):
    if row['real_text_id_v16'] == row['real_text_id_v17']:
        return row['real_text_id_v17'] # Coinciden
    else:
        # AQUÃ� ES DONDE MEJORAMOS:
        # En lugar de tirar una moneda, le damos el voto al modelo mÃ¡s fuerte.
        # Asumo que RoBERTa (v17) es mÃ¡s inteligente.
        return row['real_text_id_v17'] 

# Aplicamos
merged['real_text_id'] = merged.apply(resolver_conflicto, axis=1)

# 5. Generar Archivo Final
submission_v18 = merged[['id', 'real_text_id']]
filename = 'submission_ensemble_v18_RoBERTa_Priority.csv'
submission_v18.to_csv(filename, index=False)

print(f"\nğŸ�† Â¡Archivo generado!: {filename}")
print(submission_v18.head())


# --- AJUSTES v19 ---
# Subimos a 512 para leer TODO el artÃ­culo
MAX_LEN = 512   

# Bajamos a 4 para que la GPU aguante el tamaÃ±o extra
BATCH_SIZE = 4  

# El resto igual
LR = 1e-5


# --- AJUSTES v19 ---
MAX_LEN = 512   # Leemos el artÃ­culo completo
BATCH_SIZE = 4  # Bajamos a 4 para no saturar la memoria GPU

print(f"ğŸ”„ Re-configurando para MAX_LEN={MAX_LEN} y BATCH_SIZE={BATCH_SIZE}...")

# 1. Re-tokenizamos con la nueva longitud
# Usamos el mismo 'df_train' que ya tenÃ­as cargado
print("âš™ï¸� Re-codificando textos (puede tardar un poco mÃ¡s)...")
train_encodings = codificar_datos(X_train_txt, max_len=MAX_LEN)
val_encodings = codificar_datos(X_val_txt, max_len=MAX_LEN)

# 2. Creamos los nuevos datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_encodings, y_train)).shuffle(1000).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_encodings, y_val)).batch(BATCH_SIZE)

print("âœ… Datos listos para v19.")


import tf_keras # Aseguramos la compatibilidad

# Limpieza de memoria vital
tf.keras.backend.clear_session()
if 'model' in globals(): del model

print("ğŸ�—ï¸� Iniciando nuevo modelo RoBERTa v19...")
model = TFRobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=1)

# Optimizador compatible
optimizer = tf_keras.optimizers.Adam(learning_rate=1e-5)
loss = tf_keras.losses.BinaryCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

print("ğŸš€ Entrenando v19 (esto serÃ¡ mÃ¡s lento por el tamaÃ±o 512)...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3 
)


# --- ENTRENAMIENTO EXTRA (FINE-TUNING) ---
print("ğŸ”¥ El modelo despertÃ³. Vamos a darle 2 Ã©pocas mÃ¡s para exprimirlo...")

# Al llamar a .fit() de nuevo sin recompilar, sigue desde donde se quedÃ³
history_extra = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=4  # Solo 2 mÃ¡s para no pasarnos (Overfitting)
)

# --- RE-GENERAR SUBMISSION v19 ---
print("\nğŸ”® Re-generando submission v19 mejorada...")

submission_rows = []
carpetas = sorted([f for f in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, f))])

for folder_name in carpetas:
    try:
        article_id = int(''.join(filter(str.isdigit, folder_name)))
        folder_path = os.path.join(test_folder, folder_name)
        
        with open(os.path.join(folder_path, "file_1.txt"), "r", errors="ignore") as f: t1 = f.read()
        with open(os.path.join(folder_path, "file_2.txt"), "r", errors="ignore") as f: t2 = f.read()
        
        # Inferencia con MAX_LEN=512
        inputs = tokenizer([t1, t2], truncation=True, padding='max_length', max_length=512, return_tensors='tf')
        inputs_tf = dict(inputs)
        
        preds = model.predict(inputs_tf, verbose=0).logits
        probs = tf.nn.sigmoid(preds).numpy().flatten()
        
        pred_label = 1 if probs[0] > probs[1] else 2
        submission_rows.append({'id': article_id, 'real_text_id': pred_label})
        
    except Exception as e:
        print(f"Error: {e}")

df_sub = pd.DataFrame(submission_rows).sort_values('id')
filename = 'submission_roberta_v19_512_extended.csv'
df_sub.to_csv(filename, index=False)

print(f"ğŸ�† Â¡Listo! Archivo final mejorado: {filename}")


# --- V20: ENTRENAMIENTO INTELIGENTE (VERSIÃ“N COMPATIBLE) ---
import tf_keras
import tensorflow as tf
import os

# CAMBIO CRUCIAL: Importamos los callbacks desde tf_keras
from tf_keras.callbacks import ModelCheckpoint, EarlyStopping

# 1. Limpieza
tf.keras.backend.clear_session()
if 'model' in globals(): del model

print("ğŸ�—ï¸� Configurando RoBERTa v20 (Todo tf_keras)...")

# 2. Definir Callbacks (VersiÃ³n Legacy)
# Nota: En tf_keras volvemos a usar .h5 normal, es mÃ¡s estable
checkpoint_path = "best_roberta_v20.h5"

model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# 3. Modelo
# Usamos el modelo, optimizador y loss, todo de tf_keras o compatible
model = TFRobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=1)
optimizer = tf_keras.optimizers.Adam(learning_rate=1e-5)
loss = tf_keras.losses.BinaryCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# 4. Entrenar
print("ğŸš€ Iniciando entrenamiento v20 (8 Ã©pocas con early stopping)...")
history_v20 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=8,  
    callbacks=[model_checkpoint, early_stopping]
)

print("âœ… Entrenamiento listo. El modelo ha recuperado los pesos de la MEJOR Ã©poca.")


# --- GENERACIÃ“N v20 ---
print("ğŸ”® Generando submission con el MEJOR modelo guardado...")

# Cargamos el archivo .h5 (compatible con tf_keras)
model.load_weights("best_roberta_v20.h5")

submission_rows = []
carpetas = sorted([f for f in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, f))])

for folder_name in carpetas:
    try:
        article_id = int(''.join(filter(str.isdigit, folder_name)))
        folder_path = os.path.join(test_folder, folder_name)
        
        with open(os.path.join(folder_path, "file_1.txt"), "r", errors="ignore") as f: t1 = f.read()
        with open(os.path.join(folder_path, "file_2.txt"), "r", errors="ignore") as f: t2 = f.read()
        
        # 512 tokens
        inputs = tokenizer([t1, t2], truncation=True, padding='max_length', max_length=512, return_tensors='tf')
        inputs_tf = dict(inputs)
        
        preds = model.predict(inputs_tf, verbose=0).logits
        probs = tf.nn.sigmoid(preds).numpy().flatten()
        
        pred_label = 1 if probs[0] > probs[1] else 2
        submission_rows.append({'id': article_id, 'real_text_id': pred_label})
        
    except Exception as e:
        print(f"Error: {e}")

df_sub = pd.DataFrame(submission_rows).sort_values('id')
filename = 'submission_roberta_v20_smart.csv'
df_sub.to_csv(filename, index=False)

print(f"ğŸ�† Â¡Listo! Archivo generado: {filename}")


# --- ENSEMBLE V21: LA TRINIDAD (RoBERTa + Ensemble + BERT) ---
import pandas as pd
import os

# Nombres exactos de tus archivos (segÃºn lo que me mostraste)
file_v20 = 'submission_roberta_v20_smart.csv'     # El Rey (RoBERTa)
file_v16 = 'submission_ensemble_mayority_V16.csv' # El Veterano (Ensemble previo)
file_v14 = 'submission_bert_v14.csv'              # El Tapado (Bert con alto Private Score)

print("âš–ï¸� Cargando archivos para el Ensemble Final...")

try:
    # Cargar DataFrames
    df_v20 = pd.read_csv(file_v20)
    df_v16 = pd.read_csv(file_v16)
    df_v14 = pd.read_csv(file_v14)
    
    # Verificar que todos tengan el mismo orden de IDs
    # (Esto es crucial en Kaggle)
    df_v20 = df_v20.sort_values('id')
    df_v16 = df_v16.sort_values('id')
    df_v14 = df_v14.sort_values('id')
    
    print("âœ… Archivos cargados y alineados.")

    # --- VOTACIÃ“N DEMOCRÃ�TICA ---
    # Creamos un DataFrame con los votos
    votes = pd.DataFrame({
        'v20': df_v20['real_text_id'],
        'v16': df_v16['real_text_id'],
        'v14': df_v14['real_text_id']
    })

    # Calculamos la MODA (el valor que se repite al menos 2 veces)
    # Ej: Si v20 dice '1', v16 dice '2', v14 dice '1' -> Gana '1'
    final_vote = votes.mode(axis=1)[0].astype(int)
    
    # AnÃ¡lisis de consistencia
    total = len(votes)
    unanimous = len(votes[ (votes['v20'] == votes['v16']) & (votes['v16'] == votes['v14']) ])
    print(f"\nğŸ“Š EstadÃ­sticas del Acuerdo:")
    print(f"Total de casos: {total}")
    print(f"Unanimidad total (3/3): {unanimous} ({unanimous/total:.1%})")
    print(f"Discrepancias resueltas por voto (2/3): {total - unanimous}")

    # Guardar
    submission = pd.DataFrame({
        'id': df_v20['id'],
        'real_text_id': final_vote
    })

    filename = 'submission_ensemble_v21_Trinity.csv'
    submission.to_csv(filename, index=False)

    print(f"\nğŸ�† Â¡Archivo generado!: {filename}")
    print("SÃºbelo y crucemos los dedos. DeberÃ­a rozar el 0.88 - 0.89")

except FileNotFoundError as e:
    print(f"â�Œ Error: No encuentro algÃºn archivo. AsegÃºrate de haber subido el v14 tambiÃ©n. {e}")


# --- CELDA 1: CONFIGURACIÃ“N INICIAL PARA DEBERTA ---

# 1. Instalamos la pieza clave que le falta a menudo a Kaggle para DeBERTa
# El argumento -q es para que no llene la pantalla de texto (quiet)
!pip install -q sentencepiece

import os
import numpy as np
import pandas as pd
import tensorflow as tf

# 2. Importamos las herramientas especÃ­ficas de DeBERTa
# Nota: Usamos 'DebertaV2' porque la v3 se basa tecnicamente en la arquitectura v2
from transformers import DebertaV2TokenizerFast, TFDebertaV2ForSequenceClassification

# ConfiguraciÃ³n bÃ¡sica para evitar errores de memoria en la GPU
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    print(f"âœ… GPU Detectada: {len(gpus)}")
except RuntimeError as e:
    print(e)

print("âœ… LibrerÃ­as instaladas y entorno listo para DeBERTa v3.")


# --- CELDA 2: CARGA DE DATOS Y PRUEBA DEL "CEREBRO" (TOKENIZER) ---

# 1. Cargar el Tokenizer de DeBERTa v3 Small
# Esto descargarÃ¡ unos megas desde Hugging Face.
print("â�³ Descargando y cargando el Tokenizer de DeBERTa v3 Small...")
try:
    tokenizer = DebertaV2TokenizerFast.from_pretrained('microsoft/deberta-v3-small')
    print("âœ… Tokenizer cargado exitosamente.")
except Exception as e:
    print(f"â�Œ Error fatal cargando el tokenizer: {e}")

# 2. Localizar y Cargar el archivo train.csv
# Buscamos el archivo automÃ¡ticamente para no fallar con la ruta
train_path = ""
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename == 'train.csv':
            train_path = os.path.join(dirname, filename)
            break

if train_path:
    print(f"ğŸ“‚ Archivo encontrado en: {train_path}")
    df_train = pd.read_csv(train_path)
    print(f"âœ… Datos cargados. Tenemos {len(df_train)} ejemplos para entrenar.")
    
    # 3. Prueba rÃ¡pida: Â¿Entiende el modelo una frase?
    prueba = tokenizer("Hola, probando DeBERTa", return_tensors='tf')
    print("ğŸ§  Prueba de tokenizaciÃ³n: Ã‰XITO (El modelo convirtiÃ³ texto a nÃºmeros)")
else:
    print("â�Œ NO se encontrÃ³ el archivo train.csv. Revisa los datos de la competencia.")


# --- CELDA 3: LECTURA MASIVA DE LOS ARCHIVOS DE TEXTO ---

# 1. Definimos el directorio base usando la ruta que YA encontramos en la Celda 2
# Si train_path es ".../data/train.csv", base_dir serÃ¡ ".../data"
base_dir = os.path.dirname(train_path) 

print(f"ğŸ“‚ Directorio base detectado: {base_dir}")

# 2. FunciÃ³n auxiliar para leer file_1 y file_2 de cada carpeta
def leer_contenido_archivos(article_id):
    # Convertimos el ID a string de 4 dÃ­gitos (ej: 26 -> "0026")
    # Esto es crucial porque asÃ­ se llaman las carpetas en Kaggle
    id_str = str(article_id).zfill(4)
    
    # Construimos la ruta: data/train/article_0026/
    ruta_carpeta = os.path.join(base_dir, 'train', f'article_{id_str}')
    
    texto_1 = ""
    texto_2 = ""
    
    # Intentamos leer file_1.txt
    ruta_1 = os.path.join(ruta_carpeta, 'file_1.txt')
    if os.path.exists(ruta_1):
        with open(ruta_1, 'r', encoding='utf-8', errors='replace') as f:
            texto_1 = f.read()
            
    # Intentamos leer file_2.txt
    ruta_2 = os.path.join(ruta_carpeta, 'file_2.txt')
    if os.path.exists(ruta_2):
        with open(ruta_2, 'r', encoding='utf-8', errors='replace') as f:
            texto_2 = f.read()
            
    return pd.Series([texto_1, texto_2])

# 3. Aplicamos la funciÃ³n a toda la tabla
print("â�³ Leyendo el contenido de los archivos txt... (Paciencia, procesando)")

# Asumimos que la columna del ID se llama 'id' (estÃ¡ndar en Kaggle).
# Esto crea dos columnas nuevas: 'text_1' y 'text_2'
df_train[['text_1', 'text_2']] = df_train['id'].apply(leer_contenido_archivos)

# 4. VerificaciÃ³n
print("âœ… Proceso terminado.")
print(f"ğŸ“Š Dimensiones de la tabla: {df_train.shape}")
print("ğŸ”� Muestra de las primeras filas con texto cargado:")
display(df_train.head())


# --- CELDA 4: PREPARACIÃ“N DE DATOS (LABELS Y DATASETS) ---
from sklearn.model_selection import train_test_split

# 1. Convertir 'real_text_id' (1 y 2) a etiquetas (0 y 1)
# Restamos 1 porque los modelos siempre cuentan desde 0
# Label 0 = El Texto 1 es el Real
# Label 1 = El Texto 2 es el Real
df_train['label'] = df_train['real_text_id'] - 1

# 2. Dividir en Entrenamiento (80%) y ValidaciÃ³n (20%)
# Al ser pocos datos (95), es crucial reservar unos pocos para validar (aprox 19 filas)
train_df, val_df = train_test_split(df_train, test_size=0.2, random_state=42)

print(f"ğŸ“Š Datos de Entrenamiento: {len(train_df)} filas")
print(f"ğŸ“Š Datos de ValidaciÃ³n: {len(val_df)} filas")

# 3. FunciÃ³n para convertir los textos a formato TensorFlow
def encoded_dataset(dataframe):
    # El tokenizer procesa el PAR de oraciones juntos (Texto 1 + Texto 2)
    # DeBERTa aprenderÃ¡ la relaciÃ³n entre ambos para decidir cuÃ¡l es el real
    encodings = tokenizer(
        dataframe['text_1'].tolist(), 
        dataframe['text_2'].tolist(), 
        truncation=True, 
        padding=True, 
        max_length=512, # TamaÃ±o estÃ¡ndar seguro para memoria
        return_tensors='tf'
    )
    
    # Empaquetamos todo en un objeto tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        dict(encodings), 
        dataframe['label'].tolist()
    ))
    
    # Shuffle para mezclar y Batch de 4 (para no saturar la memoria con DeBERTa)
    dataset = dataset.shuffle(100).batch(4)
    return dataset

# 4. Generamos los datasets finales
print("âš™ï¸� Tokenizando y creando tensores... esto puede tomar unos segundos.")
train_dataset = encoded_dataset(train_df)
val_dataset = encoded_dataset(val_df)

print("âœ… Datasets listos. Estructura preparada para el entrenamiento.")


# --- CELDA 5: ENTRENAMIENTO (SOLUCIÃ“N DEFINITIVA KERAS 3) ---

# 1. Instalamos el puente de compatibilidad (tf_keras)
# El '-q' es para que no haga mucho ruido en pantalla
!pip install -q tf_keras

import tensorflow as tf
import tf_keras  # Esta es la librerÃ­a que arregla el conflicto
from transformers import TFDebertaV2ForSequenceClassification

# 2. Cargar el modelo
print("ğŸ�—ï¸� Cargando arquitectura DeBERTa v3 Small...")
model = TFDebertaV2ForSequenceClassification.from_pretrained('microsoft/deberta-v3-small', num_labels=2)

# 3. Compilar usando 'tf_keras' en lugar de 'tensorflow.keras'
# Esto crea un optimizador que el modelo SÃ� puede entender
optimizer = tf_keras.optimizers.Adam(learning_rate=2e-5)
loss = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Nota: Al compilar, pasamos el optimizador "puente"
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# 4. Entrenar
print("ğŸš€ Iniciando entrenamiento (Fine-Tuning)...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=4
)

print("âœ… Entrenamiento finalizado.")


# --- CELDA 6: GENERACIÃ“N DE PREDICCIONES (SUBMISSION) ---
import numpy as np

# 1. Identificar los IDs de la carpeta TEST
# Como no tenemos un 'test.csv', escaneamos las carpetas 'article_XXXX'
test_dir = os.path.join(base_dir, 'test')
test_ids = []

print(f"ğŸ“‚ Escaneando carpeta de prueba: {test_dir}")
for folder_name in os.listdir(test_dir):
    if folder_name.startswith('article_'):
        # Extraemos el nÃºmero (ej: article_0192 -> 192)
        try:
            art_id = int(folder_name.split('_')[1])
            test_ids.append(art_id)
        except:
            pass

# Ordenamos los IDs para que quede prolijo
test_ids.sort()
df_test = pd.DataFrame({'id': test_ids})
print(f"ğŸ“Š Encontrados {len(df_test)} artÃ­culos para predecir.")

# 2. FunciÃ³n para leer archivos de TEST
# (Es igual a la de train, pero apuntando a la carpeta 'test')
def leer_test(article_id):
    id_str = str(article_id).zfill(4)
    ruta_carpeta = os.path.join(base_dir, 'test', f'article_{id_str}')
    
    t1, t2 = "", ""
    # Leer file_1
    if os.path.exists(os.path.join(ruta_carpeta, 'file_1.txt')):
        with open(os.path.join(ruta_carpeta, 'file_1.txt'), 'r', encoding='utf-8', errors='replace') as f:
            t1 = f.read()
    # Leer file_2
    if os.path.exists(os.path.join(ruta_carpeta, 'file_2.txt')):
        with open(os.path.join(ruta_carpeta, 'file_2.txt'), 'r', encoding='utf-8', errors='replace') as f:
            t2 = f.read()
    return pd.Series([t1, t2])

# 3. Leemos los textos
print("â�³ Leyendo textos de prueba...")
df_test[['text_1', 'text_2']] = df_test['id'].apply(leer_test)

# 4. Tokenizar para Test
# No necesitamos etiquetas aquÃ­ (labels) porque es lo que vamos a predecir
def encode_test(dataframe):
    encodings = tokenizer(
        dataframe['text_1'].tolist(), 
        dataframe['text_2'].tolist(), 
        truncation=True, 
        padding=True, 
        max_length=512, 
        return_tensors='tf'
    )
    return dict(encodings)

print("âš™ï¸� Preparando datos para el modelo...")
test_encodings = encode_test(df_test)

# 5. Predecir
print("ğŸ”® Generando predicciones...")
# El modelo nos da probabilidades [prob_clase_0, prob_clase_1]
predictions_logits = model.predict(test_encodings)
predictions_probs = tf.nn.softmax(predictions_logits.logits)
predictions_labels = np.argmax(predictions_probs, axis=1)

# 6. Convertir etiqueta (0/1) al formato de Kaggle (1/2)
# Si el modelo predijo 0 -> Es el Texto 1
# Si el modelo predijo 1 -> Es el Texto 2
df_test['real_text_id'] = predictions_labels + 1

# 7. Guardar Submission
submission = df_test[['id', 'real_text_id']]
submission.to_csv('submission_Deberta_v22.csv', index=False)

print("\nâœ… Â¡LISTO! Archivo 'submission_Deberta_v22.csv' generado.")
print(submission.head())


# --- CELDA MAESTRA: ENTRENAMIENTO CON EL 100% DE DATOS ---
import tf_keras
from transformers import TFDebertaV2ForSequenceClassification

print("ğŸ”¥ ESTRATEGIA: Usando el 100% de los datos (sin validaciÃ³n) y mÃ¡s Ã©pocas.")

# 1. Etiquetas (0 y 1)
df_train['label'] = df_train['real_text_id'] - 1

# 2. Tokenizar TODO el dataset (sin train_test_split)
def crear_dataset_completo(dataframe):
    encodings = tokenizer(
        dataframe['text_1'].tolist(), 
        dataframe['text_2'].tolist(), 
        truncation=True, 
        padding=True, 
        max_length=512, 
        return_tensors='tf'
    )
    dataset = tf.data.Dataset.from_tensor_slices((
        dict(encodings), 
        dataframe['label'].tolist()
    ))
    # Batch pequeÃ±o (4) para que actualice los pesos mÃ¡s veces
    return dataset.shuffle(100).batch(4)

full_dataset = crear_dataset_completo(df_train)
print(f"ğŸ“Š Entrenando con {len(df_train)} ejemplos (100% del dataset).")

# 3. Cargar Modelo Limpio
model = TFDebertaV2ForSequenceClassification.from_pretrained('microsoft/deberta-v3-small', num_labels=2)

# 4. Compilar
optimizer = tf_keras.optimizers.Adam(learning_rate=1e-5) # Bajamos un poco el LR para ser mÃ¡s precisos
loss = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# 5. Entrenar MÃ�S tiempo (12 Ã©pocas)
# Al no haber validaciÃ³n, el 'loss' deberÃ­a bajar mucho, casi a 0.
history = model.fit(
    full_dataset,
    epochs=12  # Aumentamos de 4 a 12
)

print("âœ… Modelo entrenado a fondo.")


# --- CELDA 8: GENERACIÃ“N DE PREDICCIONES (SUBMISSION v23) ---
import numpy as np

# 1. Identificar los IDs de la carpeta TEST
# Como no tenemos un 'test.csv', escaneamos las carpetas 'article_XXXX'
test_dir = os.path.join(base_dir, 'test')
test_ids = []

print(f"ğŸ“‚ Escaneando carpeta de prueba: {test_dir}")
for folder_name in os.listdir(test_dir):
    if folder_name.startswith('article_'):
        # Extraemos el nÃºmero (ej: article_0192 -> 192)
        try:
            art_id = int(folder_name.split('_')[1])
            test_ids.append(art_id)
        except:
            pass

# Ordenamos los IDs para que quede prolijo
test_ids.sort()
df_test = pd.DataFrame({'id': test_ids})
print(f"ğŸ“Š Encontrados {len(df_test)} artÃ­culos para predecir.")

# 2. FunciÃ³n para leer archivos de TEST
# (Es igual a la de train, pero apuntando a la carpeta 'test')
def leer_test(article_id):
    id_str = str(article_id).zfill(4)
    ruta_carpeta = os.path.join(base_dir, 'test', f'article_{id_str}')
    
    t1, t2 = "", ""
    # Leer file_1
    if os.path.exists(os.path.join(ruta_carpeta, 'file_1.txt')):
        with open(os.path.join(ruta_carpeta, 'file_1.txt'), 'r', encoding='utf-8', errors='replace') as f:
            t1 = f.read()
    # Leer file_2
    if os.path.exists(os.path.join(ruta_carpeta, 'file_2.txt')):
        with open(os.path.join(ruta_carpeta, 'file_2.txt'), 'r', encoding='utf-8', errors='replace') as f:
            t2 = f.read()
    return pd.Series([t1, t2])

# 3. Leemos los textos
print("â�³ Leyendo textos de prueba...")
df_test[['text_1', 'text_2']] = df_test['id'].apply(leer_test)

# 4. Tokenizar para Test
# No necesitamos etiquetas aquÃ­ (labels) porque es lo que vamos a predecir
def encode_test(dataframe):
    encodings = tokenizer(
        dataframe['text_1'].tolist(), 
        dataframe['text_2'].tolist(), 
        truncation=True, 
        padding=True, 
        max_length=512, 
        return_tensors='tf'
    )
    return dict(encodings)

print("âš™ï¸� Preparando datos para el modelo...")
test_encodings = encode_test(df_test)

# 5. Predecir
print("ğŸ”® Generando predicciones...")
# El modelo nos da probabilidades [prob_clase_0, prob_clase_1]
predictions_logits = model.predict(test_encodings)
predictions_probs = tf.nn.softmax(predictions_logits.logits)
predictions_labels = np.argmax(predictions_probs, axis=1)

# 6. Convertir etiqueta (0/1) al formato de Kaggle (1/2)
# Si el modelo predijo 0 -> Es el Texto 1
# Si el modelo predijo 1 -> Es el Texto 2
df_test['real_text_id'] = predictions_labels + 1

# 7. Guardar Submission
submission = df_test[['id', 'real_text_id']]
submission.to_csv('submission_Deberta_V23.csv', index=False)

print("\nâœ… Â¡LISTO! Archivo 'submission_Deberta_v23.csv' generado.")
print(submission.head())


# Cuenta cuÃ¡ntos 1 y cuÃ¡ntos 2 hay en tus predicciones
print(submission['real_text_id'].value_counts())


# --- CELDA 9: DATA AUGMENTATION (MULTIPLICAR DATOS) ---
import random
import nltk
from nltk.corpus import wordnet

# 1. Descargar diccionario de sinÃ³nimos
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True) # Necesario para contexto

# 2. FunciÃ³n para reemplazar palabras por sinÃ³nimos
def sinonimo_augment(text, n_cambios=3):
    words = text.split()
    new_words = words.copy()
    
    # Intentamos cambiar n palabras aleatorias
    indexes = list(range(len(words)))
    random.shuffle(indexes)
    
    cambios_hechos = 0
    for idx in indexes:
        if cambios_hechos >= n_cambios:
            break
            
        word = words[idx]
        synonyms = []
        
        # Buscar sinÃ³nimos en WordNet
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word and '_' not in lemma.name():
                    synonyms.append(lemma.name())
        
        if len(synonyms) > 0:
            # Reemplazar por un sinÃ³nimo al azar
            new_words[idx] = random.choice(synonyms)
            cambios_hechos += 1
            
    return ' '.join(new_words)

print("ğŸ§ª Probando aumentaciÃ³n...")
ejemplo = "The quick brown fox jumps over the lazy dog"
print(f"Original: {ejemplo}")
print(f"Aumentado: {sinonimo_augment(ejemplo)}")

# 3. Â¡Multiplicar el Dataset!
print("\nğŸš€ Generando nuevos datos sintÃ©ticos...")
new_rows = []

# Recorremos cada fila original
for index, row in df_train.iterrows():
    # 1. Agregamos el original
    new_rows.append(row)
    
    # 2. Generamos VariaciÃ³n A (modificando texto 1 y texto 2)
    # Copiamos la fila
    row_aug1 = row.copy()
    row_aug1['text_1'] = sinonimo_augment(row['text_1'])
    row_aug1['text_2'] = sinonimo_augment(row['text_2'])
    new_rows.append(row_aug1)
    
    # 3. Generamos VariaciÃ³n B (modificando un poco mÃ¡s)
    row_aug2 = row.copy()
    row_aug2['text_1'] = sinonimo_augment(row['text_1'], n_cambios=5)
    row_aug2['text_2'] = sinonimo_augment(row['text_2'], n_cambios=5)
    new_rows.append(row_aug2)

# Crear el nuevo DataFrame aumentado
df_train_aug = pd.DataFrame(new_rows).reset_index(drop=True)

print(f"ğŸ“Š Dataset Original: {len(df_train)} filas")
print(f"ğŸ“ˆ Dataset Aumentado: {len(df_train_aug)} filas")
print("âœ… Datos listos para re-entrenar.")


# --- CELDA 10 MAESTRA: ENTRENAMIENTO CON DATOS AUMENTADOS ---
import tf_keras
from transformers import TFDebertaV2ForSequenceClassification

print("ğŸ”¥ ESTRATEGIA: Usando  los datos aumentados (sin validaciÃ³n) y mÃ¡s Ã©pocas.")

# 1. Etiquetas (0 y 1)
df_train_aug['label'] = df_train_aug['real_text_id'] - 1

# 2. Tokenizar TODO el dataset (sin train_test_split)
def crear_dataset_completo(dataframe):
    encodings = tokenizer(
        dataframe['text_1'].tolist(), 
        dataframe['text_2'].tolist(), 
        truncation=True, 
        padding=True, 
        max_length=512, 
        return_tensors='tf'
    )
    dataset = tf.data.Dataset.from_tensor_slices((
        dict(encodings), 
        dataframe['label'].tolist()
    ))
    # Batch pequeÃ±o (4) para que actualice los pesos mÃ¡s veces
    return dataset.shuffle(100).batch(4)

full_dataset = crear_dataset_completo(df_train_aug)
print(f"ğŸ“Š Entrenando con {len(df_train_aug)}  (datos aumentado).")

# 3. Cargar Modelo Limpio
model = TFDebertaV2ForSequenceClassification.from_pretrained('microsoft/deberta-v3-small', num_labels=2)

# 4. Compilar
optimizer = tf_keras.optimizers.Adam(learning_rate=1e-5) # Bajamos un poco el LR para ser mÃ¡s precisos
loss = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# 5. Entrenar MÃ�S tiempo (12 Ã©pocas)
# Al no haber validaciÃ³n, el 'loss' deberÃ­a bajar mucho, casi a 0.
history = model.fit(
    full_dataset,
    epochs=12  # Aumentamos de 4 a 12
)

print("âœ… Modelo entrenado a fondo.")


# --- CELDA 11: GENERACIÃ“N DE PREDICCIONES (SUBMISSION) ---
import numpy as np

# 1. Identificar los IDs de la carpeta TEST
# Como no tenemos un 'test.csv', escaneamos las carpetas 'article_XXXX'
test_dir = os.path.join(base_dir, 'test')
test_ids = []

print(f"ğŸ“‚ Escaneando carpeta de prueba: {test_dir}")
for folder_name in os.listdir(test_dir):
    if folder_name.startswith('article_'):
        # Extraemos el nÃºmero (ej: article_0192 -> 192)
        try:
            art_id = int(folder_name.split('_')[1])
            test_ids.append(art_id)
        except:
            pass

# Ordenamos los IDs para que quede prolijo
test_ids.sort()
df_test = pd.DataFrame({'id': test_ids})
print(f"ğŸ“Š Encontrados {len(df_test)} artÃ­culos para predecir.")

# 2. FunciÃ³n para leer archivos de TEST
# (Es igual a la de train, pero apuntando a la carpeta 'test')
def leer_test(article_id):
    id_str = str(article_id).zfill(4)
    ruta_carpeta = os.path.join(base_dir, 'test', f'article_{id_str}')
    
    t1, t2 = "", ""
    # Leer file_1
    if os.path.exists(os.path.join(ruta_carpeta, 'file_1.txt')):
        with open(os.path.join(ruta_carpeta, 'file_1.txt'), 'r', encoding='utf-8', errors='replace') as f:
            t1 = f.read()
    # Leer file_2
    if os.path.exists(os.path.join(ruta_carpeta, 'file_2.txt')):
        with open(os.path.join(ruta_carpeta, 'file_2.txt'), 'r', encoding='utf-8', errors='replace') as f:
            t2 = f.read()
    return pd.Series([t1, t2])

# 3. Leemos los textos
print("â�³ Leyendo textos de prueba...")
df_test[['text_1', 'text_2']] = df_test['id'].apply(leer_test)

# 4. Tokenizar para Test
# No necesitamos etiquetas aquÃ­ (labels) porque es lo que vamos a predecir
def encode_test(dataframe):
    encodings = tokenizer(
        dataframe['text_1'].tolist(), 
        dataframe['text_2'].tolist(), 
        truncation=True, 
        padding=True, 
        max_length=512, 
        return_tensors='tf'
    )
    return dict(encodings)

print("âš™ï¸� Preparando datos para el modelo...")
test_encodings = encode_test(df_test)

# 5. Predecir
print("ğŸ”® Generando predicciones...")
# El modelo nos da probabilidades [prob_clase_0, prob_clase_1]
predictions_logits = model.predict(test_encodings)
predictions_probs = tf.nn.softmax(predictions_logits.logits)
predictions_labels = np.argmax(predictions_probs, axis=1)

# 6. Convertir etiqueta (0/1) al formato de Kaggle (1/2)
# Si el modelo predijo 0 -> Es el Texto 1
# Si el modelo predijo 1 -> Es el Texto 2
df_test['real_text_id'] = predictions_labels + 1

# 7. Guardar Submission
submission = df_test[['id', 'real_text_id']]
submission.to_csv('submission_Deberta_v24.csv', index=False)

print("\nâœ… Â¡LISTO! Archivo 'submission_Deberta_v24.csv' generado.")
print(submission.head())


# Cuenta cuÃ¡ntos 1 y cuÃ¡ntos 2 hay en tus predicciones
print(submission['real_text_id'].value_counts())

