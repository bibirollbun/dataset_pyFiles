# 1. Procesamiento de audios y generaciÃ³n de imÃ¡genes RGB


import os
import numpy as np
import librosa
import cv2
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer

# Especies objetivo
especies_objetivo = [
    'bkcchi', 'grycat', 'blujay', 'bobfly1', 'hofwoo1', 'sonspa', 'amegfi', 'grekis', 'rewbla',
    'rucwar', 'brnjay', 'chswar', 'swaspa', 'norcar', 'haiwoo', 'reevir1', 'obnthr1', 'rubwre1',
    'orcpar', 'runwre1', 'comgra', 'rtlhum', 'sthwoo1', 'plawre1', 'yebsap', 'ovenbi1', 'orfpar',
    'norwat', 'comyel', 'belkin1', 'bucmot2', 'dowwoo', 'eastow', 'eawpew', 'grhcha1',
    'amecro', 'amerob', 'balori', 'bucmot2', 'cangoo', 'clcrob', 'crfpar', 'norfli', 'rebwoo',
    'whcpar', 'woothr', 'yehcar1', 'melbla1', 'gockin', 'nocall'  # incluimos nocall
]

clases = sorted(especies_objetivo)
class_to_idx = {c: i for i, c in enumerate(clases)}

# ParÃ¡metros
SPEC_SHAPE = (224, 224) # Nuevo tamaÃ±o de espectograma
SAMPLE_RATE = 32000
SIGNAL_LENGTH = 5  # segundos

def mono_to_color(X, eps=1e-6):
    mean = X.mean()
    std = X.std()
    X = (X - mean) / (std + eps)
    _min, _max = X.min(), X.max()
    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)
    return np.stack([V, V, V], axis=-1)

def audio_to_rgb_melspec(file_path):
    wav, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=SIGNAL_LENGTH, mono=True)
    if len(wav) < SAMPLE_RATE * SIGNAL_LENGTH:
        wav = np.pad(wav, (0, SAMPLE_RATE * SIGNAL_LENGTH - len(wav)))

    melspec = librosa.feature.melspectrogram(
        y=wav,
        sr=SAMPLE_RATE,
        n_mels=SPEC_SHAPE[0],
        fmin=500,
        fmax=12500
    )
    melspec = librosa.power_to_db(melspec).astype(np.float32)
    melspec = cv2.resize(melspec, SPEC_SHAPE)
    return mono_to_color(melspec)

# Carga de datos
X, y = [], []
base_path = "/kaggle/input/birdclef-2021/train_short_audio"

print("Procesando audios y generando imÃ¡genes RGB...")
for especie in tqdm(especies_objetivo):
    especie_path = os.path.join(base_path, especie)
    if not os.path.exists(especie_path): continue

    for archivo in os.listdir(especie_path):
        if archivo.endswith(".ogg"):
            file_path = os.path.join(especie_path, archivo)
            try:
                rgb_spec = audio_to_rgb_melspec(file_path)
                X.append(rgb_spec)
                y.append(class_to_idx[especie])
            except Exception as e:
                print(f"âš ï¸� Error con {file_path}: {e}")

# Convertimos a arrays
X = np.array(X)
y = np.array(y)

print(f"âœ… Datos listos: {X.shape[0]} muestras de {len(clases)} clases.")


# 2. Entrenamiento con EfficientNetB0, RGB y Fine-Tuning

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf

# One-hot encoding de etiquetas
y_cat = to_categorical(y, num_classes=len(clases))

# DivisiÃ³n entrenamiento/validaciÃ³n
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, stratify=y, random_state=42)

print(f" Datos listos: {X_train.shape[0]} entrenamiento / {X_val.shape[0]} validaciÃ³n")

# NormalizaciÃ³n de imÃ¡genes (aÃ±adido para mejorar el entrenamiento)
X_train = X_train / 255.0
X_val = X_val / 255.0

# AumentaciÃ³n de datos (opcional pero recomendado para mejorar generalizaciÃ³n)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# MÃ©tricas personalizadas para F1 score (importante para la competencia)
def f1_metric(y_true, y_pred):
    # Convierte las probabilidades en etiquetas
    y_pred_classes = tf.argmax(y_pred, axis=1)
    y_true_classes = tf.argmax(y_true, axis=1)
    
    # Calcula precisiÃ³n y recall
    precision = tf.reduce_sum(
        tf.cast(tf.logical_and(tf.equal(y_true_classes, y_pred_classes), 
                              tf.not_equal(y_true_classes, 0)), tf.float32)
    ) / (tf.reduce_sum(tf.cast(tf.not_equal(y_pred_classes, 0), tf.float32)) + tf.keras.backend.epsilon())
    
    recall = tf.reduce_sum(
        tf.cast(tf.logical_and(tf.equal(y_true_classes, y_pred_classes), 
                              tf.not_equal(y_true_classes, 0)), tf.float32)
    ) / (tf.reduce_sum(tf.cast(tf.not_equal(y_true_classes, 0), tf.float32)) + tf.keras.backend.epsilon())
    
    # F1 Score
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return f1

# Crear modelo con EfficientNetB0
def build_model(trainable=False, dropout_rate=0.2):
    input_tensor = Input(shape=(224, 224, 3))
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor
    )
    base_model.trainable = trainable

    # AÃ±adir capas superiores
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)  # AÃ±adir dropout para evitar sobreajuste
    output = Dense(len(clases), activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output)
    return model, base_model

# FASE 1: Entrenamiento con capas congeladas (feature extraction)
model, base_model = build_model(trainable=False)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', f1_metric]
)

# Callbacks
callbacks_phase1 = [
    EarlyStopping(monitor='val_f1_metric', patience=5, mode='max', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_f1_metric', patience=2, factor=0.5, mode='max'),
]

print("Fase 1: Entrenamiento inicial con capas congeladas")
history_phase1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,  # Menos Ã©pocas para la primera fase
    batch_size=32,
    callbacks=callbacks_phase1,
    verbose=1
)
model.save('/kaggle/working/modelo_efficientnet_rgb_finetuned1.h5')

# FASE 2: Fine-tuning de las Ãºltimas capas
print("Fase 2: Fine-tuning de las Ãºltimas capas")

# Cargar el mejor modelo de la fase 1
model.load_weights('/kaggle/working/modelo_efficientnet_rgb_finetuned1.h5')

# Descongelar las Ãºltimas capas (por ejemplo, los Ãºltimos 20 bloques)
# EfficientNetB0 tiene 236 capas en total
base_model.trainable = True

# Congelar todas las capas excepto las Ãºltimas N
fine_tune_at = len(base_model.layers) - 20  # Ajustar este nÃºmero segÃºn necesites
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Verificar quÃ© capas son entrenables
trainable_layers = [layer.name for layer in base_model.layers if layer.trainable]
print(f"Capas entrenables: {len(trainable_layers)} de {len(base_model.layers)}")
print(f"Primeras 5 capas entrenables: {trainable_layers[:5]}")

# Recompilar el modelo con una tasa de aprendizaje mÃ¡s baja para fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-4),  # Tasa de aprendizaje mÃ¡s baja
    loss='categorical_crossentropy',
    metrics=['accuracy', f1_metric]
)

# Callbacks para la fase de fine-tuning
callbacks_phase2 = [
    EarlyStopping(monitor='val_f1_metric', patience=7, mode='max', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_f1_metric', patience=3, factor=0.2, mode='max'),
]

# Entrenamiento con fine-tuning
history_phase2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=8,  # MÃ¡s Ã©pocas para fine-tuning
    batch_size=16,  # Batch size mÃ¡s pequeÃ±o para fine-tuning
    callbacks=callbacks_phase2,
    verbose=1
)


## Segunda parte

model.save('/kaggle/working/modelo_efficientnet_rgb_finetuned2.h5')

# Combinar historiales
history = {}
for key in history_phase1.history:
    history[key] = history_phase1.history[key] + history_phase2.history[key]

# Evaluar el modelo final
print("Evaluando modelo final...")
model.load_weights('/kaggle/working/modelo_efficientnet_rgb_finetuned2.h5')  # Cargar el mejor modelo
test_loss, test_acc, test_f1 = model.evaluate(X_val, y_val)
print(f"Rendimiento final: Accuracy = {test_acc:.4f}, F1 Score = {test_f1:.4f}")

# Hacer predicciones y calcular mÃ©tricas adicionales
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

# Calcular F1, precisiÃ³n y recall
f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
precision = precision_score(y_true_classes, y_pred_classes, average='macro')
recall = recall_score(y_true_classes, y_pred_classes, average='macro')

print(f"MÃ©tricas detalladas:")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Guardar modelo entrenado y su historial
model.save('/kaggle/working/modelo_efficientnet_rgb_finetuned2.h5')
with open('/kaggle/working/history_entrenamiento_finetuned.pkl', 'wb') as f:
    pickle.dump(history, f)

# Visualizar el entrenamiento
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Accuracy del modelo')
plt.ylabel('Accuracy')
plt.xlabel('Ã‰poca')
plt.legend(['Entrenamiento', 'ValidaciÃ³n'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history['f1_metric'])
plt.plot(history['val_f1_metric'])
plt.title('F1 Score del modelo')
plt.ylabel('F1 Score')
plt.xlabel('Ã‰poca')
plt.legend(['Entrenamiento', 'ValidaciÃ³n'], loc='lower right')

plt.tight_layout()
plt.savefig('/kaggle/working/training_metrics.png')
plt.show()

print("âœ… Modelo entrenado y guardado con EfficientNetB0 y fine-tuning.")


## Segunda parte

model.save('/kaggle/working/modelo_efficientnet_rgb_finetuned2.h5')

# Combinar historiales
history = {}
for key in history_phase1.history:
    history[key] = history_phase1.history[key] + history_phase2.history[key]

# Evaluar el modelo final
print("Evaluando modelo final...")
model.load_weights('/kaggle/working/modelo_efficientnet_rgb_finetuned2.h5')  # Cargar el mejor modelo
test_loss, test_acc, test_f1 = model.evaluate(X_val, y_val)
print(f"Rendimiento final: Accuracy = {test_acc:.4f}, F1 Score = {test_f1:.4f}")

# Hacer predicciones y calcular mÃ©tricas adicionales
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

# Calcular F1, precisiÃ³n y recall
f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
precision = precision_score(y_true_classes, y_pred_classes, average='macro')
recall = recall_score(y_true_classes, y_pred_classes, average='macro')

print(f"MÃ©tricas detalladas:")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Guardar modelo entrenado y su historial
model.save('/kaggle/working/modelo_efficientnet_rgb_finetuned2.h5')
with open('/kaggle/working/history_entrenamiento_finetuned.pkl', 'wb') as f:
    pickle.dump(history, f)

# Visualizar el entrenamiento
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Accuracy del modelo')
plt.ylabel('Accuracy')
plt.xlabel('Ã‰poca')
plt.legend(['Entrenamiento', 'ValidaciÃ³n'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history['f1_metric'])
plt.plot(history['val_f1_metric'])
plt.title('F1 Score del modelo')
plt.ylabel('F1 Score')
plt.xlabel('Ã‰poca')
plt.legend(['Entrenamiento', 'ValidaciÃ³n'], loc='lower right')

plt.tight_layout()
plt.savefig('/kaggle/working/training_metrics.png')
plt.show()

print("âœ… Modelo entrenado y guardado con EfficientNetB0 y fine-tuning.")


# PARTE A

# EvaluaciÃ³n del modelo RGB con mÃ©tricas F1 Score y exportaciÃ³n a CSV

import os
import numpy as np
import pandas as pd
import librosa
import cv2
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, multilabel_confusion_matrix

# === ConfiguraciÃ³n ===
SAMPLE_RATE = 32000
SIGNAL_LENGTH = 5  # en segundos
SPEC_SHAPE = (224, 224)
THRESHOLD = 0.1

# === Clases ===
especies_objetivo = [
    'bkcchi', 'grycat', 'blujay', 'bobfly1', 'hofwoo1', 'sonspa', 'amegfi', 'grekis', 'rewbla',
    'rucwar', 'brnjay', 'chswar', 'swaspa', 'norcar', 'haiwoo', 'reevir1', 'obnthr1', 'rubwre1',
    'orcpar', 'runwre1', 'comgra', 'rtlhum', 'sthwoo1', 'plawre1', 'yebsap', 'ovenbi1', 'orfpar',
    'norwat', 'comyel', 'belkin1', 'bucmot2', 'dowwoo', 'eastow', 'eawpew', 'grhcha1',
    'amecro', 'amerob', 'balori', 'bucmot2', 'cangoo', 'clcrob', 'crfpar', 'norfli', 'rebwoo',
    'whcpar', 'woothr', 'yehcar1', 'melbla1', 'gockin', 'nocall'
]
clases = sorted(especies_objetivo)
label_to_idx = {l: i for i, l in enumerate(clases)}

# === Funciones auxiliares ===
def mono_to_color(X, eps=1e-6):
    mean = X.mean()
    std = X.std()
    X = (X - mean) / (std + eps)
    _min, _max = X.min(), X.max()
    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)
    return np.stack([V, V, V], axis=-1)

def predict_fragment(chunk, model):
    spec = librosa.feature.melspectrogram(y=chunk, sr=SAMPLE_RATE, n_mels=SPEC_SHAPE[0], fmin=500, fmax=12500)
    spec = librosa.power_to_db(spec).astype(np.float32)
    spec = cv2.resize(spec, SPEC_SHAPE)
    rgb = mono_to_color(spec)
    rgb = np.expand_dims(rgb, axis=0)
    return model.predict(rgb, verbose=0)[0]

def compute_metrics_per_example(true_labels, pred_labels):
    """
    Calcula mÃ©tricas para un Ãºnico ejemplo segÃºn las fÃ³rmulas definidas:
    - Precision = TP / (TP + FP)
    - Recall = TP / (TP + FN)
    - F1 = 2 * (Precision * Recall) / (Precision + Recall)
    - Accuracy = (TP + TN) / (TP + TN + FP + FN)
    - Specificity = TN / (TN + FP)
    """
    true_set = set(true_labels)
    pred_set = set(pred_labels)
    
    # Para clasificaciÃ³n multilabel
    tp = len(true_set & pred_set)  # Verdaderos positivos: especies correctamente predichas
    fp = len(pred_set - true_set)  # Falsos positivos: especies predichas pero no presentes
    fn = len(true_set - pred_set)  # Falsos negativos: especies presentes pero no predichas
    
    # En clasificaciÃ³n multilabel, para calcular TN necesitamos considerar todas las clases
    # TN son las especies que correctamente NO se predicen porque NO estÃ¡n presentes
    all_classes = set(clases)
    tn = len(all_classes - true_set - pred_set)  # Verdaderos negativos
    
    # CÃ¡lculo de mÃ©tricas segÃºn las fÃ³rmulas
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }

def get_metrics_from_binary_matrices(y_true_bin, y_pred_bin):
    """
    Calcula mÃ©tricas globales a partir de matrices binarias de verdad y predicciÃ³n
    """
    # MÃ©tricas usando sklearn para el micro-promedio (global)
    precision = precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    
    # Para la especificidad, calculamos el promedio de las matrices de confusiÃ³n por clase
    mcm = multilabel_confusion_matrix(y_true_bin, y_pred_bin)
    specificity = np.mean([
        m[0, 0] / (m[0, 0] + m[0, 1]) if (m[0, 0] + m[0, 1]) > 0 else 0
        for m in mcm
    ])
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity
    }

def multilabel_to_binary(y_list, label_map):
    """
    Convierte una lista de etiquetas multilabel a su representaciÃ³n binaria
    """
    bin_labels = []
    for labels in y_list:
        row = [0] * len(label_map)
        for l in labels:
            l = l.strip()
            idx = label_map.get(l, None)
            if idx is not None and idx < len(row):
                row[idx] = 1
        bin_labels.append(row)
    return np.array(bin_labels)

# === Cargar modelo entrenado ===
print("ğŸ”„ Cargando modelo...")
model_path = "/kaggle/working/modelo_efficientnet_rgb_finetuned2.h5"
model = load_model(model_path)
print("âœ… Modelo cargado correctamente")

# === Datos ===
print("ğŸ”„ Cargando datos de etiquetas...")
gt_df = pd.read_csv("/kaggle/input/birdclef-2021/train_soundscape_labels.csv")
gt_df['birds'] = gt_df['birds'].fillna("nocall")
gt_dict = dict(zip(gt_df['row_id'], gt_df['birds']))
print(f"âœ… Cargadas {len(gt_df)} etiquetas")

# === EvaluaciÃ³n por audio ===
soundscape_dir = "/kaggle/input/birdclef-2021/train_soundscapes/"
metricas_por_audio = []

# === Para guardar las predicciones detalladas ===
all_predictions = []

# === Para mÃ©tricas globales ===
all_tp = 0
all_fp = 0
all_fn = 0
all_tn = 0

print("ğŸ”� Evaluando soundscapes...\n")

for fname in sorted(os.listdir(soundscape_dir)):
    if not fname.endswith(".ogg"): continue
    print(f"ğŸ”„ Procesando {fname}...")
    base, site = fname.split('_')[0], fname.split('_')[1]
    y, _ = librosa.load(os.path.join(soundscape_dir, fname), sr=SAMPLE_RATE)

    y_true_rows = []
    y_pred_rows = []
    
    # Para mÃ©tricas acumuladas por audio
    audio_tp = 0
    audio_fp = 0 
    audio_fn = 0
    audio_tn = 0
    
    metricas_por_fragmento = []

    for i in range(0, len(y), SIGNAL_LENGTH * SAMPLE_RATE):
        chunk = y[i:i + SIGNAL_LENGTH * SAMPLE_RATE]
        if len(chunk) < SIGNAL_LENGTH * SAMPLE_RATE:
            continue

        seconds = (i // (SAMPLE_RATE * SIGNAL_LENGTH) + 1) * SIGNAL_LENGTH
        row_id = f"{base}_{site}_{seconds}"
        
        # Predecir especies
        pred_probs = predict_fragment(chunk, model)
        
        # Obtener top 3 predicciones para el anÃ¡lisis
        top_indices = np.argsort(pred_probs)[-3:][::-1]  # Top 3 predicciones
        top_birds = [(clases[idx], pred_probs[idx]) for idx in top_indices]
        
        # Obtener etiquetas reales y predicciones basadas en umbral
        true_birds = gt_dict.get(row_id, "nocall").split()
        pred_labels = [clases[j] for j, p in enumerate(pred_probs) if p > THRESHOLD]
        if not pred_labels:
            pred_labels = ["nocall"]
            
        # Calcular mÃ©tricas por fragmento individualmente
        metrics = compute_metrics_per_example(true_birds, pred_labels)
        
        # Acumular mÃ©tricas para este audio
        audio_tp += metrics["tp"]
        audio_fp += metrics["fp"]
        audio_fn += metrics["fn"]
        audio_tn += metrics["tn"]
        
        # Acumular mÃ©tricas globales
        all_tp += metrics["tp"]
        all_fp += metrics["fp"]
        all_fn += metrics["fn"]
        all_tn += metrics["tn"]
        
        # Guardar datos para el anÃ¡lisis detallado
        prediction_data = {
            "row_id": row_id,
            "birds": " ".join(true_birds),
            "prediction": " ".join(pred_labels),
            "pred1": top_birds[0][0],
            "score1": top_birds[0][1],
            "pred2": top_birds[1][0],
            "score2": top_birds[1][1],
            "pred3": top_birds[2][0],
            "score3": top_birds[2][1],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "specificity": metrics["specificity"]
        }
        all_predictions.append(prediction_data)
        metricas_por_fragmento.append(prediction_data)
        
        # Guardar para la evaluaciÃ³n global
        y_pred_rows.append(pred_labels)
        y_true_rows.append(true_birds)

    # === MÃ©tricas por audio ===
    if y_true_rows:
        # Calcular mÃ©tricas usando matrices binarias (enfoque sklearn)
        y_true_bin = multilabel_to_binary(y_true_rows, label_to_idx)
        y_pred_bin = multilabel_to_binary(y_pred_rows, label_to_idx)
        metrics_from_bin = get_metrics_from_binary_matrices(y_true_bin, y_pred_bin)
        
        # Calcular mÃ©tricas manualmente desde los totales acumulados
        total = audio_tp + audio_fp + audio_fn + audio_tn
        precision_manual = audio_tp / (audio_tp + audio_fp) if (audio_tp + audio_fp) > 0 else 0
        recall_manual = audio_tp / (audio_tp + audio_fn) if (audio_tp + audio_fn) > 0 else 0
        f1_manual = 2 * (precision_manual * recall_manual) / (precision_manual + recall_manual) if (precision_manual + recall_manual) > 0 else 0
        accuracy_manual = (audio_tp + audio_tn) / total if total > 0 else 0
        specificity_manual = audio_tn / (audio_tn + audio_fp) if (audio_tn + audio_fp) > 0 else 0
        
        # Usar las mÃ©tricas calculadas con sklearn como las "oficiales"
        fila = {
            "Audio": fname,
            "PrecisiÃ³n": metrics_from_bin["precision"],
            "Sensibilidad": metrics_from_bin["recall"],
            "F1 Score": metrics_from_bin["f1"],
            "PrecisiÃ³n global": metrics_from_bin["accuracy"],
            "Especificidad": metrics_from_bin["specificity"],
            # Incluir tambiÃ©n las mÃ©tricas calculadas manualmente para comparaciÃ³n
            "PrecisiÃ³n_manual": precision_manual,
            "Sensibilidad_manual": recall_manual,
            "F1 Score_manual": f1_manual,
            "PrecisiÃ³n global_manual": accuracy_manual,
            "Especificidad_manual": specificity_manual,
            # Contadores
            "TP": audio_tp,
            "FP": audio_fp,
            "FN": audio_fn,
            "TN": audio_tn,
            "Fragmentos": len(y_true_rows)
        }
        metricas_por_audio.append(fila)

        print(f"ğŸ�§ {fname}:")
        print(f"  - Fragmentos evaluados: {len(y_true_rows)}")
        print(f"  - PrecisiÃ³n:         {metrics_from_bin['precision']:.4f}")
        print(f"  - Sensibilidad:      {metrics_from_bin['recall']:.4f}")
        print(f"  - F1 Score:          {metrics_from_bin['f1']:.4f}")
        print(f"  - PrecisiÃ³n global:  {metrics_from_bin['accuracy']:.4f}")
        print(f"  - Especificidad:     {metrics_from_bin['specificity']:.4f}")
        print("")
        
        # Guardar mÃ©tricas por fragmento para este audio
        audio_fragments_df = pd.DataFrame(metricas_por_fragmento)
        audio_name = fname.replace(".ogg", "")
        audio_fragments_df.to_csv(f"/kaggle/working/metricas_{audio_name}.csv", index=False)
        print(f"ğŸ’¾ MÃ©tricas por fragmento guardadas en 'metricas_{audio_name}.csv'")

# === Calcular mÃ©tricas globales ===
total_global = all_tp + all_fp + all_fn + all_tn
precision_global = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
recall_global = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
f1_global = 2 * (precision_global * recall_global) / (precision_global + recall_global) if (precision_global + recall_global) > 0 else 0
accuracy_global = (all_tp + all_tn) / total_global if total_global > 0 else 0
specificity_global = all_tn / (all_tn + all_fp) if (all_tn + all_fp) > 0 else 0

print("\nğŸ“Š MÃ‰TRICAS GLOBALES:")
print(f"  - PrecisiÃ³n:         {precision_global:.4f}")
print(f"  - Sensibilidad:      {recall_global:.4f}")
print(f"  - F1 Score:          {f1_global:.4f}")
print(f"  - PrecisiÃ³n global:  {accuracy_global:.4f}")
print(f"  - Especificidad:     {specificity_global:.4f}")
print(f"  - Total TP: {all_tp}, FP: {all_fp}, FN: {all_fn}, TN: {all_tn}")

# === Mostrar tabla final ===
tabla_metricas = pd.DataFrame(metricas_por_audio)
print("\nğŸ“Š Tabla completa de mÃ©tricas por audio:")
print(tabla_metricas[["Audio", "PrecisiÃ³n", "Sensibilidad", "F1 Score", "PrecisiÃ³n global", "Especificidad", "Fragmentos"]])

# === Guardar resultados en archivos CSV ===
tabla_metricas.to_csv("/kaggle/working/metricas_por_audio.csv", index=False)
print("\nğŸ’¾ MÃ©tricas por audio guardadas en 'metricas_por_audio.csv'")

# === Guardar todas las predicciones detalladas ===
predictions_df = pd.DataFrame(all_predictions)
predictions_df.to_csv("/kaggle/working/predicciones_detalladas.csv", index=False)
print("ğŸ’¾ Predicciones detalladas guardadas en 'predicciones_detalladas.csv'")

# === Guardar mÃ©tricas globales ===
metricas_globales = pd.DataFrame([{
    "PrecisiÃ³n": precision_global,
    "Sensibilidad": recall_global,
    "F1 Score": f1_global,
    "PrecisiÃ³n global": accuracy_global,
    "Especificidad": specificity_global,
    "TP": all_tp,
    "FP": all_fp,
    "FN": all_fn,
    "TN": all_tn
}])
metricas_globales.to_csv("/kaggle/working/metricas_globales.csv", index=False)
print("ğŸ’¾ MÃ©tricas globales guardadas en 'metricas_globales.csv'")


# PARTE B
# Mostrar tabla detallada con TODAS las predicciones y estadÃ­sticas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML

# === ConfiguraciÃ³n ===
COLUMNAS_MOSTRAR = 20  # NÃºmero mÃ¡ximo de filas a mostrar en la tabla detallada

print("ğŸ”„ Cargando predicciones detalladas...")
# Cargar los datos de predicciones generados en la Parte A
try:
    predictions_df = pd.read_csv("/kaggle/working/predicciones_detalladas.csv")
    print(f"âœ… Cargadas {len(predictions_df)} predicciones")
except FileNotFoundError:
    print("â�Œ Error: El archivo de predicciones no se encuentra. Ejecute primero la Parte A.")
    predictions_df = pd.DataFrame()

if not predictions_df.empty:
    # === 1. Crear tabla de resultados con formato visual mejorado ===
    print("\nğŸ”� Generando tabla detallada de predicciones...")
    
    # Crear nueva tabla para visualizaciÃ³n
    results_table = []
    
    for idx, row in predictions_df.iterrows():
        row_id = row['row_id']
        true_birds = row['birds']
        pred_birds = row['prediction']
        
        # Verificar si predicciÃ³n es correcta o no
        true_set = set(true_birds.split())
        pred_set = set(pred_birds.split())
        
        # Para cada especie en true y pred
        common = true_set.intersection(pred_set)
        false_pos = pred_set - true_set
        false_neg = true_set - pred_set
        
        # Formato para mostrar coincidencias
        formatted_prediction = ""
        for bird in pred_birds.split():
            if bird in common:
                formatted_prediction += f"<span style='color:green'>{bird}</span> "
            else:
                formatted_prediction += f"<span style='color:red'>{bird}</span> "
        
        # Formato para mostrar etiquetas reales
        formatted_true = ""
        for bird in true_birds.split():
            if bird in common:
                formatted_true += f"<span style='color:green'>{bird}</span> "
            else:
                formatted_true += f"<span style='color:orange'>{bird}</span> "
        
        # Resultado general: Ã©xito completo, parcial o error
        if common == true_set and common == pred_set:
            result = "âœ“"
            result_class = "success"
        elif len(common) > 0:
            result = "Â±"  # Parcialmente correcto
            result_class = "partial"
        else:
            result = "âœ—"
            result_class = "error"
            
        # MÃ©tricas para este fragmento
        precision = row.get('precision', 0)
        recall = row.get('recall', 0)
        f1 = row.get('f1', 0)
            
        # Crear fila para la tabla
        results_table.append({
            "row_id": row_id,
            "ground_truth": formatted_true.strip(),
            "prediction": formatted_prediction.strip(),
            "result": result,
            "result_class": result_class,
            "precision": f"{precision:.4f}",
            "recall": f"{recall:.4f}",
            "f1_score": f"{f1:.4f}",
            "top1": f"{row['pred1']} ({row['score1']:.4f})",
            "top2": f"{row['pred2']} ({row['score2']:.4f})",
            "top3": f"{row['pred3']} ({row['score3']:.4f})"
        })
    
    # Convertir a DataFrame
    results_df = pd.DataFrame(results_table)
    
    # Mostrar tabla interactiva con formato HTML
    css = """
    <style>
    .dataframe {
        font-family: Arial, sans-serif;
        border-collapse: collapse;
        width: 100%;
    }
    .dataframe th {
        background-color: #4b6584;
        color: white;
        text-align: left;
        padding: 8px;
    }
    .dataframe td {
        border: 1px solid #ddd;
        padding: 8px;
    }
    .dataframe tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    .dataframe tr:hover {
        background-color: #ddd;
    }
    .success {
        background-color: #badc58 !important;
        font-weight: bold;
    }
    .partial {
        background-color: #f9ca24 !important;
    }
    .error {
        background-color: #ff7979 !important;
    }
    </style>
    """
    
    # Aplicar formato condicional
    def style_result(val):
        if val == "âœ“":
            return 'background-color: #badc58'
        elif val == "Â±":
            return 'background-color: #f9ca24'
        else:
            return 'background-color: #ff7979'
    
    # Guardar tabla completa sin HTML para exportaciÃ³n
    export_df = results_df.copy()
    # Limpiar las etiquetas HTML para la exportaciÃ³n
    for col in ['ground_truth', 'prediction']:
        export_df[col] = export_df[col].str.replace('<[^<]+?>', '', regex=True)
    
    export_df.to_csv("/kaggle/working/tabla_predicciones_completa.csv", index=False)
    print(f"ğŸ’¾ Tabla completa guardada en 'tabla_predicciones_completa.csv' ({len(export_df)} filas)")
    
    # Mostrar solo las primeras N filas en consola
    print(f"\nğŸ“Š Muestra de las primeras {min(COLUMNAS_MOSTRAR, len(results_df))} predicciones:")
    display_df = results_df.head(COLUMNAS_MOSTRAR).copy()
    display(HTML(css + display_df.to_html(escape=False)))
    
    # === 2. AnÃ¡lisis estadÃ­stico de predicciones ===
    print("\nğŸ“ˆ EstadÃ­sticas generales de predicciÃ³n:")
    
    # Totales por tipo de resultado
    total_predictions = len(results_df)
    correct_predictions = len(results_df[results_df['result'] == "âœ“"])
    partial_predictions = len(results_df[results_df['result'] == "Â±"])
    incorrect_predictions = len(results_df[results_df['result'] == "âœ—"])
    
    print(f"Total de fragmentos evaluados: {total_predictions}")
    print(f"Predicciones completamente correctas: {correct_predictions} ({correct_predictions/total_predictions*100:.2f}%)")
    print(f"Predicciones parcialmente correctas: {partial_predictions} ({partial_predictions/total_predictions*100:.2f}%)")
    print(f"Predicciones incorrectas: {incorrect_predictions} ({incorrect_predictions/total_predictions*100:.2f}%)")
    
    # === 3. AnÃ¡lisis por especie ===
    # Preparar datos
    all_birds = set()
    for birds in predictions_df['birds'].str.split():
        all_birds.update(birds)
    
    # Inicializar estadÃ­sticas por especie
    species_stats = {bird: {"tp": 0, "fp": 0, "fn": 0, "total_true": 0, "total_pred": 0} for bird in all_birds}
    
    # Calcular estadÃ­sticas
    for idx, row in predictions_df.iterrows():
        true_set = set(row['birds'].split())
        pred_set = set(row['prediction'].split())
        
        for bird in all_birds:
            # Si la especie estÃ¡ en verdadero positivo
            if bird in true_set and bird in pred_set:
                species_stats[bird]["tp"] += 1
            # Si es un falso positivo
            elif bird in pred_set:
                species_stats[bird]["fp"] += 1
            # Si es un falso negativo
            elif bird in true_set:
                species_stats[bird]["fn"] += 1
                
            # Contar apariciones totales
            if bird in true_set:
                species_stats[bird]["total_true"] += 1
            if bird in pred_set:
                species_stats[bird]["total_pred"] += 1
    
    # Calcular mÃ©tricas por especie
    species_metrics = []
    for bird, stats in species_stats.items():
        if stats["total_true"] == 0 and stats["total_pred"] == 0:
            continue
            
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        species_metrics.append({
            "Especie": bird,
            "Total_real": stats["total_true"],
            "Total_pred": stats["total_pred"],
            "VP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })
    
    # Crear DataFrame y ordenar por nÃºmero de apariciones
    species_df = pd.DataFrame(species_metrics).sort_values("Total_real", ascending=False)
    
    # Formatear para mejor visualizaciÃ³n
    species_display = species_df.copy()
    for col in ["Precision", "Recall", "F1"]:
        species_display[col] = species_display[col].map("{:.4f}".format)
    
    print("\nğŸ¦œ EstadÃ­sticas por especie de ave (Top 15):")
    print(species_display.head(15).to_string(index=False))
    
    # Guardar resultados
    species_df.to_csv("/kaggle/working/estadisticas_por_especie.csv", index=False)
    print("\nğŸ’¾ EstadÃ­sticas por especie guardadas en 'estadisticas_por_especie.csv'")
    
    # === 4. VisualizaciÃ³n de distribuciÃ³n de F1 Score ===
    if 'f1' in predictions_df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(predictions_df['f1'], bins=20, kde=True)
        plt.title('DistribuciÃ³n de F1 Score por fragmento')
        plt.xlabel('F1 Score')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        plt.savefig('/kaggle/working/distribucion_f1_score.png', dpi=300, bbox_inches='tight')
        print("\nğŸ’¾ GrÃ¡fico de distribuciÃ³n de F1 Score guardado como 'distribucion_f1_score.png'")
        plt.close()
    
    # === 5. GrÃ¡fica de mÃ©tricas por audio ===
    try:
        audio_metrics_df = pd.read_csv("/kaggle/working/metricas_por_audio.csv")
        if not audio_metrics_df.empty:
            # Ordenar por F1 Score para mejor visualizaciÃ³n
            audio_metrics_df = audio_metrics_df.sort_values("F1 Score", ascending=False)
            
            # Extraer nombres cortos de audio para etiquetas
            audio_metrics_df['Audio_short'] = audio_metrics_df['Audio'].apply(lambda x: x.split('.')[0][:10])
            
            # Crear grÃ¡fica de barras agrupadas
            metrics_to_plot = ["PrecisiÃ³n", "Sensibilidad", "F1 Score", "PrecisiÃ³n global", "Especificidad"]
            
            plt.figure(figsize=(14, 8))
            x = np.arange(len(audio_metrics_df))
            width = 0.15
            offsets = np.linspace(-(len(metrics_to_plot)-1)/2*width, (len(metrics_to_plot)-1)/2*width, len(metrics_to_plot))
            
            for i, metric in enumerate(metrics_to_plot):
                plt.bar(x + offsets[i], audio_metrics_df[metric], width=width, label=metric)
            
            plt.xlabel('Audio')
            plt.ylabel('Valor')
            plt.title('MÃ©tricas por archivo de audio')
            plt.xticks(x, audio_metrics_df['Audio_short'], rotation=45, ha='right')
            plt.ylim(0, 1.05)
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('/kaggle/working/metricas_por_audio.png', dpi=300, bbox_inches='tight')
            print("\nğŸ’¾ GrÃ¡fico de mÃ©tricas por audio guardado como 'metricas_por_audio.png'")
            plt.close()
            
            # === 6. GrÃ¡fica comparativa de F1 Score vs PrecisiÃ³n global ===
            plt.figure(figsize=(10, 6))
            plt.scatter(audio_metrics_df['F1 Score'], audio_metrics_df['PrecisiÃ³n global'], 
                      s=80, alpha=0.7, c=audio_metrics_df['Fragmentos'], cmap='viridis')
            
            # AÃ±adir etiquetas a los puntos
            for i, row in audio_metrics_df.iterrows():
                plt.annotate(row['Audio_short'], 
                             (row['F1 Score'], row['PrecisiÃ³n global']),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8)
            
            plt.colorbar(label='NÃºmero de fragmentos')
            plt.xlabel('F1 Score')
            plt.ylabel('PrecisiÃ³n global (Accuracy)')
            plt.title('RelaciÃ³n entre F1 Score y PrecisiÃ³n Global por Audio')
            plt.grid(True, alpha=0.3)
            plt.savefig('/kaggle/working/f1_vs_accuracy.png', dpi=300, bbox_inches='tight')
            print("\nğŸ’¾ GrÃ¡fico de F1 vs PrecisiÃ³n guardado como 'f1_vs_accuracy.png'")
            plt.close()
            
    except Exception as e:
        print(f"\nâš ï¸� No se pudieron generar los grÃ¡ficos por audio: {e}")
    
    # === 7. AnÃ¡lisis de confusiÃ³n por clase ===
    # Preparar matriz de confusiÃ³n simplificada: para cada clase, Â¿cuÃ¡ntas veces se confunde con otra?
    confusion = {}
    
    # Inicializar contadores de confusiÃ³n
    for bird in all_birds:
        confusion[bird] = {"TP": 0}  # Inicializamos con verdaderos positivos
    
    # Analizar cada predicciÃ³n
    for idx, row in predictions_df.iterrows():
        true_birds = set(row['birds'].split())
        pred_birds = set(row['prediction'].split())
        
        # Para cada ave real, ver quÃ© se predijo errÃ³neamente
        for true_bird in true_birds:
            if true_bird in pred_birds:
                confusion[true_bird]["TP"] += 1
            else:
                # Si el ave real no fue predicha, anotamos quÃ© se predijo en su lugar
                for wrong_bird in pred_birds:
                    if wrong_bird not in true_birds:  # Solo las predicciones incorrectas
                        if wrong_bird not in confusion[true_bird]:
                            confusion[true_bird][wrong_bird] = 0
                        confusion[true_bird][wrong_bird] += 1
    
    # Crear una tabla de las aves mÃ¡s confundidas (top 15)
    confusions_list = []
    
    for bird, confusions in confusion.items():
        if "TP" not in confusions or confusions["TP"] == 0:
            continue  # Saltamos aves sin predicciones correctas
        
        # Eliminar TP y ordenar el resto por frecuencia
        del confusions["TP"]
        if not confusions:
            continue  # Si no hay confusiones, continuamos
        
        top_confusion = sorted(confusions.items(), key=lambda x: x[1], reverse=True)
        
        # Tomar solo el ave con la que mÃ¡s se confunde
        if top_confusion:
            confused_with, count = top_confusion[0]
            total_real = species_stats[bird]["total_true"]
            confusions_list.append({
                "Especie": bird,
                "Total_real": total_real,
                "Confundida_con": confused_with,
                "Veces_confundida": count,
                "Porcentaje": count / total_real * 100 if total_real > 0 else 0
            })
    
    # Si hay confusiones, mostrarlas
    if confusions_list:
        confusion_df = pd.DataFrame(confusions_list).sort_values("Veces_confundida", ascending=False)
        confusion_df["Porcentaje"] = confusion_df["Porcentaje"].map("{:.2f}%".format)
        
        print("\nğŸ”„ Top confusiones entre especies:")
        print(confusion_df.head(15).to_string(index=False))
        
        # Guardar resultados
        confusion_df.to_csv("/kaggle/working/confusiones_especies.csv", index=False)
        print("\nğŸ’¾ AnÃ¡lisis de confusiones guardado en 'confusiones_especies.csv'")
    
    # === 8. Resumen final de mÃ©tricas ===
    try:
        metricas_globales = pd.read_csv("/kaggle/working/metricas_globales.csv")
        
        print("\nğŸ“Š RESUMEN DE MÃ‰TRICAS GLOBALES:")
        for col in metricas_globales.columns:
            if col in ["TP", "FP", "FN", "TN"]:
                print(f"  - {col}: {metricas_globales[col].values[0]}")
            else:
                print(f"  - {col}: {metricas_globales[col].values[0]:.4f}")
        
        # Crear un grÃ¡fico de radar para las mÃ©tricas
        metrics = ["PrecisiÃ³n", "Sensibilidad", "F1 Score", "PrecisiÃ³n global", "Especificidad"]
        values = [metricas_globales[m].values[0] for m in metrics]
        
        # Crear figura para radar chart
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        
        # Agregar valores
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # Cerrar el polÃ­gono
        angles += angles[:1]  # Cerrar el polÃ­gono
        
        # Dibujar las mÃ©tricas
        ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
        ax.fill(angles, values, alpha=0.25, color='#3498db')
        
        # Etiquetas
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True)
        plt.title('Resumen de MÃ©tricas Globales', size=15)
        
        # Guardar
        plt.tight_layout()
        plt.savefig('/kaggle/working/radar_metricas_globales.png', dpi=300, bbox_inches='tight')
        print("\nğŸ’¾ GrÃ¡fico de radar de mÃ©tricas guardado como 'radar_metricas_globales.png'")
        plt.close()
        
    except Exception as e:
        print(f"\nâš ï¸� No se pudo generar el resumen de mÃ©tricas globales: {e}")
    
    print("\nâœ… AnÃ¡lisis completo finalizado!")
else:
    print("\nâ�Œ No hay datos de predicciones para analizar. Ejecute primero la Parte A.")


# Para descargar todo el workspace en zip

import os
import zipfile
from IPython.display import FileLink

# Directorio de trabajo de Kaggle
working_dir = "/kaggle/working"

# Crear un archivo zip con todos los archivos del directorio
zip_filename = "todos_los_archivos.zip"
zip_path = os.path.join(working_dir, zip_filename)

with zipfile.ZipFile(zip_path, 'w') as zipf:
    # AÃ±adir todos los archivos del directorio working al zip
    for file in os.listdir(working_dir):
        if file != zip_filename:  # Evitar incluir el propio zip
            file_path = os.path.join(working_dir, file)
            if os.path.isfile(file_path):
                zipf.write(file_path, arcname=file)

print(f"âœ… Todos los archivos comprimidos en '{zip_filename}'")

# Crear un enlace de descarga para el archivo zip
display(FileLink(zip_path))


# Para borrar lo que hay en "/kaggle/working" excepto dos archivos

import os
import shutil

# Directorio de trabajo de Kaggle
working_dir = "/kaggle/working"

# Archivos que no se deben eliminar
excluir = {"modelo_efficientnet_rgb_finetuned2.h5", "modelo_efficientnet_rgb_finetuned1.h5"}

# Listar todos los archivos antes de borrarlos
print("Archivos que serÃ¡n eliminados:")
for file in os.listdir(working_dir):
    if file not in excluir:
        print(f"- {file}")

# Confirmar antes de borrar
confirm = input("Â¿EstÃ¡s seguro de que quieres borrar todos los archivos excepto los excluidos? (s/n): ")

if confirm.lower() == 's':
    # Borrar todos los archivos excepto los excluidos
    for file in os.listdir(working_dir):
        if file in excluir:
            continue
        file_path = os.path.join(working_dir, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error al eliminar {file_path}: {e}")
    
    print("âœ… Archivos eliminados, excepto los excluidos.")
else:
    print("â�Œ OperaciÃ³n cancelada. No se ha eliminado ningÃºn archivo.")


# 2. Fine-Tuning con EfficientNetB0 pre-entrenado, RGB

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf

# One-hot encoding de etiquetas
y_cat = to_categorical(y, num_classes=len(clases))

# DivisiÃ³n entrenamiento/validaciÃ³n
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, stratify=y, random_state=42)

print(f" Datos listos: {X_train.shape[0]} entrenamiento / {X_val.shape[0]} validaciÃ³n")

# NormalizaciÃ³n de imÃ¡genes
X_train = X_train / 255.0
X_val = X_val / 255.0

# AumentaciÃ³n de datos (comentada segÃºn lo solicitado)
# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip("horizontal"),
#     tf.keras.layers.RandomRotation(0.1),
#     tf.keras.layers.RandomZoom(0.1),
# ])

# MÃ©tricas personalizadas para F1 score
def f1_metric(y_true, y_pred):
    # Convierte las probabilidades en etiquetas
    y_pred_classes = tf.argmax(y_pred, axis=1)
    y_true_classes = tf.argmax(y_true, axis=1)
    
    # Calcula precisiÃ³n y recall
    precision = tf.reduce_sum(
        tf.cast(tf.logical_and(tf.equal(y_true_classes, y_pred_classes), 
                              tf.not_equal(y_true_classes, 0)), tf.float32)
    ) / (tf.reduce_sum(tf.cast(tf.not_equal(y_pred_classes, 0), tf.float32)) + tf.keras.backend.epsilon())
    
    recall = tf.reduce_sum(
        tf.cast(tf.logical_and(tf.equal(y_true_classes, y_pred_classes), 
                              tf.not_equal(y_true_classes, 0)), tf.float32)
    ) / (tf.reduce_sum(tf.cast(tf.not_equal(y_true_classes, 0), tf.float32)) + tf.keras.backend.epsilon())
    
    # F1 Score
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return f1

# FASE ÃšNICA: Cargar modelo pre-entrenado y hacer fine-tuning
print("Cargando modelo pre-entrenado...")

# Cargar el modelo existente
model = load_model('/kaggle/input/modelo_normal/tensorflow2/default/1/modelo_efficientnet_rgb.h5', 
                  custom_objects={'f1_metric': f1_metric})

# Obtener el modelo base (EfficientNetB0)
#for layer in model.layers:
#    if isinstance(layer, tf.keras.Model):  # Encuentra el modelo base
#        base_model = layer
#        break

# Alternativamente, si el cÃ³digo anterior no funciona para extraer base_model
# Podemos recrear la arquitectura y copiar los pesos
input_tensor = Input(shape=(224, 224, 3))
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=input_tensor)

print("Preparando modelo para fine-tuning...")

# Descongelar las Ãºltimas capas (por ejemplo, los Ãºltimos 20 bloques)
base_model.trainable = True

# Congelar todas las capas excepto las Ãºltimas N
fine_tune_at = len(base_model.layers) - 10  # Ajustar este nÃºmero segÃºn necesites
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Verificar quÃ© capas son entrenables
trainable_layers = [layer.name for layer in base_model.layers if layer.trainable]
print(f"Capas entrenables: {len(trainable_layers)} de {len(base_model.layers)}")
print(f"Primeras 5 capas entrenables: {trainable_layers[:5]}")

# Recompilar el modelo con una tasa de aprendizaje mÃ¡s baja para fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-3),  # Tasa de aprendizaje mÃ¡s baja
    loss='categorical_crossentropy',
    metrics=['accuracy', f1_metric]
)

# Callbacks para la fase de fine-tuning
callbacks = [
    EarlyStopping(monitor='val_f1_metric', patience=7, mode='max', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_f1_metric', patience=3, factor=0.2, mode='max'),
]

print("Iniciando fine-tuning...")
# Entrenamiento con fine-tuning
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=3, 
    batch_size=16,  # Batch size mÃ¡s pequeÃ±o para fine-tuning
    callbacks=callbacks,
    verbose=1
)

model.save('/kaggle/working/modelo_efficientnet_rgb_finetuned3.h5')

# Evaluar el modelo final
print("Evaluando modelo final...")
test_loss, test_acc, test_f1 = model.evaluate(X_val, y_val)
print(f"Rendimiento final: Accuracy = {test_acc:.4f}, F1 Score = {test_f1:.4f}")

# Hacer predicciones y calcular mÃ©tricas adicionales
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

# Calcular F1, precisiÃ³n y recall
f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
precision = precision_score(y_true_classes, y_pred_classes, average='macro')
recall = recall_score(y_true_classes, y_pred_classes, average='macro')

print(f"MÃ©tricas detalladas:")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Guardar modelo entrenado y su historial
with open('/kaggle/working/history_finetuning.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Visualizar el entrenamiento
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy del modelo')
plt.ylabel('Accuracy')
plt.xlabel('Ã‰poca')
plt.legend(['Entrenamiento', 'ValidaciÃ³n'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['f1_metric'])
plt.plot(history.history['val_f1_metric'])
plt.title('F1 Score del modelo')
plt.ylabel('F1 Score')
plt.xlabel('Ã‰poca')
plt.legend(['Entrenamiento', 'ValidaciÃ³n'], loc='lower right')

plt.tight_layout()
plt.savefig('/kaggle/working/training_metrics_finetuning.png')
plt.show()

print("âœ… Modelo cargado, fine-tuneado y guardado con EfficientNetB0.")


# BORRAR 3. Fine-Tuning con EfficientNetB0 pre-entrenado, RGB

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf

# ========== BLOQUE DE PREPARACIÃ“N DE DATOS ==========
# IMPORTANTE: Este bloque debe ajustarse segÃºn cÃ³mo estÃ¡n definidos tus datos
# Asumiendo que ya tienes los datos cargados previamente en tu notebook/script
# Si no es asÃ­, debes cargarlos explÃ­citamente antes de este punto

# Verificar que X e y existen y tienen las dimensiones correctas
try:
    print(f"Forma de los datos X: {X.shape}")
    print(f"Forma de las etiquetas y: {y.shape}")
    print(f"NÃºmero de clases: {len(clases)}")
except NameError:
    print("ERROR: Las variables X, y o clases no estÃ¡n definidas.")
    print("AsegÃºrate de cargar tus datos antes de ejecutar este script.")
    # Descomenta estas lÃ­neas y ajusta las rutas si necesitas cargar los datos aquÃ­
    # X = np.load('/kaggle/path/to/your/data.npy')
    # y = np.load('/kaggle/path/to/your/labels.npy')
    # clases = [...] # Define tus clases aquÃ­

# One-hot encoding de etiquetas
y_cat = to_categorical(y, num_classes=len(clases))

# DivisiÃ³n entrenamiento/validaciÃ³n
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, stratify=y, random_state=42)

print(f"Datos listos: {X_train.shape[0]} entrenamiento / {X_val.shape[0]} validaciÃ³n")
print(f"Forma de X_train: {X_train.shape}, y_train: {y_train.shape}")

# NormalizaciÃ³n de imÃ¡genes
X_train = X_train / 255.0
X_val = X_val / 255.0

# AumentaciÃ³n de datos (comentada segÃºn lo solicitado)
# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip("horizontal"),
#     tf.keras.layers.RandomRotation(0.1),
#     tf.keras.layers.RandomZoom(0.1),
# ])

# ========== DEFINICIÃ“N DE MÃ‰TRICAS Y FUNCIONES AUXILIARES ==========
# MÃ©tricas personalizadas para F1 score
def f1_metric(y_true, y_pred):
    # Convierte las probabilidades en etiquetas
    y_pred_classes = tf.argmax(y_pred, axis=1)
    y_true_classes = tf.argmax(y_true, axis=1)
    
    # Calcula precisiÃ³n y recall
    precision = tf.reduce_sum(
        tf.cast(tf.logical_and(tf.equal(y_true_classes, y_pred_classes), 
                              tf.not_equal(y_true_classes, 0)), tf.float32)
    ) / (tf.reduce_sum(tf.cast(tf.not_equal(y_pred_classes, 0), tf.float32)) + tf.keras.backend.epsilon())
    
    recall = tf.reduce_sum(
        tf.cast(tf.logical_and(tf.equal(y_true_classes, y_pred_classes), 
                              tf.not_equal(y_true_classes, 0)), tf.float32)
    ) / (tf.reduce_sum(tf.cast(tf.not_equal(y_true_classes, 0), tf.float32)) + tf.keras.backend.epsilon())
    
    # F1 Score
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return f1

# ========== CARGA DEL MODELO PRE-ENTRENADO ==========
print("Cargando modelo pre-entrenado...")
try:
    # Intentar cargar el modelo con mÃ©tricas personalizadas
    model = load_model('/kaggle/input/modelo_normal/tensorflow2/default/1/modelo_efficientnet_rgb.h5', 
                    custom_objects={'f1_metric': f1_metric})
    print("Modelo cargado exitosamente.")
    
    # Intentar obtener el modelo base
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # Encuentra el modelo base
            base_model = layer
            print("Modelo base encontrado dentro del modelo cargado.")
            break
    
    if base_model is None:
        print("No se pudo encontrar el modelo base automÃ¡ticamente.")
        print("Creando un nuevo modelo base EfficientNetB0...")
        # Si no podemos extraer el modelo base, creamos uno nuevo
        input_shape = model.input_shape[1:] # Obtener shape de entrada del modelo cargado
        input_tensor = Input(shape=input_shape)
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=input_tensor)
        print(f"Modelo base creado con shape de entrada: {input_shape}")
        
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    print("Creando un nuevo modelo desde cero...")
    
    # Crear modelo desde cero si no se puede cargar
    input_tensor = Input(shape=(224, 224, 3))
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    output = Dense(len(clases), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    print("Nuevo modelo creado con Ã©xito.")

# ========== CONFIGURACIÃ“N DEL FINE-TUNING ==========
print("Preparando modelo para fine-tuning...")

# Descongelar las Ãºltimas capas
base_model.trainable = True

# Congelar todas las capas excepto las Ãºltimas N
fine_tune_at = len(base_model.layers) - 20  # Ajustar este nÃºmero segÃºn necesites
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Verificar quÃ© capas son entrenables
trainable_layers = [layer.name for layer in base_model.layers if layer.trainable]
print(f"Capas entrenables: {len(trainable_layers)} de {len(base_model.layers)}")
if trainable_layers:
    print(f"Primeras 5 capas entrenables (o menos): {trainable_layers[:min(5, len(trainable_layers))]}")

# Recompilar el modelo con una tasa de aprendizaje mÃ¡s baja para fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-4),  # Tasa de aprendizaje mÃ¡s baja
    loss='categorical_crossentropy',
    metrics=['accuracy', f1_metric]
)

# ========== ENTRENAMIENTO (FINE-TUNING) ==========
# Callbacks para la fase de fine-tuning
callbacks = [
    EarlyStopping(monitor='val_f1_metric', patience=7, mode='max', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_f1_metric', patience=3, factor=0.2, mode='max'),
    ModelCheckpoint('/kaggle/working/modelo_efficientnet_rgb_finetuned_best.h5', 
                    monitor='val_f1_metric', mode='max', save_best_only=True)
]

print("Iniciando fine-tuning...")
# Entrenamiento con fine-tuning
try:
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,  # Ajustar segÃºn necesidad
        batch_size=16,  # Batch size mÃ¡s pequeÃ±o para fine-tuning
        callbacks=callbacks,
        verbose=1
    )
    print("Fine-tuning completado con Ã©xito.")
    model.save('/kaggle/working/modelo_efficientnet_rgb_finetuned.h5')
except Exception as e:
    print(f"Error durante el entrenamiento: {e}")
    print("Revisa las dimensiones de tus datos y la configuraciÃ³n del modelo.")
    # Si hay error, imprimir formas para debug
    print(f"Forma de X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Forma esperada de entrada al modelo: {model.input_shape}")
    print(f"Forma esperada de salida del modelo: {model.output_shape}")

# ========== EVALUACIÃ“N DEL MODELO ==========
try:
    print("Evaluando modelo final...")
    # Cargar el mejor modelo guardado durante el entrenamiento
    model = load_model('/kaggle/working/modelo_efficientnet_rgb_finetuned_best.h5', 
                      custom_objects={'f1_metric': f1_metric})
    
    test_loss, test_acc, test_f1 = model.evaluate(X_val, y_val)
    print(f"Rendimiento final: Accuracy = {test_acc:.4f}, F1 Score = {test_f1:.4f}")

    # Hacer predicciones y calcular mÃ©tricas adicionales
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    # Calcular F1, precisiÃ³n y recall
    f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
    precision = precision_score(y_true_classes, y_pred_classes, average='macro')
    recall = recall_score(y_true_classes, y_pred_classes, average='macro')

    print(f"MÃ©tricas detalladas:")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Guardar historial
    with open('/kaggle/working/history_finetuning.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    # Visualizar el entrenamiento
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy del modelo')
    plt.ylabel('Accuracy')
    plt.xlabel('Ã‰poca')
    plt.legend(['Entrenamiento', 'ValidaciÃ³n'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['f1_metric'])
    plt.plot(history.history['val_f1_metric'])
    plt.title('F1 Score del modelo')
    plt.ylabel('F1 Score')
    plt.xlabel('Ã‰poca')
    plt.legend(['Entrenamiento', 'ValidaciÃ³n'], loc='lower right')

    plt.tight_layout()
    plt.savefig('/kaggle/working/training_metrics_finetuning.png')
    plt.show()
    
except Exception as e:
    print(f"Error durante la evaluaciÃ³n: {e}")

print("âœ… Proceso completado: modelo cargado, fine-tuneado y evaluado.")




