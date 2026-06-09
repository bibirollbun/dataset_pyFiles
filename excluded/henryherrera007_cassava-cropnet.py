# ============================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ============================================
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# Definir rutas principales
PATHS = {
    'TRAIN_CSV': '/kaggle/input/cassava-leaf-disease-classification/train.csv',
    'TEST_CSV': '/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv',
    'DISEASE_MAP': '/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json',
    'TRAIN_IMAGES': '/kaggle/input/cassava-leaf-disease-classification/train_images',
    'TEST_IMAGES': '/kaggle/input/cassava-leaf-disease-classification/test_images',
    'OUTPUT': '/kaggle/working/submission.csv',
    'MODEL_CACHE': '/kaggle/working/model_cache',
    'WEIGHTS': '/kaggle/working/weights',
    'PLOTS': '/kaggle/working/plots',
    'SAVED_MODEL': '/kaggle/working/cassava_disease_model_tf'
}

# Crear directorios
for directory in ['MODEL_CACHE', 'WEIGHTS', 'PLOTS', 'SAVED_MODEL']:
    os.makedirs(PATHS[directory], exist_ok=True)

# Cargar mapeo de enfermedades
with open(PATHS['DISEASE_MAP'], 'r') as f:
    disease_map = json.load(f)
disease_map = {int(k): v for k, v in disease_map.items()}
num_classes = len(disease_map)

print(f"Número de clases: {num_classes}")
print("Mapeo de clases:", disease_map)

# Cargar dataset de entrenamiento
train_df = pd.read_csv(PATHS['TRAIN_CSV'])
print(f"Datos de entrenamiento: {train_df.shape}")



# ============================================
# 2. DIVISIÓN DEL DATASET Y GENERADORES
# ============================================
train_df, val_df = train_test_split(
    train_df, test_size=0.2, stratify=train_df['label'], random_state=42
)

train_df['label_str'] = train_df['label'].astype(str)
val_df['label_str'] = val_df['label'].astype(str)

# Parámetros de imagen
img_height, img_width = 224, 224
batch_size = 32

# Generadores
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, directory=PATHS['TRAIN_IMAGES'],
    x_col='image_id', y_col='label_str',
    target_size=(img_height, img_width),
    batch_size=batch_size, class_mode='categorical', shuffle=True
)
val_generator = val_datagen.flow_from_dataframe(
    val_df, directory=PATHS['TRAIN_IMAGES'],
    x_col='image_id', y_col='label_str',
    target_size=(img_height, img_width),
    batch_size=batch_size, class_mode='categorical', shuffle=False
)



# ============================================
# 3. VALIDACIÓN DE SPLITS Y ETIQUETAS
# ============================================
def validate_dataset_splits(train_gen, val_gen):
    print("\n=== VALIDACIÓN DE DATASETS ===")
    total = train_gen.n + val_gen.n
    print(f"Entrenamiento: {train_gen.n} imágenes ({train_gen.n/total:.2%})")
    print(f"Validación: {val_gen.n} imágenes ({val_gen.n/total:.2%})")
validate_dataset_splits(train_generator, val_generator)



# ============================================
# 4. CARGA DEL MODELO BASE (CROPNET)
# ============================================
cropnet_path = "/kaggle/input/cropnet/tensorflow1/classifier-cassava-disease-v1/1"
base_model_layer = tf.keras.layers.TFSMLayer(cropnet_path, call_endpoint='default')

# Inspeccionar salida
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
base_outputs = base_model_layer(inputs)
x = base_outputs[list(base_outputs.keys())[0]] if isinstance(base_outputs, dict) else base_outputs



# ============================================
# 5. MODELO FINAL Y ENTRENAMIENTO
# ============================================
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)



# Fase 1: entrenar solo la cabeza
base_model_layer.trainable = True

# Contamos las capas totales
total_layers = len(base_model_layer.layers)
half = total_layers // 2

print(f"Total layers: {total_layers}, freezing first {half}")

# Congelamos la mitad inferior
for layer in base_model_layer.layers[:half]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint(os.path.join(PATHS['WEIGHTS'], 'best_model.keras'),
                    monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
]

steps_per_epoch = train_generator.n // batch_size
val_steps = val_generator.n // batch_size

print("Entrenando con modelo base congelado...")
history_frozen = model.fit(
    train_generator, steps_per_epoch=steps_per_epoch,
    validation_data=val_generator, validation_steps=val_steps,
    epochs=10, callbacks=callbacks
)



# Fase 2: fine-tuning
base_model_layer.trainable = True
model.compile(optimizer=optimizers.Adam(5e-6), loss='categorical_crossentropy', metrics=['accuracy'])

print("Entrenando con modelo base descongelado (fine-tuning)...")
history_unfrozen = model.fit(
    train_generator, steps_per_epoch=steps_per_epoch,
    validation_data=val_generator, validation_steps=val_steps,
    epochs=1, initial_epoch=history_frozen.epoch[-1]+1, callbacks=callbacks
)



# ============================================
# 6. VISUALIZACIÓN DEL ENTRENAMIENTO
# ============================================
def plot_training_history(history_frozen, history_unfrozen):
    merged = {}
    for m in history_frozen.history:
        merged[m] = list(history_frozen.history[m])
    for m in history_unfrozen.history:
        merged[m].extend(history_unfrozen.history[m])
    epochs = range(1, len(merged['accuracy']) + 1)

    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, merged['accuracy'], 'b', label='Entrenamiento')
    plt.plot(epochs, merged['val_accuracy'], 'r', label='Validación')
    plt.axvline(x=len(history_frozen.history['accuracy']), color='g', linestyle='--')
    plt.title('Precisión del modelo')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(epochs, merged['loss'], 'b', label='Entrenamiento')
    plt.plot(epochs, merged['val_loss'], 'r', label='Validación')
    plt.axvline(x=len(history_frozen.history['accuracy']), color='g', linestyle='--')
    plt.title('Pérdida del modelo')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_training_history(history_frozen, history_unfrozen)



# ============================================
# 7. GUARDADO DEL MODELO Y PREDICCIONES
# ============================================
@tf.function(input_signature=[tf.TensorSpec(shape=[None, img_height, img_width, 3], dtype=tf.float32)])
def serving_fn(input_image):
    return {'predictions': model(input_image, training=False)}

tf.saved_model.save(model, PATHS['SAVED_MODEL'], signatures={'serving_default': serving_fn})
print(f"Modelo guardado en: {PATHS['SAVED_MODEL']}")



# ============================================
# 8. MATRIZ DE CONFUSIÓN Y MÉTRICAS
# ============================================
print("\n=== EVALUACIÓN DEL MODELO EN VALIDACIÓN ===")
val_generator.reset()
preds = model.predict(val_generator, steps=val_steps + 1)
y_pred = np.argmax(preds, axis=1)
y_true = val_generator.classes[:len(y_pred)]

# Reporte de clasificación
report = classification_report(y_true, y_pred, target_names=[disease_map[i] for i in range(num_classes)])
print(report)

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[disease_map[i] for i in range(num_classes)],
            yticklabels=[disease_map[i] for i in range(num_classes)])
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de confusión - Validación")
plt.show()

acc = accuracy_score(y_true, y_pred)
print(f"Precisión global: {acc:.4f}")


