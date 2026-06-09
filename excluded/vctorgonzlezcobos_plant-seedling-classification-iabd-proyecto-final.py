from collections import Counter
import io
import math
import os
import pathlib
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import warnings
import zipfile

from google.colab import files
from IPython.display import display, Markdown, HTML
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K, callbacks, layers, models, optimizers, regularizers, initializers, Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import plot_model

# Ensure latest kaggle package
!pip install --upgrade --force-reinstall --no-deps kaggle

print("Python version:", sys.version.split()[0])
print("TensorFlow version:", tf.__version__)
print("GPUs disponibles:", len(tf.config.list_physical_devices('GPU')))



SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


DATASET_NAME = 'plant-seedlings-classification'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_CHANNELS = 3

TEST_SPLIT = 0.2
VAL_SPLIT = 0.1

BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE

EPOCHS_SCRATCH = 40
EPOCHS_AUGMENT = 30
EPOCHS_TRANSFER = 20
EPOCHS_FINE_TUNING = 10

FINE_TUNE_AT = 50 # NÃºmero de capas que dejaremos congeladas en finetuning



# En nuestro caso trabajaremos con colab pero no lo guardaremos en el drive ya que la descomprensiÃ³n es muy lenta
BASE_DIR = pathlib.Path(f'./{DATASET_NAME}')
DATA_DIR = BASE_DIR / 'data'/'train'
DATA_FOLDER = BASE_DIR / 'data'
MODEL_FOLDER = BASE_DIR / 'models'
DATA_FOLDER.mkdir(parents=True, exist_ok=True)
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)


def md(text: str, verbose = 1):
    """
    Muestra texto como Markdown.
    """
    if verbose:
      display(Markdown(text))

def plot_learning_curves(history, title_suffix=''):
    plt.figure(figsize=(12,4))
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy ' + title_suffix)
    plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()
    # Loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss ' + title_suffix)
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    plt.show()

def download_kaggle_dataset(dataset_name, extract_path=None):
    """
    Descarga un dataset de Kaggle y lo extrae en la ruta especificada.

    Parameters:
    dataset_name (str): Nombre del dataset en Kaggle (formato 'usuario/dataset' o nombre de competiciÃ³n).
    extract_path (str, optional): Ruta donde extraer los archivos. Si no se especifica, se extrae en la ruta actual dentro de una carpeta con el nombre del dataset.

    Returns:
    str: Ruta donde se extrajeron los archivos, o None en caso de error.
    """
    # Directorio de descarga/extracciÃ³n
    base_path = extract_path or os.getcwd()
    download_dir = os.path.join(base_path)

    # Si ya existe y contiene archivos, no hacemos nada
    if os.path.isdir(download_dir) and os.listdir(download_dir):
        print("El dataset ya estÃ¡ cargado en:", download_dir)
        return download_dir

    # Crear directorio de destino
    os.makedirs(download_dir, exist_ok=True)

    # Comprobar credenciales de Kaggle
    if not os.path.exists('kaggle.json'):
        print("Por favor, sube tu archivo kaggle.json con credenciales.")
        files.upload()

    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    subprocess.run(['cp', 'kaggle.json', os.path.expanduser('~/.kaggle/')], check=True)
    os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)

    # Descargar dataset o competiciÃ³n
    zip_path = os.path.join(download_dir, f"{dataset_name.replace('/', '_')}.zip")

    # Intentar como competiciÃ³n primero
    result = subprocess.run(
        ['kaggle', 'competitions', 'download', '-c', dataset_name, '-p', download_dir],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    if result.returncode == 0:
        print(f"Descargando competiciÃ³n: {dataset_name}")
    else:
        # Si falla, intentamos como dataset
        print(f"Descargando dataset: {dataset_name}")
        subprocess.run(['kaggle', 'datasets', 'download', dataset_name, '-p', download_dir], check=True)

    # Verificar descarga
    if os.path.exists(zip_path):
        print(f"Extrayendo archivos en {download_dir}")
        subprocess.run(['unzip', '-q', zip_path, '-d', download_dir], check=True)

        # Eliminar zip
        os.remove(zip_path)
        print(f"Dataset extraÃ­do con Ã©xito en {download_dir}")
        return download_dir
    else:
        print("La descarga fallÃ³. Revisa el nombre del dataset o tus credenciales de Kaggle.")
        return None

def train_plot_model(model_name, model, train_ds, val_ds, epochs=EPOCHS_SCRATCH, class_weights=None, checkpoint=True):
    """
    Entrena el modelo, guarda el mejor checkpoint y muestra las curvas de aprendizaje.
    """
    md(f"### ğŸ”„ Entrenando modelo **{model_name}** durante {epochs} epochs...\n")
    if checkpoint:
      cb_list = [
          callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
          callbacks.ModelCheckpoint(MODEL_FOLDER / f'best_{model_name}.h5', save_best_only=True),
          callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3)
      ]
    else:
      cb_list = [
          callbacks.EarlyStopping(monitor='val_loss', patience=10),
          callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3)
      ]
    if class_weights is not None:
      history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb_list,
        class_weight=class_weights,
        verbose=1
    )
    else:
      history = model.fit(
          train_ds,
          validation_data=val_ds,
          epochs=epochs,
          callbacks=cb_list,
          verbose=1
      )

    # Mostrar curvas de entrenamiento
    md(f"#### ğŸ“ˆ Curvas de aprendizaje de **{model_name}**")
    plot_learning_curves(history, title_suffix=f'({model_name})')

    return history

def plot_confusion_matrix(cm, class_names, title=None):
    """
    Dibuja la matriz de confusiÃ³n.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='Etiqueta verdadera',
        xlabel='Etiqueta predicha',
        title=title or 'Matriz de confusiÃ³n'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    # Anotar valores
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

def load_model_from(model_name ,base_path=MODEL_FOLDER):
    """
    Carga un modelo desde el sistema de archivos.
    """
    model_path = base_path / f'{model_name}.h5'

    return models.load_model(model_path)

def evaluate_model(model_name, model, test_ds, class_names, verbose=1):
    """
    EvalÃºa el modelo en el dataset de test, muestra reporte y matriz de confusiÃ³n.
    """
    md(f"### ğŸ§ª EvaluaciÃ³n del modelo **{model_name}** en test\n",verbose)

    # EvaluaciÃ³n de mÃ©tricas generales
    loss, *metrics = model.evaluate(test_ds, verbose=0)
    md(f"**PÃ©rdida:** {loss:.4f}  ")
    for name, value in zip(model.metrics_names[1:], metrics):
        md(f"**{name}:** {value:.4f}  ",verbose)
    md("---")

    # Predicciones y mÃ©tricas por clase
    y_pred = np.argmax(model.predict(test_ds), axis=1)
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).T

    md("#### ğŸ“Š Informe de clasificaciÃ³n",verbose)
    display(report_df.style.format("{:.2f}"))
    md("---",verbose)

    # Matriz de confusiÃ³n
    cm = sk_confusion_matrix(y_true, y_pred)
    md(f"#### ğŸ”� Matriz de confusiÃ³n â€“ {model_name}",verbose)
    plot_confusion_matrix(cm, class_names)

def compare_models(
    model_variants,
    test_ds
):
    accuracies = {}
    md("## ğŸ�† ComparaciÃ³n de accuracies en test")
    # EvaluaciÃ³n de cada variante
    for model, name in model_variants:
        loss, acc = model.evaluate(test_ds, verbose=0)
        accuracies[name] = acc

    # â€”â€”â€”â€”â€”â€”â€” Tabla (nombre, accuracy, peso del modelo) ordenada por accuracy â€”â€”â€”â€”â€”â€”â€”
    table_rows = []
    for model, name in model_variants:
        # Guardar pesos en un fichero temporal para medir su tamaÃ±o
        with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as tmp:
            tmp_path = tmp.name
        model.save_weights(tmp_path)
        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        os.remove(tmp_path)

        table_rows.append((name, accuracies[name], size_mb))

    # Ordenar por accuracy descendente
    table_rows.sort(key=lambda x: x[1], reverse=True)

    # Renderizar la tabla Markdown
    table = "| Modelo | Accuracy | TamaÃ±o (MB) |\n"
    table+="|:---|:---:|:---:|"
    for name, acc, size in table_rows:
        table+=f"\n| {name} | {acc:.2%} | {size:.2f} |"
    md(table)
    # â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”â€”

    # SelecciÃ³n del mejor
    best_name = max(accuracies, key=accuracies.get)
    best_model = next(m for m, n in model_variants if n == best_name)
    md(f"## ğŸ¥‡ Mejor modelo: **{best_name}** con accuracy {accuracies[best_name]:.4%}")

    return best_model


print("\n--- 1. Carga del Dataset ---")


# --- TU CÃ“DIGO AQUÃ� ---
download_kaggle_dataset(DATASET_NAME, DATA_FOLDER)
# --- FIN TU CÃ“DIGO ---


# --- TU CÃ“DIGO AQUÃ� ---
class_names = sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])
NUM_CLASSES = len(class_names)
print(f"Clases detectadas ({NUM_CLASSES}): {class_names}")
# --- FIN TU CÃ“DIGO ---


print("\n--- 2. AnÃ¡lisis Exploratorio de Datos (EDA) ---")


# --- TU CÃ“DIGO AQUÃ� ---
# Muestra algunas imÃ¡genes del conjunto de entrenamiento con sus etiquetas
plt.figure(figsize=(10,8))

# NÃºmero de clases
n = len(class_names)
# NÃºmero de columnas que quieres (por ejemplo 3)
cols = 3
# Calcula filas necesarias
rows = math.ceil(n / cols)

# Crea la figura y los ejes
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
axes = axes.flatten()  # para indexar fÃ¡cilmente

for i, class_name in enumerate(class_names):
    class_dir = DATA_DIR / class_name
    img_files = list(class_dir.glob('*.png'))
    if not img_files:
        img_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg'))
    if not img_files:
        warnings.warn(f"No se encontraron imÃ¡genes para la clase '{class_name}'. Se omitirÃ¡.")
        continue

    # Carga la imagen
    img = keras.preprocessing.image.load_img(img_files[0], target_size=IMAGE_SIZE)

    # Muestra en el subplot correspondiente
    ax = axes[i]
    ax.imshow(img)
    ax.set_title(class_name)
    ax.axis('off')
# Si hay ejes sobrantes, los desactivamos
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle('Ejemplos de cada clase')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
# --- FIN TU CÃ“DIGO ---


# --- TU CÃ“DIGO AQUÃ� ---
# Verifica las dimensiones de las imÃ¡genes cargadas y el formato de las etiquetas

# â€”â€”â€” 1. Listado de todos los ficheros de imagen â€”â€”â€”
train_filepaths = []
for ext in ('png', 'jpg', 'jpeg'):
    train_filepaths.extend(DATA_DIR.glob(f'*/*.{ext}'))

if not train_filepaths:
    raise RuntimeError(f"No se encontraron imÃ¡genes en {DATA_DIR}")

# â€”â€”â€” 2. Leer tamaÃ±os de cada imagen â€”â€”â€”
sizes = [Image.open(fp).size for fp in train_filepaths]

# â€”â€”â€” 3. DimensiÃ³n mÃ¡s frecuente â€”â€”â€”
size_counts = Counter(sizes)
most_common_size, most_common_count = size_counts.most_common(1)[0]
print(f"DimensiÃ³n mÃ¡s repetida: {most_common_size} (aparece {most_common_count} veces)")

# â€”â€”â€” 4. Ã�rea mÃ¡xima y mÃ­nima â€”â€”â€”
areas = [w * h for w, h in sizes]
idx_max, idx_min = int(np.argmax(areas)), int(np.argmin(areas))
print(f"DimensiÃ³n con Ã¡rea MÃ�XIMA: {sizes[idx_max]} (Ã¡rea = {areas[idx_max]})")
print(f"DimensiÃ³n con Ã¡rea MÃ�NIMA: {sizes[idx_min]} (Ã¡rea = {areas[idx_min]})")
# --- FIN TU CÃ“DIGO ---


# --- TU CÃ“DIGO AQUÃ� ---
# Calcula y visualiza cuÃ¡ntas imÃ¡genes hay por cada clase en el conjunto de entrenamiento
# Esto es crucial para detectar desbalanceo.
counts = {c: len(list((DATA_DIR/c).glob('*.png'))) for c in class_names}
pd.Series(counts).sort_values().plot(kind='bar', figsize=(8,4))
plt.title('NÃºmero de imÃ¡genes por clase')
plt.ylabel('Count')
plt.show()
# --- FIN TU CÃ“DIGO ---


print("\n--- 3. Preprocesamiento de Datos ---")


# --- TU CÃ“DIGO AQUÃ� ---
# --- FIN TU CÃ“DIGO ---


# Conjuntos separados para entrenar, ajustar hiperparÃ¡metros
# y evaluar el rendimiento final de forma imparcial.
# --- TU CÃ“DIGO AQUÃ� ---
# 1) Split inicial TRAIN / TEST
train_batch_ds = image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='int',
    validation_split=TEST_SPLIT,
    subset='training',
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)
test_batch_ds = image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='int',
    validation_split=TEST_SPLIT,
    subset='validation',
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# 2) "Unbatch" para quitar la estructura de batches
train_unbatched = train_batch_ds.unbatch()
test_unbatched  = test_batch_ds.unbatch()

# 3) FunciÃ³n auxiliar: pasar de tf.data.Dataset a NumPy arrays
def ds_to_numpy(ds):
    xs, ys = [], []
    for img, label in ds:
        xs.append(img.numpy())
        ys.append(label.numpy())
    return np.array(xs), np.array(ys)

x_all, y_all   = ds_to_numpy(train_unbatched)   # ~80% de las imÃ¡genes
x_test, y_test = ds_to_numpy(test_unbatched)    # ~20% de las imÃ¡genes

# 4) Split TRAIN / VALIDATION sobre el 80% restante
x_train, x_val, y_train, y_val = train_test_split(
    x_all, y_all,
    test_size=VAL_SPLIT,
    stratify=y_all
)

# 5) Reconstruir tf.data.Dataset a partir de los NumPy arrays
train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
      .shuffle(buffer_size=len(x_train))
      .batch(BATCH_SIZE)
      .cache()
      .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
      .batch(BATCH_SIZE)
      .cache()
      .prefetch(AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
      .batch(BATCH_SIZE)
      .cache()
      .prefetch(AUTOTUNE)
)

# Normalizar las imÃ¡genes a [0, 1] y mantener las etiquetas
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds   = val_ds.map(lambda x, y: (x / 255.0, y))
test_ds  = test_ds.map(lambda x, y: (x / 255.0, y))

print(f"Train size: {len(x_train)}")
print(f"Validation size: {len(x_val)}")
print(f"Test size: {len(x_test)}")
# --- FIN TU CÃ“DIGO ---


print("\n--- 4. Modelo 1: CNN desde Cero ---")


# --- TU CÃ“DIGO AQUÃ� ---
def build_cnn_scratch(input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,3), num_classes=NUM_CLASSES):
    n = 32

    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    for i in range(4):
        model.add(layers.Conv2D(n, (3,3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.1))
        n *=2

    model.add(layers.Flatten())
    model.add(layers.Dense(n, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

model_1 = build_cnn_scratch()
model_1.build(input_shape=(None, *IMAGE_SIZE, IMAGE_CHANNELS))
model_1.summary()
# --- FIN TU CÃ“DIGO ---


# --- TU CÃ“DIGO AQUÃ� ---
model_1.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# --- FIN TU CÃ“DIGO ---


# --- TU CÃ“DIGO AQUÃ� ---
# Entrenamiento
history_scratch = train_plot_model('scratch', model_1, train_ds, val_ds, epochs=EPOCHS_SCRATCH)
# --- FIN TU CÃ“DIGO ---


# --- TU CÃ“DIGO AQUÃ� ---
# Cargar
model_1 = load_model_from('best_scratch')

# EvaluaciÃ³n
evaluate_model('scratch', model_1, test_ds, class_names)
# --- FIN TU CÃ“DIGO ---


# Vamos a crear variantes de este modelo para probar diferentes arquitecturas de cnn
def build_cnn_light(input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,3), num_classes=NUM_CLASSES):
    return models.Sequential([
        layers.Input(input_shape),
        # Bloque 1
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        # Bloque 2
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        # â€”> Aplanamos antes del Dense
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])

def build_cnn_anistropic(input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,3), num_classes=NUM_CLASSES):
    return models.Sequential([
        layers.Input(input_shape),
        # Bloque 1 (3Ã—3)
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        # Bloque 2 (5Ã—3)
        layers.Conv2D(64, (5,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        # Bloque 3 (7Ã—3)
        layers.Conv2D(128, (7,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        # â€”> Aplanamos antes del Dense
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])

def build_vgg(input_shape, num_classes):
    return models.Sequential([
        layers.Input(input_shape),
        # Bloque 1: 2Ã— (Conv3x3 â†’ ReLU â†’ BN) â†’ Pool
        layers.Conv2D(32,(3,3),padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32,3,padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        # Bloque 2: 2Ã— (Conv3x3 â†’ ReLU â†’ BN) â†’ Pool
        layers.Conv2D(64,(3,3),padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64,(3,3),padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        # Bloque 3: 2Ã— (Conv3x3 â†’ ReLU â†’ BN) â†’ Pool
        layers.Conv2D(128,(3,3),padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128,(3,3),padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        # Clasificador
        layers.Flatten(),
        layers.Dense(512,activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes,activation='softmax'),
    ])

def residual_block(x, filters, downsample=False):
    y = layers.Conv2D(filters,3,padding='same',
                      strides=2 if downsample else 1,
                      activation='relu')(x)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(filters,(3,3),padding='same',activation=None)(y)
    y = layers.BatchNormalization()(y)
    if downsample:
        x = layers.Conv2D(filters,1,strides=2,padding='same')(x)
        x = layers.BatchNormalization()(x)
    out = layers.Add()([x,y])
    out = layers.Activation('relu')(out)
    return out

def build_resnetish(input_shape, num_classes, n_blocks=[2,2,2]):
    inp = layers.Input(input_shape)
    x = layers.Conv2D(64,7,strides=2,padding='same',activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3,strides=2,padding='same')(x)
    for i, blocks in enumerate(n_blocks):
        for j in range(blocks):
            x = residual_block(x, filters=64*(2**i), downsample=(j==0 and i>0))
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(num_classes,activation='softmax')(x)
    return Model(inp, out)

model_variants = [
    (build_cnn_light(), 'light'),
    (build_cnn_anistropic(), 'anistropic'),
    (build_vgg((IMAGE_HEIGHT,IMAGE_WIDTH,3), NUM_CLASSES), 'vgg'),
    (build_resnetish((IMAGE_HEIGHT,IMAGE_WIDTH,3), NUM_CLASSES), 'resnetish'),
]

for i, (orig_model, name) in enumerate(model_variants):
    K.clear_session()
    model = orig_model
    model.name = name
    model.summary()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    # Entreno la variante CORRECTA
    history = train_plot_model(name, model, train_ds, val_ds, epochs=EPOCHS_SCRATCH)
    # Cargamos y sustituimos para asegurar de que es el checkpoint que ha guardado el ModelCheckPoint
    model = load_model_from(f'best_{name}')
    model_variants[i] = (model, name)
    # La evalÃºo tambiÃ©n sobre la variante correcta
    evaluate_model(name, model, test_ds, class_names)


#Comparamos el accuracy de todos
model_variants.append((model_1, 'scratch'))
#En base a eso elegimoe el mejor modelo
best_scratch_model = compare_models(model_variants, test_ds=test_ds)


print("\n--- 5. Modelo 2: Mejoras (Data Augmentation / Class Weighting) ---")
# Esta secciÃ³n es opcional pero muy recomendada, especialmente si hay overfitting o desbalanceo


# --- TU CÃ“DIGO AQUÃ� ---
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights_array))

# Mostramos el peso asignado a cada clase: un valor mayor refuerza la atenciÃ³n del modelo sobre las clases menos frecuentes durante el entrenamiento.
# Como la mas frecuente es "Loose Silky Bent", su peso es el mas bajo.
for k, v in class_weights.items():
    label = class_names[k]
    print(f"Clase {k} ({label}): {v}")
# --- FIN TU CÃ“DIGO ---


# Se puede hacer con capas de Keras o con ImageDataGenerator
# --- TU CÃ“DIGO AQUÃ� ---

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# --- FIN TU CÃ“DIGO ---


# --- TU CÃ“DIGO AQUÃ� ---
def build_cnn_scratch_aug(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    n = 32
    for _ in range(4):
        x = layers.Conv2D(n, (3,3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.1)(x)
        n *= 2
    x = layers.Flatten()(x)
    x = layers.Dense(n, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name="scratch_aug")

def build_resnetish_aug(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    # Bloque inicial
    x = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    # Bloques residuales
    for i, blocks in enumerate([2,2,2]):
        for j in range(blocks):
            downsample = (j == 0 and i > 0)
            y = layers.Conv2D(64*(2**i), 3, strides=2 if downsample else 1,
                              padding='same', activation='relu')(x)
            y = layers.BatchNormalization()(y)
            y = layers.Conv2D(64*(2**i), 3, padding='same', activation=None)(y)
            y = layers.BatchNormalization()(y)
            if downsample:
                x = layers.Conv2D(64*(2**i), 1, strides=2, padding='same')(x)
                x = layers.BatchNormalization()(x)
            x = layers.Add()([x, y])
            x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name="resnetish_aug")

models_aug = [
    (build_cnn_scratch_aug(), 'scratch_aug'),
    (build_resnetish_aug(), 'resnetish_aug'),
]
for model, name in models_aug:
    K.clear_session()
    model.name = name
    model.summary()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    train_plot_model(name, model, train_ds, val_ds, epochs=EPOCHS_SCRATCH, class_weights=class_weights)

# --- FIN TU CÃ“DIGO ---


# --- TU CÃ“DIGO AQUÃ� ---
# Cargamos los modelos
model_scratch_aug = load_model_from('best_scratch_aug')
model_resnetish_aug = load_model_from('best_resnetish_aug')

models_aug = [
    (model_scratch_aug, 'scratch_aug'),
    (model_resnetish_aug, 'resnetish_aug'),
]

# Evaluamos modelos
for model, name in models_aug:
    evaluate_model(name, model, test_ds, class_names)

# Comparamos modelos
best_aug_model = compare_models(models_aug, test_ds=test_ds)
models_aug.append(model_variants[-2])
models_aug.append((model_1, 'scratch'))
# Comparamos todos
best_model_aug = compare_models(models_aug, test_ds=test_ds)

# --- FIN TU CÃ“DIGO ---


print("\n--- 6. Modelo 3: Transfer Learning ---")


# --- TU CÃ“DIGO AQUÃ� ---
# Elige una red base (VGG16, ResNet50V2, MobileNetV2, etc.)
# AsegÃºrate que IMAGE_SIZE sea compatible con la red elegida (ej. 224x224 para muchas)
base_models = {
    'VGG16': (tf.keras.applications.VGG16, (IMAGE_HEIGHT,IMAGE_WIDTH)),
    'ResNet50V2': (tf.keras.applications.ResNet50V2, (IMAGE_HEIGHT,IMAGE_WIDTH)),
    'MobileNetV2': (tf.keras.applications.MobileNetV2, (IMAGE_HEIGHT,IMAGE_WIDTH)),
    'EfficientNetB0': (tf.keras.applications.EfficientNetB0, (IMAGE_HEIGHT,IMAGE_WIDTH))
}
# --- FIN TU CÃ“DIGO ---


# --- TU CÃ“DIGO AQUÃ� ---
# Define el preprocesamiento especÃ­fico de la red elegida
preprocess_input_dict = {
    'VGG16':          tf.keras.applications.vgg16.preprocess_input,
    'ResNet50V2':     tf.keras.applications.resnet_v2.preprocess_input,
    'MobileNetV2':    tf.keras.applications.mobilenet_v2.preprocess_input,
    'EfficientNetB0': tf.keras.applications.efficientnet.preprocess_input
}


# --- FIN TU CÃ“DIGO ---


# --- TU CÃ“DIGO AQUÃ� ---
# Compilar el modelo
models_tl = []  # Lista para guardar los modelos de TL
for name, (base_fn, size) in base_models.items():
    K.clear_session()
    input_shape = (*size, IMAGE_CHANNELS)
    # Carga del backbone sin la parte de clasificaciÃ³n superior
    backbone = base_fn(
        weights='imagenet', include_top=False,
        input_shape=input_shape
    )
    backbone.trainable = False # Congelamos modelo base

    # ConstrucciÃ³n del modelo de transferencia
    inputs = layers.Input(shape=input_shape)
    x = layers.Lambda(preprocess_input_dict[name])(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model_tl = tf.keras.Model(inputs, outputs, name=f"{name}_tl")
    model_tl.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    models_tl.append((model_tl, name))

# Entrenamiento
histories_tl = {}
for model, name in models_tl:
    K.clear_session()
    print(f"\n--- Transfer Learning {name} ---")
    model.summary()
    hist = train_plot_model(
        model_name=name + '_tl',
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=EPOCHS_TRANSFER
    )
    histories_tl[name] = hist
# --- FIN TU CÃ“DIGO ---


# --- TU CÃ“DIGO AQUÃ� ---
# Carga de modelos
models_tl = [
    (models.load_model(str(MODEL_FOLDER/'best_EfficientNetB0_tl.h5'), custom_objects={'preprocess_input': preprocess_input_dict["EfficientNetB0"]}),"EfficientNetB0" ),
    (models.load_model(str(MODEL_FOLDER/'best_ResNet50V2_tl.h5'), custom_objects={'preprocess_input': preprocess_input_dict["ResNet50V2"]}), "ResNet50V2"),
    (models.load_model(str(MODEL_FOLDER/'best_MobileNetV2_tl.h5'), custom_objects={'preprocess_input': preprocess_input_dict["MobileNetV2"]}), "MobileNetV2"),
    (models.load_model(str(MODEL_FOLDER/'best_VGG16_tl.h5'), custom_objects={'preprocess_input': preprocess_input_dict["VGG16"]}), "VGG16")
]

for model, name in models_tl:
    evaluate_model(name + '_tl', model, test_ds, class_names)


best_transfer_model = compare_models(models_tl, test_ds=test_ds)
# --- FIN TU CÃ“DIGO ---



# --- TU CÃ“DIGO AQUÃ� ---
# Modelos guardados a cargar: (nombre, ruta al .h5)
choosen_tl_models = [
    (models.load_model('plant-seedlings-classification/models/best_ResNet50V2_tl.h5', custom_objects={'preprocess_input': preprocess_input_dict["ResNet50V2"]}),"ResNet50V2_ft" ),
    (models.load_model('plant-seedlings-classification/models/best_MobileNetV2_tl.h5', custom_objects={'preprocess_input': preprocess_input_dict["MobileNetV2"]}), "MobileNetV2_ft")
]


# Descongela algunas capas superiores del modelo base
for model, name in choosen_tl_models:
     print(f"\n--- Fine-tuning {name} ---")
    # Obtenemos el backbone que estÃ¡ en la segunda capa del modelo (despuÃ©s del Lambda)
     base_model = model.layers[2]
     # Descongelar todo el backbone
     base_model.trainable = True
    # Re-congelar las primeras `fine_tune_at` capas para regularizaciÃ³n
     for layer in base_model.layers[:FINE_TUNE_AT]:
         layer.trainable = False
      # Recompilamos con learning rate bajo para fine-tuning
     model.compile(
          optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # IMPORTANTE EL LR bajo
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy']
      )
    # Entrenamos de nuevo
     hist_ft = train_plot_model(
          model_name=name,
          model=model,
          train_ds=train_ds,
          val_ds=val_ds,
          epochs=EPOCHS_FINE_TUNING,
          checkpoint = False
      )
     histories_tl[name] = hist_ft
    # Evaluamos
     evaluate_model(name, model, test_ds, class_names)

best_tl_finetuned = compare_models(choosen_tl_models, test_ds=test_ds)

# --- FIN TU CÃ“DIGO ---


model_ResNet50V2_tl = models.load_model(str(MODEL_FOLDER/'best_ResNet50V2_tl.h5'), custom_objects={'preprocess_input': preprocess_input_dict["ResNet50V2"]})
model_MobileNetV2_tl = models.load_model(str(MODEL_FOLDER/'best_MobileNetV2_tl.h5'), custom_objects={'preprocess_input': preprocess_input_dict["MobileNetV2"]})
model_MobileNetV2_ft = choosen_tl_models[1]
model_ResNet50V2_ft = choosen_tl_models[0]

models_ft = [
    (model_ResNet50V2_tl, 'ResNet50V2_tl'),
    (model_MobileNetV2_tl, 'MobileNetV2_tl'),
    model_MobileNetV2_ft,
    model_ResNet50V2_ft
]
_ = compare_models(models_ft, test_ds=test_ds)


print("\n--- 7. ComparaciÃ³n y Conclusiones ---")



# --- TU CÃ“DIGO AQUÃ� ---
# Crea una tabla (puedes usar print o librerÃ­as como Pandas) resumiendo
# las mÃ©tricas clave (Accuracy, Precision, Recall, F1-Score en Test)
# para cada modelo entrenado (CNN Scratch, CNN Augmented, Transfer Learning, Fine-Tuned).
# Cargamos todos los modelos de nuevo
model_scratch = models.load_model(str(MODEL_FOLDER/'best_scratch.h5'))
model_light = models.load_model(str(MODEL_FOLDER/'best_light.h5'))
model_anistropic = models.load_model(str(MODEL_FOLDER/'best_anistropic.h5'))
model_vgg = models.load_model(str(MODEL_FOLDER/'best_vgg.h5'))
model_resnetish = models.load_model(str(MODEL_FOLDER/'best_resnetish.h5'))
model_scratch_aug = models.load_model(str(MODEL_FOLDER/'best_scratch_aug.h5'))
model_resnetish_aug = models.load_model(str(MODEL_FOLDER/'best_resnetish_aug.h5'))
model_EfficientNetB0_tl = models.load_model(str(MODEL_FOLDER/'best_EfficientNetB0_tl.h5'), custom_objects={'preprocess_input': preprocess_input_dict["EfficientNetB0"]})
model_VGG16_tl = models.load_model(str(MODEL_FOLDER/'best_VGG16_tl.h5'), custom_objects={'preprocess_input': preprocess_input_dict["VGG16"]})
all_models = [
    (model_scratch, 'scratch'),
    (model_light, 'light'),
    (model_anistropic, 'anistropic'),
    (model_vgg, 'vgg'),
    (model_resnetish, 'resnetish'),
    (model_scratch_aug, 'scratch_aug'),
    (model_resnetish_aug, 'resnetish_aug'),
    (model_EfficientNetB0_tl, 'EfficientNetB0_tl'),
    (model_ResNet50V2_tl, 'ResNet50V2_tl'),
    (model_MobileNetV2_tl, 'MobileNetV2_tl'),
    (model_VGG16_tl, 'VGG16_tl'),
    model_MobileNetV2_ft,
    model_ResNet50V2_ft
]

_ = compare_models(all_models, test_ds=test_ds)# Hacemos la tabla comparativa
# Se recomienda extraer las mÃ©tricas del classification_report para una comparaciÃ³n mÃ¡s detallada.
# --- FIN TU CÃ“DIGO ---


# --- ESCRIBE TU ANÃ�LISIS AQUÃ� (en celdas Markdown) ---
# * Comenta cuÃ¡l modelo obtuvo el mejor rendimiento general y por quÃ© crees que fue asÃ­.
# * Analiza las curvas de aprendizaje: Â¿Hubo overfitting? Â¿AyudÃ³ el Data Augmentation?
# * Â¿CÃ³mo afectÃ³ el Transfer Learning al rendimiento y al tiempo de entrenamiento (observado)?
# * Â¿Fue Ãºtil el Fine-Tuning (si se realizÃ³)?
# * Menciona los principales desafÃ­os encontrados durante el proyecto (carga de datos, preprocesamiento, entrenamiento, ajuste de hiperparÃ¡metros).
# * PropÃ³n posibles mejoras o siguientes pasos que podrÃ­an realizarse.
# ------------------------------------------------------------------------------

print("\n--- Fin del Proyecto ---")

