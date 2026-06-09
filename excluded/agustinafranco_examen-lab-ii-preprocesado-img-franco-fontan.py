!pip install "protobuf==3.20.3" --force-reinstall

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.notebook import tqdm
import gc

# Importaciones de Keras (Igual que tu original)
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


# --- CONFIGURACIÃ“N ---
# EfficientNetB0 usa nativamente 224x224
IMG_SIZE = 224
BATCH_SIZE = 32 # Procesamos 64 fotos a la vez para ir rÃ¡pido
print(f"ðŸ”§ ConfiguraciÃ³n: EfficientNetB0 | Batch: {BATCH_SIZE}")

# Configurar GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("âœ… GPU Activada.")
    except RuntimeError as e:
        print(e)

# --- MODELO ---
# Usamos el mismo modelo que tu script original
model = EfficientNetB0(weights='imagenet', include_top=False, pooling="max")

# --- FUNCIÃ“N OPTIMIZADA (Estilo similar al tuyo pero por lotes) ---
def process_images(ids, img_dir):
    print(f"ðŸš€ Procesando {len(ids)} imÃ¡genes en {img_dir}...")
    
    features = {}
    batch_imgs = []
    batch_ids = []
    
    for pet_id in tqdm(ids):
        # Tu ruta original
        path = os.path.join(img_dir, f"{pet_id}-1.jpg")
        
        if os.path.exists(path):
            try:
                # Carga igual que tu script
                img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
                img = img_to_array(img)
                img = preprocess_input(img)
                
                batch_imgs.append(img)
                batch_ids.append(pet_id)
                
                # Cuando llenamos el "carrito" (batch), predecimos todo junto
                if len(batch_imgs) >= BATCH_SIZE:
                    batch_stack = np.array(batch_imgs)
                    preds = model.predict(batch_stack, verbose=0)
                    
                    for i, pid in enumerate(batch_ids):
                        features[pid] = preds[i]
                    
                    batch_imgs = []
                    batch_ids = []
            except:
                pass

    # Procesar las que sobraron
    if len(batch_imgs) > 0:
        batch_stack = np.array(batch_imgs)
        preds = model.predict(batch_stack, verbose=0)
        for i, pid in enumerate(batch_ids):
            features[pid] = preds[i]

    # Crear DataFrame (EfficientNetB0 max pooling da 1280 columnas)
    if features:
        cols = [f"img_feat_{i+1}" for i in range(1280)]
        df = pd.DataFrame.from_dict(features, orient='index', columns=cols)
        df.index.name = "PetID"
        return df
    else:
        return pd.DataFrame()


# 1. Leer IDs
train_csv = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/train/train.csv')
test_csv = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/test/test.csv')


# 2. Rutas
train_dir = '/kaggle/input/petfinder-adoption-prediction/train_images'
test_dir = '/kaggle/input/petfinder-adoption-prediction/test_images'


# 3. Procesar TRAIN
print("\n--- INICIANDO TRAIN ---")
train_img = process_images_safe(train_csv, train_dir)
gc.collect() # Limpiar RAM


# 4. Procesar TEST
print("\n--- INICIANDO TEST ---")
test_img = process_images_safe(test_csv, test_dir)
gc.collect()


# 5. Guardar
print("\nðŸ’¾ Guardando archivos parquet...")
train_img.to_parquet("train_img.parquet")
test_img.to_parquet("test_img.parquet")
print("âœ… Â¡LISTO! Haz clic en 'Save & Run All' ahora.")

