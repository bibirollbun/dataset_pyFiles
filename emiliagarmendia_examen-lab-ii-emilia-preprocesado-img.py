# Importación de Librerías
import os 

import numpy as np
import pandas as pd

from tqdm import tqdm

from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input


# Para suprimir warnings innecesarios
import warnings
warnings.filterwarnings('ignore')


    # Configuración de rutas 
TRAIN_IMG_PATH = '../input/petfinder-adoption-prediction/train_images'
TEST_IMG_PATH = '../input/petfinder-adoption-prediction/test_images'

# Parámetros de procesamiento
IMG_SIZE = (224, 224)  # Tamaño estándar para EfficientNetB0
BATCH_SIZE = 32        # Procesar 32 imágenes a la vez (más eficiente que 1 a 1)
FEATURE_DIM = 1280     # Dimensión de salida de EfficientNetB0


def load_and_preprocess_image(img_path, target_size=IMG_SIZE):
    """
    Carga y preprocesa una imagen individual.
    
    Args:
        img_path: ruta completa a la imagen
        target_size: tupla (height, width) para redimensionar
    
    Returns:
        numpy array con la imagen preprocesada o None si hay error
    """
    try:
        # Cargar imagen y redimensionar
        img = load_img(img_path, target_size=target_size)
        # Convertir a array numpy
        img_array = img_to_array(img)
        # Preprocesar según EfficientNet
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error procesando {img_path}: {e}")
        return None


def extract_features_batch(image_paths, model, batch_size=BATCH_SIZE):
    """
    Extrae características de un lote de imágenes.
    
    Args:
        image_paths: lista de rutas de imágenes
        model: modelo pre-entrenado para extracción de features
        batch_size: número de imágenes a procesar simultáneamente
    
    Returns:
        DataFrame con PetID como índice y features como columnas
    """
    features_list = []
    pet_ids = []
    
    # Procesar imágenes en lotes
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Procesando lotes"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        batch_ids = []
        
        # Cargar el lote de imágenes
        for img_path in batch_paths:
            img = load_and_preprocess_image(img_path)
            if img is not None:
                batch_images.append(img)
                # Extraer PetID del nombre del archivo
                pet_id = os.path.basename(img_path).replace('-1.jpg', '')
                batch_ids.append(pet_id)
        
        # Si hay imágenes válidas en el lote, extraer features
        if batch_images:
            batch_array = np.array(batch_images)
            # Extraer features del lote completo (más eficiente)
            batch_features = model.predict(batch_array, verbose=0)
            features_list.extend(batch_features)
            pet_ids.extend(batch_ids)
    
    # Crear DataFrame con los resultados
    feature_names = [f"img_feat_{i+1}" for i in range(FEATURE_DIM)]
    df_features = pd.DataFrame(features_list, columns=feature_names, index=pet_ids)
    
    return df_features


def get_image_paths(directory, suffix='-1.jpg'):
    """
    Obtiene lista de rutas de imágenes que terminan en un sufijo específico.
    
    Args:
        directory: directorio donde buscar imágenes
        suffix: terminación del archivo a buscar (default: primera foto '-1.jpg')
    
    Returns:
        lista de rutas completas
    """
    image_files = [f for f in os.listdir(directory) if f.endswith(suffix)]
    image_paths = [os.path.join(directory, f) for f in image_files]
    return image_paths


# Cargar EfficientNetB0 pre-entrenado en ImageNet
# include_top=False: remover capa de clasificación
# pooling="max": usar max pooling global para obtener vector de features
print("Cargando modelo EfficientNetB0...")
model = EfficientNetB0(weights='imagenet', include_top=False, pooling="max")
print(f"Modelo cargado. Dimensión de features: {FEATURE_DIM}")


pics = [f for f in os.listdir('../input/petfinder-adoption-prediction/test_images') if f.endswith("-1.jpg")]
test_img  = pd.DataFrame([], columns=[f"img_feat_{i + 1}" for i in range(1280)])
for p in tqdm(pics):
    img = load_img(f'../input/petfinder-adoption-prediction/test_images/{p}')
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    petId = p.strip("-1.jpg")
    test_img.loc[petId] = model.predict(img)[0]


# Obtener rutas de imágenes de entrenamiento
train_img_paths = get_image_paths(TRAIN_IMG_PATH)
print(f"Total de imágenes de train: {len(train_img_paths)}")

# Extraer features
print("\nExtrayendo características de imágenes de train...")
train_img_features = extract_features_batch(train_img_paths, model)

# Mostrar resultado
print(f"\nFeatures extraídos: {train_img_features.shape}")
print("\nPrimeras filas:")
display(train_img_features.head())


# Obtener rutas de imágenes de test
test_img_paths = get_image_paths(TEST_IMG_PATH)
print(f"Total de imágenes de test: {len(test_img_paths)}")

# Extraer features
print("\nExtrayendo características de imágenes de test...")
test_img_features = extract_features_batch(test_img_paths, model)

# Mostrar resultado
print(f"\nFeatures extraídos: {test_img_features.shape}")
print("\nPrimeras filas:")
display(test_img_features.head())


print("=== Verificación de Train ===")
print(f"Valores nulos: {train_img_features.isnull().sum().sum()}")
print(f"Valores infinitos: {np.isinf(train_img_features.values).sum()}")
print(f"Estadísticas básicas:")
print(train_img_features.describe())

print("\n=== Verificación de Test ===")
print(f"Valores nulos: {test_img_features.isnull().sum().sum()}")
print(f"Valores infinitos: {np.isinf(test_img_features.values).sum()}")
print(f"Estadísticas básicas:")
print(test_img_features.describe())


# Guardar features de train
train_img_features.to_parquet('train_img_features.parquet')
print("✓ Train features guardados en: train_img_features.parquet")

# Guardar features de test
test_img_features.to_parquet('test_img_features.parquet')
print("✓ Test features guardados en: test_img_features.parquet")

print("\n=== Resumen ===")
print(f"Train: {train_img_features.shape[0]} mascotas, {train_img_features.shape[1]} features")
print(f"Test: {test_img_features.shape[0]} mascotas, {test_img_features.shape[1]} features")




