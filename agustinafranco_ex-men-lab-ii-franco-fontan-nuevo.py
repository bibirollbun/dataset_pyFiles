!pip install "protobuf==3.20.3" --force-reinstall
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import os
import gc
import glob
import re
import warnings
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm.notebook import tqdm


#Integrantes del grupo:
# Mariana Fontan
# Agustina Franco


# ConfiguraciÃ³n
warnings.filterwarnings("ignore")
pd.set_option('mode.chained_assignment', None)


# --- 1. CONFIGURACIÃ“N GPU ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("âœ… GPU Activada para procesamiento de imÃ¡genes.")
    except RuntimeError as e:
        print(e)
else:
    print("âš ï¸� ADVERTENCIA: No se detectÃ³ GPU. El procesamiento de imÃ¡genes serÃ¡ lento.")


# --- 2. CARGAR DATOS BASE ---
print("--- Cargando CSVs ---")
train = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/train/train.csv').set_index("PetID")
test = pd.read_csv('/kaggle/input/petfinder-adoption-prediction/test/test.csv').set_index("PetID")


# --- 3. CARGAR TEXTO (INPUT EXTERNO) ---
# Buscamos el parquet de texto que SÃ� pudiste guardar
def find_text_parquet(pattern):
    files = glob.glob(f'/kaggle/input/**/*{pattern}*', recursive=True)
    return files[0] if files else None

path_txt_train = find_text_parquet("train_text.parquet")
path_txt_test = find_text_parquet("test_text.parquet")

if path_txt_train:
    print(f"âœ… Texto encontrado: {path_txt_train}")
    train = train.join(pd.read_parquet(path_txt_train), how='left')
    test = test.join(pd.read_parquet(path_txt_test), how='left')
else:
    print("âš ï¸� NO se encontrÃ³ input de texto. Se entrenarÃ¡ sin texto.")


# --- 4. PROCESAMIENTO DE IMÃ�GENES EN VIVO ---
# Definimos la funciÃ³n aquÃ­ mismo para correrla ahora
def process_images_live(df, img_dir, dataset_name):
    print(f"ğŸš€ Procesando imÃ¡genes de {dataset_name} en vivo...")
    
    # ConfiguraciÃ³n Ultra-RÃ¡pida
    IMG_SIZE = 128
    BATCH_SIZE = 64
    
    # Cargar modelo ligero
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    ids = df.index.values
    features = {}
    batch_imgs = []
    batch_ids = []
    
    for pet_id in tqdm(ids):
        path = os.path.join(img_dir, f"{pet_id}-1.jpg")
        
        # Si existe la imagen, la procesamos
        if os.path.exists(path):
            try:
                img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
                img = img_to_array(img)
                img = preprocess_input(img)
                
                batch_imgs.append(img)
                batch_ids.append(pet_id)
                
                # Predecir batch
                if len(batch_imgs) >= BATCH_SIZE:
                    preds = model.predict(np.array(batch_imgs), verbose=0)
                    for i, pid in enumerate(batch_ids):
                        features[pid] = preds[i]
                    batch_imgs = []
                    batch_ids = []
            except:
                pass
    
    # Remanente
    if len(batch_imgs) > 0:
        preds = model.predict(np.array(batch_imgs), verbose=0)
        for i, pid in enumerate(batch_ids):
            features[pid] = preds[i]
            
    # Crear DF
    if features:
        cols = [f"img_feat_{i}" for i in range(1280)]
        return pd.DataFrame.from_dict(features, orient='index', columns=cols)
    else:
        return pd.DataFrame()

# EJECUTAR PROCESAMIENTO DE IMÃ�GENES
train_dir = '/kaggle/input/petfinder-adoption-prediction/train_images'
test_dir = '/kaggle/input/petfinder-adoption-prediction/test_images'

# Procesar y Unir Train
img_features_train = process_images_live(train, train_dir, "TRAIN")
train = train.join(img_features_train, how='left')
del img_features_train
gc.collect()

# Procesar y Unir Test
img_features_test = process_images_live(test, test_dir, "TEST")
test = test.join(img_features_test, how='left')
del img_features_test
gc.collect()

print(f"âœ… Datos listos. Dimensiones finales: Train {train.shape}, Test {test.shape}")



# --- 5. PREPARACIÃ“N PARA MODELO ---
if "AdoptionSpeed" in test.columns:
    test = test.drop("AdoptionSpeed", axis=1)

y = train['AdoptionSpeed']
X = train.drop(['AdoptionSpeed'], axis=1)

# CategorÃ­as
cat_cols = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
            'Color3', 'FurLength', 'MaturitySize', 'Vaccinated', 'Dewormed',
            'Sterilized', 'Health', 'State']

for c in [col for col in cat_cols if col in X.columns]:
    X[c] = X[c].astype('category').cat.codes
    test[c] = test[c].astype('category').cat.codes

# Solo numÃ©ricas y limpiar nombres
X = X.select_dtypes(include=[np.number]).fillna(0)
test = test.select_dtypes(include=[np.number]).fillna(0)

# Alinear
cols = X.columns.intersection(test.columns)
X = X[cols]
test = test[cols]

# Limpiar caracteres raros en nombres de columnas
X.columns = [re.sub(r'[^A-Za-z0-9_]+', '', c) for c in X.columns]
test.columns = [re.sub(r'[^A-Za-z0-9_]+', '', c) for c in test.columns]



# --- 6. ENTRENAMIENTO (LightGBM) ---
print("--- Entrenando LightGBM ---")
NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds_proba = np.zeros((len(test), 5))

# ParÃ¡metros rÃ¡pidos
lgbm_params = {
    'objective': 'multiclass',
    'num_class': 5,
    'metric': 'multi_logloss',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 7,
    'num_leaves': 31,
    'n_jobs': -1,
    'verbosity': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(
        X.iloc[train_idx], y.iloc[train_idx],
        eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
        callbacks=[lgb.early_stopping(30, verbose=False)]
    )
    oof_preds[val_idx] = model.predict(X.iloc[val_idx])
    test_preds_proba += model.predict_proba(test) / NUM_FOLDS
    print(f"Fold {fold+1} OK")



# --- 7. SUBMISSION ---
kappa = cohen_kappa_score(y, oof_preds, weights='quadratic')
print(f"--- Kappa Score: {kappa:.4f} ---")

submission = pd.DataFrame({
    'PetID': test.index,
    'AdoptionSpeed': np.argmax(test_preds_proba, axis=1).astype(int)
})
submission.to_csv('submission.csv', index=False)
print("Â¡Submission generado!")

