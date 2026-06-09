!pip install py7zr -q
# Installation de la version compatible de protobuf. RedÃ©marrage du kernel nÃ©cessaire aprÃ¨s cette Ã©tape!
!pip install --quiet protobuf==3.20.*


import py7zr
import os
import glob
from pathlib import Path

# --- DÃ©finition des Chemins ---
INPUT_DIR = '/kaggle/input/statoil-iceberg-classifier-challenge/'
OUTPUT_DIR = '/kaggle/working/data/'

# CrÃ©er le rÃ©pertoire de destination
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# DÃ©compression des fichiers .7z
compressed_files = glob.glob(os.path.join(INPUT_DIR, '*.7z'))
print(f"DÃ©compression de {len(compressed_files)} archives 7z...")

for file_path in compressed_files:
    filename = Path(file_path).name
    try:
        with py7zr.SevenZipFile(file_path, mode='r') as archive:
            archive.extractall(path=OUTPUT_DIR)
        print(f"âœ… {filename} dÃ©compressÃ©.")
    except Exception as e:
        print(f"â�Œ Erreur lors de la dÃ©compression de {filename} : {e}")

print("-" * 40)

# --- NOUVELLE INSPECTION CRITIQUE pour dÃ©terminer le chemin final ---
# Nous savons qu'un sous-dossier 'data' a Ã©tÃ© crÃ©Ã©.
# Le rÃ©pertoire qui contient nos fichiers est donc probablement OUTPUT_DIR / 'data'
FINAL_DATA_DIR = os.path.join(OUTPUT_DIR, 'data')

# Fichiers que nous recherchons
FILES_TO_LOCATE = ['train.json', 'test.json', 'sample_submission.csv']
PATHS = {}

# S'assurer que le dossier FINAL_DATA_DIR existe et contient les fichiers
if os.path.isdir(FINAL_DATA_DIR):
    for filename in FILES_TO_LOCATE:
        PATHS[filename] = os.path.join(FINAL_DATA_DIR, filename)
        
    print(f"âœ… Chemins d'accÃ¨s finaux identifiÃ©s dans {FINAL_DATA_DIR}:")
    for name, path in PATHS.items():
        print(f"   {name}: {path}")

else:
    print(f"â�Œ Erreur : Le rÃ©pertoire de donnÃ©es final {FINAL_DATA_DIR} n'a pas Ã©tÃ© trouvÃ©.")
    PATHS = None # Mettre Ã  None pour Ã©viter les erreurs de la cellule suivante

# Nous allons utiliser les chemins dans les prochaines cellules.


import pandas as pd
import json
import os

# --- DÃ©finition des Chemins (HypothÃ¨se: /data/data/processed/) ---
DATA_ROOT = '/kaggle/working/data/data/processed/'

PATHS = {
    'train.json': os.path.join(DATA_ROOT, 'train.json'), 
    'test.json': os.path.join(DATA_ROOT, 'test.json'),
    'sample_submission.csv': os.path.join(DATA_ROOT, 'sample_submission.csv'),
}

print(f"Tentative de chargement en utilisant le chemin imbriquÃ© 'processed' : {DATA_ROOT}...")

# --- Chargement des DataFrames ---
try:
    # 1. Chargement des donnÃ©es d'entraÃ®nement (JSON)
    with open(PATHS['train.json'], 'r') as f:
        train_data = json.load(f)
        df_train = pd.DataFrame(train_data)

    # 2. Chargement des donnÃ©es de test (JSON)
    with open(PATHS['test.json'], 'r') as f:
        test_data = json.load(f)
        df_test = pd.DataFrame(test_data)
        
    # 3. Chargement du fichier de soumission (CSV)
    df_submission = pd.read_csv(PATHS['sample_submission.csv'])

    print(f"âœ… Chargement des trois DataFrames rÃ©ussi !")
    print(f"- EntraÃ®nement (df_train) : {len(df_train)} lignes")
    print(f"- Test (df_test) : {len(df_test)} lignes")
    print("\nğŸ“� AperÃ§u des donnÃ©es d'entraÃ®nement :")
    print(df_train.head(2))
    
    # --- 4. VÃ©rification du dÃ©sÃ©quilibre des classes ---
    class_counts = df_train['is_iceberg'].value_counts()
    print("\nğŸ“Š RÃ©partition des classes (is_iceberg) :")
    print(class_counts)
    
    ratio_iceberg = class_counts.get(1,0) / len(df_train)
    print(f"\nğŸ’¡ Pourcentage d'icebergs : {ratio_iceberg*100:.2f}%")
    
    if ratio_iceberg < 0.4 or ratio_iceberg > 0.6:
        print("âš ï¸� Attention : classes dÃ©sÃ©quilibrÃ©es, prÃ©voir oversampling, class weights ou data augmentation.")
    else:
        print("âœ… Classes relativement Ã©quilibrÃ©es")
    
except FileNotFoundError:
    print(f"â�Œ Erreur critique : Le chemin imbriquÃ© '{DATA_ROOT}...' n'existe pas.")
    print("Action urgente : Veuillez exÃ©cuter la commande de listage ci-dessous pour trouver le chemin exact.")
    print("\n--- Diagnostic ---")
    print("Veuillez exÃ©cuter ceci dans une cellule pour nous donner l'arborescence exacte :")
    print("!ls -R /kaggle/working/data/")
    
except Exception as e:
    print(f"â�Œ Erreur lors du chargement (autre que fichier non trouvÃ©) : {e}")



import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

print("PrÃ©paration avancÃ©e des donnÃ©es d'entraÃ®nement...")

# --- 0. Conversion band_1 / band_2 en listes de float si besoin ---
df_train['band_1'] = df_train['band_1'].apply(lambda x: np.array(x, dtype=np.float32))
df_train['band_2'] = df_train['band_2'].apply(lambda x: np.array(x, dtype=np.float32))

# --- 1. Remodelage des images (5625 -> 75x75) ---
X_train_band1 = np.stack(df_train['band_1'].apply(lambda x: x.reshape(75, 75)))
X_train_band2 = np.stack(df_train['band_2'].apply(lambda x: x.reshape(75, 75)))

# --- 2. Normalisation des bandes (min-max par image) ---
def normalize_band(band):
    min_val = band.min(axis=(1,2), keepdims=True)
    max_val = band.max(axis=(1,2), keepdims=True)
    return (band - min_val) / (max_val - min_val + 1e-6)

band1_norm = normalize_band(X_train_band1)
band2_norm = normalize_band(X_train_band2)

# --- 3. CrÃ©ation d'un 3e canal moyen ---
band_avg = (band1_norm + band2_norm) / 2

# --- 4. Empilement final pour CNN/ViT ---
X_train_images = np.stack([band1_norm, band2_norm, band_avg], axis=-1)
print(f"Forme des images empilÃ©es : {X_train_images.shape} (N_images, 75, 75, 3 canaux)")

# --- 5. Gestion de l'angle d'incidence ---
inc_angle_train = df_train['inc_angle'].replace('na', np.nan).astype(float)
median_angle = np.nanmedian(inc_angle_train)
X_train_angle = inc_angle_train.fillna(median_angle).values.reshape(-1,1)

# Standardisation
angle_mean = X_train_angle.mean()
angle_std = X_train_angle.std()
X_train_angle_std = (X_train_angle - angle_mean) / (angle_std + 1e-6)
print(f"Angle standardisÃ© : mean={X_train_angle_std.mean():.2f}, std={X_train_angle_std.std():.2f}")

# --- 6. Features manuelles/statistiques par image ---
def extract_features(band1, band2):
    N = band1.shape[0]
    features = []
    for i in range(N):
        feats = []
        for b in [band1[i], band2[i]]:
            feats.append(b.mean())               # moyenne
            feats.append(b.std())                # Ã©cart type
            feats.append(skew(b.flatten()))     # skewness
            feats.append(kurtosis(b.flatten())) # kurtosis
        features.append(feats)
    return np.array(features, dtype=np.float32)

X_train_features = extract_features(X_train_band1, X_train_band2)
print(f"Forme des features manuelles/statistiques : {X_train_features.shape}")

# --- 7. Labels ---
Y_train = df_train['is_iceberg'].values.reshape(-1,1)

print("âœ… PrÃ©paration avancÃ©e terminÃ©e !")
print(f"- Images : {X_train_images.shape}")
print(f"- Angle standardisÃ© : {X_train_angle_std.shape}")
print(f"- Features manuelles : {X_train_features.shape}")
print(f"- Labels : {Y_train.shape}")


import matplotlib.pyplot as plt

# --- 1. DÃ©finition des Indices Ã  Comparer ---
# Le Navire que vous venez de voir
SHIP_INDEX = 9 
# Un Iceberg (nous prenons l'indice 1, qui est souvent un iceberg)
ICEBERG_INDEX = 1 

# --- 2. PrÃ©paration des DonnÃ©es d'Affichage ---

def get_image_data(index):
    """Fonction utilitaire pour extraire les donnÃ©es et les Ã©tiquettes par index."""
    data = {
        'band1': X_train_images[index, :, :, 0],
        'band2': X_train_images[index, :, :, 1],
        'label': "Iceberg" if Y_train[index] == 1 else "Navire (Ship)",
        'id': df_train.iloc[index]['id'],
        'angle': X_train_angle[index][0],
    }
    return data

ship_data = get_image_data(SHIP_INDEX)
iceberg_data = get_image_data(ICEBERG_INDEX)

# --- 3. Affichage de la Figure de Comparaison ---

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle("Comparaison : Navire vs. Iceberg (Bande 1 & Bande 2)", fontsize=16)

# PremiÃ¨re Ligne : Le Navire
# Bande 1 (HH) Navire
axes[0, 0].imshow(ship_data['band1'], cmap='gray')
axes[0, 0].title.set_text(f"Navire (ID: {ship_data['id']})\nBande 1 (HH)")
axes[0, 0].axis('off')

# Bande 2 (HV) Navire
axes[0, 1].imshow(ship_data['band2'], cmap='gray')
axes[0, 1].title.set_text(f"Navire (Angle: {ship_data['angle']:.2f}Â°)\nBande 2 (HV)")
axes[0, 1].axis('off')

# DeuxiÃ¨me Ligne : L'Iceberg
# Bande 1 (HH) Iceberg
axes[1, 0].imshow(iceberg_data['band1'], cmap='gray')
axes[1, 0].title.set_text(f"Iceberg (ID: {iceberg_data['id']})\nBande 1 (HH)")
axes[1, 0].axis('off')

# Bande 2 (HV) Iceberg
axes[1, 1].imshow(iceberg_data['band2'], cmap='gray')
axes[1, 1].title.set_text(f"Iceberg (Angle: {iceberg_data['angle']:.2f}Â°)\nBande 2 (HV)")
axes[1, 1].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuster pour le suptitle
plt.show()


import os
# DÃ©sactivation d'XLA pour Ã©viter les conflits CUDA (cuDNN/cuBLAS)
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Dropout,
    BatchNormalization, LayerNormalization, Reshape
)
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# ============================================================
# 1. SÃ©paration des donnÃ©es pour la validation
# ============================================================
# X_train_images: (N,75,75,3)
# X_train_angle_std: (N,1)
# X_train_features: (N,N_feat)
X_train_img, X_val_img, Y_train_data, Y_val, \
X_train_angle, X_val_angle, \
X_train_feat, X_val_feat = train_test_split(
    X_train_images, Y_train, X_train_angle_std, X_train_features,
    test_size=0.1, random_state=42
)

# --- Blindage / conversions robustes ---
Y_train_data = Y_train_data.astype("float32").reshape(-1,1)
Y_val = Y_val.astype("float32").reshape(-1,1)

X_train_angle = X_train_angle.astype(np.float32)
X_val_angle   = X_val_angle.astype(np.float32)
X_train_img   = np.nan_to_num(X_train_img, nan=0.0, posinf=1e4, neginf=-1e4)
X_val_img     = np.nan_to_num(X_val_img, nan=0.0, posinf=1e4, neginf=-1e4)
X_train_feat  = X_train_feat.astype(np.float32)
X_val_feat    = X_val_feat.astype(np.float32)

print(f"Shapes des donnÃ©es : Images {X_train_img.shape}, Angle {X_train_angle.shape}, Features {X_train_feat.shape}, Labels {Y_train_data.shape}")

# ============================================================
# 2. Bloc Transformer
# ============================================================
def transformer_block(x, embed_dim, num_heads):
    ln_1 = LayerNormalization(epsilon=1e-6)(x)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(ln_1, ln_1)
    attn_output = Dropout(0.1)(attn_output)
    x = keras.layers.add([attn_output, x])

    ln_2 = LayerNormalization(epsilon=1e-6)(x)
    ffn_output = Dense(embed_dim, activation="relu")(ln_2)
    ffn_output = Dense(embed_dim)(ffn_output)
    ffn_output = Dropout(0.1)(ffn_output)
    return keras.layers.add([ffn_output, x])

# ============================================================
# 3. Construction du modÃ¨le hybride CNN-ViT + angle + features
# ============================================================
input_img = Input(shape=(75,75,3), name='input_image')   # 3 canaux
input_angle = Input(shape=(1,), name='input_angle')
input_feat = Input(shape=(X_train_feat.shape[1],), name='input_features')

embed_dim = 64
num_heads = 4

# CNN initial
x = Conv2D(embed_dim, (3,3), activation='relu', padding='same')(input_img)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)  # pooling moins agressif

h_out, w_out = x.shape[1], x.shape[2]
num_patches = h_out * w_out
patch_tokens = Reshape((num_patches, embed_dim))(x)

x_tokens = transformer_block(patch_tokens, embed_dim, num_heads)
x_flat = Reshape((num_patches*embed_dim,))(x_tokens)
x_flat = Dense(256, activation='relu')(x_flat)
x_flat = Dropout(0.4)(x_flat)

# Angle branch
angle_out = Dense(16, activation='relu')(input_angle)

# Features branch
feat_out = Dense(32, activation='relu')(input_feat)

# Fusion
merged = concatenate([x_flat, angle_out, feat_out])
y = Dense(128, activation='relu')(merged)
y = Dropout(0.3)(y)
final_output = Dense(1, activation='sigmoid', name='output_prediction')(y)

model_vit_hybrid = Model(inputs=[input_img, input_angle, input_feat], outputs=final_output)
model_vit_hybrid.compile(loss='binary_crossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])

model_vit_hybrid.summary()

# ============================================================
# 4. EarlyStopping basÃ© sur le log loss pour Ã©viter overfitting
# ============================================================
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ============================================================
# 5. EntraÃ®nement
# ============================================================
history_vit = model_vit_hybrid.fit(
    x={'input_image': X_train_img, 'input_angle': X_train_angle, 'input_features': X_train_feat},
    y=Y_train_data,
    validation_data=({'input_image': X_val_img, 'input_angle': X_val_angle, 'input_features': X_val_feat}, Y_val),
    epochs=20,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print("\nâœ… EntraÃ®nement terminÃ©. PrÃªt pour l'infÃ©rence et la soumission.")



import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

print("--- 1. PrÃ©paration des donnÃ©es de test (df_test) ---")

# 1. Remodelage des images de test (5625 -> 75x75)
X_test_band1 = np.stack(df_test['band_1'].apply(lambda x: np.array(x).reshape(75,75)))
X_test_band2 = np.stack(df_test['band_2'].apply(lambda x: np.array(x).reshape(75,75)))

# --- Normalisation (min-max par image) ---
def normalize_band(band):
    min_val = band.min(axis=(1,2), keepdims=True)
    max_val = band.max(axis=(1,2), keepdims=True)
    return (band - min_val) / (max_val - min_val + 1e-6)

band1_norm = normalize_band(X_test_band1)
band2_norm = normalize_band(X_test_band2)
band_avg = (band1_norm + band2_norm)/2

# Empilement final 3 canaux
X_test_images = np.stack([band1_norm, band2_norm, band_avg], axis=-1)
print(f"Forme images test : {X_test_images.shape}")

# 2. Angle dâ€™incidence
inc_angle_test = df_test['inc_angle'].replace('na', np.nan).astype(float)
X_test_angle = inc_angle_test.fillna(median_angle).values.reshape(-1,1)

# Standardisation selon lâ€™entrainement
X_test_angle_std = (X_test_angle - angle_mean)/(angle_std + 1e-6)

# 3. Features manuelles/statistiques
def extract_features(band1, band2):
    N = band1.shape[0]
    features = []
    for i in range(N):
        feats = []
        for b in [band1[i], band2[i]]:
            feats.append(b.mean())
            feats.append(b.std())
            feats.append(skew(b.flatten()))
            feats.append(kurtosis(b.flatten()))
        features.append(feats)
    return np.array(features, dtype=np.float32)

X_test_features = extract_features(X_test_band1, X_test_band2)
print(f"Forme features test : {X_test_features.shape}")

# --- Blindage ---
X_test_images = np.nan_to_num(X_test_images, nan=0.0, posinf=1e4, neginf=-1e4)
X_test_angle_std = X_test_angle_std.astype(np.float32)
X_test_features = X_test_features.astype(np.float32)

print("\n--- 2. GÃ©nÃ©ration des PrÃ©dictions ---")
predictions = model_vit_hybrid.predict(
    {'input_image': X_test_images,
     'input_angle': X_test_angle_std,
     'input_features': X_test_features}
)

predictions = predictions.flatten()

# --- 2b. Clipping pour log loss ---
epsilon = 1e-5
predictions = np.clip(predictions, epsilon, 1 - epsilon)

print(f"âœ… GÃ©nÃ©ration de {len(predictions)} prÃ©dictions terminÃ©e.")

# --- 3. CrÃ©ation du fichier de soumission ---
submission = pd.DataFrame({'id': df_test['id'], 'is_iceberg': predictions})
print("ğŸ“� AperÃ§u du fichier de soumission :")
print(submission.head())

SUBMISSION_FILE_PATH = 'submission.csv'
submission.to_csv(SUBMISSION_FILE_PATH, index=False)
print(f"\nğŸ�‰ SuccÃ¨s ! Fichier de soumission crÃ©Ã© : {SUBMISSION_FILE_PATH}")

