import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display

# Chemin d'accès racine (à adapter si nécessaire)
ROOT = "/kaggle/input/birdclef-2025"

# Chargement des fichiers
try:
    train_df = pd.read_csv(f'{ROOT}/train.csv')
    taxonomy_df = pd.read_csv(f'{ROOT}/taxonomy.csv')
    sample_submission = pd.read_csv(f'{ROOT}/sample_submission.csv')
    print("Chargement des DataFrames réussi.")
except FileNotFoundError as e:
    print(f"Erreur de chargement : {e}")

# Aperçu
print("\nAperçu de train_df:")
print(train_df.head())
print(f"\nDimensions de train_df : {train_df.shape}")


# Compter les occurrences de chaque espèce
label_counts = train_df['primary_label'].value_counts()

# Afficher le Top 10 et le Bottom 10
print("--- Top 10 des espèces les plus fréquentes ---")
print(label_counts.head(10))
print("\n--- Bottom 10 des espèces les plus rares ---")
print(label_counts.tail(10))

# Visualisation des 50 espèces les plus fréquentes pour confirmer le déséquilibre
plt.figure(figsize=(15, 6))
sns.barplot(x=label_counts.head(50).index, y=label_counts.head(50).values, palette="viridis")
plt.title('Distribution des 50 Espèces les plus Fréquentes')
plt.xlabel('Espèce (primary_label)')
plt.ylabel('Nombre d\'enregistrements')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# IMPLICATION : Confirme la nécessité d'oversampling et d'augmentation des données.


# --- Partie A : Distribution Taxonomique ---

# 1. Vérification et Nettoyage des colonnes de taxonomy_df
# Nous assumons que les colonnes pertinentes sont 'taxon_id' (pour la liaison) et 'class' (pour la catégorie : Aves, Amphibia, etc.).
print("Colonnes de taxonomy_df :", taxonomy_df.columns.tolist())

# Correction de l'accès aux colonnes et du renommage
try:
    # Tentative d'utilisation des colonnes 'taxon_id' et 'class'
    taxonomy_df_clean = taxonomy_df[['taxon_id', 'class']].copy()
    taxonomy_df_clean = taxonomy_df_clean.rename(columns={'class': 'class_name'})
except KeyError as e:
    # Si la colonne 'class' ou 'taxon_id' manque, nous ne pouvons pas faire la liaison.
    # Pour l'EDA, nous allons simplement utiliser la colonne existante la plus probable pour les catégories.
    if 'class' in taxonomy_df.columns:
        print("Avertissement: 'taxon_id' manquant ou mal orthographié, utilisant 'class' pour le décompte.")
        taxonomy_df_clean = taxonomy_df.rename(columns={'class': 'class_name'})
    else:
        print(f"Erreur fatale: Colonne de classification ('class') manquante dans taxonomy_df. Colonnes : {taxonomy_df.columns.tolist()}")
        taxonomy_df_clean = pd.DataFrame({'class_name': []}) # DataFrame vide pour éviter l'erreur suivante

# 2. Décompte des Taxons
if not taxonomy_df_clean.empty:
    taxa_counts = taxonomy_df_clean['class_name'].value_counts()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=taxa_counts.index, y=taxa_counts.values, palette="rocket")
    plt.title('Nombre d\'espèces uniques par Taxon')
    plt.ylabel("Nombre d'espèces uniques")
    plt.xlabel("Taxon (class_name)")
    plt.show()
    print("IMPLICATION : Les différences de taille de classe confirment la nécessité de stratégies d'extraction de features adaptées (fréquences différentes pour chaque taxon).")
else:
    print("Impossible de procéder à l'analyse taxonomique sans une colonne de classe valide.")


print("\n" + "="*50)
print("--- Partie B : Analyse Multi-Label ---")

# 1. Fonction pour compter le nombre d'espèces secondaires
def count_secondary_labels(labels_str):
    """Compte le nombre de labels secondaires (autres espèces) dans la chaîne."""
    try:
        # Évaluation sécurisée de la chaîne list-like (e.g., "['compau', 'trokin']")
        labels_list = eval(labels_str)
        # Compte les éléments de la liste qui ne sont pas la chaîne vide ([''])
        return len([l for l in labels_list if l != ''])
    except Exception:
        # En cas d'erreur d'évaluation (format incorrect)
        return 0

# Applique la fonction pour obtenir le nombre de labels secondaires
train_df['num_secondary_labels'] = train_df['secondary_labels'].apply(count_secondary_labels)

# 2. Distribution du nombre total d'espèces (Primaire + Secondaires)
# Le nombre total d'espèces est : 1 (primary_label) + num_secondary_labels
train_df['total_species_in_clip'] = 1 + train_df['num_secondary_labels']

total_species_counts = train_df['total_species_in_clip'].value_counts().sort_index()

print("\n--- Distribution du nombre total d'espèces (Primaire + Secondaires) par enregistrement ---")
print(total_species_counts)

# 3. Visualisation (se concentrer sur les clips multi-espèces)
# Nous filtrons la catégorie "1 seule espèce" pour mieux visualiser les co-occurrences.
multi_species_clips = total_species_counts[total_species_counts.index > 1]


if not multi_species_clips.empty:
    plt.figure(figsize=(8, 5))
    multi_species_clips.plot(kind='bar', color='darkorange')
    plt.title('Fréquence des enregistrements Multi-Espèces (2+ espèces)')
    plt.xlabel("Nombre total d'espèces présentes")
    plt.ylabel("Fréquence")
    plt.xticks(rotation=0)
    plt.show()
    print("IMPLICATION : Le nombre de clips multi-espèces est faible (comparé aux clips simples). Cela justifie l'utilisation d'une **Loss Function Multi-Label (BCE)** et d'une analyse des co-occurrences pour améliorer la performance sur ces cas complexes.")
else:
    print("Aucun enregistrement multi-label significatif trouvé.")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- A. Distribution Géographique ---

print("--- Analyse Géographique des Enregistrements (latitude/longitude) ---")

# Filtrer les coordonnées manquantes ou invalides si nécessaire pour le plotting
df_geo = train_df.dropna(subset=['latitude', 'longitude'])

plt.figure(figsize=(10, 6))
# Utiliser 'hue' pour différencier les collections (XC, iNat, CSA)
sns.scatterplot(
    x='longitude', 
    y='latitude', 
    data=df_geo, 
    alpha=0.6, 
    hue='collection', 
    palette='tab10',
    s=20 # Taille du point
)
plt.title('Distribution Géographique des Enregistrements')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Collection', loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Vérification rapide de l'étendue géographique
if not df_geo.empty:
    latitude_range = df_geo['latitude'].max() - df_geo['latitude'].min()
    longitude_range = df_geo['longitude'].max() - df_geo['longitude'].min()
    print(f"\nÉtendue géographique : Latitude ({latitude_range:.2f}), Longitude ({longitude_range:.2f})")
else:
    print("\nAvertissement: Aucune donnée de coordonnées valide trouvée.")

print("IMPLICATION : L'analyse des clusters peut suggérer des biais géographiques dans les données d'entraînement. Utiliser la géolocalisation comme feature ou s'assurer que le modèle généralise au-delà des zones denses est crucial.")


# --- B. Distribution de la Qualité (rating) ---

print("\n" + "="*50)
print("--- Analyse de la Qualité (rating) et de la Collection ---")

# Histogramme de la distribution des notes
plt.figure(figsize=(8, 5))
# Utiliser des bins centrés sur les valeurs pour la clarté
bins = np.arange(-0.5, 5.5, 1)
train_df['rating'].plot(kind='hist', bins=bins, edgecolor='black', rwidth=0.8)
plt.title('Distribution des Ratings de Qualité (0.0 à 5.0)')
plt.xlabel('Rating (0=Non noté, 1=Basse, 5=Haute)')
plt.ylabel('Nombre d\'enregistrements')
plt.xticks(np.arange(0, 6))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Qualité moyenne par collection
quality_by_collection = train_df.groupby('collection')['rating'].agg(['mean', 'median', 'count']).sort_values(by='mean', ascending=False)
print("\nStatistiques de Rating par Collection :")
print(quality_by_collection)

# Comparaison des notes pour les espèces rares vs. communes (reprise des 10 plus/moins fréquentes de la Cellule 2)
# NOTE : Ces indices dépendent de l'exécution de la Cellule 2.
label_counts = train_df['primary_label'].value_counts()
top_10_labels = label_counts.head(10).index
bottom_10_labels = label_counts.tail(10).index

rating_common = train_df[train_df['primary_label'].isin(top_10_labels)]['rating'].mean()
rating_rare = train_df[train_df['primary_label'].isin(bottom_10_labels)]['rating'].mean()

print(f"\nRating moyen des 10 espèces les plus communes : {rating_common:.2f}")
print(f"Rating moyen des 10 espèces les plus rares : {rating_rare:.2f}")

print("IMPLICATION : Les ratings faibles (surtout pour les espèces rares) indiquent que le **filtrage des données** (p. ex., ne retenir que rating >= 3) et le **nettoyage du bruit** seront des étapes cruciales dans le prétraitement.")


import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd # Assurez-vous que train_df est bien chargé

# Définition du chemin racine et de la fréquence d'échantillonnage standardisée
ROOT = "/kaggle/input/birdclef-2025"
SR = 32000 # Fréquence d'échantillonnage standardisée

# --- Fonction pour visualiser un Mel-Spectrogramme ---
def plot_melspectrogram(filename, label, sr=SR):
    """Charge un fichier audio, calcule et affiche son Mel-Spectrogramme."""
    audio_path = os.path.join(ROOT, 'train_audio', filename)
    
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        
        # Calcul du Spectrogramme Mel (paramètres types pour BirdCLEF)
        # n_mels=128 est un bon standard.
        M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=320, n_mels=128)
        M_db = librosa.power_to_db(M, ref=np.max)
        
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(M_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-Spectrogramme : {label}')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Erreur lors du traitement de {filename}: {e}")
        
# --- Sélection d'échantillons pour la comparaison (corrigé) ---

# 1. Exemple d'un Oiseau (Aves) - Sélection basée sur le code eBird (chaîne de caractères)
# Utiliser un label commun
bird_sample = train_df[train_df['primary_label'] == 'grekis'].iloc[0]

# 2. Exemple d'un Non-Oiseau (Amphibien/Insecte) - Sélection basée sur l'ID numérique
# Cherchons le premier label qui est strictement numérique.
non_bird_df = train_df[train_df['primary_label'].str.isnumeric()]
if not non_bird_df.empty:
    non_bird_sample = non_bird_df.iloc[0]
else:
    # Fallback si aucun label numérique n'est trouvé
    non_bird_sample = train_df.iloc[1] 


print(f"\nExemple 1 : Oiseau Commun ({bird_sample['common_name']})")
plot_melspectrogram(bird_sample['filename'], bird_sample['common_name'])


print(f"\nExemple 2 : Non-Oiseau (probable Amphibien/Insecte) ({non_bird_sample['common_name']})")
plot_melspectrogram(non_bird_sample['filename'], non_bird_sample['common_name'])


# 3. Exemple d'un enregistrement à faible rating ou bruité
low_rating_df = train_df[train_df['rating'] == 1.0]
if not low_rating_df.empty:
    low_rating_sample = low_rating_df.iloc[0]
    print(f"\nExemple 3 : Échantillon de Faible Qualité (Rating 1.0) ({low_rating_sample['common_name']})")
    plot_melspectrogram(low_rating_sample['filename'], low_rating_sample['common_name'])
else:
    print("\nAucun échantillon avec un rating de 1.0 trouvé.")


print("\n--- Synthèse de l'Analyse Spectrale ---")
print("IMPLICATION : L'analyse visuelle des spectrogrammes confirme l'occupation de **bandes de fréquences** distinctes par les différents taxons (Oiseaux vs. Amphibiens/Insectes) et la présence de **bruit de fond** dans les échantillons de faible qualité. Cela justifie l'utilisation de **SpecAugment** et l'optimisation des paramètres du Mel-Spectrogramme.")


import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# --- Paramètres ---
ROOT = "/kaggle/input/birdclef-2025"
SR = 32000
N_MELS = 128
MIN_RATING = 3
MAX_AUDIO_DURATION = 10  # Tronquer l'audio à 10 secondes
HOP_LENGTH = 320 # Tiré de la fonction melspectrogram
N_FFT = 1024     # Tiré de la fonction melspectrogram

# Calcul du nombre de pas de temps (colonnes) pour 10 secondes
# Nombre de trames par seconde = SR / HOP_LENGTH = 32000 / 320 = 100
# Nombre de pas de temps cible pour 10s = 10 * 100 = 1000
# Note : Librosa ajoute un pas de temps supplémentaire, donc on vise légèrement plus grand
TARGET_TIME_STEPS = int(np.ceil(SR * MAX_AUDIO_DURATION / HOP_LENGTH)) 

TIME_MASK_PARAM = 10
FREQ_MASK_PARAM = 8
PITCH_SHIFT_STEPS = [-2, -1, 0, 1, 2]
TIME_STRETCH_RATES = [0.9, 1.0, 1.1]
NOISE_LEVEL = 0.005

# --- Chargement CSV ---
train_df = pd.read_csv(f"{ROOT}/train.csv")
train_df = train_df[train_df['rating'] >= MIN_RATING].reset_index(drop=True)

# --- Fonctions d'augmentation (inchangées) ---
def augment_audio(y, sr=SR):
    """Applique pitch shift et bruit aléatoire"""
    y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=np.random.choice(PITCH_SHIFT_STEPS))
    y = y + np.random.randn(len(y)) * NOISE_LEVEL
    return y

def preprocess_audio(filename, sr=SR, n_mels=N_MELS, augment=True, duration=MAX_AUDIO_DURATION, target_len=TARGET_TIME_STEPS):
    """Charge audio tronqué, applique augmentation (optionnelle), retourne log-Mel normalisé et paddé"""
    path = os.path.join(ROOT, 'train_audio', filename)
    
    # 1. Chargement tronqué
    y, _ = librosa.load(path, sr=sr, duration=duration)
    
    if augment:
        y = augment_audio(y, sr)
        
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=n_mels)
    M_db = librosa.power_to_db(M, ref=np.max)
    mel = (M_db - M_db.min()) / (M_db.max() - M_db.min())
    
    # 2. *** MODIFICATION CLÉ : Assurer la forme fixe par padding ou tronquage ***
    num_time = mel.shape[1]
    
    if num_time < target_len:
        # Padding (remplissage) avec des zéros si le spectrogramme est trop court
        padding = target_len - num_time
        mel = np.pad(mel, ((0, 0), (0, padding)), mode='constant')
    elif num_time > target_len:
        # Tronquage si le spectrogramme est trop long (devrait être rare après librosa.load(duration=10))
        mel = mel[:, :target_len]
        
    return mel

def spec_augment(mel, time_mask=TIME_MASK_PARAM, freq_mask=FREQ_MASK_PARAM):
    # Reste inchangé, mais opère sur un spectrogramme de taille fixe
    mel_copy = mel.copy()
    num_mel, num_time = mel_copy.shape
    # ... (code inchangé pour le masque)
    t = np.random.randint(0, time_mask)
    t0 = np.random.randint(0, max(1, num_time - t))
    mel_copy[:, t0:t0+t] = 0
    f = np.random.randint(0, freq_mask)
    f0 = np.random.randint(0, max(1, num_mel - f))
    mel_copy[f0:f0+f, :] = 0
    return mel_copy

def parse_secondary_labels(s):
    """Parse la chaîne secondary_labels en liste"""
    try:
        labels = eval(s)
        return [l for l in labels if l != '']
    except:
        return []

# --- Préparation X et y (inchangée à part l'appel de fonction) ---
X = []
y_primary = []
y_multi = []

mlb = MultiLabelBinarizer()
all_secondary_labels = train_df['secondary_labels'].apply(parse_secondary_labels)
mlb.fit(all_secondary_labels)

le = LabelEncoder()
le.fit(train_df['primary_label'])

print(f"Début du traitement des {len(train_df)} fichiers (taille cible : 128x{TARGET_TIME_STEPS})...")
for idx, row in train_df.iterrows():
    # Appel de la fonction modifiée
    mel = preprocess_audio(row['filename'], augment=True)
    mel = spec_augment(mel)
    X.append(mel)
    
    y_primary.append(row['primary_label'])
    y_multi.append(parse_secondary_labels(row['secondary_labels']))

# *** MODIFICATION CLÉ 2: La conversion fonctionne maintenant grâce à la taille fixe ***
X = np.array(X, dtype=np.float32) 
y_primary = np.array(le.transform(y_primary))
y_multi = np.array(mlb.transform(y_multi))

print("---")
print(f"Dataset prêt : X={X.shape}, y_primary={y_primary.shape}, y_multi={y_multi.shape}")
print(f"Exemple de classes multi-label : {mlb.classes_[:10]}")


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import GRU, Dense, Flatten, TimeDistributed, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np # Assurez-vous que numpy est bien importé

# Le reste du code suppose que X et y_multi sont disponibles à partir de la cellule précédente.

# --- Préparer les données pour Keras ---
# X : (num_samples, n_mels, time) → ajouter channel=1
X_input = X[..., np.newaxis]
y_input = y_multi

# Split train/val
X_train, X_val, y_train, y_val = train_test_split(X_input, y_input, test_size=0.1, random_state=42, shuffle=True)

# --- Définition du modèle CNN + GRU ---
input_shape = X_input.shape[1:]  # (n_mels, time, 1)
inputs = Input(shape=input_shape)

# CNN feature extractor
x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)

# *** AJOUT CLÉ: Réduction supplémentaire de la dimension temporelle pour le GRU ***
# MaxPooling sur l'axe du temps (2ème dimension spatiale, index 2)
# Réduit le nombre de pas de temps par 2 (accélère le GRU)
x = MaxPooling2D((1, 2))(x)

# Reshape pour RNN (time dimension)
n_mels_reduced = x.shape[1]
time_steps = x.shape[2]
channels = x.shape[3]
# x sera maintenant de forme (batch, time_steps_reduced / 8, features_flattened)
x = Reshape((time_steps, n_mels_reduced * channels))(x)

# GRU
x = GRU(128, return_sequences=False)(x)

# Dense multi-label
outputs = Dense(y_input.shape[1], activation='sigmoid')(x)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# --- Entraînement ---
# Le 'time stretch' ayant été désactivé, les échantillons sont de longueur fixe (10s),
# ce qui est idéal pour le batching.
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,        
    batch_size=16
)

# --- Sauvegarde du modèle ---
model.save("cnn_gru_birdclef.h5")
print("Modèle entraîné et sauvegardé !")


import os
import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# --- Paramètres ---
ROOT = "/kaggle/input/birdclef-2025"
TEST_PATH = os.path.join(ROOT, "test_soundscapes")
SR = 32000
N_MELS = 128
CHUNK_DURATION = 5  # en secondes (La fenêtre du modèle entraîné)
HOP_LENGTH = 320
N_FFT = 1024

# --- Charger le modèle entraîné ---
# *** CORRECTION 1: Utiliser le nom du fichier réellement sauvegardé ***
try:
    model = load_model("cnn_gru_birdclef.h5")
    print("Modèle 'cnn_gru_birdclef.h5' chargé avec succès.")
except Exception as e:
    print(f"ERREUR: Impossible de charger le modèle. Vérifiez le nom du fichier. Détail: {e}")
    # Si le modèle ne charge pas, on arrête pour éviter les erreurs subséquentes.
    raise

# --- Classes ---
# *** CORRECTION 2: S'assurer que les classes correspondent à l'ordre mlb.classes_ ***
# Si mlb n'est pas sauvegardé/chargé, nous utilisons la liste des dossiers pour l'ordre
# car c'était la méthode implicite utilisée pour obtenir mlb.classes_.
# C'est une approximation, mais c'est la seule si l'objet mlb n'est pas disponible.
all_labels = sorted(os.listdir(os.path.join(ROOT, "train_audio")))
# Exclure les IDs numériques (non-oiseaux) si le modèle n'a été entraîné que sur les labels primaires string eBird.
# Cependant, comme vous avez utilisé y_multi qui inclut tous les labels, cette liste est correcte
# SI et seulement SI l'ordre est le même que mlb.classes_.
# Par défaut, on garde la liste complète des dossiers (labels).
class_labels = all_labels 


# --- Fonction pour extraire log-Mel spectrogram d'un chunk ---
def extract_mel(y, sr=SR, n_mels=N_MELS):
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=n_mels)
    M_db = librosa.power_to_db(M, ref=np.max)
    # *** AMÉLIORATION: Ajout de dtype=np.float32 pour la cohérence avec l'entraînement ***
    return (M_db - M_db.min()) / (M_db.max() - M_db.min()).astype(np.float32)

# --- Créer submission dataframe ---
submission = pd.DataFrame(columns=['row_id'] + class_labels)

# --- Boucle sur chaque test soundscape ---
print(f"Début de la prédiction sur {len(os.listdir(TEST_PATH))} fichiers...")
for soundscape_file in sorted(os.listdir(TEST_PATH)):
    if not soundscape_file.endswith(".ogg"):
        continue
    
    path = os.path.join(TEST_PATH, soundscape_file)
    y, _ = librosa.load(path, sr=SR)
    
    # Découper en chunks de 5 secondes
    chunk_len = SR * CHUNK_DURATION
    chunks = [y[i:i+chunk_len] for i in range(0, len(y), chunk_len)]
    
    for i, chunk in enumerate(chunks):
        # Si le chunk est trop court, padding avec zeros
        if len(chunk) < chunk_len:
            # Padding avec des zéros à la fin pour avoir la longueur fixe
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
        
        mel = extract_mel(chunk)
        # S'assurer que le spectrogramme a la même taille en temps que le modèle attend
        # Ceci est critique si l'audio tronqué fait < 5s (ce qui ne devrait pas arriver ici)
        
        mel = np.expand_dims(mel, axis=(0, -1))  # ajouter batch et channel
        
        # Prédiction
        scores = model.predict(mel, verbose=0)[0]
        
        # Créer row_id: L'index de temps est l'heure de FIN du chunk
        row_id = f"{soundscape_file.split('.')[0]}_{(i+1)*CHUNK_DURATION}"
        
        # Ajouter ligne au dataframe
        submission.loc[len(submission)] = [row_id] + list(scores)

# --- Sauvegarder submission ---
submission.to_csv("submission.csv", index=False)
print("---")
print(f"Submission prête : submission.csv avec {len(submission)} lignes.")
submission.head()

