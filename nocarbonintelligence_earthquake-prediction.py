import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# --- 1. SETUP & CHARGEMENT DES DONNÉES PAR SEGMENTS ---

# Chemin d'accès racine (selon la convention Kaggle)
ROOT_PATH = '../input/LANL-Earthquake-Prediction'
TRAIN_FILE = 'train.csv'

# Définition des paramètres de segmentation pour éviter la panne de mémoire
SEGMENT_SIZE = 150000 
CHUNK_SIZE = 150000 
N_SEGMENTS = 4194 # Nombre total de segments de 150k

# Initialisation du nouveau DataFrame pour stocker les features
segments_features = pd.DataFrame(index=range(N_SEGMENTS), 
                                 columns=['mean', 'std', 'max', 'min', 'time_to_failure'])

# Types de données optimisés
DTYPE = {'acoustic_data': np.int16, 'time_to_failure': np.float32}

start_time = time.time()
print(f"Démarrage de l'extraction des {N_SEGMENTS} segments à partir de {TRAIN_FILE}...")

# Création de l'itérateur pour lire le fichier par morceaux
chunk_iterator = pd.read_csv(os.path.join(ROOT_PATH, TRAIN_FILE), 
                             dtype=DTYPE, 
                             iterator=True, 
                             chunksize=CHUNK_SIZE)

i = 0
for chunk in chunk_iterator:
    # Récupérer les données acoustiques
    acoustic_data = chunk['acoustic_data']
    
    # 1. Extraction de la valeur cible (Target) : dernière valeur du segment
    target = chunk['time_to_failure'].iloc[-1]
    
    # 2. Extraction des caractéristiques de base
    segments_features.loc[i, 'time_to_failure'] = target
    segments_features.loc[i, 'mean'] = acoustic_data.mean()
    segments_features.loc[i, 'std'] = acoustic_data.std()
    segments_features.loc[i, 'max'] = acoustic_data.max()
    segments_features.loc[i, 'min'] = acoustic_data.min()
    
    i += 1
    # Affichage de la progression
    if i % 1000 == 0:
        print(f"Progression : {i} segments traités / {N_SEGMENTS}")

end_time = time.time()
print(f"Analyse des {N_SEGMENTS} segments terminée en {round(end_time - start_time, 2)} secondes.")


# --- 2. INSPECTION ET STATISTIQUES SUR LE NOUVEAU DATAFRAME ---

print("\n--- Inspection du Nouveau DataFrame de Features ---")
print(f"Dimensions du DataFrame de features: {segments_features.shape}")
print("Aperçu des 5 premières lignes du DataFrame de features:")
print(segments_features.head())

print("\n--- Statistiques Descriptives des Features (Segments) ---")
print(segments_features.describe())


# --- 3. CORRÉLATION ---

# Calcul et affichage de la corrélation entre les features et la cible
print("\n--- Corrélation de Pearson (Segment Features vs. Target) ---")
correlation_matrix = segments_features.corr()
print(correlation_matrix['time_to_failure'].sort_values(ascending=False))

# Visualisation des corrélations (Heatmap)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='viridis')
plt.title('Matrice de Corrélation des Features du Segment')
plt.show()


# Fonction pour calculer un ensemble plus riche de features pour un segment de 150k
def extract_advanced_features(chunk):
    acoustic_data = chunk['acoustic_data']
    
    # 1. Caractéristiques de base (déjà calculées, mais incluses pour la complétude)
    features = {
        'mean': acoustic_data.mean(),
        'std': acoustic_data.std(),
    }
    
    # 2. Caractéristiques d'Amplitude/Énergie
    features['abs_mean'] = np.mean(np.abs(acoustic_data))
    features['abs_std'] = np.std(np.abs(acoustic_data))
    features['rms'] = np.sqrt(np.mean(acoustic_data**2))
    features['peak_to_peak'] = acoustic_data.max() - acoustic_data.min()

    # 3. Caractéristiques Statistiques Avancées
    features['skew'] = acoustic_data.skew()
    features['kurt'] = acoustic_data.kurtosis()
    features['q01'] = np.quantile(acoustic_data, 0.01)
    features['q99'] = np.quantile(acoustic_data, 0.99)
    
    # 4. Caractéristiques de Fréquence (Approximation par Z-Crossings)
    # Compte le nombre de fois où le signal passe d'une valeur positive à négative (ou vice versa)
    features['z_crossings'] = ((acoustic_data.values[:-1] * acoustic_data.values[1:]) < 0).sum()
    
    return features

# --- RE-EXECUTION AVEC LA NOUVELLE FONCTION ---

# Redéfinition du DataFrame pour inclure les nouvelles features
feature_names = ['time_to_failure', 'mean', 'std', 'abs_mean', 'abs_std', 
                 'rms', 'peak_to_peak', 'skew', 'kurt', 'q01', 'q99', 'z_crossings']
segments_features_advanced = pd.DataFrame(index=range(N_SEGMENTS), columns=feature_names)
DTYPE = {'acoustic_data': np.int16, 'time_to_failure': np.float32}

chunk_iterator = pd.read_csv(os.path.join(ROOT_PATH, TRAIN_FILE), dtype=DTYPE, iterator=True, chunksize=CHUNK_SIZE)

i = 0
for chunk in chunk_iterator:
    # Récupération de la cible
    segments_features_advanced.loc[i, 'time_to_failure'] = chunk['time_to_failure'].iloc[-1]
    
    # Calcul des features avancées
    new_features = extract_advanced_features(chunk)
    
    for key, value in new_features.items():
        segments_features_advanced.loc[i, key] = value
        
    i += 1
    if i % 1000 == 0:
        print(f"Progression (Advanced Features): {i} segments traités / {N_SEGMENTS}")

# Affichage des corrélations des features avancées
print("\n--- Corrélation des Features Avancées avec la Cible ---")
print(segments_features_advanced.corr()['time_to_failure'].sort_values(ascending=False))


# Conversion en types numériques pour s'assurer que le modèle fonctionne
segments_features_advanced = segments_features_advanced.apply(pd.to_numeric)

# Séparation des features (X) et de la cible (y)
X = segments_features_advanced.drop('time_to_failure', axis=1)
y = segments_features_advanced['time_to_failure']


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import os
import time

# --- SETUP ---
ROOT_PATH = '../input/LANL-Earthquake-Prediction'
TRAIN_FILE = 'train.csv'
SEGMENT_SIZE = 150000
CHUNK_SIZE = 150000
N_SEGMENTS = 4194
DTYPE = {'acoustic_data': np.int16, 'time_to_failure': np.float32}

# --- 1. FONCTION D'INGÉNIERIE DES CARACTÉRISTIQUES AVANCÉES ---
def extract_advanced_features(chunk):
    acoustic_data = chunk['acoustic_data'].values.astype(np.float32)

    # a) Ajouter un petit bruit gaussien
    noise = np.random.normal(0, 0.5, len(acoustic_data))
    acoustic_data += noise

    # b) Centrer par la médiane
    acoustic_data -= np.median(acoustic_data)

    features = {}

    # --- Caractéristiques de base ---
    features['mean'] = acoustic_data.mean()
    features['std'] = acoustic_data.std()
    features['max'] = acoustic_data.max()
    features['min'] = acoustic_data.min()

    # --- Amplitude / énergie ---
    features['abs_mean'] = np.mean(np.abs(acoustic_data))
    features['abs_std'] = np.std(np.abs(acoustic_data))
    features['rms'] = np.sqrt(np.mean(acoustic_data**2))
    features['peak_to_peak'] = acoustic_data.max() - acoustic_data.min()

    # --- Statistiques avancées ---
    features['skew'] = pd.Series(acoustic_data).skew()
    features['kurt'] = pd.Series(acoustic_data).kurtosis()
    features['q01'] = np.quantile(acoustic_data, 0.01)
    features['q99'] = np.quantile(acoustic_data, 0.99)

    # --- Fréquence approximative ---
    features['z_crossings'] = ((acoustic_data[:-1] * acoustic_data[1:]) < 0).sum()

    # --- Caractéristiques temporelles ---
    diff = np.diff(acoustic_data)
    features['mean_diff'] = diff.mean()
    features['std_diff'] = diff.std()
    # Rolling windows simples (ex: 100 points)
    window = 100
    features['rolling_std_mean'] = pd.Series(acoustic_data).rolling(window).std().mean()
    features['rolling_mean_mean'] = pd.Series(acoustic_data).rolling(window).mean().mean()

    return features

# --- 2. EXTRACTION DES FEATURES SUR TOUS LES SEGMENTS ---
feature_names = [
    'time_to_failure', 'mean', 'std', 'max', 'min', 'abs_mean', 'abs_std',
    'rms', 'peak_to_peak', 'skew', 'kurt', 'q01', 'q99', 'z_crossings',
    'mean_diff', 'std_diff', 'rolling_std_mean', 'rolling_mean_mean', 'cycle_id'
]

segments_features_model = pd.DataFrame(index=range(N_SEGMENTS), columns=feature_names)
chunk_iterator = pd.read_csv(os.path.join(ROOT_PATH, TRAIN_FILE), dtype=DTYPE,
                             iterator=True, chunksize=CHUNK_SIZE)

i = 0
last_ttf = 0
cycle_count = 0

for chunk in chunk_iterator:
    current_ttf = chunk['time_to_failure'].iloc[-1]

    if current_ttf > last_ttf and last_ttf < 1:
        cycle_count += 1

    segments_features_model.loc[i, 'time_to_failure'] = current_ttf
    segments_features_model.loc[i, 'cycle_id'] = cycle_count

    new_features = extract_advanced_features(chunk)
    for key, value in new_features.items():
        segments_features_model.loc[i, key] = value

    last_ttf = current_ttf
    i += 1

# Conversion numérique
segments_features_model = segments_features_model.apply(pd.to_numeric)
print("Extraction des features avancées terminée.")

# --- 3. FILTRAGE DES CYCLES POUR L'ENTRAÎNEMENT ---
CYCLES_TO_KEEP = [1, 2, 3, 5, 7, 8, 10, 12, 14, 15]
segments_features_model = segments_features_model[
    segments_features_model['cycle_id'].isin(CYCLES_TO_KEEP)
]

# --- 4. X / y ---
X = segments_features_model.drop(['time_to_failure', 'cycle_id'], axis=1)
y = segments_features_model['time_to_failure']

# --- 5. ENTRAÎNEMENT LIGHTGBM ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"\nDimensions du jeu d'entraînement filtré: {X_train.shape}")

params = {
    'objective': 'mae',
    'metric': 'mae',
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

model = lgb.LGBMRegressor(**params)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='mae',
    callbacks=[lgb.early_stopping(100, verbose=False)]
)

# --- 6. ÉVALUATION ---
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nMAE sur le jeu de test filtré: {mae:.4f}")

# --- Importance des features ---
importance_df = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
print("\nTop 10 des features par importance:")
print(importance_df.sort_values(by='importance', ascending=False).head(10))



import numpy as np
import pandas as pd
import os
import time
import lightgbm as lgb

# --- SETUP ---
ROOT_PATH = '../input/LANL-Earthquake-Prediction'
TEST_FOLDER = 'test'
SUBMISSION_FILE = 'sample_submission.csv'
SEGMENT_SIZE = 150000
DTYPE_TEST = {'acoustic_data': np.int16}  # Le fichier test n'a pas de colonne time_to_failure

# On suppose que le modèle LightGBM entraîné 'model' et le DataFrame X (pour les colonnes) sont accessibles.

# --- 1. FONCTION DE FEATURE ENGINEERING ---
def extract_advanced_features(chunk):
    acoustic_data = chunk['acoustic_data'].values
    
    # a) Ajouter un petit bruit gaussien constant (std=0.5)
    noise = np.random.normal(0, 0.5, len(acoustic_data))
    acoustic_data = acoustic_data + noise
    
    # b) Soustraire la médiane du segment
    acoustic_data = acoustic_data - np.median(acoustic_data)
    
    features = {}
    # Caractéristiques de base
    features['mean'] = acoustic_data.mean()
    features['std'] = acoustic_data.std()
    features['rms'] = np.sqrt(np.mean(acoustic_data**2))
    features['abs_mean'] = np.mean(np.abs(acoustic_data))
    features['abs_std'] = np.std(np.abs(acoustic_data))
    features['peak_to_peak'] = acoustic_data.max() - acoustic_data.min()
    
    # Caractéristiques statistiques
    features['skew'] = pd.Series(acoustic_data).skew()
    features['kurt'] = pd.Series(acoustic_data).kurtosis()
    features['q01'] = np.quantile(acoustic_data, 0.01)
    features['q99'] = np.quantile(acoustic_data, 0.99)
    
    # Fréquence approximative
    features['z_crossings'] = ((acoustic_data[:-1] * acoustic_data[1:]) < 0).sum()
    
    # Dérivées
    diff = np.diff(acoustic_data)
    features['mean_diff'] = diff.mean()
    features['std_diff'] = diff.std()
    
    # Rolling windows (taille = 1000)
    window = 1000
    series = pd.Series(acoustic_data)
    features['rolling_mean_mean'] = series.rolling(window).mean().mean()
    features['rolling_std_mean'] = series.rolling(window).std().mean()
    
    return features

# --- 2. EXTRACTION DES FEATURES TEST ---
submission = pd.read_csv(os.path.join(ROOT_PATH, SUBMISSION_FILE), index_col='seg_id')

# Même colonnes que l’entraînement
X_test_pred = pd.DataFrame(columns=X.columns, index=submission.index)

print(f"Démarrage de l'extraction des features pour les {len(submission)} segments de test...")

start_time = time.time()
i = 0

for seg_id in X_test_pred.index:
    file_path = os.path.join(ROOT_PATH, TEST_FOLDER, seg_id + '.csv')
    segment_test = pd.read_csv(file_path, dtype=DTYPE_TEST)
    
    features = extract_advanced_features(segment_test)
    
    for feature_name, value in features.items():
        if feature_name in X_test_pred.columns:
            X_test_pred.loc[seg_id, feature_name] = value

    i += 1
    if i % 1000 == 0:
        print(f"Progression : {i} segments de test traités...")

end_time = time.time()
print(f"Extraction des features de test terminée en {round(end_time - start_time, 2)} secondes.")

# --- 3. PREDICTION & SOUMISSION ---
X_test_pred = X_test_pred.apply(pd.to_numeric)
X_test_pred = X_test_pred.fillna(0)

print("\nDémarrage de la prédiction...")
predictions = model.predict(X_test_pred)

submission['time_to_failure'] = predictions

print("\nAperçu du fichier de soumission final :")
print(submission.head())

submission.to_csv('submission_corrected.csv')
print("\n--- Fichier submission_corrected.csv généré avec succès ! ---")

