import pandas as pd
import numpy as np

# --- 0. Configuration ---
# Chemin vers les fichiers de donnÃ©es (DOIT ÃŠTRE LE MÃŠME QUE PRÃ‰CÃ‰DEMMENT)
ROOT = '/kaggle/input/leap-atmospheric-physics-ai-climsim' 

SAMPLE_SUBMISSION_FILE = f'{ROOT}/sample_submission.csv' 

print(f"Chargement du masque de pondÃ©ration depuis : {SAMPLE_SUBMISSION_FILE}")

try:
    # Charger UNIQUEMENT la premiÃ¨re ligne (le masque W)
    df_weights = pd.read_csv(SAMPLE_SUBMISSION_FILE, nrows=1)
    
    # Isoler les colonnes cibles (sans 'sample_id')
    target_cols = [col for col in df_weights.columns if col != 'sample_id']
    W_vector = df_weights[target_cols].iloc[0].values
    
    # --- 1. Analyse et Noms des Colonnes MasquÃ©es (W=0) ---
    
    nombre_total = len(W_vector)
    
    # Obtenir les noms des colonnes avec W = 0
    cols_a_masquer = [target_cols[i] for i, w in enumerate(W_vector) if w == 0]
    nombre_zeros = len(cols_a_masquer)
    nombre_evaluees = nombre_total - nombre_zeros
    
    print("-" * 50)
    print(f"Dimensions du Masque W : {W_vector.shape}")
    print(f"Nombre total de cibles : {nombre_total}")
    print(f"Cibles Ã©valuÃ©es (W>0) : {nombre_evaluees}")
    print(f"Cibles masquÃ©es (W=0) : {nombre_zeros}")
    
    # --- 2. Affichage des Noms SpÃ©cifiques ---

    if nombre_zeros > 0:
        print(f"\nâœ… Les colonnes suivantes NE SONT PAS Ã©valuÃ©es par le score final (W=0) :")
        
        # Afficher le dÃ©compte et les noms des colonnes 'ptend_' masquÃ©es
        ptend_masquees = [col for col in cols_a_masquer if col.startswith('ptend_')]
        if ptend_masquees:
            print(f"   - **ptend_** ({len(ptend_masquees)} colonnes) :")
            print(f"     {ptend_masquees}")

        # Afficher le dÃ©compte et les noms des colonnes 'cam_out_' masquÃ©es
        cam_masquees = [col for col in cols_a_masquer if col.startswith('cam_out_')]
        if cam_masquees:
            print(f"   - **cam_out_** ({len(cam_masquees)} colonnes) :")
            # Pour la lisibilitÃ©, n'affichons que les premiÃ¨res et derniÃ¨res colonnes si la liste est longue
            if len(cam_masquees) > 10:
                print(f"     {cam_masquees[:5]} ... {cam_masquees[-5:]}")
            else:
                print(f"     {cam_masquees}")
                
        print("\n=> Nous pouvons potentiellement supprimer ces colonnes de Y et rÃ©duire la taille d'OUTPUT_UNITS.")
        
    else:
        print("Conclusion : Tous les poids sont 1, ou il y a une erreur de lecture. Le modÃ¨le doit prÃ©dire les 368 colonnes.")
        
except FileNotFoundError:
    print(f"â�Œ Erreur : Le fichier {SAMPLE_SUBMISSION_FILE} n'a pas Ã©tÃ© trouvÃ©. VÃ©rifiez le chemin ROOT.")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration et Chargement des DonnÃ©es ---

# Chemin racine fourni (inchangÃ©)
ROOT = '/kaggle/input/leap-atmospheric-physics-ai-climsim'
TRAIN_FILE = f'{ROOT}/train.csv'

# Charger un petit Ã©chantillon pour l'inspection (1000 premiÃ¨res lignes)
N_ROWS = 1000
try:
    df_train = pd.read_csv(TRAIN_FILE, nrows=N_ROWS)
except FileNotFoundError:
    print(f"Erreur : Le fichier {TRAIN_FILE} n'a pas Ã©tÃ© trouvÃ©.")
    exit()

# Correction des noms de variables rÃ©els (basÃ©s sur le diagnostic)
INPUT_COL_BASE_CORRECT = 'state_t' # EntrÃ©e: TempÃ©rature (OK)
TARGET_COL_BASE_CORRECT = 'ptend_t' # Cible: Tendance au chauffage (CORRIGÃ‰, Ã©tait 'ptend_the')

print("### 1. ğŸ“� Inspection de Base et de la Structure des Colonnes (VERSION FINALE) ###")
print(f"Forme du DataFrame Ã©chantillonnÃ©: {df_train.shape}")
print(f"Nombre total de colonnes: {df_train.shape[1]}")
print("-" * 50)

# --- 2. Validation de la Dimension Verticale (60 Niveaux) ---

# Identification des colonnes de profil avec les noms CORRIGÃ‰S
input_cols_60 = [col for col in df_train.columns if col.startswith(f'{INPUT_COL_BASE_CORRECT}_')]
target_cols_60 = [col for col in df_train.columns if col.startswith(f'{TARGET_COL_BASE_CORRECT}_')]

print(f"Variable d'entrÃ©e (TempÃ©rature) '{INPUT_COL_BASE_CORRECT}' : {len(input_cols_60)} colonnes.")
print(f"Variable cible (Chauffage) '{TARGET_COL_BASE_CORRECT}' : {len(target_cols_60)} colonnes.")

if len(input_cols_60) == 60 and len(target_cols_60) == 60:
    print("\nâœ… SUCCÃˆS : 60 colonnes trouvÃ©es pour l'entrÃ©e ET la cible. La nature du PROFIL VERTICAL est confirmÃ©e.")
    print(f"Indices des 5 premiÃ¨res colonnes (EntrÃ©e): {input_cols_60[:5]}")
    print(f"Indices des 5 derniÃ¨res colonnes (Cible): {target_cols_60[-5:]}")
else:
    print("\nâ�Œ ATTENTION : Ã‰chec persistant. Il faut vÃ©rifier manuellement le nom de la variable cible de chauffage.")
    # Afficher les 20 derniÃ¨res colonnes pour une inspection manuelle si le problÃ¨me persistait.
    print(f"20 derniÃ¨res colonnes: {df_train.columns[-20:].tolist()}")
    exit()
    
print("-" * 50)

# --- 3. Visualisation des Profils (Preuve Graphique de la VerticalitÃ©) ---

# SÃ©lectionner le premier exemple de la ligne 0
example_row = df_train.iloc[0]

# Extraction des profils avec les noms de colonnes CORRIGÃ‰S
temp_profile = example_row[[f'{INPUT_COL_BASE_CORRECT}_{i}' for i in range(60)]].values
heat_tendency_profile = example_row[[f'{TARGET_COL_BASE_CORRECT}_{i}' for i in range(60)]].values

# DÃ©finir les niveaux verticaux.
levels = np.arange(60)
altitude_proxy = 59 - levels # Inverser les indices pour que le niveau 59 (prÃ¨s du sol) soit en bas du graphique.

print("### 3. Visualisation du Profil Vertical pour un Point de Grille ###")

plt.figure(figsize=(12, 6))

# --- Graphique 1 : Profil de TempÃ©rature (EntrÃ©e) ---
plt.subplot(1, 2, 1)
plt.plot(temp_profile, altitude_proxy, marker='o', markersize=3, linestyle='-', color='blue')
plt.title(f'Profil Vertical de TempÃ©rature ({INPUT_COL_BASE_CORRECT})')
plt.xlabel('TempÃ©rature (K)')
plt.ylabel('Niveau Vertical (Proxy Altitude)')
plt.yticks(np.arange(0, 60, 10), labels=np.arange(59, -1, -10))
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().invert_yaxis() 

# --- Graphique 2 : Profil de Tendance au Chauffage (Cible) ---
plt.subplot(1, 2, 2)
plt.plot(heat_tendency_profile, altitude_proxy, marker='x', markersize=3, linestyle='-', color='red')
plt.title(f'Profil Vertical de Tendance au Chauffage ({TARGET_COL_BASE_CORRECT})')
plt.xlabel('Tendance au Chauffage (K/s)')
plt.ylabel('Niveau Vertical (Proxy Altitude)')
plt.yticks(np.arange(0, 60, 10), labels=np.arange(59, -1, -10))
plt.grid(True, linestyle='--', alpha=0.6)
plt.gca().invert_yaxis() 

plt.tight_layout()
plt.show()

print("\n=> La vÃ©rification est rÃ©ussie et le graphique est affichÃ©. Le problÃ¨me est bien une rÃ©gression de profil Ã  profil (60D).")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Config ---
ROOT = '/kaggle/input/leap-atmospheric-physics-ai-climsim'
TRAIN_FILE = f'{ROOT}/train.csv'
N_ROWS = 50000
LEVELS = 60
INPUT_COL_BASE = 'state_t'
TARGET_COL_BASE = 'ptend_t'

# Construire la liste des colonnes attendues
temp_cols = [f'{INPUT_COL_BASE}_{i}' for i in range(LEVELS)]
ptend_cols = [f'{TARGET_COL_BASE}_{i}' for i in range(LEVELS)]
usecols = temp_cols + ptend_cols  # charger uniquement ces colonnes

# Charger en gÃ©rant l'absence Ã©ventuelle de colonnes
try:
    df_train = pd.read_csv(TRAIN_FILE, usecols=usecols, nrows=N_ROWS)
    print(f"Ã‰chantillon chargÃ©: {df_train.shape[0]} lignes, {df_train.shape[1]} colonnes.")
except ValueError as e:
    # pd.read_csv lÃ¨ve ValueError si certaines colonnes dans usecols sont manquantes
    present_cols = pd.read_csv(TRAIN_FILE, nrows=0).columns.tolist()
    missing = [c for c in usecols if c not in present_cols]
    raise RuntimeError(f"Colonnes manquantes dans le fichier: {missing}") from e

# 1) Statistiques par niveau
temp_stats = df_train[temp_cols].agg(['mean', 'std']).T
temp_stats.index = [f'Niveau_{i}' for i in range(LEVELS)]
temp_stats.columns = ['TempÃ©rature Moyenne (K)', 'Ã‰cart-Type TempÃ©rature (K)']
print(pd.concat([temp_stats.head(), temp_stats.tail()]).to_markdown(floatfmt=".2f"))

# 2) CorrÃ©lations niveau-Ã -niveau (plus robuste et plus rapide)
corrs = []
for t_col, p_col in zip(temp_cols, ptend_cols):
    x = df_train[t_col]
    y = df_train[p_col]
    # Si std == 0 => corr = NaN ; on gÃ¨re explicitement
    if x.std() == 0 or y.std() == 0:
        corrs.append(np.nan)
    else:
        corrs.append(x.corr(y))  # Ã©quivalent Ã  pearsonr sans import
df_corr = pd.Series(corrs, index=[f'Niveau_{i}' for i in range(LEVELS)], name='CorrÃ©lation')

# 3) Plot
plt.figure(figsize=(14,10))
altitude_proxy = 59 - np.arange(LEVELS)

plt.subplot(1,2,1)
plt.plot(temp_stats['TempÃ©rature Moyenne (K)'], altitude_proxy, marker='o', markersize=3, label='Moyenne (K)')
plt.plot(temp_stats['Ã‰cart-Type TempÃ©rature (K)'], altitude_proxy, marker='x', markersize=3, linestyle='--', label='Ã‰cart-Type (K)')
plt.gca().invert_yaxis()
plt.title('Profil Vertical: TempÃ©rature Moyenne et VariabilitÃ©')
plt.xlabel('Valeur')
plt.ylabel('Niveau Vertical (Proxy Altitude)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.yticks(np.arange(0,60,10), labels=np.arange(59,-1,-10))

plt.subplot(1,2,2)
plt.plot(df_corr.values, altitude_proxy, marker='s', markersize=3)
plt.gca().invert_yaxis()
plt.title('CorrÃ©lation Pearson (state_t vs ptend_t) par Niveau')
plt.xlabel('Coefficient de corrÃ©lation')
plt.ylabel('Niveau Vertical (Proxy Altitude)')
plt.axvline(0, color='red', linewidth=0.8)
plt.grid(True, linestyle='--', alpha=0.6)
plt.yticks(np.arange(0,60,10), labels=np.arange(59,-1,-10))

plt.tight_layout()
plt.show()

# 4) Diagnostic rapide des corrÃ©lations
print("\nNombre de niveaux oÃ¹ la corrÃ©lation est NaN (std=0):", np.sum(np.isnan(df_corr)))
print("CorrÃ©lation moyenne (excl. NaN):", np.nanmean(df_corr))



import pandas as pd

# IMPORTANT : Assurez-vous que le chemin ROOT est dÃ©fini comme dans les autres cellules
ROOT = '/kaggle/input/leap-atmospheric-physics-ai-climsim' 
TRAIN_FILE = f'{ROOT}/train.csv'

print(f"Chargement du fichier d'entraÃ®nement depuis : {TRAIN_FILE}")

# Chargement d'un sous-ensemble pour des raisons de mÃ©moire et de temps
# Si vous avez assez de mÃ©moire, vous pouvez retirer 'nrows'
try:
    df_train = pd.read_csv(TRAIN_FILE, nrows=100000) 
    print(f"âœ… df_train chargÃ©. Dimensions : {df_train.shape}")
except FileNotFoundError:
    print(f"â�Œ Erreur : Le fichier {TRAIN_FILE} n'a pas Ã©tÃ© trouvÃ©. Veuillez vÃ©rifier le chemin ROOT.")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import backend as K

# --- 0. Configuration et Constantes ---
ROOT = '/kaggle/input/leap-atmospheric-physics-ai-climsim' 
SAMPLE_SUBMISSION_FILE = f'{ROOT}/sample_submission.csv' 
TRAIN_FILE = f'{ROOT}/train.csv'
LEVELS = 60
id_col = 'sample_id'

# Liste des 60 colonnes identifiÃ©es comme ayant un Poids W = 0 (NON Ã‰VALUÃ‰ES)
COLS_TO_DROP = [
    'ptend_q0001_0', 'ptend_q0001_1', 'ptend_q0001_2', 'ptend_q0001_3', 'ptend_q0001_4', 'ptend_q0001_5', 
    'ptend_q0001_6', 'ptend_q0001_7', 'ptend_q0001_8', 'ptend_q0001_9', 'ptend_q0001_10', 'ptend_q0001_11', 
    'ptend_q0002_0', 'ptend_q0002_1', 'ptend_q0002_2', 'ptend_q0002_3', 'ptend_q0002_4', 'ptend_q0002_5', 
    'ptend_q0002_6', 'ptend_q0002_7', 'ptend_q0002_8', 'ptend_q0002_9', 'ptend_q0002_10', 'ptend_q0002_11', 
    'ptend_q0003_0', 'ptend_q0003_1', 'ptend_q0003_2', 'ptend_q0003_3', 'ptend_q0003_4', 'ptend_q0003_5', 
    'ptend_q0003_6', 'ptend_q0003_7', 'ptend_q0003_8', 'ptend_q0003_9', 'ptend_q0003_10', 'ptend_q0003_11', 
    'ptend_u_0', 'ptend_u_1', 'ptend_u_2', 'ptend_u_3', 'ptend_u_4', 'ptend_u_5', 
    'ptend_u_6', 'ptend_u_7', 'ptend_u_8', 'ptend_u_9', 'ptend_u_10', 'ptend_u_11', 
    'ptend_v_0', 'ptend_v_1', 'ptend_v_2', 'ptend_v_3', 'ptend_v_4', 'ptend_v_5', 
    'ptend_v_6', 'ptend_v_7', 'ptend_v_8', 'ptend_v_9', 'ptend_v_10', 'ptend_v_11'
]

# --- CHARGEMENT DU FICHIER D'ENTRAINEMENT (Ajustez nrows si nÃ©cessaire) ---
N_ROWS_TO_LOAD = 50000 
try:
    df_train = pd.read_csv(TRAIN_FILE, nrows=N_ROWS_TO_LOAD)
    print(f"Chargement de {df_train.shape[0]} lignes d'entraÃ®nement.")
except FileNotFoundError:
    print(f"â�Œ Erreur: Fichier d'entraÃ®nement non trouvÃ© Ã  {TRAIN_FILE}")
    exit()
    
# --- 1. SÃ©paration des EntrÃ©es (X) et Cibles (Y) ---

all_cols = df_train.columns.tolist()
target_cols_original = [col for col in all_cols if col.startswith('ptend_') or col.startswith('cam_out_')]
# Y filtrÃ© : la liste des 308 cibles Ã  prÃ©dire
target_cols_filtered = [col for col in target_cols_original if col not in COLS_TO_DROP]
input_cols = [col for col in all_cols if col not in target_cols_original and col != id_col]

X = df_train[input_cols].values
Y = df_train[target_cols_filtered].values

# DÃ©finition des variables globales (Indispensables pour les cellules suivantes)
globals()['OUTPUT_UNITS'] = Y.shape[1] 
globals()['input_cols'] = input_cols
globals()['target_cols_filtered'] = target_cols_filtered
globals()['COLS_TO_DROP'] = COLS_TO_DROP
globals()['LEVELS'] = LEVELS

print("### 1. SÃ©paration et Filtrage des Cibles (Y) ###")
print(f"X (EntrÃ©es) dimensions brutes : {X.shape} ({X.shape[1]} caractÃ©ristiques)")
print(f"Y (Cibles FILTRÃ‰ES) dimensions brutes : {Y.shape} ({globals()['OUTPUT_UNITS']} cibles)")
print("-" * 50)


# --- 2. Standardisation des DonnÃ©es ---
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)

scaler_Y = StandardScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_val_scaled = scaler_Y.transform(Y_val)

globals()['scaler_Y'] = scaler_Y 
globals()['scaler_X'] = scaler_X


# --- 3. Mise en Forme 3D pour le CNN 1D ---
N_FEATURES_60D = 9 
N_SCALAR_FEATURES = X.shape[1] - (N_FEATURES_60D * LEVELS) 

globals()['N_FEATURES_60D'] = N_FEATURES_60D
globals()['N_SCALAR_FEATURES'] = N_SCALAR_FEATURES

# SÃ©paration des profils (CNN)
X_train_profile = X_train_scaled[:, :N_FEATURES_60D * LEVELS]
X_val_profile = X_val_scaled[:, :N_FEATURES_60D * LEVELS]
X_train_CNN = X_train_profile.reshape(-1, LEVELS, N_FEATURES_60D)
X_val_CNN = X_val_profile.reshape(-1, LEVELS, N_FEATURES_60D)

# SÃ©paration des scalaires (MLP)
X_train_SCALAR = X_train_scaled[:, N_FEATURES_60D * LEVELS:]
X_val_SCALAR = X_val_scaled[:, N_FEATURES_60D * LEVELS:]

globals()['X_train_CNN'] = X_train_CNN
globals()['X_val_CNN'] = X_val_CNN
globals()['X_train_SCALAR'] = X_train_SCALAR
globals()['X_val_SCALAR'] = X_val_SCALAR
globals()['Y_train_scaled'] = Y_train_scaled
globals()['Y_val_scaled'] = Y_val_scaled

print("### 3. Reforme pour le CNN 1D Multi-Input ###")
print(f"X_train reformÃ© pour CNN (Profils) : {X_train_CNN.shape}")
print("-" * 50)


# --- 4. CrÃ©ation du Masque W FiltrÃ© pour la Perte (CORRIGÃ‰) ---

df_weights_original = pd.read_csv(SAMPLE_SUBMISSION_FILE, nrows=1)
target_cols_all = [col for col in df_weights_original.columns if col != id_col]

# CrÃ©ation d'un mapping Colonne -> Poids W pour un alignement GARANTI
weights_map = df_weights_original[target_cols_all].iloc[0].to_dict()

# CrÃ©er le vecteur W filtrÃ© dans l'ordre EXACT de Y (target_cols_filtered)
W_vector_filtered = np.array([weights_map[col] for col in target_cols_filtered], dtype=np.float32)

W_tensor_filtered = tf.constant(W_vector_filtered, dtype=tf.float32)

globals()['W_tensor'] = W_tensor_filtered


# --- 5. VÃ©rifications Finales ---
print("\n### 5. VÃ©rifications Cruciales du Tenseur W et de l'Ordre des Cibles ###")

# VÃ©rification 5.1: Alignement de W avec Y
assert W_tensor_filtered.shape[0] == globals()['OUTPUT_UNITS']
print(f"âœ… Alignement W : W_tensor de taille {W_tensor_filtered.shape[0]} correspond Ã  {globals()['OUTPUT_UNITS']} sorties.")

# VÃ©rification 5.2: Conservation (ptend_t est bien au dÃ©but de Y)
ptend_t_cols = [col for col in target_cols_filtered if col.startswith('ptend_t_')]
if len(ptend_t_cols) == LEVELS and ptend_t_cols == target_cols_filtered[:LEVELS]:
    print(f"âœ… Ordre Y : Les {LEVELS} colonnes 'ptend_t' sont en tÃªte de Y. La perte de conservation est sÃ©curisÃ©e.")
else:
    print("â�Œ ERREUR CRITIQUE: L'ordre des cibles (Y) n'a pas les 60 colonnes 'ptend_t' en tÃªte. La perte de conservation sera fausse.")
    print("PremiÃ¨res colonnes de Y :", target_cols_filtered[:LEVELS])
    
print("\nğŸ�‰ Toutes les variables globales nÃ©cessaires sont dÃ©finies. Vous pouvez lancer la cellule d'entraÃ®nement V4.")


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Concatenate, Dropout 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# --- 1. Fonction de Perte PIML Robuste (CORRIGÃ‰E) ---

# W_tensor est rÃ©cupÃ©rÃ© depuis les variables globales (W_tensor_filtered, taille 308)
W = globals().get('W_tensor')

def weighted_mse_with_conservation_loss(y_true, y_pred):
    """
    Fonction de Perte PIML (Physically Informed ML) robuste et sÃ©curisÃ©e.
    Inclut : MSE PondÃ©rÃ©e (simplifiÃ©e) et PÃ©nalitÃ© de Conservation.
    """
    
    # 1. Terme MSE PondÃ©rÃ©e (la perte principale)
    err = y_pred - y_true
    # Multiplier l'erreur au carrÃ© par le masque W
    weighted_sq_err = K.square(err) * W 
    
    # Calculer la moyenne sur TOUTES les prÃ©dictions pondÃ©rÃ©es
    weighted_mse_loss = K.mean(weighted_sq_err) 

    # 2. Terme de PÃ©nalitÃ© de Conservation (sur les 60 premiÃ¨res sorties : ptend_t)
    LEVELS = globals().get('LEVELS', 60)
    # Nous utilisons ici la TENDANCE, car la propriÃ©tÃ© de conservation est ptend_t_sum = 0
    # Cependant, forcer (y_pred - y_true) Ã  Ãªtre zÃ©ro est plus robuste en pratique.
    ptend_pred = y_pred[:, :LEVELS]
    ptend_true = y_true[:, :LEVELS]

    # PÃ©nalitÃ© basÃ©e sur la non-conservation de la TENDANCE
    cons_residual = K.sum(ptend_pred - ptend_true, axis=1)
    cons_penalty = K.mean(K.square(cons_residual))

    # 3. Combinaison
    # LAMBDA ajustÃ© Ã  une petite valeur pour Ã©viter que la perte physique ne domine la MSE
    LAMBDA = 1e-4 
    total_loss = weighted_mse_loss + LAMBDA * cons_penalty
    return total_loss

# --- 2. DÃ©finition du ModÃ¨le Multi-Input (Avec RÃ©gularisation L2 et Dropout) ---

# RÃ©cupÃ©ration des variables globales
LEVELS = globals().get('LEVELS', 60)
N_FEATURES_60D = globals().get('N_FEATURES_60D', 9)
N_SCALAR_FEATURES = globals().get('N_SCALAR_FEATURES', 16)
OUTPUT_UNITS = globals().get('OUTPUT_UNITS', 308)

# DÃ©finition d'un rÃ©gularisateur L2 lÃ©ger
regularizer = tf.keras.regularizers.l2(1e-5) 

# Branche des Profils (CNN 1D)
profile_input = Input(shape=(LEVELS, N_FEATURES_60D), name='profile_input')
x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizer)(profile_input)
x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizer)(x)
x = Flatten()(x)
x = Dropout(0.1)(x) 

# Branche des Scalaires (MLP)
scalar_input = Input(shape=(N_SCALAR_FEATURES,), name='scalar_input')
y = Dense(32, activation='relu', kernel_regularizer=regularizer)(scalar_input)
y = Dense(16, activation='relu', kernel_regularizer=regularizer)(y) 
y = Dropout(0.1)(y)

# Combinaison
combined = Concatenate()([x, y])
z = Dense(64, activation='relu', kernel_regularizer=regularizer)(combined)
output = Dense(OUTPUT_UNITS, activation='linear', name='final_output')(z) 

model_optimized = Model(inputs=[profile_input, scalar_input], outputs=output)


# --- 3. Compilation et Callbacks AvancÃ©s ---

# DÃ©finition du nom du fichier
# Note: Le ModelCheckpoint ci-dessous utilise le format .h5 pour la compatibilitÃ©
MODEL_FILENAME = 'best_model_PIML_V5_CORRECT.keras' 

# Callbacks avancÃ©s
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)
# Le ModelCheckpoint ne sauvegarde que le MEILLEUR modÃ¨le (pour l'architecture V4)
checkpoint = ModelCheckpoint(
    'best_model_PIML_V4.h5', 
    monitor='val_loss', 
    save_best_only=True, 
    save_weights_only=False
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=5, 
    min_lr=1e-7
)

model_optimized.compile(
    optimizer=Adam(learning_rate=1e-4), 
    loss=weighted_mse_with_conservation_loss,
    metrics=[] 
)

print("\n### RÃ©sumÃ© du ModÃ¨le PIML Multi-Input V5 SÃ©curisÃ© ###")
model_optimized.summary()
print("-" * 50)

print("DÃ©marrage de l'entraÃ®nement V5...")

# RÃ©cupÃ©ration des tenseurs d'entraÃ®nement (supposÃ©s globaux)
X_train_CNN = globals().get('X_train_CNN')
X_train_SCALAR = globals().get('X_train_SCALAR')
Y_train_scaled = globals().get('Y_train_scaled')
X_val_CNN = globals().get('X_val_CNN')
X_val_SCALAR = globals().get('X_val_SCALAR')
Y_val_scaled = globals().get('Y_val_scaled')

history_optimized = model_optimized.fit(
    [X_train_CNN, X_train_SCALAR], 
    Y_train_scaled, 
    epochs=100, 
    batch_size=1024,
    validation_data=([X_val_CNN, X_val_SCALAR], Y_val_scaled),
    callbacks=[early_stopping, checkpoint, reduce_lr],
    verbose=1
)

# -----------------------------------------------------------
# ğŸŒŸ AJOUT CRITIQUE DE SÃ‰CURITÃ‰ : SAUVEGARDE MANUELLE DU MEILLEUR MODÃˆLE
# -----------------------------------------------------------
# Assurez-vous d'abord de restaurer les meilleurs poids trouvÃ©s pendant l'entraÃ®nement
model_optimized.set_weights(early_stopping.best_weights)

# Sauvegarder dans le nouveau format .keras pour une compatibilitÃ© maximale
model_optimized.save(MODEL_FILENAME)

print(f"\nğŸ�‰ EntraÃ®nement terminÃ©. Le MEILLEUR modÃ¨le (architecture et poids) est sauvegardÃ© dans '{MODEL_FILENAME}'.")
print("-" * 50)


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model 
from tensorflow.keras import backend as K
import os

# --- 0. Configuration et Variables Globales ---
ROOT = globals().get('ROOT', '/kaggle/input/leap-atmospheric-physics-ai-climsim')
TEST_FILE = f'{ROOT}/test.csv'
SAMPLE_SUBMISSION_FILE = f'{ROOT}/sample_submission.csv'
MODEL_PATH = 'best_model_PIML_V5_CORRECT.keras' 

# RÃ©cupÃ©ration des objets cruciaux
scaler_X = globals().get('scaler_X')
scaler_Y = globals().get('scaler_Y')
input_cols = globals().get('input_cols')
target_cols_filtered = globals().get('target_cols_filtered')
id_col = 'sample_id'
LEVELS = globals().get('LEVELS', 60)
N_FEATURES_60D = globals().get('N_FEATURES_60D', 9)

# âš ï¸� VÃ‰RIFICATION CRITIQUE âš ï¸�
if scaler_X is None or scaler_Y is None or target_cols_filtered is None:
    print("â�Œ ERREUR FATALE: Les objets de normalisation (scaler_X/Y) ou les colonnes cibles sont manquants.")
    print("Veuillez vous assurer que la cellule d'entraÃ®nement a Ã©tÃ© exÃ©cutÃ©e et que les scalers sont dans globals().")
    exit()

# --- 1. FONCTIONS CUSTOMISÃ‰ES ---
def weighted_mse_with_conservation_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

# --- 2. Chargement du ModÃ¨le et PrÃ©dictions ---
print(f"Chargement du modÃ¨le depuis : {MODEL_PATH}")
custom_objects = {'weighted_mse_with_conservation_loss': weighted_mse_with_conservation_loss }
try:
    model_optimized = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects=custom_objects,
        compile=False
    )
except Exception as e:
    print(f"â�Œ ERREUR FATALE : Le modÃ¨le est illisible. {e}")
    exit()

print("Chargement des donnÃ©es de test...")
df_test = pd.read_csv(TEST_FILE)
df_submission_sample = pd.read_csv(SAMPLE_SUBMISSION_FILE)

# PrÃ©paration de X_test
X_test_brut = df_test[input_cols].values
X_test_scaled = scaler_X.transform(X_test_brut)
X_test_profile = X_test_scaled[:, :N_FEATURES_60D * LEVELS]
X_test_CNN = X_test_profile.reshape(-1, LEVELS, N_FEATURES_60D)
X_test_SCALAR = X_test_scaled[:, N_FEATURES_60D * LEVELS:]

print("DÃ©marrage des prÃ©dictions...")
Y_pred_scaled = model_optimized.predict([X_test_CNN, X_test_SCALAR], batch_size=1024)

# ğŸš€ V12 NOUVEAUTÃ‰ 1 : CLIPPING DANS L'ESPACE NORMALISÃ‰
# Une valeur > 10 dans l'espace normalisÃ© est dÃ©jÃ  trop grande.
SCALED_CLIP_VALUE = 10.0
Y_pred_scaled_clipped = np.clip(Y_pred_scaled, a_min=-SCALED_CLIP_VALUE, a_max=SCALED_CLIP_VALUE)
print(f"âœ… Clipping dans l'espace normalisÃ© appliquÃ© (min/max: +/- {SCALED_CLIP_VALUE}).")

# DÃ©normalisation
Y_pred_denormalized = scaler_Y.inverse_transform(Y_pred_scaled_clipped)

# ğŸš€ V12 NOUVEAUTÃ‰ 2 : CLIPPING AGRESSIF (REDONDANT MAIS SÃ‰CURISÃ‰)
CLIP_VALUE = 1e+5 # 100 000.0 K/s 
Y_pred_denormalized_clipped = np.clip(Y_pred_denormalized, a_min=-CLIP_VALUE, a_max=CLIP_VALUE)
print(f"âœ… Re-Clipping de sÃ©curitÃ© AGRESSIF appliquÃ© (min/max: +/- {CLIP_VALUE} K/s).")


# --- 3. POST-TRAITEMENT PHYSIQUE (Ptend Trick) ---
col_source = 'ptend_q0002_2' 
col_target = 'ptend_q0002_26' 
try:
    idx_source = target_cols_filtered.index(col_source)
    idx_target = target_cols_filtered.index(col_target)
    Y_pred_denormalized_clipped[:, idx_target] = Y_pred_denormalized_clipped[:, idx_source]
    print(f"âœ… Correction du Ptend Trick appliquÃ©e ({col_source} -> {col_target}).")
except:
    pass # Ne pas interrompre si le Ptend Trick Ã©choue

# --- 4. CONSTRUCTION DU FICHIER DE SOUMISSION (MÃ‰THODE V10 CORRECTE) ---
df_pred_filtered = pd.DataFrame(Y_pred_denormalized_clipped, columns=target_cols_filtered)

print("DÃ©marrage de la construction (MÃ©thode V10: Multiplication par W)...")
submission_cols = [col for col in df_submission_sample.columns if col != id_col]
df_submission_final = pd.DataFrame(0.0, index=df_test.index, columns=[id_col] + submission_cols) 
df_submission_final[id_col] = df_test[id_col] 
df_submission_final[target_cols_filtered] = df_pred_filtered[target_cols_filtered]

# 4.3 RÃ©cupÃ©rer la MATRICE DES POIDS W 
W_matrix_full = df_submission_sample[submission_cols].values 

# 4.4 L'Ã‰TAPE CRITIQUE : MULTIPLIER PAR W (Conversion en W/mÂ²)
df_submission_final.iloc[:, 1:] = df_submission_final.iloc[:, 1:].values * W_matrix_full

# ğŸš€ V12 NOUVEAUTÃ‰ 3 : CLIPPING DU RÃ‰SULTAT FINAL (en unitÃ©s W/mÂ²)
FINAL_CLIP_VALUE = 1e+4 # 10 000.0 W/mÂ² (Limite ultra-stricte)
df_submission_final.iloc[:, 1:] = np.clip(
    df_submission_final.iloc[:, 1:].values,
    a_min=-FINAL_CLIP_VALUE,
    a_max=FINAL_CLIP_VALUE
)
print(f"âœ… Clipping FINAL appliquÃ© (min/max: +/- {FINAL_CLIP_VALUE} W/mÂ²).")


# 4.5 Sauvegarde
submission_filename = 'submission_PIML_V12_TRIPLE_CLIPPED.csv'
df_submission_final.to_csv(submission_filename, index=False)

print("-" * 50)
print(f"ğŸ�‰ Soumission finalisÃ©e et sauvegardÃ©e sous : {submission_filename} (TRIPLE SÃ‰CURITÃ‰)")

