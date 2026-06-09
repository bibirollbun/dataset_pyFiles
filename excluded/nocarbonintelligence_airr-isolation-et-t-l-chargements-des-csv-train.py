# ----------------------------------------------------------------------
# ANCIENNE CELLULE 1 - LOGIQUE DE CONSOLIDATION DÃ‰PLACÃ‰E
#
# Ce bloc de code a Ã©tÃ© exÃ©cutÃ© une fois dans un notebook sÃ©parÃ© pour :
# 1. Rechercher tous les fichiers metadata.csv dans les 8 datasets.
# 2. DÃ©terminner l'ensemble maximal des 19 colonnes (y compris les 12 HLA et les 4 cliniques).
# 3. Aligner toutes les structures en remplissant les colonnes manquantes avec des NaN (pour Datasets 1 Ã  7).
# 4. Consolider les 3610 rÃ©pertoires dans le fichier unique 'all_train_metadata_consolidated.csv'.
#
# Le fichier 'all_train_metadata_consolidated.csv' a Ã©tÃ© tÃ©lÃ©chargÃ© et
# rÃ©-uploadÃ© dans le rÃ©pertoire d'entrÃ©e pour un chargement simple et rapide.
# Ce code est maintenant commentÃ©.
# ----------------------------------------------------------------------
#
# import pandas as pd
# from glob import glob
# import os
# import sys
# import numpy as np # Import nÃ©cessaire pour les valeurs manquantes
#
# # --- Configuration des Chemins ---
# ROOT_DIR = "/kaggle/input/adaptive-immune-profiling-challenge-2025"
# TARGET_CSV_FOLDER = f"{ROOT_DIR}/train_datasets/train_datasets" 
# OUTPUT_METADATA_NAME = 'all_train_metadata_consolidated.csv'
# MISSING_META_VALUE = np.nan # Valeur standard pour les colonnes manquantes
# # ---------------------------------
#
# print("ğŸ”� Ã‰tape 1 : Recherche et dÃ©termination de la structure MAXIMALE")
# ... (le reste du code de consolidation est omis ici pour la clartÃ©) ...
#
# ----------------------------------------------------------------------
# ğŸš€ NOUVEAU CHARGEMENT SIMPLE ET RAPIDE DU FICHIER CONSOLIDÃ‰
#
# import pandas as pd
# import os
#
# # --- Configuration des Chemins ---
# TRAIN_METADATA_PATH = 'all_train_metadata_consolidated.csv' 
# ROOT_DIR = "/kaggle/input/adaptive-immune-profiling-challenge-2025"
# # ---------------------------------
#
# # 1. Chargement du fichier consolidÃ© (CorrigÃ© pour Ã©viter le dÃ©calage de colonnes)
# all_meta = pd.read_csv(
#     TRAIN_METADATA_PATH, 
#     sep=',', 
#     skipinitialspace=True
# )
#
# # 2. Reconstruction du chemin du fichier TSV original
# all_meta['filename'] = all_meta.apply(
#     lambda row: os.path.join(ROOT_DIR, 'train_datasets', 'train_datasets', row['dataset'], row['repertoire_id'] + '.tsv'),
#     axis=1
# )
#
# # 3. Indexation et DÃ©finition de la Cible
# all_meta = all_meta.set_index('repertoire_id').reset_index()
# TARGET_LABEL_COLUMN = 'label_positive' 
#
# print(f"âœ… CONSOLIDATION BYPASSÃ‰E. {len(all_meta)} rÃ©pertoires chargÃ©s Ã  partir du fichier consolidÃ©.")
# print(f"La table de mÃ©tadonnÃ©es contient maintenant {all_meta.shape[1]} colonnes (y compris les IDs et les NaN).")
#
# display(all_meta.head(2))
# display(all_meta.tail(2))



import pandas as pd
import os

# --- Configuration des Chemins ---
# ATTENTION : Le chemin vers le fichier consolidÃ© est mis Ã  jour
TRAIN_METADATA_PATH = '/kaggle/input/airr-all-train-metadata-consolidation/all_train_metadata_consolidated.csv' 
ROOT_DIR = "/kaggle/input/adaptive-immune-profiling-challenge-2025"
# ---------------------------------

# 1. Chargement du fichier consolidÃ©
all_meta = pd.read_csv(
    TRAIN_METADATA_PATH, 
    sep=',', 
    skipinitialspace=True
)

# 2. Reconstruction du chemin du fichier TSV original (nÃ©cessaire pour charger les donnÃ©es AIRR plus tard)
all_meta['filename'] = all_meta.apply(
    lambda row: os.path.join(ROOT_DIR, 'train_datasets', 'train_datasets', row['dataset'], row['repertoire_id'] + '.tsv'),
    axis=1
)

# 3. Indexation et DÃ©finition de la Cible
all_meta = all_meta.set_index('repertoire_id').reset_index()
TARGET_LABEL_COLUMN = 'label_positive' 

print(f"âœ… Chargement rÃ©ussi. {len(all_meta)} rÃ©pertoires chargÃ©s.")
print(f"Colonnes chargÃ©es : {all_meta.columns.tolist()}")

display(all_meta.head(2))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# NOTE: Cette cellule suppose que 'all_meta' a Ã©tÃ© chargÃ© correctement dans la Cellule 1

## 1. SÃ©paration des Features de MÃ©tadonnÃ©es (X_meta) et de la Cible (y)
# Nous extrayons toutes les colonnes qui ne sont pas des identifiants (repertoire_id, filename, dataset)
# ni la cible (label_positive).
IDENTIFIER_COLS = ['repertoire_id', 'filename', 'dataset']
TARGET_COL = 'label_positive'
ALL_META_FEATURES = [col for col in all_meta.columns if col not in IDENTIFIER_COLS + [TARGET_COL]]

X_meta = all_meta[ALL_META_FEATURES].copy() # Utilisation de .copy() pour Ã©viter SettingWithCopyWarning
y = all_meta[TARGET_COL].astype(int)

print(f"âœ… Features de MÃ©tadonnÃ©es (X_meta) extraites. Total de {X_meta.shape[1]} features.")
print(f"âœ… Cible (y) extraite et binarisÃ©e. Total de {len(y)} rÃ©pertoires.")

# ---
print("\n### 2. Inspection du Taux de Valeurs Manquantes (NaN) ###")
# Identifier les colonnes avec le plus de NaN (HLA, age, etc.)
missing_ratio = X_meta.isnull().sum().sort_values(ascending=False) / len(X_meta)
missing_ratio = missing_ratio[missing_ratio > 0]

if not missing_ratio.empty:
    plt.figure(figsize=(14, 6))
    # Utilisation d'un graphique Ã  barres pour visualiser le ratio de NaN
    sns.barplot(x=missing_ratio.index, y=missing_ratio.values, palette="viridis")
    plt.xticks(rotation=45, ha='right')
    plt.title("Ratio de Valeurs Manquantes (NaN) par Feature de MÃ©tadonnÃ©e")
    plt.ylabel("Ratio (NaN / Total RÃ©pertoires)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    print("\nâš ï¸� Taux de NaN le plus Ã©levÃ© :")
    print(missing_ratio) # Affichons tous les ratios > 0 pour Ãªtre complet
    print("\nCes colonnes, particuliÃ¨rement les colonnes HLA (DRB, DQA, etc.), sont manquantes dans la majoritÃ© des datasets (1-7), ce qui confirme le risque de Data Leakage.")
else:
    print("ğŸ‘� Aucune valeur manquante dÃ©tectÃ©e dans les features de mÃ©tadonnÃ©es.")

# ---
print("\n### 3. Type de DonnÃ©es et Nettoyage Initial (HLA & Ã‚ge) ###")
# Conversion de la colonne d'Ã¢ge en numÃ©rique
if 'age' in X_meta.columns:
    X_meta['age'] = pd.to_numeric(X_meta['age'], errors='coerce')

# AperÃ§u des types de donnÃ©es pour identifier les prochaines Ã©tapes de prÃ©-traitement
print("\nTypes de donnÃ©es des features de mÃ©tadonnÃ©es aprÃ¨s nettoyage initial :")
print(X_meta.dtypes)

