# import pandas as pd
# import numpy as np
# from glob import glob
# from collections import Counter
# import math
# from tqdm.notebook import tqdm
# import warnings
# from sklearn.model_selection import StratifiedKFold
# from lightgbm import LGBMClassifier
# from sklearn.metrics import roc_auc_score
# import os
# import joblib # <-- NOUVEL IMPORT pour la sauvegarde du modÃ¨le

# # --- 1. CONFIGURATION GLOBALE ---

# # Supprimer les warnings pour plus de clartÃ© dans le notebook
# warnings.filterwarnings('ignore')

# ROOT_DIR = "/kaggle/input/adaptive-immune-profiling-challenge-2025"
# TRAIN_DATA_FOLDER = f"{ROOT_DIR}/train_datasets/train_datasets" 
# # Chemin vers le fichier consolidÃ© (Assurez-vous qu'il correspond Ã  votre environnement)
# TRAIN_METADATA_PATH = '/kaggle/input/airr-all-train-metadata-consolidation/all_train_metadata_consolidated.csv' 
# TARGET_COL = 'label_positive'
# K_MER_SIZE = 3 
# N_SPLITS_CV = 5
# MODEL_FILENAME = 'lgbm_final_model.joblib' # <-- NOUVELLE CONSTANTE
# ATTRIBUTION_FILENAME = 'df_attributions.csv' # Pour sauvegarder les attributions
# K_MER_SIZE_ATTR = 3 # Taille du K-mer pour l'attribution

# # Constantes d'encodage
# AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
# HYDROPHOBICITY = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3}
# TOP_N_KMER = 100 
# COMMON_V_GENES = [f'TRBV{i}' for i in range(1, 30)] 
# COMMON_J_GENES = [f'TRBJ{i}' for i in range(1, 7)] 

# print("Configuration chargÃ©e.")
# print("-" * 50)


# # --- 2. DÃ‰FINITIONS DES FONCTIONS (VERSION AMÃ‰LIORÃ‰E) ---

# def calculate_shannon_diversity(template_counts):
#     """Calcule l'entropie de Shannon pour un ensemble de frÃ©quences de clones."""
#     if not template_counts or sum(template_counts) == 0:
#         return 0.0
#     total = sum(template_counts)
#     normalized_probs = [c / total for c in template_counts if c > 0]
#     shannon_entropy = -sum(p * math.log(p, 2) for p in normalized_probs)
#     return shannon_entropy


# def encode_repertoire_to_features_v2(metadata_df, data_path=None, k_mer_size=K_MER_SIZE, common_v=COMMON_V_GENES, common_j=COMMON_J_GENES, top_kmer=TOP_N_KMER):
#     """
#     Lit les TSV et extrait des features avancÃ©es (v2 + amÃ©liorations).
#     """
#     all_features = []
    
#     # Utilisation de tqdm pour la progression (si disponible)
#     import sys
#     if 'tqdm.notebook' in sys.modules:
#         iterator = tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Encodage V2 AmÃ©liorÃ©")
#     else:
#         iterator = metadata_df.iterrows()
    
#     for index, row in iterator:
#         tsv_file = row['filename']
        
#         try:
#             rep_df = pd.read_csv(tsv_file, sep='\t')
#         except FileNotFoundError:
#             continue

#         features = row.to_dict()
        
#         # --- FEATURE SET 1: Composition & Longueur ---
        
#         rep_df['len'] = rep_df['junction_aa'].astype(str).str.len()
#         features['mean_cdr3_len'] = rep_df['len'].mean()
#         features['std_cdr3_len'] = rep_df['len'].std()
        
#         # AJOUTS : Moments statistiques sur la longueur
#         features['skew_cdr3_len'] = rep_df['len'].skew()
#         features['kurtosis_cdr3_len'] = rep_df['len'].kurt()
        
#         def get_hydrophobicity(seq):
#             return sum(HYDROPHOBICITY.get(aa, 0) for aa in seq) / len(seq) if len(seq) > 0 else 0
            
#         rep_df['hydrophobicity'] = rep_df['junction_aa'].apply(get_hydrophobicity)
#         features['mean_hydrophobicity'] = rep_df['hydrophobicity'].mean()
        
#         # AJOUT : Ã‰cart-type de l'hydrophobie
#         features['std_hydrophobicity'] = rep_df['hydrophobicity'].std()
        
#         # AJOUT : Proportion d'AA invalides/non-fonctionnels
#         invalid_aa_count = rep_df['junction_aa'].str.contains(r'[^ACDEFGHIKLMNPQRSTVWY]').sum()
#         features['prop_invalid_aa'] = invalid_aa_count / len(rep_df) if len(rep_df) > 0 else 0


#         # --- FEATURE SET 2: DiversitÃ© & Structure ---
#         template_counts = rep_df['templates'].tolist() if 'templates' in rep_df.columns else [1] * len(rep_df)
#         features['shannon_diversity'] = calculate_shannon_diversity(template_counts)
#         features['unique_sequences_count'] = len(rep_df)
#         features['total_templates_sum'] = sum(template_counts)
#         if 'templates' in rep_df.columns:
#             sorted_templates = sorted(template_counts, reverse=True)
#             total_sum = sum(sorted_templates)
#             features['freq_top_10_clones'] = sum(sorted_templates[:10]) / total_sum if total_sum > 0 else 0
            
#         # --- FEATURE SET 3: GÃ¨nes V/J SpÃ©cifiques & k-mers ---
#         k_mers = []
#         for seq in rep_df['junction_aa'].astype(str):
#             if len(seq) >= k_mer_size:
#                 k_mers.extend([seq[i:i + k_mer_size] for i in range(len(seq) - k_mer_size + 1)])
        
#         kmer_counts = Counter(k_mers)
#         total_k_mers = sum(kmer_counts.values())
        
#         for kmer, count in kmer_counts.most_common(top_kmer):
#             if all(aa in AMINO_ACIDS for aa in kmer):
#                 features[f'kmer_freq_{kmer}'] = count / total_k_mers

#         v_counts = rep_df['v_call'].value_counts(normalize=True).to_dict()
#         j_counts = rep_df['j_call'].value_counts(normalize=True).to_dict()
        
#         for gene in common_v:
#             features[f'v_freq_{gene}'] = v_counts.get(gene, 0)
#         for gene in common_j:
#             features[f'j_freq_{gene}'] = j_counts.get(gene, 0)
            
#         all_features.append(features)

#     # CrÃ©ation du DataFrame final et nettoyage
#     df_features = pd.DataFrame(all_features)
    
#     # Remplacer les NaNs pour les features numÃ©riques *agrÃ©gÃ©es* si elles manquent
#     tsvs_cols = ['mean_cdr3_len', 'std_cdr3_len', 'skew_cdr3_len', 'kurtosis_cdr3_len', 'mean_hydrophobicity', 'std_hydrophobicity', 'prop_invalid_aa', 'shannon_diversity', 'unique_sequences_count', 'total_templates_sum', 'freq_top_10_clones']
#     v_j_kmer_cols = [col for col in df_features.columns if col.startswith(('v_freq_', 'j_freq_', 'kmer_freq_'))]
    
#     for col in tsvs_cols + v_j_kmer_cols:
#         if col in df_features.columns:
#             df_features[col] = df_features[col].fillna(0)
    
#     # Remplacement des NaNs dans le reste du DataFrame (mÃ©tadonnÃ©es) par 0
#     if TARGET_COL in df_features.columns:
#         cols_to_fill = df_features.columns.difference([TARGET_COL])
#         df_features[cols_to_fill] = df_features[cols_to_fill].fillna(0) 
#     else:
#         df_features = df_features.fillna(0)

#     df_features.columns = df_features.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
    
#     return df_features

# print("âœ… Fonctions d'encodage (V2 AmÃ©liorÃ©) dÃ©finies.")
# print("-" * 50)


# # --- 3. CHARGEMENT & ENCODAGE GLOBAL ---

# print("3. Chargement et prÃ©paration des donnÃ©es...")

# # 3.1 Chargement du fichier consolidÃ©
# all_meta = pd.read_csv(
#     TRAIN_METADATA_PATH, 
#     sep=',', 
#     skipinitialspace=True
# )

# # 3.2 Reconstruction du chemin du fichier TSV original
# all_meta['filename'] = all_meta.apply(
#     lambda row: os.path.join(ROOT_DIR, 'train_datasets', 'train_datasets', row['dataset'], row['repertoire_id'] + '.tsv'),
#     axis=1
# )

# # 3.3 Lancement de l'encodage lourd
# print(f"Lancement de l'encodage de {len(all_meta)} rÃ©pertoires (peut prendre du temps)...")
# full_train_df = encode_repertoire_to_features_v2(metadata_df=all_meta)

# print(f"\nâœ… Encodage global terminÃ©. Ensemble de donnÃ©es final: {full_train_df.shape}")
# print("-" * 50)


# # --- 4. PRÃ‰PARATION POUR LE ML ---

# # PrÃ©paration de X et y
# attribution_source_df = full_train_df.copy() # Garder une copie pour l'Ã©tape 6 (Attribution)
# X_cols_to_drop = ['repertoire_id', 'filename', TARGET_COL, 'dataset'] 
# X = full_train_df.drop(columns=X_cols_to_drop, errors='ignore')
# y = full_train_df[TARGET_COL].astype(int)

# # Conversion des types CatÃ©goriels pour LightGBM
# HLA_COLS = ['A', 'B', 'C', 'DPA1', 'DPB1', 'DQA1', 'DQB1', 'DRB1', 'DRB3', 'DRB4', 'DRB5']
# CATEGORICAL_COLS = [
#     'study_group_description', 
#     'sex', 
#     'race', 
#     'sequencing_run_id', 
#     'dataset'
# ] + HLA_COLS

# for col in CATEGORICAL_COLS:
#     if col in X.columns:
#         X[col] = X[col].fillna('Missing').astype('category')
        
# # Stocker les catÃ©gories pour le jeu de test
# TRAIN_CATEGORIES = {col: X[col].cat.categories for col in X.columns if X[col].dtype.name == 'category'} 
# X_train_cols = X.columns # Stocker l'ordre des colonnes
        
# print(f"âœ… PrÃ©paration terminÃ©e. X shape: {X.shape}")
# print("-" * 50)


# # --- 5. MODÃ‰LISATION ET VALIDATION CROISÃ‰E ---

# skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=42)
# lgbm = LGBMClassifier(
#     objective='binary', metric='auc', random_state=42, n_estimators=500, n_jobs=-1, verbose=-1,
#     categorical_feature=[col for col in CATEGORICAL_COLS if col in X.columns] 
# )

# cv_auc_scores = []
# oof_predictions = np.zeros(len(y))

# print(f"DÃ©but de la Cross-Validation StratifiÃ©e (K={N_SPLITS_CV})...")

# for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
#     X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
#     y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
    
#     lgbm.fit(X_train_fold, y_train_fold)
#     val_preds = lgbm.predict_proba(X_val_fold)[:, 1]
#     oof_predictions[val_index] = val_preds
    
#     fold_auc = roc_auc_score(y_val_fold, val_preds)
#     cv_auc_scores.append(fold_auc)
#     print(f"  Fold {fold+1}/{N_SPLITS_CV} - AUC: {fold_auc:.4f}")

# mean_cv_auc = np.mean(cv_auc_scores)
# oof_auc = roc_auc_score(y, oof_predictions)

# # Rappel du minimum requis (information utilisateur)
# min_score_required = 0.5 

# print(f"\nâœ¨ AUC MOYENNE de la Cross-Validation : {mean_cv_auc:.4f}")
# print(f"â­� AUC OOF GLOBAL (Estimation Leaderboard) : {oof_auc:.4f}")
# print(f"> Le minimum requis pour la compÃ©tition est de {min_score_required}")
# print("-" * 50)


# # --- 6. ENTRAÃ�NEMENT DU MODÃˆLE FINAL, SAUVEGARDE ET ATTRIBUTIONS (OPTIMISÃ‰E) ---

# print("6. EntraÃ®nement du modÃ¨le final, sauvegarde et gÃ©nÃ©ration des attributions...")
# lgbm.fit(X, y) 

# # SAUVEGARDE DU MODÃˆLE FINAL
# joblib.dump(lgbm, MODEL_FILENAME)
# print(f"âœ… ModÃ¨le LightGBM FINAL sauvegardÃ© sous : {MODEL_FILENAME}")

# # -----------------------------------------------------------------------
# # NOUVELLE SECTION OPTIMISÃ‰E POUR L'ATTRIBUTION (Moins de RAM/Timeout)
# # -----------------------------------------------------------------------

# print("DÃ©but de l'extraction des Attributions (TÃ¢che 2)...")

# # Importance des features liÃ©es aux sÃ©quences (k-mers, V/J)
# feature_importances = pd.Series(lgbm.feature_importances_, index=X.columns)
# relevant_importances = feature_importances[
#     feature_importances.index.str.startswith(('kmer_freq_', 'v_freq_', 'j_freq_'))]

# # Convertir les importances pertinentes en dictionnaire pour une recherche O(1)
# importance_dict = relevant_importances.to_dict()

# all_attributions = []
# MAX_ATTRIBUTIONS_PER_REP = 5000 # Prendre le top 5000 par rÃ©pertoire
# MAX_ATTRIBUTIONS_PER_DATASET = 50000 # Limite pour le top N par dataset avant fusion

# # Utiliser attribution_source_df pour lister les rÃ©pertoires positifs
# attribution_datasets = attribution_source_df['dataset'].unique()

# for dataset_id in tqdm(attribution_datasets, desc="Extraction des Attributions par Dataset"):
#     positive_reps = attribution_source_df[(attribution_source_df['dataset'] == dataset_id) & (attribution_source_df[TARGET_COL] == True)]
    
#     current_dataset_attributions = []
    
#     for _, row in positive_reps.iterrows():
#         tsv_file = row['filename']
#         try:
#             # 1. Lecture du TSV
#             rep_df = pd.read_csv(tsv_file, sep='\t')
#         except FileNotFoundError:
#             continue
        
#         if rep_df.empty:
#             continue
        
#         # Fonction de scoring optimisÃ©e qui utilise les dictionnaires d'importance
#         def calculate_score_fast(junction_aa, v_call, j_call):
#             score = 0.0
            
#             # Score des gÃ¨nes V et J
#             score += importance_dict.get(f'v_freq_{v_call}', 0)
#             score += importance_dict.get(f'j_freq_{j_call}', 0)
            
#             # Score des k-mers
#             if len(junction_aa) >= K_MER_SIZE_ATTR:
#                 for j in range(len(junction_aa) - K_MER_SIZE_ATTR + 1):
#                     kmer = junction_aa[j:j + K_MER_SIZE_ATTR]
#                     score += importance_dict.get(f'kmer_freq_{kmer}', 0)
            
#             return score

#         # 3. Application du score sur les colonnes
#         rep_df['importance_score'] = rep_df.apply(
#             lambda x: calculate_score_fast(x['junction_aa'], x['v_call'], x['j_call']), 
#             axis=1
#         )
        
#         # 4. Collection des Top SÃ©quences (Limitation de la RAM)
#         rep_df_sorted = rep_df.sort_values(by='importance_score', ascending=False)
#         # Prendre le top N par rÃ©pertoire pour ne pas surcharger la mÃ©moire
#         top_sequence_info = rep_df_sorted[['junction_aa', 'v_call', 'j_call', 'importance_score']].head(MAX_ATTRIBUTIONS_PER_REP) 
#         top_sequence_info['dataset'] = dataset_id
        
#         current_dataset_attributions.append(top_sequence_info)

#     # Consolidation des attributions pour le dataset en cours
#     if current_dataset_attributions:
#         combined_attributions = pd.concat(current_dataset_attributions)
        
#         # Filtrer et conserver le top 50 000 par dataset, basÃ© sur le score global
#         final_attributions_for_dataset = combined_attributions.sort_values(
#             by='importance_score', ascending=False
#         ).head(MAX_ATTRIBUTIONS_PER_DATASET)
        
#         all_attributions.append(final_attributions_for_dataset)

# # -----------------------------------------------------------------------
# # FINALISATION DE L'ARTEFACT D'ATTRIBUTION
# # -----------------------------------------------------------------------

# if all_attributions:
#     df_attributions = pd.concat(all_attributions).reset_index(drop=True)
    
#     # Nous gardons le TOP 400k des attributions selon les rÃ¨gles courantes (Ã  ajuster si besoin)
#     MAX_TOTAL_ATTRIBUTIONS = 400000 
#     df_attributions = df_attributions.sort_values(by='importance_score', ascending=False).head(MAX_TOTAL_ATTRIBUTIONS)

#     # PrÃ©paration finale pour la soumission
#     df_attributions = df_attributions[['dataset', 'junction_aa', 'v_call', 'j_call']]
    
#     # Sauvegarde finale
#     ATTRIBUTION_FILENAME = 'df_attributions.csv'
#     df_attributions.to_csv(ATTRIBUTION_FILENAME, index=False)
    
#     print(f"âœ… Attributions finalisÃ©es. Total de lignes : {len(df_attributions)}")
#     print(f"âœ… Attributions sauvegardÃ©es sous : {ATTRIBUTION_FILENAME}")

# else:
#     print("â�Œ Avertissement : Aucune attribution gÃ©nÃ©rÃ©e.")
# print("-" * 50)


# -------------------------------------------------------------
# CELLE COMPLÃˆTE POUR SOUMISSION TEST + ATTRIBUTIONS SÃ‰CURISÃ‰E
# (FINAL FIX: Remplacement du slicing dangereux par une reconstruction sÃ»re pour les IDs dupliquÃ©s)
# -------------------------------------------------------------
import pandas as pd
import numpy as np
import joblib
from glob import glob
from collections import Counter
from itertools import cycle
import math
import os
from tqdm.notebook import tqdm
from typing import List
import warnings

warnings.filterwarnings('ignore')

# ------------------- CONSTANTES -------------------
ROOT_DIR = "/kaggle/input/adaptive-immune-profiling-challenge-2025"
TEST_DATA_FOLDER = f"{ROOT_DIR}/test_datasets/test_datasets"
SUBMISSION_BASE_PATH = f"{ROOT_DIR}/sample_submissions.csv"
METADATA_STUDIES_PATH = f"{ROOT_DIR}/metadata_studies.csv"
ATTRIBUTION_FILENAME = '/kaggle/input/airr-df-attributions/df_attributions.csv'
MODEL_PATH = '/kaggle/input/lgbm-final-model/lgbm_final_model.joblib'

TARGET_COL = 'label_positive'
K_MER_SIZE = 3
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
COMMON_V_GENES = [f'TRBV{i}' for i in range(1, 30)]
COMMON_J_GENES = [f'TRBJ{i}' for i in range(1, 7)]
MISSING_VALUE_REPLACEMENT = 0.0 # Remplacement des NaNs dans les features numÃ©riques (Train/Test)
MISSING_SUBMISSION_VALUE = -999.0 # Valeur requise pour la section d'Attribution
SUBMISSION_FILENAME = 'submission.csv'

CATEGORICAL_COLS = [
    'study_group_description', 'sex', 'race', 'sequencing_run_id', 'dataset',
    'A', 'B', 'C', 'DPA1', 'DPB1', 'DQA1', 'DQB1', 'DRB1', 'DRB3', 'DRB4', 'DRB5'
]

HYDROPHOBICITY = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4,
                  'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
                  'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
                  'W': -0.9, 'Y': -1.3}

# ------------------- MOCK/SETUP POUR ROBUSTESSE -------------------
class MockModel:
    def __init__(self, feature_name): self.feature_name_ = feature_name
    def predict_proba(self, X): return np.array([[0.5, 0.5]] * len(X))

# ------------------- FONCTIONS UTILITAIRES -------------------
def calculate_shannon_diversity(template_counts):
    if not template_counts or sum(template_counts) == 0:
        return 0.0
    total = sum(template_counts)
    probs = [c / total for c in template_counts if c > 0]
    return -sum(p * math.log(p, 2) for p in probs)

def encode_repertoire_to_features_v2(metadata_df: pd.DataFrame, x_train_cols_ref: List[str], k_mer_size=K_MER_SIZE, common_v=COMMON_V_GENES, common_j=COMMON_J_GENES):
    all_features = []
    # TRAIN_KMER_COLS contient tous les K-mers que le modÃ¨le connaÃ®t
    TRAIN_KMER_COLS = [col.replace('kmer_freq_', '') for col in x_train_cols_ref if col.startswith('kmer_freq_')]
    
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Encodage Test V2"):
        tsv_file = row['filename']
        features = row.to_dict()
        
        try:
            rep_df = pd.read_csv(tsv_file, sep='\t', engine='c')
        except Exception:
            all_features.append(features)
            continue
        if 'junction_aa' not in rep_df.columns:
            all_features.append(features)
            continue

        # --- Features longueur & hydrophobicitÃ© ---
        rep_df['len'] = rep_df['junction_aa'].astype(str).str.len()
        features['mean_cdr3_len'] = rep_df['len'].mean()
        features['std_cdr3_len'] = rep_df['len'].std()
        features['skew_cdr3_len'] = rep_df['len'].skew()
        features['kurtosis_cdr3_len'] = rep_df['len'].kurt()
        rep_df['hydrophobicity'] = rep_df['junction_aa'].apply(lambda seq: sum(HYDROPHOBICITY.get(aa, 0) for aa in seq)/len(seq) if len(seq)>0 else 0)
        features['mean_hydrophobicity'] = rep_df['hydrophobicity'].mean()
        features['std_hydrophobicity'] = rep_df['hydrophobicity'].std()
        invalid_aa_count = rep_df['junction_aa'].str.contains(r'[^ACDEFGHIKLMNPQRSTVWY]').sum()
        features['prop_invalid_aa'] = invalid_aa_count / len(rep_df) if len(rep_df) > 0 else 0

        # --- DiversitÃ© & clonotypes ---
        template_counts = rep_df['templates'].tolist() if 'templates' in rep_df.columns else [1]*len(rep_df)
        features['shannon_diversity'] = calculate_shannon_diversity(template_counts)
        features['unique_sequences_count'] = len(rep_df)
        features['total_templates_sum'] = sum(template_counts)
        total_sum_templates = sum(template_counts)
        sorted_templates = sorted(template_counts, reverse=True)
        features['freq_top_10_clones'] = sum(sorted_templates[:10]) / total_sum_templates if total_sum_templates > 0 else 0

        # --- k-mers (alignement FORCÃ‰ au vocabulaire train) ---
        k_mers = []
        for seq in rep_df['junction_aa'].dropna().astype(str):
            if len(seq) >= k_mer_size:
                k_mers.extend([seq[i:i+k_mer_size] for i in range(len(seq)-k_mer_size+1)])
        kmer_counts = Counter(k_mers)
        total_k_mers = sum(kmer_counts.values())

        # CrÃ©er toutes les colonnes K-mers de Train, mÃªme si non trouvÃ©es (valeur 0.0)
        for kmer_name in TRAIN_KMER_COLS:
            count = kmer_counts.get(kmer_name, 0)
            features[f'kmer_freq_{kmer_name}'] = count / total_k_mers if total_k_mers > 0 else 0.0

        # --- V/J gene frequencies ---
        v_counts = rep_df['v_call'].fillna('').value_counts(normalize=True).to_dict() if 'v_call' in rep_df.columns else {}
        j_counts = rep_df['j_call'].fillna('').value_counts(normalize=True).to_dict() if 'j_call' in rep_df.columns else {}
        for gene in common_v: features[f'v_freq_{gene}'] = v_counts.get(gene, 0)
        for gene in common_j: features[f'j_freq_{gene}'] = j_counts.get(gene, 0)

        all_features.append(features)

    df_features = pd.DataFrame(all_features)
    df_features.columns = df_features.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
    return df_features

# ------------------- CHARGEMENT ET PRÃ‰PARATION -------------------
print("1. Chargement des artefacts de base...")

# 1) Load model (and get X_train_cols)
try:
    lgbm = joblib.load(MODEL_PATH)
    print("âœ… ModÃ¨le LGBM chargÃ©.")
    
    try:
        if hasattr(lgbm, "feature_name_"):
            X_train_cols = list(lgbm.feature_name_)
        elif hasattr(lgbm, "base_estimator_") and hasattr(lgbm.base_estimator_, "feature_name_"):
            X_train_cols = list(lgbm.base_estimator_.feature_name_)
        else:
            raise AttributeError("Attribut feature_name_ introuvable sur le modÃ¨le chargÃ©.")
        print(f"âœ… {len(X_train_cols)} features d'entraÃ®nement trouvÃ©es.")
        
    except AttributeError as ae:
        print(f"âš ï¸� Erreur lors de la rÃ©cupÃ©ration des features : {ae}. Utilisation du MockModel pour les colonnes.")
        # Utilisation du MockModel si les colonnes ne peuvent pas Ãªtre rÃ©cupÃ©rÃ©es
        mock_kmers = [f'kmer_freq_{aa1}{aa2}{aa3}' for aa1 in AMINO_ACIDS for aa2 in AMINO_ACIDS for aa3 in AMINO_ACIDS][:100]
        X_train_cols = ['mean_cdr3_len', 'shannon_diversity'] + mock_kmers + CATEGORICAL_COLS
        lgbm = MockModel(X_train_cols)
        print("âš ï¸� ALERTE : Soumission finale doit utiliser le VRAI modÃ¨le, pas le MockModel (probabilitÃ© 0.5).")

except Exception as e:
    print(f"â�Œ Impossible de charger le modÃ¨le LGBM depuis {MODEL_PATH} ({e}). Utilisation d'un MockModel.")
    # Utilisation du MockModel si le modÃ¨le ne peut pas Ãªtre chargÃ©
    mock_kmers = [f'kmer_freq_{aa1}{aa2}{aa3}' for aa1 in AMINO_ACIDS for aa2 in AMINO_ACIDS for aa3 in AMINO_ACIDS][:100]
    X_train_cols = ['mean_cdr3_len', 'shannon_diversity'] + mock_kmers + CATEGORICAL_COLS
    lgbm = MockModel(X_train_cols)
    print("âš ï¸� ALERTE : Soumission finale doit utiliser le VRAI modÃ¨le, pas le MockModel (probabilitÃ© 0.5).")


# 2) Load submission base (we want to work with repertoire_id internally)
try:
    # Lecture du fichier de soumission de base
    # NOTE IMPORTANTE: Nous renommons ID_original pour le nettoyer plus tard et Ã©viter la duplication ID/ID
    df_submission_base = pd.read_csv(SUBMISSION_BASE_PATH).rename(columns={'ID':'ID_original', 'repertoire_id':'ID_original'})
    
    # S'assurer que le nom repertoire_id est correct pour la fusion
    if 'ID_original' in df_submission_base.columns:
        # Renommer la colonne ID_original en 'repertoire_id' pour la logique de prÃ©diction
        df_submission_base.rename(columns={'ID_original':'repertoire_id'}, inplace=True)
        # Mais conserver l'autre 'ID' si elle existe pour la soumission finale (bien que vide)
        if len(df_submission_base.filter(like='ID_original').columns) > 1:
            df_submission_base.drop(columns=df_submission_base.filter(like='ID_original').columns[1:], inplace=True)


    # FIX: Supprimer toute colonne d'index non nommÃ©e (qui crÃ©e le ID dupliquÃ©)
    cols_to_drop_on_load = [col for col in df_submission_base.columns if 'Unnamed:' in str(col)]
    df_submission_base.drop(columns=cols_to_drop_on_load, inplace=True, errors='ignore')
    
    # S'assurer que l'ID principal est bien 'repertoire_id'
    if 'ID' in df_submission_base.columns and 'repertoire_id' not in df_submission_base.columns:
        df_submission_base.rename(columns={'ID':'repertoire_id'}, inplace=True)
    elif 'ID' in df_submission_base.columns and 'repertoire_id' in df_submission_base.columns:
        # Si les deux existent, garder repertoire_id et supprimer ID (le vide/dupliquÃ©)
        df_submission_base.drop(columns=['ID'], inplace=True, errors='ignore')

    # Reconfirmer l'ID Ã  prÃ©dire
    repertoire_ids_to_predict = df_submission_base[df_submission_base['repertoire_id'].astype(str)!=str(MISSING_SUBMISSION_VALUE)]['repertoire_id'].tolist()
    print(f"âœ… Fichier de soumission de base chargÃ©. {len(df_submission_base)} lignes totales.")
except FileNotFoundError:
    # MOCK DATA
    df_submission_base = pd.DataFrame({
        'repertoire_id':[f'mock_{i}' for i in range(100)],
        'dataset':['dataset_1']*100,
        'label_positive_probability':[0.5]*100,
        'junction_aa':['']*100,
        'v_call':['']*100,
        'j_call':['']*100
    })
    df_submission_base.loc[50:, 'repertoire_id'] = MISSING_SUBMISSION_VALUE
    repertoire_ids_to_predict = df_submission_base[df_submission_base['repertoire_id'].astype(str)!=str(MISSING_SUBMISSION_VALUE)]['repertoire_id'].tolist()
    print(f"âš ï¸� sample_submissions non trouvÃ© -> mock crÃ©Ã©. {len(repertoire_ids_to_predict)} rÃ©pertoires Ã  prÃ©dire.")

# MÃ©tadonnÃ©es et Attributions
try:
    df_studies = pd.read_csv(METADATA_STUDIES_PATH).rename(columns={'ID':'repertoire_id'})
    df_studies['repertoire_id'] = df_studies['repertoire_id'].astype(str)
except FileNotFoundError:
    df_studies = pd.DataFrame(columns=['repertoire_id'] + CATEGORICAL_COLS)
try:
    df_attributions = pd.read_csv(ATTRIBUTION_FILENAME).fillna("")
except FileNotFoundError:
    df_attributions = pd.DataFrame(columns=['junction_aa','v_call','j_call','dataset'])

# Mapping des fichiers TSV (test)
all_test_tsv_paths = glob(f"{TEST_DATA_FOLDER}/**/*.tsv", recursive=True)
test_tsv_path_mapping = {}
for path in all_test_tsv_paths:
    repertoire_id = os.path.splitext(os.path.basename(path))[0]
    dataset_id = os.path.basename(os.path.dirname(path))
    if repertoire_id in repertoire_ids_to_predict:
        test_tsv_path_mapping[repertoire_id] = {'repertoire_id':repertoire_id,'filename':path,'dataset':dataset_id,'dataset_id':dataset_id}

all_test_meta = pd.DataFrame.from_dict(test_tsv_path_mapping, orient='index').reset_index(drop=True)
if not all_test_meta.empty:
    all_test_meta['filename'] = all_test_meta['filename'].str.replace('\\','/',regex=False)
all_test_meta = all_test_meta.merge(df_studies.drop(columns=['dataset_id','dataset'],errors='ignore'), on='repertoire_id', how='left')

print("2. Encodage des features...")
full_test_df = encode_repertoire_to_features_v2(all_test_meta, x_train_cols_ref=X_train_cols)
VALID_TEST_DATASETS = full_test_df['dataset'].unique().tolist() if 'dataset' in full_test_df.columns else []
if not VALID_TEST_DATASETS:
    VALID_TEST_DATASETS = ['dataset_1']

# --- Logging de dÃ©bogage pour les K-mers ---
KMER_COLS_CHECK = [col for col in full_test_df.columns if col.startswith('kmer_freq_')]
if not KMER_COLS_CHECK:
    print("âš ï¸� Attention: Aucune colonne de k-mer n'a Ã©tÃ© trouvÃ©e dans les features de test.")
else:
    # Compter le nombre de colonnes de K-mers oÃ¹ au moins un rÃ©pertoire a une frÃ©quence > 0
    non_zero_kmer_count = (full_test_df[KMER_COLS_CHECK] != 0.0).any(axis=0).sum()
    print(f"ğŸ”� DEBUG K-MERS: {non_zero_kmer_count} colonnes de k-mers sur {len(KMER_COLS_CHECK)} ont des valeurs non-nulles dans le jeu de test.")
# ----------------------------------------------------------------------------------

# ------------------- ALIGNEMENT ET PRÃ‰DICTION -------------------
# Build X_test (index = repertoire_id)
if 'repertoire_id' not in full_test_df.columns:
    full_test_df = full_test_df.reset_index().rename(columns={'index':'repertoire_id'})

X_test = full_test_df.set_index('repertoire_id').drop(columns=['filename','dataset_id'], errors='ignore')

# Restrict to rows that are present in sample_submissions (safety)
pred_ids = list(set(X_test.index).intersection(set(repertoire_ids_to_predict)))
X_test = X_test.loc[pred_ids]

# Prepare aligned DF with correct columns/order from training
X_test_aligned = pd.DataFrame(index=X_test.index, columns=X_train_cols)

# Precompute numerical and categorical columns lists
NUMERICAL_COLS = [col for col in X_train_cols if col not in CATEGORICAL_COLS]

# ğŸ�¯ FIX MAJEUR D'ALIGNEMENT: Remplissage et alignement forcÃ©
for col in X_train_cols:
    if col in X_test.columns:
        X_test_aligned[col] = X_test[col]
    else:
        # Si la colonne est connue Ã  l'entraÃ®nement mais absente de l'encodage test,
        # elle sera remplie par NaN ici, puis imputÃ©e correctement par la suite.
        X_test_aligned[col] = np.nan

# Categorical handling: replace NaN with 'Missing' and set dtype 'category'
for col in CATEGORICAL_COLS:
    if col in X_test_aligned.columns:
        # Si le modÃ¨le a Ã©tÃ© entraÃ®nÃ© avec des catÃ©gories (LightGBM), on doit les conserver.
        X_test_aligned[col] = X_test_aligned[col].fillna('Missing').astype(str)
        X_test_aligned[col] = pd.Categorical(X_test_aligned[col])

# Numeric handling: coerce to numeric then impute
for col in NUMERICAL_COLS:
    if col in X_test_aligned.columns:
        X_test_aligned[col] = pd.to_numeric(X_test_aligned[col], errors='coerce')

# Imputation de 0.0 UNIQUEMENT sur les colonnes numÃ©riques (CORRECTIF)
X_test_aligned[NUMERICAL_COLS] = X_test_aligned[NUMERICAL_COLS].fillna(MISSING_VALUE_REPLACEMENT)

# Final safety: ensure index sorted same as predictions list
X_test_aligned = X_test_aligned.loc[sorted(X_test_aligned.index)]

# Predict (handle empty gracefully)
if len(X_test_aligned) > 0:
    test_preds_proba = lgbm.predict_proba(X_test_aligned)[:,1]
    df_predictions = pd.DataFrame({'repertoire_id':X_test_aligned.index,'label_positive_probability':test_preds_proba})
    print(f"âœ… PrÃ©dictions effectuÃ©es pour {len(df_predictions)} rÃ©pertoires.")
else:
    df_predictions = pd.DataFrame({'repertoire_id':[],'label_positive_probability':[]})
    print("âš ï¸� Aucun rÃ©pertoire Ã  prÃ©dire (X_test_aligned vide).")

# ------------------- SOUMISSION -------------------
# Classification section (predicted rows)
df_classification_section = df_submission_base[df_submission_base['repertoire_id'].isin(df_predictions['repertoire_id'])].copy()
if 'dataset' in full_test_df.columns:
    df_classification_section['dataset'] = df_classification_section['repertoire_id'].map(full_test_df.set_index('repertoire_id')['dataset']).fillna(df_classification_section.get('dataset', np.nan))
df_classification_section = df_classification_section.merge(df_predictions, on='repertoire_id', how='left', suffixes=('_base','_pred'))
df_classification_section['label_positive_probability'] = df_classification_section['label_positive_probability_pred'].fillna(MISSING_VALUE_REPLACEMENT)
df_classification_section = df_classification_section[['repertoire_id','dataset','label_positive_probability','junction_aa','v_call','j_call']]


# ğŸ�¯ FIX 2: Attribution section (Robust logic for identifying -999.0 rows)
classified_ids = df_classification_section['repertoire_id'].tolist()
# Get all rows from base that were NOT in the classification section (ce sont les -999.0)
df_attribution_section = df_submission_base[~df_submission_base['repertoire_id'].isin(classified_ids)].copy()

required_len = len(df_attribution_section)
cols_to_map = ['junction_aa','v_call','j_call']

# Keep only attributions that belong to VALID_TEST_DATASETS
df_attributions = df_attributions[df_attributions['dataset'].isin(VALID_TEST_DATASETS)].copy() if not df_attributions.empty else df_attributions
df_attributions_mapped = df_attributions[cols_to_map].head(required_len) if not df_attributions.empty else pd.DataFrame(columns=cols_to_map)

# If not enough attributions, pad with empty strings (Confirmation: c'est la bonne pratique)
if len(df_attributions_mapped) < required_len:
    missing_rows = required_len - len(df_attributions_mapped)
    # Assurer que les colonnes manquantes sont initialisÃ©es
    missing_data = {col: [""]*missing_rows for col in cols_to_map}
    df_attributions_mapped = pd.concat([df_attributions_mapped, pd.DataFrame(missing_data)], ignore_index=True)
    df_attributions_mapped = df_attributions_mapped.iloc[:required_len] # Tronquer si trop de lignes ajoutÃ©es/existantes

if required_len > 0:
    df_attribution_section['junction_aa'] = df_attributions_mapped['junction_aa'].values
    df_attribution_section['v_call'] = df_attributions_mapped['v_call'].values
    df_attribution_section['j_call'] = df_attributions_mapped['j_call'].values
    dataset_cycle = cycle(VALID_TEST_DATASETS)
    df_attribution_section['dataset'] = [next(dataset_cycle) for _ in range(required_len)]
    # Use -999.0 for attribution probability column as required by the competition
    df_attribution_section['label_positive_probability'] = float(MISSING_SUBMISSION_VALUE)

# rename attribution section's repertoire_id -> ID (as expected in final file)
df_attribution_section.rename(columns={'repertoire_id':'ID'}, inplace=True)

# Concatenate classification + attribution
df_final_submission = pd.concat([df_classification_section, df_attribution_section], ignore_index=True)
# For the classification rows we still have 'repertoire_id' column; rename it to 'ID' to match sample format
if 'repertoire_id' in df_final_submission.columns:
    df_final_submission.rename(columns={'repertoire_id':'ID'}, inplace=True)


# ğŸ�¯ FIX 3: Nettoyage et Ã©limination de la colonne ID dupliquÃ©e/vide (avant l'export)
# Supprimer les colonnes d'index non nÃ©cessaires ou les duplicata d'ID accidentels.
cols_to_drop = [col for col in df_final_submission.columns if 'Unnamed:' in str(col)]
df_final_submission.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# ğŸ�¯ FIX ULTIME 3.0 (CORRIGÃ‰ V5): Remplacement du Slicing Dangereux par une Reconstruction SÃ»re
if list(df_final_submission.columns).count("ID") > 1:
    cols = df_final_submission.columns
    id_cols = [i for i, c in enumerate(cols) if c == "ID"]
    keep = id_cols[-1]

    new_cols = []
    for i, c in enumerate(cols):
        if c != "ID":
            new_cols.append(c)
        else:
            if i == keep:
                new_cols.append(c)  # garder uniquement cette ID

    # Cette opÃ©ration de reconstruction est la plus robuste (remplace le slicing ambigu)
    df_final_submission = df_final_submission[new_cols] 


# Build expected column order from original sample_submissions but adapt 'repertoire_id' -> 'ID'
original_cols = df_submission_base.columns.tolist()
expected_cols = [c if c != 'repertoire_id' else 'ID' for c in original_cols]
# Rendre la liste des colonnes attendues unique (pour gÃ©rer les IDs dupliquÃ©s accidentels)
expected_cols_unique = []
seen = set()
for col in expected_cols:
    if col not in seen:
        expected_cols_unique.append(col)
        seen.add(col)
expected_cols = expected_cols_unique
    
# final selection - if some expected columns are missing, keep intersection but warn
missing_expected = [c for c in expected_cols if c not in df_final_submission.columns]
if missing_expected:
    print(f"âš ï¸� Colonnes attendues manquantes dans df_final_submission (elles seront ajoutÃ©es en tant que vide) : {missing_expected}")
    for c in missing_expected:
        df_final_submission[c] = "" # pad missing columns with empty strings

df_final_submission = df_final_submission[expected_cols]

# ------------------- SUPER-SAFE TYPE FIX (CORRIGÃ‰ V4) -------------------

# Colonnes obligatoires pour strings
STRING_COLUMNS = ['ID', 'dataset', 'junction_aa', 'v_call', 'j_call']
NUMERIC_COLUMNS = ['label_positive_probability']

# 1. Assurer que toutes les colonnes string sont bien remplies et en str
for col in STRING_COLUMNS:
    if col in df_final_submission.columns:
        df_final_submission[col] = df_final_submission[col].fillna("").astype(str)

# 2. Colonnes numÃ©riques (probabilitÃ©)
for col in NUMERIC_COLUMNS:
    if col in df_final_submission.columns:
        # Coerce to numeric (float), remplace les non-convertibles par MISSING_SUBMISSION_VALUE (-999.0)
        df_final_submission[col] = pd.to_numeric(df_final_submission[col], errors='coerce').fillna(MISSING_SUBMISSION_VALUE).astype(float)

# 3. SÃ©curitÃ© ultime: s'assurer qu'aucune colonne n'est object inattendue
# L'objet est maintenant garanti d'Ãªtre un DataFrame (grÃ¢ce Ã  FIX 3.0 V5)
for col, dtype in df_final_submission.dtypes.items():
    if dtype == 'object' and col not in STRING_COLUMNS:
        # Convertir tout objet restant en string (sÃ©curitÃ© ultime)
        df_final_submission[col] = df_final_submission[col].astype(str)

print("âœ… Types forcÃ©s avec succÃ¨s avant export.")
# ------------------------------------------------------------------------------------------------------------------------------------

# ====================================================================
# ğŸ”� BLOC D'INSPECTION DU DATAFRAME FINAL
# ====================================================================
print("\n--- ğŸ”� INSPECTION DU DATAFRAME FINAL ---")
print(f"Total des lignes: {len(df_final_submission)}")

# Afficher les 5 premiÃ¨res lignes (VÃ©rification des IDs et probabilitÃ©s)
print("\n[HEAD - 5 premiÃ¨res lignes]")
print(df_final_submission[['ID', 'label_positive_probability', 'junction_aa']].head())

# Afficher les 5 derniÃ¨res lignes (VÃ©rification du padding -999.0)
print("\n[TAIL - 5 derniÃ¨res lignes]")
print(df_final_submission[['ID', 'label_positive_probability', 'junction_aa']].tail())

# VÃ©rification des types de colonnes
print("\n[DTYPES - VÃ©rification des types de colonnes]")
print(df_final_submission.dtypes)

# VÃ©rification rapide des valeurs spÃ©ciales
num_minus_999 = (df_final_submission['label_positive_probability'] == float(MISSING_SUBMISSION_VALUE)).sum()
print(f"\nNombre de lignes d'attribution (-999.0): {num_minus_999}")
print("-------------------------------------------\n")
# ====================================================================


# ------------------- CHECK SÃ‰CURITÃ‰ -------------------
if 'dataset' in df_final_submission.columns:
    train_datasets_in_submission = [d for d in df_final_submission['dataset'].unique() if str(d).startswith('train')]
    if train_datasets_in_submission:
        raise ValueError(f"â�Œ Train datasets prÃ©sents dans la soumission : {train_datasets_in_submission}")

# Export
df_final_submission.to_csv(SUBMISSION_FILENAME, index=False)
print(f"ğŸ�‰ Fichier de soumission final crÃ©Ã©: {SUBMISSION_FILENAME}")
print(f"Nombre total de lignes: {len(df_final_submission)}")

