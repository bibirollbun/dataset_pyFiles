import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import json

# --- 1. Chargement des donnÃ©es ---
train = pd.read_csv("/kaggle/input/mercor-cheating-detection/train.csv")
test = pd.read_csv("/kaggle/input/mercor-cheating-detection/test.csv")
graph_df = pd.read_csv("/kaggle/input/mercor-cheating-detection/social_graph.csv")

# --- 2. IngÃ©nierie de Feature : Degree Centrality ---
G = nx.from_pandas_edgelist(graph_df, 'user_a', 'user_b')
degree_dict = dict(G.degree())
degree_df = pd.DataFrame(list(degree_dict.items()), columns=['user_hash', 'degree_centrality'])

# Fusionner la feature de degrÃ©
train = train.merge(degree_df, on='user_hash', how='left')
test = test.merge(degree_df, on='user_hash', how='left')

# Imputer les NaN (utilisateurs isolÃ©s) avec 0
train['degree_centrality'] = train['degree_centrality'].fillna(0)
test['degree_centrality'] = test['degree_centrality'].fillna(0)

# DÃ©finir les features Ã  utiliser
FEATURES = [f'feature_{i:03d}' for i in range(1, 19)] + ['degree_centrality']

print("PrÃ©paration des donnÃ©es terminÃ©e. Features utilisÃ©es :")
print(FEATURES)


import networkx as nx
import pandas as pd
import time

print("DÃ©but du calcul des centralitÃ©s avancÃ©es...")

# --- 1. Calcul de la CentralitÃ© d'IntermÃ©diaritÃ© (Betweenness) ---
# CORRECTION CRITIQUE : Utilisation de k=2000 pour l'Ã©chantillonnage stochastique.
# Cela permet d'obtenir une bonne approximation trÃ¨s rapidement, Ã©vitant le blocage du kernel.
start_time = time.time()
betweenness_dict = nx.betweenness_centrality(G, k=2000, normalized=True, seed=42) 

betweenness_df = pd.DataFrame(
    list(betweenness_dict.items()), 
    columns=['user_hash', 'betweenness_centrality']
)
print(f"Betweenness Centrality (approximÃ©e) calculÃ©e en {time.time() - start_time:.2f} secondes.")


# --- 2. Calcul de la CentralitÃ© de ProximitÃ© (Closeness) ---
# Moins coÃ»teux, le calcul exact peut Ãªtre conservÃ©.
start_time = time.time()
closeness_dict = nx.closeness_centrality(G)

closeness_df = pd.DataFrame(
    list(closeness_dict.items()), 
    columns=['user_hash', 'closeness_centrality']
)
print(f"Closeness Centrality calculÃ©e en {time.time() - start_time:.2f} secondes.")


# --- 3. Fusion et Imputation des nouvelles features ---

# Fusionner les deux nouvelles DataFrames de centralitÃ©
centrality_df = pd.merge(betweenness_df, closeness_df, on='user_hash', how='outer')

# Fusionner avec les DataFrames train et test existants
train = train.merge(centrality_df, on='user_hash', how='left')
test = test.merge(centrality_df, on='user_hash', how='left')


# Imputation : Les nÅ“uds isolÃ©s (valeurs NaN aprÃ¨s la fusion) reÃ§oivent une centralitÃ© de 0.
train['betweenness_centrality'] = train['betweenness_centrality'].fillna(0)
test['betweenness_centrality'] = test['betweenness_centrality'].fillna(0)

train['closeness_centrality'] = train['closeness_centrality'].fillna(0)
test['closeness_centrality'] = test['closeness_centrality'].fillna(0)


# --- 4. Mise Ã  jour de la liste des features ---
NEW_FEATURES = ['betweenness_centrality', 'closeness_centrality']
FEATURES = FEATURES + NEW_FEATURES 

print("\nâœ… Nouvelles features de graphe ajoutÃ©es avec succÃ¨s.")
print(f"Nouvelle liste de FEATURES : {FEATURES}")


# --- 1. StratÃ©gie Semi-SupervisÃ©e / CrÃ©ation de l'Ensemble d'EntraÃ®nement ---

# CrÃ©er un label pour les donnÃ©es high_conf_clean :
# L'Ã©valuation initiale suggÃ¨re que ces utilisateurs sont des non-tricheurs (0.0).
# Nous fusionnons les candidats Ã©tiquetÃ©s (is_cheating est 0 ou 1) et les high_conf_clean.

# CrÃ©er la colonne cible Y_full
train['target'] = train['is_cheating'].copy()

# Remplacer les NaN (qui sont high_conf_clean=1.0, comme observÃ©) par 0.0
# Nous supposons que high_conf_clean = 1.0 signifie non-tricheur (0.0).
train['target'] = train['target'].fillna(0.0)

# DÃ©finir les ensembles X et Y
X_train_full = train[FEATURES].copy()
y_train_full = train['target'].copy()
X_test = test[FEATURES].copy()

# --- 2. Imputation des Valeurs Manquantes ---

# Calculer la mÃ©diane pour l'entraÃ®nement (Ã  utiliser aussi sur le test)
median_imputer = X_train_full.median()

# Imputer les ensembles d'entraÃ®nement et de test
X_train_imputed = X_train_full.fillna(median_imputer)
X_test_imputed = X_test.fillna(median_imputer)

print("\nImputation (par la mÃ©diane) et gestion du semi-supervisÃ© terminÃ©es.")
print(f"Taille de l'ensemble d'entraÃ®nement (incluant les 'high_conf_clean') : {X_train_imputed.shape[0]}")


# Importations nÃ©cessaires (assumÃ©es dans les cellules prÃ©cÃ©dentes)
# from sklearn.model_selection import KFold
# from sklearn.metrics import roc_auc_score
# import lightgbm as lgb
# import numpy as np

# --- 1. Configuration de l'entraÃ®nement ---
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Calcul du scale_pos_weight pour gÃ©rer le dÃ©sÃ©quilibre et le coÃ»t FN Ã©levÃ© ($600)
# N_NEGATIVES = (y_train_full == 0.0).sum() 
# N_POSITIVES = (y_train_full == 1.0).sum() 
# SCALE_POS_WEIGHT = N_NEGATIVES / N_POSITIVES # Ancien ratio : â‰ˆ 6.92

# --- HYPOTHÃˆSE DU COÃ›T : PONDÃ‰RATION AGRESSIVE ---
# Nous fixons la pondÃ©ration Ã  15.0 pour forcer le modÃ¨le Ã  minimiser le coÃ»t FN ($600).
SCALE_POS_WEIGHT_TEST = 15.0 

# Configuration des hyperparamÃ¨tres de LightGBM (Baseline)
lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1, # Pour masquer les logs
    'n_jobs': -1,
    'seed': 42,
    # <<< MODIFICATION CLÃ‰ >>>
    'scale_pos_weight': SCALE_POS_WEIGHT_TEST 
}

# Initialisation pour stocker les rÃ©sultats
# X_train_imputed et X_test_imputed contiennent maintenant toutes les features de graphe !
oof_predictions = np.zeros(X_train_imputed.shape[0]) 
test_predictions = np.zeros(X_test_imputed.shape[0]) 

# --- 2. EntraÃ®nement par K-Fold Cross-Validation ---
print("\nDÃ©but de l'entraÃ®nement LightGBM (5-Fold CV)...")
print(f"PondÃ©ration de la classe positive (scale_pos_weight) utilisÃ©e : {SCALE_POS_WEIGHT_TEST:.2f} (HypothÃ¨se Agressive : 15.0)")

for fold, (train_index, val_index) in enumerate(kf.split(X_train_imputed, y_train_full)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    
    X_train, X_val = X_train_imputed.iloc[train_index], X_train_imputed.iloc[val_index]
    y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]
    
    model = lgb.LGBMClassifier(**lgbm_params)
    
    # EntraÃ®nement avec early stopping
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False)])
    
    # PrÃ©diction sur le fold de validation (OOF)
    oof_predictions[val_index] = model.predict_proba(X_val)[:, 1]
    
    # PrÃ©diction sur l'ensemble de test (moyennÃ©e Ã  la fin)
    test_predictions += model.predict_proba(X_test_imputed)[:, 1] / N_SPLITS

# --- 3. Ã‰valuation OOF (sur l'ensemble d'entraÃ®nement) ---
oof_auc = roc_auc_score(y_train_full, oof_predictions)
print(f"\nâœ… AUC OOF (Score de Calibration Interne) : {oof_auc:.4f}")


import pandas as pd
import numpy as np

# --- 1. Correction de sÃ©curitÃ© ---
test_predictions = np.nan_to_num(test_predictions, nan=0.0) 

# --- 2. CrÃ©er le DataFrame de soumission avec le nom de colonne correct ---
submission = pd.DataFrame({
    'user_hash': test['user_hash'],
    # CORRECTION CLÃ‰ : DOIT Ãªtre 'prediction'
    'prediction': test_predictions 
})

# --- 3. Sauvegarder le fichier ---
submission.to_csv('submission_lgbm_graphe_agressif.csv', index=False)

print("\nðŸš€ Soumission crÃ©Ã©e avec succÃ¨s : submission_lgbm_graphe_agressif.csv")
print("AperÃ§u de la soumission (VÃ©rifiez le nom de la colonne 'prediction') :")
print(submission.head())

