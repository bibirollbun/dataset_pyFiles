import pandas as pd
import numpy as np

# Le chemin d'accÃ¨s au fichier est ajustÃ© au contexte de l'interprÃ©teur
INPUT_FILE_PATH = '/kaggle/input/airr-submission/submission (1).csv'

try:
    df_submission = pd.read_csv(INPUT_FILE_PATH)
    
    print("--- ğŸ“‹ INSPECTION DE LA COLONNE ID ---")
    
    # 1. Identifier les valeurs NaN (celles qui causent la duplication des IDs d'attribution)
    nan_count = df_submission['ID'].isna().sum()
    
    # 2. Compter le nombre total de lignes dupliquÃ©es (inclut les 400 000 NaN)
    # L'argument keep=False marque toutes les occurrences d'une valeur dupliquÃ©e (y compris la premiÃ¨re)
    total_duplicate_rows = df_submission['ID'].duplicated(keep=False).sum()

    print(f"Total des lignes : {len(df_submission)}")
    print(f"Nombre de valeurs manquantes (NaN) dans 'ID' : {nan_count}")
    print(f"Nombre de lignes avec un ID non-unique (inclut les NaN et les IDs valides si dupliquÃ©s) : {total_duplicate_rows}")

    # 3. Afficher les IDs les plus frÃ©quents (pour voir le 'NaN')
    print("\n--- ğŸ“Š 5 IDs les plus frÃ©quents ---")
    
    # Utilisation de dropna=False pour inclure les NaN dans le dÃ©compte (s'ils existent)
    id_counts = df_submission['ID'].value_counts(dropna=False)
    
    # Remplacer temporairement 'NaN' par une chaÃ®ne lisible pour l'affichage
    top_ids = id_counts.head(5).rename(index={np.nan: 'NaN (Placeholder d\'Attribution)'})
    print(top_ids.to_markdown())

except FileNotFoundError:
    print(f"â�Œ Erreur: Le fichier {INPUT_FILE_PATH} n'a pas Ã©tÃ© trouvÃ©.")
except Exception as e:
    print(f"â�Œ Une erreur inattendue est survenue: {e}")


import pandas as pd
import os

# Chemin d'accÃ¨s au fichier Ã  corriger
INPUT_FILE_PATH = '/kaggle/input/airr-submission/submission (1).csv'
# Chemin oÃ¹ sauvegarder le fichier corrigÃ© (dans /kaggle/working/ pour la soumission)
OUTPUT_FILE_PATH = '/kaggle/working/submission_corrected.csv'

# Colonne Ã  supprimer
COLUMN_TO_DROP = 'ID.1'

try:
    # Lire le fichier (avec le DtypeWarning ignorÃ©)
    df_submission = pd.read_csv(INPUT_FILE_PATH, low_memory=False)
    
    if COLUMN_TO_DROP in df_submission.columns:
        
        # 1. Supprimer la colonne ID.1
        df_submission_cleaned = df_submission.drop(columns=[COLUMN_TO_DROP])
        
        # --- ğŸ”� INSPECTION DES VALEURS NULLES AJOUTÃ‰E ---
        print("\n--- ğŸ”� DÃ‰COMPTE DES VALEURS NULLES (NaN) APRÃˆS SUPPRESSION DE 'ID.1' ---")
        print("Ceci confirme les colonnes Ã  nettoyer pour l'Ã©tape suivante.")
        print(df_submission_cleaned.isnull().sum())
        # -----------------------------------------------------------------------
        
        # 2. Sauvegarder le fichier corrigÃ©
        df_submission_cleaned.to_csv(OUTPUT_FILE_PATH, index=False)
        
        print(f"\nâœ… Colonne '{COLUMN_TO_DROP}' supprimÃ©e avec succÃ¨s.")
        print(f"Nouvelle forme: {df_submission_cleaned.shape}")
        print(f"Fichier corrigÃ© sauvegardÃ© sous: {OUTPUT_FILE_PATH}")
        print("\n[HEAD du fichier corrigÃ© - CONFIRMATION UNE SEULE COLONNE ID]:")
        print(df_submission_cleaned.head(2).to_markdown(index=False))

    else:
        print(f"âœ… La colonne '{COLUMN_TO_DROP}' n'a pas Ã©tÃ© trouvÃ©e. Le fichier est dÃ©jÃ  prÃªt.")

except FileNotFoundError:
    print(f"â�Œ Erreur: Le fichier {INPUT_FILE_PATH} n'a pas Ã©tÃ© trouvÃ©. VÃ©rifiez le chemin d'accÃ¨s.")
except Exception as e:
    print(f"â�Œ Une erreur inattendue est survenue: {e}")


# --------------------------------------------------------------------------------------
# ğŸš€ CELLULE UNIQUE V7.2 : CORRECTION DU CHEMIN ET INJECTION FINALE (FIXED CSV DELIMITER)
# --------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os

# --- CHEMINS D'ACCÃˆS ---
SAMPLE_FILE_PATH = '/kaggle/input/adaptive-immune-profiling-challenge-2025/sample_submissions.csv' 
PREDICTIONS_FILE_PATH = '/kaggle/working/submission_corrected.csv'
ATTRIBUTION_FILENAME = '/kaggle/input/airr-df-attributions/df_attributions.csv'
OUTPUT_FILE_PATH = '/kaggle/working/submission_final_ready_v7.csv'

MISSING_SUBMISSION_VALUE = -999.0
START_INJECTION_INDEX = 4213 
COLUMN_TO_DROP = 'ID.1' 

try:
    # 1. Charger le fichier SAMPLE comme structure de base
    df_final = pd.read_csv(SAMPLE_FILE_PATH, low_memory=False)
    
    # 2. Charger les PRÃ‰DICTIONS du participant
    df_predictions = pd.read_csv(PREDICTIONS_FILE_PATH, low_memory=False)

    if 'label_positive_probability' not in df_predictions.columns:
         raise ValueError("Le fichier de prÃ©dictions du participant n'a pas la colonne 'label_positive_probability'.")

    # 3. INJECTION des PRÃ‰DICTIONS (lignes 0 Ã  4212)
    df_final.loc[:START_INJECTION_INDEX-1, 'label_positive_probability'] = df_predictions.loc[:START_INJECTION_INDEX-1, 'label_positive_probability'].values
    print("âœ… PrÃ©dictions du participant injectÃ©es (lignes 1 Ã  4213).")
    
    # 4. Nettoyage de la colonne ID.1
    if COLUMN_TO_DROP in df_final.columns:
        df_final = df_final.drop(columns=[COLUMN_TO_DROP])
        print(f"âœ… Colonne '{COLUMN_TO_DROP}' supprimÃ©e avec succÃ¨s.")

    # 5. Charger le contenu d'Attribution
    # ğŸ’¥ CORRECTION DE L'ERREUR ICI : Retrait de delim_whitespace=True ğŸ’¥
    df_attributions = pd.read_csv(ATTRIBUTION_FILENAME).fillna("")
    
    required_len = len(df_final) - START_INJECTION_INDEX
    
    if required_len > 0:
        # SÃ©lectionner les colonnes de contenu Ã  injecter
        # Cette ligne fonctionnera maintenant
        df_content_to_inject = df_attributions[['junction_aa', 'v_call', 'j_call']].head(required_len)
        
        # S'assurer que les longueurs correspondent (omissions pour la concision)

        # 6. REMPLACEMENT DES SÃ‰QUENCES UNIQUEMENT
        df_final.loc[START_INJECTION_INDEX:, 'junction_aa'] = df_content_to_inject['junction_aa'].values
        df_final.loc[START_INJECTION_INDEX:, 'v_call'] = df_content_to_inject['v_call'].values
        df_final.loc[START_INJECTION_INDEX:, 'j_call'] = df_content_to_inject['j_call'].values
        
        # S'assurer que la probabilitÃ© reste -999.0 dans la section Attribution
        df_final.loc[START_INJECTION_INDEX:, 'label_positive_probability'] = float(MISSING_SUBMISSION_VALUE)

    # 7. Finalisation et Sauvegarde
    expected_cols = ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
    df_final = df_final[[c for c in expected_cols if c in df_final.columns]].copy()
    
    for col in ['ID', 'dataset', 'junction_aa', 'v_call', 'j_call']:
        df_final[col] = df_final[col].fillna("").astype(str)
            
    df_final['label_positive_probability'] = pd.to_numeric(df_final['label_positive_probability'], errors='coerce').fillna(MISSING_SUBMISSION_VALUE).astype(float)
    
    df_final.to_csv(OUTPUT_FILE_PATH, index=False)

    # 8. Inspection et vÃ©rification des IDs
    print("\n--- ğŸ”� INSPECTION FINALE DU DATAFRAME CORRIGÃ‰ V7.2 ---")
    
    if df_final['ID'].nunique() == len(df_final):
        print("âœ… VÃ‰RIFICATION ID : La colonne 'ID' est entiÃ¨rement unique. C'est parfait pour la soumission.")
    else:
        num_duplicates = len(df_final) - df_final['ID'].nunique()
        print(f"â�Œ VÃ‰RIFICATION ID : ATTENTION ! {num_duplicates} doublons trouvÃ©s dans la colonne 'ID'. La soumission risque d'Ã©chouer.")

    print("\n[HEAD - 5 premiÃ¨res lignes (Classification - Vos PrÃ©dictions)]")
    print(df_final[['ID', 'dataset', 'label_positive_probability', 'junction_aa']].head().to_markdown(index=False))

    print("\n[TAIL - 5 derniÃ¨res lignes (Attribution - NOUVELLES SÃ©quences)]")
    print(df_final[['ID', 'dataset', 'label_positive_probability', 'junction_aa']].tail().to_markdown(index=False))

    print(f"\nğŸ�‰ Fichier INTERMÃ‰DIAIRE V7.2 PRÃŠT Ã€ ÃŠTRE TRAITÃ‰ PAR V8 : {OUTPUT_FILE_PATH}")

except FileNotFoundError as e:
    print(f"â�Œ Erreur: L'un des fichiers n'a pas Ã©tÃ© trouvÃ©. VÃ©rifiez le chemin : {e}")
except Exception as e:
    print(f"â�Œ Une erreur inattendue est survenue: {e}")


import pandas as pd
import numpy as np
import os

# Fichier intermÃ©diaire (gÃ©nÃ©rÃ© par le script V7)
INPUT_FILE_PATH = '/kaggle/working/submission_final_ready_v7.csv'
# Fichier FINAL prÃªt Ã  soumettre
OUTPUT_FILE_PATH = '/kaggle/working/submission_final_ready_v8.csv'

# Colonnes qui doivent Ãªtre des chaÃ®nes
STRING_COLS_FOR_RECAST = ['ID', 'dataset', 'junction_aa', 'v_call', 'j_call']
FLOAT_COLS = ['label_positive_probability']

try:
    df_final = pd.read_csv(INPUT_FILE_PATH, low_memory=False)

    print(f"Fichier intermÃ©diaire chargÃ©. Forme: {df_final.shape}")
    print("\n--- DÃ‰COMPTE DES NaN AVANT CORRECTION ---")
    print(df_final.isnull().sum())
    
    # 1. Remplir les NaN de toutes les colonnes STRING par une chaÃ®ne vide ""
    for col in STRING_COLS_FOR_RECAST:
        if col in df_final.columns:
            # S'assurer que tous les NaN sont des chaÃ®nes vides avant le type casting final
            df_final[col] = df_final[col].fillna("").astype(str)
            
    # 2. Remplir les NaN des colonnes FLOAT par -999.0 float
    for col in FLOAT_COLS:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(float(MISSING_SUBMISSION_VALUE)).astype(float)


    # 3. Sauvegarde du fichier corrigÃ©
    df_final.to_csv(OUTPUT_FILE_PATH, index=False)
    
    # 4. INSPECTION FINALE DU FICHIER EXPORTÃ‰ (vÃ©rification ultime du format)
    df_check_exported = pd.read_csv(OUTPUT_FILE_PATH, low_memory=False)
    
    print("\n--- ğŸ”� INSPECTION FINALE DU FICHIER EXPORTÃ‰ V8 (Post-Export) ---")
    
    # DÃ©compte des NaN sur le fichier relu
    nan_count_exported = df_check_exported.isnull().sum().sum()
    if nan_count_exported == 0:
        print(f"âœ… Aucune valeur nulle (NaN) trouvÃ©e dans le fichier exportÃ©.")
    else:
        print(f"â�Œ Erreur critique : {nan_count_exported} NaN sont encore prÃ©sents aprÃ¨s export.")
        print(df_check_exported.isnull().sum())
        
    # VÃ©rification des types
    print("\n[VÃ©rification des 5 premiÃ¨res lignes et des types de colonnes (doit Ãªtre clean)]:")
    print(df_check_exported.head(5).to_markdown(index=False))
    
    # VÃ©rification des types des colonnes textuelles (pour s'assurer que "" n'est pas devenu NaN Ã  l'export)
    print("\n[VÃ©rification des Types des colonnes ClÃ©s (doit Ãªtre float pour la probabilitÃ© et object/string pour le reste)]:")
    print(df_check_exported.dtypes.to_markdown())

    print(f"\nğŸ�‰ Fichier FINAL V8 PRÃŠT Ã€ ÃŠTRE SOUMIS : {OUTPUT_FILE_PATH}")

except FileNotFoundError:
    print(f"â�Œ Erreur: Le fichier {INPUT_FILE_PATH} n'a pas Ã©tÃ© trouvÃ©.")
except Exception as e:
    print(f"â�Œ Une erreur inattendue est survenue: {e}")


import pandas as pd
import os

# Chemin d'accÃ¨s au fichier sample_submissions.csv
INPUT_FILE_PATH = '/kaggle/input/adaptive-immune-profiling-challenge-2025/sample_submissions.csv'
# Index de dÃ©part pour l'inspection (ligne 4214 = index 4213)
START_INSPECTION_INDEX = 4213 
# Nombre de lignes Ã  afficher
ROWS_TO_DISPLAY = 15 

try:
    # Charger le fichier (low_memory=False pour gÃ©rer les types mixtes)
    # L'en-tÃªte est souvent dupliquÃ© ou incorrectement lu dans ce type de fichier, 
    # d'oÃ¹ la nÃ©cessitÃ© de 'low_memory=False'.
    df_sample = pd.read_csv(INPUT_FILE_PATH, low_memory=False)
    
    # Isoler les colonnes d'intÃ©rÃªt
    cols_of_interest = ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
    
    # Isoler les lignes de la section Attribution (Ã  partir de l'index 4213)
    df_attribution_section = df_sample[cols_of_interest].iloc[START_INSPECTION_INDEX:]
    
    print(f"\n--- ğŸ”� INSPECTION DE LA SECTION ATTRIBUTION ({ROWS_TO_DISPLAY} LIGNES Ã  partir de la ligne 4214) ---")
    
    # Afficher le rÃ©sultat
    print(df_attribution_section.head(ROWS_TO_DISPLAY).to_markdown(index=True))
    
    print("\n--- RAPPEL DES VALEURS UNIQUES DANS LA SECTION JONCTION ---")
    # Confirmer ce qui est dans les colonnes des sÃ©quences
    print(f"Valeurs uniques dans 'junction_aa' (section Attribution) : {df_attribution_section['junction_aa'].unique()}")
    print(f"Valeurs uniques dans 'v_call' (section Attribution) : {df_attribution_section['v_call'].unique()}")
    
except FileNotFoundError:
    print(f"â�Œ Erreur: Le fichier {INPUT_FILE_PATH} n'a pas Ã©tÃ© trouvÃ©. VÃ©rifiez le chemin d'accÃ¨s.")
except Exception as e:
    print(f"â�Œ Une erreur inattendue est survenue: {e}")

