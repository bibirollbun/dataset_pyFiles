!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
warnings.filterwarnings('ignore')

# Weighted MAE
def compute_wMAE(y_true_dict, y_pred_dict, test_ranges):
    errors = []
    sqrt_inv_ni = [1 / np.sqrt(len(y_true_dict[prop])) for prop in y_true_dict]
    normalization = sum(sqrt_inv_ni)
    K = len(y_true_dict)
    
    for i, prop in enumerate(y_true_dict):
        ni = len(y_true_dict[prop])
        ri = test_ranges[prop][1] - test_ranges[prop][0]
        if ri == 0:
            ri = 1.0
        wi = (1 / ri) * (K * (1 / np.sqrt(ni)) / normalization)
        mae = mean_absolute_error(y_true_dict[prop], y_pred_dict[prop])
        errors.append(wi * mae)
        print(f"{prop}: MAE={mae:.6f}, Weight={wi:.6f}, Weighted_MAE={wi*mae:.6f}")
    return np.mean(errors)

# Compute RDKit Descriptors
def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(best_features)
    desc_values = []
    for desc_name in best_features:
        if desc_name in [desc[0] for desc in Descriptors.descList]:
            desc_func = dict(Descriptors.descList)[desc_name]
            desc_values.append(desc_func(mol))
        else:
            # Handle fragment-based descriptors (fr_*) using getattr
            try:
                desc_values.append(getattr(Chem.Fragments, desc_name)(mol))
            except AttributeError:
                desc_values.append(0)  # Default to 0 if descriptor not found
    return desc_values

# Pre-selected features
best_features = [
    'AUTOCORR2D_1', 'AUTOCORR2D_2', 'AUTOCORR2D_3', 'AUTOCORR2D_4', 'AUTOCORR2D_5',
    'AUTOCORR2D_6', 'AUTOCORR2D_7', 'AUTOCORR2D_8', 'AUTOCORR2D_9', 'AUTOCORR2D_10',
    'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI',
    'BCUT2D_MRLOW', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n',
    'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
    'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4',
    'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'ExactMolWt',
    'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha',
    'HeavyAtomCount', 'HeavyAtomMolWt', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA',
    'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge',
    'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge', 'MolLogP',
    'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
    'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings',
    'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds',
    'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons',
    'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14',
    'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8',
    'PEOE_VSA9', 'RingCount', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4',
    'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10',
    'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5',
    'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10',
    'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7',
    'VSA_EState8', 'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO',
    'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S',
    'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1',
    'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate',
    'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
    'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
    'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen',
    'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
    'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile',
    'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime',
    'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester',
    'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
    'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole',
    'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed'
]

# Load Data
print("Loading data...")
train = pd.read_csv('/kaggle/input/super-data/super_data.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
print(f"Training data shape: {train.shape}")
print(f"Test data shape: {test.shape}")

# Feature Engineering
print("Computing descriptors...")
train_descriptors = [compute_all_descriptors(smi) for smi in train['SMILES'].to_list()]
train_features_df = pd.DataFrame(train_descriptors, columns=best_features)
test_descriptors = [compute_all_descriptors(smi) for smi in test['SMILES'].to_list()]
test_features_df = pd.DataFrame(test_descriptors, columns=best_features)

# Feature Cleaning
print("Cleaning features...")
train_features_df = train_features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
test_features_df = test_features_df.replace([np.inf, -np.inf], np.nan).fillna(0)

feats = best_features
print(f"Using {len(feats)} pre-selected features")

# LightGBM K-Fold Training
def lgb_kfold(train_df, test_df, target, feats, folds):
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': 0.03,
        'num_leaves': 127,
        'min_data_in_leaf': 10,
        'max_depth': -1,
        'max_bin': 256,
        'boosting': 'gbdt',
        'feature_fraction': 0.8,
        'bagging_freq': 1,
        'bagging_fraction': 0.9,
        'bagging_seed': 42,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbosity': -1,
        'num_boost_round': 5000,
        'device_type': 'cpu'
    }
    
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    df_importances = pd.DataFrame()
    valid_maes = []
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df)):
        print(f'n_fold: {n_fold}')
        
        train_x = train_df[feats].iloc[train_idx].values
        train_y = train_df[target].iloc[train_idx].values
        valid_x = train_df[feats].iloc[valid_idx].values
        valid_y = train_df[target].iloc[valid_idx].values
        test_x = test_df[feats].values
        
        print(f'train_x: {train_x.shape}, valid_x: {valid_x.shape}, test_x: {test_x.shape}')
        
        dtrain = lgb.Dataset(train_x, label=train_y)
        dval = lgb.Dataset(valid_x, label=valid_y, reference=dtrain)
        callbacks = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(200)
        ]
        
        bst = lgb.train(params, dtrain, valid_sets=[dval, dtrain], callbacks=callbacks)
        
        # Feature Importance
        feature_importances = sorted(zip(feats, bst.feature_importance('gain')), key=lambda x: x[1], reverse=True)
        for f in feature_importances[:30]:
            print(f)
        
        df_importance = pd.DataFrame({
            'feature': [f[0] for f in feature_importances],
            'importance': [f[1] for f in feature_importances],
            'fold': n_fold
        })
        df_importances = pd.concat([df_importances, df_importance])
        
        # OOF Predictions and Validation MAE
        oof_preds[valid_idx] = bst.predict(valid_x, num_iteration=bst.best_iteration)
        valid_mae = mean_absolute_error(valid_y, oof_preds[valid_idx])
        valid_maes.append(valid_mae)
        print(f"Fold {n_fold} Validation MAE: {valid_mae:.6f}")
        
        # Test Predictions
        sub_preds += bst.predict(test_x, num_iteration=bst.best_iteration) / folds.n_splits
    
    print(f"Mean Validation MAE: {np.mean(valid_maes):.6f}")
    return oof_preds, sub_preds, df_importances

# Run for Each Target
n_splits = 5
seed = 42
folds = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
oof_dict = {}
sub_dict = {}
test_ranges = {}
y_true_dict = {}
y_pred_dict = {}

for t in targets:
    print(f"\nTraining for {t}")
    train_df = train[train[t].notnull()].copy()
    train_df = pd.concat([train_df[['id', t]], train_features_df.loc[train_df.index]], axis=1)
    test_df = test_features_df.copy()
    
    test_ranges[t] = (train_df[t].min(), train_df[t].max())
    oof_preds, sub_preds, df_importances = lgb_kfold(train_df, test_df, t, feats, folds)
    
    oof_dict[t] = oof_preds
    sub_dict[t] = sub_preds
    y_true_dict[t] = train_df[t].values
    y_pred_dict[t] = oof_preds
    
    test[t] = sub_preds

# Compute Validation wMAE
print("\nFINAL VALIDATION EVALUATION")
wmae_score = compute_wMAE(y_true_dict, y_pred_dict, test_ranges)
print(f"Validation Weighted MAE (wMAE): {wmae_score:.6f}")

# Submission
submission = test[['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]
submission.to_csv('submission.csv', index=False)
print(f"Submission saved! Shape: {submission.shape}")





# import pandas as pd
# import numpy as np
# import lightgbm as lgb
# from rdkit import Chem
# from rdkit.Chem import Descriptors
# import warnings
# import os

# # Ignore warnings for cleaner output
# warnings.filterwarnings('ignore')

# # --- CONFIGURATION ---

# # Directory where your saved models are located
# MODEL_DIR = '/kaggle/input/chemologist' 

# # Path to the test data file
# TEST_CSV_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv'

# # Path for the output submission file
# SUBMISSION_CSV_PATH = 'submission.csv'

# # Map target variables to their corresponding model files
# # IMPORTANT: Update these filenames to match the exact names of your saved models.
# MODEL_FILES = {
#     'Tg': 'lgb_Tg_best_fold_1.txt',
#     'FFV': 'lgb_FFV_best_fold_1.txt',
#     'Tc': 'lgb_Tc_best_fold_1.txt',
#     'Density': 'lgb_Density_best_fold_2.txt',
#     'Rg': 'lgb_Rg_best_fold_2.txt'
# }

# # --- FEATURE LIST ---

# # IMPORTANT! YOU MUST PASTE THE LIST OF FEATURES USED DURING TRAINING HERE.
# # The features were selected by VarianceThreshold in your training script.
# # To get this list, add `print(feats)` to your training script after the
# # line `feats = train_features_clean.columns.tolist()` and copy the output here.
# # The script will NOT work correctly without the exact same feature list.
# #
# # EXAMPLE:
# # SELECTED_FEATURES = ['MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', ... etc. ]

# SELECTED_FEATURES = ['MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'SPS', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAmideBonds', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumAtomStereoCenters', 'NumBridgeheadAtoms', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumHeterocycles', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumUnspecifiedAtomStereoCenters', 'Phi', 'RingCount', 'MolLogP', 'MolMR', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_aniline', 'fr_aryl_methyl', 'fr_azo', 'fr_benzene', 'fr_bicyclic', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_halogen', 'fr_hdrzine', 'fr_imidazole', 'fr_imide', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactone', 'fr_methoxy', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_oxazole', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_pyridine', 'fr_sulfide', 'fr_sulfone', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']


# # --- HELPER FUNCTION ---

# def compute_all_descriptors(smiles):
#     """
#     Computes all RDKit descriptors for a given SMILES string.
#     Returns a list of descriptor values.
#     """
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         # Get the number of descriptors to return a list of Nones of the correct length
#         num_descriptors = len(Descriptors.descList)
#         return [None] * num_descriptors
#     # Calculate all descriptors from the list
#     return [desc[1](mol) for desc in Descriptors.descList]


# # --- MAIN INFERENCE LOGIC ---

# def run_inference():
#     """
#     Main function to run the inference pipeline.
#     """
#     print("Starting inference process...")

#     # --- Step 0: Validate Inputs ---
#     if not SELECTED_FEATURES:
#         print("\nERROR: The 'SELECTED_FEATURES' list is empty.")
#         print("Please paste the list of feature names from your training script into the variable.")
#         return

#     if not os.path.exists(TEST_CSV_PATH):
#         print(f"\nERROR: Test file not found at '{TEST_CSV_PATH}'")
#         return

#     if not os.path.exists(MODEL_DIR):
#         print(f"\nERROR: Models directory not found at '{MODEL_DIR}'")
#         return

#     print(f"\n--- Step 1: Loading Test Data from '{TEST_CSV_PATH}' ---")
#     try:
#         test_df = pd.read_csv(TEST_CSV_PATH)
#         print(f"Test data loaded successfully. Shape: {test_df.shape}")
#     except Exception as e:
#         print(f"Failed to load test data. Error: {e}")
#         return
        
#     # Initialize submission dataframe with the ID
#     submission_df = test_df[['id']].copy()

#     print("\n--- Step 2: Feature Engineering (RDKit Descriptors) ---")
#     # Get the names of all descriptors from RDKit
#     desc_names = [desc[0] for desc in Descriptors.descList]
    
#     # Compute descriptors for each SMILES string in the test set
#     test_descriptors = [compute_all_descriptors(smi) for smi in test_df['SMILES'].to_list()]
    
#     # Create a DataFrame with the computed descriptors
#     test_features_df = pd.DataFrame(test_descriptors, columns=desc_names)
#     print(f"Generated {len(desc_names)} RDKit descriptors.")

#     # Clean features by replacing infinity with NaN
#     test_features_df = test_features_df.replace([np.inf, -np.inf], np.nan)
#     print("Replaced infinite values with NaN.")

#     print("\n--- Step 3: Feature Selection ---")
#     # Select only the features that were used for training
#     # Also, handle cases where a feature might be missing from the generated descriptors
#     final_test_features = test_features_df[SELECTED_FEATURES]
#     print(f"Selected {len(final_test_features.columns)} features for prediction.")

#     # --- Step 4: Prediction ---
#     for target, model_filename in MODEL_FILES.items():
#         print(f"\n--- Predicting for target: {target} ---")
#         model_path = os.path.join(MODEL_DIR, model_filename)

#         if not os.path.exists(model_path):
#             print(f"  WARNING: Model file not found at '{model_path}'. Skipping this target.")
#             submission_df[target] = 0 # Or np.nan, depending on requirements
#             continue

#         try:
#             # Load the pre-trained LightGBM model
#             bst = lgb.Booster(model_file=model_path)
#             print(f"  Model '{model_filename}' loaded successfully.")

#             # Make predictions on the prepared test features
#             predictions = bst.predict(final_test_features.values)
#             print(f"  Prediction complete.")
            
#             # Add predictions to our submission DataFrame
#             submission_df[target] = predictions

#         except Exception as e:
#             print(f"  An error occurred during prediction for target {target}. Error: {e}")
#             submission_df[target] = 0 # Or np.nan

#     # --- Step 5: Save Submission File ---
#     print(f"\n--- Saving final predictions to '{SUBMISSION_CSV_PATH}' ---")
#     try:
#         submission_df.to_csv(SUBMISSION_CSV_PATH, index=False)
#         print("Submission file saved successfully!")
#         print(f"Final submission shape: {submission_df.shape}")
#         print("\nInference process complete.")
#     except Exception as e:
#         print(f"Failed to save submission file. Error: {e}")


# if __name__ == '__main__':
#     run_inference()


# import pandas as pd
# import numpy as np
# from sklearn.feature_selection import VarianceThreshold
# from rdkit import Chem
# from rdkit.Chem import Descriptors

# print("--- Loading data to generate feature list ---")
# # Make sure this path is correct for your Kaggle environment
# train = pd.read_csv('/kaggle/input/super-data/super_data.csv')

# def compute_all_descriptors(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return [None] * len(desc_names)
#     return [desc[1](mol) for desc in Descriptors.descList]

# print("--- Computing RDKit descriptors ---")
# desc_names = [desc[0] for desc in Descriptors.descList]
# train_descriptors = [compute_all_descriptors(smi) for smi in train['SMILES'].to_list()]
# train_features_df = pd.DataFrame(train_descriptors, columns=desc_names)

# print("--- Cleaning and selecting features ---")
# train_features_df = train_features_df.replace([np.inf, -np.inf], np.nan)

# # Use the same threshold as in your training script
# selector = VarianceThreshold(threshold=0.01)
# selector.fit(train_features_df)

# # Get the feature names
# final_feature_names = selector.get_feature_names_out()

# print("\n\n--- COPY THE LIST BELOW ---")
# # The output is formatted to be pasted directly into the inference script
# print(list(final_feature_names))
# print(f"\nTotal features: {len(final_feature_names)}")


# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, rdMolDescriptors
# from rdkit.ML.Descriptors import MoleculeDescriptors
# from rdkit.Chem import Descriptors3D
# from rdkit.Chem import rdMolDescriptors
# from rdkit.Chem import QED

# # Paste your best_features list here
# best_features = [
#     'AUTOCORR2D_1', 'AUTOCORR2D_2', 'AUTOCORR2D_3', 'AUTOCORR2D_4', 'AUTOCORR2D_5',
#     'AUTOCORR2D_6', 'AUTOCORR2D_7', 'AUTOCORR2D_8', 'AUTOCORR2D_9', 'AUTOCORR2D_10',
#     'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI',
#     'BCUT2D_MRLOW', 'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n',
#     'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
#     'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4',
#     'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'ExactMolWt',
#     'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha',
#     'HeavyAtomCount', 'HeavyAtomMolWt', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA',
#     'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge',
#     'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge', 'MolLogP',
#     'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
#     'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings',
#     'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds',
#     'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons',
#     'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14',
#     'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8',
#     'PEOE_VSA9', 'RingCount', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4',
#     'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10',
#     'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5',
#     'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10',
#     'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7',
#     'VSA_EState8', 'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO',
#     'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S',
#     'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1',
#     'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate',
#     'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
#     'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
#     'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen',
#     'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
#     'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile',
#     'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime',
#     'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester',
#     'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
#     'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole',
#     'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed'
# ]  # full list from your earlier message

# # Load train + test
# train_df = pd.read_csv("/kaggle/input/super-data/super_data.csv")[["id", "SMILES"]]
# test_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")[["id", "SMILES"]]
# all_df = pd.concat([train_df, test_df], ignore_index=True)

# # Generate descriptors
# calculator = MoleculeDescriptors.MolecularDescriptorCalculator(best_features)

# def extract_features(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return [None] * len(best_features)
#     try:
#         return list(calculator.CalcDescriptors(mol))
#     except:
#         return [None] * len(best_features)

# # Apply extraction
# feature_matrix = all_df["SMILES"].apply(extract_features)
# features_df = pd.DataFrame(feature_matrix.tolist(), columns=best_features)
# features_df.insert(0, "id", all_df["id"])

# # Drop rows with invalid SMILES (i.e., all NaNs)
# features_df.dropna(inplace=True)

# # Save to file
# features_df.to_csv("rdkit_features.csv", index=False)
# print("✅ Saved rdkit_features.csv")


