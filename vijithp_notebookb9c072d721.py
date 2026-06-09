!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl

import sys
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
tqdm.pandas()

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors

import warnings
import xgboost as xgb

rdBase.DisableLog('rdApp.warning')
warnings.filterwarnings('ignore')

train_df_original = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')


# Create separate files for each property
properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

for prop in properties:
    # Create a new DataFrame with SMILES and the current property
    prop_df = train_df_original[['SMILES', prop]].copy()
    
    # Remove rows where the property is null or empty
    prop_df.dropna(subset=[prop], inplace=True)
    
    # Define the output filename
    output_filename = f'train_{prop}.csv'
    
    # Save the new DataFrame to a csv file
    prop_df.to_csv(output_filename, index=False)
    
    print(f"Created {output_filename} with SMILES and {prop} data.")

import pandas as pd

# Load the datasets1
train_tc_df = pd.read_csv('/kaggle/working/train_Tc.csv')
dataset1_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv')

# Standardize column names
dataset1_df.rename(columns={'TC_mean': 'Tc'}, inplace=True)

# Concatenate the dataframes
merged_tc_df = pd.concat([train_tc_df, dataset1_df], ignore_index=True)

# Save the merged dataframe back to 'train_Tc.csv', overwriting the original file
merged_tc_df.to_csv('train_Tc.csv', index=False)

print("old 'train_Tc.csv':", len(train_tc_df))
print("Total rows in the updated 'train_Tc.csv':", len(merged_tc_df))

# Load the datasets3
train_tg_df = pd.read_csv('/kaggle/working/train_Tg.csv')
dataset3_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv')


# Concatenate the dataframes
merged_tg_df = pd.concat([train_tg_df, dataset3_df], ignore_index=True)

# Save the merged dataframe back to 'train_Tc.csv', overwriting the original file
merged_tg_df.to_csv('train_Tg.csv', index=False)

print("old 'train_Tg.csv':", len(train_tg_df))
print("Total rows in the updated 'train_Tg.csv':", len(merged_tg_df))


# Load the datasets4
train_ffv_df = pd.read_csv('/kaggle/working/train_FFV.csv')
dataset4_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv')


# Concatenate the dataframes
merged_ffv_df = pd.concat([train_ffv_df, dataset4_df], ignore_index=True)

# Save the merged dataframe back to 'train_Tc.csv', overwriting the original file
merged_ffv_df.to_csv('train_FFV.csv', index=False)

print("old 'train_FFV.csv':", len(train_ffv_df))
print("Total rows in the updated 'train_FFV.csv':", len(merged_ffv_df))


TARGET_VARIABLES = ["Tg", "FFV", "Tc", "Density", "Rg"]
RANDOM_STATE = 42
SUBMISSION_FILE = 'submission.csv'

# --- NEW: Define specific XGBoost parameters for each target variable ---
# 您可以根据每个性质的需求，在这里精细调整参数
TARGET_SPECIFIC_XGB_PARAMS = {
    "Tg": {
        'n_estimators': 600, 'learning_rate': 0.012, 'max_depth': 7, 
        'subsample': 0.816, 'colsample_bytree': 0.62, 'random_state': RANDOM_STATE,
        'reg_alpha': 8.617e-05, 'reg_lambda': 8.597e-05,
    },
    "FFV": {
        'n_estimators': 2200, 'learning_rate': 0.013, 'max_depth': 8, 
        'subsample': 0.6582, 'colsample_bytree': 0.8195, 'random_state': RANDOM_STATE,
        'reg_alpha': 0.009747, 'reg_lambda': 0.88,
    },
    "Tc": {
        'n_estimators': 1800, 'learning_rate': 0.06235, 'max_depth': 11, 
        'subsample': 0.68, 'colsample_bytree': 0.8, 'random_state': RANDOM_STATE,
        'reg_alpha': 5.0779, 'reg_lambda': 0.745,
    },
    "Density": {
        'n_estimators': 2100, 'learning_rate': 0.03375, 'max_depth': 4, 
        'subsample': 0.8018, 'colsample_bytree': 0.962, 'random_state': RANDOM_STATE,
        'reg_alpha': 5.4985, 'reg_lambda': 0.5,
    },
    "Rg": {
       'n_estimators': 2200, 'learning_rate': 0.013, 'max_depth': 4, 
        'subsample': 0.674, 'colsample_bytree': 0.76, 'random_state': RANDOM_STATE,
        'reg_alpha': 0.00953, 'reg_lambda': 5.421e-07,
    }
}

# --- feature engineering function (unchanged) ---
def generate_rdkit_features(smiles_str: str):
    """
    Generates RDKit descriptors and Morgan fingerprints from SMILES strings.
    """
    mol = Chem.MolFromSmiles(smiles_str)
    
    desc_list = [d[0] for d in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)
    
    morgan_fp_size = 1024
    
    if mol is None:
        return np.full(len(desc_list) + morgan_fp_size, np.nan)
    
    descriptors = np.array(calculator.CalcDescriptors(mol))
    mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=morgan_fp_size)
    mfp_array = np.array(list(mfp.ToBitString())).astype(int)
    
    return np.concatenate([descriptors, mfp_array])

# --- Main Program ---

# a. Load test data and generate features once
print("Loading test data...")
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

print("Generating RDKit features for the test set...")
desc_list_names = [d[0] for d in Descriptors._descList]
fp_morgan_cols = [f'mfp_{i}' for i in range(1024)]
feature_columns = desc_list_names + fp_morgan_cols

X_test_list = [generate_rdkit_features(s) for s in tqdm(test_df['SMILES'], desc="Processing test set")]
X_test = pd.DataFrame(np.vstack(X_test_list), columns=feature_columns)

# b. Train model and predict for each target
submission_df = pd.DataFrame({'id': test_df['id']})

for target in TARGET_VARIABLES:
    print(f"--- Training XGBoost model for '{target}' ---")
    
    # 1. Load the specific training data for the target
    train_target_df = pd.read_csv(f'/kaggle/working/train_{target}.csv')

    # 2. Generate features for the current training set
    print(f"Generating features for {target} training data...")
    X_train_list = [generate_rdkit_features(s) for s in tqdm(train_target_df['SMILES'], desc=f"Processing {target} train")]
    X_train = pd.DataFrame(np.vstack(X_train_list), columns=feature_columns)
    
    y_train = train_target_df[target]

    # 3. Data preprocessing
    print("Preprocessing data...")
    f32_max = np.finfo(np.float32).max
    
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    for df in [X_train_processed, X_test_processed]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df[df > f32_max] = np.nan
        df[df < -f32_max] = np.nan
    
    impute_values = X_train_processed.mean()
    X_train_processed.fillna(impute_values, inplace=True)
    X_test_processed.fillna(impute_values, inplace=True)

    # 4. Initialize and train the model using target-specific parameters
    print(f"Using specific parameters for '{target}'...")
    # --- MODIFIED LINE ---
    # Retrieve the parameters for the current target
    current_params = TARGET_SPECIFIC_XGB_PARAMS[target]
    model = xgb.XGBRegressor(**current_params)
    
    model.fit(X_train_processed, y_train)

    # 5. Make predictions
    predictions = model.predict(X_test_processed)
    submission_df[target] = predictions
    
    print(f"'{target}' prediction completed.")

# c. Save submission
submission_df.to_csv('submission.csv', index=False)
print("\nSubmission file created successfully!")
print(submission_df.head())



submission




