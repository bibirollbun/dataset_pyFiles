# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install --no-index --no-deps /kaggle/input/rdkit2025new/wheelhouse/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import pandas as pd
import numpy as np
import pickle
import random
import os

# RDKit libraries
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors

# Machine Learning libraries
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer # Used for imputing inf/missing values

# Hyperparameter tuning
import optuna

# Progress bar library
from tqdm.auto import tqdm

# For display in Jupyter Notebooks
from IPython.display import display

# --- Configuration Section ---
CONFIG = {
    'train_csv_path': '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv',
    'test_csv_path': '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv',
    'output_dir': '/kaggle/working/', # Output directory
    'id_col': 'id',
    'smiles_col': 'SMILES',
    'all_potential_targets': ['Tg', 'FFV', 'Tc', 'Density','Rg'],
    'error_indices': [576, 4783, 4836, 7129], # Indices with known issues to be dropped
    'test_size': 0.3,
    'random_state': 42,
    'n_trials': 100, # Number of trials for Optuna optimization
    'base_drop_cols': ['id', 'SMILES', 'smi_processed', 'mol'],
}

# --- Function Definitions ---

def preprocess_smiles_data(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """Preprocesses SMILES strings."""
    processed_smiles, mols = [], []
    for i, smi in enumerate(df[smiles_col]):
        try:
            # Replace wildcard '*' with a Carbon atom to create a valid molecule
            mol = Chem.MolFromSmiles(smi.replace('*', 'C'))
            if mol is None: raise ValueError("Invalid SMILES")
            processed_smiles.append(Chem.MolToSmiles(mol))
            mols.append(mol)
        except Exception as e:
            original_index = df.index[i]
            mol_id = df.loc[original_index, CONFIG['id_col']]
            processed_smiles.append(np.nan)
            mols.append(np.nan)

    df_processed = df.copy()
    df_processed['smi_processed'] = processed_smiles
    df_processed['mol'] = mols
    df_processed.dropna(subset=['smi_processed', 'mol'], inplace=True)
    return df_processed.reset_index(drop=True)


def calculate_descriptors(df: pd.DataFrame, id_col: str, smiles_col: str) -> pd.DataFrame:
    """
    Calculates 2D descriptors and handles infinite and extremely large finite values.
    """
    tqdm.pandas()
    # 2D Descriptors
    print("  Calculating 2D descriptors...")
    desc_names = [desc[0] for desc in Descriptors.descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    def calc_2d_safe(mol):
        if pd.isna(mol): return [np.nan] * len(desc_names)
        try: return calc.CalcDescriptors(mol)
        except Exception: return [np.nan] * len(desc_names)
    desc_2d_list = df['mol'].progress_apply(calc_2d_safe)
    df_desc_2d = pd.DataFrame(desc_2d_list.tolist(), columns=desc_names, index=df.index)

    # Convert existing inf -> nan
    df_desc_2d.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Get the max/min values representable by float32
    f32_max = np.finfo(np.float32).max
    f32_min = np.finfo(np.float32).min

    # Also convert "extremely large finite values" exceeding float32 range to NaN.
    # This allows them to be handled by the subsequent imputer.
    print("  Cleaning up extreme values from descriptors...")
    for col in df_desc_2d.select_dtypes(include=np.number).columns:
        # Using .where for a more memory-efficient operation
        df_desc_2d[col] = df_desc_2d[col].where((df_desc_2d[col] <= f32_max) & (df_desc_2d[col] >= f32_min), np.nan)

    df_final = pd.concat([df, df_desc_2d], axis=1)

    return df_final.reset_index(drop=True)


def train_model(features: pd.DataFrame, target: pd.Series, target_col: str):
    """
    Trains, evaluates, and saves an XGBoost model.
    Handles missing values internally.
    
    Returns:
        tuple: (trained_model, standard_scaler, imputer)
    """
    # 1. Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=CONFIG['test_size'], random_state=CONFIG['random_state']
    )

    # 2. Replace infinite values with NaN (as a safeguard, though handled in calculate_descriptors)
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 3. Fit and apply the Imputer
    #    Create a rule to impute with the median from the training data (X_train)
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    #    Apply the same rule to the test data
    X_test_imputed = imputer.transform(X_test)
    
    # 4. Fit and apply the Scaler
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_imputed)
    X_test_std = scaler.transform(X_test_imputed)

    # 5. Hyperparameter tuning and model training (Optuna)
    def objective(trial):
        params = {
            'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'booster': 'gbtree',
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        model = XGBRegressor(n_estimators=1000, **params, random_state=CONFIG['random_state'], early_stopping_rounds=50)
        model.fit(X_train_std, y_train, eval_set=[(X_test_std, y_test)], verbose=False)
        return r2_score(y_test, model.predict(X_test_std))

    print("  Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=CONFIG['n_trials'])
    
    # 6. Retrain the model with the best parameters
    best_model = XGBRegressor(n_estimators=2000, **study.best_params, random_state=CONFIG['random_state'])
    best_model.fit(X_train_std, y_train)
    
    y_pred = best_model.predict(X_test_std)
    print(f"  Best R2: {study.best_value:.4f}, Final Test Set RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

    # 7. Save the model and preprocessors
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    with open(f"{CONFIG['output_dir']}model_{target_col}.pkl", 'wb') as f: pickle.dump(best_model, f)
    with open(f"{CONFIG['output_dir']}scaler_{target_col}.pkl", 'wb') as f: pickle.dump(scaler, f)
    with open(f"{CONFIG['output_dir']}imputer_{target_col}.pkl", 'wb') as f: pickle.dump(imputer, f)
    
    return best_model, scaler, imputer

def predict_properties(df_features: pd.DataFrame, model, scaler, imputer) -> np.ndarray:
    """
    Predicts properties using a trained model.
    Handles missing values internally.
    """
    # 1. Replace infinite values with NaN (safeguard)
    features_cleaned = df_features.replace([np.inf, -np.inf], np.nan)
    
    # 2. Impute missing values with the pre-fitted imputer
    features_imputed = imputer.transform(features_cleaned)
    
    # 3. Standardize features with the pre-fitted scaler
    features_std = scaler.transform(features_imputed)
    
    # 4. Predict
    return model.predict(features_std)

# --- Main Processing ---
if __name__ == '__main__':
    # Pre-process test data and calculate descriptors once to be efficient
    print("--- Pre-processing Test Data (occurs once) ---")
    test_df_original = pd.read_csv(CONFIG['test_csv_path'])
    test_df_processed = preprocess_smiles_data(test_df_original, CONFIG['smiles_col'])
    test_df_with_features = calculate_descriptors(test_df_processed, CONFIG['id_col'], CONFIG['smiles_col'])
    
    final_predictions_df = test_df_with_features.copy()

    # Loop through each target property
    for target_col in CONFIG['all_potential_targets']:
        print(f"\n{'='*20} Processing Target: {target_col} {'='*20}")

        # 1. Prepare Training Data
        train_df = pd.read_csv(CONFIG['train_csv_path'])
        train_df.dropna(subset=[target_col], inplace=True)
        if len(train_df) == 0:
            print(f"No training data available for target '{target_col}'. Skipping.")
            continue
        train_df.reset_index(drop=True, inplace=True)
        train_df.drop(index=train_df[train_df.index.isin(CONFIG['error_indices'])].index, inplace=True, errors='ignore')
        
        train_df_processed = preprocess_smiles_data(train_df, CONFIG['smiles_col'])
        print(f"Calculating descriptors for {len(train_df_processed)} training molecules...")
        train_df_features = calculate_descriptors(train_df_processed, CONFIG['id_col'], CONFIG['smiles_col'])
        
        target = train_df_features[target_col]
        cols_to_drop = CONFIG['base_drop_cols'] + CONFIG['all_potential_targets']
        features = train_df_features.drop(columns=cols_to_drop, errors='ignore')
        
        # 2. Model Training
        model, scaler, imputer = train_model(features, target, target_col)

        # 3. Prediction on Test Data
        # Align test data columns with the features used for training
        test_features_aligned = test_df_with_features[features.columns]
        predictions = predict_properties(test_features_aligned, model, scaler, imputer)
        
        final_predictions_df[target_col] = predictions
        print(f"Predictions for {target_col} have been added.")

    # 4. Save all predictions to a single file
    print("\n--- All prediction tasks are complete. Saving final results. ---")
    output_path = f"{CONFIG['output_dir']}submission.csv"

    output_cols = [CONFIG['id_col']] + [t for t in CONFIG['all_potential_targets']]   
    output_cols_existing = [col for col in output_cols if col in final_predictions_df.columns]
    final_predictions_df[output_cols_existing].to_csv(output_path, index=False)
    
    print(f"\nFinal predictions saved to {output_path}")
    print("\n--- Script Finished ---")
    display(final_predictions_df[output_cols_existing].head())

