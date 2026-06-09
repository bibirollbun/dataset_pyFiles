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


# --------------------------------------------------------------------------
# Step 0: Install RDKit from a local file (CRUCIAL FOR SUBMISSION)
# The Kaggle submission environment for this competition does not have RDKit
# pre-installed. The command below installs it from a local dataset.
#
#
# MAKE SURE YOU HAVE ADDED THE `rdkit-pypi` KAGGLE DATASET TO THIS NOTEBOOK.
# (Click "+ Add data" -> Search for "rdkit-pypi" -> Click "Add")
# --------------------------------------------------------------------------
!pip install --no-index --find-links=/kaggle/input/rdkit-pypi/ rdkit

# --------------------------------------------------------------------------
# Step 1: Import all required packages
# --------------------------------------------------------------------------
import pandas as pd
import numpy as np
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from tqdm.auto import tqdm
import os
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Step 2: Verify dependencies
# --------------------------------------------------------------------------
print("Verifying dependencies...")
try:
    import numpy, rdkit, lightgbm
    print(f"NumPy version: {numpy.__version__}")
    print(f"RDKit version: {rdkit.__version__}")
    print(f"LightGBM version: {lightgbm.__version__}")
except ImportError as e:
    print(f"Dependency missing: {e}")
    raise SystemExit("Please ensure all dependencies are installed.")

# --------------------------------------------------------------------------
# Step 3: Feature Engineering Function (Robust Version)
# --------------------------------------------------------------------------
def generate_features(smiles):
    """
    Generates a robust feature vector from a SMILES string by combining
    hand-picked descriptors and a Morgan fingerprint. This function is
    designed to be resilient to errors from unusual molecular structures.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
    except:
        return None

    # --- Feature Set 1: Hand-picked, reliable RDKit descriptors ---
    descriptor_values = []
    try:
        descriptor_values.extend([
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol)
        ])
    except:
        pass # Proceed even if some descriptors fail

    # --- Feature Set 2: Morgan Fingerprint ---
    fp_values = []
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fp_values = [float(x) for x in fp.ToBitString()]
    except:
        pass # Proceed even if fingerprint fails

    all_features = descriptor_values + fp_values
    if not all_features: return None
        
    return np.array(all_features, dtype=float)

# --------------------------------------------------------------------------
# Step 4: Data Loading and Processing with Progress Bars
# --------------------------------------------------------------------------
def load_and_process_data(train_file, test_file):
    print("Loading data...")
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"Ensure both {train_file} and {test_file} are present.")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    print(f"Original train shape: {train_df.shape}, Original test shape: {test_df.shape}")

    smiles_col = next((col for col in train_df.columns if col.lower() in ['smiles', 'smile']), None)
    if smiles_col is None: raise ValueError("No 'smiles' column found in train.csv.")
    
    # Generate features with a progress bar
    print("Generating features for training data...")
    train_features_list = [generate_features(s) for s in tqdm(train_df[smiles_col], desc="Train Features")]
    
    print("Generating features for test data...")
    test_features_list = [generate_features(s) for s in tqdm(test_df[smiles_col], desc="Test Features")]

    # Filter out molecules where feature generation failed
    train_valid_indices = [i for i, f in enumerate(train_features_list) if f is not None]
    test_valid_indices = [i for i, f in enumerate(test_features_list) if f is not None]

    if not train_valid_indices:
        raise ValueError("No valid molecules could be processed from the training data.")

    X_train = np.vstack([train_features_list[i] for i in train_valid_indices])
    X_test = np.vstack([test_features_list[i] for i in test_valid_indices]) if test_valid_indices else np.array([[]])
    
    # Align target labels with valid molecules
    y_train = train_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].iloc[train_valid_indices]
    
    # Keep track of which original test samples are valid
    test_df_filtered = test_df.iloc[test_valid_indices]
    
    print(f"Processed valid train samples: {X_train.shape[0]}")
    print(f"Processed valid test samples: {X_test.shape[0] if X_test.size > 0 else 0}")

    # Impute NaNs in features (if any) and scale
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    if X_test.size > 0: X_test = imputer.transform(X_test)
        
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if X_test.size > 0: X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, test_df, test_df_filtered

# --------------------------------------------------------------------------
# Step 5: Multi-Task Model Training and Prediction
# --------------------------------------------------------------------------
def train_and_predict(X_train, y_train, X_test):
    """
    Trains a multi-output LightGBM model using cross-validation and returns
    test predictions. This approach allows the model to learn relationships
    between the different target properties.
    """
    
    # Define the core LightGBM model
    lgbm = lgb.LGBMRegressor(
        objective='regression_l1', metric='mae', n_estimators=2000,
        learning_rate=0.01, num_leaves=31, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.1, lambda_l2=0.1,
        verbose=-1, n_jobs=-1, seed=42
    )

    # Use MultiOutputRegressor to train one model per target
    model = MultiOutputRegressor(lgbm)
    
    # Handle the Kaggle smoke test edge case
    if X_train.shape[0] < 10:
        print("Small dataset detected, training a single model without CV.")
        # Impute NaNs in targets for this simplified training
        y_train_imputed = SimpleImputer(strategy='median').fit_transform(y_train)
        model.fit(X_train, y_train_imputed)
        test_predictions = model.predict(X_test) if X_test.size > 0 else np.array([[]])
        return test_predictions

    # Proceed with cross-validation for the full dataset
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros_like(y_train, dtype=float)
    test_predictions_list = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        print(f"--- Fold {fold+1}/{n_splits} ---")
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Impute missing target values for the current training fold ONLY
        # This prevents data leakage from the validation set.
        imputer_y = SimpleImputer(strategy='median')
        y_train_fold_imputed = imputer_y.fit_transform(y_train_fold)
        
        # Train the multi-output model
        model.fit(X_train_fold, y_train_fold_imputed)
        
        # Store predictions
        oof_predictions[val_idx] = model.predict(X_val_fold)
        if X_test.size > 0:
            test_predictions_list.append(model.predict(X_test))

    # Average predictions across folds for the final test set prediction
    if X_test.size > 0:
        final_test_predictions = np.mean(test_predictions_list, axis=0)
    else:
        final_test_predictions = np.array([[]])

    return final_test_predictions

# --------------------------------------------------------------------------
# Step 6: Main Execution Pipeline
# --------------------------------------------------------------------------
def main():
    # Define file paths for Kaggle environment
    train_file = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
    test_file = '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv'
    target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

    try:
        X_train, y_train, X_test, original_test_df, filtered_test_df = load_and_process_data(train_file, test_file)
    except Exception as e:
        print(f"Fatal Error during data loading: {e}")
        return
        
    submission = pd.DataFrame({'id': original_test_df['id']})
    
    # Calculate fallback predictions (median of each column)
    fallback_preds = y_train.median().to_dict()

    # If the test set is completely invalid, generate a fallback submission
    if X_test.shape[0] == 0:
        print("\nWARNING: No valid molecules found in the test set. Generating fallback submission.")
        for target in target_columns:
            submission[target] = fallback_preds[target]
        submission.to_csv('submission.csv', index=False)
        return

    # Train model and get predictions
    try:
        predictions = train_and_predict(X_train, y_train, X_test)
        predictions_df = pd.DataFrame(predictions, columns=target_columns, index=filtered_test_df['id'])
    except Exception as e:
        print(f"FATAL: An error occurred during model training: {e}")
        # Use fallback if training fails for any reason
        predictions_df = pd.DataFrame(index=filtered_test_df['id'])
        for target in target_columns:
            predictions_df[target] = fallback_preds[target]

    # Map predictions back to the original submission dataframe
    for target in target_columns:
        submission[target] = submission['id'].map(predictions_df[target]).fillna(fallback_preds[target])

    submission_path = 'submission.csv'
    submission.to_csv(submission_path, index=False)
    print("\n" + "="*50)
    print(f"Submission file created successfully: {submission_path}")
    print("Submission file shape:", submission.shape)
    print("Top 5 rows of submission file:")
    print(submission.head())
    print("="*50)

if __name__ == "__main__":
    main()


