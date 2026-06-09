# Environment Setup
import numpy as np
import pandas as pd
from pathlib import Path
import math
import sys
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Install RDKit from pre-compiled wheel
!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl

# Set seeds for full reproducibility
SEED = 42
np.random.seed(SEED)

# Data Loading & Configuration
# File paths (Kaggle competition specific)
train_path = Path('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_path = Path('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
subm_path = Path('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')

# Model constants
N_BITS = 2048   # Morgan fingerprint resolution
RADIUS = 3      # Atomic neighborhood radius
TEST_SIZE = 0.2 # Validation split percentage


# Molecular Processing
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors
RDLogger.DisableLog('rdApp.*')

# Quick sanity check:
mol = Chem.MolFromSmiles("CCO")
print("RDKit fingerprint size:", len(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)))

def safe_smiles_to_fp(smiles, n_bits=N_BITS, radius=RADIUS):
    """
    Convert SMILES to Morgan fingerprint with comprehensive error handling
    Returns: RDKit fingerprint object or None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Basic sanitization check
        if mol.GetNumAtoms() < 1:
            return None
            
        # Generate fingerprint
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception as e:
        # Log truncated SMILES for debugging
        if len(smiles) > 30:
            smiles_repr = f"{smiles[:15]}...{smiles[-15:]}"
        else:
            smiles_repr = smiles
        print(f"Error processing SMILES '{smiles_repr}': {str(e)}")
        return None

def fingerprints_to_array(fingerprints):
    """
    Convert RDKit fingerprint objects to numpy array
    Returns: (n_samples, n_bits) binary array
    """
    arr = np.zeros((len(fingerprints), N_BITS), dtype=np.uint8)
    for i, fp in enumerate(fingerprints):
        if fp is not None:
            DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


# Evaluation Metric
def weighted_mae(y_true, y_pred, R_ranges, N_total=1500):
    """
    Compute competition specific weighted MAE
    Args:
        y_true: Actual values (n_samples, n_properties)
        y_pred: Predicted values (n_samples, n_properties)
        R_ranges: Property ranges for normalization
        N_total: Test set size (default=1500)
    Returns:
        wMAE (float)
    """
    n_properties = y_true.shape[1]
    abs_errors = np.abs(y_true - y_pred)
    
    # Competition parameters
    n_p = {i: N_total for i in range(n_properties)}  # All properties available
    N = N_total
    
    # Calculate unnormalized weights
    w_prime = np.zeros(n_properties)
    for i in range(n_properties):
        w_prime[i] = (math.sqrt(N) / math.sqrt(n_p[i])) * (1 / R_ranges[i])
    
    # Apply competition normalization
    w_normalized = w_prime * (n_properties / np.sum(w_prime))
    
    # Compute sample errors
    sample_errors = np.zeros(y_true.shape[0])
    for i in range(y_true.shape[0]):
        sample_errors[i] = np.mean(abs_errors[i] * w_normalized)
    
    return np.mean(sample_errors)


# Modeling
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def build_model():
    """Configure competition-optimized GBM"""
    return GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        random_state=SEED,
        subsample=0.8,
        validation_fraction=0.1,
        n_iter_no_change=15,
        tol=1e-5
    )


# Main Processing Pipeline
def main():
    # Load datasets with robust column handling
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    submission_df = pd.read_csv(subm_path)
    
    # Standardize column names by stripping whitespace
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()
    submission_df.columns = submission_df.columns.str.strip()
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Define target properties
    properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Calculate property ranges (handle missing values)
    R_ranges = []
    print("\nProperty ranges (R_ranges):")
    for prop in properties:
        if prop in train_df.columns:
            # Handle NaN values in properties
            vals = train_df[prop].dropna()
            if len(vals) > 0:
                q5 = np.percentile(vals, 5)
                q95 = np.percentile(vals, 95)
                R_range = q95 - q5
                print(f"  {prop}: {R_range:.4f}")
                R_ranges.append(R_range)
            else:
                print(f"  {prop}: No valid values - using default range")
                R_ranges.append(1.0)  # Default range
        else:
            print(f"  {prop}: Column missing - using default range")
            R_ranges.append(1.0)  # Default range
    
    # Process training SMILES with strict NaN handling
    print("\nProcessing training SMILES...")
    train_fps = []
    valid_indices = []
    
    # Pre-clean SMILES and filter NaN targets
    clean_train_df = train_df.dropna(subset=properties)
    print(f"  Samples with complete properties: {len(clean_train_df)}/{len(train_df)}")
    
    for i, row in clean_train_df.iterrows():
        smiles = str(row['SMILES']).strip()
        if not smiles:
            continue
            
        fp = safe_smiles_to_fp(smiles)
        if fp is not None:
            train_fps.append(fp)
            valid_indices.append(i)
    
    # Create feature matrix and target array
    X_train = fingerprints_to_array(train_fps)
    y_train = clean_train_df.iloc[valid_indices][properties].values
    
    print(f"Valid training samples: {X_train.shape[0]}/{len(train_df)}")
    
    # Process test SMILES with robust error handling
    print("\nProcessing test SMILES...")
    test_fps = []
    for i, row in test_df.iterrows():
        smiles = str(row['SMILES']).strip()
        fp = safe_smiles_to_fp(smiles) if smiles else None
        
        if fp is None:
            # Generate valid fingerprint for invalid SMILES
            try:
                null_mol = Chem.MolFromSmiles('C')
                fp = AllChem.GetMorganFingerprintAsBitVect(null_mol, RADIUS, nBits=N_BITS)
            except:
                # Fallback to simple carbon molecule
                fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles('C'), RADIUS, nBits=N_BITS)
        test_fps.append(fp)
    
    X_test = fingerprints_to_array(test_fps)
    print(f"Test samples processed: {X_test.shape[0]}")
    
    # Only proceed if we have training data
    if X_train.shape[0] == 0:
        print("\nCRITICAL: No valid training samples found. Using fallback predictions.")
        # Generate fallback predictions (median values from training)
        fallback_preds = []
        for prop in properties:
            if prop in train_df.columns:
                # Handle all-NaN columns
                if train_df[prop].isna().all():
                    fallback_preds.append(0.0)
                else:
                    fallback_preds.append(train_df[prop].median(skipna=True))
            else:
                fallback_preds.append(0.0)
        test_preds = np.array([fallback_preds] * len(test_df))
    else:
        # Train model without validation if insufficient samples
        if X_train.shape[0] <= 10:
            print("\nInsufficient samples for validation. Training final model directly.")
            final_model = MultiOutputRegressor(build_model())
            final_model.fit(X_train, y_train)
            test_preds = final_model.predict(X_test)
        else:
            print("\nSplitting data for validation...")
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=TEST_SIZE, random_state=SEED
            )
            print(f"Train split: {X_train_split.shape[0]} samples")
            print(f"Validation split: {X_val.shape[0]} samples")
            
            # Initialize and train model
            base_model = build_model()
            model = MultiOutputRegressor(estimator=base_model)
            model.fit(X_train_split, y_train_split)
            
            # Validate model
            val_preds = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_preds, multioutput='raw_values')
            
            print("\nValidation MAE per property:")
            for prop, mae in zip(properties, val_mae):
                print(f"  {prop}: {mae:.4f}")
            
            # Calculate weighted MAE if we have ranges
            if len(R_ranges) == len(properties):
                wmae = weighted_mae(y_val, val_preds, R_ranges)
                print(f"\nValidation Weighted MAE: {wmae:.4f}")
        
            # Train final model on all available data
            print("\nTraining final model on all valid data...")
            final_model = MultiOutputRegressor(build_model())
            final_model.fit(X_train, y_train)
            test_preds = final_model.predict(X_test)
    
    # Prepare submission
    submission_df[properties] = test_preds
    submission_path = "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")
    print("First 5 predictions:")
    print(submission_df.head().to_string(index=False))
    
    return submission_path

if __name__ == "__main__":
    main()

