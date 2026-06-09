!pip install rdkit-pypi -q
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings and standard warnings
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings("ignore")

print("Setup completed.")



def augment_smiles(smiles, n_variants=5):
    """
    Generate randomized SMILES variants using RDKit.
    Returns list of size n_variants.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles] * n_variants # Fallback for invalid SMILES
        
        variants = set()
        
        # Always keep the canonical version as anchor
        variants.add(Chem.MolToSmiles(mol, canonical=True))
        
        # Generate random variants via graph traversal randomization
        attempts = 0
        max_attempts = n_variants * 5
        
        while len(variants) < n_variants and attempts < max_attempts:
            # doRandom=True generates non-canonical SMILES
            s = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            variants.add(s)
            attempts += 1
            
        # Pad with original SMILES if we couldn't generate enough unique variants
        result = list(variants)
        while len(result) < n_variants:
            result.append(smiles)
            
        return result[:n_variants]
        
    except Exception:
        # Return original on any RDKit internal error
        return [smiles] * n_variants



# Load raw data
train_original = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
print(f"Original size: {len(train_original)}")

# Config augmentation factor
N_VARIANTS = 5 
augmented_rows = []

print(f"Running augmentation (x{N_VARIANTS})...")

for _, row in tqdm(train_original.iterrows(), total=len(train_original)):
    smiles = row['SMILES']
    
    # Generate variants
    variants = augment_smiles(smiles, n_variants=N_VARIANTS)
    
    # Expand dataset
    for s in variants:
        new_row = row.copy()
        new_row['SMILES'] = s
        augmented_rows.append(new_row)

# Convert to DataFrame
train_augmented = pd.DataFrame(augmented_rows)

print(f"Done. Final shape: {train_augmented.shape}")



# Cleanup
train_augmented = train_augmented.reset_index(drop=True)

# Sanity check
print("Preview:")
print(train_augmented[['id', 'SMILES']].head())

# Export
filename = "train_augmented.csv"
train_augmented.to_csv(filename, index=False)
print(f"Saved to {filename}")


