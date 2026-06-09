# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import warnings
warnings.filterwarnings("ignore")


%%time
# install RDKit for offline
!pip install /kaggle/input//rdkit-install-whl/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install tqdm 


%%time
import polars as pl

# Define paths
train_path = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
test_path = '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv'

# Lazy load to inspect schema (columns) â€” no data loaded
lazy_train = pl.scan_csv(train_path)
lazy_test = pl.scan_csv(test_path)

print("Train columns:", lazy_train.columns)
print("Test columns:", lazy_test.columns)

# Peek at first 20 rows using Polars (faster and consistent)
train_df = pl.read_csv(train_path, n_rows=20)
test_df = pl.read_csv(test_path, n_rows=20)

print("\nTrain preview:")
print(train_df)
print("\nTest preview:")
print(test_df)   


%%time
from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import Dict, Optional

def calc_features(smiles: str) -> Dict[str, Optional[float]]:
    """
    Calculate molecular descriptors from a SMILES string.
    
    Args:
        smiles (str): Input SMILES string.
        
    Returns:
        Dict[str, Optional[float]]: Dictionary with MolWt, NumAtoms, NumRings, TPSA.
                                   Returns None values if SMILES is invalid or error occurs.
    """
    # Initialize default return values
    features = {"MolWt": None, "NumAtoms": None, "NumRings": None, "TPSA": None}
    
    if not isinstance(smiles, str) or not smiles.strip():
        return features  # Return Nones for invalid input

    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return features  # RDKit couldn't parse the SMILES

        features.update({
            "MolWt": Descriptors.MolWt(mol),
            "NumAtoms": mol.GetNumAtoms(),
            "NumRings": Chem.GetSSSR(mol),  # Returns int-like; no need to cast in modern RDKit
            "TPSA": Descriptors.TPSA(mol),
        })
    except Exception as e:
        # Optionally log or debug: print(f"Error processing SMILES '{smiles}': {e}")
        pass  # Return defaults (None) on any error

    return features   


%%time
import polars as pl
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
import pandas as pd
from typing import Dict, Optional

def extract_features(smiles: str, mol_id: int) -> Optional[Dict[str, Optional[float]]]:
    """
    Extract multiple RDKit descriptors from a SMILES string.
    
    Args:
        smiles (str): SMILES string.
        mol_id (int): Molecule ID.
        
    Returns:
        Dict of features or None if invalid.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None

        return {
            'id': mol_id,
            'MolWt': Descriptors.MolWt(mol),
            'MolLogP': Descriptors.MolLogP(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'RingCount': Descriptors.RingCount(mol)
        }
    except Exception:
        return None  # Broad but safe for cheminformatics edge cases

# --- Efficient chunked processing to avoid memory overload ---
chunk_size = 10_000
features = []

# Use Polars for fast, low-memory iteration (or pandas with chunking)
for chunk in pd.read_csv(
    "/kaggle/input/neurips-open-polymer-prediction-2025/train.csv",
    chunksize=chunk_size
):
    batch_features = [
        extract_features(smi, id_)
        for smi, id_ in zip(chunk['SMILES'], chunk['id'])
    ]
    # Filter out None results
    batch_features = [f for f in batch_features if f is not None]
    features.extend(batch_features)

# Convert to DataFrame
train_feat = pd.DataFrame(features)

# Save efficiently
train_feat.to_feather("train_feat.feather")
print(f"âœ… Extracted features for {len(train_feat)} valid molecules â†’ saved to 'train_feat.feather'")   


%%time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load processed features
train_feat = pd.read_feather("train_feat.feather")
train_targets = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")[['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]

# Merge for visualization
df = pd.merge(train_feat, train_targets, on="id")

# Set aesthetic
sns.set(style="whitegrid", palette="muted")


%%time
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

for col in target_cols:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[col], kde=True, bins=40)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()


%%time
plt.figure(figsize=(14, 10))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Feature-Target Correlation")
plt.tight_layout()
plt.show()


%%time
# Example scatter for MolWt vs Tg
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='MolWt', y='Tg')
plt.title("MolWt vs Glass Transition Temp (Tg)")
plt.tight_layout()
plt.show()


%%time
# improve code above
import lightgbm as lgb
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

# ----------------------------
# 1. Load Data
# ----------------------------
print("Loading data...")
train_feat = pd.read_feather("train_feat.feather")  # Contains SMILES + descriptors + 'id'
target_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")

# Merge features with targets using 'id'
df = train_feat.merge(target_df, on="id")  # â†� Fixed: was merging with undefined `train`
print(f"Training dataset shape: {df.shape}")

# Define features and targets
features = ['MolWt', 'MolLogP', 'NumHAcceptors', 'NumHDonors', 'TPSA', 'NumRotatableBonds', 'RingCount']
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
models = {target: [] for target in targets}

# ----------------------------
# 2. Train Models
# ----------------------------
print("\nStarting training...")
for target in targets:
    print(f"\nğŸ�¯ Training for target: {target}")
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        X_train = df.iloc[train_idx][features]
        y_train = df.iloc[train_idx][target]
        X_val = df.iloc[val_idx][features]
        y_val = df.iloc[val_idx][target]

        # Handle missing values (just in case)
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params={
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': -1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'seed': 42,
                'force_col_wise': True,
            },
            train_set=train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(100)
            ]
        )
        models[target].append(model)
        print(f"Fold {fold} - Best score: {model.best_score['valid_0']['rmse']:.4f}")  


%%time
from rdkit import Chem
from rdkit.Chem import Draw

# Example: Show 5 random molecules from your dataset
sample_smiles = df['SMILES'].dropna().sample(5, random_state=42).tolist()
mols = [Chem.MolFromSmiles(s) for s in sample_smiles]
Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200,200))


%%time
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
import numpy as np
from typing import Optional
import warnings

# Optional: enable warning tracking
TRACK_FAILURES = False
_failed_smiles = set()

def clean_polymer_smiles(smi: str) -> Optional[str]:
    """
    Lightly clean polymer SMILES to improve RDKit compatibility.
    Prefers minimal modification over aggressive rewriting.
    """
    if not isinstance(smi, str) or not smi.strip():
        return None
    smi = smi.strip()

    # Replace polymer terminal wildcard * with H
    smi = smi.replace('*', 'H')

    # Remove common chiral annotations that may block parsing
    smi = smi.replace('[C@H]', 'C').replace('[C@@H]', 'C')
    smi = smi.replace('[CH]', 'C').replace('[NH]', 'N').replace('[OH]', 'O')

    return smi.strip() or None

def smi_to_morgan(
    smi: str,
    radius: int = 2,
    n_bits: int = 2048
) -> np.ndarray:
    """
    Convert SMILES to Morgan fingerprint (ECPF) bit vector.
    
    Returns:
        np.ndarray of shape (n_bits,) with dtype float32.
        Returns zero vector on failure.
    """
    global _failed_smiles

    # Step 1: Clean input
    cleaned_smi = clean_polymer_smiles(smi)
    if not cleaned_smi:
        if TRACK_FAILURES:
            _failed_smiles.add(smi)
        return np.zeros(n_bits, dtype=np.float32)

    # Step 2: Parse molecule
    try:
        mol = Chem.MolFromSmiles(cleaned_smi, sanitize=False)
        if mol is None:
            if TRACK_FAILURES:
                _failed_smiles.add(smi)
            return np.zeros(n_bits, dtype=np.float32)
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)

    # Step 3: Light sanitization (skip problematic steps)
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        )
    except Exception:
        pass  # Proceed even if sanitization fails

    # Step 4: Generate fingerprint
    try:  
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)   


import pandas as pd
# 2
# This function run slowly ..
# Load data
train_main = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
supplement = pd.concat([
    pd.read_csv(f"/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset{i}.csv")
    for i in range(1, 5)
], ignore_index=True)

# Combine all SMILES
all_smiles = pd.concat([train_main['SMILES'], supplement['SMILES']], ignore_index=True)
print(f"Total SMILES to process: {len(all_smiles)}")

# Generate fingerprints with error resilience
print("Generating fingerprints (this may take a few minutes)...")
X_all = []
for smi in tqdm(all_smiles, desc="Processing SMILES"):
    try:
        fp = smi_to_morgan(smi)
    except Exception:
        fp = np.zeros(2048, dtype=np.float32)  # Fallback
    X_all.append(fp)

X_all = np.array(X_all, dtype=np.float32)
print(f"Fingerprint matrix shape: {X_all.shape}")   


%%time
# 3
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

# Scale all fingerprints
print("Scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)  # (16963, 2048)

print("Fitting PCA...")
pca = PCA(n_components=64)
X_pca_all = pca.fit_transform(X_scaled)

# Save for reproducibility
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')

print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")  # ~0.95+ = good   


%%time
# 4
import pandas as pd

# Load main train (has Tg, FFV, etc.)
train_main = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")

# Generate fingerprints
X_train_raw = []
for smi in tqdm(train_main['SMILES'], desc="Train Featurization"):
    X_train_raw.append(smi_to_morgan(smi))
X_train_raw = np.array(X_train_raw)

# Transform using fitted scaler + PCA
X_train_scaled = scaler.transform(X_train_raw)
X_train_final = pca.transform(X_train_scaled)  # (n_train, 64)

# Extract targets
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
y_train = train_main[targets].values   


%%time
# Load test
test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")

# Featurize
X_test_raw = np.array([smi_to_morgan(smi) for smi in test['SMILES']])
X_test_scaled = scaler.transform(X_test_raw)
X_test_final = pca.transform(X_test_scaled)  # (n_test, 64)   


from sklearn.model_selection import KFold

# Define cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"Using {kf.get_n_splits(X_train_final)}-fold cross-validation")   


%%time
from sklearn.model_selection import KFold
import lightgbm as lgb

# Define CV splitter
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize models dict
models = {t: [] for t in targets}

# Train one model per target
for target_idx, target in enumerate(targets):
    print(f"\nğŸš€ Training for {target}")
    y = y_train[:, target_idx]  # Extract target column
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_final)):
        X_tr, X_val = X_train_final[tr_idx], X_train_final[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        train_set = lgb.Dataset(X_tr, label=y_tr)
        val_set = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            params={
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'verbose': -1,
                'seed': 42,
            },
            train_set=train_set,
            valid_sets=[val_set],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        models[target].append(model)  # Save model for ensemble

print("âœ… All models trained and stored.")   


%%time
# Make predictions with shape validation
preds = {}
for target in targets:
    if target not in models or len(models[target]) == 0:
        print(f"âš ï¸� No models found for {target}")
        preds[target] = np.zeros(len(test))
        continue
        
    n_models = len(models[target])
    pred_per_model = np.zeros(len(X_test_final))
    
    for model in models[target]:
        try:
            pred = model.predict(X_test_final)
            if pred.shape[0] != len(X_test_final):
                raise ValueError(f"Prediction shape mismatch: {pred.shape} vs {len(X_test_final)}")
            pred_per_model += pred / n_models
        except Exception as e:
            print(f"Error in {target} prediction: {str(e)}")
            # Continue with other models
    
    preds[target] = pred_per_model

print("âœ… Predictions completed")   


%%time
import pandas as pd

# Create a DataFrame with predictions
submission_df = pd.DataFrame({
    'id': test['id']  # Keep original test IDs
})

# Add each target as a column
for target in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    submission_df[target] = preds[target]  # Already in order

# Ensure correct column order
submission_df = submission_df[['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]

# Convert to correct types
submission_df['id'] = submission_df['id'].astype(int)
for target in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    submission_df[target] = pd.to_numeric(submission_df[target], errors='coerce')

# Save
submission_df.to_csv('submission.csv', index=False)

print(f"âœ… Submission saved in WIDE format: 'submission.csv'")
print(f"Shape: {submission_df.shape}")
print("\nFirst 3 rows:")
print(submission_df.head(3)) 

