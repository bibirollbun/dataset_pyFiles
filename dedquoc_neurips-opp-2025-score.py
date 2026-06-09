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


!pip install tqdm


# install RDKit for offline
#!pip install /kaggle/input/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install /kaggle/input/rdkit-install-whl/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl



%%time
import pandas as pd

supplement_dfs = []
for i in range(1, 5):
    df = pd.read_csv(f"/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset{i}.csv")
    print(f"dataset{i}.csv shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    supplement_dfs.append(df)

# Combine all supplemental data
supplement = pd.concat(supplement_dfs, ignore_index=True)
print(f"Total supplemental data: {len(supplement)} samples")   


%%time
# Load main train
train_main = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")

# Check if supplement has same columns
common_cols = ['SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']
if all(col in supplement.columns for col in common_cols):
    # Use only rows where at least one target is not NaN
    supplement_clean = supplement[common_cols].dropna(how='all', subset=common_cols[1:])
    
    # Add fake 'id' for supplement (avoid conflict)
    supplement_clean['id'] = range(-len(supplement_clean), 0)
    
    # Combine
    train_full = pd.concat([train_main, supplement_clean], ignore_index=True)
    print(f"Extended training set: {len(train_full)} samples (original: {len(train_main)})")
else:
    print("Supplemental data may have different schema â€” inspect manually.")
    display(supplement.head())   


%%time
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
import numpy as np
from tqdm import tqdm

def clean_polymer_smiles(smi):
    """
    Aggressively clean polymer SMILES for RDKit compatibility.
    """
    if not isinstance(smi, str) or not smi.strip():
        return None
    smi = smi.strip()
    
    # Replace wildcard * with H (polymer endpoints)
    smi = smi.replace('*', 'H')
    
    # Fix common invalid patterns
    smi = smi.replace('H)', '')           # Remove isolated H) â†’ invalid
    smi = smi.replace('(H)', '')          # Remove (H) â†’ often invalid branch
    smi = smi.replace('[C@H]', 'CH')      # Remove chiral tags (simplify)
    smi = smi.replace('[C@@H]', 'CH')     # Remove chiral tags
    smi = smi.replace('[CH]', 'C')        # Simplify
    smi = smi.replace('[NH]', 'N')        # Simplify
    smi = smi.replace('[OH]', 'O')        # Simplify
    
    # Remove any remaining brackets if they cause issues (last resort)
    # But better to let RDKit handle them
    return smi

def smi_to_morgan(smi, radius=2, n_bits=2048):
    """
    Robust SMILES to Morgan fingerprint.
    Returns zero vector if parsing fails.
    """
    try:
        cleaned_smi = clean_polymer_smiles(smi)
        if not cleaned_smi:
            return np.zeros(n_bits, dtype=np.float32)

        # Try parsing with RDKit
        mol = Chem.MolFromSmiles(cleaned_smi, sanitize=False)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)

        # Try to sanitize with minimal checks
        try:
            Chem.SanitizeMol(mol, 
                           sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except:
            pass  # Continue even if sanitization fails

        # Generate Morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        ConvertToNumpyArray(fp, arr)
        return arr

    except Exception as e:
        # Catch ALL errors and return zero vector
        return np.zeros(n_bits, dtype=np.float32)   


import pandas as pd
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
import lightgbm as lgb
# Initialize models dictionary
models = {t: [] for t in targets}

# Train LightGBM models
for target_idx, target in enumerate(targets):
    print(f"\nğŸš€ Training {target}")
    y = y_train[:, target_idx]
    
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
        models[target].append(model)   


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


print("Predicting on test set...")
preds = {}

for target in targets:
    pred_per_model = np.zeros(len(X_test_final))
    # Average over 5-fold models
    for model in models[target]:
        pred_per_model += model.predict(X_test_final) / len(models[target])
    preds[target] = pred_per_model

print("âœ… Predictions completed")   


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
# Build submission: one row per (id, variable)
print("Creating submission file...")

submission_rows = []

# Loop over each test sample by index
for idx in range(len(test)):
    # For each target (Tg, FFV, etc.), add one row
    for target in targets:
        # Safely get prediction (ensure it's a scalar)
        pred_value = float(preds[target][idx])  # This should be a single float
        submission_rows.append({
            'id': int(test.iloc[idx]['id']),
            'variable': str(target),
            'prediction': pred_value
        })

# Create DataFrame
submission_df = pd.DataFrame(submission_rows)

# Reorder columns
submission_df = submission_df[['id', 'variable', 'prediction']]

# Final check: correct length?
expected_rows = len(test) * len(targets)
assert len(submission_df) == expected_rows, f"Expected {expected_rows} rows, got {len(submission_df)}"

print(f"âœ… Submission ready! Shape: {submission_df.shape}")
print(f"   â†’ {len(test)} samples Ã— {len(targets)} targets = {expected_rows} rows")   


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


# Load and inspect
check = pd.read_csv('submission.csv')
print("Columns:", check.columns.tolist())
print("Expected: ['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']")
print("\nFirst few rows:")
print(check)   

