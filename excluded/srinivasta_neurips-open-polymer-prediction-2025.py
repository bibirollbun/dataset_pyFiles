import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold

# -----------------------------
# 1. SMILES and Fingerprint features
# -----------------------------
def extended_smiles_features(smiles_list):
    atoms = ['C', 'O', 'N', 'F', 'S', 'H', 'Si', 'Cl', 'P', 'Br', 'B', 'Ge', 'Se', 'Te', 'Cd', 'Ca']
    features = []
    for smi in smiles_list:
        atom_counts = [smi.count(atom) for atom in atoms]
        features.append([
            len(smi), smi.count('='), smi.count('#'),
            smi.count('('), smi.count(')'), smi.count('1'),
            smi.count('2'), smi.count('3'), smi.lower().count('c'),
            *atom_counts
        ])
    return np.array(features)

def rdkit_descriptors(smiles_list):
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append([np.nan]*5)
        else:
            features.append([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol)
            ])
    return np.array(features)

def rdkit_fingerprints(smiles_list, radius=2, nBits=2048):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append([0]*nBits)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            fps.append(fp.ToList())
    return np.array(fps)

# -----------------------------
# 2. Load data
# -----------------------------
train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# -----------------------------
# 3. Featurize
# -----------------------------
def featurize(df):
    ext = extended_smiles_features(df['SMILES'])
    desc = rdkit_descriptors(df['SMILES'])
    fp = rdkit_fingerprints(df['SMILES'])
    return np.hstack([ext, desc, fp])

X_train_full = featurize(train_df)
X_test = featurize(test_df)

# -----------------------------
# 4. Remove constant features
# -----------------------------
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.0)
X_train_full = selector.fit_transform(X_train_full)
X_test = selector.transform(X_test)

# -----------------------------
# 5. Train one model per target
# -----------------------------
submission = pd.DataFrame({'id': test_df['id']})

for i, col in enumerate(target_cols):
    print(f"\nðŸ”¬ Training for target: {col}")
    target = train_df[col]
    
    not_null_idx = ~target.isna()
    X_target = X_train_full[not_null_idx]
    y_target = target[not_null_idx].values
    
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=10,
        random_state=42,
        min_gain_to_split=0.0  # Suppress no-split warnings
    )
    model.fit(X_target, y_target)
    
    y_pred = model.predict(X_test)
    submission[col] = y_pred

# -----------------------------
# 6. Save submission
# -----------------------------
submission.to_csv('submission.csv', index=False)
print("âœ… Per-target LightGBM submission saved as submission.csv")


