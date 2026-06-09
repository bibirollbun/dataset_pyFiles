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


!pip install /kaggle/input/rdkit-install-whl/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl




# Import Libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.optimize import minimize


from lightgbm import LGBMRegressor, early_stopping


# Chemistry libraries
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, AllChem,  DataStructs
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds

# New imports (add after your current imports)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
print("All libraries imported successfully!")

import optuna


from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNet




# Cell 3: Configuration
RANDOM_STATE = 42
N_SPLITS = 5
N_TRIALS = 50  # Optuna trials - reduce for faster experimentation

# Competition properties
PROPERTIES = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# File paths
DATA_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'
SUPPLEMENT_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/'

print(f"Configuration set: Random State = {RANDOM_STATE}")


modred_path = "/kaggle/input/modred-dataset/"
modred_files = {
    "FFV": "desc_ffv.csv",
    "Density": "desc_de.csv",
    "Tc": "desc_tc.csv",
    "Tg": "desc_tg.csv",
    "Rg": "desc_rg.csv"
}

modred = {}
for prop, fname in modred_files.items():
    df = pd.read_csv(f"{modred_path}{fname}")
    print(f"--- {prop} (raw) ---")
    print(f"Columns: {df.columns.tolist()[-10:]}")  # Show last 10 columns
    print(f"Shape: {df.shape}")
    
    # Remove duplicate column names, keeping the last occurrence
    df = df.loc[:, ~df.columns.duplicated(keep='last')]
    
    # Extract only SMILES and target columns
    if 'SMILES' in df.columns and prop in df.columns:
        # Select just the SMILES and target columns
        df_clean = df[['SMILES', prop]].copy()
        
        # Convert target to numeric and clean data
        df_clean[prop] = pd.to_numeric(df_clean[prop], errors='coerce')
        df_clean = df_clean.dropna()
        
        # Remove invalid values
        df_clean = df_clean[df_clean[prop] != np.inf]
        df_clean = df_clean[df_clean[prop] != -np.inf] 
        df_clean = df_clean[df_clean['SMILES'].str.len() > 0]
        
        modred[prop] = df_clean.reset_index(drop=True)
        print(f"{prop}: {len(modred[prop])} valid samples after cleaning")
    else:
        print(f"Warning: Could not find SMILES or {prop} columns")
        print(f"Available columns: {[col for col in df.columns if 'SMILES' in col.upper() or prop.lower() in col.lower()]}")

# Verify the results
for prop, df_prop in modred.items():
    print(f"\n{prop}: {df_prop.columns.tolist()} | shape={df_prop.shape}")
    if len(df_prop) > 0:
        print(f"Sample values:")
        print(df_prop.head(2))
        print(f"Target range: {df_prop[prop].min():.4f} to {df_prop[prop].max():.4f}")
    print()


# Cell 4: Load Data
train = pd.read_csv(f"{DATA_PATH}train.csv")
test = pd.read_csv(f"{DATA_PATH}test.csv")
sample_sub = pd.read_csv(f"{DATA_PATH}sample_submission.csv")

# Load supplement datasets
supp1 = pd.read_csv(f'{SUPPLEMENT_PATH}dataset1.csv')
supp2 = pd.read_csv(f'{SUPPLEMENT_PATH}dataset2.csv')
supp3 = pd.read_csv(f'{SUPPLEMENT_PATH}dataset3.csv')
supp4 = pd.read_csv(f'{SUPPLEMENT_PATH}dataset4.csv')

print("Data loaded successfully!")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Cell 5: Quick EDA
print("=== TRAIN DATA INFO ===")
train.info()
print("\n=== MISSING VALUES ===")
print(train.isnull().sum())
print("\n=== BASIC STATISTICS ===")
train.describe()


# === Polymer-safe featurization: ECFP bits + a few robust 2D counts (no 3D, no charges) ===

N_BITS = 4096  # fingerprint length

def canon_poly_smiles(s: str):
    """Cap polymer connection points and return (Mol, canonical SMILES)."""
    # Replace '*' with a neutral carbon cap to avoid valence errors
    s_cap = s.replace('*', 'C')
    mol = Chem.MolFromSmiles(s_cap)
    if mol is None:
        return None, None
    can = Chem.MolToSmiles(mol, canonical=True)
    return mol, can

def get_robust_descriptors(mol):
    """
    Get only the most robust and polymer-relevant descriptors
    Focus on descriptors that rarely fail and are known to correlate with polymer properties
    ALWAYS returns exactly 25 features
    """
    # Initialize with zeros - ensures consistent size
    descriptors = np.zeros(25, dtype=float)
    
    if mol is None:
        return descriptors
    
    try:
        # Core molecular properties (always work)
        descriptors[0] = Descriptors.MolWt(mol)
        descriptors[1] = mol.GetNumHeavyAtoms()
        
        # Atom counts (robust)
        descriptors[2] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)   # Carbon
        descriptors[3] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)   # Oxygen  
        descriptors[4] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)   # Nitrogen
        descriptors[5] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)  # Sulfur
        descriptors[6] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)   # Fluorine
        
        # Structural descriptors (usually robust)
        try:
            descriptors[7] = rdMolDescriptors.CalcTPSA(mol)
            descriptors[8] = rdMolDescriptors.CalcNumRotatableBonds(mol)
            descriptors[9] = rdMolDescriptors.CalcNumRings(mol)
            descriptors[10] = rdMolDescriptors.CalcNumAromaticRings(mol)
        except:
            pass  # Keep zeros
        
        # Lipophilicity (can occasionally fail)
        try:
            descriptors[11] = Descriptors.MolLogP(mol)
        except:
            pass
            
        try:
            descriptors[12] = Descriptors.MolMR(mol)  # Molar refractivity
        except:
            pass
        
        # Hydrogen bonding descriptors
        try:
            descriptors[13] = rdMolDescriptors.CalcNumHBD(mol)  # H-bond donors
            descriptors[14] = rdMolDescriptors.CalcNumHBA(mol)  # H-bond acceptors
        except:
            pass
        
        # Additional robust descriptors
        try:
            descriptors[15] = rdMolDescriptors.CalcNumHeteroatoms(mol)
            descriptors[16] = rdMolDescriptors.CalcFractionCsp3(mol)
            descriptors[17] = rdMolDescriptors.CalcNumAliphaticRings(mol)
            descriptors[18] = rdMolDescriptors.CalcNumSaturatedRings(mol)
        except:
            pass
        
        # Derived features that are useful for polymers (safe calculations)
        heavy_atoms = descriptors[1]
        n_c = descriptors[2]
        n_o = descriptors[3] 
        n_heteroatoms = descriptors[15]
        n_rings = descriptors[9]
        n_aromatic_rings = descriptors[10]
        
        descriptors[19] = n_c / heavy_atoms if heavy_atoms > 0 else 0.0  # c_ratio
        descriptors[20] = n_o / heavy_atoms if heavy_atoms > 0 else 0.0  # o_ratio
        descriptors[21] = n_heteroatoms / heavy_atoms if heavy_atoms > 0 else 0.0  # hetero_ratio
        descriptors[22] = n_rings / heavy_atoms if heavy_atoms > 0 else 0.0  # ring_density
        descriptors[23] = n_aromatic_rings / max(n_rings, 1) if n_rings > 0 else 0.0  # aromatic_ratio
        descriptors[24] = descriptors[8] / heavy_atoms if heavy_atoms > 0 else 0.0  # rotatable_ratio
        
        # Handle any NaN or inf values
        descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)
        
        return descriptors
        
    except Exception as e:
        print(f"Descriptor calculation failed for molecule: {e}")
        return np.zeros(25, dtype=float)


# Hybrid Pipeline: Property-specific feature selection and model choice

def fp_features_hybrid(smiles: str, property_name: str):
    """
    Property-specific featurization:
    - Tg, Tc: Use Optuna-only approach (4*2048 + 10 = 8202 features)
    - FFV, Density, Rg: Use conservative approach (4*2048 + 25 = 8217 features)
    """
    mol, can = canon_poly_smiles(smiles)
    if mol is None:
        if property_name in ['Tg', 'Tc']:
            return np.zeros(4 * N_BITS + 10, dtype=float)  # Optuna-only size
        else:
            return np.zeros(4 * N_BITS + 25, dtype=float)  # Conservative size

    # Core fingerprints (same for all properties)
    # 1. Standard bit-based ECFP2 (radius=2)
    fp2_bit = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=N_BITS, useChirality=True
    )
    bits2 = np.zeros((N_BITS,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp2_bit, bits2)

    # 2. Standard bit-based ECFP3 (radius=3) 
    fp3_bit = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=3, nBits=N_BITS, useChirality=True
    )
    bits3 = np.zeros((N_BITS,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp3_bit, bits3)

        # 3. Count-based ECFP2 (radius=2) - deterministic hashed counts
    fp2_count = AllChem.GetHashedMorganFingerprint(
        mol, radius=2, nBits=N_BITS, useChirality=True
    )
    counts2 = np.zeros((N_BITS,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp2_count, counts2)
    
    # 4. Count-based ECFP3 (radius=3) - deterministic hashed counts  
    fp3_count = AllChem.GetHashedMorganFingerprint(
        mol, radius=3, nBits=N_BITS, useChirality=True
    )
    counts3 = np.zeros((N_BITS,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp3_count, counts3)

    # Base fingerprint features (4*2048 = 8192 features)
    base_features = np.concatenate([
        bits2.astype(float),
        bits3.astype(float),
        counts2.astype(float),
        counts3.astype(float)
    ])

    # Property-specific descriptors
    if property_name in ['Tg', 'Tc']:
        # Use original 10 descriptors from Optuna-only approach (what worked better)
        def count_atomic(Z): 
            return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == Z)

        descriptors_2d = np.array([
            Descriptors.MolWt(mol),
            count_atomic(6),  # C
            count_atomic(8),  # O
            count_atomic(7),  # N
            count_atomic(16), # S
            mol.GetNumHeavyAtoms(),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            Descriptors.MolLogP(mol),
            Descriptors.MolMR(mol),
        ], dtype=float)
        
        return np.concatenate([base_features, descriptors_2d])
    else:
        # Use 25 robust descriptors for FFV, Density, Rg
        robust_descriptors = get_robust_descriptors(mol)
        return np.concatenate([base_features, robust_descriptors])

def featurize_series_hybrid(smiles_series: pd.Series, property_name: str) -> np.ndarray:
    """Property-specific featurization"""
    return np.vstack([fp_features_hybrid(s, property_name) for s in smiles_series])


# === Grouped folds by canonical SMILES to prevent leakage ===
def make_groups(df: pd.DataFrame) -> np.ndarray:
    cans = []
    for s in df['SMILES'].tolist():
        _, can = canon_poly_smiles(s)
        cans.append(can if can is not None else s)
    return np.array(cans)

def get_group_folds(df: pd.DataFrame, n_splits=5):
    groups = make_groups(df)
    gkf = GroupKFold(n_splits=n_splits)
    for tr_idx, va_idx in gkf.split(df, groups=groups):
        yield tr_idx, va_idx, groups


# Cell 8: Data Preparation Functions
def prepare_data_for_property(train_data, supplement_data, property_name, modred_data):
    """Prepare training data for a specific property with optional modred augmentation"""
    
    def clean(df, prop):
        """Ensure df has exactly two columns: SMILES + property"""
        if df is None or df.empty:
            return pd.DataFrame(columns=['SMILES', prop])
        # Drop duplicate columns if any
        df = df.loc[:, ~df.columns.duplicated()]
        # Keep only relevant cols
        if prop in df.columns:
            return df[['SMILES', prop]].dropna()
        return pd.DataFrame(columns=['SMILES', prop])
    
    if property_name == 'Tg':
        dfs = [
            clean(train_data[['SMILES', property_name]], 'Tg'),
            clean(supplement_data['supp3'], 'Tg'),
            clean(modred_data.get('Tg'), 'Tg')
        ]
        
    elif property_name == 'Tc':
        supp1_renamed = supplement_data['supp1'].rename(columns={'TC_mean': 'Tc'})
        dfs = [
            clean(train_data[['SMILES', property_name]], 'Tc'),
            clean(supp1_renamed, 'Tc'),
            clean(modred_data.get('Tc'), 'Tc')
        ]
        
    elif property_name == 'FFV':
        dfs = [
            clean(train_data[['SMILES', property_name]], 'FFV'),
            clean(supplement_data['supp4'], 'FFV'),
            clean(modred_data.get('FFV'), 'FFV')
        ]
        
    elif property_name == 'Density':
        dfs = [
            clean(train_data[['SMILES', property_name]], 'Density'),
            clean(modred_data.get('Density'), 'Density')
        ]
        
    elif property_name == 'Rg':
        dfs = [
            clean(train_data[['SMILES', property_name]], 'Rg'),
            clean(modred_data.get('Rg'), 'Rg')
        ]
    else:
        dfs = [clean(train_data[['SMILES', property_name]], property_name)]

    combined = pd.concat(dfs, ignore_index=True)
    return combined.drop_duplicates(subset='SMILES').reset_index(drop=True)




# Cell 9: Evaluation Functions
def compute_wmae(all_models, all_X_data, all_y_data, property_names=['Tg', 'FFV', 'Tc', 'Density', 'Rg']):
    """
    Compute wMAE directly from your trained models - matches your original structure
    """
    K = len(all_models)
     # Calculate n_i and r_i for each property
    n_values = [len(y) for y in all_y_data]
    r_values = [y.max() - y.min() for y in all_y_data]
    
    # Calculate denominator
    denominator = sum(np.sqrt(1 / np.array(n_values)))
    
    # Calculate weights
    weights = []
    for i in range(K):
        w_i = (1 / r_values[i]) * (K * np.sqrt(1 / n_values[i])) / denominator
        weights.append(w_i)
    
    # Calculate weighted errors
    total_error = 0
    total_count = 0
    
    for i, (model, X_data, Y_data) in enumerate(zip(all_models, all_X_data, all_y_data)):
        y_pred = model.predict(X_data)
        mae_i = np.abs(y_pred - Y_data).sum()
        total_error += weights[i] * mae_i
        total_count += len(Y_data)
    
    wmae = total_error / total_count
    return wmae, weights

def evaluate_model(model, X_val, y_val, property_name):
    """Evaluate single model"""
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    print(f"{property_name} MAE: {mae:.4f}")
    return mae, preds




# Optuna logging - set to WARNING to reduce output
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_lgbm_params(X_train, y_train, groups, n_trials=50, n_splits=5, seed=42, property_name=""):
    """
    Optimize LGBM hyperparameters using Optuna with GroupKFold CV
    
    Args:
        X_train: Training features
        y_train: Training targets  
        groups: Groups for GroupKFold (canonical SMILES)
        n_trials: Number of Optuna trials
        n_splits: Number of CV folds
        seed: Random seed
        property_name: Name of property for logging
        
    Returns:
        best_params: Dictionary of best hyperparameters
        best_score: Best CV MAE score
    """
    
    def objective(trial):
        # Define hyperparameter search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 32, 512, step=16),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100, step=5),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.05),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0, step=0.1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0, step=0.1),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0, step=0.05),
            'random_state': seed,
            'verbose': -1,
            'force_col_wise': True,  # For better performance with many features
        }
        
        # Perform GroupKFold CV
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=n_splits)
        
        cv_scores = []
        for train_idx, val_idx in gkf.split(X_train, y_train, groups):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = LGBMRegressor(**params)
            # Fit without pruning callback to avoid dependency issues
            model.fit(X_tr, y_tr, 
                     eval_set=[(X_val, y_val)],
                     eval_metric='mae',
                     callbacks=[early_stopping(stopping_rounds=100)])  # Empty callbacks list
            
            pred_val = model.predict(X_val)
            mae = mean_absolute_error(y_val, pred_val)
            cv_scores.append(mae)
        
        return np.mean(cv_scores)
    
    # Create and run study
    study = optuna.create_study(direction='minimize', 
                               sampler=optuna.samplers.TPESampler(seed=seed))
    
    print(f"Optimizing LGBM for {property_name}...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_score = study.best_value
    
    print(f"[{property_name}] Best MAE: {best_score:.4f}")
    print(f"[{property_name}] Best params: {best_params}")
    
    return best_params, best_score


def get_fast_catboost_params(property_name: str, seed: int = 42):
    """
    Fast CatBoost parameters - pre-tuned for polymer data
    No optimization needed, just good default parameters
    """
    
    # Property-specific tuned parameters based on common polymer prediction patterns
    property_configs = {
        'Tg': {
            'iterations': 1500,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3.0,
            'bagging_temperature': 0.5,
            'random_strength': 2.0,
            'border_count': 128,
            'subsample': 0.8,
        },
        'Tc': {
            'iterations': 1200,
            'learning_rate': 0.06,
            'depth': 7,
            'l2_leaf_reg': 5.0,
            'bagging_temperature': 0.3,
            'random_strength': 1.5,
            'border_count': 128,
            'subsample': 0.85,
        },
        'FFV': {
            'iterations': 1800,
            'learning_rate': 0.04,
            'depth': 8,
            'l2_leaf_reg': 2.0,
            'bagging_temperature': 0.7,
            'random_strength': 3.0,
            'border_count': 128,
            'subsample': 0.75,
        },
        'Density': {
            'iterations': 1000,
            'learning_rate': 0.07,
            'depth': 6,
            'l2_leaf_reg': 4.0,
            'bagging_temperature': 0.4,
            'random_strength': 1.0,
            'border_count': 128,
            'subsample': 0.9,
        },
        'Rg': {
            'iterations': 1600,
            'learning_rate': 0.045,
            'depth': 7,
            'l2_leaf_reg': 2.5,
            'bagging_temperature': 0.6,
            'random_strength': 2.5,
            'border_count': 128,
            'subsample': 0.8,
        }
    }
    
    # Get property-specific params or use Tg as default
    params = property_configs.get(property_name, property_configs['Tg']).copy()
    
    # Add common parameters
    params.update({
        'random_seed': seed,
        'verbose': False,
        'allow_writing_files': False,
        'task_type': 'CPU',
        'early_stopping_rounds': 100,
    })
    
    print(f"[{property_name}] Using fast CatBoost params (no optimization needed)")
    return params




def clip_by_train_quantiles(train_y: np.ndarray, preds: np.ndarray, lo=0.01, hi=0.99):
    ql, qh = np.quantile(train_y, lo), np.quantile(train_y, hi)
    return np.clip(preds, ql, qh)


# Copy these functions into your notebook to fix the NameError

def compute_wmae_weights(all_train_data, properties=['Tg', 'FFV', 'Tc', 'Density', 'Rg']):
    """
    Compute true wMAE weights for the competition metric
    w_j = (1 / sqrt(n_j)) / range_j, normalized to sum to 1
    
    This is the ACTUAL metric being optimized in the competition!
    """
    weights = {}
    raw_weights = []
    
    print("=== COMPUTING TRUE wMAE WEIGHTS ===")
    
    for prop in properties:
        # Get training data for this property
        prop_data = all_train_data[prop].dropna()
        n_j = len(prop_data)
        range_j = prop_data.max() - prop_data.min()
        
        # Compute raw weight: w_j = (1 / sqrt(n_j)) / range_j
        raw_w_j = (1.0 / np.sqrt(n_j)) / range_j
        raw_weights.append(raw_w_j)
        
        print(f"{prop}: n={n_j}, range={range_j:.4f}, raw_weight={raw_w_j:.8f}")
    
    # Normalize weights to sum to 1
    total_raw_weight = sum(raw_weights)
    normalized_weights = [w / total_raw_weight for w in raw_weights]
    
    for i, prop in enumerate(properties):
        weights[prop] = normalized_weights[i]
        print(f"{prop}: normalized_weight={normalized_weights[i]:.6f}")
    
    print(f"Weight sum check: {sum(normalized_weights):.6f} (should be 1.0)")
    
    return weights


def evaluate_weights_with_wmae(property_name: str, weights: np.ndarray, 
                              oof_lgb, oof_catboost, oof_knn, oof_xgb, 
                              y_true, groups, wmae_weight: float, n_splits=5):
    """
    Evaluate blend weights using wMAE instead of plain MAE
    This is the TRUE competition metric!
    """
    w1, w2, w3, w4 = weights
    
    # Ensure weights sum to 1 (project to simplex)
    total = w1 + w2 + w3 + w4
    if total <= 0:
        return float('inf')
    w1, w2, w3, w4 = w1/total, w2/total, w3/total, w4/total
    
    gkf = GroupKFold(n_splits=n_splits)
    cv_scores = []
    
    for train_idx, val_idx in gkf.split(oof_lgb, y_true, groups):
        # Get validation fold predictions
        lgb_val = oof_lgb[val_idx]
        cat_val = oof_catboost[val_idx]
        knn_val = oof_knn[val_idx]
        xgb_val = oof_xgb[val_idx]
        y_val = y_true[val_idx]
        
        # Create blended prediction
        blend_val = w1 * lgb_val + w2 * cat_val + w3 * knn_val + w4 * xgb_val
        
        # Calculate weighted MAE for this fold using competition weights
        mae = np.mean(np.abs(blend_val - y_val))
        weighted_mae = wmae_weight * mae  # Apply true wMAE weight
        cv_scores.append(weighted_mae)
    
    return np.mean(cv_scores)


def optimize_property_specific_quad_weights_wmae(property_name: str, 
                                               oof_lgb, oof_catboost, oof_knn, oof_xgb, 
                                               y_true, groups, wmae_weight: float,
                                               n_splits=5, n_dirichlet_samples=400, seed=42):
    """
    Property-specific 4-model blend optimization using TRUE wMAE weights
    This optimizes for the actual competition metric, not plain MAE!
    """
    np.random.seed(seed)
    
    def evaluate_weights(weights):
        """Evaluate blend weights using wMAE (TRUE competition metric)"""
        return evaluate_weights_with_wmae(
            property_name, weights, oof_lgb, oof_catboost, oof_knn, oof_xgb,
            y_true, groups, wmae_weight, n_splits
        )
    
    # Property-specific Dirichlet prior tuning (same as before)
    if property_name == 'Tg':
        alpha = np.array([3.0, 3.0, 1.5, 3.0])  # Favor trees over KNN
    elif property_name == 'Tc': 
        alpha = np.array([3.0, 3.0, 1.5, 3.0])
    elif property_name == 'FFV':
        alpha = np.array([2.5, 2.5, 3.0, 2.5])  # Favor KNN
    elif property_name == 'Density':
        alpha = np.array([2.5, 2.5, 2.0, 3.5])  # Favor XGBoost
    elif property_name == 'Rg':
        alpha = np.array([2.5, 2.5, 2.5, 2.5])  # Balanced
    else:
        alpha = np.array([2.5, 2.5, 2.5, 2.5])
    
    print(f"  [{property_name}] Using wMAE weight: {wmae_weight:.6f} | Property priors: LGBM={alpha[0]:.1f}, Cat={alpha[1]:.1f}, KNN={alpha[2]:.1f}, XGB={alpha[3]:.1f}")
    
    # Stage 1: Property-aware Dirichlet sampling with wMAE objective
    print(f"  [{property_name}] wMAE-optimized Dirichlet exploration...")
    
    best_score = float('inf')
    best_weights = None
    dirichlet_history = []
    
    for i in range(n_dirichlet_samples):
        # Sample from property-specific Dirichlet distribution
        weights = np.random.dirichlet(alpha)
        score = evaluate_weights(weights)  # This uses wMAE now!
        
        dirichlet_history.append((weights.copy(), score))
        
        if score < best_score:
            best_score = score
            best_weights = weights.copy()
    
    # Stage 2: Coordinate refinement with wMAE objective
    print(f"  [{property_name}] wMAE-optimized coordinate refinement...")
    
    # Get top 15% of candidates for refinement
    dirichlet_history.sort(key=lambda x: x[1])
    top_candidates = dirichlet_history[:max(1, int(len(dirichlet_history) * 0.15))]
    
    for start_weights, start_score in top_candidates[:7]:
        
        def objective(params):
            w1, w2, w3 = params
            w4 = 1.0 - w1 - w2 - w3
            
            if w1 < 0 or w2 < 0 or w3 < 0 or w4 < 0:
                return float('inf')
            
            return evaluate_weights([w1, w2, w3, w4])  # wMAE objective
        
        initial_params = start_weights[:3]
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        
        try:
            from scipy.optimize import minimize
            result = minimize(
                objective, initial_params, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 100}
            )
            
            if result.success and result.fun < best_score:
                refined_w1, refined_w2, refined_w3 = result.x
                refined_w4 = 1.0 - refined_w1 - refined_w2 - refined_w3
                
                if refined_w4 >= 0:
                    refined_weights = np.array([refined_w1, refined_w2, refined_w3, refined_w4])
                    verified_score = evaluate_weights(refined_weights)
                    
                    if verified_score < best_score:
                        best_score = verified_score
                        best_weights = refined_weights
                        
        except Exception:
            continue
    
    # Convert back to plain MAE for reporting (but optimization used wMAE!)
    plain_mae_score = evaluate_weights_with_wmae(
        property_name, best_weights, oof_lgb, oof_catboost, oof_knn, oof_xgb,
        y_true, groups, wmae_weight=1.0, n_splits=n_splits  # weight=1.0 gives plain MAE
    )
    
    print(f"  [{property_name}] wMAE-optimized weights complete. wMAE score: {best_score:.8f}, plain MAE: {plain_mae_score:.6f}")
    
    return best_weights, best_score, plain_mae_score


def analyze_property_specific_blend_performance(property_name: str,
                                              oof_lgb, oof_catboost, oof_knn, oof_xgb, 
                                              y_true, best_weights):
    """
    Property-specific blend analysis with insights about model contributions
    """
    w1, w2, w3, w4 = best_weights
    
    # Individual model performances
    individual_maes = {
        'LGBM': mean_absolute_error(y_true, oof_lgb),
        'CatBoost': mean_absolute_error(y_true, oof_catboost),
        'KNN': mean_absolute_error(y_true, oof_knn),
        'XGBoost': mean_absolute_error(y_true, oof_xgb)
    }
    
    # Property-specific model ranking
    model_ranking = sorted(individual_maes.items(), key=lambda x: x[1])
    best_model_name, best_model_mae = model_ranking[0]
    
    # Blend performance
    blend_pred = w1 * oof_lgb + w2 * oof_catboost + w3 * oof_knn + w4 * oof_xgb
    blend_mae = mean_absolute_error(y_true, blend_pred)
    
    # Weight analysis
    weights_dict = {'LGBM': w1, 'CatBoost': w2, 'KNN': w3, 'XGBoost': w4}
    dominant_model = max(weights_dict.items(), key=lambda x: x[1])
    
    # Property-specific insights
    if property_name == 'Tg':
        expected_pattern = "Tree models (LGBM/XGB/Cat) should dominate for glass transition temperature"
    elif property_name == 'Tc':
        expected_pattern = "Tree models should be strong for critical temperature"
    elif property_name == 'FFV':
        expected_pattern = "KNN might contribute more due to free volume structure sensitivity"
    elif property_name == 'Density':
        expected_pattern = "XGBoost often excels at density prediction tasks"
    elif property_name == 'Rg':
        expected_pattern = "Balanced ensemble often works best for radius of gyration"
    else:
        expected_pattern = "No specific pattern expected"
    
    # Check if pattern matches expectation
    tree_weight = w1 + w2 + w4  # LGBM + CatBoost + XGBoost
    knn_weight = w3
    
    analysis = {
        'property': property_name,
        'individual_maes': individual_maes,
        'model_ranking': model_ranking,
        'best_individual_model': best_model_name,
        'best_individual_mae': best_model_mae,
        'blend_mae': blend_mae,
        'blend_improvement': best_model_mae - blend_mae,
        'weights': weights_dict,
        'dominant_model': dominant_model,
        'tree_models_weight': tree_weight,
        'knn_weight': knn_weight,
        'expected_pattern': expected_pattern,
        'pattern_analysis': {
            'tree_dominant': tree_weight > 0.6,
            'knn_significant': knn_weight > 0.15,
            'balanced_ensemble': max(weights_dict.values()) < 0.4
        }
    }
    
    return analysis


# Quick test function to verify wMAE weights computation
def test_wmae_weights_computation():
    """
    Test the wMAE weights computation with sample data
    """
    print("=== TESTING wMAE WEIGHTS COMPUTATION ===")
    
    # Create sample data that mimics your training sets
    np.random.seed(42)
    sample_data = {
        'Tg': pd.Series(np.random.normal(300, 50, 1000)),      # Large dataset, wide range
        'FFV': pd.Series(np.random.normal(0.1, 0.02, 800)),    # Medium dataset, narrow range  
        'Tc': pd.Series(np.random.normal(500, 100, 600)),      # Smaller dataset, wide range
        'Density': pd.Series(np.random.normal(1.2, 0.1, 700)), # Medium dataset, narrow range
        'Rg': pd.Series(np.random.normal(10, 2, 900))          # Large dataset, medium range
    }
    
    wmae_weights = compute_wmae_weights(sample_data)
    
    print("\nExpected patterns:")
    print("- Smaller datasets should get higher weights (Tc)")
    print("- Narrower ranges should get higher weights (FFV, Density)")
    print("- Large datasets with wide ranges get lower weights (Tg)")
    
    return wmae_weights

# Run the test
print("Copy all functions above into your notebook, then run:")
print("test_wmae_weights_computation()")


# Corrected wMAE-Optimized Per-Target Pipeline - COMPLETE VERSION

import xgboost as xgb
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, ElasticNet

def compute_wmae_weights(all_train_data, properties=['Tg', 'FFV', 'Tc', 'Density', 'Rg']):
    """
    Compute true wMAE weights for the competition metric
    w_j = (1 / sqrt(n_j)) / range_j, normalized to sum to 1
    """
    weights = {}
    raw_weights = []
    
    print("=== COMPUTING TRUE wMAE WEIGHTS ===")
    
    for prop in properties:
        prop_data = all_train_data[prop].dropna()
        n_j = len(prop_data)
        range_j = prop_data.max() - prop_data.min()
        raw_w_j = (1.0 / np.sqrt(n_j)) / range_j
        raw_weights.append(raw_w_j)
        print(f"{prop}: n={n_j}, range={range_j:.4f}, raw_weight={raw_w_j:.8f}")
    
    total_raw_weight = sum(raw_weights)
    normalized_weights = [w / total_raw_weight for w in raw_weights]
    
    for i, prop in enumerate(properties):
        weights[prop] = normalized_weights[i]
        print(f"{prop}: normalized_weight={normalized_weights[i]:.6f}")
    
    print(f"Weight sum check: {sum(normalized_weights):.6f}")
    return weights


def get_fast_xgboost_params(property_name: str, seed: int = 42):
    """Fast XGBoost parameters - pre-tuned for polymer data"""
    property_configs = {
        'Tg': {'n_estimators': 1200, 'learning_rate': 0.05, 'max_depth': 7, 'subsample': 0.8,
               'colsample_bytree': 0.85, 'reg_alpha': 2.0, 'reg_lambda': 3.0, 'min_child_weight': 5, 'gamma': 0.1},
        'Tc': {'n_estimators': 1000, 'learning_rate': 0.06, 'max_depth': 6, 'subsample': 0.85,
               'colsample_bytree': 0.9, 'reg_alpha': 1.5, 'reg_lambda': 2.0, 'min_child_weight': 3, 'gamma': 0.05},
        'FFV': {'n_estimators': 1500, 'learning_rate': 0.04, 'max_depth': 8, 'subsample': 0.75,
                'colsample_bytree': 0.8, 'reg_alpha': 3.0, 'reg_lambda': 4.0, 'min_child_weight': 8, 'gamma': 0.2},
        'Density': {'n_estimators': 800, 'learning_rate': 0.08, 'max_depth': 5, 'subsample': 0.9,
                    'colsample_bytree': 0.95, 'reg_alpha': 1.0, 'reg_lambda': 1.5, 'min_child_weight': 2, 'gamma': 0.0},
        'Rg': {'n_estimators': 1300, 'learning_rate': 0.045, 'max_depth': 7, 'subsample': 0.8,
               'colsample_bytree': 0.85, 'reg_alpha': 2.5, 'reg_lambda': 3.5, 'min_child_weight': 6, 'gamma': 0.15}
    }
    
    params = property_configs.get(property_name, property_configs['Tg']).copy()
    params.update({'random_state': seed, 'verbosity': 0, 'n_jobs': -1, 'tree_method': 'auto'})
    
    print(f"[{property_name}] Using fast XGBoost params")
    return params


def evaluate_weights_with_wmae(property_name: str, weights: np.ndarray, 
                              oof_lgb, oof_catboost, oof_knn, oof_xgb, 
                              y_true, groups, wmae_weight: float, n_splits=5):
    """Evaluate blend weights using wMAE (TRUE competition metric)"""
    w1, w2, w3, w4 = weights
    
    total = w1 + w2 + w3 + w4
    if total <= 0:
        return float('inf')
    w1, w2, w3, w4 = w1/total, w2/total, w3/total, w4/total
    
    gkf = GroupKFold(n_splits=n_splits)
    cv_scores = []
    
    for train_idx, val_idx in gkf.split(oof_lgb, y_true, groups):
        lgb_val = oof_lgb[val_idx]
        cat_val = oof_catboost[val_idx]
        knn_val = oof_knn[val_idx]
        xgb_val = oof_xgb[val_idx]
        y_val = y_true[val_idx]
        
        blend_val = w1 * lgb_val + w2 * cat_val + w3 * knn_val + w4 * xgb_val
        mae = np.mean(np.abs(blend_val - y_val))
        weighted_mae = wmae_weight * mae  # Apply true wMAE weight
        cv_scores.append(weighted_mae)
    
    return np.mean(cv_scores)


def optimize_property_specific_quad_weights_wmae(property_name: str, 
                                               oof_lgb, oof_catboost, oof_knn, oof_xgb, 
                                               y_true, groups, wmae_weight: float,
                                               n_splits=5, n_dirichlet_samples=400, seed=42):
    """Property-specific 4-model blend optimization using TRUE wMAE weights"""
    np.random.seed(seed)
    
    def evaluate_weights(weights):
        return evaluate_weights_with_wmae(
            property_name, weights, oof_lgb, oof_catboost, oof_knn, oof_xgb,
            y_true, groups, wmae_weight, n_splits
        )
    
    # Property-specific Dirichlet priors
    if property_name == 'Tg':
        alpha = np.array([3.0, 3.0, 1.5, 3.0])  # Favor trees over KNN
    elif property_name == 'Tc': 
        alpha = np.array([3.0, 3.0, 1.5, 3.0])
    elif property_name == 'FFV':
        alpha = np.array([2.5, 2.5, 3.0, 2.5])  # Favor KNN
    elif property_name == 'Density':
        alpha = np.array([2.5, 2.5, 2.0, 3.5])  # Favor XGBoost
    elif property_name == 'Rg':
        alpha = np.array([2.5, 2.5, 2.5, 2.5])  # Balanced
    else:
        alpha = np.array([2.5, 2.5, 2.5, 2.5])
    
    print(f"  [{property_name}] wMAE weight: {wmae_weight:.6f} | Priors: LGBM={alpha[0]:.1f}, Cat={alpha[1]:.1f}, KNN={alpha[2]:.1f}, XGB={alpha[3]:.1f}")
    
    # Stage 1: Dirichlet sampling with wMAE objective
    best_score = float('inf')
    best_weights = None
    
    for i in range(n_dirichlet_samples):
        weights = np.random.dirichlet(alpha)
        score = evaluate_weights(weights)  # Uses wMAE!
        
        if score < best_score:
            best_score = score
            best_weights = weights.copy()
    
    # Stage 2: Coordinate refinement
    for _ in range(20):  # Multiple refinement attempts
        def objective(params):
            w1, w2, w3 = params
            w4 = 1.0 - w1 - w2 - w3
            if w1 < 0 or w2 < 0 or w3 < 0 or w4 < 0:
                return float('inf')
            return evaluate_weights([w1, w2, w3, w4])
        
        initial_params = best_weights[:3]
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        
        try:
            result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
            
            if result.success and result.fun < best_score:
                refined_w1, refined_w2, refined_w3 = result.x
                refined_w4 = 1.0 - refined_w1 - refined_w2 - refined_w3
                
                if refined_w4 >= 0:
                    refined_weights = np.array([refined_w1, refined_w2, refined_w3, refined_w4])
                    verified_score = evaluate_weights(refined_weights)
                    
                    if verified_score < best_score:
                        best_score = verified_score
                        best_weights = refined_weights
        except:
            continue
    
    # Convert to plain MAE for reporting
    plain_mae_score = evaluate_weights_with_wmae(
        property_name, best_weights, oof_lgb, oof_catboost, oof_knn, oof_xgb,
        y_true, groups, wmae_weight=1.0, n_splits=n_splits
    )
    
    print(f"  [{property_name}] wMAE-optimized complete. wMAE score: {best_score:.8f}, plain MAE: {plain_mae_score:.6f}")
    
    return best_weights, best_score, plain_mae_score


def analyze_property_specific_blend_performance(property_name: str,
                                              oof_lgb, oof_catboost, oof_knn, oof_xgb, 
                                              y_true, best_weights):
    """Property-specific blend analysis"""
    w1, w2, w3, w4 = best_weights
    
    individual_maes = {
        'LGBM': mean_absolute_error(y_true, oof_lgb),
        'CatBoost': mean_absolute_error(y_true, oof_catboost),
        'KNN': mean_absolute_error(y_true, oof_knn),
        'XGBoost': mean_absolute_error(y_true, oof_xgb)
    }
    
    model_ranking = sorted(individual_maes.items(), key=lambda x: x[1])
    best_model_name, best_model_mae = model_ranking[0]
    
    blend_pred = w1 * oof_lgb + w2 * oof_catboost + w3 * oof_knn + w4 * oof_xgb
    blend_mae = mean_absolute_error(y_true, blend_pred)
    
    weights_dict = {'LGBM': w1, 'CatBoost': w2, 'KNN': w3, 'XGBoost': w4}
    dominant_model = max(weights_dict.items(), key=lambda x: x[1])
    
    # Property-specific insights
    expected_patterns = {
        'Tg': "Tree models should dominate for glass transition temperature",
        'Tc': "Tree models should be strong for critical temperature", 
        'FFV': "KNN might contribute more due to structure-sensitive free volume",
        'Density': "XGBoost often excels at density prediction",
        'Rg': "Balanced ensemble often works best for radius of gyration"
    }
    expected_pattern = expected_patterns.get(property_name, "No specific pattern expected")
    
    tree_weight = w1 + w2 + w4  # LGBM + CatBoost + XGBoost
    knn_weight = w3
    
    analysis = {
        'property': property_name,
        'individual_maes': individual_maes,
        'model_ranking': model_ranking,
        'best_individual_model': best_model_name,
        'best_individual_mae': best_model_mae,
        'blend_mae': blend_mae,
        'blend_improvement': best_model_mae - blend_mae,
        'weights': weights_dict,
        'dominant_model': dominant_model,
        'tree_models_weight': tree_weight,
        'knn_weight': knn_weight,
        'expected_pattern': expected_pattern,
        'pattern_analysis': {
            'tree_dominant': tree_weight > 0.6,
            'knn_significant': knn_weight > 0.15,
            'balanced_ensemble': max(weights_dict.values()) < 0.4
        }
    }
    
    return analysis


def stack_with_other_targets_selective_quad(base_results: dict,
                                          property_name: str,
                                          test_df: pd.DataFrame,
                                          n_splits: int = 5,
                                          seed: int = 42):
    """
    Selective stacking for 4-model results (Ridge vs ElasticNet)
    """
    # This target's base
    base = base_results[property_name]
    y = base['train_y']
    can_self = base['canon']
    self_oof = base['oof_blend']
    self_test = base['test_blend']

    # Build aligned OOF features from other targets
    other_props = [p for p in ['Tg','FFV','Tc','Density','Rg'] if p != property_name and p in base_results]
    oof_mat = [self_oof]
    test_mat = [self_test]

    for p in other_props:
        other = base_results[p]
        mapping = dict(zip(other['canon'], other['oof_blend']))
        mean_val = float(np.mean(other['oof_blend'])) if len(other['oof_blend']) else 0.0
        aligned = np.array([mapping.get(c, mean_val) for c in can_self])
        oof_mat.append(aligned)
        test_mat.append(other['test_blend'])

    Z = np.column_stack(oof_mat)
    Zte = np.column_stack(test_mat)

    df_self = base['df']
    
    # Try Ridge
    oof_ridge = np.zeros_like(y, dtype=float)
    preds_ridge = np.zeros(len(test_df))
    
    for tr, va, groups in get_group_folds(df_self, n_splits=n_splits):
        ridge = Ridge(alpha=1.0, random_state=seed)
        ridge.fit(Z[tr], y[tr])
        oof_ridge[va] = ridge.predict(Z[va])
        preds_ridge += ridge.predict(Zte)
    
    preds_ridge /= n_splits
    cv_mae_ridge = mean_absolute_error(y, oof_ridge)
    
    # Try ElasticNet
    oof_elastic = np.zeros_like(y, dtype=float)
    preds_elastic = np.zeros(len(test_df))
    
    for tr, va, groups in get_group_folds(df_self, n_splits=n_splits):
        elastic = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=seed, max_iter=2000)
        elastic.fit(Z[tr], y[tr])
        oof_elastic[va] = elastic.predict(Z[va])
        preds_elastic += elastic.predict(Zte)
    
    preds_elastic /= n_splits  
    cv_mae_elastic = mean_absolute_error(y, oof_elastic)
    
    # Pick the better one
    if cv_mae_elastic < cv_mae_ridge:
        cv_mae_meta = cv_mae_elastic
        preds_te = preds_elastic
        meta_type = "ElasticNet"
    else:
        cv_mae_meta = cv_mae_ridge  
        preds_te = preds_ridge
        meta_type = "Ridge"

    # Choose better of base vs best meta
    use_meta = cv_mae_meta <= base['cv_mae']
    final_cv = min(cv_mae_meta, base['cv_mae'])
    final_test = preds_te if use_meta else self_test

    return final_cv, final_test, {
        'meta_cv': cv_mae_meta,
        'ridge_cv': cv_mae_ridge,
        'elastic_cv': cv_mae_elastic,
        'base_cv': base['cv_mae'], 
        'used_meta': use_meta,
        'meta_type': meta_type
    }


def train_xgboost_quad_hybrid_property_model_per_target_wmae(train_df,
                                                           supplements: dict,
                                                           modred: dict,
                                                           property_name: str,
                                                           test_df,
                                                           wmae_weight: float,
                                                           n_splits: int = 5,
                                                           seed: int = 42,
                                                           n_trials: int = 50):
    """Train 4-model ensemble with wMAE-optimized per-target blending"""
    
    # Build dataset
    df = prepare_data_for_property(train_df, supplements, property_name, modred).copy()
    df = df[['SMILES', property_name]].dropna().drop_duplicates(subset='SMILES').reset_index(drop=True)
    
    print(f"\n[{property_name}] Training samples: {len(df)}")
    
    # Property-specific features
    X_all = featurize_series_hybrid(df['SMILES'], property_name)
    X_bits = X_all[:, :2*N_BITS]  # For KNN
    y = df[property_name].values
    
    # Test features
    Xte_all = featurize_series_hybrid(test_df['SMILES'], property_name)
    Xte_bits = Xte_all[:, :2*N_BITS]
    
    feature_approach = "Optuna-only (10 desc)" if property_name in ['Tg', 'Tc'] else "Conservative (25 desc)"
    print(f"[{property_name}] Using {feature_approach}: {X_all.shape[1]} features")
    
    # Get groups for optimization
    groups = make_groups(df)
    
    # Optimize LGBM parameters
    best_lgbm_params, best_lgbm_score = optimize_lgbm_params(
        X_all, y, groups, n_trials=n_trials, n_splits=n_splits, 
        seed=seed, property_name=property_name
    )
    
    # Get fast parameters
    best_catboost_params = get_fast_catboost_params(property_name, seed)
    best_xgboost_params = get_fast_xgboost_params(property_name, seed)
    
    # Train all 4 models using GroupKFold
    oof_lgb = np.zeros(len(df))
    oof_catboost = np.zeros(len(df))
    oof_knn = np.zeros(len(df))
    oof_xgb = np.zeros(len(df))
    
    preds_te_lgb = np.zeros(len(test_df))
    preds_te_catboost = np.zeros(len(test_df))
    preds_te_knn = np.zeros(len(test_df))
    preds_te_xgb = np.zeros(len(test_df))
    
    fold_num = 0
    for tr, va, groups_fold in get_group_folds(df, n_splits=n_splits):
        fold_num += 1
        Xtr, Xva = X_all[tr], X_all[va]
        ytr, yva = y[tr], y[va]
        Xtr_bits, Xva_bits = X_bits[tr], X_bits[va]
        
        # LGBM
        lgb_params = best_lgbm_params.copy()
        lgb_params['random_state'] = seed + fold_num
        
        lgb = LGBMRegressor(**lgb_params)
        lgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric='mae',
                callbacks=[early_stopping(stopping_rounds=100)])
        oof_lgb[va] = lgb.predict(Xva)
        preds_te_lgb += lgb.predict(Xte_all)
        
        # CatBoost
        cat_params = best_catboost_params.copy()
        cat_params['random_seed'] = seed + fold_num
        
        catboost = CatBoostRegressor(**cat_params)
        catboost.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=False)
        oof_catboost[va] = catboost.predict(Xva)
        preds_te_catboost += catboost.predict(Xte_all)
        
        # XGBoost
        xgb_params = best_xgboost_params.copy()
        xgb_params['random_state'] = seed + fold_num
        
        xgboost = xgb.XGBRegressor(**xgb_params)
        xgboost.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        oof_xgb[va] = xgboost.predict(Xva)
        preds_te_xgb += xgboost.predict(Xte_all)
        
        # KNN
        knn = KNeighborsRegressor(n_neighbors=7, metric='cosine', algorithm='brute', weights='distance')
        knn.fit(Xtr_bits, ytr)
        oof_knn[va] = knn.predict(Xva_bits)
        preds_te_knn += knn.predict(Xte_bits)
    
    # Average test predictions
    preds_te_lgb /= n_splits
    preds_te_catboost /= n_splits
    preds_te_knn /= n_splits
    preds_te_xgb /= n_splits
    
    # Optimize quad blend weights using wMAE-optimized method
    print(f"[{property_name}] Optimizing with TRUE wMAE objective...")
    best_weights, best_wmae_score, plain_mae_score = optimize_property_specific_quad_weights_wmae(
        property_name, oof_lgb, oof_catboost, oof_knn, oof_xgb, y, groups, wmae_weight,
        n_splits=n_splits, n_dirichlet_samples=400, seed=seed
    )
    
    lgb_w, cat_w, knn_w, xgb_w = best_weights
    print(f"[{property_name}] wMAE-optimized weights: {lgb_w:.4f} LGBM + {cat_w:.4f} CatBoost + {knn_w:.4f} KNN + {xgb_w:.4f} XGBoost")
    
    # Analyze blend performance
    blend_analysis = analyze_property_specific_blend_performance(
        property_name, oof_lgb, oof_catboost, oof_knn, oof_xgb, y, best_weights
    )
    
    # Create final predictions
    oof_blend = lgb_w * oof_lgb + cat_w * oof_catboost + knn_w * oof_knn + xgb_w * oof_xgb
    preds_te = lgb_w * preds_te_lgb + cat_w * preds_te_catboost + knn_w * preds_te_knn + xgb_w * preds_te_xgb
    
    cv_mae = mean_absolute_error(y, oof_blend)
    canon = make_groups(df)
    
    # Individual model MAEs
    lgb_mae = mean_absolute_error(y, oof_lgb)
    cat_mae = mean_absolute_error(y, oof_catboost) 
    knn_mae = mean_absolute_error(y, oof_knn)
    xgb_mae = mean_absolute_error(y, oof_xgb)
    
    print(f"[{property_name}] Individual MAEs - LGBM: {lgb_mae:.4f}, CatBoost: {cat_mae:.4f}, KNN: {knn_mae:.4f}, XGBoost: {xgb_mae:.4f}")
    print(f"[{property_name}] Final blend MAE: {cv_mae:.4f}")
    
    return {
        'cv_mae': cv_mae,
        'oof_blend': oof_blend,
        'test_blend': preds_te,
        'train_y': y,
        'canon': canon,
        'df': df,
        'best_lgbm_params': best_lgbm_params,
        'best_catboost_params': best_catboost_params,
        'best_xgboost_params': best_xgboost_params,
        'lgbm_optimization_score': best_lgbm_score,
        'feature_approach': feature_approach,
        'best_blend_weights': best_weights,
        'blend_analysis': blend_analysis,
        'blend_mae': plain_mae_score,
        'blend_wmae_score': best_wmae_score,
        'wmae_weight': wmae_weight,
        'property_specific_analysis': blend_analysis,
        'individual_maes': {'lgbm': lgb_mae, 'catboost': cat_mae, 'knn': knn_mae, 'xgboost': xgb_mae}
    }


def run_xgboost_per_target_wmae_pipeline(n_trials=50):
    """
    wMAE-OPTIMIZED PER-TARGET 4-MODEL Pipeline
    CRITICAL: Optimizes for the actual competition metric (wMAE), not plain MAE!
    """
    
    print("=== wMAE-OPTIMIZED PER-TARGET 4-MODEL PIPELINE ===")
    print("CRITICAL: Now optimizes TRUE wMAE metric, not plain MAE!")
    
    PROPERTIES = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    supplements = {'supp1': supp1, 'supp2': supp2, 'supp3': supp3, 'supp4': supp4}
    
    # STEP 1: Compute TRUE wMAE weights
    print("\nSTEP 1: Computing true wMAE weights...")
    all_train_data = {}
    for prop in PROPERTIES:
        df = prepare_data_for_property(train, supplements, property_name=prop, modred_data=modred)
        all_train_data[prop] = df[prop]
    
    wmae_weights = compute_wmae_weights(all_train_data, PROPERTIES)
    
    print(f"\n=== TRUE wMAE WEIGHTS ===")
    for prop in PROPERTIES:
        print(f"{prop}: {wmae_weights[prop]:.6f}")
    
    # STEP 2: Train models with wMAE optimization
    print(f"\nSTEP 2: Training models with wMAE optimization...")
    base_results_wmae = {}
    
    for prop in PROPERTIES:
        res = train_xgboost_quad_hybrid_property_model_per_target_wmae(
            train, supplements, modred, prop, test, wmae_weights[prop],
            n_splits=5, seed=42, n_trials=n_trials
        )
        base_results_wmae[prop] = res
        
        lgb_w, cat_w, knn_w, xgb_w = res['best_blend_weights']
        analysis = res['property_specific_analysis']
        
        print(f"\n[wMAE-OPTIMIZED] {prop}: Plain MAE = {res['blend_mae']:.6f}, wMAE score = {res['blend_wmae_score']:.8f}")
        print(f"    wMAE weight = {wmae_weights[prop]:.6f}")
        print(f"    Optimized weights: {lgb_w:.4f} LGBM + {cat_w:.4f} CatBoost + {knn_w:.4f} KNN + {xgb_w:.4f} XGBoost")
        print(f"    Dominant: {analysis['dominant_model'][0]} ({analysis['dominant_model'][1]:.3f})")
    
    # STEP 3: Compute total wMAE score
    total_wmae_score = sum(base_results_wmae[prop]['blend_wmae_score'] for prop in PROPERTIES)
    total_plain_mae = sum(base_results_wmae[prop]['blend_mae'] for prop in PROPERTIES)
    
    print(f"\n=== TOTAL COMPETITION SCORES ===")
    print(f"Total wMAE score: {total_wmae_score:.8f} (THIS IS THE COMPETITION METRIC!)")
    print(f"Average plain MAE: {total_plain_mae/len(PROPERTIES):.6f}")
    
    # STEP 4: Selective stacking
    final_preds_wmae = {}
    cv_report_wmae = {}
    
    for prop in PROPERTIES:
        cv_best, test_pred, info = stack_with_other_targets_selective_quad(
            base_results_wmae, prop, test, n_splits=5, seed=42
        )
        test_pred = clip_by_train_quantiles(base_results_wmae[prop]['train_y'], test_pred)
        final_preds_wmae[prop] = test_pred
        cv_report_wmae[prop] = info
    
    # Build submission
    submission_wmae = test[['id']].copy()
    for prop in PROPERTIES:
        submission_wmae[prop] = final_preds_wmae[prop]
    
    print("\nwMAE-optimized submission preview:")
    print(submission_wmae.head())
    
    submission_wmae.to_csv('submission.csv', index=False)
    print("\nS# Corrected wMAE-Optimized Per-Target Pipeline")


# Usage
print("=== USAGE ===")
print("wmae_results = run_xgboost_per_target_wmae_pipeline(n_trials=70)")
print("final_score = compute_final_wmae_from_results(wmae_results[0], wmae_results[3], PROPERTIES)")


# Run the wMAE-optimized pipeline
wmae_results = run_xgboost_per_target_wmae_pipeline(n_trials=70)

# Extract results
base_results, final_preds, cv_reports, wmae_weights, patterns = wmae_results

# Compute final competition score
final_wmae = compute_final_wmae_from_results(base_results, wmae_weights, PROPERTIES)

