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


# !pip install rdkit lightgbm catboost xgboost


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, Fragments, Lipinski, GraphDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import time
import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, Crippen, Lipinski, rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import time
import warnings
warnings.filterwarnings('ignore')

print(f"Starting advanced polymer property prediction pipeline...")
print(f"Date: 2025-06-23")
print(f"User: Kowshik Debanath")
start_time = time.time()

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Load data
train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Check for missing values
print("\nMissing values in train data:")
missing_values = train.isnull().sum()
print(missing_values)

# Calculate available samples for each property
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
available_samples = {col: train.shape[0] - missing_values[col] for col in target_cols}
print("\nAvailable samples for each property:")
for col, count in available_samples.items():
    print(f"{col}: {count} ({count/train.shape[0]*100:.2f}%)")

# Calculate property ranges (needed for wMAE calculation)
property_ranges = {}
for col in target_cols:
    property_ranges[col] = train[col].dropna().max() - train[col].dropna().min()
    
print("\nEstimated property ranges:")
for col, range_val in property_ranges.items():
    print(f"{col}: {range_val:.4f}")

# Calculate weights for wMAE metric
weights = {}
for col in target_cols:
    # Weight = (1/sqrt(n_i)) / range_i, normalized
    weights[col] = (1 / np.sqrt(available_samples[col])) / property_ranges[col]

# Normalize weights to sum to number of tasks (5)
weight_sum = sum(weights.values())
for col in target_cols:
    weights[col] = weights[col] / weight_sum * len(target_cols)

print("\nEstimated weights for wMAE metric:")
for col, weight in weights.items():
    print(f"{col}: {weight:.4f}")

# Enhanced polymer-specific feature extraction
def extract_polymer_features(mol):
    """Extract polymer-specific features beyond standard RDKit descriptors"""
    features = {}
    
    # Basic physical properties
    features['mw'] = Descriptors.MolWt(mol)
    features['exact_mw'] = Descriptors.ExactMolWt(mol)
    features['logp'] = Descriptors.MolLogP(mol)
    features['mr'] = Descriptors.MolMR(mol)
    features['tpsa'] = Descriptors.TPSA(mol)
    features['heavy_atom_count'] = mol.GetNumHeavyAtoms()
    features['atom_count'] = mol.GetNumAtoms()
    features['fraction_csp3'] = Descriptors.FractionCSP3(mol)
    
    # Structural features
    features['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
    features['num_h_donors'] = Descriptors.NumHDonors(mol)
    features['num_h_acceptors'] = Descriptors.NumHAcceptors(mol)
    features['num_heteroatoms'] = Descriptors.NumHeteroatoms(mol)
    features['num_valence_electrons'] = Descriptors.NumValenceElectrons(mol)
    
    # Ring information
    features['num_rings'] = rdMolDescriptors.CalcNumRings(mol)
    features['num_aromatic_rings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
    features['num_aliphatic_rings'] = rdMolDescriptors.CalcNumAliphaticRings(mol)
    features['num_saturated_rings'] = rdMolDescriptors.CalcNumSaturatedRings(mol)
    
    # Graph theoretical indices
    features['balaban_j'] = Descriptors.BalabanJ(mol) if mol.GetNumHeavyAtoms() > 1 else 0
    features['bertz_ct'] = Descriptors.BertzCT(mol)
    
    # Polymer-specific ratios
    if mol.GetNumHeavyAtoms() > 0:
        features['rotatable_per_heavy'] = features['num_rotatable_bonds'] / features['heavy_atom_count']
        features['rings_per_heavy'] = features['num_rings'] / features['heavy_atom_count']
        features['heteroatoms_per_heavy'] = features['num_heteroatoms'] / features['heavy_atom_count']
        features['aromatic_ratio'] = features['num_aromatic_rings'] / (features['num_rings'] + 0.001)
        features['sp3_per_total'] = features['fraction_csp3'] * features['heavy_atom_count']
    else:
        features['rotatable_per_heavy'] = 0
        features['rings_per_heavy'] = 0
        features['heteroatoms_per_heavy'] = 0
        features['aromatic_ratio'] = 0
        features['sp3_per_total'] = 0
    
    # Count atom types
    atom_types = {'C': 0, 'N': 0, 'O': 0, 'S': 0, 'F': 0, 'Cl': 0, 'Br': 0, 'I': 0, 'P': 0, 'Si': 0}
    aromatic_count = 0
    sp3_count = 0
    sp2_count = 0
    sp_count = 0
    
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in atom_types:
            atom_types[symbol] += 1
        
        if atom.GetIsAromatic():
            aromatic_count += 1
        
        hybrid = atom.GetHybridization()
        if hybrid == Chem.rdchem.HybridizationType.SP3:
            sp3_count += 1
        elif hybrid == Chem.rdchem.HybridizationType.SP2:
            sp2_count += 1
        elif hybrid == Chem.rdchem.HybridizationType.SP:
            sp_count += 1
    
    # Store counts and ratios
    features['aromatic_atoms'] = aromatic_count
    features['sp3_atoms'] = sp3_count
    features['sp2_atoms'] = sp2_count
    features['sp_atoms'] = sp_count
    
    # Add atom counts to features
    for atom_type, count in atom_types.items():
        features[f'num_{atom_type}'] = count
        
        # Calculate ratios if there are heavy atoms
        if mol.GetNumHeavyAtoms() > 0:
            features[f'ratio_{atom_type}'] = count / mol.GetNumHeavyAtoms()
        else:
            features[f'ratio_{atom_type}'] = 0
    
    # Bond information
    if mol.GetNumBonds() > 0:
        bond_types = {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 0,
            Chem.rdchem.BondType.TRIPLE: 0,
            Chem.rdchem.BondType.AROMATIC: 0
        }
        
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            if bond_type in bond_types:
                bond_types[bond_type] += 1
        
        features['single_bonds'] = bond_types[Chem.rdchem.BondType.SINGLE]
        features['double_bonds'] = bond_types[Chem.rdchem.BondType.DOUBLE]
        features['triple_bonds'] = bond_types[Chem.rdchem.BondType.TRIPLE]
        features['aromatic_bonds'] = bond_types[Chem.rdchem.BondType.AROMATIC]
        
        # Calculate ratios
        total_bonds = mol.GetNumBonds()
        features['single_bond_ratio'] = bond_types[Chem.rdchem.BondType.SINGLE] / total_bonds
        features['double_bond_ratio'] = bond_types[Chem.rdchem.BondType.DOUBLE] / total_bonds
        features['triple_bond_ratio'] = bond_types[Chem.rdchem.BondType.TRIPLE] / total_bonds
        features['aromatic_bond_ratio'] = bond_types[Chem.rdchem.BondType.AROMATIC] / total_bonds
    else:
        features['single_bonds'] = 0
        features['double_bonds'] = 0
        features['triple_bonds'] = 0
        features['aromatic_bonds'] = 0
        features['single_bond_ratio'] = 0
        features['double_bond_ratio'] = 0
        features['triple_bond_ratio'] = 0
        features['aromatic_bond_ratio'] = 0
    
    # Additional polymer-specific features
    try:
        # Approximate chain length
        chain_length = features['mw'] / (14 * max(1, features['num_C']))  # Based on typical C-C chain
        features['est_chain_length'] = chain_length
        
        # Branching index (based on C atom count and total heavy atoms)
        features['branching_index'] = features['num_C'] / features['heavy_atom_count'] if features['heavy_atom_count'] > 0 else 0
        
        # Heteroatom diversity
        hetero_types = sum(1 for atom_type in ['N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'] if atom_types[atom_type] > 0)
        features['heteroatom_diversity'] = hetero_types
        
        # Functional group indicators
        features['has_amide'] = 1 if atom_types['N'] > 0 and atom_types['O'] > 0 else 0
        features['has_ester'] = 1 if atom_types['O'] > 1 else 0
        features['has_ether'] = 1 if atom_types['O'] > 0 else 0
        features['has_halogen'] = 1 if any(atom_types[x] > 0 for x in ['F', 'Cl', 'Br', 'I']) else 0
    except:
        features['est_chain_length'] = 0
        features['branching_index'] = 0
        features['heteroatom_diversity'] = 0
        features['has_amide'] = 0
        features['has_ester'] = 0
        features['has_ether'] = 0
        features['has_halogen'] = 0
    
    return features

# Comprehensive feature generation
def generate_comprehensive_features(smiles_list):
    """Generate comprehensive features for polymer property prediction"""
    # Initialize RDKit descriptor calculator
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    
    features_list = []
    valid_indices = []
    
    print("Processing SMILES strings...")
    for i, smiles in enumerate(smiles_list):
        try:
            # Multiple ways to parse SMILES for better success rate
            mol = None
            
            # Method 1: Standard parsing
            mol = Chem.MolFromSmiles(smiles)
            
            # Method 2: Handle polymer SMILES with asterisks
            if mol is None and '*' in smiles:
                modified_smiles = smiles.replace('*', 'C')
                mol = Chem.MolFromSmiles(modified_smiles)
            
            # Method 3: Try another replacement for asterisks
            if mol is None and '*' in smiles:
                modified_smiles = smiles.replace('*', '[H]')
                mol = Chem.MolFromSmiles(modified_smiles)
            
            # Method 4: Try without sanitizing
            if mol is None:
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
                if mol is not None:
                    try:
                        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    except:
                        pass
            
            if mol is not None:
                # 1. Get RDKit descriptors
                try:
                    descriptors = list(descriptor_calculator.CalcDescriptors(mol))
                except:
                    # If descriptor calculation fails, use zeros
                    descriptors = [0] * len(Descriptors._descList)
                
                # 2. Get polymer-specific features
                try:
                    polymer_features = list(extract_polymer_features(mol).values())
                except:
                    # If polymer feature extraction fails, use zeros
                    polymer_features = [0] * 80  # Approximate number of polymer features
                
                # 3. Generate fingerprints
                try:
                    # Morgan fingerprints (ECFP4)
                    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    morgan_features = np.zeros((1024,))
                    AllChem.DataStructs.ConvertToNumpyArray(morgan_fp, morgan_features)
                    
                    # MACCS keys
                    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
                    maccs_features = np.zeros((167,))
                    AllChem.DataStructs.ConvertToNumpyArray(maccs_fp, maccs_features)
                    
                    # Atom pairs
                    pairs_fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=512)
                    pairs_features = np.zeros((512,))
                    AllChem.DataStructs.ConvertToNumpyArray(pairs_fp, pairs_features)
                except:
                    # If fingerprint generation fails, use zeros
                    morgan_features = np.zeros(1024)
                    maccs_features = np.zeros(167)
                    pairs_features = np.zeros(512)
                
                # 4. Combine all features
                # Use fewer bits from Morgan to reduce dimensionality
                all_features = (
                    descriptors + 
                    polymer_features + 
                    list(morgan_features[:256]) + 
                    list(maccs_features) +
                    list(pairs_features[:128])
                )
                
                features_list.append(all_features)
                valid_indices.append(i)
            else:
                if i < 5:  # Print first few failures for debugging
                    print(f"Warning: Could not parse SMILES: {smiles}")
        except Exception as e:
            if i < 5:  # Print first few failures for debugging
                print(f"Error processing SMILES {smiles}: {str(e)}")
    
    print(f"Successfully processed {len(valid_indices)} out of {len(smiles_list)} SMILES strings")
    
    # Create feature names
    descriptor_names = [x[0] for x in Descriptors._descList]
    
    # Approximate polymer feature names
    polymer_feature_names = list(extract_polymer_features(Chem.MolFromSmiles('CC')).keys())
    
    # Fingerprint feature names
    morgan_names = [f'morgan_{i}' for i in range(256)]
    maccs_names = [f'maccs_{i}' for i in range(167)]
    pairs_names = [f'pairs_{i}' for i in range(128)]
    
    feature_names = descriptor_names + polymer_feature_names + morgan_names + maccs_names + pairs_names
    
    # Check if we have any features
    if not features_list:
        print("WARNING: No features generated. Creating fallback features.")
        # Create fallback features with random values
        np.random.seed(SEED)
        features_list = [np.random.random(len(feature_names))]
        valid_indices = [0]
    
    # Ensure all feature vectors have the same length
    expected_length = len(feature_names)
    for i, feat in enumerate(features_list):
        if len(feat) != expected_length:
            if len(feat) < expected_length:
                features_list[i] = feat + [0] * (expected_length - len(feat))
            else:
                features_list[i] = feat[:expected_length]
    
    return pd.DataFrame(features_list, index=valid_indices, columns=feature_names), valid_indices

print("\nGenerating comprehensive molecular features...")
train_features, train_valid_idx = generate_comprehensive_features(train['SMILES'])
test_features, test_valid_idx = generate_comprehensive_features(test['SMILES'])

# Join features with original data
train_with_features = pd.concat([
    train.iloc[train_valid_idx].reset_index(drop=True),
    train_features.reset_index(drop=True)
], axis=1)

test_with_features = pd.concat([
    test.iloc[test_valid_idx].reset_index(drop=True),
    test_features.reset_index(drop=True)
], axis=1)

print(f"Train with features shape: {train_with_features.shape}")
print(f"Test with features shape: {test_with_features.shape}")

# Enhanced cleaning and feature selection
def clean_and_select_features(df, feature_cols):
    """Clean features and select valid ones with improved handling"""
    df_clean = df.copy()
    
    # Replace infinity and NaN
    df_clean[feature_cols] = df_clean[feature_cols].replace([np.inf, -np.inf], np.nan)
    
    # Select valid columns (not all NaN and have some variation)
    valid_cols = []
    for col in feature_cols:
        if df_clean[col].notna().sum() > 0 and df_clean[col].nunique() > 1:
            valid_cols.append(col)
    
    print(f"Retained {len(valid_cols)} out of {len(feature_cols)} features after validation")
    
    # If we don't have enough valid columns, try to recover more
    if len(valid_cols) < 50:
        print("Few valid features, attempting to recover more...")
        for col in feature_cols:
            if col not in valid_cols and df_clean[col].notna().sum() > 0:
                valid_cols.append(col)
        print(f"Recovered to {len(valid_cols)} features")
    
    # Handle outliers using robust methods
    for col in valid_cols:
        if df_clean[col].dtype.kind in 'fc':  # Only numerical columns
            # Use quantiles for clipping to avoid extreme outliers
            q_low = df_clean[col].quantile(0.001)
            q_high = df_clean[col].quantile(0.999)
            df_clean[col] = df_clean[col].clip(lower=q_low, upper=q_high)
    
    # Fill remaining NaN values with median
    for col in valid_cols:
        if df_clean[col].isna().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
    
    return df_clean, valid_cols

# Get feature columns
feature_cols = train_features.columns.tolist()
print(f"Total number of features: {len(feature_cols)}")

# Clean and select features
train_with_features, valid_feature_cols = clean_and_select_features(train_with_features, feature_cols)
test_with_features, _ = clean_and_select_features(test_with_features, feature_cols)

# Feature selection with correlation and variance analysis
def advanced_feature_selection(df, target_col, all_features, max_features=300):
    """Select features using correlation and variance analysis"""
    df_valid = df[df[target_col].notna()].copy()
    
    if len(df_valid) < 20:
        print(f"Not enough data for {target_col}, using all features")
        return all_features[:min(len(all_features), max_features)]
    
    # 1. Calculate correlation with target
    correlations = []
    for col in all_features:
        if col in df_valid.columns:
            corr = abs(df_valid[col].corr(df_valid[target_col]))
            if not pd.isna(corr):
                correlations.append((col, corr))
    
    # Sort by correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # 2. Calculate variance for each feature
    variances = []
    for col in all_features:
        if col in df_valid.columns:
            var = df_valid[col].var()
            if not pd.isna(var) and var > 0:
                variances.append((col, var))
    
    # Sort by variance
    variances.sort(key=lambda x: x[1], reverse=True)
    
    # 3. Combine correlation and variance
    # Take top 75% by correlation and top 50% by variance
    corr_cutoff = len(correlations) * 3 // 4
    var_cutoff = len(variances) // 2
    
    top_corr = set([x[0] for x in correlations[:corr_cutoff]])
    top_var = set([x[0] for x in variances[:var_cutoff]])
    
    # Combined features (prioritize correlation)
    combined_features = list(top_corr) + [f for f in top_var if f not in top_corr]
    
    # Limit to max_features
    selected_features = combined_features[:max_features]
    
    print(f"Selected {len(selected_features)} features for {target_col} using advanced selection")
    return selected_features

# Select features for each target
target_features = {}
for col in target_cols:
    print(f"\nSelecting features for {col}...")
    target_features[col] = advanced_feature_selection(train_with_features, col, valid_feature_cols)

# Robust ensemble for optimal predictions
class RobustEnsemble:
    def __init__(self, models, transformers, feature_subsets=None, fallback_value=0):
        self.models = models
        self.transformers = transformers  # List of (scaler, imputer) tuples
        self.feature_subsets = feature_subsets  # List of feature subset lists
        self.fallback_value = fallback_value
        
    def predict(self, X):
        all_preds = []
        
        for i, (model, (scaler, imputer)) in enumerate(zip(self.models, self.transformers)):
            try:
                # Apply feature subset if available
                if self.feature_subsets and i < len(self.feature_subsets):
                    X_subset = X[self.feature_subsets[i]]
                else:
                    X_subset = X
                
                # Apply transformations
                X_imputed = imputer.transform(X_subset)
                X_scaled = scaler.transform(X_imputed)
                
                # Get predictions
                preds = model.predict(X_scaled)
                all_preds.append(preds)
            except Exception as e:
                print(f"Model {i} prediction failed: {str(e)}")
        
        # If we have any predictions, average them
        if all_preds:
            # Weight models equally
            stacked_preds = np.column_stack(all_preds)
            return np.mean(stacked_preds, axis=1)
        else:
            # Fallback to default value
            return np.full(len(X), self.fallback_value)

# Fallback values based on property-specific knowledge
fallback_values = {
    'Tg': 400,      # Glass transition temperature in K
    'FFV': 0.2,     # Fractional free volume (dimensionless)
    'Tc': 0.2,      # Critical temperature in K
    'Density': 1.2, # Density in g/cm³
    'Rg': 10.0      # Radius of gyration in Å
}

# Enhanced training with optimized models
def train_optimized_models(df, target_col, feature_cols, n_splits=5, n_repeats=2):
    """Train optimized models for polymer property prediction"""
    # Filter valid rows for this target
    valid_idx = df[target_col].notna()
    df_valid = df.loc[valid_idx]
    
    # Check if we have enough data
    if len(df_valid) < 20:
        print(f"Not enough data for {target_col}, using fallback value")
        return {'model': RobustEnsemble([], [], [], fallback_values[target_col]), 'cv_score': 0.0}
    
    # Prepare data
    X = df_valid[feature_cols].values
    y = df_valid[target_col].values
    
    print(f"Training optimized models on {len(y)} samples for {target_col}")
    
    # Initialize for ensemble
    all_models = []
    all_transformers = []
    all_feature_subsets = []
    all_scores = []
    oof_preds = np.zeros(len(y))
    fold_count = 0
    
    # Set up cross-validation
    n_splits_adjusted = min(n_splits, max(2, len(y) // 20))
    if len(y) >= 100:
        cv = RepeatedKFold(n_splits=n_splits_adjusted, n_repeats=n_repeats, random_state=SEED)
    else:
        cv = KFold(n_splits=n_splits_adjusted, shuffle=True, random_state=SEED)
    
    # Select appropriate hyperparameters based on target and dataset size
    if target_col == 'FFV' and len(y) > 1000:
        # FFV has the most data, use more complex models
        xgb_params = {
            'n_estimators': 2000,
            'learning_rate': 0.005,
            'max_depth': 7,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 3,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5,
            'random_state': SEED
        }
        
        lgb_params = {
            'n_estimators': 2000,
            'learning_rate': 0.005,
            'num_leaves': 80,
            'max_depth': 8,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5,
            'random_state': SEED
        }
        
        cat_params = {
            'iterations': 2000,
            'learning_rate': 0.01,
            'depth': 7,
            'l2_leaf_reg': 2,
            'random_seed': SEED,
            'verbose': False
        }
    elif target_col == 'Tc':
        # Tc has high weight, optimize for it
        xgb_params = {
            'n_estimators': 1500,
            'learning_rate': 0.01,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.75,
            'min_child_weight': 2,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': SEED
        }
        
        lgb_params = {
            'n_estimators': 1500,
            'learning_rate': 0.01,
            'num_leaves': 50,
            'max_depth': 7,
            'subsample': 0.8,
            'colsample_bytree': 0.75,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': SEED
        }
        
        cat_params = {
            'iterations': 1500,
            'learning_rate': 0.015,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': SEED,
            'verbose': False
        }
    else:
        # Standard conservative settings for other properties
        xgb_params = {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'reg_alpha': 0.01,
            'reg_lambda': 1.0,
            'random_state': SEED
        }
        
        lgb_params = {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'num_leaves': 31,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 1.0,
            'random_state': SEED
        }
        
        cat_params = {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': SEED,
            'verbose': False
        }
    
    # Train models using cross-validation
    for i, (train_idx, val_idx) in enumerate(cv.split(X)):
        fold_count += 1
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Prepare data transformers
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)
        
        # 1. XGBoost
        try:
            xgb_model = xgb.XGBRegressor(**xgb_params)
            xgb_model.fit(X_train_scaled, y_train)
            xgb_preds = xgb_model.predict(X_val_scaled)
            xgb_score = mean_absolute_error(y_val, xgb_preds)
            
            # Store OOF predictions
            oof_preds[val_idx] += xgb_preds
            
            # Add to ensemble
            all_models.append(xgb_model)
            all_transformers.append((scaler, imputer))
            all_feature_subsets.append(feature_cols)
            all_scores.append(xgb_score)
            
            print(f"  Fold {fold_count} XGB MAE: {xgb_score:.6f}")
        except Exception as e:
            print(f"  XGB training failed: {str(e)}")
        
        # 2. LightGBM
        try:
            lgb_model = LGBMRegressor(**lgb_params)
            lgb_model.fit(X_train_scaled, y_train)
            lgb_preds = lgb_model.predict(X_val_scaled)
            lgb_score = mean_absolute_error(y_val, lgb_preds)
            
            # Add to ensemble
            all_models.append(lgb_model)
            all_transformers.append((scaler, imputer))
            all_feature_subsets.append(feature_cols)
            all_scores.append(lgb_score)
            
            print(f"  Fold {fold_count} LGB MAE: {lgb_score:.6f}")
        except Exception as e:
            print(f"  LGB training failed: {str(e)}")
        
        # 3. CatBoost
        try:
            cat_model = CatBoostRegressor(**cat_params)
            cat_model.fit(X_train_scaled, y_train)
            cat_preds = cat_model.predict(X_val_scaled)
            cat_score = mean_absolute_error(y_val, cat_preds)
            
            # Add to ensemble
            all_models.append(cat_model)
            all_transformers.append((scaler, imputer))
            all_feature_subsets.append(feature_cols)
            all_scores.append(cat_score)
            
            print(f"  Fold {fold_count} CatBoost MAE: {cat_score:.6f}")
        except Exception as e:
            print(f"  CatBoost training failed: {str(e)}")
    
    # If all models failed, use fallback
    if not all_models:
        print(f"All models failed for {target_col}. Using fallback value.")
        return {'model': RobustEnsemble([], [], [], fallback_values[target_col]), 'cv_score': 0.0}
    
    # Calculate overall CV score
    cv_score = np.mean(all_scores) if all_scores else 0.0
    print(f"Cross-validation MAE for {target_col}: {cv_score:.6f}")
    
    # Create ensemble model
    ensemble = RobustEnsemble(all_models, all_transformers, all_feature_subsets, fallback_values[target_col])
    
    return {
        'model': ensemble,
        'cv_score': cv_score
    }

# Train optimized models for each target
models = {}
for col in target_cols:
    print(f"\nTraining optimized models for {col}...")
    models[col] = train_optimized_models(train_with_features, col, target_features[col])

# Generate test predictions
test_preds = {}
for col in target_cols:
    print(f"Generating predictions for {col}...")
    try:
        # Ensure test data has all necessary features
        missing_cols = [c for c in target_features[col] if c not in test_with_features.columns]
        if missing_cols:
            print(f"  Warning: {len(missing_cols)} feature columns missing in test data. Adding zeros.")
            for c in missing_cols:
                test_with_features[c] = 0
        
        # Generate predictions
        test_preds[col] = models[col]['model'].predict(test_with_features[target_features[col]])
        
        # Sanity check predictions
        print(f"  Prediction range: {np.min(test_preds[col]):.4f} to {np.max(test_preds[col]):.4f}")
        
        # Replace NaN or infinite values with fallback
        mask = ~np.isfinite(test_preds[col])
        if mask.any():
            print(f"  Replacing {mask.sum()} NaN/infinite values with fallback")
            test_preds[col][mask] = fallback_values[col]
    except Exception as e:
        print(f"Error generating predictions for {col}: {str(e)}")
        # Fallback to default values
        test_preds[col] = np.full(len(test_with_features), fallback_values[col])

# Create submission dataframe
submission = pd.DataFrame({'id': test_with_features['id']})
for col in target_cols:
    submission[col] = test_preds[col]

# Ensure all rows from original test set are included
if len(submission) < len(test):
    print(f"WARNING: Submission has {len(submission)} rows but test set has {len(test)} rows")
    # Fill in missing rows with fallback values
    missing_ids = set(test['id']) - set(submission['id'])
    print(f"Adding {len(missing_ids)} missing rows")
    
    for missing_id in missing_ids:
        row = {'id': missing_id}
        for col in target_cols:
            row[col] = fallback_values[col]
        submission = pd.concat([submission, pd.DataFrame([row])], ignore_index=True)

# Check submission format
print("\nSubmission preview:")
print(submission.head())

# Save submission file
submission.to_csv('submission.csv', index=False)
print("\nSubmission file saved.")

# Calculate approximate wMAE based on CV results
weighted_scores = []
for col in target_cols:
    if 'cv_score' in models[col] and models[col]['cv_score'] > 0:
        # Apply the weight formula directly
        weighted_score = models[col]['cv_score'] * weights[col]
        weighted_scores.append(weighted_score)
        print(f"Weighted score for {col}: {weighted_score:.6f}")

if weighted_scores:
    estimated_wmae = sum(weighted_scores)
    print(f"\nEstimated weighted MAE: {estimated_wmae:.6f}")
else:
    print("\nCannot estimate weighted MAE (no CV scores available)")

elapsed_time = time.time() - start_time
print(f"Total runtime: {elapsed_time/60:.2f} minutes")
print("Completed advanced polymer property prediction pipeline")




