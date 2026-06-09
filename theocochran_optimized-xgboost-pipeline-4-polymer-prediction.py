# Cell 1: Enhanced Setup with All Required Libraries
"""
Install RDKit and import all required libraries for comprehensive polymer prediction
"""

# Install RDKit from wheel file
import subprocess
import sys

try:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "/kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl",
        "--quiet"
    ])
    print("âœ… RDKit installed successfully!")
except Exception as e:
    print(f"âš ï¸� RDKit installation failed: {e}")

# Core imports
import pandas as pd
import numpy as np
import pickle
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Fragments, Lipinski
    from rdkit.Chem import rdmolops
    import networkx as nx
    RDKIT_AVAILABLE = True
    print("âœ… RDKit and NetworkX available")
except ImportError:
    RDKIT_AVAILABLE = False
    print("â�Œ RDKit not available")

# ML libraries
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Neural network libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

optuna.logging.set_verbosity(optuna.logging.WARNING)
print("ğŸš€ All libraries loaded successfully!")


# Cell 2: Configuration and Constants
"""
Enhanced configuration with optimization settings and problem-specific constants
"""

class Config:
    # Target properties
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    SEED = 42
    FOLDS = 5
    
    # Optimization settings
    N_TRIALS = 100  # Optuna trials per target
    EARLY_STOPPING = 100
    MAX_ITERATIONS = 5000
    
    # Data paths
    BASE_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'
    
    # Model settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def get_canonical_smiles(smiles):
        """Convert SMILES to canonical form for consistency"""
        if not RDKIT_AVAILABLE:
            return smiles
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        except:
            pass
        return smiles

# Competition scoring constants (from problem description)
MINMAX_DICT = {
    'Tg': [-148.0297376, 472.25],
    'FFV': [0.2269924, 0.77709707], 
    'Tc': [0.0465, 0.524],
    'Density': [0.748691234, 1.840998909],
    'Rg': [9.7283551, 34.672905605],
}

# Problematic RDKit descriptors to remove (from Dmitry's analysis)
USELESS_COLS = [
    # NaN data
    'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO',
    'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW',
    # Constant data  
    'NumRadicalElectrons', 'SMR_VSA8', 'SlogP_VSA9', 'fr_barbitur',
    'fr_benzodiazepine', 'fr_dihydropyridine', 'fr_epoxide', 'fr_isothiocyan',
    'fr_lactam', 'fr_nitroso', 'fr_prisulfonamd', 'fr_thiocyan',
    # High correlation >0.95
    'MaxEStateIndex', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
    'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Kappa1',
    'LabuteASA', 'HeavyAtomCount', 'MolMR', 'Chi3n', 'BertzCT', 'Chi2v',
    'Chi4n', 'HallKierAlpha', 'Chi3v', 'Chi4v', 'MinAbsPartialCharge',
    'MinPartialCharge', 'MaxAbsPartialCharge', 'FpDensityMorgan2',
    'FpDensityMorgan3', 'Phi', 'Kappa3', 'fr_nitrile', 'SlogP_VSA6',
    'NumAromaticCarbocycles', 'NumAromaticRings', 'fr_benzene', 'VSA_EState6',
    'NOCount', 'fr_C_O', 'fr_C_O_noCOO', 'NumHDonors', 'fr_amide',
    'fr_Nhpyrrole', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_COO2',
    'fr_halogen', 'fr_diazo', 'fr_nitro_arom', 'fr_phos_ester'
]

print(f"ğŸ�¯ Targets: {Config.TARGETS}")
print(f"ğŸ”§ Device: {Config.DEVICE}")
print(f"ğŸš« Removing {len(USELESS_COLS)} problematic descriptors")


# Cell 3: Robust Data Loading with Complete R-Group Filtering
"""
Load competition data with complete filtering of problematic polymer notation
"""

print("ğŸ“‚ Loading competition data...")
train = pd.read_csv(Config.BASE_PATH + 'train.csv')
test = pd.read_csv(Config.BASE_PATH + 'test.csv')

print(f"   Training samples: {len(train)}")
print(f"   Test samples: {len(test)}")

def clean_and_validate_smiles(smiles):
    """Completely clean and validate SMILES, removing all problematic patterns"""
    if not isinstance(smiles, str) or len(smiles) == 0:
        return None
    
    # List of all problematic patterns we've seen
    bad_patterns = [
        '[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]', 
        "[R']", '[R"]', 'R1', 'R2', 'R3', 'R4', 'R5',
        # Additional patterns that cause issues
        '([R])', '([R1])', '([R2])', 
    ]
    
    # Check for any bad patterns
    for pattern in bad_patterns:
        if pattern in smiles:
            return None
    
    # Additional check: if it contains ] followed by [ without valid atoms, likely polymer notation
    if '][' in smiles and any(x in smiles for x in ['[R', 'R]']):
        return None
    
    # Try to parse with RDKit if available
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
            else:
                return None
        except:
            return None
    
    # If RDKit not available, return cleaned SMILES
    return smiles

# Clean and validate all SMILES
print("ğŸ”„ Cleaning and validating SMILES...")
train['canonical_smiles'] = train['SMILES'].apply(clean_and_validate_smiles)
test['canonical_smiles'] = test['SMILES'].apply(clean_and_validate_smiles)

# Remove invalid SMILES
invalid_train = train['canonical_smiles'].isnull().sum()
invalid_test = test['canonical_smiles'].isnull().sum()

print(f"   Removed {invalid_train} invalid SMILES from training data")
print(f"   Removed {invalid_test} invalid SMILES from test data")

train = train[train['canonical_smiles'].notnull()].reset_index(drop=True)
test = test[test['canonical_smiles'].notnull()].reset_index(drop=True)

print(f"   Final training samples: {len(train)}")
print(f"   Final test samples: {len(test)}")

def add_extra_data_clean(df_train, df_extra, target):
    """Add external data with thorough SMILES cleaning"""
    n_samples_before = len(df_train[df_train[target].notnull()])
    
    print(f"      Processing {len(df_extra)} {target} samples...")
    
    # Clean external SMILES
    df_extra['canonical_smiles'] = df_extra['SMILES'].apply(clean_and_validate_smiles)
    
    # Remove invalid SMILES and missing targets
    before_filter = len(df_extra)
    df_extra = df_extra[df_extra['canonical_smiles'].notnull()]
    df_extra = df_extra.dropna(subset=[target])
    after_filter = len(df_extra)
    
    print(f"      Kept {after_filter}/{before_filter} valid samples")
    
    if len(df_extra) == 0:
        print(f"      No valid data remaining for {target}")
        return df_train
    
    # Group by canonical SMILES and average duplicates
    df_extra = df_extra.groupby('canonical_smiles', as_index=False)[target].mean()
    
    cross_smiles = set(df_extra['canonical_smiles']) & set(df_train['canonical_smiles'])
    unique_smiles_extra = set(df_extra['canonical_smiles']) - set(df_train['canonical_smiles'])

    # Fill missing values
    filled_count = 0
    for smile in df_train[df_train[target].isnull()]['canonical_smiles'].tolist():
        if smile in cross_smiles:
            df_train.loc[df_train['canonical_smiles']==smile, target] = \
                df_extra[df_extra['canonical_smiles']==smile][target].values[0]
            filled_count += 1
    
    # Add unique SMILES
    extra_to_add = df_extra[df_extra['canonical_smiles'].isin(unique_smiles_extra)].copy()
    if len(extra_to_add) > 0:
        for col in Config.TARGETS:
            if col not in extra_to_add.columns:
                extra_to_add[col] = np.nan
        
        extra_to_add = extra_to_add[['canonical_smiles'] + Config.TARGETS]
        df_train = pd.concat([df_train, extra_to_add], axis=0, ignore_index=True)

    n_samples_after = len(df_train[df_train[target].notnull()])
    print(f'      {target}: +{n_samples_after-n_samples_before} samples, +{len(unique_smiles_extra)} unique SMILES')
    return df_train

# Load external datasets with robust error handling
print("\nğŸ“‚ Loading external datasets...")

external_datasets = []

# Function to safely load datasets
def safe_load_dataset(path, target, processor_func, description):
    try:
        if path.endswith('.xlsx'):
            data = pd.read_excel(path)
        else:
            data = pd.read_csv(path)
        
        data = processor_func(data)
        external_datasets.append((target, data))
        print(f"   âœ… {description}: {len(data)} samples")
        return True
    except Exception as e:
        print(f"   âš ï¸� {description} failed: {str(e)[:100]}")
        return False

# Load each dataset
safe_load_dataset(
    '/kaggle/input/tc-smiles/Tc_SMILES.csv',
    'Tc',
    lambda df: df.rename(columns={'TC_mean': 'Tc'}),
    'Tc data'
)

safe_load_dataset(
    '/kaggle/input/tg-smiles-pid-polymer-class/TgSS_enriched_cleaned.csv',
    'Tg', 
    lambda df: df[['SMILES', 'Tg']] if 'Tg' in df.columns else df,
    'TgSS enriched data'
)

safe_load_dataset(
    '/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv',
    'Tg',
    lambda df: df[['SMILES', 'Tg (C)']].rename(columns={'Tg (C)': 'Tg'}),
    'JCIM Tg data'
)

safe_load_dataset(
    '/kaggle/input/smiles-extra-data/data_tg3.xlsx',
    'Tg',
    lambda df: df.rename(columns={'Tg [K]': 'Tg'}).assign(Tg=lambda x: x['Tg'] - 273.15),
    'Xlsx Tg data'
)

safe_load_dataset(
    '/kaggle/input/smiles-extra-data/data_dnst1.xlsx',
    'Density',
    lambda df: df.rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]
                .query('SMILES.notnull() and Density.notnull() and Density != "nylon"')
                .assign(Density=lambda x: x['Density'].astype(float) - 0.118),
    'Density data'
)

# Integrate external data
print("\nğŸ”„ Integrating external data...")
train_extended = train[['canonical_smiles'] + Config.TARGETS].copy()

for target, dataset in external_datasets:
    print(f"   Processing {target} data...")
    train_extended = add_extra_data_clean(train_extended, dataset, target)

print(f"\nğŸ“Š Final training data:")
print(f"   Original samples: {len(train)}")
print(f"   Extended samples: {len(train_extended)}")
print(f"   Gain: +{len(train_extended) - len(train)} samples")

for target in Config.TARGETS:
    count = train_extended[target].notna().sum()
    original_count = train[target].notna().sum() if target in train.columns else 0
    gain = count - original_count
    print(f"   {target}: {count:,} samples (+{gain})")

print(f"\nâœ… Data integration complete with clean SMILES!")


# Cell 4: Modern Feature Engineering without Deprecation Warnings
"""
Feature engineering using modern RDKit API to eliminate deprecation warnings
"""

def compute_all_descriptors_modern(smiles):
    """Compute RDKit descriptors with modern API and error handling"""
    if not RDKIT_AVAILABLE:
        return []
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            desc_names = [desc[0] for desc in Descriptors.descList if desc[0] not in USELESS_COLS]
            return [0] * len(desc_names)
        
        descriptors = []
        for desc_name, desc_func in Descriptors.descList:
            if desc_name not in USELESS_COLS:
                try:
                    value = desc_func(mol)
                    descriptors.append(value if not (np.isnan(value) or np.isinf(value)) else 0)
                except:
                    descriptors.append(0)
        
        return descriptors
    except:
        desc_names = [desc[0] for desc in Descriptors.descList if desc[0] not in USELESS_COLS]
        return [0] * len(desc_names)

def compute_graph_features_modern(smiles):
    """Compute graph features with error handling"""
    if not RDKIT_AVAILABLE:
        return {'graph_diameter': 0, 'avg_shortest_path': 0, 'num_cycles': 0}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'graph_diameter': 0, 'avg_shortest_path': 0, 'num_cycles': 0}
        
        adj = rdmolops.GetAdjacencyMatrix(mol)
        G = nx.from_numpy_array(adj)
        
        diameter = 0
        avg_path = 0
        cycles = 0
        
        try:
            if nx.is_connected(G) and len(G) > 1:
                diameter = nx.diameter(G)
                avg_path = nx.average_shortest_path_length(G)
            cycles = len(list(nx.cycle_basis(G)))
        except:
            pass
        
        return {
            'graph_diameter': diameter,
            'avg_shortest_path': avg_path,
            'num_cycles': cycles
        }
    except:
        return {'graph_diameter': 0, 'avg_shortest_path': 0, 'num_cycles': 0}

def compute_morgan_fingerprints_modern(smiles, radius=2, n_bits=1024):
    """Compute Morgan fingerprints using modern API (no deprecation warnings)"""
    if not RDKIT_AVAILABLE:
        return {f'morgan_{i}': 0 for i in range(n_bits)}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {f'morgan_{i}': 0 for i in range(n_bits)}
        
        # Use the modern MorganGenerator API
        try:
            from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
            generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
            fp = generator.GetFingerprint(mol)
            # Convert to bit string and then to dictionary
            fp_bits = fp.ToBitString()
            return {f'morgan_{i}': int(bit) for i, bit in enumerate(fp_bits)}
        except ImportError:
            # Fallback to rdMolDescriptors (still modern, less deprecated)
            try:
                from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
                fp = GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
                return {f'morgan_{i}': int(fp[i]) for i in range(n_bits)}
            except ImportError:
                # Final fallback to AllChem (will show warnings but works)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
                return {f'morgan_{i}': int(fp[i]) for i in range(n_bits)}
        
    except Exception as e:
        # Return zeros if anything fails
        return {f'morgan_{i}': 0 for i in range(n_bits)}

def preprocessing_modern(df, smiles_col='canonical_smiles'):
    """Modern feature engineering pipeline without deprecation warnings"""
    
    if not RDKIT_AVAILABLE:
        print("âš ï¸� RDKit not available - using basic string features")
        basic_features = []
        for smiles in df[smiles_col]:
            smiles_str = str(smiles)
            basic_features.append({
                'smiles_length': len(smiles_str),
                'carbon_count': smiles_str.count('C'),
                'nitrogen_count': smiles_str.count('N'),
                'oxygen_count': smiles_str.count('O'),
                'sulfur_count': smiles_str.count('S'),
                'phosphorus_count': smiles_str.count('P'),
                'fluorine_count': smiles_str.count('F'),
                'chlorine_count': smiles_str.count('Cl'),
                'bromine_count': smiles_str.count('Br'),
                'iodine_count': smiles_str.count('I'),
                'double_bonds': smiles_str.count('='),
                'triple_bonds': smiles_str.count('#'),
                'rings': smiles_str.count('('),
                'aromatic_c': smiles_str.count('c'),
                'aromatic_n': smiles_str.count('n'),
                'aromatic_o': smiles_str.count('o'),
                'branches': smiles_str.count('['),
                'polymer_stars': smiles_str.count('*')
            })
        return pd.DataFrame(basic_features)
    
    print(f"ğŸ§¬ Computing modern molecular features for {len(df)} molecules...")
    
    # Suppress RDKit warnings during feature computation
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    # RDKit descriptors
    print("   Computing RDKit descriptors (modern API)...")
    desc_names = [desc[0] for desc in Descriptors.descList if desc[0] not in USELESS_COLS]
    descriptors = []
    
    for smiles in tqdm(df[smiles_col], desc="RDKit descriptors"):
        descriptors.append(compute_all_descriptors_modern(smiles))
    
    desc_df = pd.DataFrame(descriptors, columns=desc_names)
    
    # Graph features
    print("   Computing graph features...")
    graph_features = []
    for smiles in tqdm(df[smiles_col], desc="Graph features"):
        graph_features.append(compute_graph_features_modern(smiles))
    
    graph_df = pd.DataFrame(graph_features)
    
    # Morgan fingerprints with modern API
    print("   Computing Morgan fingerprints (modern API)...")
    morgan_features = []
    for smiles in tqdm(df[smiles_col], desc="Morgan fingerprints"):
        morgan_features.append(compute_morgan_fingerprints_modern(smiles))
    
    morgan_df = pd.DataFrame(morgan_features)
    
    # Combine all features
    result = pd.concat([desc_df, graph_df, morgan_df], axis=1)
    
    # Clean up infinite and NaN values
    result = result.replace([np.inf, -np.inf], 0)
    result = result.fillna(0)
    
    print(f"âœ… Generated {len(result.columns)} features")
    print(f"   - {len(desc_df.columns)} RDKit descriptors")
    print(f"   - {len(graph_df.columns)} graph features") 
    print(f"   - {len(morgan_df.columns)} fingerprint features")
    
    return result

# Generate features
print("ğŸ”¬ Generating features for training data...")
train_features = preprocessing_modern(train_extended)
train_full = pd.concat([train_extended, train_features], axis=1)

print("\nğŸ”¬ Generating features for test data...")
test_features = preprocessing_modern(test)
test_full = pd.concat([test, test_features], axis=1)

# Feature column identification
all_features = train_features.columns.tolist()
print(f"\nâœ… Feature generation complete!")
print(f"   Total feature columns: {len(all_features)}")

# Intelligent feature filtering per target
features_by_target = {}
for target in Config.TARGETS:
    target_data = train_full[train_full[target].notnull()]
    
    if len(target_data) == 0:
        print(f"   âš ï¸� No data for {target}, skipping")
        features_by_target[target] = []
        continue
    
    # Remove truly problematic features
    good_features = []
    for col in all_features:
        if col in target_data.columns:
            values = target_data[col]
            
            # Skip if all values are the same
            if values.nunique() <= 1:
                continue
            
            # Skip if >98% zeros (likely uninformative)
            if (values == 0).mean() > 0.98:
                continue
            
            # Skip if too many missing values
            if values.isnull().mean() > 0.5:
                continue
            
            good_features.append(col)
    
    features_by_target[target] = good_features
    print(f"   {target}: {len(good_features)} features ({len(target_data)} samples)")

# Memory cleanup
del train_features, test_features
gc.collect()

print(f"\nğŸ§¹ Memory cleaned, ready for training!")




# Cell 5: Model Training with Robust Data Cleaning
"""
Train optimized XGBoost models with comprehensive data cleaning to handle infinite values
"""

def clean_feature_matrix(X, y, feature_names):
    """Thoroughly clean feature matrix to remove infinite and problematic values"""
    print(f"   Cleaning feature matrix: {X.shape}")
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(X, columns=feature_names)
    
    # Check for infinite values
    inf_counts = {}
    for col in df.columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    
    if inf_counts:
        print(f"   Found infinite values in {len(inf_counts)} columns")
        for col, count in list(inf_counts.items())[:5]:  # Show first 5
            print(f"      {col}: {count} infinite values")
    
    # Replace infinite values with 0
    df = df.replace([np.inf, -np.inf], 0)
    
    # Fill any remaining NaN values
    df = df.fillna(0)
    
    # Check for extremely large values that might cause issues
    large_value_threshold = 1e10
    for col in df.columns:
        large_mask = np.abs(df[col]) > large_value_threshold
        if large_mask.any():
            print(f"   Clipping {large_mask.sum()} extremely large values in {col}")
            df[col] = np.clip(df[col], -large_value_threshold, large_value_threshold)
    
    # Remove constant columns
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"   Removing {len(constant_cols)} constant columns")
        df = df.drop(columns=constant_cols)
    
    print(f"   Cleaned matrix shape: {df.shape}")
    return df.values, df.columns.tolist()

def objective_robust(trial, X, y, groups, feature_names):
    """Robust Optuna optimization objective with data cleaning"""
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'seed': Config.SEED,
        'verbosity': 0,
        'tree_method': 'hist',
        'missing': 0.0,  # Explicitly handle missing values
        
        # Hyperparameters to optimize
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),  # Reduced max depth
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
    }
    
    # Use GPU if available
    if torch.cuda.is_available():
        params['tree_method'] = 'gpu_hist'
        params['gpu_id'] = 0
    
    group_kfold = GroupKFold(n_splits=Config.FOLDS)
    cv_scores = []
    
    try:
        for train_idx, valid_idx in group_kfold.split(X, y, groups=groups):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]
            
            # Additional cleaning for each fold
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)
            
            dtrain = xgb.DMatrix(X_train, label=y_train, missing=0.0)
            dvalid = xgb.DMatrix(X_valid, label=y_valid, missing=0.0)
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=Config.MAX_ITERATIONS,
                evals=[(dtrain, 'train'), (dvalid, 'valid')],
                early_stopping_rounds=Config.EARLY_STOPPING,
                verbose_eval=False
            )
            
            cv_scores.append(model.best_score)
        
        return np.mean(cv_scores)
    
    except Exception as e:
        print(f"   Trial failed: {str(e)[:100]}")
        return float('inf')  # Return a large value for failed trials

def train_optimized_model_robust(target):
    """Train optimized model with robust data cleaning"""
    
    print(f"\nğŸ�¯ Training {target}...")
    
    # Prepare target-specific data
    target_data = train_full[train_full[target].notnull()].reset_index(drop=True)
    if len(target_data) < 50:
        print(f"   âš ï¸� Insufficient data for {target} ({len(target_data)} samples)")
        return None, None
    
    # Get features for this target
    target_features = features_by_target[target]
    if len(target_features) == 0:
        print(f"   âš ï¸� No features available for {target}")
        return None, None
    
    # Extract feature matrix
    X = target_data[target_features].values
    y = target_data[target].values
    groups = target_data['canonical_smiles'].factorize()[0]
    
    print(f"   Initial data: {len(target_data)} samples, {X.shape[1]} features")
    
    # Clean the feature matrix thoroughly
    X_clean, clean_features = clean_feature_matrix(X, y, target_features)
    
    if X_clean.shape[1] == 0:
        print(f"   âš ï¸� No features remaining after cleaning for {target}")
        return None, None
    
    print(f"   Starting optimization with {X_clean.shape[1]} clean features...")
    
    # Optimize hyperparameters with robust objective
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=Config.SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)  # More aggressive pruning
    )
    
    try:
        study.optimize(
            lambda trial: objective_robust(trial, X_clean, y, groups, clean_features),
            n_trials=Config.N_TRIALS,
            show_progress_bar=True,
            timeout=1800  # 30 minute timeout per target
        )
        
        if len(study.trials) == 0 or study.best_value == float('inf'):
            print(f"   âš ï¸� No successful trials for {target}")
            return None, None
        
        best_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae', 
            'seed': Config.SEED,
            'verbosity': 0,
            'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
            'missing': 0.0
        }
        best_params.update(study.best_params)
        
        print(f"   Best CV MAE: {study.best_value:.5f}")
        
        # Train final model on all cleaned data
        X_final = np.nan_to_num(X_clean, nan=0.0, posinf=0.0, neginf=0.0)
        dtrain = xgb.DMatrix(X_final, label=y, missing=0.0)
        
        final_model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=Config.MAX_ITERATIONS,
            verbose_eval=False
        )
        
        # Evaluate on training data
        train_pred = final_model.predict(dtrain)
        train_mae = mean_absolute_error(y, train_pred)
        
        print(f"   âœ… {target} complete - CV: {study.best_value:.5f}, Train: {train_mae:.5f}")
        
        return final_model, {
            'cv_mae': study.best_value,
            'train_mae': train_mae,
            'best_params': best_params,
            'n_samples': len(target_data),
            'features': clean_features  # Store the cleaned feature names
        }
    
    except Exception as e:
        print(f"   â�Œ Training failed for {target}: {str(e)[:150]}")
        return None, None

# Train all models with robust pipeline
print("ğŸš€ Training optimized models with robust data cleaning...")
models = {}
results = {}

for target in Config.TARGETS:
    model, result = train_optimized_model_robust(target)
    if model is not None:
        models[target] = model
        results[target] = result

print(f"\nâœ… Training complete!")
print(f"   Successfully trained models: {list(models.keys())}")

# Display results summary
if results:
    print(f"\nğŸ“Š Results Summary:")
    for target, result in results.items():
        print(f"   {target}: CV={result['cv_mae']:.5f}, Train={result['train_mae']:.5f}, Samples={result['n_samples']:,}")
else:
    print(f"\nâš ï¸� No models were successfully trained!")


# Cell 6: Robust Prediction Generation with Data Cleaning
"""
Generate predictions with the same data cleaning pipeline used in training
"""

def generate_predictions_robust():
    """Generate predictions with robust data cleaning"""
    
    print("ğŸ”® Generating robust predictions...")
    
    # Initialize predictions
    test_predictions = pd.DataFrame({'id': test['id']})
    for target in Config.TARGETS:
        test_predictions[target] = 0.0
    
    # Generate predictions for each target
    for target in Config.TARGETS:
        if target in models and target in results:
            print(f"   Predicting {target}...")
            
            try:
                # Get the same features used during training
                model_features = results[target]['features']
                
                # Extract test features
                X_test_raw = test_full[model_features].values
                
                # Apply the same cleaning as during training
                X_test_clean = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Create DMatrix with same settings as training
                dtest = xgb.DMatrix(X_test_clean, missing=0.0)
                
                # Generate predictions
                predictions = models[target].predict(dtest)
                test_predictions[target] = predictions
                
                print(f"   {target}: Generated {len(predictions)} predictions")
                print(f"      Range: [{predictions.min():.3f}, {predictions.max():.3f}]")
                
            except Exception as e:
                print(f"   âš ï¸� Prediction failed for {target}: {str(e)[:100]}")
                test_predictions[target] = 0.0
        else:
            print(f"   âš ï¸� No model available for {target}")
            test_predictions[target] = 0.0
    
    return test_predictions

# Generate initial predictions
predictions_raw = generate_predictions_robust()

print(f"\nâœ… Initial predictions complete")

# Handle data leakage (exact SMILES matches)
print(f"\nğŸ”� Handling data leakage (overlapping SMILES)...")

leakage_stats = {}
for target in Config.TARGETS:
    # Get training data with known values for this target
    train_known = train_full[train_full[target].notnull()][['canonical_smiles', target]].drop_duplicates()
    
    if len(train_known) == 0:
        leakage_stats[target] = 0
        continue
    
    # Create lookup dictionary
    smiles_to_value = dict(zip(train_known['canonical_smiles'], train_known[target]))
    
    # Find overlapping SMILES in test set
    test_smiles = test['canonical_smiles'].values
    overlapping_count = 0
    
    # Replace predictions with known values for overlapping SMILES
    for idx in range(len(test)):
        smiles = test_smiles[idx]
        if smiles in smiles_to_value:
            predictions_raw.loc[idx, target] = smiles_to_value[smiles]
            overlapping_count += 1
    
    leakage_stats[target] = overlapping_count
    print(f"   {target}: {overlapping_count} exact matches replaced")

# Final prediction validation and post-processing
print(f"\nğŸ”§ Final prediction validation...")

final_predictions = predictions_raw.copy()

# Clip predictions to reasonable ranges
for target in Config.TARGETS:
    if target in train_full.columns:
        train_values = train_full[target].dropna()
        if len(train_values) > 0:
            # Use 1st and 99th percentile as bounds
            lower_bound = train_values.quantile(0.01)
            upper_bound = train_values.quantile(0.99)
            
            before_clip = final_predictions[target].copy()
            final_predictions[target] = np.clip(
                final_predictions[target], 
                lower_bound, 
                upper_bound
            )
            
            clipped_count = (before_clip != final_predictions[target]).sum()
            if clipped_count > 0:
                print(f"   {target}: Clipped {clipped_count} predictions to [{lower_bound:.3f}, {upper_bound:.3f}]")

# Final validation
print(f"\nâœ… Final validation:")
submission_final = final_predictions[['id'] + Config.TARGETS].copy()

for target in Config.TARGETS:
    preds = submission_final[target]
    
    # Check for any remaining invalid values
    nan_count = preds.isna().sum()
    inf_count = np.isinf(preds).sum()
    
    if nan_count > 0:
        print(f"   âš ï¸� {target}: {nan_count} NaN values found, filling with median")
        median_val = train_full[target].median() if target in train_full.columns else 0
        submission_final[target] = preds.fillna(median_val)
    
    if inf_count > 0:
        print(f"   âš ï¸� {target}: {inf_count} infinite values found, replacing with median")
        median_val = train_full[target].median() if target in train_full.columns else 0
        submission_final[target] = preds.replace([np.inf, -np.inf], median_val)
    
    final_preds = submission_final[target]
    print(f"   {target}: âœ… [{final_preds.min():.3f}, {final_preds.max():.3f}], mean={final_preds.mean():.3f}")

# Save final submission
submission_final.to_csv('submission.csv', index=False)

print(f"\nğŸ�‰ Robust predictions complete!")
print(f"   Submission saved: submission.csv")
print(f"   Shape: {submission_final.shape}")

# Show leakage summary
total_leakage = sum(leakage_stats.values())
if total_leakage > 0:
    print(f"\nğŸ“Š Data leakage handled:")
    print(f"   Total exact matches: {total_leakage}")
    for target, count in leakage_stats.items():
        if count > 0:
            percentage = (count / len(test)) * 100
            print(f"   {target}: {count} ({percentage:.1f}%)")

# Display final sample
print(f"\nğŸ”� Final submission preview:")
display(submission_final.head())


# Cell 7: Competition Scoring and Validation with Robust Data Cleaning
"""
Calculate competition wMAE score and perform validation with the same data cleaning as training
"""

def scaling_error(labels, preds, property_name):
    """Calculate scaled absolute error for a property"""
    error = np.abs(labels - preds)
    min_val, max_val = MINMAX_DICT[property_name]
    label_range = max_val - min_val
    return np.mean(error / label_range)

def get_property_weights(labels_df):
    """Calculate property weights based on sample count"""
    property_weights = []
    for property_name in MINMAX_DICT.keys():
        valid_num = np.sum(labels_df[property_name].notna())
        property_weights.append(valid_num)
    
    property_weights = np.array(property_weights)
    property_weights = np.sqrt(1 / property_weights)
    return (property_weights / np.sum(property_weights)) * len(property_weights)

def wmae_score(solution_df, submission_df):
    """Calculate weighted Mean Absolute Error (wMAE) competition score"""
    chemical_properties = list(MINMAX_DICT.keys())
    property_maes = []
    property_weights = get_property_weights(solution_df[chemical_properties])
    
    for i, property_name in enumerate(chemical_properties):
        is_labeled = solution_df[property_name].notna()
        
        if np.any(is_labeled):
            mae_val = scaling_error(
                solution_df.loc[is_labeled, property_name],
                submission_df.loc[is_labeled, property_name], 
                property_name
            )
            property_maes.append(mae_val)
        else:
            property_maes.append(0.0)
    
    if not property_maes or np.sum(property_weights) == 0:
        return float('inf')
    
    return float(np.average(property_maes, weights=property_weights))

# Perform out-of-fold validation with robust data cleaning
print("ğŸ“Š Performing out-of-fold validation with robust data cleaning...")

def validate_model_performance_robust():
    """Generate OOF predictions using the same robust pipeline as training"""
    oof_predictions = train_full[['canonical_smiles'] + Config.TARGETS].copy()
    
    for target in Config.TARGETS:
        oof_predictions[f'{target}_pred'] = np.nan
    
    for target in Config.TARGETS:
        if target not in models or target not in results:
            print(f"   Skipping {target} - no trained model available")
            continue
            
        print(f"   Validating {target}...")
        
        try:
            # Get target-specific data (same as training)
            target_data = train_full[train_full[target].notnull()].reset_index(drop=True)
            
            # Use the same cleaned features as in training
            model_features = results[target]['features']
            X_raw = target_data[model_features].values
            y = target_data[target].values
            groups = target_data['canonical_smiles'].factorize()[0]
            
            # Apply the same data cleaning as in training
            X_clean = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Perform GroupKFold cross-validation
            group_kfold = GroupKFold(n_splits=Config.FOLDS)
            oof_preds = np.zeros(len(target_data))
            fold_scores = []
            
            for fold, (train_idx, valid_idx) in enumerate(group_kfold.split(X_clean, y, groups=groups)):
                # Use the same parameters as the final model
                params = results[target]['best_params']
                
                # Extract and clean fold data
                X_train_fold = X_clean[train_idx]
                X_valid_fold = X_clean[valid_idx]
                y_train_fold = y[train_idx]
                y_valid_fold = y[valid_idx]
                
                # Additional cleaning per fold
                X_train_fold = np.nan_to_num(X_train_fold, nan=0.0, posinf=0.0, neginf=0.0)
                X_valid_fold = np.nan_to_num(X_valid_fold, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Train fold model
                dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold, missing=0.0)
                dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold, missing=0.0)
                
                fold_model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=Config.MAX_ITERATIONS,
                    evals=[(dtrain, 'train'), (dvalid, 'valid')],
                    early_stopping_rounds=Config.EARLY_STOPPING,
                    verbose_eval=False
                )
                
                # Predict on validation fold
                fold_preds = fold_model.predict(dvalid)
                oof_preds[valid_idx] = fold_preds
                
                # Calculate fold score
                fold_mae = mean_absolute_error(y_valid_fold, fold_preds)
                fold_scores.append(fold_mae)
            
            # Store OOF predictions
            target_indices = target_data.index
            oof_predictions.loc[target_indices, f'{target}_pred'] = oof_preds
            
            # Calculate overall OOF MAE
            oof_mae = mean_absolute_error(y, oof_preds)
            print(f"   {target} OOF MAE: {oof_mae:.5f} (avg fold: {np.mean(fold_scores):.5f})")
            
        except Exception as e:
            print(f"   âš ï¸� Validation failed for {target}: {str(e)[:150]}")
            continue
    
    return oof_predictions

# Generate OOF predictions
try:
    oof_results = validate_model_performance_robust()
    validation_successful = True
except Exception as e:
    print(f"âš ï¸� Validation failed: {str(e)[:200]}")
    print("Continuing without validation...")
    validation_successful = False
    oof_results = None

# Calculate validation wMAE score if validation was successful
if validation_successful and oof_results is not None:
    print(f"\nğŸ�¯ Calculating validation wMAE score...")
    
    # Prepare validation data
    val_solution = oof_results[Config.TARGETS].copy()
    val_submission = oof_results[[f'{target}_pred' for target in Config.TARGETS if f'{target}_pred' in oof_results.columns]].copy()
    
    # Ensure we have predictions for all targets
    missing_preds = []
    for target in Config.TARGETS:
        if f'{target}_pred' not in val_submission.columns:
            val_submission[f'{target}_pred'] = 0.0
            missing_preds.append(target)
    
    if missing_preds:
        print(f"   âš ï¸� Using zeros for missing predictions: {missing_preds}")
    
    val_submission.columns = Config.TARGETS
    
    try:
        val_wmae = wmae_score(val_solution, val_submission)
        print(f"âœ… Validation wMAE: {val_wmae:.5f}")
    except Exception as e:
        print(f"âš ï¸� Could not calculate validation wMAE: {e}")
        val_wmae = None
else:
    val_wmae = None

# Individual target MAEs for validation (if available)
if validation_successful and oof_results is not None:
    print(f"\nğŸ“ˆ Individual target performance (validation):")
    for target in Config.TARGETS:
        pred_col = f'{target}_pred'
        if pred_col in oof_results.columns:
            mask = oof_results[target].notna() & oof_results[pred_col].notna()
            if mask.sum() > 0:
                target_mae = mean_absolute_error(
                    oof_results.loc[mask, target],
                    oof_results.loc[mask, pred_col]
                )
                samples = mask.sum()
                print(f"   {target}: MAE = {target_mae:.5f} ({samples:,} samples)")
            else:
                print(f"   {target}: No valid predictions for validation")
        else:
            print(f"   {target}: No predictions generated")

# Feature importance analysis (if models were trained successfully)
if models and results:
    print(f"\nğŸ”� Analyzing feature importance...")
    importance_summary = {}
    
    for target in models:
        if target in results:
            model = models[target]
            feature_names = results[target]['features']
            
            try:
                # Get feature importance
                importance_scores = model.get_score(importance_type='gain')
                
                # Convert to sorted list
                feature_importance = []
                for i, feature in enumerate(feature_names):
                    score = importance_scores.get(f'f{i}', 0)
                    feature_importance.append((feature, score))
                
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                importance_summary[target] = feature_importance[:20]  # Top 20
                
                print(f"\n   {target} - Top 10 features:")
                for feat, score in feature_importance[:10]:
                    print(f"      {feat}: {score:.2f}")
                    
            except Exception as e:
                print(f"   âš ï¸� Feature importance analysis failed for {target}: {str(e)[:100]}")

print(f"\nâœ… Validation and analysis complete!")
if val_wmae is not None:
    print(f"ğŸ�¯ Estimated competition score: {val_wmae:.5f}")
else:
    print(f"âš ï¸� No validation score available")

# Training summary
if results:
    print(f"\nğŸ“Š Training Summary:")
    for target, result in results.items():
        print(f"   {target}: CV={result['cv_mae']:.5f}, Features={len(result['features'])}, Samples={result['n_samples']:,}")
else:
    print(f"\nâš ï¸� No training results available")


# Cell 8: Model Ensemble and Final Predictions
"""
Create ensemble predictions and apply final post-processing using correct variable names
"""

print("ğŸ¤– Creating ensemble predictions...")

# Check if we have predictions from the robust pipeline
if 'final_predictions' not in locals():
    print("âš ï¸� No predictions found from previous cells. Using fallback approach...")
    
    # Fallback: create basic predictions if the robust pipeline failed
    final_predictions = pd.DataFrame({'id': test['id']})
    for target in Config.TARGETS:
        final_predictions[target] = 0.0
        
        # Try to use model if available
        if target in models and target in results:
            try:
                print(f"   Generating fallback predictions for {target}...")
                model_features = results[target]['features']
                X_test_raw = test_full[model_features].values
                X_test_clean = np.nan_to_num(X_test_raw, nan=0.0, posinf=0.0, neginf=0.0)
                dtest = xgb.DMatrix(X_test_clean, missing=0.0)
                predictions = models[target].predict(dtest)
                final_predictions[target] = predictions
                print(f"   {target}: Generated {len(predictions)} fallback predictions")
            except Exception as e:
                print(f"   âš ï¸� Fallback prediction failed for {target}: {str(e)[:100]}")
                final_predictions[target] = 0.0
        else:
            print(f"   âš ï¸� No model available for {target}")

# Apply final post-processing
print("ğŸ”§ Applying post-processing...")

# 1. Clip predictions to reasonable ranges based on training data
for target in Config.TARGETS:
    if target in train_full.columns:
        train_values = train_full[target].dropna()
        if len(train_values) > 0:
            # Use 99.5th percentile as bounds to handle outliers
            lower_bound = train_values.quantile(0.005)
            upper_bound = train_values.quantile(0.995)
            
            before_clip = final_predictions[target].copy()
            final_predictions[target] = np.clip(
                final_predictions[target], 
                lower_bound, 
                upper_bound
            )
            
            clipped_count = (before_clip != final_predictions[target]).sum()
            if clipped_count > 0:
                print(f"   {target}: Clipped {clipped_count} predictions to [{lower_bound:.3f}, {upper_bound:.3f}]")

# 2. Handle any remaining NaN or infinite values
for target in Config.TARGETS:
    preds = final_predictions[target]
    
    # Replace NaN with median of training data
    nan_count = preds.isna().sum()
    if nan_count > 0:
        median_val = train_full[target].median() if target in train_full.columns else 0
        final_predictions[target] = preds.fillna(median_val)
        print(f"   {target}: Filled {nan_count} NaN values with {median_val:.3f}")
    
    # Replace infinite values
    inf_count = np.isinf(preds).sum()
    if inf_count > 0:
        median_val = train_full[target].median() if target in train_full.columns else 0
        final_predictions[target] = preds.replace([np.inf, -np.inf], median_val)
        print(f"   {target}: Replaced {inf_count} infinite values with {median_val:.3f}")

# Handle data leakage (overlapping SMILES) - critical for competition success
print(f"\nğŸ”� Handling data leakage (overlapping SMILES)...")

leakage_stats = {}
for target in Config.TARGETS:
    # Get training data with known values for this target
    train_known = train_full[train_full[target].notnull()][['canonical_smiles', target]].drop_duplicates()
    
    if len(train_known) == 0:
        leakage_stats[target] = 0
        continue
    
    # Create lookup dictionary
    smiles_to_value = dict(zip(train_known['canonical_smiles'], train_known[target]))
    
    # Find overlapping SMILES in test set
    test_smiles = test['canonical_smiles'].values
    overlapping_count = 0
    
    # Replace predictions with known values for overlapping SMILES
    for idx in range(len(test)):
        smiles = test_smiles[idx]
        if smiles in smiles_to_value:
            final_predictions.loc[idx, target] = smiles_to_value[smiles]
            overlapping_count += 1
    
    leakage_stats[target] = overlapping_count
    print(f"   {target}: {overlapping_count} overlapping SMILES found and replaced")

print(f"\nğŸ“Š Data leakage summary:")
total_leakage = sum(leakage_stats.values())
print(f"   Total replacements: {total_leakage}")
for target, count in leakage_stats.items():
    if count > 0:
        percentage = (count / len(test)) * 100
        print(f"   {target}: {count} exact matches used ({percentage:.1f}%)")

# 3. Final validation of submission format
print(f"\nâœ… Final submission validation:")
submission_final = final_predictions[['id'] + Config.TARGETS].copy()

# Check required format
assert list(submission_final.columns) == ['id'] + Config.TARGETS, "Incorrect column order"
assert len(submission_final) == len(test), f"Incorrect number of rows: {len(submission_final)} vs {len(test)}"
assert submission_final['id'].equals(test['id']), "ID mismatch"

for target in Config.TARGETS:
    preds = submission_final[target]
    assert not preds.isna().any(), f"{target} contains NaN values"
    assert not np.isinf(preds).any(), f"{target} contains infinite values"
    print(f"   {target}: âœ… Valid [{preds.min():.3f}, {preds.max():.3f}], mean={preds.mean():.3f}")

# Save final submission
submission_final.to_csv('submission.csv', index=False)

print(f"\nğŸ�‰ Final submission ready!")
print(f"   File: submission.csv")
print(f"   Shape: {submission_final.shape}")

# Summary statistics
print(f"\nğŸ“Š Final prediction summary:")
display(submission_final[Config.TARGETS].describe().round(4))

# Show sample of final submission
print(f"\nğŸ”� Final submission sample:")
display(submission_final.head(10))


# Cell 9: Competition Summary and Model Insights
"""
Final summary of approach, performance metrics, and key insights with robust error handling
"""

print("ğŸ�† NEURIPS POLYMER PREDICTION - FINAL SUMMARY")
print("=" * 60)

# Data summary
print(f"\nğŸ“Š DATA SUMMARY:")
original_train_size = len(train) if 'train' in locals() else 0
extended_train_size = len(train_extended) if 'train_extended' in locals() else 0
test_size = len(test) if 'test' in locals() else 0

print(f"   Original training samples: {original_train_size:,}")
print(f"   Extended training samples: {extended_train_size:,}")
if extended_train_size > original_train_size:
    gain = extended_train_size - original_train_size
    percentage = (gain / original_train_size * 100) if original_train_size > 0 else 0
    print(f"   Data augmentation gain: +{gain:,} samples ({percentage:.1f}%)")
else:
    print(f"   No data augmentation applied")
print(f"   Test samples: {test_size:,}")

print(f"\nğŸ�¯ TARGET COVERAGE:")
for target in Config.TARGETS:
    original_count = 0
    extended_count = 0
    
    if 'train' in locals() and target in train.columns:
        original_count = train[target].notna().sum()
    if 'train_extended' in locals() and target in train_extended.columns:
        extended_count = train_extended[target].notna().sum()
    
    gain = extended_count - original_count
    
    if 'models' in locals() and target in models:
        print(f"   {target}: âœ… Model trained ({extended_count:,} samples, +{gain:,} from external data)")
    else:
        print(f"   {target}: â�Œ No model ({extended_count:,} samples)")

# Feature summary
print(f"\nğŸ§¬ FEATURE ENGINEERING:")
total_features = len(all_features) if 'all_features' in locals() else 0
print(f"   Total features generated: {total_features:,}")

if RDKIT_AVAILABLE and total_features > 0:
    # Estimate feature breakdown based on typical counts
    estimated_desc = min(150, total_features)  # RDKit descriptors
    estimated_morgan = min(1024, max(0, total_features - estimated_desc - 10))  # Morgan fingerprints
    estimated_graph = min(10, max(0, total_features - estimated_desc - estimated_morgan))  # Graph features
    
    print(f"   RDKit descriptors: ~{estimated_desc:,}")
    print(f"   Morgan fingerprints: ~{estimated_morgan:,}")
    print(f"   Graph features: ~{estimated_graph:,}")
    print(f"   Problematic descriptors removed: {len(USELESS_COLS)}")
else:
    print(f"   Basic string-based features used (RDKit not available)")

# Model performance
print(f"\nğŸ¤– MODEL PERFORMANCE:")
if 'results' in locals() and results:
    for target, result in results.items():
        print(f"   {target}:")
        print(f"      Cross-validation MAE: {result['cv_mae']:.5f}")
        print(f"      Training MAE: {result['train_mae']:.5f}")
        print(f"      Training samples: {result['n_samples']:,}")
        if 'features' in result:
            print(f"      Features used: {len(result['features']):,}")
else:
    print(f"   No detailed model performance available")

# Validation results
if 'val_wmae' in locals() and val_wmae is not None:
    print(f"\nğŸ�¯ ESTIMATED COMPETITION SCORE:")
    print(f"   Validation wMAE: {val_wmae:.5f}")
else:
    print(f"\nğŸ�¯ ESTIMATED COMPETITION SCORE:")
    print(f"   Validation wMAE: Not available")

# Data leakage impact
if 'leakage_stats' in locals():
    total_leakage = sum(leakage_stats.values())
    if total_leakage > 0:
        print(f"\nğŸ”� DATA LEAKAGE HANDLING:")
        print(f"   Total test samples with exact train matches: {total_leakage}")
        for target, count in leakage_stats.items():
            if count > 0:
                percentage = (count / test_size) * 100 if test_size > 0 else 0
                print(f"   {target}: {count} samples ({percentage:.1f}%)")
    else:
        print(f"\nğŸ”� DATA LEAKAGE HANDLING:")
        print(f"   No exact SMILES matches found between train and test")
else:
    print(f"\nğŸ”� DATA LEAKAGE HANDLING:")
    print(f"   Data leakage analysis not performed")

# Optimization details
print(f"\nâš™ï¸� OPTIMIZATION DETAILS:")
print(f"   Hyperparameter trials per target: {Config.N_TRIALS}")
print(f"   Cross-validation folds: {Config.FOLDS}")
print(f"   Early stopping rounds: {Config.EARLY_STOPPING}")
print(f"   Max iterations: {Config.MAX_ITERATIONS:,}")
print(f"   Device used: {Config.DEVICE}")

# Training success summary
successful_models = len(models) if 'models' in locals() else 0
total_targets = len(Config.TARGETS)
success_rate = (successful_models / total_targets * 100) if total_targets > 0 else 0

print(f"\nğŸ“ˆ TRAINING SUCCESS:")
print(f"   Models successfully trained: {successful_models}/{total_targets} ({success_rate:.1f}%)")
if 'models' in locals():
    successful_targets = list(models.keys())
    failed_targets = [t for t in Config.TARGETS if t not in successful_targets]
    if successful_targets:
        print(f"   Successful targets: {', '.join(successful_targets)}")
    if failed_targets:
        print(f"   Failed targets: {', '.join(failed_targets)}")

# Key insights and lessons learned
print(f"\nğŸ’¡ KEY INSIGHTS:")
insights = []
if extended_train_size > original_train_size:
    gain = extended_train_size - original_train_size
    insights.append(f"External data integration crucial (+{gain:,} samples)")
if 'leakage_stats' in locals() and sum(leakage_stats.values()) > 0:
    insights.append("Data leakage handling essential (exact SMILES matches)")
if total_features > 100:
    insights.append("Comprehensive feature engineering with RDKit descriptors")
if successful_models > 0:
    insights.append("Hyperparameter optimization provides significant gains")
if RDKIT_AVAILABLE:
    insights.append("RDKit molecular descriptors + fingerprints are powerful")

if insights:
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")
else:
    print(f"   Basic approach implemented with available resources")

print(f"\nğŸš€ NEXT STEPS FOR IMPROVEMENT:")
improvements = [
    "Add neural network models (CNN-LSTM, Transformer)",
    "Implement advanced ensemble methods", 
    "Add more external datasets",
    "Optimize ensemble weights",
    "Implement multi-task learning",
    "Add graph neural networks"
]

for i, improvement in enumerate(improvements, 1):
    print(f"   {i}. {improvement}")

print(f"\nğŸ“� OUTPUT FILES:")
print(f"   submission.csv - Main submission file")
if 'submission_final' in locals():
    print(f"   Contains {len(submission_final):,} predictions for {len(Config.TARGETS)} targets")
else:
    print(f"   Contains predictions for {len(Config.TARGETS)} targets")

print(f"\nğŸ�‰ READY FOR SUBMISSION!")
print(f"   Submit: submission.csv")
if successful_models >= 3:  # If we got at least 3 models working
    print(f"   Expected significant improvement over baseline approaches")
elif successful_models > 0:
    print(f"   Expected moderate improvement with partial model success")
else:
    print(f"   Basic submission generated despite training challenges")

print("\n" + "=" * 60)
print("ğŸŒŸ GOOD LUCK! ğŸŒŸ")
print("=" * 60)

# Display final submission preview if available
print(f"\nğŸ”� FINAL SUBMISSION PREVIEW:")
if 'submission_final' in locals():
    display(submission_final.head(10))
    print(f"\nSubmission shape: {submission_final.shape}")
    print(f"Columns: {list(submission_final.columns)}")
    
    # Show prediction statistics
    print(f"\nPrediction Statistics:")
    stats_df = submission_final[Config.TARGETS].describe().round(4)
    display(stats_df)
else:
    print("   Submission file not available for preview")

print("\n" + "=" * 60)
print("ğŸ“Š EXECUTION COMPLETE")
print("=" * 60)

