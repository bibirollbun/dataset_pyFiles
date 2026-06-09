# Install TabNet
!cp -r /kaggle/input/tabnet-offline/* /kaggle/working/
!pip install -f --quiet --no-index --find-links='/kaggle/input/tabnet-offline' 'pytorch_tabnet-4.1.0-py3-none-any.whl'


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Fragments, Lipinski
from rdkit.Chem import rdmolops
# Data paths
BASE_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'
RDKIT_AVAILABLE = True
TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
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


import pandas as pd
import numpy as np


#Cell 3: Robust Data Loading with Complete R-Group Filtering
"""
Load competition data with complete filtering of problematic polymer notation
"""

print("ğŸ“‚ Loading competition data...")
train = pd.read_csv(BASE_PATH + 'train.csv')
test = pd.read_csv(BASE_PATH + 'test.csv')

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
train['SMILES'] = train['SMILES'].apply(clean_and_validate_smiles)
test['SMILES'] = test['SMILES'].apply(clean_and_validate_smiles)

# Remove invalid SMILES
invalid_train = train['SMILES'].isnull().sum()
invalid_test = test['SMILES'].isnull().sum()

print(f"   Removed {invalid_train} invalid SMILES from training data")
print(f"   Removed {invalid_test} invalid SMILES from test data")

train = train[train['SMILES'].notnull()].reset_index(drop=True)
test = test[test['SMILES'].notnull()].reset_index(drop=True)

print(f"   Final training samples: {len(train)}")
print(f"   Final test samples: {len(test)}")

def add_extra_data_clean(df_train, df_extra, target):
    """Add external data with thorough SMILES cleaning"""
    n_samples_before = len(df_train[df_train[target].notnull()])
    
    print(f"      Processing {len(df_extra)} {target} samples...")
    
    # Clean external SMILES
    df_extra['SMILES'] = df_extra['SMILES'].apply(clean_and_validate_smiles)
    
    # Remove invalid SMILES and missing targets
    before_filter = len(df_extra)
    df_extra = df_extra[df_extra['SMILES'].notnull()]
    df_extra = df_extra.dropna(subset=[target])
    after_filter = len(df_extra)
    
    print(f"      Kept {after_filter}/{before_filter} valid samples")
    
    if len(df_extra) == 0:
        print(f"      No valid data remaining for {target}")
        return df_train
    
    # Group by canonical SMILES and average duplicates
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()
    
    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])

    # Fill missing values
    filled_count = 0
    for smile in df_train[df_train[target].isnull()]['SMILES'].tolist():
        if smile in cross_smiles:
            df_train.loc[df_train['SMILES']==smile, target] = \
                df_extra[df_extra['SMILES']==smile][target].values[0]
            filled_count += 1
    
    # Add unique SMILES
    extra_to_add = df_extra[df_extra['SMILES'].isin(unique_smiles_extra)].copy()
    if len(extra_to_add) > 0:
        for col in TARGETS:
            if col not in extra_to_add.columns:
                extra_to_add[col] = np.nan
        
        extra_to_add = extra_to_add[['SMILES'] + TARGETS]
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

safe_load_dataset(
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv',
    'FFV', 
    lambda df: df[['SMILES', 'FFV']] if 'FFV' in df.columns else df,
    'dataset 4'
)

# Integrate external data
print("\nğŸ”„ Integrating external data...")
train_extended = train[['SMILES'] + TARGETS].copy()

for target, dataset in external_datasets:
    print(f"   Processing {target} data...")
    train_extended = add_extra_data_clean(train_extended, dataset, target)

print(f"\nğŸ“Š Final training data:")
print(f"   Original samples: {len(train)}")
print(f"   Extended samples: {len(train_extended)}")
print(f"   Gain: +{len(train_extended) - len(train)} samples")

for target in TARGETS:
    count = train_extended[target].notna().sum()
    original_count = train[target].notna().sum() if target in train.columns else 0
    gain = count - original_count
    print(f"   {target}: {count:,} samples (+{gain})")

print(f"\nâœ… Data integration complete with clean SMILES!")



def separate_subtables(train_df):
	
	labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
	subtables = {}
	for label in labels:
		subtables[label] = train_df[['SMILES', label]][train_df[label].notna()]
	return subtables



from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds
from rdkit.Chem.Descriptors import MolWt, MolLogP
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator

import networkx as nx
def smiles_to_combined_fingerprints_with_descriptors(smiles_list, radius=2, n_bits=128):
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    atom_pair_gen = GetAtomPairGenerator(fpSize=n_bits)
    torsion_gen = GetTopologicalTorsionGenerator(fpSize=n_bits)

    fingerprints = []
    descriptors = []
    valid_smiles = []
    invalid_indices = []

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Fingerprints
            morgan_fp = generator.GetFingerprint(mol)
            #atom_pair_fp = atom_pair_gen.GetFingerprint(mol)
            #torsion_fp = torsion_gen.GetFingerprint(mol)
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)

            combined_fp = np.concatenate([
                np.array(morgan_fp),
                #np.array(atom_pair_fp),
                #np.array(torsion_fp),
                np.array(maccs_fp)
            ])
            fingerprints.append(combined_fp)

            # RDKit Descriptors
            descriptor_values = {}
            for name, func in Descriptors.descList:
                try:
                    descriptor_values[name] = func(mol)
                except:
                    descriptor_values[name] = None

            # Specific descriptors
            descriptor_values['MolWt'] = MolWt(mol)
            descriptor_values['LogP'] = MolLogP(mol)
            descriptor_values['TPSA'] = CalcTPSA(mol)
            descriptor_values['RotatableBonds'] = CalcNumRotatableBonds(mol)
            descriptor_values['NumAtoms'] = mol.GetNumAtoms()
            descriptor_values['SMILES'] = smiles

            # Graph-based features
            try:
                adj = rdmolops.GetAdjacencyMatrix(mol)
                G = nx.from_numpy_array(adj)

                if nx.is_connected(G):
                    descriptor_values['graph_diameter'] = nx.diameter(G)
                    descriptor_values['avg_shortest_path'] = nx.average_shortest_path_length(G)
                else:
                    descriptor_values['graph_diameter'] = 0
                    descriptor_values['avg_shortest_path'] = 0

                descriptor_values['num_cycles'] = len(list(nx.cycle_basis(G)))
            except:
                descriptor_values['graph_diameter'] = None
                descriptor_values['avg_shortest_path'] = None
                descriptor_values['num_cycles'] = None

            descriptors.append(descriptor_values)
            valid_smiles.append(smiles)
        else:
            #fingerprints.append(np.zeros(n_bits * 3 + 167))
            fingerprints.append(np.zeros(n_bits  + 167))
            descriptors.append(None)
            valid_smiles.append(None)
            invalid_indices.append(i)

    return np.array(fingerprints), descriptors, valid_smiles, invalid_indices

def smiles_to_combined_fingerprints_with_descriptorsOriginal(smiles_list, radius=2, n_bits=128):
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    atom_pair_gen = GetAtomPairGenerator(fpSize=n_bits)
    torsion_gen = GetTopologicalTorsionGenerator(fpSize=n_bits)

    fingerprints = []
    descriptors = []
    valid_smiles = []
    invalid_indices = []

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Fingerprints
            morgan_fp = generator.GetFingerprint(mol)
            #atom_pair_fp = atom_pair_gen.GetFingerprint(mol)
            #torsion_fp = torsion_gen.GetFingerprint(mol)
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)

            combined_fp = np.concatenate([
                np.array(morgan_fp),
                #np.array(atom_pair_fp),
                #np.array(torsion_fp),
                np.array(maccs_fp)
            ])
            fingerprints.append(combined_fp)

            # All RDKit Descriptors
            descriptor_values = {}
            for name, func in Descriptors.descList:
                try:
                    descriptor_values[name] = func(mol)
                except:
                    descriptor_values[name] = None

            # Add specific descriptors explicitly
            descriptor_values['MolWt'] = MolWt(mol)
            descriptor_values['LogP'] = MolLogP(mol)
            descriptor_values['TPSA'] = CalcTPSA(mol)
            descriptor_values['RotatableBonds'] = CalcNumRotatableBonds(mol)
            descriptor_values['NumAtoms'] = mol.GetNumAtoms()
            descriptor_values['SMILES'] = smiles
            #descriptor_values['RadiusOfGyration'] =CalcRadiusOfGyration(mol)

            descriptors.append(descriptor_values)
            valid_smiles.append(smiles)
        else:
            #fingerprints.append(np.zeros(n_bits * 3 + 167))
            fingerprints.append(np.zeros( 167))
            descriptors.append(None)
            valid_smiles.append(None)
            invalid_indices.append(i)

    return np.array(fingerprints), descriptors, valid_smiles, invalid_indices

def make_smile_canonical(smile): # To avoid duplicates, for example: canonical '*C=C(*)C' == '*C(=C*)C'
	try:
		mol = Chem.MolFromSmiles(smile)
		canon_smile = Chem.MolToSmiles(mol, canonical=True)
		return canon_smile
	except:
		return np.nan


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Descriptors
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error



#required_descriptors = {'MolWt', 'LogP', 'TPSA', 'RotatableBonds', 'NumAtoms'}
#required_descriptors = {'graph_diameter','num_cycles','avg_shortest_path'}
required_descriptors = {'graph_diameter','num_cycles','avg_shortest_path','MolWt', 'LogP', 'TPSA', 'RotatableBonds', 'NumAtoms'}
#required_descriptors = {}

filters = {
    'Tg': list(set([
        'BalabanJ','BertzCT','Chi1','Chi3n','Chi4n','EState_VSA4','EState_VSA8',
        'FpDensityMorgan3','HallKierAlpha','Kappa3','MaxAbsEStateIndex','MolLogP',
        'NumAmideBonds','NumHeteroatoms','NumHeterocycles','NumRotatableBonds',
        'PEOE_VSA14','Phi','RingCount','SMR_VSA1','SPS','SlogP_VSA1','SlogP_VSA5',
        'SlogP_VSA8','TPSA','VSA_EState1','VSA_EState4','VSA_EState6','VSA_EState7',
        'VSA_EState8','fr_C_O_noCOO','fr_NH1','fr_benzene','fr_bicyclic','fr_ether',
        'fr_unbrch_alkane'
    ]).union(required_descriptors)),

    'FFV': list(set([
        'AvgIpc','BalabanJ','BertzCT','Chi0','Chi0n','Chi0v','Chi1','Chi1n','Chi1v',
        'Chi2n','Chi2v','Chi3n','Chi3v','Chi4n','EState_VSA10','EState_VSA5',
        'EState_VSA7','EState_VSA8','EState_VSA9','ExactMolWt','FpDensityMorgan1',
        'FpDensityMorgan2','FpDensityMorgan3','FractionCSP3','HallKierAlpha',
        'HeavyAtomMolWt','Kappa1','Kappa2','Kappa3','MaxAbsEStateIndex',
        'MaxEStateIndex','MinEStateIndex','MolLogP','MolMR','MolWt','NHOHCount',
        'NOCount','NumAromaticHeterocycles','NumHAcceptors','NumHDonors',
        'NumHeterocycles','NumRotatableBonds','PEOE_VSA14','RingCount','SMR_VSA1',
        'SMR_VSA10','SMR_VSA3','SMR_VSA5','SMR_VSA6','SMR_VSA7','SMR_VSA9','SPS',
        'SlogP_VSA1','SlogP_VSA10','SlogP_VSA11','SlogP_VSA12','SlogP_VSA2',
        'SlogP_VSA3','SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7',
        'SlogP_VSA8','TPSA','VSA_EState1','VSA_EState10','VSA_EState2',
        'VSA_EState3','VSA_EState4','VSA_EState5','VSA_EState6','VSA_EState7',
        'VSA_EState8','VSA_EState9','fr_Ar_N','fr_C_O','fr_NH0','fr_NH1',
        'fr_aniline','fr_ether','fr_halogen','fr_thiophene'
    ]).union(required_descriptors)),

    'Tc': list(set([
        'BalabanJ','BertzCT','Chi0','EState_VSA5','ExactMolWt','FpDensityMorgan1',
        'FpDensityMorgan2','FpDensityMorgan3','HeavyAtomMolWt','MinEStateIndex',
        'MolWt','NumAtomStereoCenters','NumRotatableBonds','NumValenceElectrons',
        'SMR_VSA10','SMR_VSA7','SPS','SlogP_VSA6','SlogP_VSA8','VSA_EState1',
        'VSA_EState7','fr_NH1','fr_ester','fr_halogen'
    ]).union(required_descriptors)),

    'Density': list(set([
        'BalabanJ','Chi3n','Chi3v','Chi4n','EState_VSA1','ExactMolWt',
        'FractionCSP3','HallKierAlpha','Kappa2','MinEStateIndex','MolMR','MolWt',
        'NumAliphaticCarbocycles','NumHAcceptors','NumHeteroatoms',
        'NumRotatableBonds','SMR_VSA10','SMR_VSA5','SlogP_VSA12','SlogP_VSA5',
        'TPSA','VSA_EState10','VSA_EState7','VSA_EState8'
    ]).union(required_descriptors)),

    'Rg': list(set([
        'AvgIpc','Chi0n','Chi1v','Chi2n','Chi3v','ExactMolWt','FpDensityMorgan1',
        'FpDensityMorgan2','FpDensityMorgan3','HallKierAlpha','HeavyAtomMolWt',
        'Kappa3','MaxAbsEStateIndex','MolWt','NOCount','NumRotatableBonds',
        'NumUnspecifiedAtomStereoCenters','NumValenceElectrons','PEOE_VSA14',
        'PEOE_VSA6','SMR_VSA1','SMR_VSA5','SPS','SlogP_VSA1','SlogP_VSA2',
        'SlogP_VSA7','SlogP_VSA8','VSA_EState1','VSA_EState8','fr_alkyl_halide',
        'fr_halogen'
    ]).union(required_descriptors))
}




import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold


import torch
import numpy as np
from pytorch_tabnet.augmentations import ClassificationSMOTE, RegressionSMOTE

import torch
import numpy as np
from pytorch_tabnet.augmentations import ClassificationSMOTE, RegressionSMOTE

def create_tabnet_augmentations(X_train, y_train, feature_info=None, augmentation_type='molecular'):
    """
    Create custom data augmentations for TabNet molecular property prediction
    
    Args:
        X_train: Training features
        y_train: Training targets
        feature_info: Dictionary with feature type information
        augmentation_type: Type of augmentation ('molecular', 'smote', 'noise', 'mixup')
    
    Returns:
        augmentation function for TabNet
    """
    
    class MolecularAugmentation:
        """Custom molecular data augmentation for TabNet with accurate feature detection"""
        
        def __init__(self, seed=42, noise_level=0.01, mixup_alpha=0.2, feature_info=None):
            self.noise_level = noise_level
            self.mixup_alpha = mixup_alpha
            self.feature_info = feature_info or {}
            self.seed = seed  # Store the seed
            self.descriptor_indices = []
            self.fingerprint_indices = []
            self._detect_feature_types()
            if seed is not None:
                self._set_seed()  # Call _set_seed during initialization

        def _set_seed(self):
            """Set the seed for reproducibility using stored seed."""
            if self.seed is not None:
                np.random.seed(self.seed)
                torch.manual_seed(self.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.seed)
            
        def _detect_feature_types(self):
            """Robustly detect which features are descriptors vs fingerprints"""
            if 'descriptor_indices' in self.feature_info:
                self.descriptor_indices = self.feature_info['descriptor_indices']
                self.fingerprint_indices = self.feature_info['fingerprint_indices']
                return
                
            # If no feature info provided, detect automatically
            # Method 1: Check feature names if available
            if 'feature_names' in self.feature_info:
                feature_names = self.feature_info['feature_names']
                for i, name in enumerate(feature_names):
                    if name.startswith('FP_'):
                        self.fingerprint_indices.append(i)
                    else:
                        self.descriptor_indices.append(i)
                return
            
            # Method 2: Statistical detection based on data properties
            # Fingerprints are typically binary (0/1) and sparse
            # Descriptors are continuous and have wider value ranges
            if hasattr(X_train, 'numpy'):
                data = X_train.numpy() if hasattr(X_train, 'numpy') else X_train
            else:
                data = X_train
                
            n_features = data.shape[1]
            
            for i in range(n_features):
                feature_values = data[:, i]
                unique_values = np.unique(feature_values)
                
                # Fingerprint detection criteria:
                # 1. Binary values (only 0 and 1)
                # 2. High sparsity (mostly 0s)
                # 3. Integer values only
                
                is_binary = len(unique_values) == 2 and set(unique_values).issubset({0, 1})
                is_sparse = np.mean(feature_values == 0) > 0.7  # >70% zeros
                is_integer = np.allclose(feature_values, feature_values.astype(int))
                
                # Additional check: low variance for fingerprints
                variance = np.var(feature_values)
                is_low_variance = variance < 0.25  # For binary data, max variance is 0.25
                
                if is_binary and is_sparse:
                    self.fingerprint_indices.append(i)
                elif is_integer and is_sparse and is_low_variance:
                    self.fingerprint_indices.append(i)
                else:
                    self.descriptor_indices.append(i)
            
            print(f"   ğŸ”� Auto-detected features:")
            print(f"     - Descriptor features: {len(self.descriptor_indices)}")
            print(f"     - Fingerprint features: {len(self.fingerprint_indices)}")
            
        def __call__(self, X_batch, y_batch):
            """Apply molecular-specific augmentations with accurate feature detection"""
            batch_size = X_batch.shape[0]
            augmented_X = X_batch.clone()
            augmented_y = y_batch.clone()
            
            # Apply augmentations only if we have detected features
            if len(self.descriptor_indices) > 0:
                # 1. Gaussian Noise on molecular descriptors only
                desc_indices = torch.tensor(self.descriptor_indices, device=X_batch.device)
                
                # Scale noise based on feature statistics to maintain realistic ranges
                desc_features = augmented_X[:, desc_indices]
                feature_stds = torch.std(desc_features, dim=0, keepdim=True)
                # Use adaptive noise: smaller noise for features with small std
                adaptive_noise = torch.randn_like(desc_features) * (feature_stds * self.noise_level)
                augmented_X[:, desc_indices] += adaptive_noise
                
                # 2. Feature dropout on descriptors (simulate missing measurements)
                if torch.rand(1) < 0.3:  # 30% chance of feature dropout
                    dropout_mask = torch.rand(batch_size, len(desc_indices), device=X_batch.device) > 0.05
                    augmented_X[:, desc_indices] *= dropout_mask
            
            if len(self.fingerprint_indices) > 0:
                # 3. Molecular fingerprint perturbation (conservative bit flipping)
                fp_indices = torch.tensor(self.fingerprint_indices, device=X_batch.device)
                
                if torch.rand(1) < 0.15:  # 15% chance of fingerprint perturbation
                    fp_section = augmented_X[:, fp_indices]
                    
                    # Conservative bit flipping: only flip bits that are chemically meaningful
                    # Flip rate based on fingerprint sparsity
                    sparsity = torch.mean((fp_section == 0).float(), dim=0)  # Cast to float before mean
                    adaptive_flip_prob = 0.005 * (1 - sparsity)  # Less flipping for sparse features
                    
                    flip_mask = torch.rand_like(fp_section) < adaptive_flip_prob.unsqueeze(0)
                    augmented_X[:, fp_indices] = torch.where(
                        flip_mask, 
                        1 - fp_section,  # Flip binary values
                        fp_section
                    )
                    
                    # Ensure fingerprints remain binary (0 or 1)
                    augmented_X[:, fp_indices] = torch.clamp(augmented_X[:, fp_indices], 0, 1)
            
            return augmented_X, augmented_y
    
    class SMOTEAugmentation:
        """SMOTE-based augmentation for continuous targets"""
        
        def __init__(self, k_neighbors=5, sampling_rate=0.2):
            self.k_neighbors = k_neighbors
            self.sampling_rate = sampling_rate
            
        def __call__(self, X_batch, y_batch):
            try:
                return RegressionSMOTE(device_name='cpu')(X_batch, y_batch)
            except:
                # Fallback to simple noise if SMOTE fails
                noise = torch.randn_like(X_batch) * 0.01
                return X_batch + noise, y_batch
    
    class MixUpAugmentation:
        """MixUp augmentation for molecular data"""
        
        def __init__(self, alpha=0.2, seed=42):
            self.alpha = alpha
            self.seed = seed  # Store the seed
            if seed is not None:
                self._set_seed()  # Call _set_seed during initialization
            
        def _set_seed(self):
            """Set the seed for reproducibility using stored seed."""
            if self.seed is not None:
                np.random.seed(self.seed)
                torch.manual_seed(self.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.seed)
            
        def __call__(self, X_batch, y_batch):
            batch_size = X_batch.shape[0]
            
            # Generate lambda from Beta distribution
            if self.alpha > 0:
                lam = torch.from_numpy(np.random.beta(self.alpha, self.alpha, (batch_size, 1))).float()
                if X_batch.is_cuda:
                    lam = lam.cuda()
            else:
                lam = torch.ones(batch_size, 1, device=X_batch.device)
            
            # Random permutation
            index = torch.randperm(batch_size, device=X_batch.device)
            
            # MixUp
            mixed_X = lam * X_batch + (1 - lam) * X_batch[index]
            mixed_y = lam * y_batch + (1 - lam) * y_batch[index]
            
            return mixed_X, mixed_y
    
    # Select augmentation type
    if augmentation_type == 'molecular':
        return MolecularAugmentation(seed=42, feature_info=feature_info)
    elif augmentation_type == 'smote':
        return SMOTEAugmentation()
    elif augmentation_type == 'mixup':
        return MixUpAugmentation(seed=42)  # Pass seed for consistency
    else:
        return None


# TabNet Internal Data Augmentation Implementation
import torch.nn.functional as F
from pytorch_tabnet.augmentations import ClassificationSMOTE, RegressionSMOTE
import time
from pytorch_tabnet.tab_model import TabNetRegressor
from tqdm.auto import tqdm
from datetime import datetime, timedelta

print("ğŸš€ GPU SETUP")
print("=" * 50)
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))
    print("GPU Memory Available:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
    print("GPU Memory Cached:", torch.cuda.memory_reserved() / 1024**3, "GB")
    device = 'cuda'
else:
    device = 'cpu'
    print("Using CPU")
print("=" * 50)

def create_feature_info(X_df, fp_df):
    """
    Create feature information dictionary for accurate augmentation
    
    Args:
        X_df: DataFrame with descriptor features
        fp_df: DataFrame with fingerprint features
    
    Returns:
        Dictionary with feature type information
    """
    descriptor_count = X_df.shape[1] if X_df is not None else 0
    fingerprint_count = fp_df.shape[1] if fp_df is not None else 0
    
    # Create feature indices
    descriptor_indices = list(range(descriptor_count))
    fingerprint_indices = list(range(descriptor_count, descriptor_count + fingerprint_count))
    
    # Create feature names
    descriptor_names = list(X_df.columns) if X_df is not None else []
    fingerprint_names = list(fp_df.columns) if fp_df is not None else []
    
    feature_info = {
        'descriptor_indices': descriptor_indices,
        'fingerprint_indices': fingerprint_indices,
        'feature_names': descriptor_names + fingerprint_names,
        'total_features': descriptor_count + fingerprint_count,
        'descriptor_count': descriptor_count,
        'fingerprint_count': fingerprint_count
    }
    
    return feature_info

# Modified training loop with TabNet's internal augmentation
print("\nğŸ�¯ STARTING TABNET TRAINING PIPELINE")
print("=" * 60)

train_df = train_extended
test_df = test
subtables = separate_subtables(train_df)

test_smiles = test_df['SMILES'].tolist()
test_ids = test_df['id'].values
labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

output_df = pd.DataFrame({'id': test_ids})

# Training statistics
total_start_time = time.time()
label_times = {}
label_results = {}

for label_idx, label in enumerate(labels):
    label_start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"ğŸ§ª PROCESSING LABEL [{label_idx+1}/{len(labels)}]: {label}")
    print(f"{'='*60}")
    print(f"â�° Started at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"ğŸ“Š Data shape: {subtables[label].shape}")
    print(f"ğŸ“ˆ Available samples: {len(subtables[label])}")
    
    original_smiles = subtables[label]['SMILES'].tolist()
    original_labels = subtables[label][label].values
    
    print(f"\nğŸ“‹ DATA STATISTICS FOR {label}:")
    print(f"   â€¢ Mean: {np.mean(original_labels):.4f}")
    print(f"   â€¢ Std:  {np.std(original_labels):.4f}")
    print(f"   â€¢ Min:  {np.min(original_labels):.4f}")
    print(f"   â€¢ Max:  {np.max(original_labels):.4f}")
    
    # Feature extraction with progress
    print(f"\nğŸ”¬ EXTRACTING MOLECULAR FEATURES...")
    feature_start = time.time()
    
    fingerprints, descriptors, valid_smiles, invalid_indices = \
        smiles_to_combined_fingerprints_with_descriptors(
            tqdm(original_smiles, desc="Processing SMILES", leave=False), 
            radius=2, n_bits=128
        )
    
    feature_time = time.time() - feature_start
    print(f"   âœ… Feature extraction completed in {feature_time:.2f}s")
    print(f"   ğŸ“Š Valid molecules: {len(original_smiles) - len(invalid_indices)}/{len(original_smiles)}")
    
    if invalid_indices:
        print(f"   âš ï¸�  Invalid SMILES found: {len(invalid_indices)} molecules")
    
    X = pd.DataFrame(descriptors)
    X = X.drop(['BCUT2D_MWLOW','BCUT2D_MWHI','BCUT2D_CHGHI','BCUT2D_CHGLO',
                'BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BCUT2D_MRLOW','BCUT2D_MRHI',
                'MinAbsPartialCharge','MaxPartialCharge','MinPartialCharge',
                'MaxAbsPartialCharge'], axis=1, errors='ignore')
    
    y = np.delete(original_labels, invalid_indices)
    
    print(f"\nğŸ�›ï¸�  FEATURE ENGINEERING:")
    print(f"   â€¢ Initial descriptor features: {X.shape[1]}")
    
    # Filter features
    if label in filters:
        available_cols = [col for col in filters[label] if col in X.columns]
        X = X[available_cols]
        print(f"   â€¢ After feature filtering: {X.shape[1]} descriptors")
    
    # Add fingerprints
    print(f"   â€¢ Adding molecular fingerprints...")
    fp_df = pd.DataFrame(fingerprints, columns=[f'FP_{i}' for i in range(fingerprints.shape[1])])
    fp_df.reset_index(drop=True, inplace=True)
    X.reset_index(drop=True, inplace=True)
    X = pd.concat([X, fp_df], axis=1)
    
    print(f"   â€¢ Total features (descriptors + fingerprints): {X.shape[1]}")
    print(f"   â€¢ Total samples: {X.shape[0]}")
    
    # Variance filtering
    print(f"\nğŸ”� VARIANCE FILTERING:")
    threshold = 0.01
    selector = VarianceThreshold(threshold=threshold)
    X_before_var = X.shape[1]
    feature_names = X.columns
    X = selector.fit_transform(X)
    features_after_var_filter = feature_names[selector.get_support()]
    print(f"   â€¢ Features before variance filtering: {X_before_var}")
    print(f"   â€¢ Features after variance filtering: {X.shape[1]}")
    print(f"   â€¢ Removed {X_before_var - X.shape[1]} low-variance features")
    
    # Split data
    print(f"\nğŸ“Š DATA SPLITTING:")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    print(f"   â€¢ Training samples: {X_train.shape[0]}")
    print(f"   â€¢ Validation samples: {X_test.shape[0]}")
    print(f"   â€¢ Features: {X_train.shape[1]}")
    
    # Scale features for TabNet
    print(f"\nâš–ï¸�  FEATURE SCALING:")
    scaler = StandardScaler()
    scaling_start = time.time()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    scaling_time = time.time() - scaling_start
    print(f"   âœ… Scaling completed in {scaling_time:.3f}s")
    
    # Get the new feature names and indices after variance filtering for augmentation
    descriptor_names_final = [name for name in features_after_var_filter if not name.startswith('FP_')]
    fingerprint_names_final = [name for name in features_after_var_filter if name.startswith('FP_')]
    
    new_desc_indices = [i for i, name in enumerate(features_after_var_filter) if name in descriptor_names_final]
    new_fp_indices = [i for i, name in enumerate(features_after_var_filter) if name in fingerprint_names_final]
    
    updated_feature_info = {
        'descriptor_indices': new_desc_indices,
        'fingerprint_indices': new_fp_indices,
        'feature_names': list(features_after_var_filter)
    }
    
    # TabNet model with GPU support
    print(f"\nğŸ¤– TABNET MODEL WITH ADVANCED AUGMENTATION:")
    print(f"   â€¢ Device: {device.upper()}")
    print(f"   â€¢ Architecture: n_d=32, n_a=32, n_steps=5")
    print(f"   â€¢ Regularization: gamma=1.3, lambda_sparse=1e-3")
    print(f"   â€¢ Optimizer: Adam (lr=2e-2)")
    print(f"   â€¢ Scheduler: StepLR (step_size=50, gamma=0.9)")
    
    # Create custom molecular augmentation for this label
    molecular_augmentation = create_tabnet_augmentations(
        torch.tensor(X_train_scaled, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.float32), 
        feature_info=updated_feature_info,
        augmentation_type='molecular'
    )
    
    print(f"   â€¢ Custom Molecular Augmentation: ENABLED")
    print(f"     - Gaussian noise on descriptors: Â±{molecular_augmentation.noise_level}")
    print(f"     - Feature dropout probability: 30%")
    print(f"     - Fingerprint bit flip rate: 1%")
    print(f"     - MixUp alpha: {molecular_augmentation.mixup_alpha}")
    
    # TabNet with Internal Data Augmentation
    tabnet_params = {
        'device_name': device,
        'n_d': 32,
        'n_a': 32,
        'n_steps': 5,
        'gamma': 1.3,
        'n_independent': 2,
        'n_shared': 2,
        'lambda_sparse': 1e-3,
        'optimizer_fn': torch.optim.Adam,
        'optimizer_params': dict(lr=2e-2),
        'mask_type': 'entmax',
        'scheduler_params': dict(step_size=50, gamma=0.9),
        'scheduler_fn': torch.optim.lr_scheduler.StepLR,
        'verbose': 2,  # Maximum verbosity
        'seed': 42
    }
    
    print(f"   â€¢ Ghost Batch Normalization: ENABLED")
    print(f"   â€¢ TabNet Internal Augmentation: ENABLED")
    
    model = TabNetRegressor(**tabnet_params)
    
    # Train model with detailed output
    print(f"\nğŸš€ TRAINING TABNET MODEL:")
    print(f"   â€¢ Max epochs: 200")
    print(f"   â€¢ Batch size: 256")
    print(f"   â€¢ Early stopping patience: 20")
    print(f"   â€¢ Evaluation metric: MAE")
    
    # Calculate optimal virtual batch size for data augmentation
    # Virtual batch size should be smaller than actual batch size for better augmentation
    batch_size = 256
    virtual_batch_size = min(64, batch_size // 4)  # Optimal ratio for augmentation
    
    print(f"ğŸ”„ TABNET DATA AUGMENTATION CONFIG:")
    print(f"   â€¢ Actual batch size: {batch_size}")
    print(f"   â€¢ Virtual batch size: {virtual_batch_size}")
    print(f"   â€¢ Augmentation ratio: {batch_size/virtual_batch_size:.1f}x")
    print(f"   â€¢ Ghost Batch Normalization: Active")
    print(f"   â€¢ Internal feature shuffling: Active")
    
    training_start = time.time()
    
    # Train with enhanced molecular augmentation
    model.fit(
        X_train_scaled, y_train.reshape(-1, 1),
        eval_set=[(X_test_scaled, y_test.reshape(-1, 1))],
        eval_name=['validation'],
        eval_metric=['mae'],
        max_epochs=200,
        patience=20,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,  # Key for internal augmentation
        drop_last=False,
        augmentations=molecular_augmentation,  # Custom molecular augmentation
        loss_fn=torch.nn.MSELoss(),  # Regression loss
    )
    
    print(f"   âœ… Training with molecular augmentation completed!")
    
    training_time = time.time() - training_start
    
   # Get training history
    train_history = model.history

    # Calculate metrics
    best_val_mae = min(train_history['validation_mae'])

    print(f"\nğŸ“ˆ TRAINING RESULTS:")
    print(f"   â€¢ Training time: {training_time:.2f}s")
    print(f"   â€¢ Total epochs: {len(train_history['validation_mae'])}")
    print(f"   â€¢ Best validation MAE: {best_val_mae:.6f}")
    
    # Evaluate on test set
    print(f"\nğŸ”� MODEL EVALUATION:")
    y_pred = model.predict(X_test_scaled).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"   â€¢ Test MAE: {mae:.6f}")
    print(f"   â€¢ Test MSE: {mse:.6f}")
    print(f"   â€¢ Test RMSE: {rmse:.6f}")
    print(f"   â€¢ RÂ² Score: {1 - mse/np.var(y_test):.6f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-10:][::-1]
        print(f"\nğŸ�¯ TOP 10 FEATURE IMPORTANCES:")
        for i, idx in enumerate(top_features_idx[:5], 1):  # Show top 5
            print(f"   {i}. Feature {idx}: {feature_importance[idx]:.4f}")
    
    # Retrain on full data with same augmentation settings
    print(f"\nğŸ”„ RETRAINING ON FULL DATASET:")
    X_full_scaled = scaler.fit_transform(X)
    model_full = TabNetRegressor(**tabnet_params)
    
    full_train_start = time.time()
    
    # Calculate optimal virtual batch size for full dataset
    full_batch_size = min(512, len(X_full_scaled) // 4)  # Adaptive batch size
    full_virtual_batch_size = min(128, full_batch_size // 4)
    
    print(f"   â€¢ Full dataset size: {len(X_full_scaled)}")
    print(f"   â€¢ Full training batch size: {full_batch_size}")
    print(f"   â€¢ Full training virtual batch size: {full_virtual_batch_size}")
    print(f"   â€¢ Data augmentation factor: {full_batch_size/full_virtual_batch_size:.1f}x")
    
    # Create augmentation for full training
    full_molecular_augmentation = create_tabnet_augmentations(
        torch.tensor(X_full_scaled, dtype=torch.float32), 
        torch.tensor(y, dtype=torch.float32),
        feature_info=updated_feature_info,
        augmentation_type='mixup'  # Use MixUp for full training
    )
    
    print(f"   â€¢ Full training augmentation: MixUp (alpha=0.2)")
    
    model_full.fit(
        X_full_scaled, y.reshape(-1, 1), 
        max_epochs=200, 
        batch_size=full_batch_size,
        virtual_batch_size=full_virtual_batch_size,  # Internal augmentation for full training
        augmentations=full_molecular_augmentation,  # Use MixUp augmentation
        loss_fn=torch.nn.MSELoss(),
    )
    full_train_time = time.time() - full_train_start
    print(f"   âœ… Full training with augmentation completed in {full_train_time:.2f}s")
    
    # Predict on test set
    print(f"\nğŸ�¯ GENERATING FINAL PREDICTIONS:")
    test_start = time.time()
    
    test_fingerprints, test_descriptors, test_valid_smiles, test_invalid_indices = \
        smiles_to_combined_fingerprints_with_descriptors(
            tqdm(test_smiles, desc="Processing test SMILES", leave=False), 
            radius=2, n_bits=128
        )
    
    X_test_final = pd.DataFrame(test_descriptors)
    X_test_final = X_test_final.drop(['BCUT2D_MWLOW','BCUT2D_MWHI','BCUT2D_CHGHI','BCUT2D_CHGLO',
                                     'BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BCUT2D_MRLOW','BCUT2D_MRHI',
                                     'MinAbsPartialCharge','MaxPartialCharge','MinPartialCharge',
                                     'MaxAbsPartialCharge'], axis=1, errors='ignore')
    
    if label in filters:
        available_cols = [col for col in filters[label] if col in X_test_final.columns]
        X_test_final = X_test_final[available_cols]
    
    test_fp_df = pd.DataFrame(test_fingerprints, columns=[f'FP_{i}' for i in range(test_fingerprints.shape[1])])
    test_fp_df.reset_index(drop=True, inplace=True)
    X_test_final.reset_index(drop=True, inplace=True)
    X_test_final = pd.concat([X_test_final, test_fp_df], axis=1)
    
    X_test_final = selector.transform(X_test_final)
    X_test_final_scaled = scaler.transform(X_test_final)
    
    # Final predictions
    y_pred_final = model_full.predict(X_test_final_scaled).flatten()
    test_time = time.time() - test_start
    
    print(f"   â€¢ Test prediction time: {test_time:.3f}s")
    print(f"   â€¢ Predictions: {y_pred_final}")
    print(f"   â€¢ Prediction stats:")
    print(f"     - Mean: {np.mean(y_pred_final):.4f}")
    print(f"     - Std:  {np.std(y_pred_final):.4f}")
    print(f"     - Min:  {np.min(y_pred_final):.4f}")
    print(f"     - Max:  {np.max(y_pred_final):.4f}")
    
    output_df[label] = y_pred_final
    
    # Store results for summary
    label_time = time.time() - label_start_time
    label_times[label] = label_time
    label_results[label] = {
        'samples': len(y),
        'features': X.shape[1],
        'mae': mae,
        'rmse': rmse,
        'training_time': training_time,
        'total_time': label_time
    }
    
    print(f"\nâœ… {label} COMPLETED!")
    print(f"   â€¢ Total time for {label}: {label_time:.2f}s")
    
    # Progress update
    progress = (label_idx + 1) / len(labels) * 100
    elapsed_time = time.time() - total_start_time
    estimated_total = elapsed_time / (label_idx + 1) * len(labels)
    eta = estimated_total - elapsed_time
    
    print(f"\nğŸ“Š OVERALL PROGRESS: {progress:.1f}%")
    print(f"   â€¢ Elapsed time: {elapsed_time:.1f}s")
    print(f"   â€¢ ETA: {eta:.1f}s ({eta/60:.1f} minutes)")
    
    # Clear GPU memory if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"   â€¢ GPU memory cleared")

# Final summary
total_time = time.time() - total_start_time
print(f"\n{'='*60}")
print(f"ğŸ�‰ TRAINING PIPELINE COMPLETED!")
print(f"{'='*60}")
print(f"â�° Total execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
print(f"\nğŸ“Š SUMMARY BY LABEL:")
print(f"{'Label':<10} {'Samples':<8} {'Features':<10} {'MAE':<12} {'RMSE':<12} {'Time(s)':<10}")
print("-" * 70)
for label in labels:
    stats = label_results[label]
    print(f"{label:<10} {stats['samples']:<8} {stats['features']:<10} "
          f"{stats['mae']:<12.6f} {stats['rmse']:<12.6f} {stats['total_time']:<10.1f}")

print(f"\nğŸ“ˆ PERFORMANCE METRICS:")
avg_mae = np.mean([label_results[label]['mae'] for label in labels])
avg_rmse = np.mean([label_results[label]['rmse'] for label in labels])
print(f"   â€¢ Average MAE across all labels: {avg_mae:.6f}")
print(f"   â€¢ Average RMSE across all labels: {avg_rmse:.6f}")

print("\nğŸ“‹ FINAL PREDICTIONS:")
print(output_df)

# Save submission
print(f"\nğŸ’¾ SAVING RESULTS...")
save_start = time.time()
output_df.to_csv('submission.csv', index=False)
save_time = time.time() - save_start
print(f"   âœ… Submission saved to 'submission.csv' in {save_time:.3f}s")

print(f"\nğŸ�¯ PIPELINE FINISHED SUCCESSFULLY!")
print(f"   Total time: {total_time:.1f}s")
print(f"   Average time per label: {total_time/len(labels):.1f}s")
print("=" * 60)

