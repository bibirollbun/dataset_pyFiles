# Kaggleã‚ªãƒ•ãƒ©ã‚¤ãƒ³ç’°å¢ƒç”¨ä¾�å­˜é–¢ä¿‚ã‚¤ãƒ³ã‚¹ãƒˆãƒ¼ãƒ«ï¼ˆä¿®æ­£ç‰ˆï¼‰
import subprocess
import sys
import os
import warnings
warnings.filterwarnings('ignore')

def install_rdkit_from_wheels():
    """wheelãƒ•ã‚¡ã‚¤ãƒ«ã�‹ã‚‰ç›´æ�¥RDKitã‚’ã‚¤ãƒ³ã‚¹ãƒˆãƒ¼ãƒ«"""
    rdkit_dataset_path = "/kaggle/input/rdkit-install-whl"
    wheel_dir = os.path.join(rdkit_dataset_path, "rdkit_wheel")
    
    if not os.path.exists(wheel_dir):
        print(f"â�Œ wheelãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªã�Œè¦‹ã�¤ã�‹ã‚Šã�¾ã�›ã‚“: {wheel_dir}")
        return False
    
    print(f"ğŸ“¦ wheelãƒ•ã‚¡ã‚¤ãƒ«ã�‹ã‚‰ç›´æ�¥RDKitã‚’ã‚¤ãƒ³ã‚¹ãƒˆãƒ¼ãƒ«")
    print(f"ğŸ“‚ wheelãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒª: {wheel_dir}")
    
    # wheelãƒ•ã‚¡ã‚¤ãƒ«ä¸€è¦§è¡¨ç¤º
    wheel_files = [f for f in os.listdir(wheel_dir) if f.endswith('.whl')]
    print("ğŸ“‚ åˆ©ç”¨å�¯èƒ½ã�ªwheelãƒ•ã‚¡ã‚¤ãƒ«:")
    for wf in wheel_files:
        print(f"  - {wf}")
    
    try:
        # RDKitã�®wheelãƒ•ã‚¡ã‚¤ãƒ«ã‚’ç‰¹å®šã�—ã�¦ã‚¤ãƒ³ã‚¹ãƒˆãƒ¼ãƒ«
        rdkit_wheel = None
        for wf in wheel_files:
            if 'rdkit' in wf.lower():
                rdkit_wheel = os.path.join(wheel_dir, wf)
                break
        
        if rdkit_wheel:
            print(f"ğŸ“¦ RDKitã‚¤ãƒ³ã‚¹ãƒˆãƒ¼ãƒ«é–‹å§‹: {os.path.basename(rdkit_wheel)}")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", rdkit_wheel, "--quiet"
            ])
            print("âœ… RDKit ã‚¤ãƒ³ã‚¹ãƒˆãƒ¼ãƒ«å®Œäº†")
            return True
        else:
            print("â�Œ RDKit wheelãƒ•ã‚¡ã‚¤ãƒ«ã�Œè¦‹ã�¤ã�‹ã‚Šã�¾ã�›ã‚“")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"â�Œ RDKit ã‚¤ãƒ³ã‚¹ãƒˆãƒ¼ãƒ«å¤±æ•—: {e}")
        return False

# RDKitã‚¤ãƒ³ã‚¹ãƒˆãƒ¼ãƒ«è©¦è¡Œ
print("ğŸš€ RDKit wheelãƒ•ã‚¡ã‚¤ãƒ«ã�‹ã‚‰ç›´æ�¥ã‚¤ãƒ³ã‚¹ãƒˆãƒ¼ãƒ«é–‹å§‹")
print("=" * 60)

install_success = install_rdkit_from_wheels()

print("=" * 60)

# RDKitã�®å‹•ä½œç¢ºèª�ã�¨ãƒ•ãƒ©ã‚°è¨­å®š
RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, MACCSkeys
    from rdkit.ML.Descriptors import MoleculeDescriptors
    RDKIT_AVAILABLE = True
    print("âœ… RDKité–¢é€£ã‚¤ãƒ³ãƒ�ãƒ¼ãƒˆãƒ†ã‚¹ãƒˆæˆ�åŠŸ")
    
    # ç°¡å�˜ã�ªå‹•ä½œãƒ†ã‚¹ãƒˆ
    mol = Chem.MolFromSmiles('CCO')
    if mol is not None:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        print(f"âœ… RDKitå‹•ä½œãƒ†ã‚¹ãƒˆæˆ�åŠŸ")
        print(f"  ã‚¨ã‚¿ãƒ�ãƒ¼ãƒ«åˆ†å­�é‡�: {mw:.2f}")
        print(f"  ã‚¨ã‚¿ãƒ�ãƒ¼ãƒ«LogP: {logp:.2f}")
        
except ImportError as e:
    print(f"â�Œ RDKitã‚¤ãƒ³ãƒ�ãƒ¼ãƒˆã‚¨ãƒ©ãƒ¼: {e}")
    print("âš ï¸�  RDKitã�ªã�—ã�§ãƒ•ã‚©ãƒ¼ãƒ«ãƒ�ãƒƒã‚¯å®Ÿè£…ã‚’ä½¿ç”¨")

print(f"\nRDKitåˆ©ç”¨å�¯èƒ½: {RDKIT_AVAILABLE}")
print("\nğŸ�¯ ä¾�å­˜é–¢ä¿‚ã‚¤ãƒ³ã‚¹ãƒˆãƒ¼ãƒ«å®Œäº†")

# NeurIPS Open Polymer Prediction 2025 - é«˜åº¦ã�ªã‚¢ãƒ³ã‚µãƒ³ãƒ–ãƒ«ãƒ¢ãƒ‡ãƒ«
import numpy as np
import pandas as pd

# å…¥åŠ›ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªã�®ç¢ºèª�
print("\nğŸ“� åˆ©ç”¨å�¯èƒ½ã�ªå…¥åŠ›ãƒ‡ãƒ¼ã‚¿:")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# å¿…è¦�ã�ªãƒ©ã‚¤ãƒ–ãƒ©ãƒªã�®ã‚¤ãƒ³ãƒ�ãƒ¼ãƒˆï¼ˆRDKitãƒ•ã‚©ãƒ¼ãƒ«ãƒ�ãƒƒã‚¯å¯¾å¿œï¼‰
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
import time
import warnings
warnings.filterwarnings('ignore')

# RDKité–¢é€£ã�®ã‚¤ãƒ³ãƒ�ãƒ¼ãƒˆï¼ˆãƒ•ã‚©ãƒ¼ãƒ«ãƒ�ãƒƒã‚¯å¯¾å¿œï¼‰
RDKIT_AVAILABLE = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, MACCSkeys
    from rdkit.ML.Descriptors import MoleculeDescriptors
    RDKIT_AVAILABLE = True
    print("âœ… RDKitåˆ©ç”¨å�¯èƒ½ - é«˜ç²¾åº¦åˆ†å­�ç‰¹å¾´é‡�ã‚’ä½¿ç”¨")
except ImportError:
    print("âš ï¸�  RDKitåˆ©ç”¨ä¸�å�¯ - åŸºæœ¬ç‰¹å¾´é‡�ã�®ã�¿ã‚’ä½¿ç”¨")

print("ãƒ©ã‚¤ãƒ–ãƒ©ãƒªã‚¤ãƒ³ãƒ�ãƒ¼ãƒˆå®Œäº†")


# ãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿ã�¨åŸºæœ¬åˆ†æ��
print(f"é«˜åº¦ã�ªãƒ�ãƒªãƒ�ãƒ¼ç‰¹æ€§äºˆæ¸¬ãƒ‘ã‚¤ãƒ—ãƒ©ã‚¤ãƒ³é–‹å§‹...")
print(f"å®Ÿè¡Œæ—¥æ™‚: 2025-06-23")
start_time = time.time()

# å†�ç�¾æ€§ã�®ã�Ÿã‚�ã�®ãƒ©ãƒ³ãƒ€ãƒ ã‚·ãƒ¼ãƒ‰è¨­å®š
SEED = 42
np.random.seed(SEED)

# ãƒ‡ãƒ¼ã‚¿èª­ã�¿è¾¼ã�¿
train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')

print(f"è¨“ç·´ãƒ‡ãƒ¼ã‚¿å½¢çŠ¶: {train.shape}")
print(f"ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿å½¢çŠ¶: {test.shape}")

# æ¬ æ��å€¤ç¢ºèª�
print("\nè¨“ç·´ãƒ‡ãƒ¼ã‚¿ã�®æ¬ æ��å€¤:")
missing_values = train.isnull().sum()
print(missing_values)

# å�„ç‰¹æ€§ã�®åˆ©ç”¨å�¯èƒ½ã‚µãƒ³ãƒ—ãƒ«æ•°è¨ˆç®—
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
available_samples = {col: train.shape[0] - missing_values[col] for col in target_cols}
print("\nå�„ç‰¹æ€§ã�®åˆ©ç”¨å�¯èƒ½ã‚µãƒ³ãƒ—ãƒ«æ•°:")
for col, count in available_samples.items():
    print(f"{col}: {count} ({count/train.shape[0]*100:.2f}%)")

# ã‚¿ãƒ¼ã‚²ãƒƒãƒˆç‰¹æ€§ã�®çµ±è¨ˆæƒ…å ±
print("\nã‚¿ãƒ¼ã‚²ãƒƒãƒˆç‰¹æ€§çµ±è¨ˆ:")
print(train[target_cols].describe())

# ç‰¹æ€§ã�®ç¯„å›²è¨ˆç®—ï¼ˆwMAEè¨ˆç®—ã�«å¿…è¦�ï¼‰
property_ranges = {}
for col in target_cols:
    property_ranges[col] = train[col].dropna().max() - train[col].dropna().min()
    
print("\nç‰¹æ€§ã�®æ�¨å®šç¯„å›²:")
for col, range_val in property_ranges.items():
    print(f"{col}: {range_val:.4f}")

# wMAEãƒ¡ãƒˆãƒªãƒƒã‚¯ç”¨ã�®é‡�ã�¿è¨ˆç®—
weights = {}
for col in target_cols:
    # é‡�ã�¿ = (1/sqrt(n_i)) / range_i, æ­£è¦�åŒ–æ¸ˆã�¿
    weights[col] = (1 / np.sqrt(available_samples[col])) / property_ranges[col]

# é‡�ã�¿ã‚’æ­£è¦�åŒ–ï¼ˆã‚¿ã‚¹ã‚¯æ•°=5ã�«å�ˆã‚�ã�›ã‚‹ï¼‰
weight_sum = sum(weights.values())
for col in target_cols:
    weights[col] = weights[col] / weight_sum * len(target_cols)

print("\nwMAEãƒ¡ãƒˆãƒªãƒƒã‚¯ç”¨æ�¨å®šé‡�ã�¿:")
for col, weight in weights.items():
    print(f"{col}: {weight:.4f}")

# RDKitè¨˜è¿°å­�è¨ˆç®—å™¨ã�®åˆ�æœŸåŒ–ï¼ˆRDKitåˆ©ç”¨å�¯èƒ½æ™‚ã�®ã�¿ï¼‰
if RDKIT_AVAILABLE:
    descriptor_names = [x[0] for x in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    print("RDKitè¨˜è¿°å­�è¨ˆç®—å™¨åˆ�æœŸåŒ–å®Œäº†")
else:
    print("RDKitä¸�ä½¿ç”¨ã�®ã�Ÿã‚�åŸºæœ¬ç‰¹å¾´é‡�ã�®ã�¿ã‚’ä½¿ç”¨")


# ãƒ�ãƒªãƒ�ãƒ¼å›ºæœ‰ç‰¹å¾´é‡�æŠ½å‡ºé–¢æ•°ã�®å®šç¾©
def get_safe_polymer_features(mol):
    """å•�é¡Œã�®ã�‚ã‚‹é–¢æ•°ã‚’ä½¿ã‚�ã�šã�«åŸºæœ¬çš„ã�ªãƒ�ãƒªãƒ�ãƒ¼å›ºæœ‰ç‰¹å¾´é‡�ã‚’æŠ½å‡º"""
    features = {}
    
    # åŸºæœ¬çš„ã�ªã‚«ã‚¦ãƒ³ãƒˆ
    features['num_atoms'] = mol.GetNumAtoms()
    features['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
    features['num_bonds'] = mol.GetNumBonds()
    
    # å�Ÿå­�ã‚¿ã‚¤ãƒ—ã�®ã‚«ã‚¦ãƒ³ãƒˆ
    atom_types = {'C': 0, 'N': 0, 'O': 0, 'S': 0, 'F': 0, 'Cl': 0, 'Br': 0, 'I': 0}
    aromatic_atoms = 0
    
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in atom_types:
            atom_types[symbol] += 1
        
        # èŠ³é¦™æ—�å�Ÿå­�ã�®ã‚«ã‚¦ãƒ³ãƒˆ
        if atom.GetIsAromatic():
            aromatic_atoms += 1
    
    features['aromatic_atoms'] = aromatic_atoms
    
    # å�Ÿå­�ã‚¿ã‚¤ãƒ—ã‚«ã‚¦ãƒ³ãƒˆã‚’ç‰¹å¾´é‡�ã�«è¿½åŠ 
    for atom_type, count in atom_types.items():
        features[f'num_{atom_type}'] = count
        
        # é‡�å�Ÿå­�æ•°ã�«å¯¾ã�™ã‚‹æ¯”ç�‡è¨ˆç®—
        if mol.GetNumHeavyAtoms() > 0:
            features[f'ratio_{atom_type}'] = count / mol.GetNumHeavyAtoms()
        else:
            features[f'ratio_{atom_type}'] = 0
    
    # çµ�å�ˆã‚¿ã‚¤ãƒ—ã�®ã‚«ã‚¦ãƒ³ãƒˆ
    bond_types = {Chem.rdchem.BondType.SINGLE: 0, 
                  Chem.rdchem.BondType.DOUBLE: 0, 
                  Chem.rdchem.BondType.TRIPLE: 0,
                  Chem.rdchem.BondType.AROMATIC: 0}
    
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type in bond_types:
            bond_types[bond_type] += 1
    
    # çµ�å�ˆã‚¿ã‚¤ãƒ—ã‚«ã‚¦ãƒ³ãƒˆã‚’ç‰¹å¾´é‡�ã�«è¿½åŠ 
    features['num_single_bonds'] = bond_types[Chem.rdchem.BondType.SINGLE]
    features['num_double_bonds'] = bond_types[Chem.rdchem.BondType.DOUBLE]
    features['num_triple_bonds'] = bond_types[Chem.rdchem.BondType.TRIPLE]
    features['num_aromatic_bonds'] = bond_types[Chem.rdchem.BondType.AROMATIC]
    
    # ä¿¡é ¼æ€§ã�®é«˜ã�„è¨˜è¿°å­�ã�®è¨ˆç®—
    try:
        features['mw'] = Descriptors.MolWt(mol)
        features['logp'] = Descriptors.MolLogP(mol)
        features['tpsa'] = Descriptors.TPSA(mol)
        features['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
        features['num_h_donors'] = Descriptors.NumHDonors(mol)
        features['num_h_acceptors'] = Descriptors.NumHAcceptors(mol)
        features['num_rings'] = Descriptors.RingCount(mol)
        features['num_aromatic_rings'] = Descriptors.NumAromaticRings(mol)
        features['num_aliphatic_rings'] = Descriptors.NumAliphaticRings(mol)
    except:
        # è¨˜è¿°å­�ã�®è¨ˆç®—ã�«å¤±æ•—ã�—ã�Ÿå ´å�ˆã�¯0ã�«è¨­å®š
        for desc in ['mw', 'logp', 'tpsa', 'num_rotatable_bonds', 'num_h_donors', 
                     'num_h_acceptors', 'num_rings', 'num_aromatic_rings', 'num_aliphatic_rings']:
            if desc not in features:
                features[desc] = 0
    
    # ãƒ�ãƒªãƒ�ãƒ¼é–¢é€£ã�®ã‚«ã‚¹ã‚¿ãƒ æ¯”ç�‡
    if mol.GetNumHeavyAtoms() > 0:
        features['rotatable_per_heavy'] = features['num_rotatable_bonds'] / mol.GetNumHeavyAtoms()
        features['rings_per_heavy'] = features.get('num_rings', 0) / mol.GetNumHeavyAtoms()
        features['aromatic_atom_ratio'] = features.get('aromatic_atoms', 0) / mol.GetNumHeavyAtoms()
    else:
        features['rotatable_per_heavy'] = 0
        features['rings_per_heavy'] = 0
        features['aromatic_atom_ratio'] = 0
    
    return features

print("ãƒ�ãƒªãƒ�ãƒ¼å›ºæœ‰ç‰¹å¾´é‡�æŠ½å‡ºé–¢æ•°å®šç¾©å®Œäº†")


# SMILESæ–‡å­—åˆ—ã�‹ã‚‰ã�®ç‰¹å¾´é‡�ç”Ÿæˆ�é–¢æ•°ï¼ˆRDKitãƒ•ã‚©ãƒ¼ãƒ«ãƒ�ãƒƒã‚¯å¯¾å¿œï¼‰
def generate_basic_smiles_features(smiles_list):
    """RDKitã�ªã�—ã�§ã�®SMILESåŸºæœ¬ç‰¹å¾´é‡�ç”Ÿæˆ�"""
    print("RDKitã�ªã�—ã�§ã�®åŸºæœ¬SMILESç‰¹å¾´é‡�ç”Ÿæˆ�ä¸­...")
    
    features = []
    feature_names = [
        'smiles_length', 'num_C', 'num_N', 'num_O', 'num_S', 'num_F', 'num_Cl', 'num_Br',
        'num_equals', 'num_hash', 'num_parens', 'num_brackets', 'num_rings_estimated',
        'aromatic_estimated', 'double_bonds_estimated', 'triple_bonds_estimated'
    ]
    
    for smiles in smiles_list:
        try:
            # åŸºæœ¬çš„ã�ªæ–‡å­—åˆ—è§£æ��ã�«ã‚ˆã‚‹ç‰¹å¾´é‡�
            feat = {}
            feat['smiles_length'] = len(smiles)
            feat['num_C'] = smiles.count('C')
            feat['num_N'] = smiles.count('N')
            feat['num_O'] = smiles.count('O')
            feat['num_S'] = smiles.count('S')
            feat['num_F'] = smiles.count('F')
            feat['num_Cl'] = smiles.count('Cl')
            feat['num_Br'] = smiles.count('Br')
            feat['num_equals'] = smiles.count('=')
            feat['num_hash'] = smiles.count('#')
            feat['num_parens'] = smiles.count('(') + smiles.count(')')
            feat['num_brackets'] = smiles.count('[') + smiles.count(']')
            feat['num_rings_estimated'] = smiles.count('1') + smiles.count('2') + smiles.count('3')
            feat['aromatic_estimated'] = smiles.count('c') + smiles.count('n') + smiles.count('o')
            feat['double_bonds_estimated'] = smiles.count('=')
            feat['triple_bonds_estimated'] = smiles.count('#')
            
            features.append([feat[name] for name in feature_names])
        except:
            # ã‚¨ãƒ©ãƒ¼æ™‚ã�¯ã‚¼ãƒ­åŸ‹ã‚�
            features.append([0] * len(feature_names))
    
    return pd.DataFrame(features, columns=feature_names), list(range(len(smiles_list)))

def generate_molecule_features(smiles_list):
    """SMILESæ–‡å­—åˆ—ã�‹ã‚‰åˆ†å­�ç‰¹å¾´é‡�ã‚’ç”Ÿæˆ�ï¼ˆRDKitãƒ•ã‚©ãƒ¼ãƒ«ãƒ�ãƒƒã‚¯å¯¾å¿œï¼‰"""
    if RDKIT_AVAILABLE:
        return generate_rdkit_features(smiles_list)
    else:
        return generate_basic_smiles_features(smiles_list)

def generate_rdkit_features(smiles_list):
    """RDKitä½¿ç”¨æ™‚ã�®é«˜ç²¾åº¦åˆ†å­�ç‰¹å¾´é‡�ç”Ÿæˆ�"""
    print("RDKitä½¿ç”¨ã�§ã�®é«˜ç²¾åº¦åˆ†å­�ç‰¹å¾´é‡�ç”Ÿæˆ�ä¸­...")
    
    # ã‚µãƒ³ãƒ—ãƒ«åˆ†å­�ã�§æ­£ç¢ºã�ªç‰¹å¾´é‡�æ§‹é€ ã‚’ä½œæˆ�
    sample_mol = Chem.MolFromSmiles('CC')
    sample_descriptors = list(calculator.CalcDescriptors(sample_mol))
    sample_polymer_features = get_safe_polymer_features(sample_mol)
    
    # ã‚µãƒ³ãƒ—ãƒ«ãƒ•ã‚£ãƒ³ã‚¬ãƒ¼ãƒ—ãƒªãƒ³ãƒˆã�®è¨ˆç®—
    sample_morgan_fp = AllChem.GetMorganFingerprintAsBitVect(sample_mol, 2, nBits=256)
    sample_morgan_features = np.zeros((256,))
    AllChem.DataStructs.ConvertToNumpyArray(sample_morgan_fp, sample_morgan_features)
    
    sample_maccs_fp = MACCSkeys.GenMACCSKeys(sample_mol)
    sample_maccs_features = np.zeros((167,))
    AllChem.DataStructs.ConvertToNumpyArray(sample_maccs_fp, sample_maccs_features)
    
    # ã‚µãƒ³ãƒ—ãƒ«ã�«åŸºã�¥ã��ç‰¹å¾´é‡�å��ã�®ä½œæˆ�
    descriptor_names = [x[0] for x in Descriptors._descList]
    polymer_feature_names = list(sample_polymer_features.keys())
    morgan_feature_names = [f'morgan_{i}' for i in range(len(sample_morgan_features))]
    maccs_feature_names = [f'maccs_{i}' for i in range(len(sample_maccs_features))]
    
    # å…¨ç‰¹å¾´é‡�å��ã‚’çµ�å�ˆ
    feature_names = descriptor_names + polymer_feature_names + morgan_feature_names + maccs_feature_names
    
    # ç©ºã�®ç‰¹å¾´é‡�ãƒªã‚¹ãƒˆã‚’ä½œæˆ�
    features = []
    valid_indices = []
    
    for i, smiles in enumerate(smiles_list):
        try:
            # æ§˜ã€…ã�ªSMILESè§£æ��ã‚¢ãƒ—ãƒ­ãƒ¼ãƒ�ã‚’è©¦è¡Œ
            mol = None
            
            # 1. ç›´æ�¥è§£æ��ã‚’è©¦è¡Œ
            mol = Chem.MolFromSmiles(smiles)
            
            # 2. å¤±æ•—æ™‚ã€�ãƒ�ãƒªãƒ�ãƒ¼SMILESè¨˜æ³•ã�®å�¯èƒ½æ€§ã‚’è€ƒæ…®
            if mol is None and '*' in smiles:
                modified_smiles = smiles.replace('*', 'C')
                mol = Chem.MolFromSmiles(modified_smiles)
            
            # 3. ã�¾ã� å¤±æ•—ã�®å ´å�ˆã€�ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ä»˜ã��ã‚µãƒ‹ã‚¿ã‚¤ã‚¼ãƒ¼ã‚·ãƒ§ãƒ³ã‚’è©¦è¡Œ
            if mol is None:
                mol = Chem.MolFromSmiles(smiles, sanitize=False)
                if mol is not None:
                    try:
                        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    except:
                        pass
            
            if mol is not None:
                # RDKitè¨˜è¿°å­�ã�®è¨ˆç®—
                descriptors = list(calculator.CalcDescriptors(mol))
                
                # ãƒ�ãƒªãƒ�ãƒ¼å›ºæœ‰ç‰¹å¾´é‡�ã�®è¨ˆç®—
                polymer_features = list(get_safe_polymer_features(mol).values())
                
                # ãƒ•ã‚£ãƒ³ã‚¬ãƒ¼ãƒ—ãƒªãƒ³ãƒˆã�®è¨ˆç®—
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Morganãƒ•ã‚£ãƒ³ã‚¬ãƒ¼ãƒ—ãƒªãƒ³ãƒˆ
                    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256)
                    morgan_features = np.zeros((256,))
                    AllChem.DataStructs.ConvertToNumpyArray(morgan_fp, morgan_features)
                    
                    # MACCSã‚­ãƒ¼
                    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
                    maccs_features = np.zeros((167,))
                    AllChem.DataStructs.ConvertToNumpyArray(maccs_fp, maccs_features)
                
                # å…¨ç‰¹å¾´é‡�ã‚’çµ�å�ˆ
                all_features = descriptors + polymer_features + list(morgan_features) + list(maccs_features)
                
                # æ­£ã�—ã�„ç‰¹å¾´é‡�æ•°ã‚’ç¢ºä¿�
                if len(all_features) != len(feature_names):
                    if len(all_features) < len(feature_names):
                        all_features = all_features + [0] * (len(feature_names) - len(all_features))
                    else:
                        all_features = all_features[:len(feature_names)]
                
                features.append(all_features)
                valid_indices.append(i)
            else:
                if i < 5:  # ãƒ‡ãƒ�ãƒƒã‚°ç”¨ã�«æœ€åˆ�ã�®æ•°å€‹ã�®å¤±æ•—ã‚’è¡¨ç¤º
                    print(f"è­¦å‘Š: SMILESè§£æ��ä¸�å�¯: {smiles}")
        except Exception as e:
            if i < 5:  # ãƒ‡ãƒ�ãƒƒã‚°ç”¨ã�«æœ€åˆ�ã�®æ•°å€‹ã�®å¤±æ•—ã‚’è¡¨ç¤º
                print(f"SMILES {smiles} å‡¦ç�†ã‚¨ãƒ©ãƒ¼: {str(e)}")
    
    print(f"{len(smiles_list)} ä¸­ {len(valid_indices)} ã�®SMILESæ–‡å­—åˆ—ã‚’æ­£å¸¸ã�«å‡¦ç�†")
    print(f"ç‰¹å¾´é‡�ãƒ™ã‚¯ãƒˆãƒ«é•·: {len(feature_names)}")
    
    # å…¨ç‰¹å¾´é‡�ãƒ™ã‚¯ãƒˆãƒ«ã�Œ feature_names ã�¨å�Œã�˜é•·ã�•ã�§ã�‚ã‚‹ã�“ã�¨ã‚’å†�ç¢ºèª�
    for i, feat in enumerate(features):
        if len(feat) != len(feature_names):
            features[i] = feat[:len(feature_names)] if len(feat) > len(feature_names) else feat + [0] * (len(feature_names) - len(feat))
    
    return pd.DataFrame(features, index=valid_indices, columns=feature_names), valid_indices

print("åˆ†å­�ç‰¹å¾´é‡�ç”Ÿæˆ�é–¢æ•°å®šç¾©å®Œäº†")


# åˆ†å­�ç‰¹å¾´é‡�ç”Ÿæˆ�å®Ÿè¡Œ
print("\nSMILESã�‹ã‚‰ã�®é«˜åº¦ã�ªåˆ†å­�ç‰¹å¾´é‡�ç”Ÿæˆ�ä¸­...")
train_features, train_valid_idx = generate_molecule_features(train['SMILES'])
test_features, test_valid_idx = generate_molecule_features(test['SMILES'])

# ç‰¹å¾´é‡�ã‚’å…ƒãƒ‡ãƒ¼ã‚¿ã�¨çµ�å�ˆ
train_with_features = pd.concat([
    train.iloc[train_valid_idx].reset_index(drop=True),
    train_features.reset_index(drop=True)
], axis=1)

test_with_features = pd.concat([
    test.iloc[test_valid_idx].reset_index(drop=True),
    test_features.reset_index(drop=True)
], axis=1)

print(f"ç‰¹å¾´é‡�ä»˜ã��è¨“ç·´ãƒ‡ãƒ¼ã‚¿å½¢çŠ¶: {train_with_features.shape}")
print(f"ç‰¹å¾´é‡�ä»˜ã��ãƒ†ã‚¹ãƒˆãƒ‡ãƒ¼ã‚¿å½¢çŠ¶: {test_with_features.shape}")


# ç‰¹å¾´é‡�ã‚¯ãƒªãƒ¼ãƒ‹ãƒ³ã‚°é–¢æ•°
def clean_features(df, feature_cols):
    """infå€¤é™¤å�»ã�¨æ¥µç«¯ã�ªå¤–ã‚Œå€¤ã�®å‡¦ç�†"""
    df_clean = df.copy()
    
    # ç„¡é™�å€¤ã‚’NaNã�§ç½®æ�›
    df_clean[feature_cols] = df_clean[feature_cols].replace([np.inf, -np.inf], np.nan)
    
    # å…¨ã�¦NaNã�¾ã�Ÿã�¯å®šæ•°å€¤ã�®åˆ—ã‚’ãƒ�ã‚§ãƒƒã‚¯
    valid_cols = []
    for col in feature_cols:
        if df_clean[col].notna().sum() > 0 and df_clean[col].nunique() > 1:
            valid_cols.append(col)
    
    print(f"ç„¡åŠ¹åˆ—é™¤å�»å¾Œ: {len(valid_cols)} / {len(feature_cols)} ç‰¹å¾´é‡�ã‚’ä¿�æŒ�")
    
    # æ¥µç«¯ã�ªå¤–ã‚Œå€¤ã�®æ¤œå‡ºã�¨ã‚­ãƒ£ãƒƒãƒ—ï¼ˆ5æ¨™æº–å��å·®è¶…ï¼‰
    for col in valid_cols:
        if df_clean[col].dtype.kind in 'fc':  # æ•°å€¤åˆ—ã�®ã�¿
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            if std > 0:
                lower_bound = mean - 5 * std
                upper_bound = mean + 5 * std
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean, valid_cols

# å…¨ã�¦ã�®å�¯èƒ½ã�ªç‰¹å¾´é‡�åˆ—ã‚’å�–å¾—
all_feature_cols = train_features.columns.tolist()
print(f"ç‰¹å¾´é‡�ç·�æ•°: {len(all_feature_cols)}")

train_with_features, valid_feature_cols = clean_features(train_with_features, all_feature_cols)
test_with_features, _ = clean_features(test_with_features, all_feature_cols)


# ã‚¿ãƒ¼ã‚²ãƒƒãƒˆåˆ¥ç‰¹å¾´é‡�é�¸æŠ�
def select_features_for_target(df, target_col, all_features, max_features=500):
    """ç‰¹å®šã‚¿ãƒ¼ã‚²ãƒƒãƒˆã�®æœ€é‡�è¦�ç‰¹å¾´é‡�é�¸æŠ�"""
    # ã‚¿ãƒ¼ã‚²ãƒƒãƒˆå€¤ã�®ã�‚ã‚‹è¡Œã�«ãƒ•ã‚£ãƒ«ã‚¿
    df_valid = df[df[target_col].notna()].copy()
    
    if len(df_valid) < 50:  # ãƒ‡ãƒ¼ã‚¿ä¸�è¶³
        return all_features[:min(len(all_features), max_features)]
    
    # ã‚¿ãƒ¼ã‚²ãƒƒãƒˆã�¨ã�®ç›¸é–¢ç¢ºèª�
    correlations = []
    for col in all_features:
        if df_valid[col].dtype.kind in 'fc':  # æ•°å€¤åˆ—ã�®ã�¿
            corr = df_valid[col].corr(df_valid[target_col])
            if not pd.isna(corr):
                correlations.append((col, abs(corr)))
    
    # ç›¸é–¢é †ã�§ã‚½ãƒ¼ãƒˆã�—ã€�ä¸Šä½�ç‰¹å¾´é‡�ã‚’é�¸æŠ�
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_features = [x[0] for x in correlations[:max_features]]
    
    print(f"{target_col} ç”¨ã�«ç›¸é–¢ãƒ™ãƒ¼ã‚¹ã�§ {len(top_features)} ç‰¹å¾´é‡�ã‚’é�¸æŠ�")
    return top_features

# ã‚¿ãƒ¼ã‚²ãƒƒãƒˆå›ºæœ‰ç‰¹å¾´é‡�ã‚»ãƒƒãƒˆã�®æº–å‚™
target_features = {}
for col in target_cols:
    target_features[col] = select_features_for_target(train_with_features, col, valid_feature_cols)


# å�„ç‰¹æ€§ã�®æœ€é�©åŒ–æ¸ˆã�¿ãƒ�ã‚¤ãƒ‘ãƒ¼ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿
def get_property_hyperparams(property_name):
    """å�„ç‰¹æ€§ã�®æœ€é�©åŒ–æ¸ˆã�¿ãƒ�ã‚¤ãƒ‘ãƒ¼ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ã‚’å�–å¾—"""
    # å�„ç‰¹æ€§ç”¨ã�®äº‹å‰�æœ€é�©åŒ–æ¸ˆã�¿ãƒ�ã‚¤ãƒ‘ãƒ¼ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿
    params = {
        'Tg': {
            'xgb': {
                'n_estimators': 1000,
                'learning_rate': 0.01,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'reg_alpha': 0.01,
                'reg_lambda': 1.0
            },
            'cat': {
                'iterations': 1000,
                'learning_rate': 0.03,
                'depth': 6,
                'l2_leaf_reg': 3
            }
        },
        'FFV': {
            'xgb': {
                'n_estimators': 2000,
                'learning_rate': 0.005,
                'max_depth': 8,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_weight': 2,
                'reg_alpha': 0.1,
                'reg_lambda': 0.5
            },
            'cat': {
                'iterations': 1500,
                'learning_rate': 0.02,
                'depth': 7,
                'l2_leaf_reg': 2
            }
        },
        'Tc': {
            'xgb': {
                'n_estimators': 1500,
                'learning_rate': 0.01,
                'max_depth': 7,
                'subsample': 0.85,
                'colsample_bytree': 0.75,
                'min_child_weight': 3,
                'reg_alpha': 0.05,
                'reg_lambda': 1.0
            },
            'cat': {
                'iterations': 1200,
                'learning_rate': 0.02,
                'depth': 5,
                'l2_leaf_reg': 4
            }
        },
        'Density': {
            'xgb': {
                'n_estimators': 1200,
                'learning_rate': 0.01,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 2,
                'reg_alpha': 0.1,
                'reg_lambda': 0.5
            },
            'cat': {
                'iterations': 1000,
                'learning_rate': 0.03,
                'depth': 6,
                'l2_leaf_reg': 3
            }
        },
        'Rg': {
            'xgb': {
                'n_estimators': 1000,
                'learning_rate': 0.02,
                'max_depth': 7,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'min_child_weight': 3,
                'reg_alpha': 0.05,
                'reg_lambda': 1.0
            },
            'cat': {
                'iterations': 1200,
                'learning_rate': 0.02,
                'depth': 7,
                'l2_leaf_reg': 3
            }
        }
    }
    
    return params.get(property_name, params['Tg'])  # ç‰¹æ€§ã�Œè¦‹ã�¤ã�‹ã‚‰ã�ªã�„å ´å�ˆã�¯Tgã‚’ãƒ‡ãƒ•ã‚©ãƒ«ãƒˆ

print("ãƒ�ã‚¤ãƒ‘ãƒ¼ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿å®šç¾©å®Œäº†")


# ã‚¢ãƒ³ã‚µãƒ³ãƒ–ãƒ«äºˆæ¸¬ç”¨ã‚¯ãƒ©ã‚¹
class AveragingEnsemble:
    def __init__(self, models, imputer, scaler):
        self.models = models
        self.imputer = imputer
        self.scaler = scaler
        
    def predict(self, X):
        try:
            X = X.copy()
            X = X.replace([np.inf, -np.inf], np.nan)
            X_imputed = self.imputer.transform(X)
            X_imputed = np.nan_to_num(X_imputed, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = self.scaler.transform(X_imputed)
            
            preds = np.column_stack([model.predict(X_scaled) for model in self.models])
            return np.mean(preds, axis=1)
        except Exception as e:
            print(f"äºˆæ¸¬ã‚¨ãƒ©ãƒ¼: {str(e)}")
            return np.zeros(X.shape[0])

# ãƒ•ã‚©ãƒ¼ãƒ«ãƒ�ãƒƒã‚¯ç”¨å¹³å�‡äºˆæ¸¬å™¨
class MeanPredictor:
    def __init__(self, value):
        self.value = value
        
    def predict(self, X):
        return np.full(len(X), self.value)

print("ã‚¢ãƒ³ã‚µãƒ³ãƒ–ãƒ«ãƒ¢ãƒ‡ãƒ«ã‚¯ãƒ©ã‚¹å®šç¾©å®Œäº†")


# é«˜åº¦ã�ªã‚¢ãƒ³ã‚µãƒ³ãƒ–ãƒ«ãƒ¢ãƒ‡ãƒ«è¨“ç·´
def train_advanced_model(df, target_col, feature_cols, n_splits=5):
    """ç‰¹å®šã�®ã‚¿ãƒ¼ã‚²ãƒƒãƒˆç‰¹æ€§ç”¨ã�®é«˜åº¦ã�ªã‚¢ãƒ³ã‚µãƒ³ãƒ–ãƒ«ãƒ¢ãƒ‡ãƒ«è¨“ç·´"""
    # ã�“ã�®ã‚¿ãƒ¼ã‚²ãƒƒãƒˆã�®æœ‰åŠ¹è¡Œã�«ãƒ•ã‚£ãƒ«ã‚¿
    valid_idx = df[target_col].notna()
    df_valid = df.loc[valid_idx]
    
    # ãƒ‡ãƒ¼ã‚¿å��åˆ†æ€§ãƒ�ã‚§ãƒƒã‚¯
    if len(df_valid) < 30:
        print(f"{target_col} ã�®ãƒ‡ãƒ¼ã‚¿ä¸�è¶³ã€�å¹³å�‡å€¤äºˆæ¸¬ã‚’ä½¿ç”¨")
        mean_val = df_valid[target_col].mean() if len(df_valid) > 0 else 0
        return {'model': MeanPredictor(mean_val), 'cv_score': 0.0}
    
    # ãƒ‡ãƒ¼ã‚¿æº–å‚™
    X = df_valid[feature_cols].copy()
    y = df_valid[target_col].values
    
    # ã‚ˆã‚Šé«˜ç²¾åº¦ã�®ã�Ÿã‚�ã�®KNNè£œå®Œã�«ã‚ˆã‚‹æ¬ æ��å€¤å‡¦ç�†
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    
    # ãƒ¢ãƒ‡ãƒ«æ€§èƒ½å�‘ä¸Šã�®ã�Ÿã‚�ã�®ç‰¹å¾´é‡�å¤‰æ�›
    scaler = PowerTransformer(method='yeo-johnson')
    X_scaled = scaler.fit_transform(X_imputed)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"{target_col} ç”¨ã�®é«˜åº¦ã�ªãƒ¢ãƒ‡ãƒ«ã‚’ {len(y)} ã‚µãƒ³ãƒ—ãƒ«ã�§è¨“ç·´")
    
    # ã‚¯ãƒ­ã‚¹ãƒ�ãƒªãƒ‡ãƒ¼ã‚·ãƒ§ãƒ³è¨­å®š
    n_splits = min(n_splits, len(y) // 10 or 2)  # ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰æ¯�ã�®å��åˆ†ã�ªã‚µãƒ³ãƒ—ãƒ«ç¢ºä¿�
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    # ã�“ã�®ç‰¹æ€§ã�®æœ€é�©åŒ–æ¸ˆã�¿ãƒ�ã‚¤ãƒ‘ãƒ¼ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿å�–å¾—
    hyperparams = get_property_hyperparams(target_col)
    
    # ãƒ™ãƒ¼ã‚¹ãƒ¢ãƒ‡ãƒ«è¨­å®š
    all_models = []
    oof_preds = np.zeros(len(y))
    cv_scores = []
    
    # è¨“ç·´ã�¨ã‚¯ãƒ­ã‚¹ãƒ�ãƒªãƒ‡ãƒ¼ã‚·ãƒ§ãƒ³
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        fold_models = []
        fold_preds = []
        
        # 1. XGBoostãƒ¢ãƒ‡ãƒ«
        try:
            xgb_model = xgb.XGBRegressor(
                **hyperparams['xgb'],
                random_state=SEED+fold
            )
            xgb_model.fit(X_train, y_train)
            xgb_preds = xgb_model.predict(X_val)
            fold_preds.append(xgb_preds)
            fold_models.append(xgb_model)
            print(f"  ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰ {fold+1} XGB MAE: {mean_absolute_error(y_val, xgb_preds):.6f}")
        except Exception as e:
            print(f"  XGBãƒ¢ãƒ‡ãƒ«å¤±æ•—: {str(e)}")
        
        # 2. CatBoostãƒ¢ãƒ‡ãƒ«
        try:
            cat_model = CatBoostRegressor(
                **hyperparams['cat'],
                random_seed=SEED+fold,
                verbose=False
            )
            cat_model.fit(X_train, y_train)
            cat_preds = cat_model.predict(X_val)
            fold_preds.append(cat_preds)
            fold_models.append(cat_model)
            print(f"  ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰ {fold+1} CatBoost MAE: {mean_absolute_error(y_val, cat_preds):.6f}")
        except Exception as e:
            print(f"  CatBoostãƒ¢ãƒ‡ãƒ«å¤±æ•—: {str(e)}")
        
        # 3. Random Forestãƒ¢ãƒ‡ãƒ«
        try:
            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=SEED+fold,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            rf_preds = rf_model.predict(X_val)
            fold_preds.append(rf_preds)
            fold_models.append(rf_model)
            print(f"  ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰ {fold+1} RF MAE: {mean_absolute_error(y_val, rf_preds):.6f}")
        except Exception as e:
            print(f"  RFãƒ¢ãƒ‡ãƒ«å¤±æ•—: {str(e)}")
        
        # 4. Gradient Boostingãƒ¢ãƒ‡ãƒ«
        try:
            gb_model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                random_state=SEED+fold
            )
            gb_model.fit(X_train, y_train)
            gb_preds = gb_model.predict(X_val)
            fold_preds.append(gb_preds)
            fold_models.append(gb_model)
            print(f"  ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰ {fold+1} GB MAE: {mean_absolute_error(y_val, gb_preds):.6f}")
        except Exception as e:
            print(f"  GBãƒ¢ãƒ‡ãƒ«å¤±æ•—: {str(e)}")
        
        # 5. å°�ãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆç”¨KNNãƒ¢ãƒ‡ãƒ«
        if len(y_train) < 1000:
            try:
                knn_model = KNeighborsRegressor(n_neighbors=7)
                knn_model.fit(X_train, y_train)
                knn_preds = knn_model.predict(X_val)
                fold_preds.append(knn_preds)
                fold_models.append(knn_model)
                print(f"  ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰ {fold+1} KNN MAE: {mean_absolute_error(y_val, knn_preds):.6f}")
            except Exception as e:
                print(f"  KNNãƒ¢ãƒ‡ãƒ«å¤±æ•—: {str(e)}")
        
        # æˆ�åŠŸã�—ã�Ÿãƒ¢ãƒ‡ãƒ«ã‚’ä¿�å­˜
        all_models.extend(fold_models)
        
        # ã‚¢ãƒ³ã‚µãƒ³ãƒ–ãƒ«äºˆæ¸¬è¨ˆç®—
        if fold_preds:
            ensemble_preds = np.mean(np.column_stack(fold_preds), axis=1)
            oof_preds[val_idx] = ensemble_preds
            fold_score = mean_absolute_error(y_val, ensemble_preds)
            cv_scores.append(fold_score)
            print(f"  ãƒ•ã‚©ãƒ¼ãƒ«ãƒ‰ {fold+1} ã‚¢ãƒ³ã‚µãƒ³ãƒ–ãƒ« MAE: {fold_score:.6f}")
    
    # å…¨ãƒ¢ãƒ‡ãƒ«å¤±æ•—æ™‚ã�¯å¹³å�‡äºˆæ¸¬å™¨ã‚’ä½¿ç”¨
    if not all_models:
        print("  å…¨ãƒ¢ãƒ‡ãƒ«å¤±æ•—ã€‚å¹³å�‡å€¤äºˆæ¸¬ã‚’ä½¿ç”¨ã€‚")
        mean_val = np.mean(y)
        return {
            'model': MeanPredictor(mean_val),
            'cv_score': 0.0
        }
    
    # å…¨ä½“CVã‚¹ã‚³ã‚¢è¨ˆç®—
    if cv_scores:
        cv_score = mean_absolute_error(y[~np.isnan(oof_preds)], oof_preds[~np.isnan(oof_preds)])
        print(f"{target_col} ã�®ã‚¯ãƒ­ã‚¹ãƒ�ãƒªãƒ‡ãƒ¼ã‚·ãƒ§ãƒ³ MAE: {cv_score:.6f}")
    else:
        cv_score = 0.0
    
    # æœ€çµ‚ã‚¢ãƒ³ã‚µãƒ³ãƒ–ãƒ«ãƒ¢ãƒ‡ãƒ«ä½œæˆ�
    final_model = AveragingEnsemble(all_models, imputer, scaler)
    
    return {
        'model': final_model,
        'cv_score': cv_score
    }

print("é«˜åº¦ã�ªãƒ¢ãƒ‡ãƒ«è¨“ç·´é–¢æ•°å®šç¾©å®Œäº†")


# å�„ã‚¿ãƒ¼ã‚²ãƒƒãƒˆã�®é«˜åº¦ã�ªãƒ¢ãƒ‡ãƒ«è¨“ç·´
models = {}
for col in target_cols:
    print(f"\n{col} ç”¨ã�®é«˜åº¦ã�ªãƒ¢ãƒ‡ãƒ«è¨“ç·´ä¸­...")
    models[col] = train_advanced_model(
        train_with_features, col, target_features[col]
    )

print("\nå…¨ãƒ¢ãƒ‡ãƒ«è¨“ç·´å®Œäº†")


# ãƒ†ã‚¹ãƒˆäºˆæ¸¬ç”Ÿæˆ�
test_preds = {}
for col in target_cols:
    print(f"{col} ã�®äºˆæ¸¬ç”Ÿæˆ�ä¸­...")
    try:
        test_preds[col] = models[col]['model'].predict(test_with_features[target_features[col]])
    except Exception as e:
        print(f"{col} ã�®äºˆæ¸¬ç”Ÿæˆ�ã‚¨ãƒ©ãƒ¼: {str(e)}")
        # ãƒ•ã‚©ãƒ¼ãƒ«ãƒ�ãƒƒã‚¯ç”¨ãƒ‡ãƒ•ã‚©ãƒ«ãƒˆå€¤
        fallbacks = {
            'Tg': 400,
            'FFV': 0.2,
            'Tc': 0.2,
            'Density': 1.0,
            'Rg': 10.0
        }
        test_preds[col] = np.full(len(test_with_features), fallbacks[col])

# æ��å‡ºãƒ‡ãƒ¼ã‚¿ãƒ•ãƒ¬ãƒ¼ãƒ ä½œæˆ�
submission = pd.DataFrame({'id': test_with_features['id']})
for col in target_cols:
    submission[col] = test_preds[col]

# æ��å‡ºãƒ•ã‚©ãƒ¼ãƒ�ãƒƒãƒˆç¢ºèª�
print("\næ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ãƒ—ãƒ¬ãƒ“ãƒ¥ãƒ¼:")
print(submission.head())

# å…ƒã�®ãƒ†ã‚¹ãƒˆã‚»ãƒƒãƒˆã�‹ã‚‰ã�®å…¨è¡ŒåŒ…å�«ç¢ºèª�
if len(submission) < len(test):
    print(f"è­¦å‘Š: æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ã�« {len(submission)} è¡Œã€�ãƒ†ã‚¹ãƒˆã‚»ãƒƒãƒˆã�« {len(test)} è¡Œ")
    # ä¸�è¶³è¡Œã‚’ãƒ‡ãƒ•ã‚©ãƒ«ãƒˆå€¤ã�§è£œå®Œ
    missing_ids = set(test['id']) - set(submission['id'])
    print(f"{len(missing_ids)} ä¸�è¶³è¡Œã‚’è¿½åŠ ")
    
    fallbacks = {
        'Tg': 400,
        'FFV': 0.2,
        'Tc': 0.2,
        'Density': 1.0,
        'Rg': 10.0
    }
    
    for missing_id in missing_ids:
        row = {'id': missing_id}
        row.update(fallbacks)
        submission = pd.concat([submission, pd.DataFrame([row])], ignore_index=True)

# æ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ä¿�å­˜
submission.to_csv('submission.csv', index=False)
print("\næ��å‡ºãƒ•ã‚¡ã‚¤ãƒ«ä¿�å­˜å®Œäº†")

# CVãƒ™ãƒ¼ã‚¹ã�®è¿‘ä¼¼wMAEè¨ˆç®—
weighted_scores = []
for col in target_cols:
    if 'cv_score' in models[col] and models[col]['cv_score'] > 0:
        weighted_score = models[col]['cv_score'] * weights[col] / property_ranges[col]
        weighted_scores.append(weighted_score)
        print(f"{col} ã�®é‡�ã�¿ä»˜ã��ã‚¹ã‚³ã‚¢: {weighted_score:.6f}")

if weighted_scores:
    estimated_wmae = sum(weighted_scores)
    print(f"\næ�¨å®šé‡�ã�¿ä»˜ã��MAE: {estimated_wmae:.6f}")
else:
    print("\né‡�ã�¿ä»˜ã��MAEæ�¨å®šä¸�å�¯ï¼ˆCVã‚¹ã‚³ã‚¢åˆ©ç”¨ä¸�å�¯ï¼‰")

elapsed_time = time.time() - start_time
print(f"ç·�å®Ÿè¡Œæ™‚é–“: {elapsed_time/60:.2f} åˆ†")

print("\n=== é«˜åº¦ã�ªãƒ�ãƒªãƒ�ãƒ¼ç‰¹æ€§äºˆæ¸¬ãƒ‘ã‚¤ãƒ—ãƒ©ã‚¤ãƒ³å®Œäº† ===")

