# Install RDKit from wheel for SMILES canonicalization
import sys
import subprocess
import os

RDKIT_AVAILABLE = False  # Default to False

print("Installing RDKit from wheel...")

# Use exact path provided
wheel_path = '/kaggle/input/d/wpixiu/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl'

try:
    if os.path.exists(wheel_path):
        print(f"âœ“ Found wheel: {wheel_path}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', wheel_path])
        print("âœ“ RDKit installed from wheel successfully")
        RDKIT_AVAILABLE = True
    else:
        print(f"âš  Wheel not found at {wheel_path}")
        print("Attempting pip install as fallback...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'rdkit'])
        print("âœ“ RDKit installed from pip")
        RDKIT_AVAILABLE = True
except Exception as e:
    print(f"âš  RDKit installation failed: {e}")
    print("Continuing without RDKit (will use simple features only)...")
    RDKIT_AVAILABLE = False

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Try to import additional RDKit modules if available
Chem = None  # Initialize Chem to None
if RDKIT_AVAILABLE:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        from rdkit.Chem import AllChem
    except ImportError:
        RDKIT_AVAILABLE = False
        Chem = None
        print("Note: RDKit core loaded but some modules unavailable")

from tqdm import tqdm

# ============================================================================
# Feature Strategy: 21 Chemistry-Based Features
# ============================================================================
# We use a balanced approach combining:
# - 10 simple string-based features (fast, reliable)
# - 11 chemistry-based features (polymer-specific domain knowledge)
# This 21-feature set captures both molecular structure and chemistry properties

USE_SIMPLE_FEATURES_ONLY = True

# SMILES canonicalization function
def make_smile_canonical(smile):
    """To avoid duplicates, for example: canonical '*C=C(*)C' == '*C(=C*)C'"""
    if not RDKIT_AVAILABLE or Chem is None:
        return smile  # Return as-is if RDKit not available
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return np.nan
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except:
        return np.nan

print()
print("=" * 70)
print("FEATURE STRATEGY: 21 CHEMISTRY-BASED FEATURES (v85)")
print("=" * 70)
print("Simple Features (10):")
print("  smiles_length, carbon_count, nitrogen_count, oxygen_count,")
print("  sulfur_count, fluorine_count, ring_count, double_bond_count,")
print("  triple_bond_count, branch_count")
print()
print("Chemistry Features (11):")
print("  num_side_chains, backbone_carbons, branching_ratio,")
print("  aromatic_count, h_bond_donors, h_bond_acceptors,")
print("  num_rings, single_bonds, halogen_count,")
print("  heteroatom_count, mw_estimate")
print("=" * 70)
print()

print("Setup complete!")


# Load data with error handling
try:
    train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
    test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
    sample_submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
    print("Data loaded from Kaggle input")
except:
    try:
        # Fallback for local testing
        train_df = pd.read_csv('data/raw/train.csv')
        test_df = pd.read_csv('data/raw/test.csv')
        sample_submission = pd.read_csv('data/raw/sample_submission.csv')
        print("Data loaded from local files")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Sample submission shape: {sample_submission.shape}")

# Target columns
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

print("\nTarget availability:")
for col in target_cols:
    n_avail = train_df[col].notna().sum()
    print(f"{col}: {n_avail} samples ({n_avail/len(train_df)*100:.1f}%)")


# Canonicalize SMILES to avoid duplicates and standardize representations
print("=" * 70)
print("CANONICALIZING SMILES")
print("=" * 70)

if RDKIT_AVAILABLE:
    print("Applying SMILES canonicalization...")
    
    # Store original counts
    orig_train_count = len(train_df)
    orig_test_count = len(test_df)
    
    # Apply canonicalization
    train_df['SMILES_canonical'] = train_df['SMILES'].apply(make_smile_canonical)
    test_df['SMILES_canonical'] = test_df['SMILES'].apply(make_smile_canonical)
    
    # Count successes
    train_success = train_df['SMILES_canonical'].notna().sum()
    test_success = test_df['SMILES_canonical'].notna().sum()
    
    print(f"Train: {train_success}/{orig_train_count} successfully canonicalized ({train_success/orig_train_count*100:.1f}%)")
    print(f"Test: {test_success}/{orig_test_count} successfully canonicalized ({test_success/orig_test_count*100:.1f}%)")
    
    # For failed canonicalizations, keep original SMILES
    train_df['SMILES_canonical'] = train_df['SMILES_canonical'].fillna(train_df['SMILES'])
    test_df['SMILES_canonical'] = test_df['SMILES_canonical'].fillna(test_df['SMILES'])
    
    # Replace SMILES with canonical versions
    train_df['SMILES'] = train_df['SMILES_canonical']
    test_df['SMILES'] = test_df['SMILES_canonical']
    
    # Drop temporary column
    train_df = train_df.drop('SMILES_canonical', axis=1)
    test_df = test_df.drop('SMILES_canonical', axis=1)
    
    print("âœ“ SMILES canonicalization complete!")
    
    # Show example
    print("\nExample canonical SMILES:")
    print(train_df['SMILES'].head(3).tolist())
else:
    print("âš  RDKit not available - skipping canonicalization")
    print("Using original SMILES as-is")

print("=" * 70)
print()



# Load external Tc dataset
print("=" * 70)
print("LOADING EXTERNAL Tc DATASET")
print("=" * 70)

try:
    # Load the external Tc data - try multiple possible paths
    tc_path = None
    possible_paths = [
        '/kaggle/input/tc-smiles/Tc_SMILES.csv',
        '/kaggle/input/tc-smiles/TC_SMILES.csv',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            tc_path = path
            break
    
    if not tc_path:
        # List available files in tc-smiles directory
        import os
        tc_dir = '/kaggle/input/tc-smiles'
        if os.path.exists(tc_dir):
            files = os.listdir(tc_dir)
            print(f"Available files in {tc_dir}: {files}")
            for f in files:
                if f.endswith('.csv'):
                    tc_path = os.path.join(tc_dir, f)
                    break
    
    if not tc_path:
        raise FileNotFoundError("No Tc CSV file found")
    
    tc_external = pd.read_csv(tc_path)
    print(f"Loaded from: {tc_path}")
    print(f"âœ“ Loaded external Tc dataset: {len(tc_external)} samples")
    print(f"Columns: {list(tc_external.columns)}")
    print(f"\nSample data:")
    print(tc_external.head())
    
    # Canonicalize external SMILES
    if RDKIT_AVAILABLE:
        print("\nCanonicalizing external SMILES...")
        tc_external['SMILES_canonical'] = tc_external['SMILES'].apply(make_smile_canonical)
        tc_success = tc_external['SMILES_canonical'].notna().sum()
        print(f"External Tc: {tc_success}/{len(tc_external)} successfully canonicalized ({tc_success/len(tc_external)*100:.1f}%)")
        
        # For failed canonicalizations, keep original
        tc_external['SMILES_canonical'] = tc_external['SMILES_canonical'].fillna(tc_external['SMILES'])
        tc_external['SMILES'] = tc_external['SMILES_canonical']
        tc_external = tc_external.drop('SMILES_canonical', axis=1)
    
    # Rename TC_mean to Tc to match training data
    tc_external = tc_external.rename(columns={'TC_mean': 'Tc'})
    
    # Check for overlap with training data
    train_smiles = set(train_df['SMILES'])
    external_smiles = set(tc_external['SMILES'])
    overlap = train_smiles & external_smiles
    print(f"\nðŸ“Š Dataset overlap analysis:")
    print(f"Training SMILES: {len(train_smiles)}")
    print(f"External SMILES: {len(external_smiles)}")
    print(f"Overlapping SMILES: {len(overlap)}")
    
    # Get original Tc count in training
    orig_tc_count = train_df['Tc'].notna().sum()
    print(f"\nOriginal training Tc samples: {orig_tc_count}")
    
    # Merge strategy: Add external data for SMILES NOT in training set
    # For overlapping SMILES, we keep training data (more reliable)
    tc_new = tc_external[~tc_external['SMILES'].isin(train_smiles)].copy()
    print(f"New Tc samples to add: {len(tc_new)}")
    
    if len(tc_new) > 0:
        # Create rows with only SMILES and Tc filled
        tc_new_rows = []
        for _, row in tc_new.iterrows():
            new_row = {
                'SMILES': row['SMILES'],
                'Tg': np.nan,
                'FFV': np.nan,
                'Tc': row['Tc'],
                'Density': np.nan,
                'Rg': np.nan
            }
            tc_new_rows.append(new_row)
        
        tc_new_df = pd.DataFrame(tc_new_rows)
        
        # Append to training data
        train_df_original = train_df.copy()
        train_df = pd.concat([train_df, tc_new_df], ignore_index=True)
        
        new_tc_count = train_df['Tc'].notna().sum()
        print(f"\nâœ… AUGMENTATION COMPLETE!")
        print(f"Training set size: {len(train_df_original)} â†’ {len(train_df)} (+{len(tc_new)})")
        print(f"Tc samples: {orig_tc_count} â†’ {new_tc_count} (+{new_tc_count - orig_tc_count})")
        print(f"Tc improvement: {((new_tc_count - orig_tc_count) / orig_tc_count * 100):.1f}% increase")
        
        print(f"\nðŸ“ˆ Final training data statistics:")
        for col in target_cols:
            n_avail = train_df[col].notna().sum()
            print(f"  {col}: {n_avail} samples ({n_avail/len(train_df)*100:.1f}%)")
    else:
        print("\nâš  All external SMILES already in training set - no augmentation needed")
        
except FileNotFoundError:
    print("âš  External Tc dataset not found - skipping augmentation")
    print("Continuing with original training data only")
except Exception as e:
    print(f"âš  Error loading external Tc data: {e}")
    print("Continuing with original training data only")

print("=" * 70)
print()



# Load external Tg dataset
print("=" * 70)
print("LOADING EXTERNAL Tg DATASET")
print("=" * 70)

try:
    # Load the external Tg data
    tg_external = pd.read_csv('/kaggle/input/tg-of-polymer-dataset/Tg_SMILES_class_pid_polyinfo_median.csv')
    print(f"âœ“ Loaded external Tg dataset: {len(tg_external)} samples")
    print(f"Columns: {list(tg_external.columns)}")
    print(f"\nSample data:")
    print(tg_external.head())
    
    # Canonicalize external SMILES
    if RDKIT_AVAILABLE:
        print("\nCanonicalizing external SMILES...")
        tg_external['SMILES_canonical'] = tg_external['SMILES'].apply(make_smile_canonical)
        tg_success = tg_external['SMILES_canonical'].notna().sum()
        print(f"External Tg: {tg_success}/{len(tg_external)} successfully canonicalized ({tg_success/len(tg_external)*100:.1f}%)")
        
        # For failed canonicalizations, keep original
        tg_external['SMILES_canonical'] = tg_external['SMILES_canonical'].fillna(tg_external['SMILES'])
        tg_external['SMILES'] = tg_external['SMILES_canonical']
        tg_external = tg_external.drop('SMILES_canonical', axis=1)
    
    # Check for overlap with training data
    train_smiles = set(train_df['SMILES'])
    external_smiles = set(tg_external['SMILES'])
    overlap = train_smiles & external_smiles
    print(f"\nðŸ“Š Dataset overlap analysis:")
    print(f"Training SMILES: {len(train_smiles)}")
    print(f"External SMILES: {len(external_smiles)}")
    print(f"Overlapping SMILES: {len(overlap)}")
    
    # Get original Tg count in training
    orig_tg_count = train_df['Tg'].notna().sum()
    print(f"\nOriginal training Tg samples: {orig_tg_count}")
    
    # Merge strategy: Add external data for SMILES NOT in training set
    # For overlapping SMILES, we keep training data (more reliable)
    tg_new = tg_external[~tg_external['SMILES'].isin(train_smiles)].copy()
    print(f"New Tg samples to add: {len(tg_new)}")
    
    if len(tg_new) > 0:
        # Create rows with only SMILES and Tg filled
        tg_new_rows = []
        for _, row in tg_new.iterrows():
            new_row = {
                'SMILES': row['SMILES'],
                'Tg': row['Tg'],
                'FFV': np.nan,
                'Tc': np.nan,
                'Density': np.nan,
                'Rg': np.nan
            }
            tg_new_rows.append(new_row)
        
        tg_new_df = pd.DataFrame(tg_new_rows)
        
        # Append to training data
        train_df_before_tg = train_df.copy()
        train_df = pd.concat([train_df, tg_new_df], ignore_index=True)
        
        new_tg_count = train_df['Tg'].notna().sum()
        print(f"\nâœ… Tg AUGMENTATION COMPLETE!")
        print(f"Training set size: {len(train_df_before_tg)} â†’ {len(train_df)} (+{len(tg_new)})")
        print(f"Tg samples: {orig_tg_count} â†’ {new_tg_count} (+{new_tg_count - orig_tg_count})")
        print(f"Tg improvement: {((new_tg_count - orig_tg_count) / orig_tg_count * 100):.1f}% increase")
        
        print(f"\nðŸ“ˆ Final training data statistics:")
        for col in target_cols:
            n_avail = train_df[col].notna().sum()
            print(f"  {col}: {n_avail} samples ({n_avail/len(train_df)*100:.1f}%)")
    else:
        print("\nâš  All external SMILES already in training set - no augmentation needed")
        
except FileNotFoundError:
    print("âš  External Tg dataset not found - skipping augmentation")
    print("Continuing with original training data only")
except Exception as e:
    print(f"âš  Error loading external Tg data: {e}")
    print("Continuing with original training data only")

print("=" * 70)
print()



# Load and Integrate External Datasets
print("=" * 70)
print("LOADING EXTERNAL DATASETS FOR AUGMENTATION")
print("=" * 70)

# Load PI1070 dataset (Density + Rg)
print("\n[1] Loading PI1070.csv (Density + Rg)...")
try:
    pi1070_df = pd.read_csv('/kaggle/input/more-data/PI1070.csv')
    print(f"âœ“ Loaded {len(pi1070_df)} samples")
    print(f"  Columns: {list(pi1070_df.columns)[:5]}... (truncated)")
    
    # Extract SMILES, Density, Rg
    pi1070_subset = pi1070_df[['smiles', 'density', 'Rg']].copy()
    pi1070_subset = pi1070_subset.rename(columns={'smiles': 'SMILES'})
    
    # Check for overlaps
    pi1070_smiles = set(pi1070_subset['SMILES'].dropna())
    train_smiles_set = set(train_df['SMILES'].dropna())
    overlap_pi1070 = len(pi1070_smiles & train_smiles_set)
    pi1070_new = pi1070_subset[~pi1070_subset['SMILES'].isin(train_smiles_set)].copy()
    
    print(f"  New non-overlapping samples: {len(pi1070_new)}")
    print(f"  Density values available: {pi1070_new['density'].notna().sum()}")
    print(f"  Rg values available: {pi1070_new['Rg'].notna().sum()}")
except Exception as e:
    print(f"âš  Failed to load PI1070: {e}")
    pi1070_new = None

# Load LAMALAB Tg dataset
print("\n[2] Loading LAMALAB_CURATED_Tg_structured_polymerclass.csv...")
try:
    lamalab_df = pd.read_csv('/kaggle/input/more-data/LAMALAB_CURATED_Tg_structured_polymerclass.csv')
    print(f"âœ“ Loaded {len(lamalab_df)} samples")
    
    # Extract SMILES and Tg (convert from Kelvin to Celsius)
    lamalab_subset = lamalab_df[['PSMILES', 'labels.Exp_Tg(K)']].copy()
    lamalab_subset = lamalab_subset.rename(columns={'PSMILES': 'SMILES', 'labels.Exp_Tg(K)': 'Tg'})
    
    # Convert Tg from Kelvin to Celsius
    lamalab_subset['Tg'] = lamalab_subset['Tg'] - 273.15
    
    # Check for overlaps
    lamalab_smiles = set(lamalab_subset['SMILES'].dropna())
    overlap_lamalab = len(lamalab_smiles & train_smiles_set)
    lamalab_new = lamalab_subset[~lamalab_subset['SMILES'].isin(train_smiles_set)].copy()
    
    print(f"  New non-overlapping samples: {len(lamalab_new)}")
    print(f"  Tg values available: {lamalab_new['Tg'].notna().sum()}")
    print(f"  Tg range (Â°C): [{lamalab_new['Tg'].min():.1f}, {lamalab_new['Tg'].max():.1f}]")
except Exception as e:
    print(f"âš  Failed to load LAMALAB Tg: {e}")
    lamalab_new = None

# Augment training data
print("\n[3] Augmenting training data...")
train_df_before = len(train_df)

# Add PI1070 data (Density + Rg)
if pi1070_new is not None and len(pi1070_new) > 0:
    for idx, row in pi1070_new.iterrows():
        if pd.notna(row['density']) or pd.notna(row['Rg']):
            train_df = pd.concat([train_df, pd.DataFrame([{
                'SMILES': row['SMILES'],
                'Tg': np.nan,
                'FFV': np.nan,
                'Tc': np.nan,
                'Density': row['density'] if pd.notna(row['density']) else np.nan,
                'Rg': row['Rg'] if pd.notna(row['Rg']) else np.nan
            }])], ignore_index=True)
    print(f"âœ“ Added {len(pi1070_new)} PI1070 samples")

# Add LAMALAB Tg data
if lamalab_new is not None and len(lamalab_new) > 0:
    lamalab_new_valid = lamalab_new[lamalab_new['Tg'].notna()].copy()
    if len(lamalab_new_valid) > 0:
        for idx, row in lamalab_new_valid.iterrows():
            train_df = pd.concat([train_df, pd.DataFrame([{
                'SMILES': row['SMILES'],
                'Tg': row['Tg'],
                'FFV': np.nan,
                'Tc': np.nan,
                'Density': np.nan,
                'Rg': np.nan
            }])], ignore_index=True)
        print(f"âœ“ Added {len(lamalab_new_valid)} LAMALAB Tg samples")

train_df = train_df.reset_index(drop=True)

print(f"\nðŸ“Š Training data augmented:")
print(f"  Before: {train_df_before} samples")
print(f"  After: {len(train_df)} samples")
print(f"  Net increase: +{len(train_df) - train_df_before} samples ({100*(len(train_df)-train_df_before)/train_df_before:.1f}%)")

print(f"\nðŸ“ˆ Updated target availability:")
for col in target_cols:
    n_avail = train_df[col].notna().sum()
    print(f"    {col}: {n_avail} samples ({n_avail/len(train_df)*100:.1f}%)")

print("=" * 70)
print()




# Load pseudo-labeled dataset
print("=" * 70)
print("LOADING PSEUDO-LABELED DATASET (Ensemble: BERT + AutoGluon + Uni-Mol)")
print("=" * 70)

try:
    # Try loading from Kaggle input first
    pseudo_label_path = None
    
    # First, check what files are in the pi1m-pseudolabels directory
    pi1m_dir = '/kaggle/input/pi1m-pseudolabels'
    if os.path.exists(pi1m_dir):
        print(f"Files in {pi1m_dir}:")
        try:
            files = os.listdir(pi1m_dir)
            for f in files[:10]:  # Show first 10 files
                print(f"  - {f}")
        except:
            pass
    
    # Try various possible paths
    possible_paths = [
        '/kaggle/input/pi1m-pseudolabels/PI1M_50000_v2.1.csv',
        '/kaggle/input/pi1m-pseudolabels/pi1m_50000_v2.1.csv',  # lowercase
        '/kaggle/input/pi1m-pseudolabels/data.csv',  # might be renamed
        '/kaggle/input/pseudo-labels/PI1M_50000_v2.1.csv',
        'data/PI1M_50000_v2.1.csv',
    ]
    
    # Also try to find any CSV file in pi1m directory
    if os.path.exists(pi1m_dir):
        try:
            for f in os.listdir(pi1m_dir):
                if f.endswith('.csv'):
                    possible_paths.insert(0, os.path.join(pi1m_dir, f))
        except:
            pass
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                pseudo_label_path = path
                print(f"âœ“ Found pseudo-label file at: {path}")
                break
        except:
            pass
    
    if pseudo_label_path:
        pseudo_df = pd.read_csv(pseudo_label_path)
        print(f"âœ“ Loaded pseudo-labeled dataset from: {pseudo_label_path}")
        print(f"  Samples: {len(pseudo_df)}")
        print(f"  Columns: {list(pseudo_df.columns)}")
        print(f"  Source: Ensemble of BERT, AutoGluon, Uni-Mol")
        
        # Show sample data
        print(f"\n  Sample data:")
        print(pseudo_df.head(2))
        
        # Check for overlap with training data
        train_smiles_set = set(train_df['SMILES'].dropna())
        pseudo_smiles = set(pseudo_df['SMILES'].dropna())
        overlap = len(train_smiles_set & pseudo_smiles)
        
        print(f"\n  ðŸ“Š Dataset overlap analysis:")
        print(f"    Training SMILES: {len(train_smiles_set)}")
        print(f"    Pseudo-label SMILES: {len(pseudo_smiles)}")
        print(f"    Overlapping SMILES: {overlap}")
        
        # Get new non-overlapping samples
        pseudo_new = pseudo_df[~pseudo_df['SMILES'].isin(train_smiles_set)].copy()
        print(f"    New samples to add: {len(pseudo_new)}")
        
        if len(pseudo_new) > 0:
            # Store original sizes
            orig_train_size = len(train_df)
            orig_counts = {col: train_df[col].notna().sum() for col in target_cols}
            
            # Append pseudo-labeled data
            train_df = pd.concat([train_df, pseudo_new], ignore_index=True)
            
            print(f"\n  âœ… PSEUDO-LABEL AUGMENTATION COMPLETE!")
            print(f"    Training set size: {orig_train_size} â†’ {len(train_df)} (+{len(pseudo_new)})")
            print(f"    Size increase: +{len(pseudo_new)/orig_train_size*100:.1f}%")
            
            print(f"\n  ðŸ“ˆ Updated target availability:")
            for col in target_cols:
                new_count = train_df[col].notna().sum()
                increase = new_count - orig_counts[col]
                print(f"    {col}: {orig_counts[col]} â†’ {new_count} (+{increase}, +{increase/orig_counts[col]*100:.1f}%)")
        else:
            print(f"\n  âš  All pseudo-label SMILES already in training set - no augmentation needed")
    else:
        print("âš  Pseudo-labeled dataset not found in any expected location")
        print("Continuing with original training data only")
        
except Exception as e:
    print(f"âš  Error loading pseudo-labeled data: {e}")
    print("Continuing with original training data only")

print("=" * 70)
print()



"
"
"

U
p
d
a
t
e
d
 
R
o
b
u
s
t
M
o
l
e
c
u
l
a
r
P
r
o
c
e
s
s
o
r
 
f
o
r
 
b
e
s
t
.
i
p
y
n
b


T
h
i
s
 
v
e
r
s
i
o
n
:

-
 
E
x
t
r
a
c
t
s
 
b
a
s
e
l
i
n
e
 
2
1
 
f
e
a
t
u
r
e
s
 
+
 
t
o
p
 
1
0
 
n
e
w
 
f
e
a
t
u
r
e
s
 
(
3
1
 
t
o
t
a
l
)

-
 
A
s
s
u
m
e
s
 
R
D
K
i
t
 
i
s
 
a
l
w
a
y
s
 
a
v
a
i
l
a
b
l
e

-
 
K
e
e
p
s
 
s
a
m
e
 
i
n
t
e
r
f
a
c
e
:
 
p
r
o
c
e
s
s
o
r
.
p
r
e
p
a
r
e
_
f
e
a
t
u
r
e
s
(
d
f
)

-
 
S
i
m
p
l
i
f
i
e
d
 
a
n
d
 
c
l
e
a
n
e
d
 
u
p


C
o
p
y
 
t
h
i
s
 
c
l
a
s
s
 
i
n
t
o
 
b
e
s
t
.
i
p
y
n
b
 
t
o
 
r
e
p
l
a
c
e
 
t
h
e
 
e
x
i
s
t
i
n
g
 
R
o
b
u
s
t
M
o
l
e
c
u
l
a
r
P
r
o
c
e
s
s
o
r

"
"
"


d
e
f
 
_
i
s
_
r
o
t
a
t
a
b
l
e
_
b
o
n
d
(
b
o
n
d
)
:

 
 
 
 
"
"
"
C
h
e
c
k
 
i
f
 
a
 
b
o
n
d
 
i
s
 
r
o
t
a
t
a
b
l
e
 
(
c
o
m
p
a
t
i
b
l
e
 
w
i
t
h
 
d
i
f
f
e
r
e
n
t
 
R
D
K
i
t
 
v
e
r
s
i
o
n
s
)
"
"
"

 
 
 
 
t
r
y
:

 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
b
o
n
d
.
I
s
R
o
t
o
r
(
)

 
 
 
 
e
x
c
e
p
t
 
A
t
t
r
i
b
u
t
e
E
r
r
o
r
:

 
 
 
 
 
 
 
 
#
 
F
a
l
l
b
a
c
k
:
 
s
i
n
g
l
e
 
b
o
n
d
,
 
n
o
t
 
i
n
 
r
i
n
g
,
 
n
o
t
 
t
e
r
m
i
n
a
l

 
 
 
 
 
 
 
 
i
f
 
b
o
n
d
.
G
e
t
B
o
n
d
T
y
p
e
(
)
 
!
=
 
C
h
e
m
.
B
o
n
d
T
y
p
e
.
S
I
N
G
L
E
:

 
 
 
 
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
F
a
l
s
e

 
 
 
 
 
 
 
 
i
f
 
b
o
n
d
.
I
s
I
n
R
i
n
g
(
)
:

 
 
 
 
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
F
a
l
s
e

 
 
 
 
 
 
 
 
#
 
C
h
e
c
k
 
i
f
 
e
i
t
h
e
r
 
a
t
o
m
 
i
s
 
t
e
r
m
i
n
a
l
 
(
d
e
g
r
e
e
 
1
)

 
 
 
 
 
 
 
 
b
e
g
i
n
_
a
t
o
m
 
=
 
b
o
n
d
.
G
e
t
B
e
g
i
n
A
t
o
m
(
)

 
 
 
 
 
 
 
 
e
n
d
_
a
t
o
m
 
=
 
b
o
n
d
.
G
e
t
E
n
d
A
t
o
m
(
)

 
 
 
 
 
 
 
 
i
f
 
b
e
g
i
n
_
a
t
o
m
.
G
e
t
D
e
g
r
e
e
(
)
 
=
=
 
1
 
o
r
 
e
n
d
_
a
t
o
m
.
G
e
t
D
e
g
r
e
e
(
)
 
=
=
 
1
:

 
 
 
 
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
F
a
l
s
e

 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
T
r
u
e

 
 
 
 
 
 
 
 

c
l
a
s
s
 
R
o
b
u
s
t
M
o
l
e
c
u
l
a
r
P
r
o
c
e
s
s
o
r
:

 
 
 
 
"
"
"

 
 
 
 
M
o
l
e
c
u
l
a
r
 
f
e
a
t
u
r
e
 
p
r
o
c
e
s
s
o
r
 
-
 
e
x
t
r
a
c
t
s
 
3
1
 
f
e
a
t
u
r
e
s
:

 
 
 
 
-
 
2
1
 
b
a
s
e
l
i
n
e
 
f
e
a
t
u
r
e
s
 
(
s
t
r
i
n
g
-
b
a
s
e
d
 
c
h
e
m
i
s
t
r
y
 
f
e
a
t
u
r
e
s
)

 
 
 
 
-
 
1
0
 
n
e
w
 
f
e
a
t
u
r
e
s
 
f
r
o
m
 
s
y
s
t
e
m
a
t
i
c
 
a
n
a
l
y
s
i
s
 
(
w
M
A
E
-
w
e
i
g
h
t
e
d
 
t
o
p
 
1
0
)

 
 
 
 
"
"
"

 
 
 
 

 
 
 
 
d
e
f
 
_
_
i
n
i
t
_
_
(
s
e
l
f
)
:

 
 
 
 
 
 
 
 
"
"
"
I
n
i
t
i
a
l
i
z
e
 
p
r
o
c
e
s
s
o
r
 
-
 
R
D
K
i
t
 
i
s
 
r
e
q
u
i
r
e
d
"
"
"

 
 
 
 
 
 
 
 
p
a
s
s

 
 
 
 

 
 
 
 
d
e
f
 
e
x
t
r
a
c
t
_
t
o
p
_
1
0
_
n
e
w
_
f
e
a
t
u
r
e
s
(
s
e
l
f
,
 
s
m
i
l
e
s
)
:

 
 
 
 
 
 
 
 
"
"
"

 
 
 
 
 
 
 
 
E
x
t
r
a
c
t
 
t
o
p
 
1
0
 
f
e
a
t
u
r
e
s
 
b
y
 
w
M
A
E
-
w
e
i
g
h
t
e
d
 
i
m
p
o
r
t
a
n
c
e
:

 
 
 
 
 
 
 
 
N
o
w
 
e
x
t
e
n
d
e
d
 
t
o
 
3
0

 
 
 
 
 
 
 
 
"
"
"

 
 
 
 
 
 
 
 
t
r
y
:

 
 
 
 
 
 
 
 
 
 
 
 
m
o
l
 
=
 
C
h
e
m
.
M
o
l
F
r
o
m
S
m
i
l
e
s
(
s
m
i
l
e
s
)

 
 
 
 
 
 
 
 
 
 
 
 
i
f
 
m
o
l
 
i
s
 
N
o
n
e
:

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
{
f
:
 
0
.
0
 
f
o
r
 
f
 
i
n
 
s
e
l
f
.
T
O
P
_
1
0
_
F
E
A
T
U
R
E
S
}

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
 
=
 
{
}

 
 
 
 
 
 
 
 
 
 
 
 
n
u
m
_
a
t
o
m
s
 
=
 
m
o
l
.
G
e
t
N
u
m
A
t
o
m
s
(
)

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
#
 
4

 
 
 
 
 
 
 
 
 
 
 
 
s
t
a
r
s
 
=
 
[
a
.
G
e
t
I
d
x
(
)
 
f
o
r
 
a
 
i
n
 
m
o
l
.
G
e
t
A
t
o
m
s
(
)
 
i
f
 
a
.
G
e
t
S
y
m
b
o
l
(
)
 
=
=
 
'
*
'
]

 
 
 
 
 
 
 
 
 
 
 
 
i
f
 
l
e
n
(
s
t
a
r
s
)
 
=
=
 
2
:

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
f
r
o
m
 
r
d
k
i
t
.
C
h
e
m
 
i
m
p
o
r
t
 
r
d
m
o
l
o
p
s

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
d
m
a
t
 
=
 
r
d
m
o
l
o
p
s
.
G
e
t
D
i
s
t
a
n
c
e
M
a
t
r
i
x
(
m
o
l
)

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
g
r
a
p
h
_
s
t
a
r
_
d
i
s
t
a
n
c
e
'
]
 
=
 
f
l
o
a
t
(
d
m
a
t
[
s
t
a
r
s
[
0
]
,
 
s
t
a
r
s
[
1
]
]
)

 
 
 
 
 
 
 
 
 
 
 
 
e
l
s
e
:

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
g
r
a
p
h
_
s
t
a
r
_
d
i
s
t
a
n
c
e
'
]
 
=
 
N
o
n
e

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
#
 
1

 
 
 
 
 
 
 
 
 
 
 
 
m
w
 
=
 
D
e
s
c
r
i
p
t
o
r
s
.
M
o
l
W
t
(
m
o
l
)

 
 
 
 
 
 
 
 
 
 
 
 
#
 
V
D
W
 
r
a
d
i
i
 
f
r
o
m
 
B
o
n
d
i
/
B
a
t
s
a
n
o
v
 
(
s
a
m
e
 
a
s
 
t
r
a
i
n
i
n
g
 
d
a
t
a
)

 
 
 
 
 
 
 
 
 
 
 
 
v
d
w
_
r
a
d
i
i
 
=
 
{
1
:
 
1
.
2
0
,
 
6
:
 
1
.
7
0
,
 
7
:
 
1
.
5
5
,
 
8
:
 
1
.
5
2
,
 
9
:
 
1
.
4
7
,
 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
1
5
:
 
1
.
8
0
,
 
1
6
:
 
1
.
8
0
,
 
1
7
:
 
1
.
7
5
,
 
3
5
:
 
1
.
8
5
}

 
 
 
 
 
 
 
 
 
 
 
 
v
d
w
_
v
o
l
u
m
e
 
=
 
0

 
 
 
 
 
 
 
 
 
 
 
 
f
o
r
 
a
t
o
m
 
i
n
 
m
o
l
.
G
e
t
A
t
o
m
s
(
)
:

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
i
f
 
a
t
o
m
.
G
e
t
A
t
o
m
i
c
N
u
m
(
)
 
>
 
1
:

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
#
 
U
s
e
 
B
o
n
d
i
/
B
a
t
s
a
n
o
v
 
i
f
 
a
v
a
i
l
a
b
l
e
,
 
e
l
s
e
 
R
D
K
i
t

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
r
a
d
i
u
s
 
=
 
v
d
w
_
r
a
d
i
i
.
g
e
t
(
a
t
o
m
.
G
e
t
A
t
o
m
i
c
N
u
m
(
)
,
 
C
h
e
m
.
G
e
t
P
e
r
i
o
d
i
c
T
a
b
l
e
(
)
.
G
e
t
R
v
d
w
(
a
t
o
m
.
G
e
t
A
t
o
m
i
c
N
u
m
(
)
)
)

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
v
d
w
_
v
o
l
u
m
e
 
+
=
 
(
4
/
3
)
 
*
 
3
.
1
4
1
5
9
 
*
 
(
r
a
d
i
u
s
 
*
*
 
3
)

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
f
f
v
_
m
w
_
p
e
r
_
v
d
w
_
v
o
l
u
m
e
'
]
 
=
 
m
w
 
/
 
v
d
w
_
v
o
l
u
m
e
 
i
f
 
v
d
w
_
v
o
l
u
m
e
 
>
 
0
 
e
l
s
e
 
0
.
0

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
#
 
2

 
 
 
 
 
 
 
 
 
 
 
 
n
u
m
_
r
o
t
a
t
a
b
l
e
 
=
 
s
u
m
(
1
 
f
o
r
 
b
 
i
n
 
m
o
l
.
G
e
t
B
o
n
d
s
(
)
 
i
f
 
b
.
G
e
t
B
o
n
d
T
y
p
e
(
)
 
=
=
 
C
h
e
m
.
B
o
n
d
T
y
p
e
.
S
I
N
G
L
E
 
a
n
d
 
n
o
t
 
b
.
I
s
I
n
R
i
n
g
(
)
)

 
 
 
 
 
 
 
 
 
 
 
 
#
D
e
s
c
r
i
p
t
o
r
s
.
N
u
m
R
o
t
a
t
a
b
l
e
B
o
n
d
s
(
m
o
l
)

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
t
h
e
r
m
a
l
_
f
l
e
x
i
b
i
l
i
t
y
_
p
e
r
_
a
t
o
m
'
]
 
=
 
n
u
m
_
r
o
t
a
t
a
b
l
e
 
/
 
n
u
m
_
a
t
o
m
s
 
i
f
 
n
u
m
_
a
t
o
m
s
 
>
 
0
 
e
l
s
e
 
0
.
0

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
#
 
1
4

 
 
 
 
 
 
 
 
 
 
 
 
b
a
c
k
b
o
n
e
_
a
t
o
m
s
 
=
 
s
e
t
(
)

 
 
 
 
 
 
 
 
 
 
 
 
i
f
 
l
e
n
(
s
t
a
r
s
)
 
>
=
 
2
:

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
B
G
 
=
 
n
x
.
G
r
a
p
h
(
)

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
f
o
r
 
b
 
i
n
 
m
o
l
.
G
e
t
B
o
n
d
s
(
)
:

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
B
G
.
a
d
d
_
e
d
g
e
(
b
.
G
e
t
B
e
g
i
n
A
t
o
m
I
d
x
(
)
,
 
b
.
G
e
t
E
n
d
A
t
o
m
I
d
x
(
)
)

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
p
a
t
h
 
=
 
n
x
.
s
h
o
r
t
e
s
t
_
p
a
t
h
(
B
G
,
 
s
t
a
r
s
[
0
]
,
 
s
t
a
r
s
[
-
1
]
)

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
b
a
c
k
b
o
n
e
_
a
t
o
m
s
 
=
 
s
e
t
(
p
a
t
h
)

 
 
 
 
 
 
 
 
 
 
 
 
e
l
s
e
:

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
b
a
c
k
b
o
n
e
_
a
t
o
m
s
 
=
 
s
e
t
(
r
a
n
g
e
(
m
o
l
.
G
e
t
N
u
m
A
t
o
m
s
(
)
)
)

 
 
 
 
 
 
 
 
 
 
 
 
a
r
o
m
a
t
i
c
_
c
o
u
n
t
 
=
 
s
u
m
(
1
 
f
o
r
 
i
d
x
 
i
n
 
b
a
c
k
b
o
n
e
_
a
t
o
m
s
 
i
f
 
m
o
l
.
G
e
t
A
t
o
m
W
i
t
h
I
d
x
(
i
d
x
)
.
G
e
t
I
s
A
r
o
m
a
t
i
c
(
)
)

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
b
a
c
k
b
o
n
e
_
a
r
o
m
a
t
i
c
_
r
a
t
i
o
_
t
o
t
a
l
'
]
 
=
 
a
r
o
m
a
t
i
c
_
c
o
u
n
t
 
/
 
m
o
l
.
G
e
t
N
u
m
A
t
o
m
s
(
)
 
i
f
 
m
o
l
.
G
e
t
N
u
m
A
t
o
m
s
(
)
 
>
 
0
 
e
l
s
e
 
0

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
#
 
3

 
 
 
 
 
 
 
 
 
 
 
 
s
p
2
_
c
o
u
n
t
 
=
 
s
u
m
(
1
 
f
o
r
 
a
 
i
n
 
m
o
l
.
G
e
t
A
t
o
m
s
(
)
 
i
f
 
s
t
r
(
a
.
G
e
t
H
y
b
r
i
d
i
z
a
t
i
o
n
(
)
)
 
=
=
 
"
S
P
2
"
)

 
 
 
 
 
 
 
 
 
 
 
 
s
p
3
_
c
o
u
n
t
 
=
 
s
u
m
(
1
 
f
o
r
 
a
 
i
n
 
m
o
l
.
G
e
t
A
t
o
m
s
(
)
 
i
f
 
s
t
r
(
a
.
G
e
t
H
y
b
r
i
d
i
z
a
t
i
o
n
(
)
)
 
=
=
 
"
S
P
3
"
)

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
s
p
2
_
s
p
3
_
r
a
t
i
o
_
a
l
l
_
a
t
o
m
s
'
]
 
=
 
s
p
2
_
c
o
u
n
t
 
/
 
s
p
3
_
c
o
u
n
t
 
i
f
 
s
p
3
_
c
o
u
n
t
 
>
 
0
 
e
l
s
e
 
1
0
0
0

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
#
 
5

 
 
 
 
 
 
 
 
 
 
 
 
#
 
N
,
 
O
,
 
F
,
 
S
 
(
s
t
r
o
n
g
l
y
 
p
o
l
a
r
,
 
H
-
b
o
n
d
i
n
g
 
a
t
o
m
s
)

 
 
 
 
 
 
 
 
 
 
 
 
p
o
l
a
r
_
a
t
o
m
s
 
=
 
s
u
m
(
1
 
f
o
r
 
a
 
i
n
 
m
o
l
.
G
e
t
A
t
o
m
s
(
)
 
i
f
 
a
.
G
e
t
A
t
o
m
i
c
N
u
m
(
)
 
i
n
 
[
7
,
 
8
,
 
9
,
 
1
6
]
)

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
t
h
e
r
m
a
l
_
p
o
l
a
r
_
a
t
o
m
_
f
r
a
c
t
i
o
n
'
]
 
=
 
p
o
l
a
r
_
a
t
o
m
s
 
/
 
n
u
m
_
a
t
o
m
s
 
i
f
 
n
u
m
_
a
t
o
m
s
 
>
 
0
 
e
l
s
e
 
0
.
0

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
#
 
1
7

 
 
 
 
 
 
 
 
 
 
 
 
b
a
c
k
b
o
n
e
_
l
e
n
g
t
h
 
=
 
l
e
n
(
b
a
c
k
b
o
n
e
_
a
t
o
m
s
)

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
b
a
c
k
b
o
n
e
_
l
e
n
g
t
h
'
]
 
=
 
b
a
c
k
b
o
n
e
_
l
e
n
g
t
h

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
#
 
8

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
r
d
k
i
t
_
2
d
_
N
u
m
H
e
t
e
r
o
a
t
o
m
s
'
]
 
=
 
f
l
o
a
t
(
D
e
s
c
r
i
p
t
o
r
s
.
N
u
m
H
e
t
e
r
o
a
t
o
m
s
(
m
o
l
)
)

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
#
 
7
,
 
1
3

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
r
d
k
i
t
_
2
d
_
V
S
A
_
E
S
t
a
t
e
7
'
]
 
=
 
f
l
o
a
t
(
D
e
s
c
r
i
p
t
o
r
s
.
V
S
A
_
E
S
t
a
t
e
7
(
m
o
l
)
)

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
r
d
k
i
t
_
2
d
_
V
S
A
_
E
S
t
a
t
e
8
'
]
 
=
 
f
l
o
a
t
(
D
e
s
c
r
i
p
t
o
r
s
.
V
S
A
_
E
S
t
a
t
e
8
(
m
o
l
)
)

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
a
t
o
m
s
_
p
e
r
_
u
n
i
t
 
=
 
s
u
m
(
1
 
f
o
r
 
a
 
i
n
 
m
o
l
.
G
e
t
A
t
o
m
s
(
)
 
i
f
 
a
.
G
e
t
A
t
o
m
i
c
N
u
m
(
)
 
>
 
1
)

 
 
 
 
 
 
 
 
 
 
 
 
d
p
 
=
 
6
0
0
 
/
 
a
t
o
m
s
_
p
e
r
_
u
n
i
t

 
 
 
 
 
 
 
 
 
 
 
 
b
a
c
k
b
o
n
e
_
c
a
r
b
o
n
s
 
=
 
s
u
m
(
1
 
f
o
r
 
a
 
i
n
 
m
o
l
.
G
e
t
A
t
o
m
s
(
)
 
i
f
 
a
.
G
e
t
A
t
o
m
i
c
N
u
m
(
)
 
=
=
 
6
)

 
 
 
 
 
 
 
 
 
 
 
 
#
 
1
0

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
d
p
_
e
s
t
i
m
a
t
e
d
_
b
a
c
k
b
o
n
e
_
l
e
n
g
t
h
_
a
n
g
s
t
r
o
m
'
]
 
=
 
b
a
c
k
b
o
n
e
_
c
a
r
b
o
n
s
 
*
 
d
p
 
*
 
1
.
5
4

 
 
 
 
 
 
 
 
 
 
 
 
m
w
_
p
e
r
_
u
n
i
t
 
=
 
D
e
s
c
r
i
p
t
o
r
s
.
M
o
l
W
t
(
m
o
l
)

 
 
 
 
 
 
 
 
 
 
 
 
#
 
2
2

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
d
p
_
e
s
t
i
m
a
t
e
d
_
m
o
l
e
c
u
l
a
r
_
w
e
i
g
h
t
'
]
 
=
 
m
w
_
p
e
r
_
u
n
i
t
 
*
 
d
p

 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
G
 
=
 
n
x
.
G
r
a
p
h
(
)

 
 
 
 
 
 
 
 
 
 
 
 
f
o
r
 
a
t
o
m
 
i
n
 
m
o
l
.
G
e
t
A
t
o
m
s
(
)
:

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
G
.
a
d
d
_
n
o
d
e
(
a
t
o
m
.
G
e
t
I
d
x
(
)
)

 
 
 
 
 
 
 
 
 
 
 
 
f
o
r
 
b
o
n
d
 
i
n
 
m
o
l
.
G
e
t
B
o
n
d
s
(
)
:

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
G
.
a
d
d
_
e
d
g
e
(
b
o
n
d
.
G
e
t
B
e
g
i
n
A
t
o
m
I
d
x
(
)
,
 
b
o
n
d
.
G
e
t
E
n
d
A
t
o
m
I
d
x
(
)
)

 
 
 
 
 
 
 
 
 
 
 
 
d
e
g
r
e
e
s
 
=
 
[
d
 
f
o
r
 
_
,
 
d
 
i
n
 
G
.
d
e
g
r
e
e
(
)
]

 
 
 
 
 
 
 
 
 
 
 
 
#
 
1
6
,
 
6

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
"
d
e
g
r
e
e
_
d
i
s
t
r
i
b
u
t
i
o
n
_
m
e
a
n
"
]
 
=
 
n
p
.
m
e
a
n
(
d
e
g
r
e
e
s
)

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
"
d
e
g
r
e
e
_
d
i
s
t
r
i
b
u
t
i
o
n
_
s
t
d
"
]
 
=
 
n
p
.
s
t
d
(
d
e
g
r
e
e
s
)

 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
#
 
1
1

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
"
b
e
t
w
e
e
n
n
e
s
s
_
c
e
n
t
r
a
l
i
t
y
_
m
e
a
n
"
]
 
=
 
n
p
.
m
e
a
n
(
l
i
s
t
(
n
x
.
b
e
t
w
e
e
n
n
e
s
s
_
c
e
n
t
r
a
l
i
t
y
(
G
)
.
v
a
l
u
e
s
(
)
)
)


 
 
 
 
 
 
 
 
 
 
 
 
#
 
9

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
r
d
k
i
t
_
2
d
_
F
r
a
c
t
i
o
n
C
S
P
3
'
]
 
=
 
f
l
o
a
t
(
D
e
s
c
r
i
p
t
o
r
s
.
F
r
a
c
t
i
o
n
C
S
P
3
(
m
o
l
)
)

 
 
 
 
 
 
 
 
 
 
 
 
#
 
1
2
,
 
2
3

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
r
d
k
i
t
_
2
d
_
S
M
R
_
V
S
A
5
'
]
 
=
 
f
l
o
a
t
(
D
e
s
c
r
i
p
t
o
r
s
.
S
M
R
_
V
S
A
5
(
m
o
l
)
)

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
r
d
k
i
t
_
2
d
_
S
M
R
_
V
S
A
1
'
]
 
=
 
f
l
o
a
t
(
D
e
s
c
r
i
p
t
o
r
s
.
S
M
R
_
V
S
A
1
(
m
o
l
)
)

 
 
 
 
 
 
 
 
 
 
 
 
#
 
1
5

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
r
d
k
i
t
_
2
d
_
N
H
O
H
C
o
u
n
t
'
]
 
=
 
i
n
t
(
D
e
s
c
r
i
p
t
o
r
s
.
N
H
O
H
C
o
u
n
t
(
m
o
l
)
)


 
 
 
 
 
 
 
 
 
 
 
 
s
i
d
e
c
h
a
i
n
_
a
t
o
m
s
 
=
 
[
i
d
x
 
f
o
r
 
i
d
x
 
i
n
 
r
a
n
g
e
(
m
o
l
.
G
e
t
N
u
m
A
t
o
m
s
(
)
)
 
i
f
 
i
d
x
 
n
o
t
 
i
n
 
b
a
c
k
b
o
n
e
_
a
t
o
m
s
 
a
n
d
 
m
o
l
.
G
e
t
A
t
o
m
W
i
t
h
I
d
x
(
i
d
x
)
.
G
e
t
A
t
o
m
i
c
N
u
m
(
)
>
1
]

 
 
 
 
 
 
 
 
 
 
 
 
#
 
1
9

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
s
i
d
e
c
h
a
i
n
_
c
o
u
n
t
'
]
 
=
 
l
e
n
(
s
i
d
e
c
h
a
i
n
_
a
t
o
m
s
)


 
 
 
 
 
 
 
 
 
 
 
 
a
r
o
m
a
t
i
c
_
a
t
o
m
s
_
i
n
_
b
a
c
k
b
o
n
e
 
=
 
s
u
m
(
1
 
f
o
r
 
i
d
x
 
i
n
 
b
a
c
k
b
o
n
e
_
a
t
o
m
s
 
i
f
 
m
o
l
.
G
e
t
A
t
o
m
W
i
t
h
I
d
x
(
i
d
x
)
.
G
e
t
I
s
A
r
o
m
a
t
i
c
(
)
)

 
 
 
 
 
 
 
 
 
 
 
 
r
o
t
a
t
a
b
l
e
_
b
o
n
d
s
_
i
n
_
b
a
c
k
b
o
n
e
 
=
 
s
u
m
(
1
 
f
o
r
 
b
 
i
n
 
m
o
l
.
G
e
t
B
o
n
d
s
(
)
 
i
f
 
b
.
G
e
t
B
e
g
i
n
A
t
o
m
I
d
x
(
)
 
i
n
 
b
a
c
k
b
o
n
e
_
a
t
o
m
s
 
a
n
d
 
b
.
G
e
t
E
n
d
A
t
o
m
I
d
x
(
)
 
i
n
 
b
a
c
k
b
o
n
e
_
a
t
o
m
s
 
a
n
d
 
_
i
s
_
r
o
t
a
t
a
b
l
e
_
b
o
n
d
(
b
)
)

 
 
 
 
 
 
 
 
 
 
 
 
b
a
c
k
b
o
n
e
_
p
l
a
n
a
r
i
t
y
_
s
c
o
r
e
 
=
 
(
a
r
o
m
a
t
i
c
_
a
t
o
m
s
_
i
n
_
b
a
c
k
b
o
n
e
 
/
 
b
a
c
k
b
o
n
e
_
l
e
n
g
t
h
 
i
f
 
b
a
c
k
b
o
n
e
_
l
e
n
g
t
h
 
>
 
0
 
e
l
s
e
 
0
)
 
-
 
(
r
o
t
a
t
a
b
l
e
_
b
o
n
d
s
_
i
n
_
b
a
c
k
b
o
n
e
 
/
 
b
a
c
k
b
o
n
e
_
l
e
n
g
t
h
 
i
f
 
b
a
c
k
b
o
n
e
_
l
e
n
g
t
h
 
>
 
0
 
e
l
s
e
 
0
)

 
 
 
 
 
 
 
 
 
 
 
 
b
a
c
k
b
o
n
e
_
r
o
t
a
t
a
b
l
e
_
b
o
n
d
s
 
=
 
s
u
m
(
1
 
f
o
r
 
b
 
i
n
 
m
o
l
.
G
e
t
B
o
n
d
s
(
)
 
i
f
 
b
.
G
e
t
B
e
g
i
n
A
t
o
m
I
d
x
(
)
 
i
n
 
b
a
c
k
b
o
n
e
_
a
t
o
m
s
 
a
n
d
 
b
.
G
e
t
E
n
d
A
t
o
m
I
d
x
(
)
 
i
n
 
b
a
c
k
b
o
n
e
_
a
t
o
m
s
 
a
n
d
 
_
i
s
_
r
o
t
a
t
a
b
l
e
_
b
o
n
d
(
b
)
)

 
 
 
 
 
 
 
 
 
 
 
 
b
a
c
k
b
o
n
e
_
f
l
e
x
i
b
i
l
i
t
y
_
s
c
o
r
e
 
=
 
b
a
c
k
b
o
n
e
_
r
o
t
a
t
a
b
l
e
_
b
o
n
d
s
/
b
a
c
k
b
o
n
e
_
l
e
n
g
t
h
 
i
f
 
b
a
c
k
b
o
n
e
_
l
e
n
g
t
h
>
0
 
e
l
s
e
 
0

 
 
 
 
 
 
 
 
 
 
 
 
#
 
2
0
,
 
2
5

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
"
r
i
g
i
d
i
t
y
_
i
n
d
e
x
"
]
 
=
 
b
a
c
k
b
o
n
e
_
p
l
a
n
a
r
i
t
y
_
s
c
o
r
e
 
-
 
b
a
c
k
b
o
n
e
_
f
l
e
x
i
b
i
l
i
t
y
_
s
c
o
r
e

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
b
a
c
k
b
o
n
e
_
p
l
a
n
a
r
i
t
y
_
s
c
o
r
e
'
]
 
=
 
b
a
c
k
b
o
n
e
_
p
l
a
n
a
r
i
t
y
_
s
c
o
r
e

 
 
 
 
 
 
 
 
 
 
 
 
#
 
3
0

 
 
 
 
 
 
 
 
 
 
 
 
a
r
o
m
a
t
i
c
_
c
o
u
n
t
 
=
 
s
u
m
(
1
 
f
o
r
 
i
d
x
 
i
n
 
b
a
c
k
b
o
n
e
_
a
t
o
m
s
 
i
f
 
m
o
l
.
G
e
t
A
t
o
m
W
i
t
h
I
d
x
(
i
d
x
)
.
G
e
t
I
s
A
r
o
m
a
t
i
c
(
)
)

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
b
a
c
k
b
o
n
e
_
a
r
o
m
a
t
i
c
_
f
r
a
c
t
i
o
n
'
]
 
=
 
a
r
o
m
a
t
i
c
_
c
o
u
n
t
 
/
 
b
a
c
k
b
o
n
e
_
l
e
n
g
t
h
 
i
f
 
b
a
c
k
b
o
n
e
_
l
e
n
g
t
h
 
>
 
0
 
e
l
s
e
 
0

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
#
 
3
4

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
s
i
d
e
c
h
a
i
n
_
p
o
l
a
r
i
t
y
_
i
n
d
e
x
'
]
 
=
 
s
u
m
(
1
 
f
o
r
 
i
d
x
 
i
n
 
s
i
d
e
c
h
a
i
n
_
a
t
o
m
s
 
i
f
 
m
o
l
.
G
e
t
A
t
o
m
W
i
t
h
I
d
x
(
i
d
x
)
.
G
e
t
A
t
o
m
i
c
N
u
m
(
)
 
i
n
 
{
7
,
8
,
9
,
1
6
,
1
7
,
3
5
,
5
3
}
)
/
l
e
n
(
s
i
d
e
c
h
a
i
n
_
a
t
o
m
s
)
 
i
f
 
s
i
d
e
c
h
a
i
n
_
a
t
o
m
s
 
e
l
s
e
 
0


 
 
 
 
 
 
 
 
 
 
 
 
#
 
2
1

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
r
d
k
i
t
_
2
d
_
f
r
_
h
a
l
o
g
e
n
'
]
 
=
 
f
l
o
a
t
(
D
e
s
c
r
i
p
t
o
r
s
.
f
r
_
h
a
l
o
g
e
n
(
m
o
l
)
)

 
 
 
 
 
 
 
 
 
 
 
 
#
 
2
8

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
r
d
k
i
t
_
2
d
_
M
o
l
L
o
g
P
'
]
 
=
 
f
l
o
a
t
(
D
e
s
c
r
i
p
t
o
r
s
.
M
o
l
L
o
g
P
(
m
o
l
)
)

 
 
 
 
 
 
 
 
 
 
 
 
#
 
2
9
,
 
3
2

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
r
d
k
i
t
_
2
d
_
S
l
o
g
P
_
V
S
A
1
2
'
]
 
=
 
f
l
o
a
t
(
D
e
s
c
r
i
p
t
o
r
s
.
S
l
o
g
P
_
V
S
A
1
2
(
m
o
l
)
)

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
r
d
k
i
t
_
2
d
_
S
l
o
g
P
_
V
S
A
5
'
]
 
=
 
f
l
o
a
t
(
D
e
s
c
r
i
p
t
o
r
s
.
S
l
o
g
P
_
V
S
A
5
(
m
o
l
)
)

 
 
 
 
 
 
 
 
 
 
 
 
#
 
3
1

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
r
d
k
i
t
_
2
d
_
E
S
t
a
t
e
_
V
S
A
1
0
'
]
 
=
 
f
l
o
a
t
(
D
e
s
c
r
i
p
t
o
r
s
.
E
S
t
a
t
e
_
V
S
A
1
0
(
m
o
l
)
)

 
 
 
 
 
 
 
 
 
 
 
 
#
 
3
3

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
[
'
r
d
k
i
t
_
2
d
_
N
u
m
H
D
o
n
o
r
s
'
]
 
=
 
i
n
t
(
D
e
s
c
r
i
p
t
o
r
s
.
N
u
m
H
D
o
n
o
r
s
(
m
o
l
)
)

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
f
e
a
t
u
r
e
s

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
e
x
c
e
p
t
 
E
x
c
e
p
t
i
o
n
 
a
s
 
e
:

 
 
 
 
 
 
 
 
 
 
 
 
#
 
R
e
t
u
r
n
 
z
e
r
o
s
 
o
n
 
e
r
r
o
r

 
 
 
 
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
{
f
:
 
0
.
0
 
f
o
r
 
f
 
i
n
 
s
e
l
f
.
T
O
P
_
1
0
_
F
E
A
T
U
R
E
S
}

 
 
 
 

 
 
 
 
d
e
f
 
e
x
t
r
a
c
t
_
b
a
s
e
l
i
n
e
_
2
1
_
f
e
a
t
u
r
e
s
(
s
e
l
f
,
 
s
m
i
l
e
s
)
:

 
 
 
 
 
 
 
 
"
"
"

 
 
 
 
 
 
 
 
E
x
t
r
a
c
t
 
2
1
 
b
a
s
e
l
i
n
e
 
c
h
e
m
i
s
t
r
y
 
f
e
a
t
u
r
e
s
 
(
s
t
r
i
n
g
-
b
a
s
e
d
 
+
 
s
i
m
p
l
e
 
R
D
K
i
t
)

 
 
 
 
 
 
 
 
T
h
e
s
e
 
a
r
e
 
t
h
e
 
f
e
a
t
u
r
e
s
 
f
r
o
m
 
t
h
e
 
0
.
0
7
5
3
3
 
b
a
s
e
l
i
n
e
 
m
o
d
e
l

 
 
 
 
 
 
 
 
"
"
"

 
 
 
 
 
 
 
 
t
r
y
:

 
 
 
 
 
 
 
 
 
 
 
 
s
m
i
l
e
s
_
s
t
r
 
=
 
s
t
r
(
s
m
i
l
e
s
)
 
i
f
 
p
d
.
n
o
t
n
a
(
s
m
i
l
e
s
)
 
e
l
s
e
 
"
"

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
#
 
S
t
r
i
n
g
-
b
a
s
e
d
 
f
e
a
t
u
r
e
s
 
(
1
0
)

 
 
 
 
 
 
 
 
 
 
 
 
b
a
s
i
c
 
=
 
{

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
s
m
i
l
e
s
_
l
e
n
g
t
h
'
:
 
l
e
n
(
s
m
i
l
e
s
_
s
t
r
)
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
c
a
r
b
o
n
_
c
o
u
n
t
'
:
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
C
'
)
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
n
i
t
r
o
g
e
n
_
c
o
u
n
t
'
:
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
N
'
)
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
o
x
y
g
e
n
_
c
o
u
n
t
'
:
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
O
'
)
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
s
u
l
f
u
r
_
c
o
u
n
t
'
:
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
S
'
)
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
f
l
u
o
r
i
n
e
_
c
o
u
n
t
'
:
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
F
'
)
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
r
i
n
g
_
c
o
u
n
t
'
:
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
c
'
)
 
+
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
C
1
'
)
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
d
o
u
b
l
e
_
b
o
n
d
_
c
o
u
n
t
'
:
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
=
'
)
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
t
r
i
p
l
e
_
b
o
n
d
_
c
o
u
n
t
'
:
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
#
'
)
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
b
r
a
n
c
h
_
c
o
u
n
t
'
:
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
(
'
)
,

 
 
 
 
 
 
 
 
 
 
 
 
}

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
#
 
C
h
e
m
i
s
t
r
y
-
b
a
s
e
d
 
f
e
a
t
u
r
e
s
 
(
1
1
)

 
 
 
 
 
 
 
 
 
 
 
 
n
u
m
_
s
i
d
e
_
c
h
a
i
n
s
 
=
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
(
'
)

 
 
 
 
 
 
 
 
 
 
 
 
b
a
c
k
b
o
n
e
_
c
a
r
b
o
n
s
 
=
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
C
'
)
 
-
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
C
(
'
)

 
 
 
 
 
 
 
 
 
 
 
 
a
r
o
m
a
t
i
c
_
c
o
u
n
t
 
=
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
c
'
)

 
 
 
 
 
 
 
 
 
 
 
 
h
_
b
o
n
d
_
d
o
n
o
r
s
 
=
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
O
'
)
 
+
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
N
'
)

 
 
 
 
 
 
 
 
 
 
 
 
h
_
b
o
n
d
_
a
c
c
e
p
t
o
r
s
 
=
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
O
'
)
 
+
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
N
'
)

 
 
 
 
 
 
 
 
 
 
 
 
n
u
m
_
r
i
n
g
s
 
=
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
1
'
)
 
+
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
2
'
)

 
 
 
 
 
 
 
 
 
 
 
 
s
i
n
g
l
e
_
b
o
n
d
s
 
=
 
l
e
n
(
s
m
i
l
e
s
_
s
t
r
)
 
-
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
=
'
)
 
-
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
#
'
)
 
-
 
a
r
o
m
a
t
i
c
_
c
o
u
n
t

 
 
 
 
 
 
 
 
 
 
 
 
h
a
l
o
g
e
n
_
c
o
u
n
t
 
=
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
F
'
)
 
+
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
C
l
'
)
 
+
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
B
r
'
)

 
 
 
 
 
 
 
 
 
 
 
 
h
e
t
e
r
o
a
t
o
m
_
c
o
u
n
t
 
=
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
N
'
)
 
+
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
O
'
)
 
+
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
S
'
)

 
 
 
 
 
 
 
 
 
 
 
 
m
w
_
e
s
t
i
m
a
t
e
 
=
 
(

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
C
'
)
 
*
 
1
2
 
+
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
O
'
)
 
*
 
1
6
 
+
 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
N
'
)
 
*
 
1
4
 
+
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
S
'
)
 
*
 
3
2
 
+
 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
s
m
i
l
e
s
_
s
t
r
.
c
o
u
n
t
(
'
F
'
)
 
*
 
1
9

 
 
 
 
 
 
 
 
 
 
 
 
)

 
 
 
 
 
 
 
 
 
 
 
 
b
r
a
n
c
h
i
n
g
_
r
a
t
i
o
 
=
 
n
u
m
_
s
i
d
e
_
c
h
a
i
n
s
 
/
 
m
a
x
(
b
a
c
k
b
o
n
e
_
c
a
r
b
o
n
s
,
 
1
)

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
#
 
C
o
m
b
i
n
e

 
 
 
 
 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
 
=
 
{

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
*
*
b
a
s
i
c
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
n
u
m
_
s
i
d
e
_
c
h
a
i
n
s
'
:
 
n
u
m
_
s
i
d
e
_
c
h
a
i
n
s
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
b
a
c
k
b
o
n
e
_
c
a
r
b
o
n
s
'
:
 
b
a
c
k
b
o
n
e
_
c
a
r
b
o
n
s
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
a
r
o
m
a
t
i
c
_
c
o
u
n
t
'
:
 
a
r
o
m
a
t
i
c
_
c
o
u
n
t
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
h
_
b
o
n
d
_
d
o
n
o
r
s
'
:
 
h
_
b
o
n
d
_
d
o
n
o
r
s
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
h
_
b
o
n
d
_
a
c
c
e
p
t
o
r
s
'
:
 
h
_
b
o
n
d
_
a
c
c
e
p
t
o
r
s
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
n
u
m
_
r
i
n
g
s
'
:
 
n
u
m
_
r
i
n
g
s
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
s
i
n
g
l
e
_
b
o
n
d
s
'
:
 
s
i
n
g
l
e
_
b
o
n
d
s
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
h
a
l
o
g
e
n
_
c
o
u
n
t
'
:
 
h
a
l
o
g
e
n
_
c
o
u
n
t
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
h
e
t
e
r
o
a
t
o
m
_
c
o
u
n
t
'
:
 
h
e
t
e
r
o
a
t
o
m
_
c
o
u
n
t
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
m
w
_
e
s
t
i
m
a
t
e
'
:
 
m
w
_
e
s
t
i
m
a
t
e
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
'
b
r
a
n
c
h
i
n
g
_
r
a
t
i
o
'
:
 
b
r
a
n
c
h
i
n
g
_
r
a
t
i
o
,

 
 
 
 
 
 
 
 
 
 
 
 
}

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
f
e
a
t
u
r
e
s

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
e
x
c
e
p
t
 
E
x
c
e
p
t
i
o
n
 
a
s
 
e
:

 
 
 
 
 
 
 
 
 
 
 
 
#
 
R
e
t
u
r
n
 
z
e
r
o
s
 
o
n
 
e
r
r
o
r

 
 
 
 
 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
{
f
:
 
0
.
0
 
f
o
r
 
f
 
i
n
 
s
e
l
f
.
B
A
S
E
L
I
N
E
_
2
1
_
F
E
A
T
U
R
E
S
}

 
 
 
 

 
 
 
 
d
e
f
 
p
r
e
p
a
r
e
_
f
e
a
t
u
r
e
s
(
s
e
l
f
,
 
d
f
)
:

 
 
 
 
 
 
 
 
"
"
"

 
 
 
 
 
 
 
 
M
a
i
n
 
m
e
t
h
o
d
:
 
E
x
t
r
a
c
t
 
a
l
l

 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
T
h
i
s
 
m
e
t
h
o
d
 
i
s
 
c
a
l
l
e
d
 
b
y
 
t
h
e
 
r
e
s
t
 
o
f
 
t
h
e
 
n
o
t
e
b
o
o
k
,
 
s
o
 
k
e
e
p
 
t
h
e
 
s
a
m
e
 
s
i
g
n
a
t
u
r
e
.

 
 
 
 
 
 
 
 
"
"
"

 
 
 
 
 
 
 
 
p
r
i
n
t
(
f
"
E
x
t
r
a
c
t
i
n
g
 
(
2
1
 
b
a
s
e
l
i
n
e
 
+
 
n
e
w
)
 
f
o
r
 
{
l
e
n
(
d
f
)
}
 
m
o
l
e
c
u
l
e
s
.
.
.
"
)

 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
#
 
D
e
f
i
n
e
 
f
e
a
t
u
r
e
 
l
i
s
t
s

 
 
 
 
 
 
 
 
s
e
l
f
.
B
A
S
E
L
I
N
E
_
2
1
_
F
E
A
T
U
R
E
S
 
=
 
[

 
 
 
 
 
 
 
 
 
 
 
 
'
s
m
i
l
e
s
_
l
e
n
g
t
h
'
,
 
'
c
a
r
b
o
n
_
c
o
u
n
t
'
,
 
'
n
i
t
r
o
g
e
n
_
c
o
u
n
t
'
,
 
'
o
x
y
g
e
n
_
c
o
u
n
t
'
,

 
 
 
 
 
 
 
 
 
 
 
 
'
s
u
l
f
u
r
_
c
o
u
n
t
'
,
 
'
f
l
u
o
r
i
n
e
_
c
o
u
n
t
'
,
 
'
r
i
n
g
_
c
o
u
n
t
'
,
 
'
d
o
u
b
l
e
_
b
o
n
d
_
c
o
u
n
t
'
,

 
 
 
 
 
 
 
 
 
 
 
 
'
t
r
i
p
l
e
_
b
o
n
d
_
c
o
u
n
t
'
,
 
'
b
r
a
n
c
h
_
c
o
u
n
t
'
,
 
'
n
u
m
_
s
i
d
e
_
c
h
a
i
n
s
'
,
 
'
b
a
c
k
b
o
n
e
_
c
a
r
b
o
n
s
'
,

 
 
 
 
 
 
 
 
 
 
 
 
'
a
r
o
m
a
t
i
c
_
c
o
u
n
t
'
,
 
'
h
_
b
o
n
d
_
d
o
n
o
r
s
'
,
 
'
h
_
b
o
n
d
_
a
c
c
e
p
t
o
r
s
'
,
 
'
n
u
m
_
r
i
n
g
s
'
,

 
 
 
 
 
 
 
 
 
 
 
 
'
s
i
n
g
l
e
_
b
o
n
d
s
'
,
 
'
h
a
l
o
g
e
n
_
c
o
u
n
t
'
,
 
'
h
e
t
e
r
o
a
t
o
m
_
c
o
u
n
t
'
,
 
'
m
w
_
e
s
t
i
m
a
t
e
'
,

 
 
 
 
 
 
 
 
 
 
 
 
'
b
r
a
n
c
h
i
n
g
_
r
a
t
i
o
'

 
 
 
 
 
 
 
 
]

 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
s
e
l
f
.
T
O
P
_
1
0
_
F
E
A
T
U
R
E
S
 
=
 
[

 
 
 
 
 
 
 
 
 
 
 
 
'
g
r
a
p
h
_
s
t
a
r
_
d
i
s
t
a
n
c
e
'
,
 
'
f
f
v
_
m
w
_
p
e
r
_
v
d
w
_
v
o
l
u
m
e
'
,
 
'
t
h
e
r
m
a
l
_
f
l
e
x
i
b
i
l
i
t
y
_
p
e
r
_
a
t
o
m
'
,

 
 
 
 
 
 
 
 
 
 
 
 
'
b
a
c
k
b
o
n
e
_
a
r
o
m
a
t
i
c
_
r
a
t
i
o
_
t
o
t
a
l
'
,
 
'
s
p
2
_
s
p
3
_
r
a
t
i
o
_
a
l
l
_
a
t
o
m
s
'
,
 

 
 
 
 
 
 
 
 
 
 
 
 
'
t
h
e
r
m
a
l
_
p
o
l
a
r
_
a
t
o
m
_
f
r
a
c
t
i
o
n
'
,
 
'
b
a
c
k
b
o
n
e
_
l
e
n
g
t
h
'
,
 
'
r
d
k
i
t
_
2
d
_
N
u
m
H
e
t
e
r
o
a
t
o
m
s
'
,

 
 
 
 
 
 
 
 
 
 
 
 
'
r
d
k
i
t
_
2
d
_
V
S
A
_
E
S
t
a
t
e
8
'
,
 
'
d
p
_
e
s
t
i
m
a
t
e
d
_
b
a
c
k
b
o
n
e
_
l
e
n
g
t
h
_
a
n
g
s
t
r
o
m
'

 
 
 
 
 
 
 
 
]

 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
a
l
l
_
f
e
a
t
u
r
e
s
 
=
 
[
]

 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
f
o
r
 
i
d
x
,
 
s
m
i
l
e
s
 
i
n
 
t
q
d
m
(
d
f
[
'
S
M
I
L
E
S
'
]
.
i
t
e
m
s
(
)
,
 
t
o
t
a
l
=
l
e
n
(
d
f
)
)
:

 
 
 
 
 
 
 
 
 
 
 
 
#
 
E
x
t
r
a
c
t
 
b
a
s
e
l
i
n
e
 
f
e
a
t
u
r
e
s

 
 
 
 
 
 
 
 
 
 
 
 
b
a
s
e
l
i
n
e
_
f
e
a
t
s
 
=
 
s
e
l
f
.
e
x
t
r
a
c
t
_
b
a
s
e
l
i
n
e
_
2
1
_
f
e
a
t
u
r
e
s
(
s
m
i
l
e
s
)

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
#
 
E
x
t
r
a
c
t
 
n
e
w
 
f
e
a
t
u
r
e
s

 
 
 
 
 
 
 
 
 
 
 
 
n
e
w
_
f
e
a
t
s
 
=
 
s
e
l
f
.
e
x
t
r
a
c
t
_
t
o
p
_
1
0
_
n
e
w
_
f
e
a
t
u
r
e
s
(
s
m
i
l
e
s
)

 
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
#
 
C
o
m
b
i
n
e

 
 
 
 
 
 
 
 
 
 
 
 
c
o
m
b
i
n
e
d
 
=
 
{
*
*
b
a
s
e
l
i
n
e
_
f
e
a
t
s
,
 
*
*
n
e
w
_
f
e
a
t
s
}

 
 
 
 
 
 
 
 
 
 
 
 
a
l
l
_
f
e
a
t
u
r
e
s
.
a
p
p
e
n
d
(
c
o
m
b
i
n
e
d
)

 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
f
e
a
t
u
r
e
s
_
d
f
 
=
 
p
d
.
D
a
t
a
F
r
a
m
e
(
a
l
l
_
f
e
a
t
u
r
e
s
,
 
i
n
d
e
x
=
d
f
.
i
n
d
e
x
)

 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
p
r
i
n
t
(
f
"
Ã¢
Å“
â€¦
 
E
x
t
r
a
c
t
e
d
 
{
l
e
n
(
f
e
a
t
u
r
e
s
_
d
f
.
c
o
l
u
m
n
s
)
}
 
f
e
a
t
u
r
e
s
:
"
)

 
 
 
 
 
 
 
 
p
r
i
n
t
(
f
"
 
 
 
S
h
a
p
e
:
 
{
f
e
a
t
u
r
e
s
_
d
f
.
s
h
a
p
e
}
"
)

 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
f
e
a
t
u
r
e
s
_
d
f

 
 
 
 

 
 
 
 
#
 
K
e
e
p
 
t
h
e
s
e
 
m
e
t
h
o
d
s
 
f
o
r
 
c
o
m
p
a
t
i
b
i
l
i
t
y
 
(
t
h
e
y
'
r
e
 
n
o
t
 
u
s
e
d
 
b
u
t
 
m
i
g
h
t
 
b
e
 
r
e
f
e
r
e
n
c
e
d
)

 
 
 
 
d
e
f
 
c
r
e
a
t
e
_
c
h
e
m
i
s
t
r
y
_
f
e
a
t
u
r
e
s
(
s
e
l
f
,
 
d
f
)
:

 
 
 
 
 
 
 
 
"
"
"
C
o
m
p
a
t
i
b
i
l
i
t
y
 
m
e
t
h
o
d
 
-
 
r
e
d
i
r
e
c
t
s
 
t
o
 
p
r
e
p
a
r
e
_
f
e
a
t
u
r
e
s
"
"
"

 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
s
e
l
f
.
p
r
e
p
a
r
e
_
f
e
a
t
u
r
e
s
(
d
f
)

 
 
 
 

 
 
 
 
d
e
f
 
c
r
e
a
t
e
_
d
e
s
c
r
i
p
t
o
r
_
f
e
a
t
u
r
e
s
(
s
e
l
f
,
 
d
f
)
:

 
 
 
 
 
 
 
 
"
"
"
C
o
m
p
a
t
i
b
i
l
i
t
y
 
m
e
t
h
o
d
 
-
 
r
e
d
i
r
e
c
t
s
 
t
o
 
p
r
e
p
a
r
e
_
f
e
a
t
u
r
e
s
"
"
"

 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
s
e
l
f
.
p
r
e
p
a
r
e
_
f
e
a
t
u
r
e
s
(
d
f
)

 
 
 
 

 
 
 
 
d
e
f
 
c
r
e
a
t
e
_
f
i
n
g
e
r
p
r
i
n
t
_
f
e
a
t
u
r
e
s
(
s
e
l
f
,
 
d
f
,
 
n
_
b
i
t
s
=
1
0
2
4
)
:

 
 
 
 
 
 
 
 
"
"
"
C
o
m
p
a
t
i
b
i
l
i
t
y
 
m
e
t
h
o
d
 
-
 
r
e
t
u
r
n
s
 
e
m
p
t
y
 
(
f
i
n
g
e
r
p
r
i
n
t
s
 
n
o
t
 
u
s
e
d
)
"
"
"

 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
p
d
.
D
a
t
a
F
r
a
m
e
(
i
n
d
e
x
=
d
f
.
i
n
d
e
x
)

 
 
 
 

 
 
 
 
d
e
f
 
c
r
e
a
t
e
_
f
a
l
l
b
a
c
k
_
f
e
a
t
u
r
e
s
(
s
e
l
f
,
 
d
f
)
:

 
 
 
 
 
 
 
 
"
"
"
C
o
m
p
a
t
i
b
i
l
i
t
y
 
m
e
t
h
o
d
 
-
 
r
e
d
i
r
e
c
t
s
 
t
o
 
p
r
e
p
a
r
e
_
f
e
a
t
u
r
e
s
"
"
"

 
 
 
 
 
 
 
 
r
e
t
u
r
n
 
s
e
l
f
.
p
r
e
p
a
r
e
_
f
e
a
t
u
r
e
s
(
d
f
)



#
 
I
n
i
t
i
a
l
i
z
e
 
p
r
o
c
e
s
s
o
r

p
r
o
c
e
s
s
o
r
 
=
 
R
o
b
u
s
t
M
o
l
e
c
u
l
a
r
P
r
o
c
e
s
s
o
r
(
)

p
r
i
n
t
(
"
Ã¢
Å“
â€œ
 
F
e
a
t
u
r
e
 
p
r
o
c
e
s
s
o
r
 
i
n
i
t
i
a
l
i
z
e
d
"
)




class RobustRandomForestModel:
    """Random Forest ensemble model - sklearn compatible"""
    
    def __init__(self, n_targets=5, n_ensemble=5):
        self.n_targets = n_targets
        self.n_ensemble = n_ensemble
        self.models = {}
        self.scalers = {}
        self.feature_names = None
    
    def train(self, X_train, y_train, X_val, y_val, target_names):
        """Train ensemble of Random Forest models for each target"""
        results = {}
        
        for i, target in enumerate(target_names):
            print(f"\nTraining Random Forest Ensemble for {target}...")
            print(f"  Training {self.n_ensemble} models with different random seeds...")
            
            try:
                y_train_target = y_train[:, i]
                y_val_target = y_val[:, i]
                
                train_mask = ~np.isnan(y_train_target)
                val_mask = ~np.isnan(y_val_target)
                
                if train_mask.sum() == 0:
                    print(f"No training data for {target}")
                    continue
                
                X_train_filtered = X_train[train_mask]
                y_train_filtered = y_train_target[train_mask]
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_filtered)
                self.scalers[target] = scaler
                
                ensemble_models = []
                ensemble_scores = []
                
                for j in range(self.n_ensemble):
                    model = RandomForestRegressor(
                        n_estimators=500,
                        max_depth=15,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        max_features='sqrt',
                        random_state=42 + i * 10 + j,
                        n_jobs=-1
                    )
                    
                    # Random Forest doesn't support eval_set - train without it
                    model.fit(X_train_scaled, y_train_filtered)
                    
                    if val_mask.sum() > 0:
                        X_val_filtered = X_val[val_mask]
                        y_val_filtered = y_val_target[val_mask]
                        X_val_scaled = scaler.transform(X_val_filtered)
                        
                        y_pred = model.predict(X_val_scaled)
                        mae = mean_absolute_error(y_val_filtered, y_pred)
                        ensemble_scores.append(mae)
                    
                    ensemble_models.append(model)
                
                self.models[target] = ensemble_models
                
                if val_mask.sum() > 0:
                    ensemble_preds = np.array([m.predict(X_val_scaled) for m in ensemble_models])
                    ensemble_pred_mean = ensemble_preds.mean(axis=0)
                    
                    results[target] = {
                        'rmse': np.sqrt(mean_squared_error(y_val_filtered, ensemble_pred_mean)),
                        'mae': mean_absolute_error(y_val_filtered, ensemble_pred_mean),
                        'r2': r2_score(y_val_filtered, ensemble_pred_mean),
                        'individual_maes': ensemble_scores,
                        'ensemble_improvement': np.mean(ensemble_scores) - mean_absolute_error(y_val_filtered, ensemble_pred_mean)
                    }
                    
                    print(f"  Individual model MAEs: {np.mean(ensemble_scores):.4f} Ã‚Â± {np.std(ensemble_scores):.4f}")
                    print(f"  Ensemble MAE: {results[target]['mae']:.4f} (Ã¢â€ â€œ {results[target]['ensemble_improvement']:.4f})")
                    print(f"  Ensemble RMSE: {results[target]['rmse']:.4f}")
                    print(f"  Ensemble RÃ‚Â²: {results[target]['r2']:.4f}")
                else:
                    print(f"  Trained {self.n_ensemble} models on {len(y_train_filtered)} samples (no validation)")
                
            except Exception as e:
                print(f"  Training failed for {target}: {e}")
                continue
        
        return results
    
    def predict(self, X_test, target_names):
        """Predict on test data using ensemble averaging"""
        predictions = np.zeros((len(X_test), len(target_names)))
        
        for i, target in enumerate(target_names):
            try:
                if target in self.models and target in self.scalers:
                    scaler = self.scalers[target]
                    ensemble_models = self.models[target]
                    
                    X_test_clean = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
                    X_test_scaled = scaler.transform(X_test_clean)
                    
                    ensemble_preds = np.array([model.predict(X_test_scaled) for model in ensemble_models])
                    pred = ensemble_preds.mean(axis=0)
                    predictions[:, i] = pred
                    
                    print(f"Predicted {target}: range [{pred.min():.4f}, {pred.max():.4f}] (ensemble of {len(ensemble_models)} models)")
                else:
                    print(f"No model available for {target}, using zeros")
                    predictions[:, i] = 0.0
                    
            except Exception as e:
                print(f"Prediction failed for {target}: {e}, using zeros")
                predictions[:, i] = 0.0
        
        return predictions



# Prepare features with comprehensive error handling
print("Preparing training features...")
try:
    train_features = processor.prepare_features(train_df)
    print(f"Training features shape: {train_features.shape}")
except Exception as e:
    print(f"Training feature preparation failed: {e}")
    raise

# Align with training data and prepare targets
try:
    common_indices = train_df.index.intersection(train_features.index)
    train_df_filtered = train_df.loc[common_indices]
    train_features_filtered = train_features.loc[common_indices]
    
    print(f"Aligned samples: {len(common_indices)}")
    
    # Prepare targets
    y = train_df_filtered[target_cols].values
    X = train_features_filtered.values
    
    # Remove samples with NaN/inf in features
    feature_mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    X = X[feature_mask]
    y = y[feature_mask]
    
    print(f"Final training set: {len(X)} samples with {X.shape[1]} features")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Train: {X_train.shape}, Validation: {X_val.shape}")
    
except Exception as e:
    print(f"Data preparation failed: {e}")
    raise


# Train Random Forest model
print("Training Random Forest model...")
try:
    xgb_model = RobustRandomForestModel(n_targets=len(target_cols))
    xgb_results = xgb_model.train(X_train, y_train, X_val, y_val, target_cols)
    
    print("\nRandom Forest Training Results:")
    for target, metrics in xgb_results.items():
        print(f"{target}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, RÃ‚Â²={metrics['r2']:.4f}")
        
except Exception as e:
    print(f"Model training failed: {e}")
    raise


# Prepare test features with robust error handling
print("Preparing test features...")
try:
    test_features = processor.prepare_features(test_df)
    print(f"Test features shape: {test_features.shape}")
    
    # Align test features with training features
    if hasattr(train_features_filtered, 'columns'):
        common_features = train_features_filtered.columns.intersection(test_features.columns)
        print(f"Common features: {len(common_features)}")
        
        if len(common_features) > 0:
            # Use common features
            test_features_aligned = test_features[common_features].copy()
            
            # Fill missing values with training medians
            train_medians = train_features_filtered[common_features].median()
            test_features_filled = test_features_aligned.fillna(train_medians)
            
            # Ensure same feature order as training
            missing_features = set(train_features_filtered.columns) - set(test_features_filled.columns)
            for feature in missing_features:
                test_features_filled[feature] = 0.0
            
            test_features_final = test_features_filled[train_features_filtered.columns]
        else:
            print("Warning: No common features, using test features as-is")
            test_features_final = test_features.fillna(0.0)
    else:
        test_features_final = test_features.fillna(0.0)
    
    print(f"Final test features shape: {test_features_final.shape}")
    
except Exception as e:
    print(f"Test feature preparation failed: {e}")
    # Create minimal fallback features
    test_features_final = pd.DataFrame({
        'smiles_length': test_df['SMILES'].str.len().fillna(0),
        'constant_feature': 1.0
    }, index=test_df.index)
    print(f"Using fallback features: {test_features_final.shape}")


# Generate predictions with robust error handling
print("Generating predictions...")
try:
    X_test = test_features_final.values
    
    # Handle any remaining NaN/inf values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Make predictions
    xgb_predictions = xgb_model.predict(X_test, target_cols)
    
    print(f"Predictions shape: {xgb_predictions.shape}")
    print("Prediction summary:")
    for i, target in enumerate(target_cols):
        pred_min, pred_max = xgb_predictions[:, i].min(), xgb_predictions[:, i].max()
        pred_mean = xgb_predictions[:, i].mean()
        print(f"  {target}: [{pred_min:.4f}, {pred_max:.4f}], mean: {pred_mean:.4f}")

except Exception as e:
    print(f"Prediction generation failed: {e}")
    # Ultimate fallback: use zeros
    xgb_predictions = np.zeros((len(test_df), len(target_cols)))
    print("Using zero predictions as fallback")


# Create submission with robust error handling
print("Creating submission...")
try:
    submission = sample_submission.copy()
    
    # Ensure we have the right number of predictions
    if len(xgb_predictions) != len(submission):
        print(f"Warning: Prediction length {len(xgb_predictions)} != submission length {len(submission)}")
        # Pad or truncate as needed
        if len(xgb_predictions) < len(submission):
            padding = np.zeros((len(submission) - len(xgb_predictions), len(target_cols)))
            xgb_predictions = np.vstack([xgb_predictions, padding])
        else:
            xgb_predictions = xgb_predictions[:len(submission)]
    
    # Fill submission
    for i, target in enumerate(target_cols):
        submission[target] = xgb_predictions[:, i]
    
    # ========================================================================
    # CRITICAL: Apply Tg transformation discovered by 2nd place winner
    # ========================================================================
    # Analysis of winning solutions revealed that the competition was determined
    # by a Tg (glass transition temperature) distribution shift in the test data.
    # The 2nd place winner (Private LB: 0.066) discovered that applying a simple
    # transformation to Tg predictions was worth 10-20x more than model complexity.
    #
    # Transformation: (9/5) * Tg + 45
    # This is similar to Celsius->Fahrenheit conversion, suggesting a units/scale
    # issue between train and test datasets for Tg specifically.
    #
    # Impact: A basic ExtraTreesRegressor with this transformation (0.077) performed
    # as well as complex BERT ensembles with 1.1M external data (0.075).
    #
    # Reference: 2nd place solution write-up on Kaggle competition discussion
    # ========================================================================
    
    print("\n" + "="*70)
    print("APPLYING TG TRANSFORMATION (2nd Place Discovery)")
    print("="*70)
    print(f"Original Tg range: [{submission['Tg'].min():.2f}, {submission['Tg'].max():.2f}]")
    print(f"Original Tg mean: {submission['Tg'].mean():.2f}")
    
    # Apply the transformation
    submission['Tg'] = (9/5) * submission['Tg'] + 45
    
    print(f"Transformed Tg range: [{submission['Tg'].min():.2f}, {submission['Tg'].max():.2f}]")
    print(f"Transformed Tg mean: {submission['Tg'].mean():.2f}")
    print("="*70 + "\n")
    
    # Sanity checks
    print("Submission validation:")
    print(f"Shape: {submission.shape}")
    print(f"Columns: {list(submission.columns)}")
    print(f"Any NaN: {submission.isnull().any().any()}")
    print(f"Any inf: {np.isinf(submission.select_dtypes(include=[np.number])).any().any()}")
    
    # Replace any remaining NaN/inf values
    submission = submission.fillna(0.0)
    numeric_cols = submission.select_dtypes(include=[np.number]).columns
    submission[numeric_cols] = submission[numeric_cols].replace([np.inf, -np.inf], 0.0)
    
    print("\nSubmission preview:")
    print(submission.head())
    
    print("\nSubmission statistics:")
    print(submission[target_cols].describe())
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    print("\nÃ¢Å“â€¦ Submission saved to submission.csv successfully!")
    print("   Includes Tg transformation for improved leaderboard performance.")
    
except Exception as e:
    print(f"Submission creation failed: {e}")
    # Create minimal fallback submission
    try:
        submission = sample_submission.copy()
        for target in target_cols:
            submission[target] = 0.0
        submission.to_csv('submission.csv', index=False)
        print("Created fallback submission with zeros")
    except Exception as e2:
        print(f"Even fallback submission failed: {e2}")
        raise

