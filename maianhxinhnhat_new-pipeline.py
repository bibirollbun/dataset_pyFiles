!pip install /kaggle/input/my-rdkit-whl/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


!pip install /kaggle/input/tqdm-joblib/tqdm_joblib-0.0.4-py3-none-any.whl


!cp -r /kaggle/input/autogluon-package/* /kaggle/working/
!pip install -f --no-index --find-links='/kaggle/input/autogluon-package' 'autogluon.tabular-1.3.1-py3-none-any.whl'

!cp -r /kaggle/input/scikit-package/* /kaggle/working/
!pip install -f --no-index --find-links='/kaggle/input/scikit-package' 'scikit_learn-1.5.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl' 


!pip install /kaggle/input/mordred-1-2-0-py3-none-any/networkx-2.8.8-py3-none-any.whl
!pip install /kaggle/input/mordred-1-2-0-py3-none-any/mordred-1.2.0-py3-none-any.whl


from autogluon.tabular import TabularDataset, TabularPredictor
from mordred import Calculator, descriptors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import polars as pl
import gc
import pickle
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

import lightgbm as lgb
from sklearn.model_selection import KFold

import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Descriptors3D import (
    Asphericity, Eccentricity, InertialShapeFactor,
    RadiusOfGyration, SpherocityIndex, NPR1, NPR2,
    PMI1, PMI2, PMI3
)
from rdkit.Chem import rdmolops

from joblib import Parallel, delayed

class CFG:
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    SEED = 42
    FOLDS = 5
    PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'
    PREPROCESSED_PATH = '/kaggle/working/preprocessed_data'

# === 1. LOAD DATA ===
train = pd.read_csv(CFG.PATH + 'train.csv')
test  = pd.read_csv(CFG.PATH + 'test.csv')

# === 2. FILTER INVALID SMILES ===
for df, name in [(train, 'train'), (test, 'test')]:
    mask = df['SMILES'].apply(lambda s: Chem.MolFromSmiles(s) is None)
    n = mask.sum()
    df.drop(df[mask].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Dropped {n} invalid SMILES rows from {name}")

# === 3. CANONICALIZE SMILES ===
def make_smile_canonical(smile):
    try:
        m = Chem.MolFromSmiles(smile)
        return Chem.MolToSmiles(m, canonical=True)
    except:
        return np.nan

for df in (train, test):
    df['SMILES'] = df['SMILES'].apply(make_smile_canonical)
    df.dropna(subset=['SMILES'], inplace=True)
    df.reset_index(drop=True, inplace=True)

# === 4. LOAD SUPPLEMENTAL DATA AND DEDUPE data_tc2 ===
data_tc   = pd.read_csv('/kaggle/input/tc-smiles/Tc_SMILES.csv').rename(columns={'TC_mean':'Tc'})

data_tc2  = pd.read_csv(f"{CFG.PATH}train_supplement/dataset1.csv").rename(columns={'TC_mean':'Tc'})
# loại bỏ SMILES duplicate trong data_tc2
counts    = data_tc2['SMILES'].value_counts()
dups      = counts[counts > 1].index
n_dup     = data_tc2['SMILES'].isin(dups).sum()
data_tc2  = data_tc2[~data_tc2['SMILES'].isin(dups)].reset_index(drop=True)
print(f"Dropped {n_dup} duplicated SMILES rows from data_tc2")

data_tg2  = pd.read_csv(
    '/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv',
    usecols=['SMILES','Tg (C)']
).rename(columns={'Tg (C)':'Tg'})

data_tg3  = pd.read_excel('/kaggle/input/smiles-extra-data/data_tg3.xlsx') \
               .rename(columns={'Tg [K]':'Tg'})
data_tg3['Tg'] -= 273.15

data_tg4  = pd.read_csv(f"{CFG.PATH}train_supplement/dataset3.csv")
data_FFV  = pd.read_csv(f"{CFG.PATH}train_supplement/dataset4.csv")

data_dnst = pd.read_excel('/kaggle/input/smiles-extra-data/data_dnst1.xlsx') \
               .rename(columns={'density(g/cm3)':'Density'})[['SMILES','Density']]
data_dnst['SMILES'] = data_dnst['SMILES'].apply(make_smile_canonical)
data_dnst = data_dnst[(data_dnst['SMILES'].notna()) &
                      (data_dnst['Density'].notna()) &
                      (data_dnst['Density']!='nylon')]
data_dnst['Density'] = data_dnst['Density'].astype(float) - 0.118

# === 5. ADD_EXTRA_DATA FUNCTION ===
def add_extra_data(df_train, df_extra, target):
    n_before = df_train[target].notna().sum()
    df_e = df_extra.copy()
    df_e['SMILES'] = df_e['SMILES'].apply(make_smile_canonical)
    df_e = df_e.groupby('SMILES', as_index=False)[target].mean()
    cross = set(df_e['SMILES']) & set(df_train['SMILES'])
    unique_extra = set(df_e['SMILES']) - set(df_train['SMILES'])
    # cập nhật giá trị chồng lắp
    for smi in df_train[df_train[target].notna()]['SMILES']:
        cross.discard(smi)
    for smi in cross:
        val = df_e.loc[df_e['SMILES']==smi, target].values[0]
        df_train.loc[df_train['SMILES']==smi, target] = val
    # append mẫu mới
    df_train = pd.concat([df_train, df_e[df_e['SMILES'].isin(unique_extra)]],
                         ignore_index=True)
    n_after = df_train[target].notna().sum()
    print(f'For target "{target}" added {n_after-n_before} samples (unique_extra={len(unique_extra)})')
    return df_train

# === 6. MERGE EXTRA DATA INTO TRAIN ===
train = add_extra_data(train, data_tc,   'Tc')
train = add_extra_data(train, data_tc2,  'Tc')
train = add_extra_data(train, data_tg2,  'Tg')
train = add_extra_data(train, data_tg3,  'Tg')
train = add_extra_data(train, data_dnst, 'Density')
train = add_extra_data(train, data_tg4,  'Tg')
train = add_extra_data(train, data_FFV,  'FFV')

# === 7. SETUP DESCRIPTORS LISTS ===
useless_cols = [
    'BCUT2D_MWHI','BCUT2D_MWLOW','BCUT2D_CHGHI','BCUT2D_CHGLO',
    'BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BCUT2D_MRHI','BCUT2D_MRLOW',
    'NumRadicalElectrons','SMR_VSA8','SlogP_VSA9','fr_barbitur',
    'fr_benzodiazepine','fr_dihydropyridine','fr_epoxide',
    'fr_isothiocyan','fr_lactam','fr_nitroso','fr_prisulfonamd',
    'fr_thiocyan','MaxEStateIndex','HeavyAtomMolWt','ExactMolWt',
    'NumValenceElectrons','Chi0','Chi0n','Chi0v','Chi1','Chi1n',
    'Chi1v','Chi2n','Kappa1','LabuteASA','HeavyAtomCount','MolMR',
    'Chi3n','BertzCT','Chi2v','Chi4n','HallKierAlpha','Chi3v',
    'Chi4v','MinAbsPartialCharge','MinPartialCharge','MaxAbsPartialCharge',
    'FpDensityMorgan2','FpDensityMorgan3','Phi','Kappa3','fr_nitrile',
    'SlogP_VSA6','NumAromaticCarbocycles','NumAromaticRings','fr_benzene',
    'VSA_EState6','NOCount','fr_C_O','fr_C_O_noCOO','NumHDonors','fr_amide',
    'fr_Nhpyrrole','fr_phenol','fr_phenol_noOrthoHbond','fr_COO2','fr_halogen',
    'fr_diazo','fr_nitro_arom','fr_phos_ester'
]
desc_names = [n for n,_ in Descriptors.descList if n not in useless_cols]

# === 8. 2D DESCRIPTORS ===
def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan]*len(desc_names)
    return [func(mol) for name,func in Descriptors.descList if name not in useless_cols]

# === 9. GRAPH FEATURES ===
def compute_graph_features(smiles, graph_feats):
    mol = Chem.MolFromSmiles(smiles)
    adj = rdmolops.GetAdjacencyMatrix(mol)
    G = nx.from_numpy_array(adj)
    if nx.is_connected(G):
        dia = nx.diameter(G)
        avg_sp = nx.average_shortest_path_length(G)
    else:
        dia, avg_sp = 0, 0
    cyc = len(nx.cycle_basis(G))
    graph_feats['graph_diameter'].append(dia)
    graph_feats['avg_shortest_path'].append(avg_sp)
    graph_feats['num_cycles'].append(cyc)

# === 10. 3D DESCRIPTORS ===
import os, sys, contextlib

import os, sys
from contextlib import contextmanager

@contextmanager
def suppress_stderr_fd():
    """
    Tạm thời redirect toàn bộ stderr (C level) vào /dev/null.
    """
    stderr_fd = sys.stderr.fileno()
    # lưu lại bản sao của stderr
    saved_stderr_fd = os.dup(stderr_fd)
    # mở /dev/null
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    try:
        # redirect stderr → /dev/null
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        # khôi phục stderr
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_stderr_fd)


def compute_3d_descriptors(smiles, max_attempts=3):
    names3d = [
        "PMI1","PMI2","PMI3",
        "NPR1","NPR2",
        "RadiusOfGyration","InertialShapeFactor",
        "Eccentricity","Asphericity",
        "SpherocityIndex","PBF"
    ]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {n: np.nan for n in names3d}

    m3 = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    # suppress C-level stderr trong embed + optimize
    try:
        with suppress_stderr_fd():
            for _ in range(max_attempts):
                if AllChem.EmbedMolecule(m3, params) == 0:
                    break
            else:
                return {n: np.nan for n in names3d}
            AllChem.UFFOptimizeMolecule(m3)
    except Exception:
        return {n: np.nan for n in names3d}

    # tính descriptor
    try:
        return {
            "PMI1": PMI1(m3), "PMI2": PMI2(m3), "PMI3": PMI3(m3),
            "NPR1": NPR1(m3), "NPR2": NPR2(m3),
            "RadiusOfGyration": RadiusOfGyration(m3),
            "InertialShapeFactor": InertialShapeFactor(m3),
            "Eccentricity": Eccentricity(m3), "Asphericity": Asphericity(m3),
            "SpherocityIndex": SpherocityIndex(m3),
            "PBF": rdMolDescriptors.CalcPBF(m3)
        }
    except Exception:
        return {n: np.nan for n in names3d}
        
# === 11. MORGAN FINGERPRINT ===
def smiles_to_morgan_fp(smiles, radius=3, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0]*n_bits
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits))

from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

# === 12. MORDRED DESCRIPTORS ===
# Initialize mordred calculator (2D descriptors only)
mordred_calc = Calculator(descriptors, ignore_3D=True)

def compute_mordred_descriptors(smiles):
    """
    Compute Mordred 2D descriptors for a SMILES string
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        # Calculate all 2D descriptors
        result = mordred_calc(mol)
        
        # Convert to dictionary, handling missing values and errors
        result_dict = {}
        for desc, value in result.items():
            desc_name = f"mordred_{str(desc)}"
            try:
                # Convert to float, handle missing/invalid values
                if value is None or str(value).lower() in ['nan', 'inf', '-inf', 'missing']:
                    result_dict[desc_name] = np.nan
                else:
                    result_dict[desc_name] = float(value)
            except (ValueError, TypeError):
                result_dict[desc_name] = np.nan
        
        return result_dict
    
    except Exception as e:
        # Return empty dict if calculation fails
        return {}

# === 13. COMPOSITE FEATURIZER ===
def get_graph_feats(smiles):
    tmp = {'graph_diameter': [], 'avg_shortest_path': [], 'num_cycles': []}
    compute_graph_features(smiles, tmp)
    return {k: tmp[k][0] for k in tmp}

def featurize_base(smiles):
    return {
        **dict(zip(desc_names, compute_all_descriptors(smiles))),
        **get_graph_feats(smiles),
        **{f'fp_{i}': bit for i, bit in enumerate(smiles_to_morgan_fp(smiles))}
    }

# === 14. PARALLEL FEATURIZATION ===

# 14.1 Base features (2D + graph + MorganFP)
print("Computing base features...")
res_base_train = Parallel(n_jobs=-1, backend='loky')(
    delayed(featurize_base)(s) for s in tqdm(train['SMILES'], desc="Base features (train)")
)
df_base_train = pd.DataFrame(res_base_train)
train = pd.concat([train.reset_index(drop=True), df_base_train], axis=1)
del res_base_train, df_base_train
gc.collect()

res_base_test = Parallel(n_jobs=-1, backend='loky')(
    delayed(featurize_base)(s) for s in tqdm(test['SMILES'], desc="Base features (test)")
)
df_base_test = pd.DataFrame(res_base_test)
test = pd.concat([test.reset_index(drop=True), df_base_test], axis=1)
del res_base_test, df_base_test
gc.collect()

# 14.2 3D descriptors with progress bar
print("Computing 3D descriptors...")
with tqdm_joblib(tqdm(desc="3D descriptors (train)", total=len(train))):
    res_3d_train = Parallel(n_jobs=-1, backend='loky')(
        delayed(compute_3d_descriptors)(s) for s in train['SMILES']
    )
df_3d_train = pd.DataFrame(res_3d_train)
train = pd.concat([train, df_3d_train], axis=1)
del res_3d_train, df_3d_train
gc.collect()

with tqdm_joblib(tqdm(desc="3D descriptors (test)", total=len(test))):
    res_3d_test = Parallel(n_jobs=-1, backend='loky')(
        delayed(compute_3d_descriptors)(s) for s in test['SMILES']
    )
df_3d_test = pd.DataFrame(res_3d_test)
test = pd.concat([test, df_3d_test], axis=1)
del res_3d_test, df_3d_test
gc.collect()

# 14.3 Mordred descriptors
print("Computing Mordred descriptors...")
with tqdm_joblib(tqdm(desc="Mordred descriptors (train)", total=len(train))):
    res_mordred_train = Parallel(n_jobs=-1, backend='loky')(
        delayed(compute_mordred_descriptors)(s) for s in train['SMILES']
    )
df_mordred_train = pd.DataFrame(res_mordred_train)
train = pd.concat([train, df_mordred_train], axis=1)
del res_mordred_train, df_mordred_train
gc.collect()

with tqdm_joblib(tqdm(desc="Mordred descriptors (test)", total=len(test))):
    res_mordred_test = Parallel(n_jobs=-1, backend='loky')(
        delayed(compute_mordred_descriptors)(s) for s in test['SMILES']
    )
df_mordred_test = pd.DataFrame(res_mordred_test)
test = pd.concat([test, df_mordred_test], axis=1)
del res_mordred_test, df_mordred_test
gc.collect()

print("All feature computation completed!")

# === Kiểm tra số phân tử bị fail embedding 3D ===
# Danh sách cột 3D descriptor bạn đã thêm vào train/test
cols3d = [
    "PMI1","PMI2","PMI3",
    "NPR1","NPR2",
    "RadiusOfGyration","InertialShapeFactor",
    "Eccentricity","Asphericity",
    "SpherocityIndex","PBF"
]

# Với train
fails_train = train[cols3d].isna().all(axis=1).sum()
total_train = len(train)
print(f"{fails_train} molecules failed 3D embedding in train / total {total_train}")

# Với test
fails_test = test[cols3d].isna().all(axis=1).sum()
total_test = len(test)
print(f"{fails_test} molecules failed 3D embedding in test / total {total_test}")

print(f"Total features: {len([c for c in train.columns if c not in ['id', 'SMILES'] + CFG.TARGETS])}")
print(f"Mordred features: {len([c for c in train.columns if c.startswith('mordred_')])}")

# === 15. POST-PROCESSING ===
# Remove constant columns
const_cols = [c for c in train.columns if train[c].nunique() <= 1]
if const_cols:
    train.drop(columns=const_cols, inplace=True)
    test.drop(columns=const_cols, inplace=True)
    print(f"Removed {len(const_cols)} constant columns")

# Log transform Ipc if present
if 'Ipc' in train.columns:
    train['Ipc'] = np.log10(train['Ipc'].replace(0, np.nan))
    test['Ipc'] = np.log10(test['Ipc'].replace(0, np.nan))

# Handle inf values and fill NaN
feature_cols = [c for c in train.columns if c not in ['id', 'SMILES'] + CFG.TARGETS]
for col in feature_cols:
    if col in train.columns and col in test.columns:
        # Replace inf values with NaN
        train[col].replace([-np.inf, np.inf], np.nan, inplace=True)
        test[col].replace([-np.inf, np.inf], np.nan, inplace=True)
        
        # Fill NaN with mean from train
        if train[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            mean_val = train[col].mean()
            train[col].fillna(mean_val, inplace=True)
            test[col].fillna(mean_val, inplace=True)

print("Pre-processing completed!")
print(f"Final shape - Train: {train.shape}, Test: {test.shape}")



# === 16. SAVE FEATURES ===
def save_preprocessed_data(train, test):
    """
    Save preprocessed data for each target in separate folders
    """
    # Create main preprocessed directory
    os.makedirs(CFG.PREPROCESSED_PATH, exist_ok=True)
    
    for target in CFG.TARGETS:
        # Create target-specific directory
        target_dir = os.path.join(CFG.PREPROCESSED_PATH, f'{target}_data')
        os.makedirs(target_dir, exist_ok=True)
        
        # Prepare data for this target (remove other targets)
        other_targets = [t for t in CFG.TARGETS if t != target]
        
        # Drop other target columns (ignore if they don't exist)
        train_target = train.drop(columns=other_targets, errors='ignore')
        test_target = test.drop(columns=other_targets, errors='ignore')
        
        # Save train and test data
        train_target.to_pickle(os.path.join(target_dir, 'train.pkl'))
        test_target.to_pickle(os.path.join(target_dir, 'test.pkl'))
        
        # Save features list
        features = [c for c in train_target.columns if c not in [target, 'id', 'SMILES']]
        with open(os.path.join(target_dir, 'features.pkl'), 'wb') as f:
            pickle.dump(features, f)
        
        print(f"Saved preprocessed data for {target} to {target_dir}")

def load_preprocessed_data(target):
    """
    Load preprocessed data for a specific target
    """
    target_dir = os.path.join(CFG.PREPROCESSED_PATH, f'{target}_data')
    
    # Load train and test data
    train = pd.read_pickle(os.path.join(target_dir, 'train.pkl'))
    test = pd.read_pickle(os.path.join(target_dir, 'test.pkl'))
    
    # Load features
    with open(os.path.join(target_dir, 'features.pkl'), 'rb') as f:
        features = pickle.load(f)
    
    return train, test, features

def train_single_model(target, time_limit=6000):
    """
    Train model for a single target using preprocessed data
    """
    print(f"Training model for {target}...")
    import torch
    NUM_GPUS = 1 if torch.cuda.is_available() else 0
    print(f"GPUs: {NUM_GPUS}")
    
    # Load preprocessed data
    train, test, features = load_preprocessed_data(target)
    
    # Prepare training data (remove rows with null target)
    df_fit = train[train[target].notnull()].reset_index(drop=True)
    
    # Train model
    model_path = f'model_{target}'
    predictor = TabularPredictor(
        label=target,
        path=model_path,
        eval_metric='mean_absolute_error'
    ).fit(
        train_data=df_fit,
        time_limit=time_limit,
        feature_prune_kwargs={'force_prune': True},
        hyperparameters={
            'GBM': {'device': 'gpu' if NUM_GPUS else 'cpu', 'seed': CFG.SEED},
            'XGB': {'tree_method': 'hist', 'device': 'cuda' if NUM_GPUS else 'cpu', 'seed': CFG.SEED},
            'CAT': {'random_seed': CFG.SEED}, 
            'XT': {'random_state': CFG.SEED},
            'NN_TORCH': {}
        },
        presets=['best_quality'],
        auto_stack=True
    )
    
    print(f"Model for {target} trained successfully!")
    return predictor

def train_all_models(time_limit=4500):
    """
    Train models for all targets using preprocessed data
    """
    predictors = {}
    
    for target in CFG.TARGETS:
        predictors[target] = train_single_model(target, time_limit)
    
    return predictors

def generate_predictions_and_submission():
    """
    Generate predictions and create submission file
    """
    print("Generating predictions...")
    
    # Load predictors and generate predictions
    predictions = {}
    test_data = None
    
    for target in CFG.TARGETS:
        # Load test data for this target
        _, test, _ = load_preprocessed_data(target)
        if test_data is None:
            test_data = test[['id', 'SMILES']].copy()
        
        # Load model and predict
        predictor = TabularPredictor.load(f'model_{target}')
        predictions[target] = predictor.predict(test)
    
    # Combine all predictions
    for target, pred in predictions.items():
        test_data[target] = pred
    
    # Apply leak fix
    print("Applying leak fix...")
    leak_df_list = []
    for target in CFG.TARGETS:
        train, _, _ = load_preprocessed_data(target)
        target_leak = train[['SMILES', target]].dropna()
        leak_df_list.append(target_leak)
    
    # Merge all leak data
    leak_df = leak_df_list[0]
    for i in range(1, len(leak_df_list)):
        leak_df = leak_df.merge(leak_df_list[i], on='SMILES', how='outer')
    
    leak_df = leak_df.drop_duplicates().set_index('SMILES')
    
    # Apply leak fix
    for target in CFG.TARGETS:
        test_data[target] = test_data['SMILES'].map(leak_df[target]).fillna(test_data[target])
    
    # Save submission
    submission = test_data[['id'] + CFG.TARGETS]
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")
    
    return submission

# === MAIN EXECUTION ===
# Assuming train and test are already prepared from previous steps
save_preprocessed_data(train, test)

# === TRAIN & SAVE MODELS ===
predictors = train_all_models(time_limit=6000)

# === PREDICT & SUBMISSION ===
submission = generate_predictions_and_submission()




