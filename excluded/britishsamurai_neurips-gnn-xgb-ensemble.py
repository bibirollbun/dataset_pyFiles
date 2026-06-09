!cp -r /kaggle/input/autogluon-package/* /kaggle/working/
!pip install -f --quiet --no-index --find-links='/kaggle/input/autogluon-package' 'autogluon.tabular-1.3.1-py3-none-any.whl'


!cp -r /kaggle/input/scikit-package/* /kaggle/working/
!pip install -f --quiet --no-index --find-links='/kaggle/input/scikit-package' 'scikit_learn-1.5.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl' 



from autogluon.tabular import TabularDataset, TabularPredictor


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


!pip install mordred --no-index --find-links=file:///kaggle/input/mordred-1-2-0-py3-none-any/


!rm -rf /kaggle/working/*


BASE_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'


output_dfs = []


import glob
import os
import time
import random
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import networkx as nx

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdmolops, Lipinski, Crippen
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator

from mordred import Calculator, descriptors as mordred_descriptors

import shap

from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from torchinfo import summary
except ImportError:
    summary = None

# Data paths
RDKIT_AVAILABLE = True
TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Create the NeurIPS directory if it does not exist
os.makedirs("NeurIPS", exist_ok=True)

class Config:
    useAllDataForTraining = False
    use_standard_scaler = True
    use_least_important_features_all_methods = True
    use_variance_threshold = False
    enable_param_tuning = False
    debug = False

    # >>> é‡�è¦�: ç‰¹å¾´é‡�æ§‹æˆ�ã‚’LBæ•´å�ˆç”¨ã�«å¤‰æ›´ <<<
    use_descriptors = True        # RDKitå …ç‰¢ã‚µãƒ–ã‚»ãƒƒãƒˆã�®ã�¿
    use_mordred = False           # Mordredã�¯OFFï¼ˆãƒ�ã‚¤ã‚º/NaNå¤šï¼‰
    use_maccs_fp = False
    use_morgan_fp = True          # MorganæŒ‡ç´‹ON
    use_atom_pair_fp = False
    use_torsion_fp = False
    use_chemberta = False
    chemberta_pooling = 'max'

    search_nn = False
    use_stacking = False
    model_name = 'xgb'            # XGBã�Œä¸»åŠ›

    feature_importance_method = 'permutation_importance'
    use_cross_validation = True
    use_pca = True                # æŒ‡ç´‹+è¨˜è¿°å­�ã‚’PCAåœ§ç¸®
    pca_variance = 0.999
    use_external_data = True
    use_augmentation = False
    add_gaussian = False
    random_state = 42

    # ãƒ©ãƒ™ãƒ«åˆ¥ã�®ç›¸é–¢åˆˆã‚Šé–¾å€¤ï¼ˆå¯†â†’å¼·ã‚�ï¼‰
    correlation_thresholds = {
        "FFV": 0.92,
        "Density": 0.92,
        "Tg": 0.95,
        "Tc": 0.95,
        "Rg": 0.95
    }

config = Config()
if config.debug or config.search_nn:
    config.use_cross_validation = False

# --- XGB Hyperparameter Tuning DB Utilities (unchanged API) ---
import sqlite3
import hashlib
import json

def init_chemberta():
    model_name = "/kaggle/input/c/transformers/default/1/ChemBERTa-77M-MLM"
    chemberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
    chemberta_model = AutoModel.from_pretrained(model_name)
    chemberta_model.eval()
    return chemberta_tokenizer, chemberta_model

def get_chemberta_embedding(smiles, embedding_dim=384):
    if smiles is None or not isinstance(smiles, str) or len(smiles) == 0:
        return np.zeros(embedding_dim)
    try:
        pooling = getattr(config, 'chemberta_pooling', 'mean')
        chemberta_tokenizer, chemberta_model = init_chemberta()
        inputs = chemberta_tokenizer([smiles], padding=True, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = chemberta_model(**inputs)
            if pooling == 'pooler' and hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                emb = outputs.pooler_output.squeeze(0)
            elif pooling == 'cls' and hasattr(outputs, 'last_hidden_state'):
                emb = outputs.last_hidden_state[:, 0, :].squeeze(0)
            elif pooling == 'max' and hasattr(outputs, 'last_hidden_state'):
                emb = outputs.last_hidden_state.max(dim=1).values.squeeze(0)
            elif pooling == 'mean' and hasattr(outputs, 'last_hidden_state'):
                emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            else:
                raise ValueError("Cannot extract embedding from model output")
            emb_np = emb.cpu().numpy()
            if emb_np.shape[0] < embedding_dim:
                emb_np = np.pad(emb_np, (0, embedding_dim - emb_np.shape[0]))
            elif emb_np.shape[0] > embedding_dim:
                emb_np = emb_np[:embedding_dim]
            return emb_np
    except Exception as e:
        print(f"ChemBERTa embedding failed for SMILES '{smiles}': {e}")
        return np.zeros(embedding_dim)

def init_xgb_tuning_db(db_path="xgb_tuning.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS xgb_tuning
                 (param_hash TEXT PRIMARY KEY, params TEXT, score REAL)''')
    c.execute('SELECT params, score FROM xgb_tuning')
    results = c.fetchall()
    conn.close()
    return [(json.loads(params), score) for params, score in results]

def get_param_hash(params):
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode('utf-8')).hexdigest()

def check_db_for_params(db_path, param_hash):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT score FROM xgb_tuning WHERE param_hash=?', (param_hash,))
    result = c.fetchone()
    conn.close()
    return result is not None

def save_result_to_db(db_path, param_hash, params, score):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO xgb_tuning (param_hash, params, score)
                 VALUES (?, ?, ?)''', (param_hash, json.dumps(params, sort_keys=True), score))
    conn.commit()
    conn.close()

from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid

def xgb_grid_search_with_db(X, y, param_grid, db_path="xgb_tuning.db"):
    tried = 0
    best_score = None
    best_params = None
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    for params in ParameterGrid(param_grid):
        param_hash = get_param_hash(params)
        if check_db_for_params(db_path, param_hash):
            print(f"Skipping already tried params: {params}")
            continue
        # fitå�´ã�§early stoppingï¼ˆMAEï¼‰
        p = params.copy()
        early_rounds = p.pop("early_stopping_rounds", 200)
        model = XGBRegressor(**p)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=early_rounds, verbose=False)
        y_pred = model.predict(X_val)
        score = mean_absolute_error(y_val, y_pred)
        print(f"Result: MAE={score:.6f} for {json.dumps(params, sort_keys=True)}")
        if (best_score is None) or (score < best_score):
            best_score = score
            best_params = params.copy()
            print(f"New best MAE: {best_score:.6f} with {json.dumps(best_params, sort_keys=True)}")
        save_result_to_db(db_path, param_hash, params, score)
        tried += 1
    print(f"Tried {tried} new parameter sets.")
    if best_score is not None:
        print(f"Best score overall: {best_score:.6f} with {json.dumps(best_params, sort_keys=True)}")

from sklearn.linear_model import RidgeCV, ElasticNetCV

def drop_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop

def get_canonical_smiles(smiles):
    if not RDKIT_AVAILABLE:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return smiles

print("Loading competition data...")
train = pd.read_csv(BASE_PATH + 'train.csv')
test = pd.read_csv(BASE_PATH + 'test.csv')

if config.debug:
    print("   Debug mode: sampling 1000 training examples")
    train = train.sample(n=1000, random_state=42).reset_index(drop=True)

print(f"Training data shape: {train.shape}, Test data shape: {test.shape}")

def clean_and_validate_smiles(smiles):
    if not isinstance(smiles, str) or len(smiles) == 0:
        return None
    bad_patterns = ['[R]','[R1]','[R2]','[R3]','[R4]','[R5]',"[R']",'[R\"]','R1','R2','R3','R4','R5','([R])','([R1])','([R2])']
    for pattern in bad_patterns:
        if pattern in smiles:
            return None
    if '][' in smiles and any(x in smiles for x in ['[R','R]']):
        return None
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
            else:
                return None
        except:
            return None
    return smiles

print("Cleaning and validating SMILES...")
train['SMILES'] = train['SMILES'].apply(clean_and_validate_smiles)
test['SMILES'] = test['SMILES'].apply(clean_and_validate_smiles)

invalid_train = train['SMILES'].isnull().sum()
invalid_test = test['SMILES'].isnull().sum()
print(f"   Removed {invalid_train} invalid SMILES from training data")
print(f"   Removed {invalid_test} invalid SMILES from test data")

train = train[train['SMILES'].notnull()].reset_index(drop=True)
test = test[test['SMILES'].notnull()].reset_index(drop=True)

print(f"   Final training samples: {len(train)}")
print(f"   Final test samples: {len(test)}")

def add_extra_data_clean(df_train, df_extra, target):
    n_samples_before = len(df_train[df_train[target].notnull()])
    print(f"      Processing {len(df_extra)} {target} samples...")
    df_extra['SMILES'] = df_extra['SMILES'].apply(clean_and_validate_smiles)
    before_filter = len(df_extra)
    df_extra = df_extra[df_extra['SMILES'].notnull()]
    df_extra = df_extra.dropna(subset=[target])
    after_filter = len(df_extra)
    print(f"      Kept {after_filter}/{before_filter} valid samples")
    if len(df_extra) == 0:
        print(f"      No valid data remaining for {target}")
        return df_train
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()
    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])
    filled_count = 0
    for smile in df_train[df_train[target].isnull()]['SMILES'].tolist():
        if smile in cross_smiles:
            df_train.loc[df_train['SMILES']==smile, target] = df_extra[df_extra['SMILES']==smile][target].values[0]
            filled_count += 1
    extra_to_add = df_extra[df_extra['SMILES'].isin(unique_smiles_extra)].copy()
    if len(extra_to_add) > 0:
        for col in TARGETS:
            if col not in extra_to_add.columns:
                extra_to_add[col] = np.nan
        extra_to_add = extra_to_add[['SMILES'] + TARGETS]
        df_train = pd.concat([df_train, extra_to_add], axis=0, ignore_index=True)
    n_samples_after = len(df_train[df_train[target].notnull()])
    print(f'      {target}: +{n_samples_after-n_samples_before} samples, +{len(unique_smiles_extra)} unique SMILES')
    print(f"      Filled {filled_count} missing entries in train for {target}")
    print(f"      Added {len(extra_to_add)} new entries for {target}")
    return df_train

print("\nğŸ“‚ Loading external datasets...")
external_datasets = []

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

safe_load_dataset('/kaggle/input/tc-smiles/Tc_SMILES.csv','Tc',lambda df: df.rename(columns={'TC_mean': 'Tc'}),'Tc data')
safe_load_dataset('/kaggle/input/tg-smiles-pid-polymer-class/TgSS_enriched_cleaned.csv','Tg',lambda df: df[['SMILES','Tg']] if 'Tg' in df.columns else df,'TgSS enriched data')
safe_load_dataset('/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv','Tg',lambda df: df[['SMILES','Tg (C)']].rename(columns={'Tg (C)':'Tg'}),'JCIM Tg data')
safe_load_dataset('/kaggle/input/smiles-extra-data/data_tg3.xlsx','Tg',lambda df: df.rename(columns={'Tg [K]':'Tg'}).assign(Tg=lambda x: x['Tg']-273.15),'Xlsx Tg data')
safe_load_dataset('/kaggle/input/smiles-extra-data/data_dnst1.xlsx','Density',
    lambda df: df.rename(columns={'density(g/cm3)':'Density'})[['SMILES','Density']].query('SMILES.notnull() and Density.notnull() and Density != "nylon"').assign(Density=lambda x: x['Density'].astype(float)-0.118),
    'Density data')
safe_load_dataset(BASE_PATH + 'train_supplement/dataset4.csv','FFV',lambda df: df[['SMILES','FFV']] if 'FFV' in df.columns else df,'dataset 4')

print("\nğŸ”„ Integrating external data...")
train_extended = train[['SMILES'] + TARGETS].copy()
if getattr(config, "use_external_data", True) and not config.debug:
    for target, dataset in external_datasets:
        print(f"   Processing {target} data...")
        train_extended = add_extra_data_clean(train_extended, dataset, target)
print(f"\nğŸ“Š Final training data:")
print(f"   Original samples: {len(train)}")
print(f"   Extended samples: {len(train_extended)}")
for target in TARGETS:
    count = train_extended[target].notna().sum()
    original_count = train[target].notna().sum() if target in train.columns else 0
    print(f"   {target}: {count:,} samples (+{count - original_count})")
print(f"\nâœ… Data integration complete with clean SMILES!")

def separate_subtables(train_df):
    labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    return {label: train_df[train_df[label].notna()][['SMILES', label]].reset_index(drop=True) for label in labels}

def augment_smiles_dataset(smiles_list, labels, num_augments=3):
    augmented_smiles, augmented_labels = [], []
    for smiles, label in zip(smiles_list, labels):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        augmented_smiles.append(smiles); augmented_labels.append(label)
        for _ in range(num_augments):
            rand_smiles = Chem.MolToSmiles(mol, doRandom=True)
            augmented_smiles.append(rand_smiles); augmented_labels.append(label)
    return augmented_smiles, np.array(augmented_labels)

# --- Mordredã�¯ä½¿ã‚�ã�ªã�„ï¼ˆOFFï¼‰ ---
mordred_calc = Calculator(mordred_descriptors, ignore_3D=True)

# RDKitå …ç‰¢ã‚µãƒ–ã‚»ãƒƒãƒˆ
RD_SUBSET = {
    'MolWt': Descriptors.MolWt,
    'TPSA': Descriptors.TPSA,
    'MolMR': Crippen.MolMR,
    'MolLogP': Crippen.MolLogP,
    'NumHAcceptors': Lipinski.NumHAcceptors,
    'NumHDonors': Lipinski.NumHDonors,
    'RingCount': Descriptors.RingCount,
    'FractionCSP3': Descriptors.FractionCSP3,
    'NumRotatableBonds': Descriptors.NumRotatableBonds,
    'NumHeteroatoms': Descriptors.NumHeteroatoms
}
HALOGENS = [9, 17, 35, 53]  # F, Cl, Br, I

def build_halogen_counts(mol):
    counts = {f'Num{sym}': 0 for sym in ['F','Cl','Br','I']}
    for atom in mol.GetAtoms():
        Z = atom.GetAtomicNum()
        if Z == 9: counts['NumF']  += 1
        if Z == 17: counts['NumCl'] += 1
        if Z == 35: counts['NumBr'] += 1
        if Z == 53: counts['NumI']  += 1
    return counts

def build_mordred_descriptors(smiles_list):
    # ä¸�ä½¿ç”¨
    return pd.DataFrame()

def smiles_to_combined_fingerprints_with_descriptors(smiles_list):
    # --- æ”¹è‰¯: Morgan 2048bit å›ºå®šã€�å¿…è¦�ã�«å¿œã�˜PCAã�§å¾Œæ®µåœ§ç¸® ---
    radius = 2
    n_bits = 2048
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits) if getattr(Config, "use_morgan_fp", True) else None

    fp_len = (n_bits if getattr(Config, 'use_morgan_fp', False) else 0)
    if getattr(Config, 'use_chemberta', False):
        fp_len += 384

    fingerprints, descriptors, valid_smiles, invalid_indices = [], [], [], []
    use_any_fp = getattr(Config, "use_morgan_fp", False) or getattr(Config, "use_chemberta", False)

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # FP
            if use_any_fp:
                fps = []
                if getattr(Config, "use_morgan_fp", True) and generator is not None:
                    fps.append(np.array(generator.GetFingerprint(mol), dtype=np.int8))
                if getattr(Config, "use_chemberta", False):
                    fps.append(get_chemberta_embedding(smiles))
                combined_fp = np.concatenate(fps) if len(fps) else np.zeros(fp_len, dtype=np.float32)
                fingerprints.append(combined_fp)
            # RDKitå …ç‰¢ã‚µãƒ–ã‚»ãƒƒãƒˆ
            if getattr(Config, 'use_descriptors', True):
                d = {}
                for name, func in RD_SUBSET.items():
                    try:
                        d[name] = float(func(mol))
                    except Exception:
                        d[name] = np.nan
                d.update(build_halogen_counts(mol))
                descriptors.append(d)
            else:
                descriptors.append(None)
            valid_smiles.append(smiles)
        else:
            if use_any_fp:
                fingerprints.append(np.zeros(fp_len, dtype=np.float32))
            descriptors.append(None)
            valid_smiles.append(None)
            invalid_indices.append(i)

    fingerprints_df = pd.DataFrame(fingerprints, columns=[f'FP_{i}' for i in range(fp_len)]) if use_any_fp else pd.DataFrame()
    descriptors_df = pd.DataFrame([d for d in descriptors if d is not None]) if any(d is not None for d in descriptors) else pd.DataFrame()
    if not descriptors_df.empty:
        descriptors_df = descriptors_df.loc[:, ~descriptors_df.columns.duplicated()]
    return fingerprints_df, descriptors_df, valid_smiles, invalid_indices

required_descriptors = {'graph_diameter','num_cycles','avg_shortest_path','MolWt','LogP','TPSA','RotatableBonds','NumAtoms'}

def combine_train_val(X_train, X_val, y_train, y_val):
    X_train = pd.DataFrame(X_train) if isinstance(X_train, np.ndarray) else X_train
    X_val   = pd.DataFrame(X_val)   if isinstance(X_val, np.ndarray)   else X_val
    y_train = pd.Series(y_train)    if isinstance(y_train, np.ndarray)  else y_train
    y_val   = pd.Series(y_val)      if isinstance(y_val, np.ndarray)    else y_val
    X_all = pd.concat([X_train, X_val], axis=0)
    y_all = pd.concat([y_train, y_val], axis=0)
    return X_all, y_all

def apply_pca(X_train, X_test=None, verbose=True):
    pca = PCA(n_components=config.pca_variance, svd_solver='full', random_state=getattr(config,'random_state',42))
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test) if X_test is not None else None
    if verbose:
        print(f"[PCA] {X_train.shape} -> {X_train_pca.shape} ({100*pca.explained_variance_ratio_.sum():.3f}% var)")
    return X_train_pca, X_test_pca, pca

def augment_dataset(X, y, n_samples=1000, n_components=5, random_state=None):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    X.columns = X.columns.astype(str)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    df = X.copy(); df['Target'] = y.values
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(df)
    synthetic_data, _ = gmm.sample(n_samples)
    synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)
    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
    return augmented_df.drop(columns='Target'), augmented_df['Target']

def display_outlier_summary(y, X=None, name="target", z_thresh=3, iqr_factor=1.5, iso_contamination=0.01, lof_contamination=0.01):
    print(f"\nOutlier summary for: {name}")
    y = np.asarray(y); n = len(y)
    z_scores = (y - np.mean(y)) / (np.std(y) + 1e-9)
    z_outliers = np.abs(z_scores) > z_thresh
    print(f"Z-score > {z_thresh}: {np.sum(z_outliers)} / {n} ({100*np.mean(z_outliers):.2f}%)")
    Q1 = np.percentile(y, 25); Q3 = np.percentile(y, 75); IQR = Q3 - Q1
    lower = Q1 - iqr_factor*IQR; upper = Q3 + iqr_factor*IQR
    iqr_outliers = (y < lower) | (y > upper)
    print(f"IQR (factor {iqr_factor}): {np.sum(iqr_outliers)} / {n} ({100*np.mean(iqr_outliers):.2f}%)")
    if X is not None:
        try:
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(contamination=iso_contamination, random_state=42)
            iso_out = iso.fit_predict(X); iso_outliers = iso_out == -1
            print(f"Isolation Forest (cont={iso_contamination}): {np.sum(iso_outliers)} / {len(iso_outliers)} ({100*np.mean(iso_outliers):.2f}%)")
        except Exception as e:
            print(f"Isolation Forest failed: {e}")
        try:
            from sklearn.neighbors import LocalOutlierFactor
            lof = LocalOutlierFactor(n_neighbors=20, contamination=lof_contamination)
            lof_out = lof.fit_predict(X); lof_outliers = lof_out == -1
            print(f"LOF (cont={lof_contamination}): {np.sum(lof_outliers)} / {len(lof_outliers)} ({100*np.mean(lof_outliers):.2f}%)")
        except Exception as e:
            print(f"LOF failed: {e}")
    else:
        print("Isolation Forest/LOF skipped (X not provided)")

train_df = train_extended
test_df  = test
subtables = separate_subtables(train_df)
test_smiles = test_df['SMILES'].tolist()
test_ids    = test_df['id'].values
labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

def save_importance_to_excel(importance_df, label, log_path):
    import os
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.exists(log_path):
        with pd.ExcelWriter(log_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            importance_df.to_excel(writer, sheet_name=label, index=False)
    else:
        with pd.ExcelWriter(log_path, engine='openpyxl') as writer:
            importance_df.to_excel(writer, sheet_name=label, index=False)

def get_least_important_features_all_methods(X, y, label, model_name=None):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=config.random_state)
    model_type = (model_name or getattr(config, 'model_name', 'xgb'))
    if model_type == 'xgb':
        model = XGBRegressor(random_state=config.random_state, n_jobs=-1, verbosity=0,
                             eval_metric="mae", objective="reg:absoluteerror",
                             tree_method="hist")
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False, early_stopping_rounds=200)
    elif model_type == 'catboost':
        model = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=6, loss_function='MAE',
                                  eval_metric='MAE', random_seed=config.random_state, verbose=False)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=200, use_best_model=True)
    elif model_type == 'lgbm':
        model = LGBMRegressor(n_estimators=5000, learning_rate=0.02, max_depth=6, reg_lambda=1.0,
                              objective='mae', random_state=config.random_state, verbose=-1)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='mae',
                  callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])
    else:
        model = XGBRegressor(random_state=config.random_state, n_jobs=-1, verbosity=0, eval_metric="mae",
                             objective="reg:absoluteerror", tree_method="hist")
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False, early_stopping_rounds=200)
    feature_names = X_train.columns
    fi_mask = model.feature_importances_ <= 0
    fi_features = set(feature_names[fi_mask])
    fi_importance_df = pd.DataFrame({'feature': feature_names,'importance_mean': model.feature_importances_,'importance_std': [0]*len(feature_names)}).sort_values('importance_mean', ascending=False)
    save_importance_to_excel(fi_importance_df, label + '_fi', getattr(Config,'permutation_importance_log_path','log/permutation_importance_log.xlsx'))
    perm_result = permutation_importance(model, X_valid, y_valid, n_repeats=10, random_state=config.random_state, scoring='neg_mean_absolute_error')
    perm_importance_df = pd.DataFrame({'feature': feature_names,'importance_mean': perm_result.importances_mean,'importance_std': perm_result.importances_std}).sort_values('importance_mean', ascending=False)
    save_importance_to_excel(perm_importance_df, label + '_perm', getattr(Config,'permutation_importance_log_path','log/permutation_importance_log.xlsx'))
    perm_mask = perm_result.importances_mean <= 0
    perm_features = set(feature_names[perm_mask])
    explainer = shap.Explainer(model, X_valid)
    shap_values = explainer(X_valid)
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    shap_mask = shap_importance <= 0
    shap_features = set(feature_names[shap_mask])
    shap_importance_df = pd.DataFrame({'feature': feature_names,'importance_mean': shap_importance,'importance_std': [0]*len(feature_names)}).sort_values('importance_mean', ascending=False)
    save_importance_to_excel(shap_importance_df, label + '_shap', getattr(Config,'permutation_importance_log_path','log/permutation_importance_log.xlsx'))
    features_to_remove = fi_features | perm_features | shap_features
    print(f"Removed {len(features_to_remove)} features for {label} (fi:{len(fi_features)}, perm:{len(perm_features)}, shap:{len(shap_features)})")
    return list(features_to_remove)

def get_least_important_features(X, y, label, model_name=None):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=config.random_state)
    if (model_name or getattr(config, 'model_name', 'xgb')) == 'xgb':
        model = XGBRegressor(random_state=config.random_state, n_jobs=-1, verbosity=0, eval_metric="mae",
                             objective="reg:absoluteerror", tree_method="hist")
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False, early_stopping_rounds=200)
    else:
        model = ExtraTreesRegressor(random_state=config.random_state, criterion='absolute_error')
        model.fit(X, y)
    importance_method = getattr(config, 'feature_importance_method', 'feature_importances_')
    if importance_method == 'feature_importances_':
        importance_df = pd.DataFrame({'feature': X_train.columns,'importance_mean': model.feature_importances_,'importance_std':[0]*len(X_train.columns)})
    else:
        result = permutation_importance(model, X_valid, y_valid, n_repeats=30, random_state=Config.random_state, scoring='r2')
        importance_df = pd.DataFrame({'feature': X_train.columns,'importance_mean': result.importances_mean,'importance_std': result.importances_std})
    model_name_used = (model_name or getattr(config,'model_name','xgb'))
    n = config.n_least_important_features.get(model_name_used, {}).get(label, 5)
    negative_importance = importance_df[importance_df['importance_mean'] <= 0]
    num_negative = len(negative_importance)
    least_important = negative_importance
    if num_negative < n:
        remaining = importance_df[~importance_df['feature'].isin(negative_importance['feature'])]
        additional = remaining.sort_values(by='importance_mean').head(n - num_negative)
        least_important = pd.concat([least_important, additional], ignore_index=True)
    print(f"Removed {len(least_important)} least important features for {label} (<=0: {num_negative})")
    importance_df = importance_df.sort_values(by='importance_mean', ascending=True)
    importance_df['removed'] = importance_df['feature'].isin(least_important['feature'])
    save_importance_to_excel(importance_df, label, Config.permutation_importance_log_path)
    return least_important['feature'].tolist()

def save_model(Model, label, fold, model_name):
    model_path = f"models/{label}_fold{fold+1}_{model_name}"
    try:
        if 'torch' in str(type(Model)).lower():
            model_path += ".pt"
            torch.save(Model.state_dict(), model_path)
        else:
            model_path += ".joblib"
            joblib.dump(Model, model_path)
        print(f"Saved model for {label} fold {fold+1} to {model_path}")
    except Exception as e:
        print(f"Failed to save model for {label} fold {fold+1}: {e}")

def train_with_other_models(model_name, label, X_train, y_train, X_val, y_val):
    print(f"Training {model_name} model for label: {label}")
    if model_name == 'lgbm':
        params = {'n_estimators': 5000, 'learning_rate': 0.02, 'objective': 'mae', 'random_state': Config.random_state, 'verbose': -1}
        if label in ['Rg','Tc']:
            params.update({'max_depth': 4, 'num_leaves': 20, 'reg_lambda': 5.0})
        elif label == 'FFV':
            params.update({'max_depth': 7, 'num_leaves': 48, 'reg_lambda': 1.0})
        else:
            params.update({'max_depth': 6, 'num_leaves': 36, 'reg_lambda': 1.0})
        Model = LGBMRegressor(**params)
        Model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae',
                  callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])
    elif model_name == 'catboost':
        params = {'iterations': 5000, 'learning_rate': 0.02, 'loss_function':'MAE','eval_metric':'MAE','random_seed':Config.random_state,'verbose':False, 'depth':6}
        if label in ['Rg','Tc']: params.update({'depth':5})
        Model = CatBoostRegressor(**params)
        Model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=200, use_best_model=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Model

def train_with_autogluon(label, X_train, y_train):
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        raise ImportError("AutoGluon is not installed.")
    import uuid
    X_train_df = pd.DataFrame(X_train)
    y_train_series = pd.Series(y_train, name=label)
    train_data = X_train_df.copy(); train_data[label] = y_train_series.values
    unique_path = f"autogluon_{label}_{int(time.time())}_{uuid.uuid4().hex}"
    hyperparameters = {"GBM": {}, "CAT": {}, "XGB": {}, "NN_TORCH": {}, "RF": {}, "XT": {}}
    hyperparameter_tune_kwargs = {"num_trials": 50, "scheduler": "local", "searcher": "auto"}
    time_limit = 3600
    predictor = TabularPredictor(label=label, eval_metric="mae", path=unique_path).fit(
        train_data, presets="best_quality", hyperparameters=hyperparameters,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs, num_bag_folds=5, num_stack_levels=2, time_limit=time_limit
    )
    leaderboard = predictor.leaderboard(silent=False)
    fi_df = predictor.feature_importance(train_data)
    fi_df.to_csv(f"NeurIPS/autogluon_feature_importance_{label}.csv")
    return predictor

def train_with_xgb(label, X_train, y_train, X_val, y_val):
    print(f"Training XGB model for label: {label}")
    # >>> æ”¹è‰¯: ãƒ©ãƒ™ãƒ«åˆ¥ã�«å …ã‚�ã�®æ­£å‰‡åŒ–ã‚’è¿½åŠ ï¼ˆmin_child_weight / reg_alpha / colsample_bynodeï¼‰
    common = dict(
        n_estimators=6000, learning_rate=0.01,
        objective="reg:absoluteerror", eval_metric="mae",
        subsample=0.8, colsample_bytree=0.9, colsample_bynode=0.9,
        reg_lambda=5.0, reg_alpha=0.1, gamma=0.1,
        tree_method="hist", n_jobs=-1, random_state=Config.random_state
    )
    if label == "FFV":
        params = dict(common, max_depth=7, min_child_weight=6, subsample=0.7, colsample_bytree=0.8)
    elif label in ["Rg","Tc"]:
        params = dict(common, max_depth=4, min_child_weight=6, subsample=0.8)
    elif label == "Tg":
        params = dict(common, max_depth=5, min_child_weight=3)
    else:  # Density
        params = dict(common, max_depth=5, min_child_weight=4, subsample=0.8)

    Model = XGBRegressor(**params)
    Model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=200)
    return Model

def preprocess_numerical_features(X, label=None):
    X_num = X.select_dtypes(include=[np.number]).copy()
    X_num.replace([np.inf, -np.inf], np.nan, inplace=True)
    valid_cols = X_num.columns
    median_values = X_num.median()
    if getattr(Config, 'use_standard_scaler', False):
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
        X_num = pd.DataFrame(X_num_scaled, columns=valid_cols, index=X.index)
    else:
        scaler = None
        X_num = X_num.copy()
    return X_num, valid_cols, scaler, median_values

def select_features_with_lasso(X, y, label):
    X_filled = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)
    lasso_cv = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=3000)
    feature_selector = SelectFromModel(lasso_cv, prefit=False, threshold=None)
    print(f"[{label}] Fitting LassoCV to find optimal features...")
    feature_selector.fit(X_scaled, y)
    selected_feature_names = X.columns[feature_selector.get_support()]
    print(f"[{label}] Original: {X.shape[1]}  Selected: {len(selected_feature_names)}")
    return selected_feature_names

def check_inf_nan(X, y, label=None):
    X_inf = np.isinf(X.values).any(); X_nan = np.isnan(X.values).any()
    y_inf = np.isinf(y).any(); y_nan = np.isnan(y).any()
    if X_inf or X_nan or y_inf or y_nan:
        print(f"âš ï¸� inf/nan in X or y [{label}] -> X_inf={X_inf}, X_nan={X_nan}, y_inf={y_inf}, y_nan={y_nan}")
        if X_inf: print(f"  X columns with inf: {X.columns[np.isinf(X.values).any(axis=0)].tolist()}")
        if X_nan: print(f"  X columns with nan: {X.columns[np.isnan(X.values).any(axis=0)].tolist()}")
    else:
        print(f"No inf/nan in X or y [{label}].")
    return X_inf or X_nan or y_inf or y_nan

def show_model_summary(model, input_dim, batch_size=32):
    try:
        from torchinfo import summary
        print(summary(model, input_size=(batch_size, input_dim)))
    except ImportError:
        pass

def train_model(model, X_train, X_val, y_train, y_val, epochs=3000, batch_size=32, lr=1e-3, weight_decay=1e-4, patience=30, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_train = np.asarray(X_train); y_train = np.asarray(y_train)
    X_val   = np.asarray(X_val);   y_val   = np.asarray(y_val)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # AdamWã�«
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=verbose)
    best_val_loss = float('inf'); best_model_state = None; epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad(); preds = model(xb); loss = criterion(preds, yb); loss.backward(); optimizer.step()
        if verbose and (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        with torch.no_grad():
            model.eval(); val_preds = model(X_val_tensor); val_loss = criterion(val_preds, y_val_tensor).item()
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss; best_model_state = model.state_dict(); epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            if verbose: print(f"Early stopping at epoch {epoch+1}, best val loss: {best_val_loss:.4f}")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
    return model

class FeedforwardNet(nn.Module):
    def __init__(self, input_dim, neurons, dropouts):
        super().__init__()
        layers = []
        for i, n in enumerate(neurons):
            layers.append(nn.Linear(input_dim, n))
            layers.append(nn.ReLU())
            if i < len(dropouts) and dropouts[i] > 0:
                layers.append(nn.Dropout(dropouts[i]))
            input_dim = n
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

def train_with_nn(label, X_train, X_val, y_train, y_val):
    input_dim = X_train.shape[1]
    best_configs = {
        "Tg":      {"neurons": [256, 128, 64], "dropouts": [0.4, 0.3, 0.2]},
        "Density": {"neurons": [256, 128, 64], "dropouts": [0.4, 0.3, 0.2]},
        "FFV":     {"neurons": [512, 256, 128], "dropouts": [0.3, 0.2, 0.2]},
        "Tc":      {"neurons": [128, 64],       "dropouts": [0.4, 0.3]},
        "Rg":      {"neurons": [128, 64],       "dropouts": [0.4, 0.3]},
    }
    cfg = best_configs.get(label)
    best_model = FeedforwardNet(input_dim, cfg["neurons"], cfg["dropouts"])
    show_model_summary(best_model, input_dim)
    best_model = train_model(best_model, X_train, X_val, y_train, y_val, verbose=True)
    return best_model

def set_global_random_seed(seed, config=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    random.seed(seed); np.random.seed(seed)
    if config is not None: config.random_state = seed

import hashlib
def stable_hash(obj, max_value=1_000_000):
    s = str(obj).encode("utf-8"); h = hashlib.md5(s).hexdigest()
    return int(h, 16) % max_value

def train_and_evaluate_models(label, X_main, y_main, splits, nfold, Config):
    FOLD_PRIME = 9973
    models, fold_maes = [], []
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Fold {fold+1}/{nfold} ---")
        base_seed = getattr(Config, 'random_state', 42)
        label_hash = stable_hash(label)
        fold_seed = base_seed + fold * FOLD_PRIME + label_hash
        set_global_random_seed(fold_seed, config=Config)
        if isinstance(X_main, np.ndarray):
            X_train, X_val = X_main[train_idx], X_main[val_idx]
        else:
            X_train, X_val = X_main.iloc[train_idx], X_main.iloc[val_idx]
        if isinstance(y_main, np.ndarray):
            y_train, y_val = y_main[train_idx], y_main[val_idx]
        else:
            y_train, y_val = y_main.iloc[train_idx], y_main.iloc[val_idx]
        if Config.model_name == 'xgb':
            Model = train_with_xgb(label, X_train, y_train, X_val, y_val)
        elif Config.model_name in ['catboost','lgbm','nn']:
            Model = train_with_other_models(Config.model_name, label, X_train, y_train, X_val, y_val) if Config.model_name!='nn' else train_with_nn(label, X_train, X_val, y_train, y_val)
        else:
            raise ValueError("Unsupported model_name")
        models.append(Model)
        # OOF MAE
        if hasattr(Model, 'forward') and not hasattr(Model, 'predict'):
            Model.eval(); dev = next(Model.parameters()).device
            with torch.no_grad():
                X_val_tensor = torch.tensor(np.asarray(X_val), dtype=torch.float32).to(dev)
                y_val_pred = Model(X_val_tensor).cpu().numpy().flatten()
        else:
            y_val_pred = Model.predict(X_val)
        fold_mae = mean_absolute_error(y_val, y_val_pred)
        print(f"Fold {fold+1} MAE: {fold_mae:.6f}")
        fold_maes.append(fold_mae)
    mean_fold_mae = float(np.mean(fold_maes)); std_fold_mae = float(np.std(fold_maes))
    print(f"{label} CV MAE: {mean_fold_mae:.6f} Â± {std_fold_mae:.6f}")
    return models, fold_maes, mean_fold_mae, std_fold_mae

def save_feature_selection_info(label, kept_columns, least_important_features, correlated_features_dropped, scaler, X_holdout, y_holdout, median_values):
    holdout_dir = f"NeurIPS/feature_selection/{label}"
    os.makedirs(holdout_dir, exist_ok=True)
    feature_info = {
        "kept_columns": list(kept_columns),
        "least_important_features": list(least_important_features),
        "correlated_features_dropped": list(correlated_features_dropped),
        "median_values": median_values.to_dict() if hasattr(median_values,'to_dict') else {}
    }
    pd.DataFrame(X_holdout).to_csv(os.path.join(holdout_dir,"X_holdout.csv"), index=False)
    pd.DataFrame({"y_holdout": y_holdout}).to_csv(os.path.join(holdout_dir,"y_holdout.csv"), index=False)
    with open(os.path.join(holdout_dir,f"{label}_feature_info.json"),"w") as f:
        json.dump(feature_info, f, indent=2)
    if scaler is not None:
        joblib.dump(scaler, os.path.join(holdout_dir,"scaler.joblib"))

def load_feature_selection_info(label, base_dir):
    holdout_dir = os.path.join(base_dir, f"NeurIPS/feature_selection/{label}")
    feature_info_path = os.path.join(holdout_dir, f"{label}_feature_info.json")
    X_holdout_path = os.path.join(holdout_dir, "X_holdout.csv")
    y_holdout_path = os.path.join(holdout_dir, "y_holdout.csv")
    if not os.path.exists(feature_info_path):
        raise FileNotFoundError(f"Feature info file not found: {feature_info_path}")
    with open(feature_info_path, "r") as f:
        feature_info = json.load(f)
    X_holdout = pd.read_csv(X_holdout_path)
    y_holdout = pd.read_csv(y_holdout_path)["y_holdout"].values
    scaler_path = os.path.join(holdout_dir, "scaler.joblib")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    median_values = pd.Series(feature_info.get("median_values", {}))
    return {
        "kept_columns": feature_info.get("kept_columns", []),
        "least_important_features": feature_info.get("least_important_features", []),
        "correlated_features_dropped": feature_info.get("correlated_features_dropped", []),
        "X_holdout": X_holdout,
        "y_holdout": y_holdout,
        "scaler": scaler,
        "median_values": median_values
    }

def load_models_for_label(label, models_dir="models"):
    models = []
    if not os.path.exists(models_dir):
        print(f"Models directory '{models_dir}' does not exist.")
        return models
    pattern_joblib = os.path.join(models_dir, f"{label}_*.joblib")
    pattern_pt     = os.path.join(models_dir, f"{label}_*.pt")
    model_files = glob.glob(pattern_joblib) + glob.glob(pattern_pt)
    if not model_files:
        print(f"No models found for label '{label}' in '{models_dir}'.")
        return models
    for model_file in sorted(model_files):
        if model_file.endswith(".joblib"):
            try:
                model = joblib.load(model_file); models.append(model)
            except Exception as e:
                print(f"Failed to load model {model_file}: {e}")
        elif model_file.endswith(".pt"):
            print(f"Skipping torch model {model_file} (requires model class).")
    print(f"Loaded {len(models)} models for label '{label}'.")
    return models

output_df = pd.DataFrame({'id': test_ids})
mae_results = []

def prepare_label_data(label, subtables, Config):
    print(f"Processing label: {label}")
    original_smiles = subtables[label]['SMILES'].tolist()
    original_labels = subtables[label][label].values
    canonical_smiles = [get_canonical_smiles(s) for s in original_smiles]
    smiles_label_df = pd.DataFrame({'SMILES': canonical_smiles, 'label': original_labels}).drop_duplicates(subset=['SMILES'], keep='first').reset_index(drop=True)
    original_smiles = smiles_label_df['SMILES'].tolist()
    original_labels = smiles_label_df['label'].values
    fp_df, descriptor_df, valid_smiles, invalid_indices = smiles_to_combined_fingerprints_with_descriptors(original_smiles)
    y = np.delete(original_labels, invalid_indices)
    if not descriptor_df.empty:
        X = pd.DataFrame(descriptor_df)
        X, kept_columns, scaler, median_values = preprocess_numerical_features(X, label)
        X.reset_index(drop=True, inplace=True)
        if not fp_df.empty:
            X = pd.concat([X, fp_df.reset_index(drop=True)], axis=1)
    else:
        kept_columns = []
        scaler = None
        median_values = pd.Series(dtype=float)
        X = fp_df
    X_dup = X.duplicated(keep='first')
    if X_dup.any():
        print(f"Found {X_dup.sum()} duplicate rows in X for {label}, removing them.")
        X = X[~X_dup]; y = y[~X_dup]
    check_inf_nan(X, y, label)

    # --- ãƒ©ãƒ™ãƒ«åˆ¥L1é�¸æŠœï¼ˆä½�ãƒ‡ãƒ¼ã‚¿ã�®Rg/Tcã�§ç‰¹ã�«åŠ¹ã��ï¼‰ ---
    if label in ['Rg','Tc'] and X.shape[1] > 64:
        feats = select_features_with_lasso(X, y, label)
        X = X[feats]; kept_columns = list(feats)

    # ç›¸é–¢åˆˆã‚Š
    corr_th = Config.correlation_thresholds.get(label, 0.96)
    if corr_th < 1.0:
        X, correlated_features_dropped = drop_correlated_features(pd.DataFrame(X), threshold=corr_th)
    else:
        correlated_features_dropped = []
    check_inf_nan(X, y, label)

    # PCAï¼ˆå…¨ä½“ã‚’åœ§ç¸®ï¼‰
    if getattr(Config, 'use_pca', False):
        X_main, X_holdout, y_main, y_holdout = train_test_split(X, y, test_size=0.10, random_state=Config.random_state,
                                                                stratify=pd.qcut(y, q=5, duplicates='drop', labels=False))
        X_main_pca, X_holdout_pca, pca = apply_pca(X_main, X_holdout, verbose=True)
        X_main, X_holdout = X_main_pca, X_holdout_pca
    else:
        y_bins = pd.qcut(y, q=5, duplicates='drop', labels=False)
        X_main, X_holdout, y_main, y_holdout = train_test_split(X, y, test_size=0.10, random_state=Config.random_state, stratify=y_bins)
        pca = None

    # StratifiedKFoldï¼ˆå›�å¸°â†’ãƒ“ãƒ‹ãƒ³ã‚°ï¼‰
    if getattr(Config, 'use_cross_validation', True):
        nfold = 10
        from sklearn.model_selection import StratifiedKFold
        y_bins = pd.qcut(y_main, q=nfold, duplicates='drop', labels=False)
        skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=Config.random_state)
        splits = skf.split(X_main, y_bins)
    else:
        idx = np.arange(len(X_main))
        tr, va = train_test_split(idx, test_size=0.2, random_state=Config.random_state)
        splits = [(tr, va)]; nfold = 1

    return {
        "X_main": X_main,
        "X_holdout": X_holdout,
        "y_main": y_main,
        "y_holdout": y_holdout,
        "kept_columns": kept_columns,
        "scaler": scaler,
        "median_values": median_values,
        "least_important_features": [],
        "correlated_features_dropped": correlated_features_dropped,
        "selector": None,
        "selected_cols_variance": None,
        "pca": pca,
        "splits": splits,
        "nfold": nfold,
        "fold_maes": [],
        "test_preds": [],
        "val_preds": np.zeros(len(y_main))
    }

def load_label_data(label, model_dir=None):
    if model_dir is not None:
        model_path = os.path.join(model_dir, f"{label}_model.pkl")
        data_path  = os.path.join(model_dir, f"{label}_data.pkl")
        model = joblib.load(model_path); data = joblib.load(data_path)
        return model, data
    return None, None

def train_or_predict(train_model=True, model_dir=None):
    for label in labels:
        if train_model:
            print(f"\n=== Training/Predicting for label: {label} ===")
            label_data = prepare_label_data(label, subtables, config)
            X_main = label_data["X_main"]; X_holdout = label_data["X_holdout"]
            y_main = label_data["y_main"]; y_holdout = label_data["y_holdout"]
            kept_columns = label_data["kept_columns"]; scaler = label_data["scaler"]
            median_values = label_data["median_values"]; correlated_features_dropped = label_data["correlated_features_dropped"]
            pca = label_data["pca"]; splits = label_data["splits"]; nfold = label_data["nfold"]
            fold_maes = label_data["fold_maes"]; test_preds = label_data["test_preds"]

            save_feature_selection_info(label, kept_columns, [], correlated_features_dropped, scaler, X_holdout, y_holdout, median_values)
            os.makedirs('models', exist_ok=True)

            models, fold_maes, mean_fold_mae, std_fold_mae = train_and_evaluate_models(label, X_main, y_main, splits, nfold, Config)
        else:
            print(f"\n=== Loading models and data for label: {label} ===")
            feature_info = load_feature_selection_info(label, model_dir)
            kept_columns = feature_info["kept_columns"]; correlated_features_dropped = feature_info["correlated_features_dropped"]
            scaler = feature_info.get("scaler", None); X_holdout = feature_info["X_holdout"]; y_holdout = feature_info["y_holdout"]
            median_values = feature_info["median_values"]; pca = None
            models = load_models_for_label(label, os.path.join(model_dir, 'models'))
            test_preds = []; fold_maes = []

        # --- ãƒ†ã‚¹ãƒˆç‰¹å¾´ã�®æ§‹ç¯‰ï¼ˆå­¦ç¿’æ™‚ã�¨å�Œã�˜åˆ—é †/ã‚¹ã‚±ãƒ¼ãƒªãƒ³ã‚°/PCAï¼‰ ---
        fp_df, descriptor_df, valid_smiles, invalid_indices = smiles_to_combined_fingerprints_with_descriptors(test_smiles)
        if not descriptor_df.empty:
            descriptor_df = descriptor_df.reindex(columns=kept_columns) if len(kept_columns) > 0 else descriptor_df
            if getattr(Config, 'use_standard_scaler', False) and scaler is not None:
                descriptor_df = pd.DataFrame(scaler.transform(descriptor_df), columns=descriptor_df.columns, index=descriptor_df.index)
            descriptor_df.reset_index(drop=True, inplace=True)
            test = pd.concat([descriptor_df, fp_df.reset_index(drop=True)], axis=1) if not fp_df.empty else descriptor_df
        else:
            test = fp_df
        if len(correlated_features_dropped) > 0:
            test = test.drop(correlated_features_dropped, axis=1, errors='ignore')
        if getattr(Config, 'use_pca', False) and pca is not None:
            test = pca.transform(test.values)

        # --- Holdoutè©•ä¾¡ & äºˆæ¸¬ ---
        holdout_maes = []
        for i, Model in enumerate(models):
            if hasattr(Model, 'forward') and not hasattr(Model, 'predict'):
                Model.eval()
                X_holdout_np = np.asarray(X_holdout) if isinstance(X_holdout, pd.DataFrame) else X_holdout
                test_np = np.asarray(test) if isinstance(test, (pd.DataFrame, pd.Series)) else test
                device = next(Model.parameters()).device
                with torch.no_grad():
                    X_holdout_tensor = torch.tensor(X_holdout_np, dtype=torch.float32).to(device)
                    test_tensor      = torch.tensor(test_np,      dtype=torch.float32).to(device)
                    y_holdout_pred   = Model(X_holdout_tensor).detach().cpu().numpy().flatten()
                    y_test_pred      = Model(test_tensor).detach().cpu().numpy().flatten()
            else:
                y_holdout_pred = Model.predict(X_holdout)
                y_test_pred    = Model.predict(test)
            holdout_mae = mean_absolute_error(y_holdout, y_holdout_pred)
            holdout_maes.append(holdout_mae)
            y_test_pred = y_test_pred.values.flatten() if isinstance(y_test_pred, pd.Series) else y_test_pred.flatten()
            test_preds.append(y_test_pred)

        mean_holdout_mae = np.mean(holdout_maes) if len(holdout_maes)>0 else np.mean(fold_maes) if len(fold_maes)>0 else np.nan
        std_holdout_mae  = np.std(holdout_maes)  if len(holdout_maes)>0 else np.std(fold_maes)  if len(fold_maes)>0 else np.nan
        print(f"{label} Holdout MAE (mean Â± std): {mean_holdout_mae:.5f} Â± {std_holdout_mae:.5f}")

        mae_results.append({
            'label': label,
            'fold_mae_mean': float(np.mean(fold_maes)) if len(fold_maes)>0 else None,
            'fold_mae_std': float(np.std(fold_maes)) if len(fold_maes)>0 else None,
            'holdout_mae_mean': float(mean_holdout_mae) if mean_holdout_mae==mean_holdout_mae else None,
            'holdout_mae_std': float(std_holdout_mae) if std_holdout_mae==std_holdout_mae else None
        })

        # --- é€†MAEé‡�ã�¿ã�§ã�®ãƒ–ãƒ¬ãƒ³ãƒ‰ï¼ˆå¹³å�‡ã‚ˆã‚Šå …ã�„ï¼‰ ---
        test_preds = np.array(test_preds)  # (n_models, n_test)
        if len(holdout_maes) > 0:
            w_src = np.array(holdout_maes)
        elif len(fold_maes) > 0:
            w_src = np.array(fold_maes)
        else:
            w_src = np.ones(test_preds.shape[0], dtype=float)
        w = 1.0 / (w_src + 1e-9); w = w / w.sum()
        y_pred = (w[:, None] * test_preds).sum(axis=0)

        output_df[label] = y_pred

    mae_df = pd.DataFrame(mae_results)
    mae_df.to_csv('NeurIPS/mae_results.csv', index=False)
    print("\nMean Absolute Error for each label:")
    print(mae_df)

# ===== å®Ÿè¡Œ =====
output_df = pd.DataFrame({'id': test_ids})
# äº‹å‰�å­¦ç¿’æ¸ˆã�¿ãƒ¢ãƒ‡ãƒ«ã�§ã�¯åŠ¹æ�œã�Œå��æ˜ ã�•ã‚Œã�ªã�„ã�Ÿã‚�ã€�å­¦ç¿’ãƒ¢ãƒ¼ãƒ‰ã�§å®Ÿè¡Œ
train_or_predict(train_model=True, model_dir=None)
print(output_df)
output_dfs.append(output_df.copy())



!pip install /kaggle/input/torch-geometric-2-6-1/torch_geometric-2.6.1-py3-none-any.whl


import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import joblib
import os
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool, global_max_pool
import torch.nn.functional as F
import warnings
import json
import torch
from sklearn.preprocessing import RobustScaler
import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

RDKIT_AVAILABLE = True
TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

os.makedirs("NeurIPS", exist_ok=True)
class Config:
    debug = False
    use_cross_validation = True  # Set to False to use a single split for speed
    use_external_data = True  # Set to True to use external datasets
    random_state = 42

# Create a single config instance to use everywhere
config = Config()

"""
Load competition data with complete filtering of problematic polymer notation
"""
print("Loading competition data...")
train = pd.read_csv(BASE_PATH + 'train.csv')
test = pd.read_csv(BASE_PATH + 'test.csv')

if config.debug:
    print("   Debug mode: sampling 1000 training examples")
    train = train.sample(n=1000, random_state=42).reset_index(drop=True)

print(f"Training data shape: {train.shape}, Test data shape: {test.shape}")

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
print("Cleaning and validating SMILES...")
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
    print(f"      Filled {filled_count} missing entries in train for {target}")
    print(f"      Added {len(extra_to_add)} new entries for {target}")
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
    BASE_PATH + 'train_supplement/dataset4.csv',
    'FFV', 
    lambda df: df[['SMILES', 'FFV']] if 'FFV' in df.columns else df,
    'dataset 4'
)

# Integrate external data
print("\nğŸ”„ Integrating external data...")
train_extended = train[['SMILES'] + TARGETS].copy()

if getattr(config, "use_external_data", True) and  not config.debug:
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
        # Filter out NaNs, select columns, reset index
        subtables[label] = train_df[train_df[label].notna()][['SMILES', label]].reset_index(drop=True)

    return subtables

def augment_smiles_dataset(smiles_list, labels, num_augments=3):
    """
    Augments a list of SMILES strings by generating randomized versions.

    Parameters:
        smiles_list (list of str): Original SMILES strings.
        labels (list or np.array): Corresponding labels.
        num_augments (int): Number of augmentations per SMILES.

    Returns:
        tuple: (augmented_smiles, augmented_labels)
    """
    augmented_smiles = []
    augmented_labels = []

    for smiles, label in zip(smiles_list, labels):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        # Add original
        augmented_smiles.append(smiles)
        augmented_labels.append(label)
        # Add randomized versions
        for _ in range(num_augments):
            rand_smiles = Chem.MolToSmiles(mol, doRandom=True)
            augmented_smiles.append(rand_smiles)
            augmented_labels.append(label)

    return augmented_smiles, np.array(augmented_labels)

required_descriptors = {'graph_diameter','num_cycles','avg_shortest_path','MolWt', 'LogP', 'TPSA', 'RotatableBonds', 'NumAtoms'}

def augment_dataset(X, y, n_samples=1000, n_components=5, random_state=None):
    """
    Augments a dataset using Gaussian Mixture Models.

    Parameters:
    - X: pd.DataFrame or np.ndarray â€” feature matrix
    - y: pd.Series or np.ndarray â€” target values
    - n_samples: int â€” number of synthetic samples to generate
    - n_components: int â€” number of GMM components
    - random_state: int â€” random seed for reproducibility

    Returns:
    - X_augmented: pd.DataFrame â€” augmented feature matrix
    - y_augmented: pd.Series â€” augmented target values
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    elif not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame or a NumPy array")

    X.columns = X.columns.astype(str)

    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    elif not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series or a NumPy array")

    df = X.copy()
    df['Target'] = y.values

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(df)

    synthetic_data, _ = gmm.sample(n_samples)
    synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)

    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)

    X_augmented = augmented_df.drop(columns='Target')
    y_augmented = augmented_df['Target']

    return X_augmented, y_augmented


train_df=train_extended
test_df=test
subtables = separate_subtables(train_df)

test_smiles = test_df['SMILES'].tolist()
test_ids = test_df['id'].values
labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# ------------------------------------------------------------------
# --- GNN MODEL AND DATA PREPARATION ---
# ------------------------------------------------------------------

# A dictionary to map atom symbols to integer indices for the GNN
ATOM_MAP = {
    'C': 0, 'N': 1, 'O': 2, 'F': 3, 'P': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8, 'H': 9,
    # --- NEWLY ADDED SYMBOLS ---
    'Si': 10, # Silicon
    'Na': 11, # Sodium
    '*' : 12, # Wildcard atom
    # --- NEWLY ADDED SYMBOLS ---
    'B': 13,  # Boron
    'Ge': 14, # Germanium
    'Sn': 15, # Tin
    'Se': 16, # Selenium
    'Te': 17, # Tellurium
    'Ca': 18, # Calcium
    'Cd': 19, # Cadmium
}

def smiles_to_graph(smiles_str: str, y_val=None):
    """
    Converts a SMILES string to a graph, adding selected global
    molecular features to each node's feature vector.
    """
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None: return None

        # 1. Calculate global features once per molecule
        global_features = [
            Descriptors.MolWt(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.MolLogP(mol)
        ]

        node_features = []
        for atom in mol.GetAtoms():
            # Initialize atom-specific features (one-hot encoding)
            atom_features = [0] * len(ATOM_MAP)
            symbol = atom.GetSymbol()
            if symbol in ATOM_MAP:
                atom_features[ATOM_MAP[symbol]] = 1

            # Add other standard atom features
            atom_features.extend([
                atom.GetAtomicNum(),
                atom.GetTotalDegree(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                int(atom.GetIsAromatic())
            ])
            
            # 2. Append the global features to each atom's feature vector
            atom_features.extend(global_features)
            
            node_features.append(atom_features)
        
        if not node_features: return None
        x = torch.tensor(node_features, dtype=torch.float)

        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.extend([(i, j), (j, i)])
            bond_type = bond.GetBondTypeAsDouble()
            edge_attrs.extend([[bond_type], [bond_type]])

        if not edge_indices:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        if y_val is not None:
            y_tensor = torch.tensor([[y_val]], dtype=torch.float)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
        else:
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except Exception as e:
        return None

from rdkit.Chem import Descriptors

# A dictionary mapping labels to their most important global features from XGBoost
LABEL_SPECIFIC_FEATURES = {
    'Tg': [
        "HallKierAlpha", # Topological charge index
        "MolLogP",       # Lipophilicity
        "NumRotatableBonds", # Flexibility
        "TPSA",          # Polarity
    ],
    'FFV': [
        "NHOHCount",     # Count of NH and OH groups (H-bonding)
        "NumRotatableBonds",
        "MolWt",         # Size
        "TPSA",
    ],
    'Tc': [
        "MolLogP",
        "NumValenceElectrons",
        "SPS",           # Molecular shape index
        "MolWt",
    ],
    'Density': [
        "MolWt",
        "MolMR",         # Molar refractivity (related to volume)
        "FractionCSP3",  # Proportion of sp3 hybridized carbons (related to saturation)
        "NumHeteroatoms",
    ],
    'Rg': [
        "HallKierAlpha",
        "MolWt",
        "NumValenceElectrons",
        "qed",           # Quantitative Estimation of Drug-likeness
    ]
}

# A helper dictionary to easily call RDKit functions from their string names
RDKIT_DESC_CALCULATORS = {name: func for name, func in Descriptors.descList}
RDKIT_DESC_CALCULATORS['qed'] = Descriptors.qed # Add qed as it's not in the default list

from rdkit import Chem
import numpy as np

# This ATOM_MAP dictionary must be defined globally in your script (it already is)
# ATOM_MAP = {'C': 0, 'N': 1, ...}

def smiles_to_graph_label_specific(smiles_str: str, label: str, y_val=None):
    """
    (BASELINE VERSION - SIMPLE FEATURES)
    - This is the original hybrid GNN featurizer that produced your best score.
    - Node Features (x): Atom one-hot (20) + 5 atom features = 25 features.
    - Edge Features (edge_attr): Bond type as double = 1 feature.
    - Global Features (u): Label-specific descriptors are stored separately in 'data.u'.
    """
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None: 
            return None

        # --- 1. Calculate and store label-specific GLOBAL features ---
        global_features = []
        features_to_calculate = LABEL_SPECIFIC_FEATURES.get(label, [])
        
        for feature_name in features_to_calculate:
            calculator_func = RDKIT_DESC_CALCULATORS.get(feature_name)
            if calculator_func:
                try:
                    val = calculator_func(mol)
                    # Ensure value is valid, replace inf/nan with 0
                    global_features.append(val if np.isfinite(val) else 0.0)
                except Exception as e:
                    global_features.append(0.0)
            else:
                global_features.append(0.0)

        # --- 2. Create Node Features (SIMPLE) ---
        node_features = []
        for atom in mol.GetAtoms():
            # One-Hot Symbol (len 20, from global ATOM_MAP)
            atom_features = [0] * len(ATOM_MAP)
            symbol = atom.GetSymbol()
            if symbol in ATOM_MAP:
                atom_features[ATOM_MAP[symbol]] = 1

            # Standard Features (len 5)
            atom_features.extend([
                atom.GetAtomicNum(),
                atom.GetTotalDegree(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                int(atom.GetIsAromatic())
            ])
            # Total features = 25
            node_features.append(atom_features)
        
        if not node_features: return None
        x = torch.tensor(node_features, dtype=torch.float)

        # --- 3. Create Edge Features (SIMPLE) ---
        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.extend([(i, j), (j, i)])
            bond_type = bond.GetBondTypeAsDouble()
            edge_attrs.extend([[bond_type], [bond_type]]) # 1-dim feature

        if not edge_indices:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float) # Shape (0, 1)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        # --- 4. Create Data Object ---
        data_obj = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data_obj.u = torch.tensor([global_features], dtype=torch.float) # Store globals in 'u'

        if y_val is not None:
            data_obj.y = torch.tensor([[y_val]], dtype=torch.float)
        
        return data_obj
        
    except Exception as e:
        # Catch any other unexpected molecule-level errors
        print(f"CRITICAL ERROR converting SMILES '{smiles_str}': {e}")
        return None
            
class GNNModel(torch.nn.Module):
    """
    Defines the Graph Neural Network architecture.
    """
    def __init__(self, num_node_features, hidden_channels=128):
        super(GNNModel, self).__init__()
        torch.manual_seed(42)
        
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.conv3 = GCNConv(hidden_channels * 2, hidden_channels * 4)
        self.lin = torch.nn.Linear(hidden_channels * 4, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = global_max_pool(x, batch) # Aggregate node features to get a graph-level embedding
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin(x)
        
        return x


def predict_with_gnn(trained_model, test_smiles):
    """
    Uses a pre-trained GNN model to make predictions on a list of test SMILES.
    """
    if trained_model is None:
        print("Prediction skipped because the GNN model is invalid.")
        return np.full(len(test_smiles), np.nan)

    print("--- Making predictions with trained GNN... ---")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert test SMILES to graph data
    test_data_list = [smiles_to_graph(s) for s in test_smiles]
    
    # We need to keep track of which original indices are valid
    valid_indices = [i for i, data in enumerate(test_data_list) if data is not None]
    valid_test_data = [data for data in test_data_list if data is not None]

    if not valid_test_data:
        print("Warning: No valid test molecules could be converted to graphs.")
        return np.full(len(test_smiles), np.nan)
        
    test_loader = PyGDataLoader(valid_test_data, batch_size=32, shuffle=False)

    trained_model.eval()
    all_preds = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            out = trained_model(data)
            all_preds.append(out.cpu())

    # Combine predictions from all batches
    test_preds_tensor = torch.cat(all_preds, dim=0).numpy().flatten()
    
    # Create a full-sized prediction array and fill in the values at their original positions
    final_predictions = np.full(len(test_smiles), np.nan)
    if len(test_preds_tensor) == len(valid_indices):
        final_predictions[valid_indices] = test_preds_tensor
    else:
        print(f"Warning: Mismatch in GNN prediction count. This can happen with invalid SMILES.")
        fill_count = min(len(valid_indices), len(test_preds_tensor))
        final_predictions[valid_indices[:fill_count]] = test_preds_tensor[:fill_count]

    return final_predictions

import json
import os

def save_gnn_model(model, label, model_dir="models/gnn"):
    """
    (MODIFIED) Saves the GNN model state_dict and its full constructor config.
    """
    if model is None:
        print(f"Skipping save for {label}, model is None.")
        return

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"gnn_model_{label}.pth")
    config_path = os.path.join(model_dir, f"gnn_config_{label}.json")

    # Save the model parameters (the weights)
    torch.save(model.state_dict(), model_path)
    
    # Save the full configuration dictionary
    with open(config_path, 'w') as f:
        json.dump(model.config_args, f, indent=4)
        
    print(f"Saved final model for {label} to {model_path}")


def load_gnn_model(label, model_dir="models/gnn"):
    """
    (MODIFIED) Loads a saved GNN model using its full config file.
    """
    model_path = os.path.join(model_dir, f"gnn_model_{label}.pth")
    config_path = os.path.join(model_dir, f"gnn_config_{label}.json")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print(f"Warning: Model or config file not found for {label}. Cannot load model.")
        return None

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    try:
        # Re-initialize the model using all saved config args via dictionary unpacking
        model = TaskSpecificGNN(**config).to(DEVICE)
        
        # Load the saved model weights
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval() # Set model to evaluation mode
        print(f"Successfully loaded saved model for {label} from {model_path}")
        return model

    except Exception as e:
        print(f"CRITICAL ERROR loading model for {label}: {e}")
        print("This may be due to a mismatch between the saved model and the current model class definition.")
        return None
    

def create_dynamic_mlp(input_dim, layer_list, dropout_list):
    """
    Helper function to dynamically build the task-specific MLP.
    """
    layers = []
    current_dim = input_dim
    
    for neurons, dropout in zip(layer_list, dropout_list):
        layers.append(torch.nn.Linear(current_dim, neurons))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))
        current_dim = neurons
        
    # Add the final single-output prediction layer
    layers.append(torch.nn.Linear(current_dim, 1))
    
    return torch.nn.Sequential(*layers)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class TaskSpecificGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_global_features,
                 hidden_channels_gnn, mlp_neurons, mlp_dropouts, heads=8):
        super().__init__()
        torch.manual_seed(42)

        # --- 1. GNN Backbone (Using GATConv, No BatchNorm) ---
        self.convs = torch.nn.ModuleList()

        # Layer 1
        self.convs.append(
            GATConv(num_node_features, hidden_channels_gnn, heads=heads,
                    edge_dim=num_edge_features)
        )

        # Layer 2
        self.convs.append(
            GATConv(hidden_channels_gnn * heads, hidden_channels_gnn * 2, heads=heads,
                    edge_dim=num_edge_features)
        )

        # Layer 3 (Final GNN layer)
        self.convs.append(
            GATConv(hidden_channels_gnn * 2 * heads, hidden_channels_gnn * 4, heads=heads,
                    concat=False, edge_dim=num_edge_features)
        )

        gnn_output_dim = hidden_channels_gnn * 4

        # --- 2. Readout Head ---
        combined_feature_size = gnn_output_dim + num_global_features

        self.readout_mlp = create_dynamic_mlp(
            input_dim=combined_feature_size,
            layer_list=mlp_neurons,
            dropout_list=mlp_dropouts
        )

        # --- 3. Store config for saving/loading ---
        self.config_args = {
            'num_node_features': num_node_features,
            'num_edge_features': num_edge_features,
            'num_global_features': num_global_features,
            'hidden_channels_gnn': hidden_channels_gnn,
            'mlp_neurons': mlp_neurons,
            'mlp_dropouts': mlp_dropouts,
            'heads': heads
        }

    def forward(self, data):
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch

        # GNN Layers with ReLU and Dropout
        x = F.relu(self.convs[0](x, edge_index, edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.convs[1](x, edge_index, edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.convs[2](x, edge_index, edge_attr))

        # Readout
        graph_embedding = global_mean_pool(x, batch)
        combined_features = torch.cat([graph_embedding, u], dim=1)
        output = self.readout_mlp(combined_features)

        return output
        
# This is a new helper, just to make scaling code cleaner inside the loops
def scale_graph_features(data_list, u_scaler, x_scaler, atom_map_len):
    """Applies fitted scalers in-place to a list of Data objects."""
    try:
        for data in data_list:
            # 1. Scale global features (u)
            data.u = torch.tensor(u_scaler.transform(data.u.numpy()), dtype=torch.float)
            
            # 2. Scale continuous part of node features (x)
            x_one_hot = data.x[:, :atom_map_len]
            x_continuous = data.x[:, atom_map_len:]
            
            x_continuous_scaled = x_scaler.transform(x_continuous.numpy())
            x_continuous_scaled_tensor = torch.tensor(x_continuous_scaled, dtype=torch.float)
            
            # Recombine scaled features
            data.x = torch.cat([x_one_hot, x_continuous_scaled_tensor], dim=1)
            
    except Exception as e:
        print(f"CRITICAL ERROR applying scalers: {e}. Check feature dimensions. AtomMapLen={atom_map_len}")
        raise e
    return data_list


def train_gnn_model(label, train_data_list, val_data_list, mlp_neurons, mlp_dropouts, epochs=300): # Increased default epochs
    """
    (REVISED)
    - Accepts both train and val data lists.
    - Implements ReduceLROnPlateau scheduler based on val_loss.
    - Implements Early Stopping based on val_loss patience.
    """
    print(f"--- Training GNN for label: {label} ---")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not train_data_list:
        print(f"Warning: Empty train data list passed for {label}.")
        return None
    if not val_data_list:
        print(f"Warning: Empty validation data list passed for {label}.")
        return None

    # drop_last=True is important for training stability, prevents variance from tiny final batches.
    train_loader = PyGDataLoader(train_data_list, batch_size=32, shuffle=True, drop_last=True) 
    val_loader = PyGDataLoader(val_data_list, batch_size=32, shuffle=False) # No shuffle/drop for val

    # Get feature dimensions from the first data object
    first_data = train_data_list[0]
    num_node_features = first_data.x.shape[1]
    num_global_features = first_data.u.shape[1]
    num_edge_features = first_data.edge_attr.shape[1]
    
    print(f"Model Features (Scaled): Nodes={num_node_features}, Edges={num_edge_features}, Global={num_global_features}")

    model = TaskSpecificGNN(  # This should be your (no-BN) model class
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_global_features=num_global_features,
        hidden_channels_gnn=128, 
        mlp_neurons=mlp_neurons,
        mlp_dropouts=mlp_dropouts
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = torch.nn.L1Loss() 

    # --- 1. ADD SCHEDULER ---
    # This will cut the LR by half (factor=0.5) if val loss doesn't improve for 10 epochs (patience=10)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # --- 2. ADD EARLY STOPPING VARS ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    PATIENCE_EPOCHS = 30  # Stop training if val loss doesn't improve for 30 straight epochs

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        for data in train_loader:
            if data.x.shape[0] <= 1: # Skip batches with one node (can happen)
                continue
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item() * data.num_graphs
        
        if len(train_loader.dataset) == 0:
            avg_train_loss = 0
        else:
            avg_train_loss = total_train_loss / len(train_loader.dataset)

        # --- 3. ADD VALIDATION LOOP (INSIDE EPOCH LOOP) ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE)
                out = model(data)
                loss = criterion(out, data.y)
                total_val_loss += loss.item() * data.num_graphs
        
        if len(val_loader.dataset) == 0:
             avg_val_loss = 0
        else:
            avg_val_loss = total_val_loss / len(val_loader.dataset)

        if epoch % 10 == 0 or epoch == 1:
             print(f"Epoch: {epoch:03d}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # --- 4. SCHEDULER & EARLY STOPPING LOGIC ---
        scheduler.step(avg_val_loss) # Feed validation loss to the scheduler

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE_EPOCHS and epoch > 50: # Give it at least 50 epochs to warm up
            print(f"--- Early stopping triggered at epoch {epoch} ---")
            break
            
    print(f"--- GNN training for {label} complete. Best Val Loss: {best_val_loss:.6f} ---")
    return model

def predict_with_gnn(trained_model, test_smiles, label, u_scaler, x_scaler, atom_map_len):
    """
    (MODIFIED for Full Scaling)
    - Requires both u_scaler (global) and x_scaler (node) to transform features.
    - Returns SCALED predictions.
    """
    if trained_model is None or u_scaler is None or x_scaler is None:
        print(f"Prediction skipped for {label} due to missing model or scaler.")
        return np.full(len(test_smiles), np.nan)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Featurize test data (features are NOT scaled yet)
    test_data_list = [smiles_to_graph_label_specific(s, label, y_val=None) for s in test_smiles]
    
    valid_indices = [i for i, data in enumerate(test_data_list) if data is not None]
    valid_test_data = [data for data in test_data_list if data is not None]

    if not valid_test_data:
        print(f"Warning: No valid test molecules could be converted for {label}.")
        return np.full(len(test_smiles), np.nan)
        
    # 2. Apply fitted scalers to all valid test features
    try:
        valid_test_data = scale_graph_features(valid_test_data, u_scaler, x_scaler, atom_map_len)
    except Exception as e:
        print(f"CRITICAL ERROR applying scalers during prediction: {e}.")
        return np.full(len(test_smiles), np.nan)

    test_loader = PyGDataLoader(valid_test_data, batch_size=32, shuffle=False) 

    trained_model.eval()
    all_preds = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            out = trained_model(data)
            all_preds.append(out.cpu())

    test_preds_tensor = torch.cat(all_preds, dim=0).numpy().flatten()
    
    # Fill predictions array (these are SCALED predictions)
    final_predictions = np.full(len(test_smiles), np.nan)
    if len(test_preds_tensor) == len(valid_indices):
        final_predictions[valid_indices] = test_preds_tensor
    else:
        print(f"Warning: Mismatch in GNN prediction count for {label}.")
        fill_count = min(len(valid_indices), len(test_preds_tensor))
        final_predictions[valid_indices[:fill_count]] = test_preds_tensor[:fill_count]

    return final_predictions # These predictions are on the SCALED range


def train_or_predict_gnn(train_model=True, model_dir="models/gnn", n_splits=10):
    """
    (FINAL COMPLETE VERSION)
    - All data hardening (coerce, filter) and RobustScaler logic is included.
    - CV loop is modified to create a val_data_list.
    - Calls the new, optimized train_gnn_model with scheduler/early stopping.
    - Correctly passes all arguments (config['neurons'], config['dropouts']) to fix the TypeError.
    """
    
    ATOM_MAP_LEN = 20  # Make sure this matches your global ATOM_MAP
    
    # Plausible physical ranges to filter catastrophic outliers BEFORE scaling
    VALID_RANGES = {
        'Tg':      (-100, 500),  
        'FFV':     (0.01, 0.99), 
        'Tc':      (0, 1000),    
        'Density': (0.1, 3.0),   
        'Rg':      (0.1, 200)    
    }

    # MLP configs for the GNN readout head
    best_configs = {
        # Classic funnel, slightly lower final dropout
        "Tg":      {"neurons": [512, 256, 128], "dropouts": [0.5, 0.4, 0.2]},
        # Original wide funnel for this complex feature
        "Density": {"neurons": [1024, 256, 64], "dropouts": [0.5, 0.4, 0.3]},
        # Even wider and deeper, with strong regularization for presumed complexity
        "FFV":     {"neurons": [1024, 512, 64], "dropouts": [0.6, 0.5, 0.4]},
        # Slightly deeper than the simplest model to capture more features
        "Tc":      {"neurons": [128, 64], "dropouts": [0.4, 0.3]},
        # A gentle funnel instead of a pure block to encourage feature compression
        "Rg":      {"neurons": [128, 64, 64], "dropouts": [0.4, 0.3, 0.3]},
    }
    default_config = {"neurons": [128, 64], "dropouts": [0.3, 0.3]}

    output_df = pd.DataFrame({'id': test_df['id']})
    cv_mae_results = []
    os.makedirs(model_dir, exist_ok=True)
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)

    for label in labels: 
        print(f"\n{'='*20} Processing GNN for label: {label} {'='*20}")
        
        config = best_configs.get(label, default_config)
        print(f"Using MLP Config: Neurons={config['neurons']}, Dropouts={config['dropouts']}")
        
        ensemble_models = []
        y_scaler_path = os.path.join(model_dir, f"gnn_yscaler_{label}.joblib")
        u_scaler_path = os.path.join(model_dir, f"gnn_uscaler_{label}.joblib")
        x_scaler_path = os.path.join(model_dir, f"gnn_xscaler_{label}.joblib")
        
        if train_model:
            # --- START DATA HARDENING ---
            all_smiles_raw = subtables[label]['SMILES']
            all_y_raw = subtables[label][label] 
            
            all_y_numeric = pd.to_numeric(all_y_raw, errors='coerce')
            original_count = len(all_y_numeric)

            valid_min, valid_max = VALID_RANGES.get(label, (-np.inf, np.inf))
            valid_mask = (all_y_numeric >= valid_min) & (all_y_numeric <= valid_max) & (all_y_numeric.notna())
            
            all_y = all_y_numeric[valid_mask].reset_index(drop=True)
            all_smiles = all_smiles_raw[valid_mask].reset_index(drop=True)
            
            print(f"FILTERING: Coerced {original_count} rows. Kept {len(all_y)} valid rows within range ({valid_min}, {valid_max}).")
            
            if len(all_y) < (2 * n_splits): 
                print(f"CRITICAL: Not enough valid data ({len(all_y)}) to train for {label} with {n_splits} splits. Skipping.")
                continue
            # --- END DATA HARDENING ---

            # --- 1. FIT Y-SCALER (ROBUST) ---
            print("Using RobustScaler for Y-Scaler.")
            y_scaler = RobustScaler()  
            all_y_scaled = y_scaler.fit_transform(all_y.values.reshape(-1, 1)).flatten()
            joblib.dump(y_scaler, y_scaler_path)
            print(f"Saved Y-Scaler for {label}")

            # --- 2. FIT INPUT SCALERS (ROBUST) ---
            print("Pre-computing all graph features to fit input scalers...")
            all_train_graphs_raw = [smiles_to_graph_label_specific(s, label, None) for s in all_smiles]
            
            # Sync graph list with all data (skipping any SMILES that fail featurization)
            all_train_graphs_synced = []
            all_y_scaled_synced = [] 
            all_y_original_synced = [] # Also sync original Y for the CV split
            all_smiles_synced = []     # Also sync SMILES for the CV split
            
            for i, graph in enumerate(all_train_graphs_raw):
                if graph is not None:
                    all_train_graphs_synced.append(graph)
                    all_y_scaled_synced.append(all_y_scaled[i]) 
                    all_y_original_synced.append(all_y[i]) # Keep the original, unscaled, clean Y
                    all_smiles_synced.append(all_smiles[i]) # Keep the matching SMILES
            
            all_train_graphs = all_train_graphs_synced 
            all_y_scaled = np.array(all_y_scaled_synced)
            all_y_original_df = pd.Series(all_y_original_synced) # Store as Series for .iloc
            all_smiles_df = pd.Series(all_smiles_synced)         # Store as Series for .iloc

            if not all_train_graphs:
                print(f"CRITICAL: No valid training graphs could be featurized for {label}. Skipping.")
                continue
                
            all_u_data = np.concatenate([d.u.numpy() for d in all_train_graphs], axis=0)
            print("Using RobustScaler for U-Scaler.")
            u_scaler = RobustScaler().fit(all_u_data)  # Use RobustScaler
            joblib.dump(u_scaler, u_scaler_path)
            print(f"Saved U-Scaler for {label}")

            all_x_data = torch.cat([d.x for d in all_train_graphs], dim=0)
            all_x_continuous = all_x_data[:, ATOM_MAP_LEN:].numpy()
            print("Using RobustScaler for X-Scaler.")
            x_scaler = RobustScaler().fit(all_x_continuous)  # Use RobustScaler
            joblib.dump(x_scaler, x_scaler_path)
            print(f"Saved X-Scaler for {label}")

            # --- 3. APPLY SCALERS ---
            all_data_objects_scaled = scale_graph_features(all_train_graphs, u_scaler, x_scaler, ATOM_MAP_LEN)
            for i, data_obj in enumerate(all_data_objects_scaled):
                data_obj.y = torch.tensor([[all_y_scaled[i]]], dtype=torch.float)
            
            # --- 4. K-FOLD CV LOOP (MODIFIED) ---
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_val_scores = []
            fold_indices_gen = kf.split(all_data_objects_scaled) # Split the synced, valid, scaled data

            for fold, (train_idx, val_idx) in enumerate(fold_indices_gen):
                print(f"\n--- Fold {fold+1}/{n_splits} for {label} ---")
                
                train_data_list = [all_data_objects_scaled[i] for i in train_idx]
                val_data_list = [all_data_objects_scaled[i] for i in val_idx] # <-- CREATE VAL LIST
                
                val_smiles_list = all_smiles_df.iloc[val_idx].tolist()
                y_val_original = all_y_original_df.iloc[val_idx].values 

                fold_model = train_gnn_model(
                    label,
                    train_data_list, # Pass train data
                    val_data_list,   # <-- Pass val data
                    config['neurons'],    # <-- PASSES mlp_neurons
                    config['dropouts'],   # <-- PASSES mlp_dropouts (FIXES ERROR)
                    epochs=300       # <-- Train longer (will stop early)
                )
                
                if fold_model:
                    print("Running final validation prediction on the best model...")
                    val_preds_scaled = predict_with_gnn(fold_model, val_smiles_list, label, u_scaler, x_scaler, ATOM_MAP_LEN)
                    
                    train_y_scaled_median = 0.0 # RobustScaler median is 0
                    val_preds_scaled_filled = pd.Series(val_preds_scaled).fillna(train_y_scaled_median)
                    
                    val_preds_original = y_scaler.inverse_transform(
                        val_preds_scaled_filled.values.reshape(-1, 1)
                    ).flatten()

                    mae = mean_absolute_error(y_val_original, val_preds_original)
                    print(f"âœ… Fold {fold+1} Validation MAE (Original Scale): {mae:.4f}")
                    fold_val_scores.append(mae)
                    
                    model_save_name = f"{label}_fold{fold}"
                    save_gnn_model(fold_model, model_save_name, model_dir)
                    ensemble_models.append(fold_model)
                else:
                    print(f"Warning: Training failed for Fold {fold+1}. Model will be skipped.")
            
            if fold_val_scores:
                avg_cv_mae = np.mean(fold_val_scores)
                print(f"\n{'*'*10} Average CV MAE for {label} (Original Scale): {avg_cv_mae:.4f} {'*'*10}")
                cv_mae_results.append({'label': label, 'avg_cv_mae': avg_cv_mae})

        else:
            # --- PREDICTION-ONLY MODE ---
            print(f"Loading {n_splits} models and ALL 3 RobustScalers for {label} ensemble...")
            model_path = '/kaggle/input/neurips-2025/GATConv_v29/models/gnn/'
            try:
                y_scaler = joblib.load(f'{model_path}gnn_yscaler_{label}.joblib')
                u_scaler = joblib.load(f'{model_path}gnn_uscaler_{label}.joblib')
                x_scaler = joblib.load(f'{model_path}gnn_xscaler_{label}.joblib')
                print("Loaded Y, U, and X RobustScalers.")
            except FileNotFoundError:
                print(f"CRITICAL: Scaler files not found for {label}. Cannot make predictions.")
                continue

            for fold in range(n_splits):
                loaded_model = load_gnn_model(f"{label}_fold{fold}", model_path.rstrip('/'))
                if loaded_model:
                    ensemble_models.append(loaded_model)
            
            if not ensemble_models: print(f"Warning: No models found for label {label}.")
            else: print(f"Successfully loaded {len(ensemble_models)} models for ensemble.")


        # --- ENSEMBLE PREDICTION STEP (Test Set) ---
        test_smiles = test_df['SMILES'].tolist()
        
        if ensemble_models and y_scaler and u_scaler and x_scaler:
            print(f"Making ensemble (scaled) predictions for {label} using {len(ensemble_models)} models...")
            all_fold_preds_scaled = []
            for model in ensemble_models:
                fold_test_preds_scaled = predict_with_gnn(model, test_smiles, label, u_scaler, x_scaler, ATOM_MAP_LEN)
                all_fold_preds_scaled.append(fold_test_preds_scaled)
            
            preds_stack_scaled = np.stack(all_fold_preds_scaled)
            final_ensemble_preds_scaled = np.nanmean(preds_stack_scaled, axis=0) 
            pred_series_scaled = pd.Series(final_ensemble_preds_scaled)
            
            pred_series_scaled_filled = pred_series_scaled.fillna(0.0) # Impute with scaled median (0.0)

            final_preds_original = y_scaler.inverse_transform(
                pred_series_scaled_filled.values.reshape(-1, 1)
            ).flatten()
            
            output_df[label] = final_preds_original
            
        else:
            print(f"No models or scalers available for {label}. Filling with (filtered) training median.")
            # Robust median fallback logic
            fallback_median = 0.0
            try:
                if 'all_y' in locals() and not all_y.empty:
                     fallback_median = all_y.median()
                else: 
                     print("Loading data to calculate fallback median...")
                     fb_y_raw = subtables[label][label]
                     fb_y_num = pd.to_numeric(fb_y_raw, errors='coerce')
                     valid_min, valid_max = VALID_RANGES.get(label, (-np.inf, np.inf))
                     fb_mask = (fb_y_num >= valid_min) & (fb_y_num <= valid_max) & (fb_y_num.notna())
                     fallback_median = fb_y_num[fb_mask].median()
                print(f"Using filtered median fallback: {fallback_median}")
            except Exception as e:
                 print(f"Error getting median, falling back to 0: {e}")
                 fallback_median = 0.0 
                 
            output_df[label] = fallback_median

    # --- Display final CV MAE summary ---
    if train_model and cv_mae_results:
        print("\n" + "="*40)
        print("ğŸ“Š HYBRID GNN 5-Fold CV MAE Summary (Original Scale):")
        print("="*40)
        mae_df = pd.DataFrame(cv_mae_results)
        print(mae_df.to_string(index=False))
        mae_df.to_csv("gnn_hybrid_cv_mae_results.csv", index=False)
        print("\nCV results saved to gnn_hybrid_cv_mae_results.csv")

    submission_path = 'submission_hybrid_gnn_final.csv'
    output_df.to_csv(submission_path, index=False)
    print(f"\nâœ… GNN Ensemble predictions (Original Scale) saved to {submission_path}")
    
    warnings.filterwarnings("default", "Mean of empty slice", RuntimeWarning)
    
    return output_df

# To train the models and then predict:
gnn_submission_df = train_or_predict_gnn(train_model=False)

output_dfs.append(gnn_submission_df)

print("\nGNN Submission Preview:")
print(gnn_submission_df.head())


len(output_dfs)


# Average predictions from all output DataFrames
final_df = pd.concat(output_dfs, axis=0).groupby('id').mean().reset_index()
final_df.to_csv('submission.csv', index=False)
final_df

