!pip install /kaggle/input/rdkit-2025-3-3-cp311-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Install RDKit if needed
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Libraries imported successfully!")



import pandas as pd
base_dir = ""
def make_smile_canonical(smile): # To avoid duplicates, for example: canonical '*C=C(*)C' == '*C(=C*)C'
    try:
        mol = Chem.MolFromSmiles(smile)
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except:
        return np.nan
def add_tc(df, base_dir=base_dir, smiles_canon_fn=lambda s: make_smile_canonical(s)):
    """
    Adds extra data for Tc, Tg, and Density from external sources to the input DataFrame.
    Automatically fills missing values for known SMILES and adds new rows for unseen ones.

    Parameters:
    - df: original DataFrame with columns ['id','SMILES','Tg','FFV','Tc','Density','Rg']
    - base_dir: path prefix for Kaggle input folders
    - smiles_canon_fn: function to canonicalize SMILES strings

    Returns:
    - updated DataFrame with additional rows and filled values
    """
    df = df.copy()

    # ─────────────────────────────────────────────────────────────────────
    # Load and process extra Tc data
    # ─────────────────────────────────────────────────────────────────────
    extra_tc_df = pd.read_csv(base_dir + '/kaggle/input/tc-smiles/Tc_SMILES.csv')
    extra_tc_df = extra_tc_df[['SMILES', 'TC_mean']].rename(columns={'TC_mean': 'Tc'})
    extra_tc_df['SMILES'] = extra_tc_df['SMILES'].apply(smiles_canon_fn)
    extra_tc_df = extra_tc_df.dropna(subset=['SMILES', 'Tc'])

    # ─────────────────────────────────────────────────────────────────────
    # Load and process extra Tg data (two sources)
    # ─────────────────────────────────────────────────────────────────────
    data_tg2 = pd.read_csv(base_dir+'/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv',
                           usecols=['SMILES', 'Tg (C)'])
    data_tg2 = data_tg2.rename(columns={'Tg (C)': 'Tg'})
    data_tg2['SMILES'] = data_tg2['SMILES'].apply(smiles_canon_fn)
    data_tg2 = data_tg2.dropna(subset=['SMILES', 'Tg'])

    data_tg3 = pd.read_excel(base_dir+'/kaggle/input/smiles-extra-data/data_tg3.xlsx')
    data_tg3 = data_tg3.rename(columns={'Tg [K]': 'Tg'})
    data_tg3['Tg'] = data_tg3['Tg'] - 273.15  # convert Kelvin to Celsius
    data_tg3['SMILES'] = data_tg3['SMILES'].apply(smiles_canon_fn)
    data_tg3 = data_tg3.dropna(subset=['SMILES', 'Tg'])

    extra_tg_df = pd.concat([data_tg2, data_tg3], ignore_index=True)
    extra_tg_df = extra_tg_df.groupby('SMILES', as_index=False)['Tg'].mean()

    # ─────────────────────────────────────────────────────────────────────
    # Load and process Density data
    # ─────────────────────────────────────────────────────────────────────
    data_dnst = pd.read_excel(base_dir+'/kaggle/input/smiles-extra-data/data_dnst1.xlsx')
    data_dnst = data_dnst.rename(columns={'density(g/cm3)': 'Density'})
    data_dnst = data_dnst[['SMILES', 'Density']]
    data_dnst['SMILES'] = data_dnst['SMILES'].apply(smiles_canon_fn)
    data_dnst = data_dnst.dropna(subset=['SMILES', 'Density'])
    data_dnst = data_dnst[data_dnst['Density'] != 'nylon']
    data_dnst['Density'] = data_dnst['Density'].astype('float64')
    data_dnst['Density'] -= 0.118  # calibration offset

    # ─────────────────────────────────────────────────────────────────────
    # Merge all extra data into a single frame
    # ─────────────────────────────────────────────────────────────────────
    df_extra = pd.merge(extra_tc_df, extra_tg_df, on='SMILES', how='outer')
    df_extra = pd.merge(df_extra, data_dnst, on='SMILES', how='outer')
    df_extra = df_extra.dropna(subset=['Tc', 'Tg', 'Density'], how='all')
    df_extra = df_extra.groupby('SMILES', as_index=False).mean()

    # ─────────────────────────────────────────────────────────────────────
    # Determine overlaps and new entries
    # ─────────────────────────────────────────────────────────────────────
    df_extra['SMILES'] = df_extra['SMILES'].apply(smiles_canon_fn)
    df['SMILES'] = df['SMILES'].apply(smiles_canon_fn)

    existing_smiles = set(df['SMILES'])
    extra_smiles = set(df_extra['SMILES'])

    cross_smiles = existing_smiles & extra_smiles
    new_smiles = extra_smiles - existing_smiles

    # ─────────────────────────────────────────────────────────────────────
    # Fill missing targets in df for overlapping SMILES
    # ─────────────────────────────────────────────────────────────────────
    for target in ['Tc', 'Tg', 'Density']:
        for smile in cross_smiles:
            value = df_extra.loc[df_extra['SMILES'] == smile, target].values[0]
            mask = (df['SMILES'] == smile) & (df[target].isna())
            df.loc[mask, target] = value

    # ─────────────────────────────────────────────────────────────────────
    # Append new SMILES rows with default NaNs for missing columns
    # ─────────────────────────────────────────────────────────────────────
    new_rows = df_extra[df_extra['SMILES'].isin(new_smiles)].copy()
    new_rows = new_rows.reset_index(drop=True)
    new_rows['id'] = range(len(df), len(df) + len(new_rows))
    for col in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
        if col not in new_rows.columns:
            new_rows[col] = pd.NA
    new_rows = new_rows[['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]
    df = pd.concat([df, new_rows], ignore_index=True)

    # ─────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────
    print(f'\nAdded {len(new_smiles)} new unique SMILES!')
    for target in ['Tc', 'Tg', 'Density']:
        n_added = df[target].notnull().sum()
        print(f'→ Total samples with {target}: {n_added}')

    return df



# Load data and define constants

train = pd.read_csv(base_dir + '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv(base_dir + '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

train['SMILES'] = train['SMILES'].apply(lambda s: make_smile_canonical(s))
test['SMILES'] = test['SMILES'].apply(lambda s: make_smile_canonical(s))

train = add_tc(train)

targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
epochs = 50

# Training configuration
MODE = "train"  # "train" or "infer"
MODEL_DIR = base_dir + "/kaggle/working/hybrid_models"  # Directory to save/load models
import os
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Metrics constants (Official competition metrics)
MINMAX_DICT = {
    'Tg': [-148.0297376, 472.25],
    'FFV': [0.2269924, 0.77709707],
    'Tc': [0.0465, 0.524],
    'Density': [0.748691234, 1.840998909],
    'Rg': [9.7283551, 34.672905605],
}
NULL_FOR_SUBMISSION = -9999

def scaling_error(labels, preds, property):
    error = np.abs(labels - preds)
    min_val, max_val = MINMAX_DICT[property]
    label_range = max_val - min_val
    return np.mean(error / label_range)

def get_property_weights(labels):
    """Calculate property weights based on sample counts (official competition logic)"""
    property_weight = []
    for property in MINMAX_DICT.keys():
        valid_num = np.sum(labels[property] != NULL_FOR_SUBMISSION)
        property_weight.append(valid_num)
    property_weight = np.array(property_weight)
    property_weight = np.sqrt(1 / property_weight)
    return (property_weight / np.sum(property_weight)) * len(property_weight)

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = "id") -> float:
    """
    Official competition scoring function: Compute weighted Mean Absolute Error (wMAE)
    """
    chemical_properties = list(MINMAX_DICT.keys())
    property_maes = []
    property_weights = get_property_weights(solution[chemical_properties])
    print(property_weights)
    for property in chemical_properties:
        is_labeled = solution[property] != NULL_FOR_SUBMISSION
        if is_labeled.sum() > 0:  # Only calculate if there are valid labels
            property_maes.append(scaling_error(
                solution.loc[is_labeled, property], 
                submission.loc[is_labeled, property], 
                property
            ))
        else:
            property_maes.append(0.0)  # No valid samples for this property
    print(property_maes)

    if len(property_maes) == 0:
        raise RuntimeError('No labels')
    return float(np.average(property_maes, weights=property_weights))

print(f"\nTarget availability:")
for target in targets:
    count = train[target].notna().sum()
    pct = count / len(train) * 100
    print(f"{target}: {count:,} samples ({pct:.1f}%)")

print(f"\nSample SMILES:")
for i in range(3):
    smiles = train.iloc[i]['SMILES']
    print(f"{i+1}. {smiles[:80]}...")



from rdkit.Chem import rdmolops
import networkx as nx
from rdkit.Chem import Descriptors
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import re


# Columns to exclude from final feature set
useless_cols = [
    'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW',
    'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'NumRadicalElectrons', 'SMR_VSA8', 'SlogP_VSA9', 'fr_barbitur',
    'fr_benzodiazepine', 'fr_dihydropyridine', 'fr_epoxide', 'fr_isothiocyan', 'fr_lactam', 'fr_nitroso',
    'fr_prisulfonamd', 'fr_thiocyan', 'MaxEStateIndex', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
    'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Kappa1', 'LabuteASA', 'HeavyAtomCount',
    'MolMR', 'Chi3n', 'BertzCT', 'Chi2v', 'Chi4n', 'HallKierAlpha', 'Chi3v', 'Chi4v', 'MinAbsPartialCharge',
    'MinPartialCharge', 'MaxAbsPartialCharge', 'FpDensityMorgan2', 'FpDensityMorgan3', 'Phi', 'Kappa3',
    'fr_nitrile', 'SlogP_VSA6', 'NumAromaticCarbocycles', 'NumAromaticRings', 'fr_benzene', 'VSA_EState6',
    'NOCount', 'fr_C_O', 'fr_C_O_noCOO', 'NumHDonors', 'fr_amide', 'fr_Nhpyrrole', 'fr_phenol',
    'fr_phenol_noOrthoHbond', 'fr_COO2', 'fr_halogen', 'fr_diazo', 'fr_nitro_arom', 'fr_phos_ester'
]

# Precompute descriptor names
DESC_NAMES = [desc[0] for desc in Descriptors.descList if desc[0] not in useless_cols]

# Compute 1D descriptors
def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * len(DESC_NAMES)
    return [getattr(Descriptors, name)(mol) for name in DESC_NAMES]

# Compute graph-based features
def compute_graph_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan, np.nan, np.nan]
    try:
        adj = rdmolops.GetAdjacencyMatrix(mol)
        G = nx.from_numpy_array(adj)
        if nx.is_connected(G):
            dia = nx.diameter(G)
            sp = nx.average_shortest_path_length(G)
        else:
            dia, sp = 0, 0
        num_cycles = len(list(nx.cycle_basis(G)))
        return [dia, sp, num_cycles]
    except Exception:
        return [np.nan, np.nan, np.nan]

# Full featurization function
def featurize_smiles_with_graph(df):
    smiles_list = df['SMILES'].tolist()

    print("Extracting molecular descriptors and graph features...")
    desc_rows = Parallel(n_jobs=-1)(delayed(compute_all_descriptors)(smi) for smi in tqdm(smiles_list))
    graph_rows = Parallel(n_jobs=-1)(delayed(compute_graph_features)(smi) for smi in tqdm(smiles_list))

    desc_df = pd.DataFrame(desc_rows, columns=DESC_NAMES)
    graph_df = pd.DataFrame(graph_rows, columns=['graph_diameter', 'avg_shortest_path', 'num_cycles'])

    result = pd.concat([desc_df, graph_df], axis=1)
    result = result.replace([np.inf, -np.inf], np.nan)

    return result

print("Extracting molecular descriptors…")

train_desc_df = featurize_smiles_with_graph(train)
test_desc_df  = featurize_smiles_with_graph(test)
features = train_desc_df.columns
train_full = pd.concat([train.reset_index(drop=True),
                        train_desc_df.reset_index(drop=True)], axis=1)
test_full  = pd.concat([test.reset_index(drop=True),
                        test_desc_df.reset_index(drop=True)],  axis=1)

print(f"Extracted {len(features)} descriptors; ")
print("Train shape:", train_full.shape)
print("Test  shape:", test_full.shape)



def compute_medians(df_num: pd.DataFrame) -> pd.Series:

    med = df_num.replace([np.inf, -np.inf], np.nan).median()
    return med.fillna(0.0)

def clean_features(df: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    cleaned = df.copy()
    num_cols = cleaned.select_dtypes(include=[np.number]).columns

    # Count NaNs and infs before cleaning
    report = {
        "total_numerical_columns": len(num_cols),
        "nan_before": cleaned[num_cols].isna().sum().sum(),
        "inf_before": np.isinf(cleaned[num_cols]).sum().sum(),
    }

    # Perform cleaning
    cleaned[num_cols] = (
        cleaned[num_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(medians[num_cols])
    )

    # Count remaining
    report["nan_after"] = cleaned[num_cols].isna().sum().sum()
    report["inf_after"] = np.isinf(cleaned[num_cols]).sum().sum()
    display(report)
    return cleaned

train_num = train_full[features].select_dtypes(include=[np.number])
medians = compute_medians(train_num)

train_full[features] = clean_features(train_full[features], medians)
test_full[features]  = clean_features(test_full[features],  medians)


XGB_params= {"Tg":{'n_estimators': 6897, 'learning_rate': 0.010131894286581853, 'max_depth': 8, 'subsample': 0.7165086480925237, 'colsample_bytree': 0.8447773274458091, 'gamma': 0.00025415678607289814, 'reg_alpha': 1.9726346720527255e-08, 'reg_lambda': 5.422882478150009e-05, 'min_child_weight': 1},

"FFV":{'n_estimators': 3142, 'learning_rate': 0.02510493042061551, 'max_depth': 7, 'subsample': 0.6004615548625559, 'colsample_bytree': 0.8161487242506853, 'gamma': 1.0056288626374738e-07, 'reg_alpha': 0.02961381419142807, 'reg_lambda': 2.8406654285701748e-05, 'min_child_weight': 1},

"Tc":{'n_estimators': 3905, 'learning_rate': 0.03832624981509266, 'max_depth': 7, 'subsample': 0.6927712931932059, 'colsample_bytree': 0.7666942724475903, 'gamma': 0.00017859161335665862, 'reg_alpha': 0.0001196126376415241, 'reg_lambda': 1.7086492695916624e-06, 'min_child_weight': 7},

"Density":{'n_estimators': 6000, 'learning_rate': 0.03858095080716189, 'max_depth': 9, 'subsample': 0.6023345124755703, 'colsample_bytree': 0.7722519503306169, 'gamma': 3.647044197455384e-07, 'reg_alpha': 0.015338593021609734, 'reg_lambda': 0.2036662630778741, 'min_child_weight': 4},

"Rg":{'n_estimators': 5045, 'learning_rate': 0.03843820258005981, 'max_depth': 12, 'subsample': 0.6512356217265035, 'colsample_bytree': 0.6966789697706135, 'gamma': 0.009780097850843522, 'reg_alpha': 0.24280831082898113, 'reg_lambda': 6.287743088973779e-05, 'min_child_weight': 8}}


def to_numpy_for_xgb(df: pd.DataFrame) -> np.ndarray:

    arr = df.to_numpy(dtype=np.float32, copy=False)

    # 将 ±inf / NaN → 0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # 裁剪极端异常值（> 1e6 举例，可按需放宽）
    np.clip(arr, -1e6, 1e6, out=arr)

    return arr



import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================

def train_target_model(
    train_df,
    test_df,
    target,
    feature_cols,
    n_splits: int = 5,
    include_ffv_pred: bool = False,
    ffv_oof=None,
    ffv_test=None,
):
    """
    Train ensemble model (LightGBM + CatBoost + XGBoost) for a single target
    with proper CV calculation.  Optionally include FFV predictions as features.
    """
    print(f"\n{'='*50}")
    print(f"Training ensemble model for: {target}")
    if include_ffv_pred:
        print(f"Including FFV predictions as features")
    print(f"{'='*50}")

    # ---------------------------

    target_data = train_df[train_df[target].notna()].copy()
    if len(target_data) < 50:
        print(f"Insufficient data for {target}: {len(target_data)} samples")
        return None, None, 0.0, None

    print(f"Training samples: {len(target_data):,}")

    # ---------------------------

    current_feature_cols = feature_cols.copy()

    if include_ffv_pred and ffv_oof is not None and ffv_test is not None:
        train_indices = target_data.index
        target_data = target_data.copy()
        target_data["FFV_pred"] = ffv_oof[train_indices]

        test_df_enhanced = test_df.copy()
        test_df_enhanced["FFV_pred"] = ffv_test

        current_feature_cols += ["FFV_pred"]
        print(f"Added FFV predictions as feature. Total features: {len(current_feature_cols)}")
    else:
        test_df_enhanced = test_df

    X = target_data[current_feature_cols].fillna(0)
    y = target_data[target]
    X_test = test_df_enhanced[current_feature_cols].fillna(0)

    # ---------------------------

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds_lgb = np.zeros(len(X))
    oof_preds_cat = np.zeros(len(X))
    oof_preds_xgb = np.zeros(len(X))
    oof_preds_ensemble = np.zeros(len(X))
    oof_true_label = np.zeros(len(y))

    test_preds_lgb = np.zeros(len(X_test))
    test_preds_cat = np.zeros(len(X_test))
    test_preds_xgb = np.zeros(len(X_test))
    test_preds_ensemble = np.zeros(len(X_test))

    cv_scores_lgb, cv_wmae_scores_lgb = [], []
    cv_scores_cat, cv_wmae_scores_cat = [], []
    cv_scores_xgb, cv_wmae_scores_xgb = [], []
    cv_scores_ensemble, cv_wmae_scores_ensemble = [], []

    # ---------------------------

    lgb_params = {
        'n_estimators': 1_000_000,
        'objective': 'regression_l1',
        'metric': 'mae',
        'verbosity': -1,
        
        'num_leaves': 50,
        'min_data_in_leaf': 2,
        'learning_rate': 0.01,
        'max_bin': 500,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'lambda_l1': 2,
        'lambda_l2': 2,
    }

    cat_params = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "learning_rate": 0.01,
        "max_depth": 7,
        "l2_leaf_reg": 0.1,
        "random_seed": 42,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.9,
        "rsm": 0.8,
        "verbose": 0,
        "iterations": 1_000_000,
        "task_type": "CPU",
    }

    xgb_params = XGB_params[target]

    lgb_models, cat_models, xgb_models = [], [], []


    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"  Fold {fold + 1}/{n_splits}...", end=" ")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        oof_true_label[val_idx] = y_val.values

        # -------- LightGBM --------
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        lgb_model = lgb.train(
            lgb_params,
            lgb_train,
            valid_sets=[lgb_val],
            # num_boost_round=1_000_000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0),
            ],
        )

        # -------- CatBoost --------
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False,
        )

        # -------- XGBoost --------
        X_train_xgb = to_numpy_for_xgb(X_train)
        X_val_xgb   = to_numpy_for_xgb(X_val)
        xgb_model = XGBRegressor(**xgb_params, objective="reg:squarederror", eval_metric="rmse",)
        xgb_model.fit(
            X_train_xgb,
            y_train,
            eval_set=[(X_val_xgb, y_val)],
            early_stopping_rounds=50,
            # verbose=False, 

            verbose=1000
        )

        if target == "FFV":
            lgb_models.append(lgb_model)
            cat_models.append(cat_model)
            xgb_models.append(xgb_model)

        # ---------- Predict ----------
        lgb_val_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        cat_val_pred = cat_model.predict(X_val)
        xgb_val_pred = xgb_model.predict(X_val_xgb)
    
        oof_preds_lgb[val_idx] = lgb_val_pred
        oof_preds_cat[val_idx] = cat_val_pred
        oof_preds_xgb[val_idx] = xgb_val_pred
    
        # ---------- Compute adaptive ensemble weights ----------
        fold_wmae_lgb = scaling_error(y_val.values, lgb_val_pred, target)
        fold_wmae_cat = scaling_error(y_val.values, cat_val_pred, target)
        fold_wmae_xgb = scaling_error(y_val.values, xgb_val_pred, target)
    
        wmae_scores = {
            "lgb": fold_wmae_lgb,
            "cat": fold_wmae_cat,
            "xgb": fold_wmae_xgb
        }
        sorted_models = sorted(wmae_scores, key=wmae_scores.get)
        weights = {model: 0.25 for model in wmae_scores}
        weights[sorted_models[0]] = 0.5  # best gets 0.5
    
        # ---------- Ensemble prediction ----------
        ensemble_val_pred = (
            weights["lgb"] * lgb_val_pred +
            weights["cat"] * cat_val_pred +
            weights["xgb"] * xgb_val_pred
        )
        oof_preds_ensemble[val_idx] = ensemble_val_pred
    
        # ---------- Test prediction ----------
        lgb_test_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
        cat_test_pred = cat_model.predict(X_test)
        X_test_xgb = to_numpy_for_xgb(X_test)
        xgb_test_pred = xgb_model.predict(X_test_xgb)
    
        test_preds_lgb += lgb_test_pred / n_splits
        test_preds_cat += cat_test_pred / n_splits
        test_preds_xgb += xgb_test_pred / n_splits
        test_preds_ensemble += (
            weights["lgb"] * lgb_test_pred +
            weights["cat"] * cat_test_pred +
            weights["xgb"] * xgb_test_pred
        ) / n_splits
    
        # ---------- Metrics ----------
        fold_mae_lgb = mean_absolute_error(y_val, lgb_val_pred)
        fold_mae_cat = mean_absolute_error(y_val, cat_val_pred)
        fold_mae_xgb = mean_absolute_error(y_val, xgb_val_pred)
        fold_mae_ens = mean_absolute_error(y_val, ensemble_val_pred)
    
        cv_scores_lgb.append(fold_mae_lgb)
        cv_scores_cat.append(fold_mae_cat)
        cv_scores_xgb.append(fold_mae_xgb)
        cv_scores_ensemble.append(fold_mae_ens)
    
        fold_wmae_ens = scaling_error(y_val.values, ensemble_val_pred, target)
        cv_wmae_scores_lgb.append(fold_wmae_lgb)
        cv_wmae_scores_cat.append(fold_wmae_cat)
        cv_wmae_scores_xgb.append(fold_wmae_xgb)
        cv_wmae_scores_ensemble.append(fold_wmae_ens)
    
        print(
            f"wMAE - LGB: {fold_wmae_lgb:.4f} | CAT: {fold_wmae_cat:.4f} "
            f"| XGB: {fold_wmae_xgb:.4f} | ENS: {fold_wmae_ens:.4f}"
        )


    # ====================================================
    ori_oof_preds_ensemble = oof_preds_ensemble.copy()
    if target == "FFV":
        missing_ffv_mask = train_df["FFV"].isna()
        if missing_ffv_mask.any():
            missing_X = train_df[missing_ffv_mask][current_feature_cols].fillna(0)

            missing_preds_lgb = np.column_stack([m.predict(missing_X) for m in lgb_models]).mean(axis=1)
            missing_preds_cat = np.column_stack([m.predict(missing_X) for m in cat_models]).mean(axis=1)
            missing_preds_xgb = np.column_stack([m.predict(missing_X) for m in xgb_models]).mean(axis=1)

            missing_preds_ensemble = (missing_preds_lgb + missing_preds_cat + missing_preds_xgb) / 3

            oof_preds_ensemble_full = np.zeros(len(train_df))
            oof_preds_ensemble_full[target_data.index] = oof_preds_ensemble
            oof_preds_ensemble_full[missing_ffv_mask] = missing_preds_ensemble
            oof_preds_ensemble = oof_preds_ensemble_full

    # ====================================================
    property_weights = get_property_weights(train_df[targets])
    target_weight = property_weights[targets.index(target)]

    # ---- LightGBM
    overall_mae_lgb = mean_absolute_error(y, oof_preds_lgb)
    overall_wmae_lgb = scaling_error(y.values, oof_preds_lgb, target) * target_weight
    print(f"LightGBM CV MAE: {overall_mae_lgb:.4f} (±{np.std(cv_scores_lgb):.4f})")
    print(f"LightGBM CV wMAE: {overall_wmae_lgb:.4f} (±{np.std(cv_wmae_scores_lgb):.4f})")

    # ---- CatBoost
    overall_mae_cat = mean_absolute_error(y, oof_preds_cat)
    overall_wmae_cat = scaling_error(y.values, oof_preds_cat, target) * target_weight
    print(f"CatBoost CV MAE: {overall_mae_cat:.4f} (±{np.std(cv_scores_cat):.4f})")
    print(f"CatBoost CV wMAE: {overall_wmae_cat:.4f} (±{np.std(cv_wmae_scores_cat):.4f})")

    # ---- XGBoost
    overall_mae_xgb = mean_absolute_error(y, oof_preds_xgb)
    overall_wmae_xgb = scaling_error(y.values, oof_preds_xgb, target) * target_weight
    print(f"XGBoost CV MAE: {overall_mae_xgb:.4f} (±{np.std(cv_scores_xgb):.4f})")
    print(f"XGBoost CV wMAE: {overall_wmae_xgb:.4f} (±{np.std(cv_wmae_scores_xgb):.4f})")

    # ---- Ensemble
    if target == "FFV":
        overall_mae_ensemble = mean_absolute_error(y, ori_oof_preds_ensemble)
        overall_wmae_ensemble = scaling_error(y.values, ori_oof_preds_ensemble, target) * target_weight
    else:
        overall_mae_ensemble = mean_absolute_error(y, oof_preds_ensemble)
        overall_wmae_ensemble = scaling_error(y.values, oof_preds_ensemble, target) * target_weight

    print(f"Ensemble CV MAE: {overall_mae_ensemble:.4f} (±{np.std(cv_scores_ensemble):.4f})")
    print(f"Ensemble CV wMAE: {overall_wmae_ensemble:.4f} (±{np.std(cv_wmae_scores_ensemble):.4f})")

    if target != "FFV":
        return oof_preds_ensemble, test_preds_ensemble, overall_wmae_ensemble, oof_true_label
    else:
        return (
            oof_preds_ensemble,
            test_preds_ensemble,
            overall_wmae_ensemble,
            oof_true_label,
            ori_oof_preds_ensemble,
        )



# Get feature columns (same as original)
feature_cols = [col for col in features if col not in ['id', 'SMILES'] + targets]
print(f"Using {len(feature_cols)} features")

# Step 1: First train FFV model (highest data availability)
print("\n" + "="*60)
print("STEP 1: Training FFV model first")
print("="*60)

ffv_oof, ffv_test_pred, ffv_cv_score, ffv_true, ori_oof_preds_ensemble = train_target_model(
    train_full, test_full, 'FFV', feature_cols
)

print(f"\nFFV model complete. CV score: {ffv_cv_score:.4f}")
print(f"FFV OOF shape: {ffv_oof.shape}")
print(f"FFV test predictions shape: {ffv_test_pred.shape}")

# Step 2: Train other models using FFV predictions as features
print("\n" + "="*60)
print("STEP 2: Training other models with FFV predictions as features")
print("="*60)

results = {'FFV': {
    'oof_predictions': ori_oof_preds_ensemble,
    'oof_true_label':  ffv_true,
    'test_predictions': ffv_test_pred,
    'cv_score': ffv_cv_score
}}

test_predictions = {'FFV': ffv_test_pred}
cv_summary = {'FFV': ffv_cv_score}

# Train remaining targets with FFV predictions as features
remaining_targets = [target for target in targets if target != 'FFV']

for target in remaining_targets:
    if target == "Tg":
        include_ffv_pred = False
    else:
        include_ffv_pred = True
    oof_preds, test_preds, cv_score, oof_true = train_target_model(
        train_full, test_full, target, feature_cols, 
        include_ffv_pred=include_ffv_pred, ffv_oof=ffv_oof, ffv_test=ffv_test_pred
    )
    
    results[target] = {
        'oof_predictions': oof_preds,
        'oof_true_label':  oof_true,
        'test_predictions': test_preds,
        'cv_score': cv_score
    }
    
    if test_preds is not None:
        test_predictions[target] = test_preds
        cv_summary[target] = cv_score
    else:
        # Fallback to mean for targets with insufficient data
        test_predictions[target] = np.full(len(test), train[target].mean() if train[target].notna().sum() > 0 else 0)
        cv_summary[target] = None

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
for target in targets:
    if cv_summary[target] is not None:
        print(f"{target:>8s}: {cv_summary[target]:.4f} wMAE")
    else:
        print(f"{target:>8s}: No model (using mean)")
        
# Calculate average CV score for trained models
trained_scores = [score for score in cv_summary.values() if score is not None]
avg_cv = np.mean(trained_scores)
print(f"\nAverage CV wMAE: {avg_cv:.4f}")



import numpy as np
import pandas as pd
import open_polymer_2025_metric as metric


all_cv_true        = {}
all_cv_predictions = {}
cv_scores          = []

for target in targets:
    oof_true = results[target]['oof_true_label']
    oof_pred = results[target]['oof_predictions']
    
    all_cv_true[target]        = np.asarray(oof_true, dtype=float)
    all_cv_predictions[target] = np.asarray(oof_pred, dtype=float)
    print(target)
    cv_score = metric.scaling_error(all_cv_true[target], all_cv_predictions[target], target)
    cv_scores.append(cv_score)


max_len = max(len(v) for v in all_cv_true.values())
NULL = metric.NULL_FOR_SUBMISSION

cv_true_df = pd.DataFrame({'id': range(max_len)})
cv_pred_df = pd.DataFrame({'id': range(max_len)})

for target in targets:
    true_pad = list(all_cv_true[target]) + [NULL]*(max_len - len(all_cv_true[target]))
    pred_pad = list(all_cv_predictions[target]) + [NULL]*(max_len - len(all_cv_predictions[target]))
    
    cv_true_df[target] = true_pad
    cv_pred_df[target] = pred_pad


competition_scores = [
    metric.scaling_error(all_cv_true[t], all_cv_predictions[t], t) for t in targets
]

estimated_lb_score = score(cv_true_df, cv_pred_df, row_id_column_name='id')

print("="*50)
print(f"Trained: {len(targets)} targets × 5 CV folds = {len(targets)*5} models")
print(f"Average CV MAE across all targets: {np.mean(cv_scores):.4f}")
print(f"{targets}")
print("Individual competition scores:", [f"{s:.4f}" for s in competition_scores])
print(f"ESTIMATED LB SCORE: {estimated_lb_score:.4f}")
print("="*50)



targets


cv_pred_df.info()


# Create submission file
submission = test[['id']].copy()

for target in targets:
    submission[target] = test_predictions[target]
for t in targets:
    for s in train[train[t].notnull()]['SMILES']:
        if s in test['SMILES'].tolist():
            submission.loc[test['SMILES']==s, t] = train[train['SMILES']==s][t].values[0]
submission.to_csv('submission.csv', index=False)
print(submission)



