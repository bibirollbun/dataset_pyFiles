# ============================================================
# Open Polymer Prediction 2025 â€” Full Stack (paths tuned to your Kaggle inputs)
# Prefers Kaggle dataset mounts shown in your screenshot, falls back to /mnt/data/*
# Integrates: data_tg3.xlsx (sheet 'Ğ›Ğ¸Ñ�Ñ‚1' if present), data_dnst1.xlsx, JCIM_sup_bigsmiles.csv, tc-smiles
# RDKit wheel from rdkit-2025-3-3-cp311 (if needed)
# Scaffold-aware CV, per-target features (2D/3D/FP), blended ensemble (Cat/LGBM/XGB/SVR)
# Produces submission.csv (id,Tg,FFV,Tc,Density,Rg)
# ============================================================

import os, warnings, gc, sys, math, random, subprocess
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd

SEED = 42
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
set_seed()

# --------------------------
# Path detection (prefer Kaggle mounts from your screenshot)
# --------------------------
def first_existing(paths):
    for p in paths:
        if isinstance(p, str) and os.path.exists(p):
            return p
    return None

# Core competition train/test (Kaggle first, then /mnt/data)
TRAIN_PATH = first_existing([
    "/kaggle/input/neurips-open-polymer-prediction-2025/train.csv",
    "/mnt/data/train (2).csv",
    "/mnt/data/train.csv",
])
TEST_PATH = first_existing([
    "/kaggle/input/neurips-open-polymer-prediction-2025/test.csv",
    "/mnt/data/test.csv",
])
assert TRAIN_PATH and TEST_PATH, "Could not find train/test files."

# Extras (Kaggle names per screenshot, then /mnt/data)
EXTRA_TG3_PATH   = first_existing([
    "/kaggle/input/smiles-extra-data/data_tg3.xlsx",
    "/mnt/data/data_tg3.xlsx",
])
EXTRA_DNST1_PATH = first_existing([
    "/kaggle/input/smiles-extra-data/data_dnst1.xlsx",
    "/mnt/data/data_dnst1.xlsx",
])
JCIM_SMILES_PATH = first_existing([
    "/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv",
    "/mnt/data/JCIM_sup_bigsmiles.csv",
])
TC_SMILES_PATH   = first_existing([
    "/kaggle/input/tc-smiles/Tc_SMILES.csv",
    "/mnt/data/Tc_SMILES.csv",
])

# RDKit wheel (if needed on Kaggle)
RDKit_WHEEL = first_existing([
    "/kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl"
])

try:
    from rdkit import Chem  # noqa: F401
except Exception:
    if RDKit_WHEEL:
        print("Installing RDKit wheelâ€¦")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", RDKit_WHEEL])
    from rdkit import Chem  # noqa: F401

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# ML libs
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
BAD_PATTERNS = [
    '[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]', "[R']", '[R"]',
    'R1','R2','R3','R4','R5','([R])','([R1])','([R2])'
]

# --------------------------
# Helpers
# --------------------------
def clean_and_validate_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles:
        return None
    for p in BAD_PATTERNS:
        if p in smiles:
            return None
    try:
        m = Chem.MolFromSmiles(smiles, sanitize=False)
        if m is None:
            return None
        try:
            Chem.SanitizeMol(m, catchErrors=True)
        except Exception:
            pass
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None

def get_scaffold(smiles: str):
    try:
        m = Chem.MolFromSmiles(smiles, sanitize=False)
        if m is None:
            return smiles
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return smiles

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def nnls_like_weights(P: np.ndarray, y: np.ndarray):
    """Non-negative weights, sumâ‰ˆ1; stable closed form with tiny ridge."""
    if P.ndim != 2:
        P = P.reshape(len(P), -1)
    reg = 1e-6 * np.eye(P.shape[1])
    w = np.linalg.lstsq(P.T @ P + reg, P.T @ y, rcond=None)[0]
    w = np.clip(w, 0, None)
    s = w.sum()
    return (w / s) if s > 0 else np.ones_like(w) / len(w)

# --------------------------
# Load core data
# --------------------------
train_raw = pd.read_csv(TRAIN_PATH)
test_raw  = pd.read_csv(TEST_PATH)

assert 'SMILES' in train_raw.columns and 'SMILES' in test_raw.columns, "SMILES column missing."
assert 'id' in test_raw.columns, "test needs id."

train_raw['SMILES'] = train_raw['SMILES'].apply(clean_and_validate_smiles)
test_raw['SMILES']  = test_raw['SMILES'].apply(clean_and_validate_smiles)
train_raw = train_raw.dropna(subset=['SMILES']).reset_index(drop=True)
test_raw  = test_raw.dropna(subset=['SMILES']).reset_index(drop=True)

# Ensure all target columns exist
for t in TARGETS:
    if t not in train_raw.columns:
        train_raw[t] = np.nan

# --------------------------
# Read your extra datasets (robust headers/sheets; â€œĞ›Ğ¸Ñ�Ñ‚1â€�)
# --------------------------
def maybe_read_csv(path, **kw):
    try:    return pd.read_csv(path, **kw)
    except: return None

def maybe_read_xlsx(path, **kw):
    try:    return pd.read_excel(path, **kw)
    except: return None

def maybe_read_xlsx_anysheet(path):
    """Use sheet 'Ğ›Ğ¸Ñ�Ñ‚1' if present, else first sheet."""
    try:
        xls = pd.ExcelFile(path)
        sheet_to_use = 'Ğ›Ğ¸Ñ�Ñ‚1' if 'Ğ›Ğ¸Ñ�Ñ‚1' in xls.sheet_names else xls.sheet_names[0]
        return pd.read_excel(path, sheet_name=sheet_to_use)
    except Exception:
        return maybe_read_xlsx(path)

external_datasets = []

# Tg extra
if EXTRA_TG3_PATH:
    tg3 = maybe_read_xlsx_anysheet(EXTRA_TG3_PATH)
    if tg3 is not None and 'SMILES' in tg3.columns:
        if 'Tg' in tg3.columns:
            ext_tg = tg3[['SMILES','Tg']].copy()
        elif 'Tg [K]' in tg3.columns:
            ext_tg = tg3.rename(columns={'Tg [K]':'Tg'})[['SMILES','Tg']].copy()
        else:
            ext_tg = None
        if ext_tg is not None:
            ext_tg['SMILES'] = ext_tg['SMILES'].apply(clean_and_validate_smiles)
            ext_tg['Tg'] = pd.to_numeric(ext_tg['Tg'], errors='coerce')
            ext_tg = ext_tg.dropna(subset=['SMILES','Tg']).groupby('SMILES', as_index=False)['Tg'].mean()
            external_datasets.append(('Tg', ext_tg))
            print(f"âœ… Tg extras: {len(ext_tg)} rows from {os.path.basename(EXTRA_TG3_PATH)}")

# Density extra
if EXTRA_DNST1_PATH:
    dnst1 = maybe_read_xlsx_anysheet(EXTRA_DNST1_PATH)
    if dnst1 is not None and 'SMILES' in dnst1.columns:
        if 'Density' in dnst1.columns:
            ext_den = dnst1[['SMILES','Density']].copy()
        elif 'density(g/cm3)' in dnst1.columns:
            ext_den = dnst1.rename(columns={'density(g/cm3)':'Density'})[['SMILES','Density']].copy()
        else:
            ext_den = None
        if ext_den is not None:
            ext_den['SMILES'] = ext_den['SMILES'].apply(clean_and_validate_smiles)
            ext_den['Density'] = pd.to_numeric(ext_den['Density'], errors='coerce')
            ext_den = ext_den.dropna(subset=['SMILES','Density']).groupby('SMILES', as_index=False)['Density'].mean()
            external_datasets.append(('Density', ext_den))
            print(f"âœ… Density extras: {len(ext_den)} rows from {os.path.basename(EXTRA_DNST1_PATH)}")

# Tc extra (tc-smiles)
if TC_SMILES_PATH:
    tc_smiles = maybe_read_csv(TC_SMILES_PATH)
    if tc_smiles is not None and 'SMILES' in tc_smiles.columns:
        if 'Tc' in tc_smiles.columns:
            ext_tc = tc_smiles[['SMILES','Tc']].copy()
        elif 'TC_mean' in tc_smiles.columns:
            ext_tc = tc_smiles.rename(columns={'TC_mean':'Tc'})[['SMILES','Tc']].copy()
        else:
            ext_tc = None
        if ext_tc is not None:
            ext_tc['SMILES'] = ext_tc['SMILES'].apply(clean_and_validate_smiles)
            ext_tc['Tc'] = pd.to_numeric(ext_tc['Tc'], errors='coerce')
            ext_tc = ext_tc.dropna(subset=['SMILES','Tc']).groupby('SMILES', as_index=False)['Tc'].mean()
            external_datasets.append(('Tc', ext_tc))
            print(f"âœ… Tc extras: {len(ext_tc)} rows from {os.path.basename(TC_SMILES_PATH)}")

# SMILES-only (JCIM)
if JCIM_SMILES_PATH:
    jcim = maybe_read_csv(JCIM_SMILES_PATH)
    if jcim is not None and 'SMILES' in jcim.columns:
        jcim_smiles = jcim[['SMILES']].dropna().copy()
        jcim_smiles['SMILES'] = jcim_smiles['SMILES'].apply(clean_and_validate_smiles)
        jcim_smiles = jcim_smiles.dropna().drop_duplicates()
        print(f"âœ… JCIM SMILES-only: {len(jcim_smiles)} unique SMILES (no targets)")

# Merge extras into train
def merge_external(train_df, ext_df, target):
    ext_df = ext_df.copy()
    ext_df['SMILES'] = ext_df['SMILES'].apply(clean_and_validate_smiles)
    ext_df = ext_df.dropna(subset=['SMILES', target]).groupby('SMILES', as_index=False)[target].mean()

    existing = set(train_df['SMILES'])
    # fill existing NaNs
    to_fill = ext_df[ext_df['SMILES'].isin(existing)]
    if len(to_fill):
        train_df = train_df.merge(to_fill, on='SMILES', how='left', suffixes=('', '_ext'))
        mask = train_df[target].isna() & train_df[f"{target}_ext"].notna()
        train_df.loc[mask, target] = train_df.loc[mask, f"{target}_ext"]
        train_df = train_df.drop(columns=[f"{target}_ext"])
    # append new SMILES rows
    new_smiles = set(ext_df['SMILES']) - existing
    if new_smiles:
        add = ext_df[ext_df['SMILES'].isin(new_smiles)][['SMILES', target]].copy()
        for t in TARGETS:
            if t not in add.columns: add[t] = np.nan
        train_df = pd.concat([train_df, add[['SMILES'] + TARGETS]], ignore_index=True)
    return train_df

train_ext = train_raw[['SMILES'] + TARGETS].copy()
for tgt, ext in external_datasets:
    train_ext = merge_external(train_ext, ext, tgt)

train_ext = (train_ext
             .replace([np.inf, -np.inf], np.nan)
             .dropna(how='all', subset=TARGETS)
             .drop_duplicates(subset=['SMILES'])
             .reset_index(drop=True))

print("\nğŸ“Š Labeled counts after merging extras:")
for t in TARGETS:
    print(f" â€¢ {t:<8}: {train_ext[t].notna().sum()}")

# --------------------------
# Target-specific preprocessing
# --------------------------
def preprocess_tg(df):       return df.drop_duplicates(subset='SMILES').copy()
def preprocess_ffv(df):
    d = df.copy()
    if 'FFV' in d.columns: d = d[d['FFV'].between(0.0, 1.0)]
    return d
def preprocess_tc(df):
    d = df.copy()
    if 'Tc' in d.columns and d['Tc'].notna().any():
        q1, q3 = d['Tc'].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        d = d[d['Tc'].between(lo, hi)]
    return d
def preprocess_density(df, cap_per_bin=300):
    d = df.copy()
    if 'Density' in d.columns:
        d = d[d['Density'].between(0.5, 2.0)]
        d['SMILES_norm'] = d['SMILES'].apply(clean_and_validate_smiles)
        d = d[d['SMILES_norm'].notnull()].copy()
        d['SMILES'] = d['SMILES_norm']; d.drop(columns=['SMILES_norm'], inplace=True)
        d['density_bin'] = pd.cut(d['Density'], bins=[0.5,1.0,1.5,2.0])
        d = (d.groupby('density_bin', observed=False)
               .apply(lambda g: g.sample(min(len(g), cap_per_bin), random_state=SEED))
               .reset_index(drop=True))
    return d
def preprocess_rg(df):       return df.drop_duplicates(subset='SMILES').copy()

# --------------------------
# Features
# --------------------------
def morgan_fp_bits(m, radius=2, nBits=1024):
    if m is None:
        return np.zeros(nBits, dtype=np.uint8)
    try:
        bv = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
        arr = np.zeros((nBits,), dtype=np.uint8)
        Chem.DataStructs.ConvertToNumpyArray(bv, arr)
        return arr
    except Exception:
        return np.zeros(nBits, dtype=np.uint8)

def base_2d_desc(m):
    if m is None:
        return dict(MolWt=np.nan, TPSA=np.nan, MolLogP=np.nan, NumHAcceptors=np.nan, NumHDonors=np.nan,
                    NumRotatableBonds=np.nan, HeavyAtomCount=np.nan, FractionCSP3=np.nan,
                    NumValenceElectrons=np.nan, NumAliphaticRings=np.nan)
    return {
        'MolWt': Descriptors.MolWt(m),
        'TPSA': rdMolDescriptors.CalcTPSA(m),
        'MolLogP': Descriptors.MolLogP(m),
        'NumHAcceptors': Descriptors.NumHAcceptors(m),
        'NumHDonors': Descriptors.NumHDonors(m),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(m),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(m),
        'FractionCSP3': Descriptors.FractionCSP3(m),
        'NumValenceElectrons': Descriptors.NumValenceElectrons(m),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(m),
    }

def shape_3d_features_from_smiles(smi):
    try:
        m = Chem.MolFromSmiles(smi, sanitize=False)
        if m is None:
            return dict(MolVolume3D=np.nan, MolSurfaceArea3D=np.nan,
                        InteratomicDistancesMean=np.nan, RgRadius=np.nan, RgMoI=np.nan)
        mH = Chem.AddHs(m)
        params = AllChem.ETKDGv3(); params.randomSeed = SEED
        res = AllChem.EmbedMolecule(mH, params)
        if res != 0:
            return dict(MolVolume3D=np.nan, MolSurfaceArea3D=np.nan,
                        InteratomicDistancesMean=np.nan, RgRadius=np.nan, RgMoI=np.nan)
        AllChem.UFFOptimizeMolecule(mH)
        vol = AllChem.ComputeMolVolume(mH)
        sa  = AllChem.ComputeMolSurfaceArea(mH)
        conf = mH.GetConformer()
        n = mH.GetNumAtoms()
        coords = np.array([list(conf.GetAtomPosition(i)) for i in range(n)])
        center = coords.mean(axis=0)
        dists = np.linalg.norm(coords - center, axis=1)
        radius = float(dists.max()) if n>0 else np.nan
        mass = np.array([a.GetMass() for a in mH.GetAtoms()])
        moi = float(np.sum(mass * np.sum((coords-center)**2, axis=1))) if n>0 else np.nan
        pair_d = []
        for i in range(n):
            for j in range(i+1, n):
                pair_d.append(np.linalg.norm(coords[i]-coords[j]))
        mean_inter = float(np.mean(pair_d)) if pair_d else np.nan
        return dict(MolVolume3D=vol, MolSurfaceArea3D=sa,
                    InteratomicDistancesMean=mean_inter, RgRadius=radius, RgMoI=moi)
    except Exception:
        return dict(MolVolume3D=np.nan, MolSurfaceArea3D=np.nan,
                    InteratomicDistancesMean=np.nan, RgRadius=np.nan, RgMoI=np.nan)

def featurize(df, use_fp=False, add_3d=False):
    out_rows, fp_rows = [], []
    for smi in df['SMILES']:
        m = Chem.MolFromSmiles(smi)
        out_rows.append(base_2d_desc(m))
        fp_rows.append(morgan_fp_bits(m, radius=2, nBits=1024) if use_fp else None)
    X2d = pd.DataFrame(out_rows, index=df.index)
    if add_3d:
        X3d = pd.DataFrame([shape_3d_features_from_smiles(s) for s in df['SMILES']], index=df.index)
        X = pd.concat([X2d, X3d], axis=1)
    else:
        X = X2d
    if use_fp:
        FP = np.stack([fp_rows[i] for i in range(len(df))])
        X_fp = pd.DataFrame(FP, columns=[f'FP_{i}' for i in range(FP.shape[1])], index=df.index)
        X = pd.concat([X, X_fp], axis=1)
    return X

# --------------------------
# Models + training wrappers
# --------------------------
CFG = {
    'FFV':     dict(use_fp=False, add_3d=False, base_models=['cat','lgbm','xgb']),
    'Density': dict(use_fp=False, add_3d=True,  base_models=['lgbm','cat','xgb']),
    'Tc':      dict(use_fp=True,  add_3d=True,  base_models=['cat','lgbm','svr']),
    'Tg':      dict(use_fp=True,  add_3d=False, base_models=['cat','lgbm','xgb']),
    'Rg':      dict(use_fp=False, add_3d=True,  base_models=['svr','cat','lgbm']),
}

def fit_cat(X, y, Xv, yv):
    m = CatBoostRegressor(
        iterations=5000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
        loss_function='RMSE', random_seed=SEED, early_stopping_rounds=200,
        verbose=False, task_type='CPU'
    )
    m.fit(Pool(X, y), eval_set=Pool(Xv, yv), verbose=False)
    return m

def fit_lgbm(X, y, Xv, yv):
    m = lgb.LGBMRegressor(
        objective='regression', metric='rmse', learning_rate=0.03,
        num_leaves=64, min_data_in_leaf=20, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=1e-2, seed=SEED,
        n_estimators=5000
    )
    m.fit(X, y, eval_set=[(Xv, yv)], eval_metric='rmse',
          callbacks=[lgb.early_stopping(200, verbose=False)])
    return m

def fit_xgb(X, y, Xv, yv):
    m = xgb.XGBRegressor(
        n_estimators=5000, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1e-2,
        random_state=SEED, tree_method='hist'
    )
    m.fit(X, y, eval_set=[(Xv, yv)], eval_metric='rmse',
          verbose=False, early_stopping_rounds=200)
    return m

def fit_svr(X, y, Xv, yv):
    # Impute NaNs â†’ median, then scale, then SVR (SVR can't handle NaNs).
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=False)),
        ('svr', SVR(C=10.0, epsilon=0.01, kernel='rbf', gamma='scale'))
    ])
    pipe.fit(X, y)
    return pipe

def train_one_target(tgt, df_all, cfg, n_splits=5):
    # preprocess per target
    if tgt=='FFV':      df = preprocess_ffv(df_all[df_all['FFV'].notna()])
    elif tgt=='Density':df = preprocess_density(df_all[df_all['Density'].notna()], cap_per_bin=300)
    elif tgt=='Tc':     df = preprocess_tc(df_all[df_all['Tc'].notna()])
    elif tgt=='Tg':     df = preprocess_tg(df_all[df_all['Tg'].notna()])
    elif tgt=='Rg':     df = preprocess_rg(df_all[df_all['Rg'].notna()])
    else: raise ValueError(tgt)

    if df.empty:
        print(f"âš ï¸� No data for {tgt}")
        return None, None, None

    X = featurize(df, use_fp=cfg['use_fp'], add_3d=cfg['add_3d'])
    y = df[tgt].values

    groups = df['SMILES'].apply(get_scaffold).values
    n_splits = min(n_splits, max(2, len(np.unique(groups))))
    gkf = GroupKFold(n_splits=n_splits)

    names = cfg['base_models']
    per_model_oof = {n: np.zeros(len(df)) for n in names}
    oof_blend = np.zeros(len(df))
    models_by_fold = []

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups), 1):
        Xtr, Xv = X.iloc[tr], X.iloc[va]
        ytr, yv = y[tr], y[va]

        fold_models, fold_preds = {}, []
        for n in names:
            if   n=='cat': mdl = fit_cat(Xtr,ytr,Xv,yv)
            elif n=='lgbm':mdl = fit_lgbm(Xtr,ytr,Xv,yv)
            elif n=='xgb': mdl = fit_xgb(Xtr,ytr,Xv,yv)
            elif n=='svr': mdl = fit_svr(Xtr,ytr,Xv,yv)
            else: continue
            pva = mdl.predict(Xv)
            fold_models[n] = mdl
            per_model_oof[n][va] = pva
            fold_preds.append(pva)

        P = np.vstack(fold_preds).T
        w = nnls_like_weights(P, yv)
        blend = (P @ w).ravel()
        oof_blend[va] = blend

        print(f"[{tgt}] Fold {fold}/{n_splits} | MAE {mean_absolute_error(yv, blend):.5f} | w={dict(zip(names, np.round(w,3)))}")
        models_by_fold.append((fold_models, w))
        gc.collect()

    print(f"[{tgt}] OOF  MAE {mean_absolute_error(y, oof_blend):.5f} | RMSE {rmse(y, oof_blend):.5f} | R2 {r2_score(y, oof_blend):.4f}")
    return dict(df=df, X=X, y=y, base_model_names=names, models=models_by_fold, oof_blend=oof_blend), oof_blend, X.columns

def predict_one_target(art, test_df, cfg):
    Xtest = featurize(test_df, use_fp=cfg['use_fp'], add_3d=cfg['add_3d'])
    # align columns
    train_cols = art['X'].columns
    for c in train_cols:
        if c not in Xtest.columns: Xtest[c] = 0
    extra = [c for c in Xtest.columns if c not in train_cols]
    if extra: Xtest = Xtest.drop(columns=extra)
    Xtest = Xtest[train_cols]

    preds = np.zeros(len(Xtest))
    B = len(art['models'])
    for fold_models, w in art['models']:
        fold_preds = []
        for n in art['base_model_names']:
            fold_preds.append(fold_models[n].predict(Xtest))
        Ptest = np.vstack(fold_preds).T
        preds += (Ptest @ w).ravel() / B
    return preds

# --------------------------
# Build per-target frames
# --------------------------
tg_df      = preprocess_tg(     train_ext[train_ext['Tg'].notna()])
ffv_df     = preprocess_ffv(    train_ext[train_ext['FFV'].notna()])
tc_df      = preprocess_tc(     train_ext[train_ext['Tc'].notna()])
density_df = preprocess_density(train_ext[train_ext['Density'].notna()], cap_per_bin=300)
rg_df      = preprocess_rg(     train_ext[train_ext['Rg'].notna()])

print("\nğŸ§ª Dataset sizes:")
for name, d in [('Tg',tg_df),('FFV',ffv_df),('Tc',tc_df),('Density',density_df),('Rg',rg_df)]:
    print(f" â€¢ {name:<8}: {len(d)} rows")

# --------------------------
# Train all targets
# --------------------------
SPLITS = 5
artifacts = {}
for tgt, d in [('Tg', tg_df), ('FFV', ffv_df), ('Tc', tc_df), ('Density', density_df), ('Rg', rg_df)]:
    if d.empty:
        print(f"âš ï¸� Skip {tgt}: no labels.")
        continue
    print(f"\n==== Training {tgt} ====")
    art, oof, cols = train_one_target(tgt, train_ext, CFG[tgt], n_splits=SPLITS)
    artifacts[tgt] = art
    gc.collect()

# --------------------------
# Predict on test
# --------------------------
test_df = test_raw[['id','SMILES']].copy()
results = {}
print("\nğŸ”® Predicting test...")
for tgt in TARGETS:
    if tgt in artifacts and artifacts[tgt] is not None:
        print(f"-> {tgt}")
        results[tgt] = predict_one_target(artifacts[tgt], test_df[['SMILES']], CFG[tgt])
    else:
        print(f"-> {tgt}: no model; median fill.")
        med = np.nanmedian(train_ext[tgt]) if tgt in train_ext.columns else 0.0
        results[tgt] = np.full(len(test_df), med, dtype=float)

# --------------------------
# Build submission
# --------------------------
sub = pd.DataFrame({
    'id':      test_df['id'].values,
    'Tg':      results['Tg'],
    'FFV':     results['FFV'],
    'Tc':      results['Tc'],
    'Density': results['Density'],
    'Rg':      results['Rg'],
})

# Safety fill
for c in ['Tg','FFV','Tc','Density','Rg']:
    if sub[c].isna().any():
        med = np.nanmedian(train_ext[c]) if c in train_ext.columns else 0.0
        sub[c] = sub[c].fillna(med)

sub.to_csv("submission.csv", index=False)
print("\nâœ… Saved submission.csv (id,Tg,FFV,Tc,Density,Rg)")
print(sub.head())


