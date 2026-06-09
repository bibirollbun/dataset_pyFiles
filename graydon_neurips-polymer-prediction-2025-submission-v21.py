# Offline install from attached dataset
!pip install rdkit --no-index --find-links=/kaggle/input/polymer-wheels-py311 --quiet 2>/dev/null || \
 echo "Note: rdkit install from offline source failed or skipped"



import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import lightgbm as lgb
import catboost as cb

print("All imports OK")



# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def canonicalize(smiles):
    """Convert SMILES to canonical form for deduplication."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol)
    except Exception:
        return smiles


def get_mol_descriptors(mol):
    """Calculate all RDKit 2D descriptors for a molecule."""
    if mol is None:
        return {}
    descriptors = {}
    for name, func in Descriptors.descList:
        try:
            descriptors[name] = func(mol)
        except Exception:
            descriptors[name] = 0.0
    return descriptors


def get_morgan_fingerprint(mol, radius=2, n_bits=2048):
    """Morgan / ECFP fingerprint as numpy array."""
    if mol is None:
        return np.zeros(n_bits)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(n_bits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def get_maccs_keys(mol):
    """MACCS 167-bit structural keys."""
    if mol is None:
        return np.zeros(167)
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(167, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def get_atompair_fingerprint(mol, n_bits=1024):
    """Atom pair fingerprint."""
    if mol is None:
        return np.zeros(n_bits)
    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(n_bits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def get_torsion_fingerprint(mol, n_bits=1024):
    """Topological torsion fingerprint."""
    if mol is None:
        return np.zeros(n_bits)
    gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(n_bits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def generate_features(smiles_series):
    """Generate full feature matrix from SMILES strings."""
    mols = smiles_series.apply(Chem.MolFromSmiles)
    all_frames = []

    # 1. RDKit 2D Descriptors (~210 features)
    print("  Calculating RDKit 2D descriptors...")
    desc_list = mols.apply(get_mol_descriptors).tolist()
    all_frames.append(pd.DataFrame(desc_list))

    # 2. Morgan Fingerprints (2048 bits)
    print("  Calculating Morgan fingerprints...")
    fp_list = mols.apply(get_morgan_fingerprint).tolist()
    all_frames.append(pd.DataFrame(fp_list, columns=[f'morgan_{i}' for i in range(2048)]))

    # 3. MACCS Keys (167 bits)
    print("  Calculating MACCS keys...")
    maccs_list = mols.apply(get_maccs_keys).tolist()
    all_frames.append(pd.DataFrame(maccs_list, columns=[f'maccs_{i}' for i in range(167)]))

    # 4. Atom Pair Fingerprints (1024 bits)
    print("  Calculating Atom Pair fingerprints...")
    ap_list = mols.apply(get_atompair_fingerprint).tolist()
    all_frames.append(pd.DataFrame(ap_list, columns=[f'atompair_{i}' for i in range(1024)]))

    # 5. Topological Torsion Fingerprints (1024 bits)
    print("  Calculating Topological Torsion fingerprints...")
    tt_list = mols.apply(get_torsion_fingerprint).tolist()
    all_frames.append(pd.DataFrame(tt_list, columns=[f'torsion_{i}' for i in range(1024)]))

    features_df = pd.concat(all_frames, axis=1)
    features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features_df



# ============================================================================
# MODEL DEFINITION - PER-TARGET ENSEMBLE
# ============================================================================

def build_ensemble(n_estimators=500, random_state=42):
    """Build a HGB + LightGBM + CatBoost voting ensemble for a SINGLE target."""
    hgb = HistGradientBoostingRegressor(
        max_iter=n_estimators,
        learning_rate=0.05,
        max_depth=7,
        min_samples_leaf=10,
        random_state=random_state,
        loss='absolute_error',
    )

    lgbm_model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=7,
        min_child_samples=10,
        random_state=random_state,
        objective='mae',
        n_jobs=-1,
        verbosity=-1,
    )

    cat_model = cb.CatBoostRegressor(
        iterations=n_estimators,
        learning_rate=0.05,
        depth=7,
        min_data_in_leaf=10,
        random_seed=random_state,
        loss_function='MAE',
        verbose=False,
        thread_count=-1,
    )

    ensemble = VotingRegressor(
        estimators=[
            ('hgb', hgb),
            ('lgbm', lgbm_model),
            ('cat', cat_model),
        ]
    )
    return ensemble


# Per-target hyperparameter configs (from best practices + 4th place insights)
TARGET_CONFIGS = {
    'Tg':      {'n_estimators': 800, 'log_transform': False},
    'FFV':     {'n_estimators': 600, 'log_transform': True},
    'Tc':      {'n_estimators': 600, 'log_transform': False},
    'Density': {'n_estimators': 600, 'log_transform': True},
    'Rg':      {'n_estimators': 600, 'log_transform': True},
}



# ============================================================================
# DATA LOADING, TRAINING, AND PREDICTION
# ============================================================================

# --- Config ---
TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

DATA_DIR = '/kaggle/input/neurips-open-polymer-prediction-2025'
if not os.path.exists(DATA_DIR):
    DATA_DIR = 'data/neurips-2025'  # Local fallback

TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE  = os.path.join(DATA_DIR, 'test.csv')
SUPPLEMENT_DIR = os.path.join(DATA_DIR, 'train_supplement')

print(f"Data dir: {DATA_DIR}")

# ============== LOAD AND MERGE DATA ==============

print("Loading training data...")
df = pd.read_csv(TRAIN_FILE)
print(f"  Host data: {len(df)} rows")

# -- Supplementary datasets --
# dataset1: SMILES, TC_mean  -> maps to Tc
# dataset3: SMILES, Tg
# dataset4: SMILES, FFV
# dataset2: SMILES only (skip - no labels)

supplementary_mappings = {
    'dataset1.csv': {'TC_mean': 'Tc'},
    'dataset3.csv': {},          # Already has 'Tg' column
    'dataset4.csv': {},          # Already has 'FFV' column
}

for fname, col_map in supplementary_mappings.items():
    fpath = os.path.join(SUPPLEMENT_DIR, fname)
    if os.path.exists(fpath):
        ext_df = pd.read_csv(fpath)
        if col_map:
            ext_df.rename(columns=col_map, inplace=True)

        if 'SMILES' not in ext_df.columns:
            continue

        # Add missing target columns as NaN
        for t in TARGETS:
            if t not in ext_df.columns:
                ext_df[t] = np.nan

        # Keep only SMILES + target columns
        ext_df = ext_df[['SMILES'] + TARGETS]
        df = pd.concat([df, ext_df], ignore_index=True)
        print(f"  + {fname}: {len(ext_df)} rows added")

# -- Also try Kaggle external datasets --
EXTRA_PATHS = [
    '/kaggle/input/extra-dataset-with-smiles-tg-pid-polymers-class',
    '/kaggle/input/extended-polymer-dataset',
]
for extra_dir in EXTRA_PATHS:
    if os.path.exists(extra_dir):
        for f in os.listdir(extra_dir):
            if f.endswith('.csv'):
                try:
                    ext_df = pd.read_csv(os.path.join(extra_dir, f))
                    ext_df.columns = [c.strip() for c in ext_df.columns]
                    rename_map = {'smiles': 'SMILES', 'tg': 'Tg', 'density': 'Density'}
                    ext_df.rename(columns=rename_map, inplace=True)
                    if 'SMILES' in ext_df.columns:
                        for t in TARGETS:
                            if t not in ext_df.columns:
                                ext_df[t] = np.nan
                        ext_df = ext_df[['SMILES'] + TARGETS]
                        df = pd.concat([df, ext_df], ignore_index=True)
                        print(f"  + Kaggle external {f}: {len(ext_df)} rows")
                except Exception as e:
                    print(f"  ! Failed {f}: {e}")

print(f"Total rows after merge: {len(df)}")

# ============== CANONICALIZE & DEDUPLICATE ==============

print("Canonicalizing SMILES...")
df['SMILES_canonical'] = df['SMILES'].apply(canonicalize)

# Deduplicate: keep first occurrence (host data comes first = higher quality)
before = len(df)
df = df.drop_duplicates(subset=['SMILES_canonical'], keep='first').reset_index(drop=True)
print(f"Deduplication: {before} -> {len(df)} rows (dropped {before - len(df)} duplicates)")

# ============== GENERATE FEATURES ==============

print("Generating features...")
X_all = generate_features(df['SMILES'])

# Feature cleaning: remove constant and highly correlated
from sklearn.feature_selection import VarianceThreshold

print(f"  Raw features: {X_all.shape[1]}")
sel = VarianceThreshold(threshold=0)
X_clean = sel.fit_transform(X_all)
clean_cols = X_all.columns[sel.get_support()]
X_all = pd.DataFrame(X_clean, columns=clean_cols)

# Remove highly correlated (>0.95) features
corr_matrix = X_all.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
X_all = X_all.drop(columns=to_drop)
selected_features = X_all.columns.tolist()
print(f"  Features after cleaning: {X_all.shape[1]}")

# ============== wMAE WEIGHTS ==============

# Compute competition wMAE weights from HOST data only (first ~7974 rows)
host_mask = df.index < 7974  # approximate host data boundary
host_df = df[host_mask]

raw_weights = {}
for col in TARGETS:
    if col in df.columns:
        vals = host_df[col].dropna()
        n_i = len(vals)
        r_i = vals.max() - vals.min() if n_i > 1 else 1.0
        if r_i > 0 and n_i > 0:
            raw_weights[col] = (1 / np.sqrt(n_i)) / r_i
        else:
            raw_weights[col] = 0
total_w = sum(raw_weights.values())
weights = {col: w / total_w for col, w in raw_weights.items()}
print(f"wMAE weights: { {k: round(v, 4) for k, v in weights.items()} }")

# ============== PER-TARGET CROSS-VALIDATION ==============

print("\n" + "="*60)
print("5-FOLD CROSS-VALIDATION (per-target, original space)")
print("="*60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_target_scores = {t: [] for t in TARGETS}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
    fold_wmae = 0.0
    for target in TARGETS:
        if target not in df.columns:
            continue

        config = TARGET_CONFIGS[target]

        # Get rows where this target is available
        y_full = df[target]
        train_mask = train_idx[y_full.iloc[train_idx].notna()]
        val_mask   = val_idx[y_full.iloc[val_idx].notna()]

        if len(train_mask) < 10 or len(val_mask) < 3:
            continue

        X_tr = X_all.iloc[train_mask]
        y_tr = y_full.iloc[train_mask].values.copy()
        X_va = X_all.iloc[val_mask]
        y_va = y_full.iloc[val_mask].values.copy()

        # Optional log-transform
        shift = 0
        if config['log_transform']:
            min_val = y_tr.min()
            shift = -min_val + 1 if min_val <= 0 else 0
            y_tr = np.log1p(y_tr + shift)

        model = build_ensemble(n_estimators=config['n_estimators'])
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)

        # Inverse transform
        if config['log_transform']:
            preds = np.expm1(preds) - shift

        mae = mean_absolute_error(y_va, preds)
        cv_target_scores[target].append(mae)
        fold_wmae += weights.get(target, 0) * mae

    print(f"  Fold {fold+1} wMAE: {fold_wmae:.5f}")

print("\nPer-target CV MAE:")
for t in TARGETS:
    scores = cv_target_scores[t]
    if scores:
        print(f"  {t:>8s}: {np.mean(scores):.5f} (+/- {np.std(scores):.5f})  [n_folds={len(scores)}]")

overall_wmae = sum(
    weights.get(t, 0) * np.mean(cv_target_scores[t])
    for t in TARGETS if cv_target_scores[t]
)
print(f"\nOverall CV wMAE: {overall_wmae:.5f}")

# ============== TRAIN FINAL PER-TARGET MODELS ==============

print("\nTraining final per-target models on ALL data...")
trained_models = {}
target_shifts = {}

for target in TARGETS:
    if target not in df.columns:
        continue
    config = TARGET_CONFIGS[target]

    # Only rows where target is available
    mask = df[target].notna()
    X_t = X_all[mask]
    y_t = df[target][mask].values.copy()

    shift = 0
    if config['log_transform']:
        min_val = y_t.min()
        shift = -min_val + 1 if min_val <= 0 else 0
        y_t = np.log1p(y_t + shift)

    model = build_ensemble(n_estimators=config['n_estimators'])
    model.fit(X_t, y_t)

    trained_models[target] = model
    target_shifts[target] = (config['log_transform'], shift)
    print(f"  {target}: trained on {len(y_t)} samples")

# ============== PREDICT TEST SET ==============

if os.path.exists(TEST_FILE):
    print("\nLoading and predicting test set...")
    test_df = pd.read_csv(TEST_FILE)
    X_test_raw = generate_features(test_df['SMILES'])
    X_test_raw = X_test_raw.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test_raw.reindex(columns=selected_features, fill_value=0.0)

    submission = pd.DataFrame()
    submission['id'] = test_df['id']

    for target in TARGETS:
        if target in trained_models:
            preds = trained_models[target].predict(X_test)

            # Inverse log-transform
            do_log, shift = target_shifts[target]
            if do_log:
                preds = np.expm1(preds) - shift

            submission[target] = preds
        else:
            submission[target] = 0.0

    # ======= POST-PROCESSING =======
    # Tg distribution shift correction
    # Best known from 2nd place: (9/5)*Tg + 45  -> 0.066 private LB
    if 'Tg' in submission.columns:
        print("Applying Tg post-processing: Tg * 1.8 + 45")
        submission['Tg'] = submission['Tg'] * 1.8 + 45

    submission.to_csv('submission.csv', index=False)
    print(f"\nSaved submission.csv ({len(submission)} rows)")
    print(submission.head())
else:
    print("No test file found — skipping prediction.")

print("\nDone!")


