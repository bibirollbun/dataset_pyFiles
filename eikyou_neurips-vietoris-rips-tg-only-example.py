!pip install rdkit-pypi persim ripser openbabel-wheel


import pandas as pd
import numpy as np
import warnings
from tqdm.notebook import tqdm
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
from openbabel import pybel, openbabel
import gc
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
train_tg = train.loc[~train.Tg.isnull(), ['SMILES', 'Tg']]


import gc
import numpy as np
from tqdm.auto import tqdm
from openbabel import pybel, openbabel
from ripser import ripser


def smiles_to_point_cloud(smiles: str) -> np.ndarray:
    """
    Generate 3D atom coordinates from a SMILES string, optimize the conformer,
    and return an array of shape (N_atoms, 3).
    """
    # Read molecule from SMILES and prepare 3D coordinates
    mol = pybel.readstring("smi", smiles)
    mol.addh()      # add explicit hydrogens
    mol.make3D()    # generate initial 3D conformer

    # Extract internal OBMol for force field optimization
    obmol = mol.OBMol

    # Setup and run MMFF94 optimization
    ff = openbabel.OBForceField.FindForceField("MMFF94")
    if ff is None:
        raise RuntimeError("MMFF94 force field not found")
    ff.Setup(obmol)
    ff.SteepestDescent(250)                  # first run steepest descent for 250 steps
    ff.ConjugateGradients(1000, 1.0e-6)       # then conjugate gradients until RMS 1e-6 or 1000 steps
    ff.GetCoordinates(obmol)                 # update molecule coordinates

    # Collect all atom coordinates
    coords = np.array([atom.coords for atom in mol.atoms], dtype=np.float32)

    # Explicitly delete objects to free memory
    del ff, obmol, mol
    return coords


def compute_persistence_diagram(pc: np.ndarray, maxdim=2) -> np.ndarray:
    """
    Compute a persistence diagram from a point cloud array of shape (N, 3).
    Returns an array of shape (n_points, 3) with rows [birth, death, dimension].
    """
    result = ripser(pc, maxdim=maxdim)
    diagrams = result['dgms']
    all_points = []
    for dim, dgm in enumerate(diagrams):
        if dgm.size == 0:
            continue
        births = dgm[:, 0:1]
        deaths = dgm[:, 1:2]
        dims = np.full((dgm.shape[0], 1), fill_value=dim, dtype=np.int32)
        # Stack as [birth, death, dimension]
        all_points.append(np.hstack([births, deaths, dims]))
    if all_points:
        return np.vstack(all_points)
    else:
        return np.empty((0, 3), dtype=np.float32)


def manual_betti_curve(dgm: np.ndarray,
                       n_bins: int = 64,
                       maxdim: int = 2) -> np.ndarray:
    """
    Compute Betti curves from a persistence diagram array of shape (n_points, 3).
    The columns are [birth, death, dimension].
    Returns a concatenated vector of Betti counts for each dimension,
    with length = (maxdim+1) * n_bins.
    """
    # Handle empty diagram
    if dgm.size == 0:
        return np.zeros((maxdim + 1) * n_bins, dtype=int)

    # Filter out infinite death values
    finite_mask = np.isfinite(dgm[:, 1])
    dgm_finite = dgm[finite_mask]
    if dgm_finite.size == 0:
        return np.zeros((maxdim + 1) * n_bins, dtype=int)

    births = dgm_finite[:, 0]
    deaths = dgm_finite[:, 1]
    dims = dgm_finite[:, 2].astype(int)

    tmin = births.min()
    tmax = deaths.max()
    ts = np.linspace(tmin, tmax, n_bins)

    curves = []
    # Compute counts for each homology dimension
    for dim in range(maxdim + 1):
        mask_dim = (dims == dim)
        b_dim = births[mask_dim]
        d_dim = deaths[mask_dim]
        if b_dim.size == 0:
            # no intervals for this dimension
            curves.append(np.zeros(n_bins, dtype=int))
            continue
        # build matrix (n_intervals, n_bins): True if interval covers threshold
        M = (b_dim[:, None] <= ts[None, :]) & (ts[None, :] < d_dim[:, None])
        counts = M.sum(axis=0)
        curves.append(counts)

    # Concatenate counts across all dimensions
    return np.concatenate(curves)


def get_betti_curves(smiles_list, n_bins=4, maxdim=2):
    """
    Given a list of SMILES strings, generate persistence diagrams and
    return a matrix of Betti curves.

    Parameters
    ----------
    smiles_list : iterable of str
        List of SMILES strings for molecules.
    n_bins : int, default=4
        Number of bins for each Betti curve.
    maxdim : int, default=2
        Maximum homology dimension to include (0..maxdim).

    Returns
    -------
    betti_curves : ndarray, shape (n_samples, (maxdim+1)*n_bins)
        Matrix where each row is the Betti curve vector of a molecule.
    """
    diagrams = []
    errors = {}
    for smile in tqdm(smiles_list, desc="SMILES â†’ diagrams"):
        try:
            pc = smiles_to_point_cloud(smile)
            dgm = compute_persistence_diagram(pc, maxdim=maxdim)
            diagrams.append(dgm)
        except Exception as e:
            errors[smile] = str(e)
        gc.collect()

    print(f"Diagrams obtained: {len(diagrams)}")
    if errors:
        print(f"Errors for {len(errors)} SMILES, examples:")
        for sm, msg in list(errors.items())[:5]:
            print(f"  {sm} â†’ {msg}")

    betti_curves = np.vstack([
        manual_betti_curve(d, n_bins=n_bins, maxdim=maxdim)
        for d in diagrams
    ])

    return betti_curves



def get_molecular_descriptors(max_autocorr=10):
    """Get molecular descriptors - either hardcoded list or auto-discovered"""

    descriptor_list_all = []
    test_mol = Chem.MolFromSmiles('CCO')

    # Collect all valid descriptors first
    for name in dir(Descriptors):
        if not name.startswith('_'):
            try:
                func = getattr(Descriptors, name)
                if callable(func):
                    result = func(test_mol)
                    if isinstance(result, (int, float)) and not np.isnan(result):
                        descriptor_list_all.append((name, func))
            except:
                pass

    print(f"ğŸ”� Total discovered descriptors before filtering: {len(descriptor_list_all)}")

    # Sort AUTOCORR2D descriptors by their numeric suffix
    autocorr_descriptors = [
        (name, func)
        for name, func in descriptor_list_all
        if name.startswith('AUTOCORR2D_')
    ]
    autocorr_descriptors.sort(key=lambda x: int(x[0].split('_')[-1]))

    # Select only the lowest-numbered ones
    limited_autocorr = autocorr_descriptors[:max_autocorr]

    # Include all other descriptors
    other_descriptors = [
        (name, func)
        for name, func in descriptor_list_all
        if not name.startswith('AUTOCORR2D_')
    ]

    # Final descriptor list
    descriptor_list = limited_autocorr + other_descriptors

    print(f"âœ… Auto-discovered {len(descriptor_list)} descriptors (limited to {max_autocorr} AUTOCORR2D):")
    names = [name for name, _ in descriptor_list]
    print("  " + ", ".join(names))

    feature_names = [name for name, _ in descriptor_list]
    return descriptor_list

def smiles_to_features(smiles_list, molecular_descriptors, clean_descriptors=False):
    features = []
    total = len(smiles_list)

    print(f"Processing {total} SMILES...", end="", flush=True)

    for smiles in tqdm(smiles_list):

        mol_features = []
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                mol_features = [np.nan] * len(molecular_descriptors)
            else:
                for name, func in molecular_descriptors:
                    try:
                        value = func(mol)
                        if np.isinf(value) or abs(value) > 1e10:
                            value = np.nan
                        mol_features.append(value)
                    except:
                        mol_features.append(np.nan)
        except:
            mol_features = [np.nan] * len(molecular_descriptors)

        features.append(mol_features)

    features = np.array(features, dtype=float)
    print(" âœ…", flush=True)

    if clean_descriptors:
        # % of NaNs per column
        nan_ratio = np.isnan(features).mean(axis=0)
        dropped_mask = nan_ratio > 0.95

        if np.any(dropped_mask):
            dropped_names = [molecular_descriptors[i][0] for i, drop in enumerate(dropped_mask) if drop]
            print(f"âš ï¸� Dropping {len(dropped_names)} descriptors with >98% missing values:")
            print("   " + ", ".join(dropped_names))

            features = features[:, ~dropped_mask]
            molecular_descriptors = [d for i, d in enumerate(molecular_descriptors) if not dropped_mask[i]]

    return features, molecular_descriptors

def clean_features(X, feature_names=None):
    """Handle NaN/inf values and impute missing data, with concise missing summary."""
    X_clean = X.copy()
    X_clean[np.isinf(X_clean)] = np.nan

    total_missing = np.isnan(X_clean).sum()
    print(f"ğŸ§¹ Cleaned {total_missing:,} missing values ({total_missing / X_clean.size:.1%})")

    for i in range(X_clean.shape[1]):
        col = X_clean[:, i]
        if np.isnan(col).any():
            missing_pct = np.isnan(col).mean() * 100
            name = feature_names[i] if feature_names else f"col_{i}"
            print(f"   â€¢ {name}: {missing_pct:.1f}% missing")
            median = np.nanmedian(col)
            X_clean[np.isnan(col), i] = median if not np.isnan(median) else 0

    return X_clean

def get_descriptors(smiles):
    molecular_descriptors = get_molecular_descriptors(max_autocorr=10) 
    raw_descriptors, molecular_descriptors = smiles_to_features(smiles, molecular_descriptors, clean_descriptors=True)
    descriptors = clean_features(raw_descriptors)
    return descriptors


from lightgbm import LGBMRegressor
import lightgbm as lgb
print("Using LightGBM version",lgb.__version__)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error

FOLDS = 5
kf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'num_leaves': 127,           
        'learning_rate': 0.07,       
        'feature_fraction': 0.8,     
        'bagging_fraction': 0.9,     
        'bagging_freq': 1,           # Bag every iteration
        'lambda_l1': 0.1,            # L1 regularization
        'lambda_l2': 0.1,            # L2 regularization
        'min_data_in_leaf': 10,      # Prevent overfitting
        'n_estimators': 2000,
        'verbose': -1,
        'random_state': 42
    }


descriptors = get_descriptors(train_tg.SMILES.values)


X = descriptors
y = train_tg.Tg.values
y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')

cv_scores = []
models = []
all_val_true = []
all_val_pred = []

print('Training Tg......')

for fold, (train_index, test_index) in enumerate(kf.split(X, y_binned)):

    X_train = X[train_index].copy()
    y_train = y[train_index].copy()
    X_val = X[test_index].copy()
    y_val = y[test_index].copy()

    
    model = LGBMRegressor(**params)
    model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(50, verbose=0), lgb.log_evaluation(0)]
        )
    val_pred = model.predict(X_val)
    cv_score = mean_absolute_error(y_val, val_pred)
    cv_scores.append(cv_score)

    all_val_true.extend(y_val)
    all_val_pred.extend(val_pred)

    print(f"----Fold {fold+1} Complete / MAE = {cv_score:.4f}", flush=True)


cv_mean = np.mean(cv_scores)
print(f"===CV: {cv_mean:.4f} Â± {np.std(cv_scores):.3f}===")


betti_curves = get_betti_curves(train_tg.SMILES.values)


X = np.hstack([betti_curves, descriptors])
y = train_tg.Tg.values
y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')

cv_scores = []
models = []
all_val_true = []
all_val_pred = []

print('Training Tg......')

for fold, (train_index, test_index) in enumerate(kf.split(X, y_binned)):

    X_train = X[train_index].copy()
    y_train = y[train_index].copy()
    X_val = X[test_index].copy()
    y_val = y[test_index].copy()

    
    model = LGBMRegressor(**params)
    model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(50, verbose=0), lgb.log_evaluation(0)]
        )
    val_pred = model.predict(X_val)
    cv_score = mean_absolute_error(y_val, val_pred)
    cv_scores.append(cv_score)

    all_val_true.extend(y_val)
    all_val_pred.extend(val_pred)

    print(f"----Fold {fold+1} Complete / MAE = {cv_score:.4f}", flush=True)


cv_mean = np.mean(cv_scores)
print(f"===CV: {cv_mean:.4f} Â± {np.std(cv_scores):.3f}===")

