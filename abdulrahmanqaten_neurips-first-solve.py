import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

sample = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')


train.head()


train.info()


test.info()


sample.info()


# --- IMPORTS ---
import pandas as pd
import numpy as np
import warnings

# Modeling
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

# RDKit for Feature Engineering
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors

# CRITICAL: Suppress all RDKit warnings for a clean output
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')
warnings.filterwarnings('ignore')


# --- CONFIGURATION ---
TARGET_VARIABLES = ["Tg", "FFV", "Tc", "Density", "Rg"]
N_SPLITS = 5
RANDOM_STATE = 42


# --- RDKIT FEATURE ENGINEERING (Stable Version) ---
def generate_rdkit_features(smiles_str: str):
    """
    Generates Descriptors and Morgan Fingerprints. Atom-Pair was removed for compatibility.
    """
    mol = Chem.MolFromSmiles(smiles_str)
    
    # Define descriptor list and fingerprint sizes
    desc_list = [d[0] for d in Descriptors._descList]
    morgan_fp_size = 1024
    
    if mol is None:
        # Return NaNs for invalid SMILES
        total_feature_count = len(desc_list) + morgan_fp_size
        return np.full(total_feature_count, np.nan)

    # 1. Descriptors (using the direct, compatible method)
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)
    descriptors = np.array(calculator.CalcDescriptors(mol))
    
    # 2. Morgan Fingerprints (using the direct, compatible function call)
    mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=morgan_fp_size)
    mfp_array = np.array(list(mfp.ToBitString())).astype(int)

    # We only concatenate the two working feature sets
    return np.concatenate([descriptors, mfp_array])


# --- DATA PREPARATION ---
print("Loading data...")
train_df, test_df, sample_df = train, test, sample

print("Generating RDKit features...")
# Create column names for the working features
desc_list_names = [d[0] for d in Descriptors._descList]
fp_morgan_cols = [f'mfp_{i}' for i in range(1024)]
feature_columns = desc_list_names + fp_morgan_cols

# Generate features
X = pd.DataFrame(np.vstack([generate_rdkit_features(s) for s in train_df['SMILES']]), columns=feature_columns)
X_test = pd.DataFrame(np.vstack([generate_rdkit_features(s) for s in test_df['SMILES']]), columns=feature_columns)

# Robust Data Cleaning
f32_max = np.finfo(np.float32).max
for df in [X, X_test]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[df > f32_max] = np.nan
    df[df < -f32_max] = np.nan

impute_values = X.mean()
X.fillna(impute_values, inplace=True)
X_test = X_test.reindex(columns=X.columns).fillna(impute_values)
print(f"Feature generation complete. Shape of X: {X.shape}")


# --- STACKING MODEL TRAINING ---
# The modeling pipeline remains the same
XGB_PARAMS = {
    'n_estimators': 2000, 'learning_rate': 0.02, 'max_depth': 6, 'subsample': 0.7, 
    'colsample_bytree': 0.6, 'random_state': RANDOM_STATE, 'n_jobs': -1, 'tree_method': 'hist'
}
LGBM_PARAMS = {
    'n_estimators': 2000, 'learning_rate': 0.02, 'max_depth': 7, 'subsample': 0.7,
    'colsample_bytree': 0.6, 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': -1
}
META_MODEL = Ridge(alpha=1.0, random_state=RANDOM_STATE)

final_predictions = pd.DataFrame({'id': test_df['id']})
for target in TARGET_VARIABLES:
    print(f"--- Stacking models for target: {target} ---")
    y = train_df[target].dropna()
    X_subset = X.loc[y.index]
    
    oof_preds_xgb = pd.Series(np.zeros(len(X_subset)), index=X_subset.index)
    oof_preds_lgb = pd.Series(np.zeros(len(X_subset)), index=X_subset.index)
    test_preds_xgb_folds = np.zeros((len(X_test), N_SPLITS))
    test_preds_lgb_folds = np.zeros((len(X_test), N_SPLITS))
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_subset, y)):
        print(f"  Fold {fold+1}/{N_SPLITS}...")
        X_train, y_train = X_subset.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X_subset.iloc[val_idx], y.iloc[val_idx]
        
        xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[xgb.callback.EarlyStopping(rounds=50, save_best=True)], verbose=0)
        oof_preds_xgb.iloc[val_idx] = xgb_model.predict(X_val)
        test_preds_xgb_folds[:, fold] = xgb_model.predict(X_test)

        lgb_model = lgb.LGBMRegressor(**LGBM_PARAMS)
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        oof_preds_lgb.iloc[val_idx] = lgb_model.predict(X_val)
        test_preds_lgb_folds[:, fold] = lgb_model.predict(X_test)
        
    X_meta_train = pd.concat([oof_preds_xgb, oof_preds_lgb], axis=1)
    X_meta_test = pd.DataFrame({
        'xgb': np.mean(test_preds_xgb_folds, axis=1),
        'lgb': np.mean(test_preds_lgb_folds, axis=1)
    })
    
    print("  Training meta-model...")
    meta_model = META_MODEL
    meta_model.fit(X_meta_train, y)
    final_predictions[target] = meta_model.predict(X_meta_test)

# --- SUBMISSION ---
final_predictions = final_predictions[sample_df.columns]
final_predictions.to_csv('submission.csv', index=False)
print("\nSubmission file created using a stable feature set and stacking strategy.")




