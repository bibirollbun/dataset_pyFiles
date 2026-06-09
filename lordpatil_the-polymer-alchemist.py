print("### 1. Setting up the environment ###")


import os
import subprocess


try:
    # Using subprocess to capture output and errors cleanly
    rdkit_path = '/kaggle/input/rdkit-install-whl/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
    if os.path.exists(rdkit_path):
        print(f"Installing RDKit from {rdkit_path}...")
        subprocess.run(['pip', 'install', rdkit_path], check=True, capture_output=True)
        print("RDKit installed successfully.")
    else:
        print("RDKit wheel not found, attempting to install from pip.")
        subprocess.run(['pip', 'install', 'rdkit'], check=True, capture_output=True)

except (subprocess.CalledProcessError, FileNotFoundError) as e:
    print(f"Could not install RDKit. Error: {e}")
    # As a fallback, try the older wheel path seen in another notebook
    try:
        rdkit_path_alt = '/kaggle/input/rdkit-install-whl/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
        if os.path.exists(rdkit_path_alt):
            print(f"Trying alternative RDKit path: {rdkit_path_alt}...")
            subprocess.run(['pip', 'install', rdkit_path_alt], check=True, capture_output=True)
            print("Alternative RDKit installed successfully.")
        else:
            print("Alternative RDKit path not found. Proceeding without installation.")
    except Exception as alt_e:
        print(f"Alternative RDKit installation failed. Error: {alt_e}")


import gc
import warnings


import numpy as np
import pandas as pd


from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import Draw
from rdkit import RDLogger


from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb


import matplotlib.pyplot as plt
import seaborn as sns


from tqdm.auto import tqdm


warnings.filterwarnings('ignore')
# Suppress RDKit logs
RDLogger.DisableLog('rdApp.*')
# Environment setup for tokenizers (though not used, good practice)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


print("Setup complete. Libraries imported.")


class Config:
    """Holds all configuration parameters for the notebook."""
    # Data Paths
    COMP_DATA_DIR = '/kaggle/input/neurips-open-polymer-prediction-2025'
    EXTRA_TG_PATH = "/kaggle/input/smiles-tg/Tg_SMILES_class_pid_polyinfo_median.csv"
    EXTRA_TC_PATH = "/kaggle/input/tc-smiles/Tc_SMILES.csv"
    OUTPUT_DIR = '/kaggle/working/'

    # Competition specifics
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    ID_COL = 'id'
    SMILES_COL = 'SMILES'

    # Modeling
    N_SPLITS = 5
    RANDOM_STATE = 42
    MAX_AUTOCORR_DESCRIPTORS = 10 # Limit to prevent feature explosion

    # LGBM Parameters
    LGBM_PARAMS = {
        'objective': 'regression_l1', # MAE
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'num_leaves': 31,
        'max_depth': -1,
        'seed': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
        'colsample_bytree': 0.7,
        'subsample': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }

    # XGBoost Parameters
    XGB_PARAMS = {
        'objective': 'reg:absoluteerror', # MAE
        'eval_metric': 'mae',
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'max_depth': 7,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    }


print("\n### 3. Loading Data and Performing EDA ###")


def load_and_combine_data():
    """Loads competition data and combines it with external datasets."""
    print("Loading competition data...")
    comp_train_df = pd.read_csv(os.path.join(Config.COMP_DATA_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(Config.COMP_DATA_DIR, 'test.csv'))
    print(f"Loaded {len(comp_train_df)} competition training samples and {len(test_df)} test samples.")
    print("Loading additional datasets for Tg and Tc...")
    extra_tg_df = pd.read_csv(Config.EXTRA_TG_PATH)
    extra_tc_df = pd.read_csv(Config.EXTRA_TC_PATH)
    print(f"Loaded {len(extra_tg_df)} additional Tg samples and {len(extra_tc_df)} additional Tc samples.")

    # Prepare extra_tg_df
    extra_tg_clean = extra_tg_df[['SMILES', 'PID', 'Tg']].rename(columns={'PID': 'id'})
    for col in Config.TARGETS:
        if col not in extra_tg_clean.columns:
            extra_tg_clean[col] = np.nan

    # Prepare extra_tc_df
    extra_tc_clean = extra_tc_df[['SMILES', 'TC_mean']].rename(columns={'TC_mean': 'Tc'})
    # Create unique IDs for the new data
    start_id = comp_train_df['id'].max() + 1
    extra_tc_clean['id'] = range(start_id, start_id + len(extra_tc_clean))
    for col in Config.TARGETS:
        if col not in extra_tc_clean.columns:
            extra_tc_clean[col] = np.nan
    
    # Ensure column order matches
    extra_tg_clean = extra_tg_clean[[Config.ID_COL, Config.SMILES_COL] + Config.TARGETS]
    extra_tc_clean = extra_tc_clean[[Config.ID_COL, Config.SMILES_COL] + Config.TARGETS]

    # Combine all datasets
    train_df = pd.concat([comp_train_df, extra_tg_clean, extra_tc_clean], ignore_index=True)
    print(f"Combined dataset has {len(train_df)} total training samples.")
    return train_df, test_df


train_df, test_df = load_and_combine_data()


print("\n--- Performing EDA ---")


print("\nTrain Data Head:")
display(train_df.head())


print("\nTest Data Head:")
display(test_df.head())


print("\nTarget Value Counts (Original + External Data):")
print(train_df[Config.TARGETS].count())


print("\nTarget Distributions:")
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, target in enumerate(Config.TARGETS):
    sns.histplot(train_df[target].dropna(), ax=axes[i], bins=30, kde=True)
    axes[i].set_title(f'Distribution of {target}')
plt.tight_layout()
plt.show()


print("\nCorrelation Heatmap of Targets:")
plt.figure(figsize=(8, 6))
correlation_matrix = train_df[Config.TARGETS].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Target Properties')
plt.show()


print("\nSample Polymer Structures:")
sample_smiles = train_df[Config.SMILES_COL].dropna().sample(8, random_state=Config.RANDOM_STATE).tolist()
mols = [Chem.MolFromSmiles(s) for s in sample_smiles]
img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(300, 300), legends=[f'Sample {i+1}' for i in range(len(mols))])
display(img)


del sample_smiles, mols, img, correlation_matrix
gc.collect()


print("\n### 4. Engineering Features from SMILES ###")


def get_rdkit_descriptors(max_autocorr=Config.MAX_AUTOCORR_DESCRIPTORS):
    """
    Auto-discovers a list of RDKit descriptors, limiting the number of
    AUTOCORR2D descriptors to prevent feature explosion.
    """
    descriptor_list_all = []
    test_mol = Chem.MolFromSmiles('CCO') # A simple molecule for testing descriptor functions
    for name, func in Descriptors.descList:
        try:
            val = func(test_mol)
            if isinstance(val, (int, float)) and not np.isnan(val):
                descriptor_list_all.append((name, func))
        except:
            continue
    autocorr = [(n, f) for n, f in descriptor_list_all if n.startswith('AUTOCORR2D_')]
    autocorr.sort(key=lambda x: int(x[0].split('_')[-1]))
    other = [(n, f) for n, f in descriptor_list_all if not n.startswith('AUTOCORR2D_')]
    final_descriptors = autocorr[:max_autocorr] + other
    feature_names = [name for name, _ in final_descriptors]
    print(f"Auto-discovered {len(final_descriptors)} RDKit descriptors.")
    return final_descriptors, feature_names


def smiles_to_features(smiles_list, descriptor_functions):
    """Converts a list of SMILES strings to a matrix of molecular features."""
    features = []
    for smiles in tqdm(smiles_list, desc="Calculating RDKit Features"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            features.append([np.nan] * len(descriptor_functions))
            continue
        
        mol_features = []
        for name, func in descriptor_functions:
            try:
                val = func(mol)
                if np.isinf(val) or abs(val) > 1e10: # Handle extreme values
                    val = np.nan
                mol_features.append(val)
            except:
                mol_features.append(np.nan)
        features.append(mol_features)
    
    return np.array(features, dtype=float)


def clean_features(X):
    """Imputes missing values in the feature matrix with column medians."""
    X_clean = X.copy()
    X_clean[np.isinf(X_clean)] = np.nan
    
    missing_before = np.isnan(X_clean).sum()
    if missing_before > 0:
        print(f"Imputing {missing_before} missing values...")
        for i in range(X_clean.shape[1]):
            col = X_clean[:, i]
            if np.isnan(col).any():
                median_val = np.nanmedian(col)
                if np.isnan(median_val): # If entire column is NaN
                    median_val = 0
                X_clean[np.isnan(col), i] = median_val
    return X_clean


descriptor_funcs, feature_names = get_rdkit_descriptors()
X_train_raw = smiles_to_features(train_df[Config.SMILES_COL].values, descriptor_funcs)
X_test_raw = smiles_to_features(test_df[Config.SMILES_COL].values, descriptor_funcs)


X_train = clean_features(X_train_raw)
X_test = clean_features(X_test_raw)


X_train = pd.DataFrame(X_train, columns=feature_names)
X_test = pd.DataFrame(X_test, columns=feature_names)


print(f"Feature engineering complete. Shape of training features: {X_train.shape}")


del X_train_raw, X_test_raw
gc.collect()


print("\n### 5. Training Models ###")


def check_gpu_availability():
    """Check for GPU and return appropriate device parameters for XGBoost."""
    try:
        xgb.XGBRegressor(tree_method='gpu_hist')
        print("GPU detected! Using 'gpu_hist' for XGBoost.")
        return {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor', 'gpu_id': 0}
    except Exception:
        print("GPU not available. Using 'hist' (CPU) for XGBoost.")
        return {'tree_method': 'hist', 'predictor': 'cpu_predictor'}


def train_predict(model_type, X_train, y_train, X_test):
    """
    Trains a model (LGBM or XGB) for a single target using K-Fold CV.
    
    Returns:
        - oof_preds: Out-of-fold predictions on the training data.
        - test_preds: Averaged predictions on the test data.
    """
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    kf = KFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.RANDOM_STATE)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"--- Fold {fold+1}/{Config.N_SPLITS} ---")
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Scale features per fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        X_test_scaled = scaler.transform(X_test)
        if model_type == 'lgbm':
            model = lgb.LGBMRegressor(**Config.LGBM_PARAMS)
            model.fit(X_train_scaled, y_train_fold,
                      eval_set=[(X_val_scaled, y_val_fold)],
                      eval_metric='mae',
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        
        elif model_type == 'xgb':
            gpu_params = check_gpu_availability()
            params = {**Config.XGB_PARAMS, **gpu_params}
            model = xgb.XGBRegressor(**params)
            model.fit(X_train_scaled, y_train_fold,
                      eval_set=[(X_val_scaled, y_val_fold)],
                      early_stopping_rounds=100,
                      verbose=False)
        else:
            raise ValueError("model_type must be 'lgbm' or 'xgb'")
        oof_preds[val_idx] = model.predict(X_val_scaled)
        test_preds += model.predict(X_test_scaled) / Config.N_SPLITS
    return oof_preds, test_preds


lgbm_predictions = pd.DataFrame({Config.ID_COL: test_df[Config.ID_COL]})
xgb_predictions = pd.DataFrame({Config.ID_COL: test_df[Config.ID_COL]})


for target in Config.TARGETS:
    print(f"\n===== Training models for target: {target} =====")
    
    # Filter data for the current target
    mask = train_df[target].notna()
    X_target = X_train[mask]
    y_target = train_df[target][mask]
    
    print(f"Training on {len(X_target)} samples.")
    
    # Train LGBM
    print(f"\n--- Training LightGBM for {target} ---")
    _, lgbm_test_preds = train_predict('lgbm', X_target, y_target, X_test)
    lgbm_predictions[target] = lgbm_test_preds
    
    # Train XGB
    print(f"\n--- Training XGBoost for {target} ---")
    _, xgb_test_preds = train_predict('xgb', X_target, y_target, X_test)
    xgb_predictions[target] = xgb_test_preds
    gc.collect()


print("\nModel training complete for all targets.")


print("\n### 6. Ensembling Predictions and Creating Submission ###")


ensemble_predictions = pd.DataFrame({Config.ID_COL: test_df[Config.ID_COL]})
for target in Config.TARGETS:
    ensemble_predictions[target] = (lgbm_predictions[target] + xgb_predictions[target]) / 2


submission_path = os.path.join(Config.OUTPUT_DIR, 'submission.csv')
ensemble_predictions.to_csv(submission_path, index=False)


print(f"\nSubmission file created at: {submission_path}")
print("\nSubmission Head:")
display(ensemble_predictions.head())


print("\nProcess completed successfully!")

