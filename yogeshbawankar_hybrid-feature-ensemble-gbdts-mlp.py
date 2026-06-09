# Install libraries quietly using -q flag
!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl -q
!pip install --no-index --find-links=file:///kaggle/input/mordred-1-2-0-py3-none-any/ mordred -q
!pip install lightgbm -q
!pip install catboost -q
print("✅ Libraries installed.")


import pandas as pd
import numpy as np
import os
import random
import warnings
import gc

# Chemoinformatics
from rdkit import Chem
from mordred import Calculator, descriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Scikit-learn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import nnls

# Modeling
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Global Settings ---
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


class CFG:
    """
    Configuration class to hold all our settings in one place.
    """
    # Paths to the datasets provided
    BASE_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'
    TC_SMILES_PATH = '/kaggle/input/tc-smiles/Tc_SMILES.csv'
    EXTRA_DATA_PATH = '/kaggle/input/smiles-extra-data/'

    # Modeling settings
    N_SPLITS = 5
    SEEDS = [42, 2025] # Using multiple seeds makes the final prediction more stable
    TARGET_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

def set_seed(seed):
    """
    Sets the random seed for all relevant libraries to ensure our results
    can be reproduced.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

print("✅ Setup and configuration complete.")


print("\nStep 2: Loading and integrating data...")

# --- Load Base Competition Data ---
train = pd.read_csv(os.path.join(CFG.BASE_PATH, 'train.csv'))
test = pd.read_csv(os.path.join(CFG.BASE_PATH, 'test.csv'))
print(f"Original train samples: {len(train)}")

# --- Helper Function for SMILES Cleaning (CORRECTED) ---
def clean_and_validate_smiles(smiles):
    """
    Takes a SMILES string, validates it, and returns a standardized
    (canonical) version. Returns None if invalid.
    
    This corrected version now explicitly checks for and removes SMILES
    with generic '[R]' placeholders.
    """
    if not isinstance(smiles, str) or not smiles:
        return None
        
    # List of known invalid patterns that RDKit cannot handle
    bad_patterns = ['[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]', "[R']"]
    if any(pattern in smiles for pattern in bad_patterns):
        return None
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
        return None
    except:
        return None

train['SMILES'] = train['SMILES'].apply(clean_and_validate_smiles)
test['SMILES'] = test['SMILES'].apply(clean_and_validate_smiles)
train.dropna(subset=['SMILES'], inplace=True)
test.dropna(subset=['SMILES'], inplace=True)

# --- Helper Function for Data Integration ---
def add_external_data(df_train, df_external, target_name):
    """
    Integrates an external dataset with the main training data.
    It fills missing values and adds new, unique polymers.
    """
    print(f"  Processing {len(df_external)} samples for target '{target_name}'...")
    df_external['SMILES'] = df_external['SMILES'].apply(clean_and_validate_smiles)
    df_external.dropna(subset=['SMILES', target_name], inplace=True)

    # Average values for any duplicate SMILES
    df_external = df_external.groupby('SMILES', as_index=False)[target_name].mean()

    # Merge to fill in missing values in the original training data
    df_train = df_train.merge(df_external, on='SMILES', how='left', suffixes=('', '_ext'))
    
    # Prioritize original values, but fill NaNs with external data
    df_train[target_name] = df_train[target_name].fillna(df_train[target_name + '_ext'])
    df_train.drop(columns=[target_name + '_ext'], inplace=True)

    # Add new polymer SMILES from the external dataset that are not in the train set
    new_smiles = df_external[~df_external['SMILES'].isin(df_train['SMILES'])]
    if not new_smiles.empty:
        # Create a temporary DataFrame for new data to avoid column mismatch
        temp_df = pd.DataFrame(new_smiles)
        df_train = pd.concat([df_train, temp_df], ignore_index=True)

    return df_train

# --- Integrate External Datasets ---
# Create a copy to work with
train_extended = train.copy()

# 1. Tc Data
df_tc = pd.read_csv(CFG.TC_SMILES_PATH).rename(columns={'TC_mean': 'Tc'})
train_extended = add_external_data(train_extended, df_tc, 'Tc')

# 2. JCIM Tg Data
df_jcim = pd.read_csv(os.path.join(CFG.EXTRA_DATA_PATH, 'JCIM_sup_bigsmiles.csv'))
df_jcim = df_jcim[['SMILES', 'Tg (C)']].rename(columns={'Tg (C)': 'Tg'})
train_extended = add_external_data(train_extended, df_jcim, 'Tg')

# 3. Excel Tg Data (converting from Kelvin to Celsius)
df_tg_xls = pd.read_excel(os.path.join(CFG.EXTRA_DATA_PATH, 'data_tg3.xlsx'))
df_tg_xls = df_tg_xls.rename(columns={'Tg [K]': 'Tg'})
# Ensure Tg column is numeric before subtraction
df_tg_xls['Tg'] = pd.to_numeric(df_tg_xls['Tg'], errors='coerce')
df_tg_xls.dropna(subset=['Tg'], inplace=True)
df_tg_xls['Tg'] = df_tg_xls['Tg'] - 273.15
train_extended = add_external_data(train_extended, df_tg_xls, 'Tg')

# 4. Density Data
df_density = pd.read_excel(os.path.join(CFG.EXTRA_DATA_PATH, 'data_dnst1.xlsx'))
df_density = df_density.rename(columns={'density(g/cm3)': 'Density'})
# Clean non-numeric entries
df_density = df_density[pd.to_numeric(df_density['Density'], errors='coerce').notnull()]
df_density['Density'] = df_density['Density'].astype(float)
train_extended = add_external_data(train_extended, df_density, 'Density')

print(f"Final extended train samples: {len(train_extended)}")
print("✅ Data loading and integration complete.")


print("\nStep 3: Engineering hybrid features...")

def generate_mordred_features(smiles_list):
    """
    Generates a wide range of physicochemical descriptors using the
    'mordred' library. These are calculated properties like molecular
    weight, complexity, etc.
    """
    print("  Generating Mordred descriptors...")
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    df_mordred = calc.pandas(mols, quiet=True)
    df_mordred = df_mordred.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    df_mordred.columns = ["mordred_" + str(c) for c in df_mordred.columns]
    return df_mordred

def generate_rdkit_fingerprints(smiles_list):
    """
    Generates molecular fingerprints using RDKit. These are like a unique
    'fingerprint' for a molecule's structure, represented as a series of
    0s and 1s. They are great for finding similar structures.
    """
    print("  Generating RDKit fingerprints...")
    # Morgan Fingerprints (ECFP)
    fp_gen = GetMorganGenerator(radius=2, fpSize=512)
    morgan_fps = [fp_gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]
    df_morgan = pd.DataFrame(np.array(morgan_fps, dtype=np.int8), columns=[f'morgan_{i}' for i in range(512)])

    # MACCS Keys
    maccs_fps = [MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s)) for s in smiles_list]
    df_maccs = pd.DataFrame(np.array(maccs_fps, dtype=np.int8), columns=[f'maccs_{i}' for i in range(167)])
    
    return pd.concat([df_morgan, df_maccs], axis=1)

# --- Generate and Combine Features for Train and Test data ---
# This might take a few minutes to run.
df_mordred_train = generate_mordred_features(train_extended['SMILES'].tolist())
df_fp_train = generate_rdkit_fingerprints(train_extended['SMILES'].tolist())

df_mordred_test = generate_mordred_features(test['SMILES'].tolist())
df_fp_test = generate_rdkit_fingerprints(test['SMILES'].tolist())

# Create master feature matrices
X_master = pd.concat([df_mordred_train, df_fp_train], axis=1)
X_test_master = pd.concat([df_mordred_test, df_fp_test], axis=1)

# Align columns to ensure train and test have the same features
common_cols = sorted(list(set(X_master.columns) & set(X_test_master.columns)))
X_master = X_master[common_cols]
X_test_master = X_test_master[common_cols]

# Create master target matrix
y_master = train_extended[CFG.TARGET_COLS]

print(f"  Master training features shape: {X_master.shape}")
print(f"  Master testing features shape: {X_test_master.shape}")
print("✅ Hybrid feature engineering complete.")


print("\nStep 4: Defining the advanced modeling pipeline...")

# --- PyTorch MLP Model Definition ---
class PolymerDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.is_test = y is None
        self.y = torch.tensor(y, dtype=torch.float32) if not self.is_test else torch.zeros(len(X), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.is_test:
            return self.X[idx]
        return self.X[idx], self.y[idx].unsqueeze(-1)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout_rate=0.3):
        super(MLP, self).__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h_dim), nn.BatchNorm1d(h_dim),
                nn.GELU(), nn.Dropout(dropout_rate)
            ])
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# --- Helper functions for the pipeline ---
def get_transforms(y, target):
    """
    Applies a transformation to targets like FFV and Density to make their
    distribution more normal, which helps models learn better.
    Returns the transformed target and a function to reverse the transform.
    """
    if target == "FFV":
        eps = 1e-3
        y_clipped = np.clip(y, eps, 1 - eps)
        transform = lambda x: np.log(x / (1 - x))
        inverse = lambda z: 1.0 / (1.0 + np.exp(-z))
        return transform(y_clipped), inverse
    if target == "Density":
        transform = lambda x: np.log(np.clip(x, 1e-4, None))
        inverse = lambda x: np.exp(x)
        return transform(y), inverse
    return y, lambda z: z # Default: no transform

def run_training_pipeline(X, y, X_test, target, random_state):
    """
    The main training function. It performs a full cross-validated training
    run for a single target property.
    """
    set_seed(random_state)
    
    # --- Stratified Cross-Validation Setup ---
    # Stratification ensures that each fold has a similar distribution of the
    # target variable, making our validation scores more reliable.
    bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
    splitter = StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=random_state)
    
    oof_preds = {name: np.zeros_like(y, dtype=float) for name in ['lgb', 'xgb', 'cat', 'mlp']}
    test_preds = {name: [] for name in ['lgb', 'xgb', 'cat', 'mlp']}
    
    # --- Imputation and Initial Cleaning ---
    median_vals = X.median()
    X = X.fillna(median_vals)
    X_test = X_test.fillna(median_vals)
    
    variances = X.var()
    keep_cols = variances[variances > 1e-8].index
    X = X[keep_cols]
    X_test = X_test[keep_cols]
    
    # Get the inverse transform function for later use
    _, inv_transform = get_transforms(y.values, target)
    
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, bins), 1):
        print(f"    Fold {fold}/{CFG.N_SPLITS}")
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        
        # --- Model Training ---
        # We train three Gradient Boosted Decision Tree models (LGB, XGB, Cat)
        # and one Neural Network (MLP). This diversity is key for a strong ensemble.
        
        # 1. LightGBM
        lgb_model = lgb.LGBMRegressor(random_state=random_state, objective='mae', n_estimators=2000, n_jobs=-1, verbosity=-1)
        lgb_model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
        
        # 2. XGBoost
        xgb_model = xgb.XGBRegressor(random_state=random_state, objective='reg:absoluteerror', tree_method='hist', n_estimators=2000)
        xgb_model.fit(Xtr, ytr, eval_set=[(Xva, yva)], early_stopping_rounds=100, verbose=False)
        
        # 3. CatBoost
        cat_model = CatBoostRegressor(random_seed=random_state, loss_function='MAE', iterations=3000, verbose=False)
        cat_model.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=100)

        # 4. PyTorch MLP
        ytr_mlp, _ = get_transforms(ytr.values, target)
        scaler = StandardScaler()
        Xtr_s, Xva_s, X_test_s = scaler.fit_transform(Xtr), scaler.transform(Xva), scaler.transform(X_test)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP(input_dim=Xtr_s.shape[1]).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        criterion = nn.L1Loss()
        
        train_loader = DataLoader(PolymerDataset(Xtr_s, ytr_mlp), batch_size=256, shuffle=True)
        val_loader = DataLoader(PolymerDataset(Xva_s), batch_size=1024, shuffle=False)
        test_loader = DataLoader(PolymerDataset(X_test_s), batch_size=1024, shuffle=False)
        
        best_val_mae, patience_counter = float('inf'), 0
        for epoch in range(150):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch.to(device))
                loss = criterion(outputs, y_batch.to(device))
                loss.backward()
                optimizer.step()
            
            model.eval()
            val_preds_transformed = []
            with torch.no_grad():
                for X_batch_val in val_loader:
                    val_preds_transformed.extend(model(X_batch_val.to(device)).cpu().numpy().flatten())
            
            current_val_mae = mean_absolute_error(yva, inv_transform(np.array(val_preds_transformed)))
            scheduler.step(current_val_mae)
            
            if current_val_mae < best_val_mae:
                best_val_mae, patience_counter = current_val_mae, 0
                torch.save(model.state_dict(), 'best_mlp.pth')
            else:
                patience_counter += 1
                if patience_counter >= 20: break
        
        model.load_state_dict(torch.load('best_mlp.pth'))
        model.eval()
        with torch.no_grad():
            oof_mlp_transformed = np.concatenate([model(X_batch.to(device)).cpu().numpy() for X_batch in val_loader]).flatten()
            test_mlp_transformed = np.concatenate([model(X_batch.to(device)).cpu().numpy() for X_batch in test_loader]).flatten()
        
        # --- Collect and Calibrate Predictions for the Fold ---
        # Calibration (Isotonic Regression) helps to correct any systematic
        # biases in a model's predictions.
        fold_preds = {
            'lgb': lgb_model.predict(Xva), 'xgb': xgb_model.predict(Xva),
            'cat': cat_model.predict(Xva), 'mlp': inv_transform(oof_mlp_transformed)
        }
        
        test_fold_raw = {
            'lgb': lgb_model.predict(X_test), 'xgb': xgb_model.predict(X_test),
            'cat': cat_model.predict(X_test), 'mlp': inv_transform(test_mlp_transformed)
        }
        
        for name, oof_fold in fold_preds.items():
            oof_preds[name][va_idx] = oof_fold
            ir = IsotonicRegression(out_of_bounds="clip").fit(oof_fold, yva)
            test_preds[name].append(ir.predict(test_fold_raw[name]))
    
    # --- Final Blending ---
    # We use Non-Negative Least Squares (NNLS) to find the optimal weights
    # to combine our four models. This is smarter than simple averaging.
    final_test_preds = {name: np.mean(preds, axis=0) for name, preds in test_preds.items()}
    oof_stack = np.column_stack(list(oof_preds.values()))
    weights, _ = nnls(oof_stack, y)
    weights /= weights.sum()
    print(f"    Blend weights: LGB={weights[0]:.3f}, XGB={weights[1]:.3f}, CAT={weights[2]:.3f}, MLP={weights[3]:.3f}")
    
    test_stack = np.column_stack(list(final_test_preds.values()))
    ensemble_preds = test_stack @ weights
    
    y_min, y_max = np.percentile(y, [0.5, 99.5])
    return np.clip(ensemble_preds, y_min, y_max)

print("✅ Advanced modeling pipeline defined.")


print("\nStep 5: Starting full training and prediction process...")
final_preds = {}

for target in CFG.TARGET_COLS:
    print(f"\n--- Training for target: {target} ---")
    
    # Prepare data for this specific target (dropping rows with no label)
    data_for_target = pd.concat([X_master, y_master], axis=1)
    data_for_target.dropna(subset=[target], inplace=True)
    
    X_target = data_for_target[X_master.columns]
    y_target = data_for_target[target]
    
    seed_preds = []
    for seed in CFG.SEEDS:
        print(f"\n  Running with seed: {seed}")
        preds = run_training_pipeline(X_target, y_target, X_test_master, target, random_state=seed)
        seed_preds.append(preds)
        gc.collect() # Free up memory
        
    # Average predictions across all seeds
    final_preds[target] = np.mean(seed_preds, axis=0)

# --- Create Submission File ---
submission = pd.DataFrame({'id': test['id'], **final_preds})

# Apply physical constraints to predictions
submission['FFV'] = np.clip(submission['FFV'], 0.01, 0.99)
submission['Density'] = np.clip(submission['Density'], 0.1, 5.0)

submission.to_csv('submission.csv', index=False)
print(f"\n✅ Submission.csv created successfully! Shape: {submission.shape}")
print("Final Predictions Head:")
print(submission.head())

