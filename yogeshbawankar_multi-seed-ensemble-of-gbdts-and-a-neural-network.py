!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl
!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl -q
!pip install mordred --no-index --find-links=file:///kaggle/input/mordred-1-2-0-py3-none-any/ -q
!pip install lightgbm xgboost catboost -q
!pip install optuna -q


import pandas as pd
import numpy as np
import gc
import warnings
import os
import random
from typing import List, Dict

# --- SKLearn ---
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.optimize import nnls

# --- Gradient Boosting Models ---
import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xgb
from catboost import CatBoostRegressor

# --- PyTorch for Neural Network ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Chemoinformatics ---
from rdkit import Chem
from mordred import Calculator, descriptors

# --- Initial Setup ---
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


class CFG:
    """Configuration class for storing global parameters."""
    N_SPLITS: int = 5
    SEEDS: List[int] = [42, 2025]
    TARGET_COLS: List[str] = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    BASE_PATH: str = '/kaggle/input/neurips-open-polymer-prediction-2025/'

def set_seed(seed: int) -> None:
    """Sets the random seed for all relevant libraries for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

# Set a global seed for initial operations
set_seed(42)



def generate_mordred_features(smiles_list: List[str]) -> pd.DataFrame:
    """Calculates Mordred chemical descriptors for a list of SMILES strings."""
    print(f"Calculating Mordred descriptors for {len(smiles_list)} SMILES...")
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    
    df = calc.pandas(mols, quiet=True)
    df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    df.columns = df.columns.map(str)
    return df

def prepare_data_for_target(
    target: str,
    use_gmm_augmentation: bool = False,
    gmm_samples: int = 1500
) -> pd.DataFrame:
    """Creates the final training table for a single target by loading, enriching, and augmenting the data."""
    print(f"--- Preparing data for {target} ---")
    
    # 1. Load original data
    train_df = pd.read_csv(CFG.BASE_PATH + 'train.csv')
    sub_df = train_df[['SMILES', target]].dropna().copy()
    print(f"Loaded {len(sub_df)} original non-null samples for {target}.")

    # 2. Augment with randomized SMILES strings
    augmented_smiles, augmented_labels = [], []
    for _, row in sub_df.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if not mol: continue
        
        augmented_smiles.append(row['SMILES'])
        augmented_labels.append(row[target])
        augmented_smiles.append(Chem.MolToSmiles(mol, doRandom=True))
        augmented_labels.append(row[target])

    augmented_df = pd.DataFrame({'SMILES': augmented_smiles, target: augmented_labels})
    print(f"Data size after SMILES augmentation: {len(augmented_df)} samples.")

    # 3. Generate Mordred features
    mordred_df = generate_mordred_features(augmented_df['SMILES'].tolist())
    full_df = pd.concat([augmented_df[target], mordred_df], axis=1)

    # 4. (Optional) Augment with GMM
    if use_gmm_augmentation:
        print(f"Applying GMM augmentation with {gmm_samples} samples...")
        X = full_df.drop(columns=target).fillna(full_df.drop(columns=target).median())
        y = full_df[target]
        
        gmm_df = X.copy()
        gmm_df['Target'] = y.values
        
        gmm = GaussianMixture(n_components=15, random_state=42, max_iter=250, n_init=5)
        gmm.fit(gmm_df)
        
        synthetic_data, _ = gmm.sample(gmm_samples)
        synthetic_df = pd.DataFrame(synthetic_data, columns=gmm_df.columns)
        
        full_df = pd.concat([
            full_df, 
            synthetic_df.rename(columns={'Target': target})
        ], ignore_index=True)
    
    print(f"Final data size for {target}: {len(full_df)} samples.")
    return full_df



def prepare_unsupervised_features(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> tuple:
    """Prepares features without using the target variable (unsupervised)."""
    tr = train_df.select_dtypes(include=[np.number]).copy()
    te = test_df.select_dtypes(include=[np.number]).copy()
    tr = tr[tr[target].notna()].copy()
    y_raw = tr[target].astype(np.float32).values
    
    feat_cols = sorted(list(set(tr.columns) & set(te.columns)))
    X_df, X_test_df = tr[feat_cols].copy(), te[feat_cols].copy()
    
    median_vals = X_df.median()
    X_df.fillna(median_vals, inplace=True)
    X_test_df.fillna(median_vals, inplace=True)
    
    variances = X_df.var()
    keep_cols = variances[variances > 1e-8].index
    return X_df[keep_cols], X_test_df[keep_cols], y_raw

def select_features_in_fold(Xtr_df: pd.DataFrame, ytr: np.ndarray, k: int = 400, corr_th: float = 0.98) -> list:
    """Performs supervised feature selection using ONLY fold training data."""
    if Xtr_df.shape[1] <= 800: return Xtr_df.columns.tolist()
    
    sel_f = SelectKBest(f_regression, k=min(k, Xtr_df.shape[1] - 1)).fit(Xtr_df, ytr)
    selected_cols = Xtr_df.columns[sel_f.get_support()]
    
    corr = Xtr_df[selected_cols].corr().abs()
    f_vals, _ = f_regression(Xtr_df[selected_cols], ytr)
    strength = pd.Series(f_vals, index=selected_cols).fillna(0.0)
    
    ordered_features = strength.sort_values(ascending=False).index
    kept_features = []
    for feature in ordered_features:
        if not kept_features or not (corr.loc[feature, kept_features] > corr_th).any():
            kept_features.append(feature)
            
    return kept_features

def get_transforms(y: np.ndarray, target: str) -> tuple:
    """Gets target transformation and its inverse based on the target name."""
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
    return y, lambda z: z


class PolymerDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.is_test = y is None
        self.y = torch.tensor(y, dtype=torch.float32) if not self.is_test else torch.zeros(len(X), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if not self.is_test:
            return self.X[idx], self.y[idx].unsqueeze(-1)
        return self.X[idx]

class ImprovedMLP(nn.Module):
    """
    An improved MLP with a deeper architecture, SiLU activation, and a residual connection.
    """
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout_rate=0.4):
        super(ImprovedMLP, self).__init__()
        
        layers = []
        current_dim = input_dim
        # Create sequential blocks: Linear -> BatchNorm -> Activation -> Dropout
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.SiLU(), # Efficient Swish-like activation
                nn.Dropout(dropout_rate)
            ])
            current_dim = h_dim
            
        layers.append(nn.Linear(current_dim, 1)) # Final output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def run_training_pipeline(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    target: str, 
    random_state: int, 
    lgb_params: Dict, 
    xgb_params: Dict, 
    cat_params: Dict
) -> np.ndarray:
    """Runs the full 4-model training and blending pipeline for a single seed."""
    set_seed(random_state)
    
    X_df, X_test_df, y_raw = prepare_unsupervised_features(train_df, test_df, target)
    _, inv_transform = get_transforms(y_raw, target)
    
    bins = pd.qcut(y_raw, q=10, labels=False, duplicates='drop')
    splitter = StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=random_state)
    
    oof_preds = {name: np.zeros_like(y_raw, dtype=float) for name in ['lgb', 'xgb', 'cat', 'mlp']}
    test_preds = {name: [] for name in ['lgb', 'xgb', 'cat', 'mlp']}
    
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X_df, bins), 1):
        print(f"  Fold {fold}/{CFG.N_SPLITS}")
        Xtr_df, Xva_df = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        ytr_raw, yva_raw = y_raw[tr_idx], y_raw[va_idx]
        
        selected_cols = select_features_in_fold(Xtr_df, ytr_raw)
        Xtr, Xva, X_test_fold = Xtr_df[selected_cols].values, Xva_df[selected_cols].values, X_test_df[selected_cols].values
        
        # --- GBDT Models ---
        lgb_model = LGBMRegressor(**lgb_params, random_state=random_state, n_estimators=3000)
        lgb_model.fit(Xtr, ytr_raw, eval_set=[(Xva, yva_raw)], callbacks=[lgb.early_stopping(100, verbose=False)])
        
        xgb_model = xgb.XGBRegressor(**xgb_params, random_state=random_state, n_estimators=3000)
        xgb_model.fit(Xtr, ytr_raw, eval_set=[(Xva, yva_raw)], early_stopping_rounds=100, verbose=False)
        
        cat_model = CatBoostRegressor(**cat_params, random_seed=random_state, iterations=4000)
        cat_model.fit(Xtr, ytr_raw, eval_set=(Xva, yva_raw), early_stopping_rounds=100)

        # --- PyTorch MLP ---
        scaler = StandardScaler()
        Xtr_s, Xva_s, X_test_s = scaler.fit_transform(Xtr), scaler.transform(Xva), scaler.transform(X_test_fold)
        ytr_mlp, _ = get_transforms(ytr_raw, target)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ImprovedMLP(input_dim=Xtr_s.shape[1]).to(device)
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
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad(); outputs = model(X_batch); loss = criterion(outputs, y_batch); loss.backward(); optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_preds_t = np.concatenate([model(X_batch_val.to(device)).cpu().numpy().flatten() for X_batch_val in val_loader])
            
            current_val_mae = mean_absolute_error(yva_raw, inv_transform(val_preds_t))
            scheduler.step(current_val_mae)
            
            if current_val_mae < best_val_mae:
                best_val_mae, patience_counter = current_val_mae, 0
                torch.save(model.state_dict(), 'best_mlp.pth')
            else:
                patience_counter += 1
                if patience_counter >= 20: break
        
        model.load_state_dict(torch.load('best_mlp.pth'))
        model.eval()
        
        # Also wrap final predictions in torch.no_grad()
        with torch.no_grad():
            oof_mlp_t = np.concatenate([model(X.to(device)).cpu().numpy().flatten() for X in val_loader])
            test_mlp_t = np.concatenate([model(X.to(device)).cpu().numpy().flatten() for X in test_loader])
        
        
        # --- Collect and Calibrate Predictions ---
        fold_preds = {
            'lgb': lgb_model.predict(Xva), 'xgb': xgb_model.predict(Xva),
            'cat': cat_model.predict(Xva), 'mlp': inv_transform(oof_mlp_t)
        }
        
        for name, oof_fold in fold_preds.items():
            oof_preds[name][va_idx] = oof_fold
            ir = IsotonicRegression(out_of_bounds="clip").fit(oof_fold, yva_raw)
            
            if name == 'mlp': raw_test_pred = inv_transform(test_mlp_t)
            else: raw_test_pred = {'lgb': lgb_model, 'xgb': xgb_model, 'cat': cat_model}[name].predict(X_test_fold)
            test_preds[name].append(ir.predict(raw_test_pred))

        print(f"    Scores -> LGB: {mean_absolute_error(yva_raw, fold_preds['lgb']):.4f} | XGB: {mean_absolute_error(yva_raw, fold_preds['xgb']):.4f} | CAT: {mean_absolute_error(yva_raw, fold_preds['cat']):.4f} | MLP: {best_val_mae:.4f}")

    # --- Final Blending  ---
    final_test_preds = {name: np.mean(preds, axis=0) for name, preds in test_preds.items()}
    oof_stack = np.column_stack(list(oof_preds.values()))
    weights, _ = nnls(oof_stack, y_raw)
    weights /= weights.sum()
    print(f"  Blend weights: LGB={weights[0]:.3f}, XGB={weights[1]:.3f}, CAT={weights[2]:.3f}, MLP={weights[3]:.3f}")
    
    test_stack = np.column_stack(list(final_test_preds.values()))
    ensemble_preds = test_stack @ weights
    
    y_min, y_max = np.percentile(y_raw, [0.5, 99.5])
    return np.clip(ensemble_preds, y_min, y_max)


if __name__ == "__main__":
    # --- Load Test Data and Generate Features ---
    test_base = pd.read_csv(CFG.BASE_PATH + 'test.csv')
    test_mordred_df = generate_mordred_features(test_base.SMILES.tolist())

    # --- Main Training Loop ---
    final_preds = {}
    for target in CFG.TARGET_COLS:
        # 1. Prepare augmented data. GMM is disabled for a stable baseline.
        # To enable it, set use_gmm_augmentation=True
        train_table = prepare_data_for_target(target, use_gmm_augmentation=False)
        
        # 2. Define model parameters (placeholders for a full hyperparameter search)
        default_lgbm_params = {'objective': 'mae', 'metric': 'l1', 'learning_rate': 0.02, 'num_leaves': 31, 'verbosity': -1}
        default_xgb_params = {'objective': 'reg:absoluteerror', 'tree_method': 'hist', 'learning_rate': 0.02, 'max_depth': 6}
        default_cat_params = {'loss_function': 'MAE', 'learning_rate': 0.03, 'depth': 6, 'verbose': False}
        
        # 3. Run the pipeline for each seed and average results
        seed_preds = []
        for seed in CFG.SEEDS:
            print(f"\n--- Training {target} with seed: {seed} ---")
            preds = run_training_pipeline(
                train_df=train_table,
                test_df=test_mordred_df,
                target=target,
                random_state=seed,
                lgb_params=default_lgbm_params,
                xgb_params=default_xgb_params,
                cat_params=default_cat_params
            )
            seed_preds.append(preds)
            gc.collect()
            
        final_preds[target] = np.mean(seed_preds, axis=0)

    # --- Create Submission File ---
    submission = pd.DataFrame({'id': test_base['id'], **final_preds})
    submission['FFV'] = np.clip(submission['FFV'], 0.01, 0.99)
    submission['Density'] = np.clip(submission['Density'], 0.1, 5.0)
    submission.to_csv('submission.csv', index=False)

    print(f"\n✅ Submission.csv created successfully!")
    print(submission.head())




