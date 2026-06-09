!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl -q
!pip install mordred --no-index --find-links=file:///kaggle/input/mordred-1-2-0-py3-none-any/ -q
!pip install lightgbm -q
print("✅ Core libraries installed.")


import pandas as pd
import numpy as np
import gc
import warnings
import os
import random

# --- Scikit-Learn ---

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

# --- GBDT Models (for Rg only) ---
import xgboost as xgb

# --- PyTorch (for all other targets) ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Chemoinformatics ---
from rdkit import Chem
from mordred import Calculator, descriptors

# --- Global Settings ---
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


class CFG:
    N_SPLITS = 5
    SEEDS = [42, 2025]
    TARGET_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

def set_seed(seed):
    """Sets the seed for reproducibility across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

print("✅ Configuration and seeding function defined.")


print("Loading data...")
data_files = {
    'tg': '/kaggle/input/modred-dataset/desc_tg.csv',
    'tc': '/kaggle/input/modred-dataset/desc_tc.csv',
    'rg': '/kaggle/input/modred-dataset/desc_rg.csv',
    'ffv': '/kaggle/input/modred-dataset/desc_ffv.csv',
    'density': '/kaggle/input/modred-dataset/desc_de.csv'
}
data = {name: pd.read_csv(path) for name, path in data_files.items()}
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
ID = test['id']

TRAIN_TABLES = {'Tg': data['tg'], 'FFV': data['ffv'], 'Tc': data['tc'], 'Density': data['density'], 'Rg': data['rg']}

print("Generating test features...")
def generate_test_features(smiles_list):
    """Calculates mordred descriptors for a list of SMILES strings."""
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    df = calc.pandas(mols, quiet=True)
    df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    df.columns = df.columns.map(str)
    return df

desc_test = generate_test_features(test.SMILES.tolist())
print("✅ Data loaded and test features generated.")


def prepare_unsupervised_features(train_df, test_df, target):
    """Prepares features without using the target variable."""
    tr = train_df.select_dtypes(include=[np.number]).copy()
    te = test_df.select_dtypes(include=[np.number]).copy()

    tr = tr[tr[target].notna()].copy()
    y_raw = tr[target].astype(np.float32).values

    feat_cols = sorted(list(set(tr.columns) & set(te.columns)))
    X_df = tr[feat_cols].copy()
    X_test_df = te[feat_cols].copy()

    median_vals = X_df.median()
    X_df.fillna(median_vals, inplace=True)
    X_test_df.fillna(median_vals, inplace=True)

    variances = X_df.var()
    keep_cols = variances[variances > 1e-8].index
    X_df = X_df[keep_cols]
    X_test_df = X_test_df[keep_cols]

    return X_df, X_test_df, y_raw

def select_features_in_fold(Xtr_df, ytr, k=400, corr_th=0.98):
    """Performs supervised feature selection using ONLY fold training data."""
    if Xtr_df.shape[1] <= 800:
        return Xtr_df.columns.tolist()

    sel_f = SelectKBest(f_regression, k=min(k, Xtr_df.shape[1] - 1)).fit(Xtr_df, ytr)
    selected_cols = Xtr_df.columns[sel_f.get_support()]
    Xtr_df_selected = Xtr_df[selected_cols]

    corr = Xtr_df_selected.corr().abs()
    f_vals, _ = f_regression(Xtr_df_selected, ytr)
    strength = pd.Series(f_vals, index=Xtr_df_selected.columns).fillna(0.0)

    ordered_features = strength.sort_values(ascending=False).index
    kept_features = []
    for feature in ordered_features:
        if not kept_features:
            kept_features.append(feature)
            continue
        if not (corr.loc[feature, kept_features] > corr_th).any():
            kept_features.append(feature)

    return kept_features

print("✅ Leak-proof feature engineering functions defined.")


# --- PyTorch Dataset Class ---
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

# --- Stronger MLP Model ---
class EnhancedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256], dropout_rate=0.4):
        super(EnhancedMLP, self).__init__()
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# --- Target Transformation Logic ---
def get_transforms(y, target):
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

# --- XGBoost Training Pipeline (for 'Rg') ---
def run_xgb_pipeline(train_df, test_df, target, random_state):
    set_seed(random_state)
    X_df, X_test_df, y_raw = prepare_unsupervised_features(train_df, test_df, target)
    bins = pd.qcut(y_raw, q=10, labels=False, duplicates='drop')
    splitter = StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=random_state)

    test_preds = []
    oof_preds = np.zeros_like(y_raw, dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X_df, bins), 1):
        print(f"  Fold {fold}/{CFG.N_SPLITS}")
        Xtr_df, Xva_df = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        ytr_raw, yva_raw = y_raw[tr_idx], y_raw[va_idx]

        selected_cols = select_features_in_fold(Xtr_df, ytr_raw)
        Xtr, Xva = Xtr_df[selected_cols].values, Xva_df[selected_cols].values
        X_test_fold = X_test_df[selected_cols].values

        model = xgb.XGBRegressor(random_state=random_state, objective='reg:absoluteerror', tree_method='hist',
                                 n_estimators=2000, learning_rate=0.02, max_depth=6, subsample=0.8,
                                 colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
        model.fit(Xtr, ytr_raw, eval_set=[(Xva, yva_raw)], early_stopping_rounds=100, verbose=False)

        oof_preds[va_idx] = model.predict(Xva)
        test_preds.append(model.predict(X_test_fold))
        print(f"    XGB MAE: {mean_absolute_error(yva_raw, oof_preds[va_idx]):.4f}")

    final_test_preds = np.mean(test_preds, axis=0)
    y_min, y_max = np.percentile(y_raw, [0.5, 99.5])
    return np.clip(final_test_preds, y_min, y_max)


# --- MLP Training Pipeline (for 'Tg', 'FFV', 'Tc', 'Density') ---
def run_mlp_pipeline(train_df, test_df, target, random_state):
    set_seed(random_state)
    X_df, X_test_df, y_raw = prepare_unsupervised_features(train_df, test_df, target)
    _, inv_transform = get_transforms(y_raw, target)
    bins = pd.qcut(y_raw, q=10, labels=False, duplicates='drop')
    splitter = StratifiedKFold(n_splits=CFG.N_SPLITS, shuffle=True, random_state=random_state)

    test_preds = []
    oof_preds = np.zeros_like(y_raw, dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X_df, bins), 1):
        print(f"  Fold {fold}/{CFG.N_SPLITS}")
        Xtr_df, Xva_df = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        ytr_raw, yva_raw = y_raw[tr_idx], y_raw[va_idx]

        ytr_transformed, _ = get_transforms(ytr_raw, target)
        selected_cols = select_features_in_fold(Xtr_df, ytr_raw)
        Xtr, Xva = Xtr_df[selected_cols].values, Xva_df[selected_cols].values
        X_test_fold = X_test_df[selected_cols].values

        scaler = StandardScaler()
        Xtr_s, Xva_s, X_test_s = scaler.fit_transform(Xtr), scaler.transform(Xva), scaler.transform(X_test_fold)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EnhancedMLP(input_dim=Xtr_s.shape[1]).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        criterion = nn.L1Loss()

        train_loader = DataLoader(PolymerDataset(Xtr_s, ytr_transformed), batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(PolymerDataset(Xva_s), batch_size=1024, shuffle=False)
        test_loader = DataLoader(PolymerDataset(X_test_s), batch_size=1024, shuffle=False)

        best_val_mae, patience_counter = float('inf'), 0
        for epoch in range(150):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            val_preds_transformed = []
            with torch.no_grad():
                for X_batch_val in val_loader:
                    val_preds_transformed.extend(model(X_batch_val.to(device)).cpu().numpy().flatten())

            current_val_mae = mean_absolute_error(yva_raw, inv_transform(np.array(val_preds_transformed)))
            scheduler.step(current_val_mae)

            if current_val_mae < best_val_mae:
                best_val_mae = current_val_mae
                patience_counter = 0
                torch.save(model.state_dict(), 'best_mlp.pth')
            else:
                patience_counter += 1
                if patience_counter >= 20: break

        model.load_state_dict(torch.load('best_mlp.pth'))
        model.eval()
        with torch.no_grad():
            oof_mlp_transformed = np.concatenate([model(X_batch.to(device)).cpu().numpy().flatten() for X_batch in val_loader])
            test_mlp_transformed = np.concatenate([model(X_batch.to(device)).cpu().numpy().flatten() for X_batch in test_loader])

        oof_preds[va_idx] = inv_transform(oof_mlp_transformed)
        test_preds.append(inv_transform(test_mlp_transformed))
        print(f"    MLP MAE: {best_val_mae:.4f}")

    final_test_preds = np.mean(test_preds, axis=0)
    y_min, y_max = np.percentile(y_raw, [0.5, 99.5])
    return np.clip(final_test_preds, y_min, y_max)

print("✅ Model pipelines defined.")


final_preds = {}

print("\n=== Initiating Training ===")
for target in CFG.TARGET_COLS:
    print(f"\n[{target}]")
    seed_preds = []
    for seed in CFG.SEEDS:
        print(f"\n--- Training with seed: {seed} ---")
        if target == 'Rg':
            # Use the dedicated XGBoost pipeline for Rg
            preds = run_xgb_pipeline(TRAIN_TABLES[target], desc_test, target, random_state=seed)
        else:
            # Use the dedicated MLP pipeline for all other targets
            preds = run_mlp_pipeline(TRAIN_TABLES[target], desc_test, target, random_state=seed)

        seed_preds.append(preds)
        gc.collect()

    # Average predictions across all seeds for the final result
    final_preds[target] = np.mean(seed_preds, axis=0)
print("\n✅ Training complete.")


submission = pd.DataFrame({'id': ID, **final_preds})

# Apply physical constraints to predictions
submission['FFV'] = np.clip(submission['FFV'], 0.01, 0.99)
submission['Density'] = np.clip(submission['Density'], 0.1, 5.0)

submission.to_csv('submission.csv', index=False)
print(f"\n✅ Submission.csv created: {submission.shape}")
print("Submission file head:")
print(submission.head())

