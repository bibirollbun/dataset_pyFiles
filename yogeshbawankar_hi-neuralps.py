# Install RDKit from the competition's provided wheel file
!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl -q

# Install all other packages from your utility notebook's output directory
!pip install --no-index --find-links=/kaggle/input/neuralps-pip-installs/pip_packages torch torch_geometric optuna lightgbm -q


print("Imports...")
import os
import gc
import re
import warnings
import random
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

# General ML
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge

# RDKit for molecular processing
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors

# PyTorch and PyG for GNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool

# Suppress warnings
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')


class CFG:
    # General
    seed = 42
    data_path = "/kaggle/input/neurips-open-polymer-prediction-2025/"
    output_path = "./"
    
    # Model Training
    target_cols = ["Tg", "FFV", "Tc", "Density", "Rg"]
    n_folds = 5
    
    # GNN parameters
    gnn_epochs = 100
    gnn_batch_size = 64
    gnn_lr = 1e-3
    gnn_patience = 10 # Patience for early stopping
    
    # LightGBM + Optuna parameters
    optuna_trials = 30 # Increased trials for better search

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CFG.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading data...")
# Main data
train_df = pd.read_csv(os.path.join(CFG.data_path, "train.csv"))
test_df = pd.read_csv(os.path.join(CFG.data_path, "test.csv"))

# Supplementary data
supplement_path = os.path.join(CFG.data_path, "train_supplement/")
dataset1 = pd.read_csv(os.path.join(supplement_path, "dataset1.csv"))
dataset3 = pd.read_csv(os.path.join(supplement_path, "dataset3.csv"))
dataset4 = pd.read_csv(os.path.join(supplement_path, "dataset4.csv"))

# Combine into a single training dataframe
dataset1.rename(columns={"TC_mean": "Tc"}, inplace=True)
train_df = pd.concat([train_df, dataset1, dataset3, dataset4], ignore_index=True)
train_df.dropna(subset=CFG.target_cols, how='all', inplace=True)

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")


def get_lgbm_features(df):
    """Creates a comprehensive feature set for LightGBM."""
    all_features = []
    descriptor_names = [d[0] for d in Descriptors._descList]
    desc_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    for smiles in tqdm(df['SMILES'], desc="Calculating LGBM Features"):
        mol = Chem.MolFromSmiles(smiles)
        features = {}
        
        if mol is None:
            features = {name: np.nan for name in descriptor_names}
            for i in range(2048): features[f"morgan_{i}"] = np.nan
            for i in range(167): features[f"maccs_{i}"] = np.nan
        else:
            descriptors = desc_calculator.CalcDescriptors(mol)
            features = {descriptor_names[i]: descriptors[i] for i in range(len(descriptor_names))}
            
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            for i in range(2048):
                features[f"morgan_{i}"] = morgan_fp[i]
                
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            for i in range(167):
                features[f"maccs_{i}"] = maccs_fp[i]

        all_features.append(features)
        
    return pd.DataFrame(all_features, index=df.index)

print("Creating advanced features for LightGBM...")
X_lgbm = get_lgbm_features(train_df)
X_test_lgbm = get_lgbm_features(test_df)

X_lgbm.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_lgbm.replace([np.inf, -np.inf], np.nan, inplace=True)

X_lgbm = X_lgbm.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
X_test_lgbm = X_test_lgbm.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
print(f"LGBM features created with shape: {X_lgbm.shape}")


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    atom_features = [[
        atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
        atom.GetHybridization(), atom.GetIsAromatic(), atom.GetNumRadicalElectrons()
    ] for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.extend([(i, j), (j, i)])
        bond_type = [
            bond.GetBondType() == Chem.rdchem.BondType.SINGLE,
            bond.GetBondType() == Chem.rdchem.BondType.DOUBLE,
            bond.GetBondType() == Chem.rdchem.BondType.TRIPLE,
            bond.GetBondType() == Chem.rdchem.BondType.AROMATIC,
            bond.IsInRing()
        ]
        edge_attrs.extend([bond_type, bond_type])
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Create graph objects once to save time
print("Creating graph objects for GNN...")
train_graphs = [smiles_to_graph(s) for s in tqdm(train_df['SMILES'], desc="Train Graphs")]
test_graphs = [smiles_to_graph(s) for s in tqdm(test_df['SMILES'], desc="Test Graphs")]




class PolymerDataset(Dataset):
    def __init__(self, graphs, targets=None):
        self.graphs = graphs
        self.is_test = targets is None
        if not self.is_test:
            self.targets = torch.tensor(targets, dtype=torch.float)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        if graph is None: return None
        if self.is_test:
            return graph
        else:
            return graph, self.targets[idx]

def collate_fn_train(batch):
    batch = [b for b in batch if b is not None and b[0] is not None]
    if not batch: return None, None
    graphs, targets = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(targets)

def collate_fn_test(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return Batch.from_data_list(batch)

class PolymerGNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=256, num_heads=8, n_outputs=5):
        super(PolymerGNN, self).__init__()
        self.conv1 = GATv2Conv(node_in_dim, hidden_dim, heads=num_heads, edge_dim=edge_in_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.conv2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, edge_dim=edge_in_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_outputs)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = x.relu()
        x = global_mean_pool(x, batch)
        return self.out(x)



# This section trains the base models and generates the out-of-fold (OOF)
# predictions that will be used to train the stacking model.

kf = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

# Placeholders for OOF predictions and test predictions from base models
oof_lgbm = np.zeros((len(train_df), len(CFG.target_cols)))
test_preds_lgbm = np.zeros((len(test_df), len(CFG.target_cols)))
oof_gnn = np.zeros((len(train_df), len(CFG.target_cols)))
test_preds_gnn = np.zeros((len(test_df), len(CFG.target_cols)))
y = train_df[CFG.target_cols].values


for i, target in enumerate(CFG.target_cols):
    print(f"\n--- Training LightGBM for target: {target} ---")
    
    target_y = y[:, i]
    valid_indices = ~np.isnan(target_y)
    
    def objective(trial):
        params = {
            'objective': 'mae', 'metric': 'mae', 'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'verbose': -1, 'n_jobs': -1, 'seed': CFG.seed,
        }
        
        kf_optuna = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
        scores = []
        for _, (train_idx, val_idx) in enumerate(kf_optuna.split(X_lgbm[valid_indices])):
            model = lgb.LGBMRegressor(**params)
            model.fit(X_lgbm[valid_indices].iloc[train_idx], target_y[valid_indices][train_idx],
                      eval_set=[(X_lgbm[valid_indices].iloc[val_idx], target_y[valid_indices][val_idx])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            preds = model.predict(X_lgbm[valid_indices].iloc[val_idx])
            scores.append(mean_absolute_error(target_y[valid_indices][val_idx], preds))
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=CFG.optuna_trials, show_progress_bar=True)
    best_params = study.best_params
    
    final_params = {'objective': 'mae', 'metric': 'mae', 'n_estimators': 2000,
                    'verbose': -1, 'n_jobs': -1, 'seed': CFG.seed, **best_params}
    
    fold_preds = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_lgbm[valid_indices])):
        model = lgb.LGBMRegressor(**final_params)
        model.fit(X_lgbm[valid_indices].iloc[train_idx], target_y[valid_indices][train_idx],
                  eval_set=[(X_lgbm[valid_indices].iloc[val_idx], target_y[valid_indices][val_idx])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
        
        # Store OOF preds only for the valid indices
        valid_val_indices = np.where(valid_indices)[0][val_idx]
        oof_lgbm[valid_val_indices, i] = model.predict(X_lgbm.iloc[valid_val_indices])
        fold_preds.append(model.predict(X_test_lgbm))
    test_preds_lgbm[:, i] = np.mean(fold_preds, axis=0)


for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    print(f"\n--- Training GNN Fold {fold+1}/{CFG.n_folds} ---")
    
    # Datasets for this fold
    fold_train_graphs = [train_graphs[i] for i in train_idx]
    fold_val_graphs = [train_graphs[i] for i in val_idx]
    fold_train_y = y[train_idx]
    fold_val_y = y[val_idx]

    train_dataset = PolymerDataset(fold_train_graphs, fold_train_y)
    val_dataset = PolymerDataset(fold_val_graphs, fold_val_y)
    train_loader = DataLoader(train_dataset, batch_size=CFG.gnn_batch_size, shuffle=True, collate_fn=collate_fn_train)
    val_loader = DataLoader(val_dataset, batch_size=CFG.gnn_batch_size, shuffle=False, collate_fn=collate_fn_train)

    # Instantiate model for this fold
    node_dim = next(g for g in train_graphs if g is not None).num_node_features
    edge_dim = next(g for g in train_graphs if g is not None).num_edge_features
    gnn_model = PolymerGNN(node_dim, edge_dim, n_outputs=len(CFG.target_cols)).to(device)

    optimizer = optim.Adam(gnn_model.parameters(), lr=CFG.gnn_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    criterion = nn.L1Loss(reduction='none')

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(CFG.gnn_epochs):
        gnn_model.train()
        total_train_loss = 0
        for graphs, targets in train_loader:
            if graphs is None: continue
            graphs, targets = graphs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = gnn_model(graphs)
            mask = ~torch.isnan(targets)
            loss = criterion(preds[mask], targets[mask]).mean()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        gnn_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for graphs, targets in val_loader:
                if graphs is None: continue
                graphs, targets = graphs.to(device), targets.to(device)
                preds = gnn_model(graphs)
                mask = ~torch.isnan(targets)
                loss = criterion(preds[mask], targets[mask]).mean()
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(gnn_model.state_dict(), f"best_gnn_model_fold{fold}.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")

        if patience_counter >= CFG.gnn_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Inference for this fold
    gnn_model.load_state_dict(torch.load(f"best_gnn_model_fold{fold}.pth"))
    gnn_model.eval()
    
    # OOF predictions
    with torch.no_grad():
        val_preds = []
        for graphs, _ in val_loader:
            if graphs is None: continue
            graphs = graphs.to(device)
            val_preds.append(gnn_model(graphs).cpu().numpy())
    oof_gnn[val_idx] = np.concatenate(val_preds)
    
    # Test predictions
    test_dataset_fold = PolymerDataset(test_graphs)
    test_loader_fold = DataLoader(test_dataset_fold, batch_size=CFG.gnn_batch_size, shuffle=False, collate_fn=collate_fn_test)
    with torch.no_grad():
        fold_test_preds = []
        for graphs in test_loader_fold:
            if graphs is None: continue
            graphs = graphs.to(device)
            fold_test_preds.append(gnn_model(graphs).cpu().numpy())
    test_preds_gnn += np.concatenate(fold_test_preds) / CFG.n_folds


print("\n--- Training Stacking Meta-Model ---")

final_predictions = pd.DataFrame({'id': test_df['id']})

for i, target in enumerate(CFG.target_cols):
    print(f"Stacking for target: {target}")
    
    # Prepare data for the meta-model
    target_y = y[:, i]
    valid_indices = ~np.isnan(target_y)
    
    # Features are the OOF predictions from base models
    X_meta_train = np.vstack([oof_lgbm[valid_indices, i], oof_gnn[valid_indices, i]]).T
    y_meta_train = target_y[valid_indices]
    
    # Test features are the test predictions from base models
    X_meta_test = np.vstack([test_preds_lgbm[:, i], test_preds_gnn[:, i]]).T
    
    # Train a simple meta-model
    meta_model = Ridge(alpha=1.0, random_state=CFG.seed)
    meta_model.fit(X_meta_train, y_meta_train)
    
    # Final prediction for this target
    final_predictions[target] = meta_model.predict(X_meta_test)


print("\n--- Creating Final Submission ---")

# Apply competition-specific post-processing
final_predictions['Tg'] += 273.15

# Save submission file
final_predictions.to_csv("submission.csv", index=False)

print("Submission file created successfully: submission.csv")
print("Final Predictions Head:")
print(final_predictions.head())

# Clean up memory
del gnn_model, X_lgbm, X_test_lgbm, train_graphs, test_graphs
gc.collect()

