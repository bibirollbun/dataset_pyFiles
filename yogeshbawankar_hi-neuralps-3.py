# Install RDKit from the competition's provided wheel file
!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl -q

# Install all other packages from your utility notebook's output directory
!pip install --no-index --find-links=/kaggle/input/neuralps-pip-installs/pip_packages torch torch_geometric optuna lightgbm -q


print("Importing libraries...")
import os
import gc
import re
import warnings
import random
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import joblib # Used for saving models

# --- Machine Learning ---
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# --- RDKit for Cheminformatics ---
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

# --- PyTorch & PyG for GNNs and CNNs ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool

# --- Suppress Warnings ---
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

print("Libraries imported successfully.")


class CFG:
    # --- General ---
    seed = 42
    data_path = "/kaggle/input/neurips-open-polymer-prediction-2025/"
    output_path = "./"
    
    # --- Model Training ---
    target_cols = ["Tg", "FFV", "Tc", "Density", "Rg"]
    n_folds = 5
    
    # --- GNN Parameters ---
    gnn_epochs = 100
    gnn_batch_size = 64
    gnn_lr = 1e-3
    gnn_patience = 10
    
    # --- 1D-CNN Parameters (NEW) ---
    cnn_epochs = 75
    cnn_batch_size = 128
    cnn_lr = 1e-3
    cnn_patience = 10
    
    # --- LightGBM + Optuna Parameters ---
    optuna_trials = 30 # For a serious run, consider increasing to 50-100
    
    # --- Stacking Meta-Model Parameters (NEW) ---
    meta_model_params = {
        'objective': 'mae',
        'metric': 'mae',
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 16,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

# --- Seeding for reproducibility ---
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

# --- Data Loading (Unchanged) ---
print("Loading data...")
train_df = pd.read_csv(os.path.join(CFG.data_path, "train.csv"))
test_df = pd.read_csv(os.path.join(CFG.data_path, "test.csv"))
supplement_path = os.path.join(CFG.data_path, "train_supplement/")
dataset1 = pd.read_csv(os.path.join(supplement_path, "dataset1.csv"))
dataset3 = pd.read_csv(os.path.join(supplement_path, "dataset3.csv"))
dataset4 = pd.read_csv(os.path.join(supplement_path, "dataset4.csv"))
dataset1.rename(columns={"TC_mean": "Tc"}, inplace=True)
train_df = pd.concat([train_df, dataset1, dataset3, dataset4], ignore_index=True)
train_df.dropna(subset=CFG.target_cols, how='all', inplace=True)

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")


# This list includes all 200+ RDKit descriptor names
DESCRIPTOR_NAMES = [d[0] for d in Descriptors._descList]

def get_2d_features(mol):
    """Calculates 2D RDKit descriptors and fingerprints for a molecule."""
    features = {}
    # 2D Descriptors
    desc_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(DESCRIPTOR_NAMES)
    descriptors = desc_calculator.CalcDescriptors(mol)
    features = {DESCRIPTOR_NAMES[i]: descriptors[i] for i in range(len(DESCRIPTOR_NAMES))}
    
    # Morgan Fingerprints (ECFP)
    morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    for i in range(2048):
        features[f"morgan_{i}"] = morgan_fp[i]
        
    # MACCS Keys
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    for i in range(167):
        features[f"maccs_{i}"] = maccs_fp[i]
        
    return features

def get_3d_features(mol):
    """Generates a 3D conformer and calculates 3D descriptors."""
    features = {}
    try:
        AllChem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)
        All_mol_3D_features = {
            'Asphericity': rdMolDescriptors.CalcAsphericity(mol),
            'Eccentricity': rdMolDescriptors.CalcEccentricity(mol),
            'InertialShapeFactor': rdMolDescriptors.CalcInertialShapeFactor(mol),
            'RadiusOfGyration': rdMolDescriptors.CalcRadiusOfGyration(mol),
            'SpherocityIndex': rdMolDescriptors.CalcSpherocityIndex(mol)
        }
        features.update(All_mol_3D_features)
    except:
        # If conformer generation fails, fill with NaN
        for key in ['Asphericity', 'Eccentricity', 'InertialShapeFactor', 'RadiusOfGyration', 'SpherocityIndex']:
            features[key] = np.nan
    return features

def get_lgbm_features(df):
    """Orchestrates the creation of all features for the LightGBM model."""
    all_features_list = []

    for smiles in tqdm(df['SMILES'], desc="Calculating LGBM Features"):
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            # If RDKit can't parse the SMILES, we create a row of NaNs
            # The number of features might need adjustment if you change feature sets
            all_features_list.append({}) 
            continue

        # Combine all feature types
        features_2d = get_2d_features(mol)
        features_3d = get_3d_features(mol)
        
        # Merge all dictionaries
        all_mol_features = {**features_2d, **features_3d}
        all_features_list.append(all_mol_features)
        
    # Create DataFrame and handle potential issues
    features_df = pd.DataFrame(all_features_list, index=df.index)
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Sanitize column names for LightGBM
    features_df = features_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    return features_df

print("Creating advanced features for LightGBM...")
X_lgbm = get_lgbm_features(train_df)
X_test_lgbm = get_lgbm_features(test_df)
print(f"LGBM features created. Train shape: {X_lgbm.shape}, Test shape: {X_test_lgbm.shape}")




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
    
    # We select a small subset of powerful global descriptors
    global_features = torch.tensor([
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol)
    ], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_features=global_features)

print("Creating graph objects for GNN...")
train_graphs = [smiles_to_graph(s) for s in tqdm(train_df['SMILES'], desc="Train Graphs")]
test_graphs = [smiles_to_graph(s) for s in tqdm(test_df['SMILES'], desc="Test Graphs")]


# --- GNN Architecture ---
class PolymerGNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, global_in_dim, hidden_dim=256, num_heads=8, n_outputs=5):
        super(PolymerGNN, self).__init__()
        self.conv1 = GATv2Conv(node_in_dim, hidden_dim, heads=num_heads, edge_dim=edge_in_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.conv2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, edge_dim=edge_in_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim * num_heads)
        
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * num_heads + global_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_outputs)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch, global_features = data.x, data.edge_index, data.edge_attr, data.batch, data.global_features
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = x.relu()
        x_pooled = global_mean_pool(x, batch)
        
        # Concatenate the learned graph embedding with the global features
        x_combined = torch.cat([x_pooled, global_features], dim=1)
        
        return self.out(x_combined)

# --- 1D-CNN Architecture for SMILES strings ---
class SMILESTokenizer:
    """A simple tokenizer for SMILES strings."""
    def __init__(self, full_smiles_list):
        self.vocab = sorted(list(set("".join(full_smiles_list))))
        self.vocab = ['<pad>', '<unk>'] + self.vocab
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.token_to_id['<pad>']

    def encode(self, smiles, max_length):
        tokens = [self.token_to_id.get(char, self.token_to_id['<unk>']) for char in smiles]
        padding_needed = max_length - len(tokens)
        return tokens + [self.pad_token_id] * padding_needed

class PolymerCNNDataset(Dataset):
    def __init__(self, smiles_list, targets, tokenizer, max_length):
        self.smiles_list = smiles_list
        self.targets = torch.tensor(targets, dtype=torch.float)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        encoded_smiles = self.tokenizer.encode(smiles, self.max_length)
        return torch.tensor(encoded_smiles, dtype=torch.long), self.targets[idx]

class Polymer1DCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_filters=256, kernel_sizes=[3, 5, 7], n_outputs=5):
        super(Polymer1DCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes) * num_filters, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_outputs)
        )

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1) # [B, EmbDim, SeqLen]
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        return self.fc(cat)

# --- GNN Datasets and Collate Functions ---
class PolymerGNNDataset(Dataset):
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

def collate_fn_gnn(batch):
    batch = [b for b in batch if b is not None and b[0] is not None]
    if not batch: return None
    graphs, targets = zip(*batch)
    batch_data = Batch.from_data_list(list(graphs))
    
    global_features_list = [g.global_features for g in graphs]
    batch_data.global_features = torch.stack(global_features_list)
    
    if targets[0] is not None:
        return batch_data, torch.stack(list(targets))
    return batch_data


# --- Placeholders for OOF and Test Predictions ---
oof_lgbm = np.zeros((len(train_df), len(CFG.target_cols)))
test_preds_lgbm = np.zeros((len(test_df), len(CFG.target_cols)))

oof_gnn = np.zeros((len(train_df), len(CFG.target_cols)))
test_preds_gnn = np.zeros((len(test_df), len(CFG.target_cols)))

oof_cnn = np.zeros((len(train_df), len(CFG.target_cols)))
test_preds_cnn = np.zeros((len(test_df), len(CFG.target_cols)))

y = train_df[CFG.target_cols].values
kf = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

# --- 1. LightGBM Training (with Optuna) ---
for i, target in enumerate(CFG.target_cols):
    print(f"\n--- Training LightGBM for target: {target} ---")
    
    target_y = y[:, i]
    valid_indices = ~np.isnan(target_y)
    X_lgbm_valid = X_lgbm[valid_indices]
    y_valid = target_y[valid_indices]
    
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
        for _, (train_idx_opt, val_idx_opt) in enumerate(kf_optuna.split(X_lgbm_valid)):
            model = lgb.LGBMRegressor(**params)
            model.fit(X_lgbm_valid.iloc[train_idx_opt], y_valid[train_idx_opt],
                      eval_set=[(X_lgbm_valid.iloc[val_idx_opt], y_valid[val_idx_opt])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            preds = model.predict(X_lgbm_valid.iloc[val_idx_opt])
            scores.append(mean_absolute_error(y_valid[val_idx_opt], preds))
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=CFG.optuna_trials, show_progress_bar=True)
    best_params = study.best_params
    
    final_params = {'objective': 'mae', 'metric': 'mae', 'n_estimators': 2000,
                    'verbose': -1, 'n_jobs': -1, 'seed': CFG.seed, **best_params}
    
    fold_preds = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_lgbm_valid)):
        model = lgb.LGBMRegressor(**final_params)
        model.fit(X_lgbm_valid.iloc[train_idx], y_valid[train_idx],
                  eval_set=[(X_lgbm_valid.iloc[val_idx], y_valid[val_idx])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
        
        valid_val_indices = np.where(valid_indices)[0][val_idx]
        oof_lgbm[valid_val_indices, i] = model.predict(X_lgbm.iloc[valid_val_indices])
        fold_preds.append(model.predict(X_test_lgbm))
    test_preds_lgbm[:, i] = np.mean(fold_preds, axis=0)

# Utility function for DL model training
def train_dl_model_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data, targets in loader:
        if data is None: continue
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(data)
        mask = ~torch.isnan(targets)
        if mask.sum() == 0: continue
        loss = criterion(preds[mask], targets[mask]).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_dl_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, targets in loader:
            if data is None: continue
            data, targets = data.to(device), targets.to(device)
            preds = model(data)
            mask = ~torch.isnan(targets)
            if mask.sum() == 0: continue
            loss = criterion(preds[mask], targets[mask]).mean()
            total_loss += loss.item()
    return total_loss / len(loader)

# --- 2 & 3. GNN and CNN Training in a Single CV Loop ---
# We prepare data for both models here to use the same folds
smiles_tokenizer = SMILESTokenizer(train_df['SMILES'].tolist() + test_df['SMILES'].tolist())
max_smiles_len = max(len(s) for s in train_df['SMILES'].tolist() + test_df['SMILES'].tolist()) + 2

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    print(f"\n--- Training GNN & CNN Fold {fold+1}/{CFG.n_folds} ---")
    
    # --- GNN Data Prep ---
    train_graphs_fold = [train_graphs[i] for i in train_idx]
    val_graphs_fold = [train_graphs[i] for i in val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    gnn_train_dataset = PolymerGNNDataset(train_graphs_fold, y_train_fold)
    gnn_val_dataset = PolymerGNNDataset(val_graphs_fold, y_val_fold)
    gnn_train_loader = DataLoader(gnn_train_dataset, batch_size=CFG.gnn_batch_size, shuffle=True, collate_fn=collate_fn_gnn)
    gnn_val_loader = DataLoader(gnn_val_dataset, batch_size=CFG.gnn_batch_size, shuffle=False, collate_fn=collate_fn_gnn)
    
    # --- GNN Model Init ---
    # Need to find first valid graph to get dimensions
    first_valid_graph = next(g for g in train_graphs if g is not None)
    node_dim = first_valid_graph.num_node_features
    edge_dim = first_valid_graph.num_edge_features
    global_dim = first_valid_graph.global_features.shape[0]

    gnn_model = PolymerGNN(node_dim, edge_dim, global_dim, n_outputs=len(CFG.target_cols)).to(device)
    gnn_optimizer = optim.Adam(gnn_model.parameters(), lr=CFG.gnn_lr)
    gnn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(gnn_optimizer, 'min', patience=5)
    
    # --- CNN Data Prep ---
    smiles_train_fold = train_df['SMILES'].iloc[train_idx].tolist()
    smiles_val_fold = train_df['SMILES'].iloc[val_idx].tolist()
    cnn_train_dataset = PolymerCNNDataset(smiles_train_fold, y_train_fold, smiles_tokenizer, max_smiles_len)
    cnn_val_dataset = PolymerCNNDataset(smiles_val_fold, y_val_fold, smiles_tokenizer, max_smiles_len)
    cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=CFG.cnn_batch_size, shuffle=True)
    cnn_val_loader = DataLoader(cnn_val_dataset, batch_size=CFG.cnn_batch_size, shuffle=False)

    # --- CNN Model Init ---
    cnn_model = Polymer1DCNN(vocab_size=smiles_tokenizer.vocab_size, n_outputs=len(CFG.target_cols)).to(device)
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=CFG.cnn_lr)
    cnn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(cnn_optimizer, 'min', patience=5)

    criterion = nn.L1Loss(reduction='none') # Same NaN-aware loss for both
    
    # --- Unified Training Loop ---
    # GNN Training
    print("Training GNN...")
    best_gnn_loss = float('inf')
    patience_counter = 0
    for epoch in range(CFG.gnn_epochs):
        train_loss = train_dl_model_epoch(gnn_model, gnn_train_loader, criterion, gnn_optimizer, device)
        val_loss = eval_dl_model(gnn_model, gnn_val_loader, criterion, device)
        gnn_scheduler.step(val_loss)
        if val_loss < best_gnn_loss:
            best_gnn_loss = val_loss
            torch.save(gnn_model.state_dict(), f"best_gnn_model_fold{fold}.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= CFG.gnn_patience:
            print(f"GNN early stopping at epoch {epoch+1}")
            break
            
    # CNN Training
    print("Training 1D-CNN...")
    best_cnn_loss = float('inf')
    patience_counter = 0
    for epoch in range(CFG.cnn_epochs):
        train_loss = train_dl_model_epoch(cnn_model, cnn_train_loader, criterion, cnn_optimizer, device)
        val_loss = eval_dl_model(cnn_model, cnn_val_loader, criterion, device)
        cnn_scheduler.step(val_loss)
        if val_loss < best_cnn_loss:
            best_cnn_loss = val_loss
            torch.save(cnn_model.state_dict(), f"best_cnn_model_fold{fold}.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= CFG.cnn_patience:
            print(f"CNN early stopping at epoch {epoch+1}")
            break
            
    # --- Inference for this fold ---
    # Load best models
    gnn_model.load_state_dict(torch.load(f"best_gnn_model_fold{fold}.pth"))
    cnn_model.load_state_dict(torch.load(f"best_cnn_model_fold{fold}.pth"))
    
    


print("\n--- Training Stacking Meta-Model ---")

final_predictions = pd.DataFrame({'id': test_df['id']})

for i, target in enumerate(CFG.target_cols):
    print(f"Stacking for target: {target}")
    
    target_y = y[:, i]
    valid_indices = ~np.isnan(target_y)
    
    # Features are the OOF predictions from the 3 base models
    X_meta_train = np.vstack([
        oof_lgbm[valid_indices, i], 
        oof_gnn[valid_indices, i],
        oof_cnn[valid_indices, i]
    ]).T
    y_meta_train = target_y[valid_indices]
    
    # Test features are the test predictions from base models
    X_meta_test = np.vstack([
        test_preds_lgbm[:, i], 
        test_preds_gnn[:, i],
        test_preds_cnn[:, i]
    ]).T
    
    # --- OPTION 1: LGBM as Meta-Model ---
    meta_model = lgb.LGBMRegressor(**CFG.meta_model_params)
    meta_model.fit(X_meta_train, y_meta_train)
    final_predictions[target] = meta_model.predict(X_meta_test)
    joblib.dump(meta_model, f'meta_model_{target}.pkl')



print("\n--- Creating Final Submission ---")

# Apply competition-specific post-processing
final_predictions['Tg'] += 273.15

# Save submission file
final_predictions.to_csv("submission.csv", index=False)

print("Submission file created successfully: submission.csv")
print("Final Predictions Head:")
print(final_predictions.head())

# Clean up memory
del gnn_model, cnn_model, X_lgbm, X_test_lgbm, train_graphs, test_graphs
gc.collect()




