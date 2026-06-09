# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
import os
warnings.filterwarnings('ignore')

# Check if optional libraries are available
RDKIT_AVAILABLE = False
TORCH_GEOMETRIC_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
    RDKIT_AVAILABLE = True
    print("RDKit is available")
except ImportError:
    print("RDKit not available, using simplified molecular representation")

try:
    from torch_geometric.data import Data as GeoData
    from torch_geometric.data import InMemoryDataset
    from torch_geometric.loader import DataLoader as GeoLoader
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
    TORCH_GEOMETRIC_AVAILABLE = True
    print("PyTorch Geometric is available")
except ImportError:
    print("PyTorch Geometric not available, using alternative approach")

# Property names for this competition
PROPERTY_NAMES = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Enhanced molecular features without RDKit
def smiles_to_features_enhanced(smiles):
    """Enhanced SMILES to feature conversion without RDKit"""
    features = []
    
    # Basic counts
    features.extend([
        len(smiles),  # Length
        smiles.count('C'),  # Carbon count
        smiles.count('N'),  # Nitrogen count
        smiles.count('O'),  # Oxygen count
        smiles.count('F'),  # Fluorine count
        smiles.count('S'),  # Sulfur count
        smiles.count('P'),  # Phosphorus count
        smiles.count('Cl'),  # Chlorine count
        smiles.count('Br'),  # Bromine count
        smiles.count('I'),  # Iodine count
        smiles.count('Si'),  # Silicon count
    ])
    
    # Structural features
    features.extend([
        smiles.count('('),  # Branching
        smiles.count('='),  # Double bonds
        smiles.count('#'),  # Triple bonds
        smiles.count('['),  # Special atoms
        smiles.count('@'),  # Chirality
        smiles.count('+'),  # Positive charge
        smiles.count('-'),  # Negative charge
    ])
    
    # Ring features
    ring_counts = [smiles.count(str(i)) for i in range(1, 10)]
    features.extend(ring_counts)
    
    # Aromatic features
    features.extend([
        smiles.count('c'),  # Aromatic carbon
        smiles.count('n'),  # Aromatic nitrogen
        smiles.count('o'),  # Aromatic oxygen
        smiles.count('s'),  # Aromatic sulfur
    ])
    
    # Polymer-specific patterns
    polymer_patterns = {
        'ester': ['COO', 'OOC'],
        'amide': ['CON', 'NOC', 'C(=O)N'],
        'ether': ['COC'],
        'sulfone': ['S(=O)(=O)', 'S(O)(O)'],
        'ketone': ['C(=O)C'],
        'carbonate': ['OC(=O)O'],
        'urethane': ['NC(=O)O', 'OC(=O)N'],
        'imide': ['C(=O)NC(=O)', 'C(=O)N(C(=O))'],
    }
    
    for pattern_name, patterns in polymer_patterns.items():
        count = sum(smiles.count(p) for p in patterns)
        features.append(count)
    
    # Ratios and derived features
    total_atoms = max(1, sum([smiles.count(atom) for atom in ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si']]))
    features.extend([
        features[1] / total_atoms if total_atoms > 0 else 0,  # C ratio
        features[2] / total_atoms if total_atoms > 0 else 0,  # N ratio
        features[3] / total_atoms if total_atoms > 0 else 0,  # O ratio
        (features[4] + features[7] + features[8] + features[9]) / total_atoms if total_atoms > 0 else 0,  # Halogen ratio
        features[12] / max(1, features[0]),  # Branching density
        (features[13] + features[14]) / max(1, features[0]),  # Unsaturation density
        sum(ring_counts) / max(1, features[0]),  # Ring density
    ])
    
    return features

# Enhanced RDKit features
def smiles_to_rdkit_features(smiles):
    """Extract comprehensive RDKit features"""
    if not RDKIT_AVAILABLE:
        return []
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        features = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.RingCount(mol),
            Descriptors.FractionCsp3(mol),
            Descriptors.NumAromaticCarbocycles(mol),
            Descriptors.NumAromaticHeterocycles(mol),
            Descriptors.BalabanJ(mol),
            Descriptors.BertzCT(mol),
            Descriptors.Chi0(mol),
            Descriptors.HallKierAlpha(mol),
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
        ]
        return features
    except:
        return []

# Combined feature extraction
def extract_all_features(smiles):
    """Extract all available features"""
    features = smiles_to_features_enhanced(smiles)
    rdkit_features = smiles_to_rdkit_features(smiles)
    return features + rdkit_features

# Dataset classes
class EnhancedFeatureDataset(Dataset):
    """Dataset with enhanced features"""
    def __init__(self, smiles_list, targets=None):
        self.features = []
        self.targets = targets
        
        print("Extracting features...")
        for i, smiles in enumerate(smiles_list):
            if i % 1000 == 0:
                print(f"  Processing {i}/{len(smiles_list)}")
            feat = extract_all_features(smiles)
            self.features.append(feat)
        
        # Pad features to same length
        max_len = max(len(f) for f in self.features)
        self.features = [f + [0] * (max_len - len(f)) for f in self.features]
        
        self.features = torch.tensor(self.features, dtype=torch.float32)
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        
        print(f"Feature shape: {self.features.shape}")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]

# Multi-task MLP with separate heads for each property
class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, shared_dims=[512, 256], task_dims=[128, 64], dropout=0.3):
        super().__init__()
        
        # Shared layers
        self.shared_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for dim in shared_dims:
            self.shared_layers.append(nn.Linear(prev_dim, dim))
            self.shared_layers.append(nn.BatchNorm1d(dim))
            self.shared_layers.append(nn.ReLU())
            self.shared_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for prop in PROPERTY_NAMES:
            task_layers = []
            task_prev_dim = prev_dim
            
            for dim in task_dims:
                task_layers.extend([
                    nn.Linear(task_prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                task_prev_dim = dim
            
            task_layers.append(nn.Linear(task_prev_dim, 1))
            self.task_heads[prop] = nn.Sequential(*task_layers)
    
    def forward(self, x, task=None):
        # Shared representation
        for layer in self.shared_layers:
            x = layer(x)
        
        # Task-specific predictions
        if task is not None:
            return self.task_heads[task](x).squeeze(-1)
        else:
            outputs = {}
            for prop in PROPERTY_NAMES:
                outputs[prop] = self.task_heads[prop](x).squeeze(-1)
            return outputs

# Training function for multi-task learning
def train_multitask_model(model, train_data, val_data, device, epochs=150, lr=0.001, patience=25):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Create data loaders for each property
    train_loaders = {}
    val_loaders = {}
    
    for prop in PROPERTY_NAMES:
        if prop in train_data:
            train_loaders[prop] = DataLoader(
                train_data[prop], batch_size=64, shuffle=True, num_workers=2
            )
            val_loaders[prop] = DataLoader(
                val_data[prop], batch_size=64, shuffle=False
            )
    
    print(f"Training on properties: {list(train_loaders.keys())}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        n_batches = 0
        
        # Train on each property
        for prop, loader in train_loaders.items():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                out = model(batch_x, task=prop)
                
                # Mask out NaN values if any
                mask = ~torch.isnan(batch_y)
                if mask.sum() > 0:
                    loss = F.mse_loss(out[mask], batch_y[mask])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    n_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0
        val_metrics = {}
        n_val_batches = 0
        
        with torch.no_grad():
            for prop, loader in val_loaders.items():
                prop_preds = []
                prop_targets = []
                
                for batch_x, batch_y in loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    out = model(batch_x, task=prop)
                    
                    mask = ~torch.isnan(batch_y)
                    if mask.sum() > 0:
                        loss = F.mse_loss(out[mask], batch_y[mask])
                        val_loss += loss.item()
                        n_val_batches += 1
                        
                        prop_preds.extend(out[mask].cpu().numpy())
                        prop_targets.extend(batch_y[mask].cpu().numpy())
                
                if len(prop_preds) > 0:
                    mae = mean_absolute_error(prop_targets, prop_preds)
                    rmse = np.sqrt(mean_squared_error(prop_targets, prop_preds))
                    val_metrics[prop] = {'MAE': mae, 'RMSE': rmse}
        
        if n_batches > 0:
            train_loss /= n_batches
        if n_val_batches > 0:
            val_loss /= n_val_batches
            
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_multitask_model.pth')
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if val_metrics:
                print("Validation Metrics:")
                for prop, metrics in val_metrics.items():
                    print(f"  {prop}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_multitask_model.pth'))
    
    return train_losses, val_losses

def create_property_datasets(features, targets, properties, indices):
    """Create separate datasets for each property"""
    datasets = {}
    
    for i, prop in enumerate(PROPERTY_NAMES):
        # Find samples with valid values for this property
        valid_mask = ~np.isnan(targets[:, i])
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            # Get features and targets for valid samples
            prop_features = features[valid_indices]
            prop_targets = targets[valid_indices, i]
            
            # Create dataset
            datasets[prop] = list(zip(
                [prop_features[j] for j in range(len(prop_features))],
                [prop_targets[j] for j in range(len(prop_targets))]
            ))
            
            print(f"  {prop}: {len(valid_indices)} samples")
    
    return datasets

def main():
    print("=== Enhanced Polymer Property Prediction ===")
    print(f"Target properties: {PROPERTY_NAMES}")
    print(f"Available libraries: RDKit={RDKIT_AVAILABLE}, PyTorch Geometric={TORCH_GEOMETRIC_AVAILABLE}")
    
    # Set paths
    base_path = "/kaggle/input/neurips-open-polymer-prediction-2025"
    train_file = f"{base_path}/train.csv"
    test_file = f"{base_path}/test.csv"
    
    supplement_files = [
        f"{base_path}/train_supplement/dataset1.csv",
        f"{base_path}/train_supplement/dataset2.csv",
        f"{base_path}/train_supplement/dataset3.csv",
        f"{base_path}/train_supplement/dataset4.csv"
    ]
    
    # Load main training data
    print("\nLoading data...")
    df = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    # Check column names and standardize
    if 'smiles' in df.columns:
        df = df.rename(columns={'smiles': 'SMILES'})
    if 'smiles' in df_test.columns:
        df_test = df_test.rename(columns={'smiles': 'SMILES'})
    
    # Load supplemental datasets
    all_dfs = [df]
    for supp_file in supplement_files:
        if os.path.exists(supp_file):
            try:
                supp_df = pd.read_csv(supp_file)
                # Check if it has the right columns
                if 'SMILES' in supp_df.columns or 'smiles' in supp_df.columns:
                    # Standardize column names
                    if 'smiles' in supp_df.columns:
                        supp_df = supp_df.rename(columns={'smiles': 'SMILES'})
                    
                    # Check for property columns
                    has_properties = any(prop in supp_df.columns for prop in PROPERTY_NAMES)
                    if has_properties:
                        all_dfs.append(supp_df)
                        print(f"  Loaded {supp_file}: {supp_df.shape[0]} samples")
            except Exception as e:
                print(f"  Error loading {supp_file}: {e}")
    
    # Combine all datasets
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined dataset shape: {df.shape}")
    
    # Remove duplicates
    original_size = len(df)
    df = df.drop_duplicates(subset='SMILES', keep='first')
    print(f"After removing duplicates: {df.shape[0]} samples ({original_size - len(df)} duplicates removed)")
    
    # Handle missing values
    print("\nMissing values in target properties:")
    for prop in PROPERTY_NAMES:
        if prop in df.columns:
            missing = df[prop].isna().sum()
            valid = df.shape[0] - missing
            print(f"  {prop}: {missing} missing ({missing/len(df)*100:.1f}%), {valid} valid samples")
    
    # Extract features for all samples
    print("\nExtracting features for all samples...")
    all_smiles = df['SMILES'].tolist()
    all_features = []
    
    for i, smiles in enumerate(all_smiles):
        if i % 1000 == 0:
            print(f"  Processing {i}/{len(all_smiles)}")
        feat = extract_all_features(smiles)
        all_features.append(feat)
    
    # Pad features to same length
    max_len = max(len(f) for f in all_features)
    all_features = np.array([f + [0] * (max_len - len(f)) for f in all_features])
    
    print(f"Feature shape: {all_features.shape}")
    
    # Get target values (with NaN for missing)
    all_targets = df[PROPERTY_NAMES].values
    
    # Create scalers for each property
    scalers = {}
    scaled_targets = np.zeros_like(all_targets)
    
    for i, prop in enumerate(PROPERTY_NAMES):
        valid_mask = ~np.isnan(all_targets[:, i])
        if valid_mask.sum() > 0:
            scaler = StandardScaler()
            scaled_targets[valid_mask, i] = scaler.fit_transform(
                all_targets[valid_mask, i].reshape(-1, 1)
            ).ravel()
            scaled_targets[~valid_mask, i] = np.nan
            scalers[prop] = scaler
        else:
            print(f"Warning: No valid samples for {prop}")
    
    # Split into train/validation
    n_samples = len(all_features)
    indices = np.arange(n_samples)
    train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=42)
    
    X_train = all_features[train_idx]
    y_train = scaled_targets[train_idx]
    X_val = all_features[val_idx]
    y_val = scaled_targets[val_idx]
    
    print(f"\nTrain size: {len(X_train)}, Validation size: {len(X_val)}")
    
    # Create property-specific datasets
    print("\nCreating property-specific datasets:")
    train_datasets = create_property_datasets(X_train, y_train, PROPERTY_NAMES, train_idx)
    val_datasets = create_property_datasets(X_val, y_val, PROPERTY_NAMES, val_idx)
    
    # Convert to PyTorch datasets
    train_data = {}
    val_data = {}
    
    for prop in train_datasets:
        features = torch.tensor([x[0] for x in train_datasets[prop]], dtype=torch.float32)
        targets = torch.tensor([x[1] for x in train_datasets[prop]], dtype=torch.float32)
        train_data[prop] = list(zip(features, targets))
        
        features = torch.tensor([x[0] for x in val_datasets[prop]], dtype=torch.float32)
        targets = torch.tensor([x[1] for x in val_datasets[prop]], dtype=torch.float32)
        val_data[prop] = list(zip(features, targets))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Get feature dimension
    feature_dim = all_features.shape[1]
    print(f"Feature dimension: {feature_dim}")
    
    # Train multi-task model
    print("\n=== Training Multi-Task Model ===")
    model = MultiTaskMLP(input_dim=feature_dim).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_losses, val_losses = train_multitask_model(
        model, train_data, val_data, device, epochs=150, lr=0.001
    )
    
    # Make test predictions
    print("\n=== Making Test Predictions ===")
    test_features = []
    test_smiles = df_test['SMILES'].tolist()
    
    for i, smiles in enumerate(test_smiles):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(test_smiles)}")
        feat = extract_all_features(smiles)
        test_features.append(feat)
    
    # Pad features
    test_features = np.array([f + [0] * (max_len - len(f)) for f in test_features])
    test_features = torch.tensor(test_features, dtype=torch.float32).to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        predictions = model(test_features)
    
    # Convert predictions to numpy and inverse transform
    test_preds = np.zeros((len(test_smiles), len(PROPERTY_NAMES)))
    
    for i, prop in enumerate(PROPERTY_NAMES):
        if prop in predictions and prop in scalers:
            pred_values = predictions[prop].cpu().numpy()
            test_preds[:, i] = scalers[prop].inverse_transform(
                pred_values.reshape(-1, 1)
            ).ravel()
        else:
            # Use median value from training data if no scaler available
            valid_values = all_targets[:, i][~np.isnan(all_targets[:, i])]
            if len(valid_values) > 0:
                test_preds[:, i] = np.median(valid_values)
            else:
                test_preds[:, i] = 0.0
            print(f"Warning: Using default value for {prop}")
    
    # Create submission
    submission_df = df_test[['id']].copy()
    for i, prop in enumerate(PROPERTY_NAMES):
        submission_df[prop] = test_preds[:, i]
    
    # Save submission
    submission_df.to_csv('submission.csv', index=False)
    print("\nSubmission saved as submission.csv")
    
    print("\nPrediction statistics on test set:")
    for i, prop in enumerate(PROPERTY_NAMES):
        pred_values = test_preds[:, i]
        print(f"{prop}: mean={pred_values.mean():.3f}, std={pred_values.std():.3f}, "
              f"min={pred_values.min():.3f}, max={pred_values.max():.3f}")

if __name__ == "__main__":
    main()

