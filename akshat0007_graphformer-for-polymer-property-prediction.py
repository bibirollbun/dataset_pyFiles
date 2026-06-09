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


pip install --no-index --find-links="/kaggle/input/notebookb88bbda203/offline-libraries" scikit-learn


# pip install --no-index --find-links="/kaggle/input/notebookb88bbda203/offline-libraries" catboost


pip install --no-index --find-links="/kaggle/input/notebookb88bbda203/offline-libraries" rdkit 


pip install --no-index --find-links="/kaggle/input/notebookb88bbda203/offline-libraries" torch-scatter


pip install --no-index --find-links="/kaggle/input/notebookb88bbda203/offline-libraries" torch-sparse 


pip install --no-index --find-links="/kaggle/input/notebookb88bbda203/offline-libraries" torch-cluster 


pip install --no-index --find-links="/kaggle/input/notebookb88bbda203/offline-libraries" torch-spline-conv


pip install --no-index --find-links="/kaggle/input/notebookb88bbda203/offline-libraries" torch-geometric


import warnings
warnings.filterwarnings('ignore')


import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# Enable CUDA Debugging for more precise error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# --- RDKit Imports ---
from rdkit import Chem

# --- PyTorch & PyG Imports ---
import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

# --- PyTorch Lightning Imports ---
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# --- Sklearn Imports ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
TRAINER_CONFIG = {"accelerator": "gpu", "devices": 1, "max_epochs": 200}
BASE_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'
SUBMISSION_FILE = '/kaggle/working/submission.csv'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
SUPPLEMENT_FILES = { 'd1': 'train_supplement/dataset1.csv', 'd3': 'train_supplement/dataset3.csv', 'd4': 'train_supplement/dataset4.csv' }
EMBEDDING_VOCAB_SIZE = 256

# --- 1. Data Loading & Preprocessing ---
print("--- 1. Loading and Preprocessing Data ---")
train_df = pd.read_csv(os.path.join(BASE_PATH, TRAIN_FILE))
test_df = pd.read_csv(os.path.join(BASE_PATH, TEST_FILE))
supp_dfs = {}
for key, path in SUPPLEMENT_FILES.items():
    try: supp_dfs[key] = pd.read_csv(os.path.join(BASE_PATH, path))
    except FileNotFoundError: supp_dfs[key] = pd.DataFrame()
if not supp_dfs.get('d1', pd.DataFrame()).empty: supp_dfs['d1'] = supp_dfs['d1'].rename(columns={'smiles': 'SMILES', 'tc': 'Tc'})
if not supp_dfs.get('d3', pd.DataFrame()).empty: supp_dfs['d3'] = supp_dfs['d3'].rename(columns={'smiles': 'SMILES', 'tg': 'Tg', 'ffv': 'FFV'})
if not supp_dfs.get('d4', pd.DataFrame()).empty: supp_dfs['d4'] = supp_dfs['d4'].rename(columns={'smiles': 'SMILES', 'density': 'Density', 'rg': 'Rg'})
combined_df = pd.concat([df for df in [train_df, supp_dfs.get('d1'), supp_dfs.get('d3'), supp_dfs.get('d4')] if not df.empty], ignore_index=True, sort=False)
properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
for col in properties:
    if col not in combined_df.columns: combined_df[col] = np.nan
train_df_processed = combined_df.groupby('SMILES')[properties].mean().reset_index()

print("\n--- Finding max graph size for padding ---")
all_smiles = pd.concat([train_df_processed['SMILES'], test_df['SMILES']]).unique()
max_nodes = 0
for smiles in tqdm(all_smiles, desc="Finding max atoms"):
    mol = Chem.MolFromSmiles(smiles)
    if mol: max_nodes = max(max_nodes, mol.GetNumAtoms())
print(f"Max atoms found in any molecule: {max_nodes}")

# --- 2. PyTorch Lightning DataModule ---
print("\n--- 2. Setting up Lightning DataModule ---")
def get_atom_features(atom):
    return [atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(), int(atom.GetHybridization()), int(atom.GetIsAromatic())]

def smiles_to_graph(smiles, max_nodes, embedding_size):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0: return None
    dist_matrix_np = Chem.GetDistanceMatrix(mol)
    if np.any(np.isinf(dist_matrix_np)) or np.any(np.isnan(dist_matrix_np)): return None
    if np.any(dist_matrix_np >= embedding_size): return None
    edge_indices = []
    for bond in mol.GetBonds():
        edge_indices.extend([(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()), (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())])
    graph = Data(
        x=torch.tensor([get_atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float),
        edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    )
    sp_matrix_real = torch.from_numpy(dist_matrix_np).long()
    padded_sp = torch.zeros((max_nodes, max_nodes), dtype=torch.long)
    padded_sp[:mol.GetNumAtoms(), :mol.GetNumAtoms()] = sp_matrix_real
    graph.sp_matrix = padded_sp
    return graph

class PolymerDataset(Dataset):
    def __init__(self, df, target_cols, max_nodes, embedding_size):
        super().__init__()
        self.df, self.target_cols, self.max_nodes, self.embedding_size = df, target_cols, max_nodes, embedding_size
        self.graphs = [smiles_to_graph(s, self.max_nodes, self.embedding_size) for s in tqdm(df['SMILES'], desc="Creating Graphs")]
        self.valid_indices = [i for i, g in enumerate(self.graphs) if g is not None]
    def len(self): return len(self.valid_indices)
    def get(self, idx):
        real_idx = self.valid_indices[idx]
        graph = self.graphs[real_idx].clone()
        if self.target_cols: 
            # Ensure y is stored as a 2D tensor [1, num_properties]
            target_values = self.df.loc[real_idx, self.target_cols].values.astype(np.float32)
            graph.y = torch.tensor(target_values, dtype=torch.float).view(1, -1)
        return graph

class PolymerDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, target_properties, max_nodes, embedding_size, batch_size=32):
        super().__init__()
        self.save_hyperparameters('batch_size')
        self.train_df, self.test_df = train_df, test_df
        self.target_properties, self.max_nodes, self.embedding_size = target_properties, max_nodes, embedding_size
        self.scalers = {target: StandardScaler() for target in self.target_properties}

    def setup(self, stage=None):
        for target in self.target_properties:
            valid_data = self.train_df[[target]].dropna()
            self.scalers[target].fit(valid_data)
            valid_indices = self.train_df[target].notna()
            self.train_df.loc[valid_indices, target] = self.scalers[target].transform(self.train_df.loc[valid_indices, [target]])
        train_data, val_data = train_test_split(self.train_df, test_size=0.15, random_state=42)
        self.train_dataset = PolymerDataset(train_data.reset_index(drop=True), self.target_properties, self.max_nodes, self.embedding_size)
        self.val_dataset = PolymerDataset(val_data.reset_index(drop=True), self.target_properties, self.max_nodes, self.embedding_size)
        self.predict_dataset = PolymerDataset(self.test_df.reset_index(drop=True), [], self.max_nodes, self.embedding_size)

    def train_dataloader(self): return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    def val_dataloader(self): return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size * 2, num_workers=os.cpu_count(), pin_memory=True)
    def predict_dataloader(self): return DataLoader(self.predict_dataset, batch_size=self.hparams.batch_size * 2, num_workers=os.cpu_count(), pin_memory=True)

# --- 3. PyTorch Lightning Model ---
print("\n--- 3. Defining LightningModule ---")
class GraphformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, embedding_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim); self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim*4), nn.GELU(), nn.Linear(embed_dim*4, embed_dim))
        self.sp_embedding = nn.Embedding(embedding_size, 1)
        self.num_heads = num_heads

    def forward(self, x, sp_matrix, padding_mask):
        if sp_matrix.dim() == 2:
            sp_matrix = sp_matrix.unsqueeze(0)
            
        batch_size, batch_seq_len = x.size(0), x.size(1)
        
        sp_matrix_clipped = torch.clamp(sp_matrix, min=0, max=self.sp_embedding.num_embeddings - 1)
        attn_bias = self.sp_embedding(sp_matrix_clipped).squeeze(-1)
        # Remove slicing since we're using global max_nodes
        attn_mask = attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1).reshape(batch_size * self.num_heads, batch_seq_len, batch_seq_len)
        attn_output, _ = self.attention(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask)
        x = self.norm1(x + attn_output); x = self.norm2(x + self.ffn(x))
        return x

class Graphformer(nn.Module):
    def __init__(self, model_config, max_nodes):
        super().__init__()
        self.max_nodes = max_nodes
        self.atom_encoder = nn.Linear(model_config["node_in_dim"], model_config["embed_dim"])
        self.layers = nn.ModuleList([GraphformerLayer(model_config["embed_dim"], model_config["num_heads"], model_config["embedding_size"]) for _ in range(model_config["num_layers"])])
        self.predictor = nn.Sequential(nn.Linear(model_config["embed_dim"], model_config["embed_dim"] // 2), nn.GELU(), nn.Linear(model_config["embed_dim"] // 2, model_config["num_outputs"]))

    def forward(self, data):
        x, batch, sp_matrix = data.x, data.batch, data.sp_matrix
        # Use global max_nodes for consistent padding
        x_dense, padding_mask = to_dense_batch(x, batch, max_num_nodes=self.max_nodes)
        x_dense = self.atom_encoder(x_dense)
        for layer in self.layers:
            x_dense = layer(x_dense, sp_matrix, ~padding_mask)
        return self.predictor(x_dense[:, 0, :])

class GraphformerLightning(pl.LightningModule):
    def __init__(self, model_config, scalers, max_nodes):
        super().__init__()
        self.save_hyperparameters(); 
        self.model = Graphformer(self.hparams.model_config, max_nodes)
        self.scalers = scalers
        
    def forward(self, batch): return self.model(batch)
    
    def _shared_step(self, batch):
        # Get model predictions
        out = self(batch)  # shape: [batch_size, num_properties]
        
        # Ensure targets are 2D: [batch_size, num_properties]
        targets = batch.y
        
        # Create mask for valid targets
        mask = ~torch.isnan(targets)  # shape: [batch_size, num_properties]
        
        # Only calculate loss if there are valid targets
        if mask.any():
            # Calculate MAE only for valid targets
            loss = torch.abs(out - targets)[mask].mean()
        else:
            loss = torch.tensor(0.0, device=self.device)
            
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log('train_loss', loss, batch_size=batch.num_graphs)
        return loss
        
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log('val_loss', loss, batch_size=batch.num_graphs)
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0): 
        return self(batch)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

# --- 4. Training ---
print("\n--- 4. Initializing Trainer and Starting Training ---")
data_module = PolymerDataModule(train_df_processed, test_df, properties, max_nodes, EMBEDDING_VOCAB_SIZE, batch_size=64)
model_config = {"node_in_dim": 5, "embed_dim": 128, "num_heads": 8, "num_layers": 6, "num_outputs": len(properties), "embedding_size": EMBEDDING_VOCAB_SIZE}
lightning_model = GraphformerLightning(model_config, data_module.scalers, max_nodes)
checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints', filename='best-model', save_top_k=1, mode='min')
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15)
trainer = pl.Trainer(**TRAINER_CONFIG, callbacks=[checkpoint_callback, early_stopping_callback], logger=pl.loggers.CSVLogger("logs/"))
trainer.fit(lightning_model, datamodule=data_module)

# --- 5. Prediction & Submission ---
print("\n--- 5. Generating Predictions ---")
predictions_batches = trainer.predict(lightning_model, datamodule=data_module, ckpt_path='best')
predictions_np = torch.cat(predictions_batches).cpu().numpy()
predictions_rescaled = {target: data_module.scalers[target].inverse_transform(predictions_np[:, i].reshape(-1, 1)).flatten() for i, target in enumerate(properties)}
submission_df = pd.DataFrame(predictions_rescaled)
submission_df['id'] = test_df['id']
submission_df = submission_df[['id'] + properties]
submission_df.to_csv(SUBMISSION_FILE, index=False)
print(f"\nSubmission file '{SUBMISSION_FILE}' created successfully!")
print("Top 5 rows of the submission file:")
print(submission_df.head())




