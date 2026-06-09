# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#NB: redundant and messy imports here
from sklearn.model_selection import train_test_split
import torch as pt
#!pip install torch_geometric
!pip install \
    torch_geometric \
    --no-index \
    --find-links=file:///kaggle/input/pt-geometric/
import torch_geometric as torch_geometric
from torch_geometric.data import Data   #PyTorch: "A data object describing a homogeneous graph. The data object can hold node-level, link-level and graph-level attributes. In general, Data tries to mimic the behavior of a regular Python dictionary. In addition, it provides useful functionality for analyzing graph structures, and provides basic PyTorch tensor functionalities. Tut: https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs."
from torch_geometric.utils import from_smiles # Converts a SMILES string to a `torch_geometric.data.Data` instance

!pip install \
    rdkit \
    --no-index \
    --find-links=file:///kaggle/input/kaggle-compatible-rdkit-dataset/
#!pip install rdkit
import rdkit as rdkit

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ALL IMPORTS
from torch_geometric.data import Batch
from tqdm import tqdm
import random

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt


#from torch.nn import Linear
#import torch.nn.functional as F
#from torch_geometric.nn import GCNConv
#from torch_geometric.nn import global_mean_pool
#from torch_geometric.loader import DataLoader

# train_loader = DataLoader(mol_data_train, batch_size=64, shuffle=True) # pytorch_geometric DataLoader objects needed as inputs to model.
# val_loader = DataLoader(mol_data_val, batch_size=64, shuffle=True)

class GCN(pt.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        pt.manual_seed(12345)
        self.conv1 = GCNConv(9, hidden_channels) # NB replaced automated num_node_features from "Batch" batch object with number 9
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
model = GCN(hidden_channels=64).to(device)
print(model)


from torch.nn import BatchNorm1d

class TweakedGCN(pt.nn.Module):
    def __init__(self, hidden_channels, dropout_p=0.5):
        super(TweakedGCN, self).__init__()
        # Using a shallower 3-layer architecture
        self.conv1 = GCNConv(9, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)

        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.bn4 = BatchNorm1d(hidden_channels)
        
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.bn5 = BatchNorm1d(hidden_channels)

        
        # Regression head with dropout
        # Regression head with dropout
        # Regression head with dropout
        self.lin = Linear(hidden_channels, 1)
        self.dropout = pt.nn.Dropout(p=dropout_p)

    def forward(self, x, edge_index, batch):
        # Graph Convolutional Layers with Batch Norm and LeakyReLU
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        
        h = self.conv3(h, edge_index)
        h = self.bn3(h)
        h = F.relu(h)

        h = self.conv4(h, edge_index)
        h = self.bn4(h)
        h = F.relu(h)

        h = self.conv5(h, edge_index)
        h = self.bn5(h)
        h = F.relu(h)
      
        # Global Pooling (aggregate node features to get a graph-level representation)
        h_graph = global_mean_pool(h, batch)
        
        # Dropout before the final linear layer
        h_graph = self.dropout(h_graph)
        
        # Final prediction
        out = self.lin(h_graph)
        
        return out


model = TweakedGCN(hidden_channels=64).to(device)
print(model)


from torch_geometric.nn import GINConv, global_max_pool
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN

class AdvancedGCN(pt.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, dropout_p=0.5):
        super(AdvancedGCN, self).__init__()
        
        # GIN requires a small MLP for each layer
        nn1 = Sequential(Linear(in_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn1)
        self.bn1 = BN(hidden_channels)

        nn2 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv2 = GINConv(nn2)
        self.bn2 = BN(hidden_channels)
        
        nn3 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv3 = GINConv(nn3)
        self.bn3 = BN(hidden_channels)
        
        nn4 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv4 = GINConv(nn4)
        self.bn4 = BN(hidden_channels)
        
        nn5 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv5 = GINConv(nn5)
        self.bn5 = BN(hidden_channels)
        
        # Regression head
        self.lin1 = Linear(hidden_channels * 2, hidden_channels) # *2 from concatenating mean and max pool
        self.lin2 = Linear(hidden_channels, out_channels)
        self.dropout = pt.nn.Dropout(p=dropout_p)

    def forward(self, x, edge_index, batch):
        # Initial transformation
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        
        # Residual Blocks
        h = h + F.relu(self.bn2(self.conv2(h, edge_index)))
        h = h + F.relu(self.bn3(self.conv3(h, edge_index)))
        h = h + F.relu(self.bn4(self.conv4(h, edge_index)))
        h = h + F.relu(self.bn5(self.conv5(h, edge_index)))
        
        # --- Global Pooling ---
        # Concatenate mean and max pooling to get a richer graph representation
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h = pt.cat([h_mean, h_max], dim=1)
        
        # --- Regression Head ---
        h = self.dropout(h)
        h = F.relu(self.lin1(h))
        out = self.lin2(h)
        
        return out



model = AdvancedGCN(in_channels = 9, hidden_channels=64).to(device)
print(model)


optimizer = pt.optim.Adam(model.parameters(), lr=0.01)
criterion = pt.nn.MSELoss()

def train():
    model.train()

    for data in val_loader:  # Iterate in batches over the training dataset.
         data = data.to(device)
         x = data.x.float()
         out = model(x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y.view(-1,1).float())  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     total_error = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         data = data.to(device)
         x = data.x.float()
         out = model(x, data.edge_index, data.batch)  
         y = data.y.view(-1, 1).float() # REVIEW THIS, For regression, the target data.y must be a float tensor. The .view(-1, 1) reshapes it to match the model's output shape of [batch_size, 1].
         error = (out - y).abs().sum().item() 
         total_error += error
     return total_error / len(loader.dataset)  # Derive ratio of correct predictions.


df_predict = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv") 
print(df_predict)
# df_predict = pd.DataFrame(data = {'id': [674895, 6738949], 'SMILES': ['shitSmiles1', 'shitSmiles2']}) rubbish data
predict_smiles_list = df_predict.loc[:, "SMILES"]

MAX_LEN_SMILES = 2000

placeholder_smiles = '*Nc1ccc([C@H](CCC)c2ccc(C3(c4ccc([C@@H](CCC)c5ccc(N*)cc5)cc4)CCC(CCCCC)CC3)cc2)cc1'

def robust_from_smiles(smiles): 
    try:
        
        if len(smiles) > MAX_LEN_SMILES:
            clean_data = Data(x=pt.zeros((1, 9)), edge_index=pt.empty((2, 0), dtype=pt.long))
        # Return the same placeholder as the error case
        else:
            clean_data = from_smiles(smiles)

        if clean_data.num_nodes == 0:
            clean_data = from_smiles(placeholder_smiles)
        
        return clean_data
    except Exception as e:
        print(f"Could not parse SMILES: {smiles}. Error: {e}")
        return from_smiles(placeholder_smiles)


print('Robust', robust_from_smiles('pathological_smiles'))
print('Default', from_smiles(placeholder_smiles))

def predict_batch(model, smiles_list):
    """Predicts on a list of SMILES strings."""
    model.eval()
    
    # Create a list of Data objects from the SMILES strings
    data_list = []
    for smiles in smiles_list:
        data = robust_from_smiles(smiles)
        data.x = data.x.to(pt.float)
        data_list.append(data)
        
    # Create a DataLoader for the batch
    print(len(data_list))
    loader = DataLoader(data_list, batch_size=len(data_list))
    batch = next(iter(loader))
    batch = batch.to(device)

    with pt.no_grad():
        predictions = model(batch.x, batch.edge_index, batch.batch)

    # Return the predictions as a simple list of numbers
    return predictions.view(-1).tolist()






df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
target_name_list = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
df_submission_str  = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv", dtype = str)
#all_ids = pd.DataFrame(df_submission_str['id'].copy())
df_submission = pd.DataFrame(data = df_predict.loc[:, 'id'])

predict_smiles_list = df_predict.loc[:, 'SMILES']
# START LOOP HERE

for target_name in target_name_list:
    df_clean = df.dropna(subset=[target_name]).copy()
    
    smiles = df_clean.loc[:, ['SMILES']]
    
    mol_data = []
    
    for index, row in tqdm(df_clean.iterrows(), total=df_clean.shape[0]): # Creates a list of pytorch Data (graph) objects with pytorch tensor target (y)
        smiles = row['SMILES']
        target = row[target_name]
    
        data = robust_from_smiles(smiles)
    
        data.y = pt.tensor([[target]], dtype=pt.float)
    
        mol_data.append(data)
    
    random.shuffle(mol_data)
    
    split_point = round(len(mol_data)*0.9)
    mol_data_train = mol_data[:split_point]
    mol_data_val = mol_data[split_point + 1:]

    print(mol_data_train)
    
    train_loader = DataLoader(mol_data_train, batch_size=64, shuffle=True) # pytorch_geometric DataLoader objects needed as inputs to model.
    val_loader = DataLoader(mol_data_val, batch_size=64, shuffle=True)
    
    #batch = Batch.from_data_list(mol_data) # only used to find num_node_features in model definition (9)
    
    train_mae_history = []
    val_mae_history = []
    
    for epoch in range(1, 200):
        train()
        train_mae = test(train_loader)
        val_mae = test(val_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_mae:.4f}, Val Acc: {val_mae:.4f}')
    
        train_mae_history.append(train_mae)
        val_mae_history.append(val_mae)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_mae_history, label='Train MAE')
    plt.plot(val_mae_history, label='Val MAE')
    title = 'MAE over Epochs' + target_name 
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend()
    plt.grid(True)
    #plt.show()
    
    predictions = predict_batch(model, predict_smiles_list)
    print(predictions)
    df_submission[target_name] = predictions
    print(df_submission)
    
    
    #for smiles, pred, value in zip(test_smiles_list, predictions, test_values_list):
    #    print(f"SMILES: {smiles:<10} | Predicted Value: {pred:.4f} | Actual Value: {value:.4f}")
#final_submission = all_ids.merge(df_submission, on='id',how='left')



tg_mean = 96.452313684
ffv_mean = 0.367211995
density_mean = 0.9854843785473083
tc_mean = 0.25633409252439104
rg_mean = 16.41978670954397

df_submission = df_submission.replace(-np.inf,np.nan)
df_submission = df_submission.replace(np.inf,np.nan)
df_submission['Tg'].fillna(tg_mean, inplace=True)
df_submission['FFV'].fillna(ffv_mean, inplace=True)
df_submission['Tc'].fillna(tc_mean, inplace=True)
df_submission['Density'].fillna(density_mean, inplace=True)
df_submission['Rg'].fillna(rg_mean, inplace=True)


df_submission['id'] = df_submission['id'].astype(str)

print(df_submission)

df_submission.to_csv('submission.csv', index = False)

