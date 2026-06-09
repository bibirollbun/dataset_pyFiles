!pip install rdkit torch torchvision torchaudio


import torch


print(torch.__version__)
print(torch.version.cuda)


!pip install rdkit torch torchvision torchaudio
!pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
!pip install torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
!pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
!pip install torch-geometric


import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from tqdm.notebook import tqdm
import warnings
from torch_geometric.nn import global_mean_pool
import joblib


warnings.filterwarnings('ignore')


TARGET_COLUMNS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']


train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')

dataset1_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv') # Tc data
dataset2_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset2.csv') # SMILES only for Tg
dataset3_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv') # Older simulation results
dataset4_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv') # Older simulation results

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Sample Submission shape: {sample_submission_df.shape}")


def generate_rdkit_features(smiles_series):
    all_desc_names = [desc[0] for desc in Descriptors._descList]
    fp_size = 2048 # Morgan Fingerprint size
    all_fp_cols = [f'FP_{i}' for i in range(fp_size)]
    all_cols = all_desc_names + all_fp_cols

    all_features = []
    for s in tqdm(smiles_series, desc="Processing SMILES"):
        mol = Chem.MolFromSmiles(s)
        current_features = [np.nan] * len(all_cols)

        if mol is not None:
            desc_values = []
            for desc_name, desc_func in Descriptors._descList:
                try:
                    desc_values.append(desc_func(mol))
                except:
                    desc_values.append(np.nan)

            fp_values = []
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_size)
                fp_values = fp.ToList()
            except:
                fp_values = [np.nan] * fp_size

            current_features = desc_values + fp_values
            
        all_features.append(current_features)

    return pd.DataFrame(all_features, columns=all_cols, index=smiles_series.index)


train_rdkit_features = generate_rdkit_features(train_df['SMILES'])
train_features_df = pd.concat([train_df[['id', 'SMILES'] + TARGET_COLUMNS], train_rdkit_features], axis=1)


test_rdkit_features = generate_rdkit_features(test_df['SMILES'])
test_features_df = pd.concat([test_df[['id', 'SMILES']], test_rdkit_features], axis=1)


print(f"Train feature shape: {train_features_df.shape}")
print(f"Test feature shape: {test_features_df.shape}")


for col in TARGET_COLUMNS:
    train_features_df[col] = pd.to_numeric(train_features_df[col], errors='coerce')


print("Target column dtypes:")
print(train_features_df[TARGET_COLUMNS].dtypes)


target_means = train_features_df[TARGET_COLUMNS].mean()
target_means = target_means.fillna(0)


for col in TARGET_COLUMNS:
    if train_features_df[col].isnull().any():
        train_features_df[col].fillna(target_means[col], inplace=True)
        print(f"Imputed NaNs in '{col}' with mean: {target_means[col]:.4f}")


print(f"Train data shape after imputing NaNs in targets: {train_features_df.shape}")


print(f"NaNs in TARGET_COLUMNS after imputation: \n{train_features_df[TARGET_COLUMNS].isnull().sum()}")


feature_cols = train_rdkit_features.columns.tolist()


X_train_rdkit = train_features_df[feature_cols].copy()
Y_train_rdkit = train_features_df[TARGET_COLUMNS].copy()
X_test_rdkit = test_features_df[feature_cols].copy()


X_train_rdkit.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_rdkit.replace([np.inf, -np.inf], np.nan, inplace=True)


feature_means = X_train_rdkit.mean()
feature_means = feature_means.fillna(0)


for col in feature_cols:
    if X_train_rdkit[col].isnull().any():
        X_train_rdkit[col].fillna(feature_means[col], inplace=True)
    if X_test_rdkit[col].isnull().any():
        X_test_rdkit[col].fillna(feature_means[col], inplace=True)


print(f"Any NaNs in X_train_rdkit features just before scaling: {X_train_rdkit.isnull().any().any()}")


print(f"Any Inf in X_train_rdkit features just before scaling: {np.isinf(X_train_rdkit).any().any()}") 


scaler = StandardScaler()
X_train_rdkit = pd.DataFrame(scaler.fit_transform(X_train_rdkit), columns=feature_cols, index=X_train_rdkit.index)
X_test_rdkit = pd.DataFrame(scaler.transform(X_test_rdkit), columns=feature_cols, index=X_test_rdkit.index)


train_features_df[feature_cols] = X_train_rdkit
test_features_df[feature_cols] = X_test_rdkit


xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
)


multi_output_xgb = MultiOutputRegressor(xgb_model)
multi_output_xgb.fit(X_train_rdkit, Y_train_rdkit)


xgb_predictions = multi_output_xgb.predict(X_test_rdkit)
xgb_predictions_df = pd.DataFrame(xgb_predictions, columns=TARGET_COLUMNS)
xgb_predictions_df['id'] = test_df['id']


xgb_predictions_df.head()


def smiles_to_graph_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
        ])
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        bond_type = bond.GetBondTypeAsDouble()
        edge_features.append([bond_type])
        edge_features.append([bond_type])

    if not edge_indices:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


train_graph_data_list = []
for idx, row in tqdm(train_features_df.iterrows(), total=len(train_features_df), desc="Converting Train SMILES to Graphs"):
    graph_data = smiles_to_graph_data(row['SMILES'])
    if graph_data:
        target_array = np.array(row[TARGET_COLUMNS], dtype=float).reshape(1, -1)
        graph_data.y = torch.tensor(target_array, dtype=torch.float)
        train_graph_data_list.append(graph_data)


test_graph_data_list = []
test_ids_for_gnn_pred = []
for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Converting Test SMILES to Graphs"):
    graph_data = smiles_to_graph_data(row['SMILES'])
    if graph_data:
        test_graph_data_list.append(graph_data)
        test_ids_for_gnn_pred.append(row['id'])


class GNNRegressor(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_targets):
        super(GNNRegressor, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.pool = global_mean_pool
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, num_targets)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.pool(x, batch)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        return x


if train_graph_data_list:
    num_node_features = train_graph_data_list[0].x.shape[1]
else:
    print("WARNING: No valid graphs generated for training GNN!")
    num_node_features = 0 

if num_node_features > 0:
    model = GNNRegressor(num_node_features=num_node_features, hidden_channels=64, num_targets=len(TARGET_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loader = DataLoader(train_graph_data_list, batch_size=32, shuffle=True)
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            out = model(batch_data)

            target_y = batch_data.y.squeeze(1)

            loss = criterion(out, target_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_data.num_graphs

        avg_loss = total_loss / len(train_graph_data_list)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}')


model.eval()
gnn_predictions = []
gnn_ids_ordered = []

test_loader = DataLoader(test_graph_data_list, batch_size=32, shuffle=False)

with torch.no_grad():
    for i, batch_data in enumerate(test_loader):
        batch_data = batch_data.to(device)
        out = model(batch_data)
        gnn_predictions.extend(out.cpu().numpy())
        
        start_idx = i * test_loader.batch_size
        end_idx = min((i + 1) * test_loader.batch_size, len(test_ids_for_gnn_pred))
        gnn_ids_ordered.extend(test_ids_for_gnn_pred[start_idx:end_idx])

gnn_predictions_df = pd.DataFrame(gnn_predictions, columns=TARGET_COLUMNS)
gnn_predictions_df['id'] = gnn_ids_ordered


final_predictions_df = sample_submission_df[['id']].copy()


final_predictions_df = pd.merge(final_predictions_df, xgb_predictions_df, on='id', how='left', suffixes=('_base', '_xgb'))
final_predictions_df = pd.merge(final_predictions_df, gnn_predictions_df, on='id', how='left', suffixes=('_xgb', '_gnn'))


for col in TARGET_COLUMNS:
    xgb_col = f'{col}_xgb'
    gnn_col = f'{col}_gnn'

    available_cols = []
    if xgb_col in final_predictions_df.columns and not final_predictions_df[xgb_col].isnull().all():
        available_cols.append(xgb_col)
    if gnn_col in final_predictions_df.columns and not final_predictions_df[gnn_col].isnull().all():
        available_cols.append(gnn_col)

    if available_cols:
        final_predictions_df[col] = final_predictions_df[available_cols].mean(axis=1)
    else:
        final_predictions_df[col] = target_means[col] 


submission_df = final_predictions_df[['id'] + TARGET_COLUMNS]
submission_df.to_csv('submission.csv', index=False)


submission_df.head()

