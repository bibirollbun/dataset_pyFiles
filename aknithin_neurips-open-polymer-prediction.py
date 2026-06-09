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


!pip uninstall numpy -y
!pip install numpy==1.26.4 rdkit-pypi torch torch-geometric lightgbm pandas scikit-learn


# === Import Dependencies ===
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import os
import warnings
warnings.filterwarnings("ignore")

# === Verify dependencies ===
print("Verifying dependencies...")
try:
    import numpy, rdkit, torch
    print(f"NumPy: {numpy.__version__}, RDKit: {rdkit.__version__}, Torch: {torch.__version__}")
except ImportError as e:
    print(f"Dependency missing: {e}")
    raise SystemExit("Please install dependencies")

# === SMILES to Graph Conversion ===
def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetTotalValence(),
                atom.GetFormalCharge(),
                int(atom.GetIsAromatic()),
            ]
            node_features.append(features)
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
        node_features = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return Data(x=node_features, edge_index=edge_index)
    except:
        return None

# === RDKit-based descriptors ===
def compute_physicochemical_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
        ]
    except:
        return None

# === GNN Model ===
class PolymerGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        return x

# === Load and Process Data ===
def load_and_process_data(train_file, test_file, target_columns):
    print("Loading data...")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    smiles_col = [col for col in train_df.columns if col.lower() in ['smiles', 'smile']][0]

    imputer = SimpleImputer(strategy='median')
    train_targets = pd.DataFrame(imputer.fit_transform(train_df[target_columns]), columns=target_columns)

    train_graphs, train_phys, valid_idx = [], [], []
    for i, smi in enumerate(train_df[smiles_col]):
        g = smiles_to_graph(smi)
        f = compute_physicochemical_features(smi)
        if g is not None and f is not None:
            train_graphs.append(g)
            train_phys.append(f)
            valid_idx.append(i)

    test_graphs, test_phys, test_idx = [], [], []
    for i, smi in enumerate(test_df[smiles_col]):
        g = smiles_to_graph(smi)
        f = compute_physicochemical_features(smi)
        if g is not None and f is not None:
            test_graphs.append(g)
            test_phys.append(f)
            test_idx.append(i)

    train_df = train_df.iloc[valid_idx]
    train_targets = train_targets.iloc[valid_idx]
    test_df = test_df.iloc[test_idx]

    scaler = StandardScaler()
    train_phys = scaler.fit_transform(train_phys)
    test_phys = scaler.transform(test_phys)
    train_raw = scaler.fit_transform(train_df[target_columns])
    test_raw = scaler.transform(np.zeros((len(test_df), len(target_columns))))

    return train_graphs, train_phys, train_raw, train_targets, test_graphs, test_phys, test_raw, test_df

# === Train GNN ===
def train_gnn(model, data_list, targets, epochs=50, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model.train()
    for epoch in range(epochs):
        for data, y in zip(data_list, targets):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.l1_loss(out, torch.tensor([y], dtype=torch.float, device=device))
            loss.backward()
            optimizer.step()

# === GNN Feature Extraction ===
def extract_gnn_features(model, data_list, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    model.eval()
    feats = []
    with torch.no_grad():
        for d in data_list:
            d = d.to(device)
            out = model(d)
            feats.append(out.cpu().numpy())
    return np.array(feats)

# === Train LightGBM ===
def train_gbdt(gnn_feat, phys_feat, raw_feat, targets, target_name):
    print(f"Training GBDT for {target_name}")
    X = np.hstack([gnn_feat, phys_feat, raw_feat])
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    models = []
    for train_idx, val_idx in kf.split(X):
        model = lgb.LGBMRegressor(
            objective='regression',
            metric='mae',
            learning_rate=0.03,
            n_estimators=300,
            num_leaves=15,
            random_state=42
        )
        model.fit(X[train_idx], targets[train_idx])
        models.append(model)
    return models

def predict_ensemble(models, gnn_feat, phys_feat, raw_feat):
    X = np.hstack([gnn_feat, phys_feat, raw_feat])
    preds = np.mean([m.predict(X) for m in models], axis=0)
    return preds

# === Main Entry Point ===
def main():
    train_path = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
    test_path = '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv'
    target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

    train_graphs, train_phys, train_raw, train_targets, test_graphs, test_phys, test_raw, test_df = load_and_process_data(
        train_path, test_path, target_columns
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = train_graphs[0].x.shape[1]
    gnn_model = PolymerGNN(input_dim=input_dim, hidden_dim=32, output_dim=16)

    submission = pd.DataFrame({'id': test_df['id']})

    for target in target_columns:
        print(f"ðŸ§  Training for target: {target}")
        train_gnn(gnn_model, train_graphs, train_targets[target].values, epochs=50, device=device)
        train_gnn_feat = extract_gnn_features(gnn_model, train_graphs, device)
        test_gnn_feat = extract_gnn_features(gnn_model, test_graphs, device)

        models = train_gbdt(train_gnn_feat, train_phys, train_raw, train_targets[target].values, target)
        preds = predict_ensemble(models, test_gnn_feat, test_phys, test_raw)
        submission[target] = preds

    # âœ… Save to Kaggle output directory
    submission.to_csv('/kaggle/working/submission.csv', index=False)
    print("âœ… submission.csv is ready in /kaggle/working!")

if __name__ == "__main__":
    main()


