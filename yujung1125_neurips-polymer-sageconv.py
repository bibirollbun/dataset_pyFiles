!pip install /kaggle/input/polymer-whl-file-dataset/whl_folder/*.whl


import deepchem
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from deepchem.feat.molecule_featurizers import MolGraphConvFeaturizer
import pandas as pd
from rdkit import Chem
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
from sklearn.metrics import mean_absolute_error, r2_score
import math
from tqdm import tqdm


class MolDataset(Dataset):
    def __init__(self, smiles, labels):
        self.featurizer = MolGraphConvFeaturizer(use_edges=True)

        smiles = np.asarray(smiles)
        labels = np.asarray(labels)

        assert len(smiles) == len(labels), "difference lens (smiles labels)"

        self._to_label = lambda v: torch.tensor(v, dtype=torch.float32)

        self._xs = []
        self._ys = []
        self.failed = 0
        self.orig_idx = []

        for i, (smi, yv) in enumerate(zip(smiles, labels)):
            try:
                feats = self.featurizer.featurize(smi)
                if len(feats) == 0:
                    raise ValueError("featurize is none")
                mg = feats[0] # deepchem MolGraph

                x = torch.tensor(mg.node_features, dtype=torch.float)
                edge_index = torch.tensor(mg.edge_index, dtype=torch.long)
                pos = None
                if getattr(mg, "node_pos_features", None) is not None:
                    pos = torch.tensor(mg.node_pos_features, dtype=torch.float)

                data = Data(x=x, edge_index=edge_index, pos=pos)
                data.smiles = smi

                y = self._to_label(yv)

                self._xs.append(data)
                self._ys.append(y)
                self.orig_idx.append(i)

            except Exception as e:
                #print(e)  #print it if you want to know errors
                self.failed += 1
                continue

        if self.failed:
            print(f"MolDataset featurize failed: {self.failed}")

    def __len__(self):
        return len(self._xs)

    def __getitem__(self, idx):
        return self._xs[idx], self._ys[idx]


class SAGEConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(SAGEConv, self).__init__()
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, x, edge_index):
        # edge_index: [2, E]
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float().unsqueeze(1)
        deg[deg == 0] = 1
        neighbor_mean = torch.zeros_like(x)
        neighbor_mean.index_add_(0, row, x[col])
        neighbor_mean = neighbor_mean / deg
        concat = torch.cat([x, neighbor_mean], dim=1)
        return self.linear(concat)
        
class SAGEClassifier(nn.Module):
    def __init__(self, n_feature, hidden, dropout=0.5):
        super(SAGEClassifier, self).__init__()
        self.sage1 = SAGEConv(n_feature, hidden)
        self.sage2 = SAGEConv(hidden, hidden)
        self.linear = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.sage1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.sage2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x


def train_model(train_dataloader,test_dataloader,n_feature,epochs=100):
    # hyper parmas
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_feature=n_feature
    hidden_dim=256
    model = SAGEClassifier(n_feature, hidden_dim)
    model.to(device)

    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    epochs = epochs

    # train
    for epoch in tqdm(range(epochs),desc="Training..."):
        # train
        model.train()
        total_loss, total = 0.0, 0
        train_true, train_pred = [], []
        for data, y in train_dataloader:
            data = data.to(device) 
            y = y.to(device).view(-1)

            # loss + optim
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch).squeeze(-1)  
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            # record loss and mse
            total_loss += loss.item() * y.size(0)
            total += y.size(0)

            train_true.extend(y.detach().cpu().numpy())
            train_pred.extend(out.detach().cpu().numpy())

        train_loss = total_loss / max(total, 1)
        
        train_true = np.array(train_true)
        train_pred = np.array(train_pred)
        train_mae = mean_absolute_error(train_true, train_pred)
        train_rmse = math.sqrt(train_loss)
        train_r2 = r2_score(train_true, train_pred)
    
        # test
        model.eval()
        total_loss, total = 0.0, 0
        test_true, test_pred = [], []
        with torch.no_grad():
            for data, y in test_dataloader:
                data = data.to(device)
                y = y.to(device).view(-1)

                # loss + optim
                out = model(data.x, data.edge_index, data.batch).squeeze(-1)
                loss = criterion(out, y)
                total_loss += loss.item() * y.size(0)

                # record loss and mse
                total += y.size(0)
                
                test_true.extend(y.detach().cpu().numpy())
                test_pred.extend(out.detach().cpu().numpy())
        test_loss = total_loss / max(total, 1)
        
        test_true = np.array(test_true)
        test_pred = np.array(test_pred)
        test_mae = mean_absolute_error(test_true, test_pred)
        test_rmse = math.sqrt(test_loss)
        test_r2 = r2_score(test_true, test_pred)
        
        #print(f"[Epoch {epoch+1}] "
        #    f"Train Loss: {train_loss:.4f} | MAE: {train_mae:.4f} | RMSE: {train_rmse:.4f} | R2Score: {train_r2:.4f} / "
        #    f"Test Loss: {test_loss:.4f} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | R2Score: {test_r2:.4f}")
    print(f"[Result] Epoch {epoch+1} : "
        f"Train Loss: {train_loss:.4f} | MAE: {train_mae:.4f} | RMSE: {train_rmse:.4f} | R2Score: {train_r2:.4f} / "
        f"Test Loss: {test_loss:.4f} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | R2Score: {test_r2:.4f}")
    return model


def make_model(df, target, epochs):
    # split X and Y
    df = df.dropna(subset=[target])
    X = df['SMILES']
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # define dataset
    train_dataset = MolDataset(X_train, y_train)
    test_dataset = MolDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, 1)
    test_dataloader = DataLoader(test_dataset, 1)
    print()
    
    # train
    print(f'start training {target} model')
    model = train_model(train_dataset,train_dataloader,train_dataset[0][0].x.shape[1],epochs=epochs)
    return model


train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')


models={
    "Tg":make_model(train_df[["SMILES", "Tg"]],'Tg',epochs=300),
    "FFV":make_model(train_df[["SMILES", "FFV"]],'FFV',epochs=10),
    "Tc":make_model(train_df[["SMILES", "Tc"]],'Tc',epochs=30),
     "Density":make_model(train_df[["SMILES", "Density"]],'Density',epochs=30),
    "Rg":make_model(train_df[["SMILES", "Rg"]],'Rg',epochs=100)
}


def predict(X, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy = [0.0 for i in range(len(X))]
    val_dataset = MolDataset(X, dummy)
    val_dataloader = DataLoader(val_dataset, len(X))

    model.eval()
    with torch.no_grad():
        for data, _ in val_dataloader:
            data = data.to(device)
            pred_val = model(data.x, data.edge_index, data.batch).detach().cpu().numpy().flatten()
    return pred_val


# prepare data and predict
test_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")

X_result = test_df['SMILES']

Tg_pred = predict(X_result, models['Tg'])
FFV_pred = predict(X_result, models['FFV'])
Tc_pred = predict(X_result, models['Tc'])
Density_pred = predict(X_result, models['Density'])
Rg_pred = predict(X_result, models['Rg'])

# make submission file
test_pd = pd.DataFrame({'id':test_df['id'],'Tg':Tg_pred,'FFV':FFV_pred,'Tc':Tc_pred,'Density':Density_pred,'Rg':Rg_pred})
test_pd.to_csv("submission.csv",index=False)

