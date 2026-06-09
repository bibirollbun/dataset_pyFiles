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


pip install torch_geometric


############################################################
# 0. Imports, env-var, paths
############################################################
import os, warnings, gc
os.environ["TORCH_ALLOW_DUPLICATE_LIBRARY_REGISTRATIONS"] = "1"

import h5py, numpy as np, pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# PyG imports
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data as GeoData, DataLoader as GeoDataLoader
from torch_geometric.nn import SAGEConv

from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

# configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H5_PATH = "/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5"
epochs, batch_size, patch_size = 10, 16, 224
k_neighbors = 8
lr = 1e-4

############################################################
# 1. Spatial patch dataset
############################################################
class SpatialDataset(Dataset):
    def __init__(self, h5_path, split, slide_ids=None, patch_size=224, transform=None):
        self.r = patch_size // 2
        self.transform = transform
        with h5py.File(h5_path, 'r') as f:
            img_grp = f[f"images/{split}"]
            keys = slide_ids or list(img_grp.keys())
            self.images = {sid: img_grp[sid][:] for sid in keys}
            spot_grp = f[f"spots/{split}"]
            self.spots  = {sid: spot_grp[sid][:] for sid in keys}
        sample = next(iter(self.spots.values()))
        self.has_label = 'C1' in sample.dtype.names
        self.index = [(sid, i)
                      for sid, arr in self.spots.items()
                      for i in range(arr.shape[0])]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sid, i = self.index[idx]
        img = self.images[sid]
        rec = self.spots[sid]
        x, y = int(rec['x'][i]), int(rec['y'][i])
        r, H, W = self.r, *img.shape[:2]
        patch = img[max(0, y-r):min(H, y+r), max(0, x-r):min(W, x+r), :]
        pad = [(0,0),(0,0),(0,0)]
        if patch.shape[0] < 2*r:
            d = 2*r - patch.shape[0]
            pad[0] = (d//2, d-d//2)
        if patch.shape[1] < 2*r:
            d = 2*r - patch.shape[1]
            pad[1] = (d//2, d-d//2)
        if any(p != (0,0) for p in pad[:2]):
            patch = np.pad(patch, pad, mode='reflect')
        patch = torch.from_numpy(patch.transpose(2,0,1)).float().div(255)
        if self.transform:
            patch = self.transform(patch)
        label = torch.zeros(35, dtype=torch.float32)
        if self.has_label:
            label = torch.tensor([rec[f'C{k}'][i] for k in range(1,36)], dtype=torch.float32)
        coord = torch.tensor([x, y], dtype=torch.float32)
        return patch, label, coord

# transforms
train_tf = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = T.Compose([
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

############################################################
# 2. CNN feature extractor using ResNet101
############################################################
class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # outputs (B,2048)

############################################################
# 3. GraphDataset builder
############################################################
class GraphDataset(Dataset):
    def __init__(self, h5_path, split, slide_ids, transform):
        self.h5_path = h5_path
        self.split   = split
        self.slide_ids = slide_ids
        self.transform = transform

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        sid = self.slide_ids[idx]
        ds = SpatialDataset(self.h5_path, self.split, [sid], patch_size, self.transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        feats, labels, coords = [], [], []
        for xb, yb, coord in loader:
            xb = xb.to(device)
            with torch.no_grad():
                feats.append(cnn(xb).cpu())
            labels.append(yb)
            coords.append(coord)
        x = torch.cat(feats, dim=0)
        y = torch.cat(labels, dim=0)
        coord_np = torch.cat(coords, dim=0).numpy()
        nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(coord_np)
        _, idxs = nbrs.kneighbors(coord_np)
        edges = []
        for src, nbr in enumerate(idxs):
            for dst in nbr[1:]:
                edges += [[src, dst], [dst, src]]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return GeoData(x=x, edge_index=edge_index, y=y)

############################################################
# 4. GraphSAGE-based GNN
############################################################
class GraphCellFracNet(nn.Module):
    def __init__(self, in_dim=2048, hidden=256, out_dim=35, num_layers=3, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        # first layer
        self.convs.append(SAGEConv(in_dim, hidden))
        self.bns.append(nn.BatchNorm1d(hidden))
        # intermediate layers
        for _ in range(num_layers-2):
            self.convs.append(SAGEConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        # final
        self.convs.append(SAGEConv(hidden, out_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return x  # logits

############################################################
# 5. Instantiate models, optimizer, scheduler, loss
############################################################
cnn       = CNNBackbone().to(device)
gnn       = GraphCellFracNet().to(device)
optimizer = AdamW(list(cnn.parameters())+list(gnn.parameters()), lr=lr, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
loss_fn   = nn.MSELoss()

############################################################
# 6. Prepare data loaders
############################################################
train_ids = [f"S_{i}" for i in range(1,6)]
val_ids   = ["S_6"]
test_ids  = ["S_7"]
train_ds  = GraphDataset(H5_PATH, 'Train', train_ids, train_tf)
val_ds    = GraphDataset(H5_PATH, 'Train', val_ids,   val_tf)
test_ds   = GraphDataset(H5_PATH, 'Test',  test_ids,  val_tf)
train_loader = GeoDataLoader(train_ds, batch_size=1, shuffle=True)
val_loader   = GeoDataLoader(val_ds,   batch_size=1, shuffle=False)
############################################################
# 7. Training and validation loop
############################################################
best_sp = -1.0
for ep in range(1, epochs+1):
    cnn.train(); gnn.train()
    total_loss = 0.0
    for data in tqdm(train_loader, desc=f"Epoch {ep} train"):
        optimizer.zero_grad()
        logits = gnn(data)
        loss = loss_fn(torch.softmax(logits, dim=1), data.y.to(device))
        loss.backward(); optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    cnn.eval(); gnn.eval()
    sp_list = []
    with torch.no_grad():
        for data in val_loader:
            logits = gnn(data)
            preds  = torch.softmax(logits, dim=1).cpu().numpy()
            true   = data.y.numpy()
            for i in range(preds.shape[0]):
                sp_list.append(spearmanr(preds[i], true[i]).correlation)
    val_sp = np.nanmean(sp_list)
    print(f"Epoch {ep}/{epochs} - Loss: {total_loss/len(train_loader):.4f} | Val SP: {val_sp:.4f} (best {best_sp:.4f})")
    if val_sp > best_sp:
        best_sp = val_sp
        torch.save({'cnn': cnn.state_dict(), 'gnn': gnn.state_dict()}, 'best_resnet101.pth')

############################################################
# 8. Inference and submission
############################################################
ckpt = torch.load('best_resnet101.pth', map_location=device)
cnn.load_state_dict(ckpt['cnn']); gnn.load_state_dict(ckpt['gnn'])
cnn.eval(); gnn.eval()
all_preds = []
with torch.no_grad():
    for data in GeoDataLoader(test_ds, batch_size=1, shuffle=False):
        logits = gnn(data)
        all_preds.append(torch.softmax(logits, dim=1).cpu().numpy())

test_preds = np.vstack(all_preds)
with h5py.File(H5_PATH,'r') as f:
    spots = np.array(f['spots/Test']['S_7'])
idx_df = pd.DataFrame(spots).reset_index()
sub = pd.DataFrame(test_preds, columns=[f"C{i+1}" for i in range(35)])
sub.insert(0, 'ID', idx_df['index'])
sub.to_csv('submission_resnet101.csv', index=False)
print('✅ submission_resnet101.csv created', sub.shape)

