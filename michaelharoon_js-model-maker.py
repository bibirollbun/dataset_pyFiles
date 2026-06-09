##########################################
# This is a LightGBM regression model.
# (gradient boosting decision tree model)
##########################################
import os
import polars as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans


# 1) Configuration
DATA_PATH = '/kaggle/input/jane-street-real-time-market-data-forecasting/train.parquet'
BATCH_SIZE = 512
EPOCHS = 50
LR = 1e-3
TEMPERATURE = 0.5
EMBED_DIM = 128
PROJ_DIM = 64
NUM_CLUSTERS = 2  # change as needed
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 2) Data loading & sampling
# Load only feature columns, drop target
FEATURES = [f'feature_{i:02d}' for i in range(79)]
df = pl.read_parquet(DATA_PATH, columns=FEATURES).sample(n=200_000, seed=42).to_pandas()
X = df.values.astype(np.float32)



# 3) Define augmentations
class TabularAugment:
    def __init__(self, noise_scale=0.01, dropout_prob=0.1):
        self.noise_scale = noise_scale
        self.dropout_prob = dropout_prob
    def __call__(self, x):
        # Gaussian noise
        x = x + np.random.normal(scale=self.noise_scale, size=x.shape)
        # Feature dropout
        mask = np.random.rand(*x.shape) > self.dropout_prob
        return x * mask

# 4) Dataset that returns two augmented views
class ContrastiveDataset(Dataset):
    def __init__(self, X, transform):
        self.X = X
        self.transform = transform
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]
        return self.transform(x), self.transform(x)

# 5) Encoder and projection head
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim=EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
    def forward(self, x):
        return self.net(x)

class ProjectionHead(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, proj_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, proj_dim)
        )
    def forward(self, z):
        return self.net(z)

# 6) Contrastive loss (NT-Xent)
def nt_xent_loss(z1, z2, temperature=TEMPERATURE):
    z = torch.cat([z1, z2], dim=0)  # 2N x D
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # 2N x 2N

    N = z1.size(0)
    mask = torch.eye(2*N, dtype=torch.bool).to(DEVICE)
    sim = sim / temperature

    # numerator: sim between positive pairs
    positive = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)], dim=0)
    # denominator: all except self
    sim_masked = sim.masked_fill(mask, -9e15)
    logits = sim_masked.exp().sum(dim=1)

    loss = -torch.log(positive.exp() / logits)
    return loss.mean()


# 7) Prepare dataloader
transform = TabularAugment()
dataset = ContrastiveDataset(X, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# 8) Model instantiation
encoder = Encoder(input_dim=X.shape[1]).to(DEVICE)
proj_head = ProjectionHead().to(DEVICE)
optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(proj_head.parameters()), lr=LR)



# 9) Training loop
for epoch in range(1, EPOCHS+1):
    encoder.train(); proj_head.train()
    total_loss = 0
    for x1, x2 in dataloader:
        x1 = x1.to(DEVICE).float()
        x2 = x2.to(DEVICE).float()
        z1 = proj_head(encoder(x1))
        z2 = proj_head(encoder(x2))
        loss = nt_xent_loss(z1, z2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch:02d}: Loss = {total_loss/len(dataloader):.4f}")

# 10) Extract embeddings for clustering
encoder.eval()
with torch.no_grad():
    X_tensor = torch.tensor(X).to(DEVICE)
    embeddings = encoder(X_tensor.float()).cpu().numpy()

# 11) K-Means clustering on embeddings
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42).fit(embeddings)
clusters = kmeans.labels_
print("Cluster distribution:", np.bincount(clusters))


# 12) Save model and clusters if desired
os.makedirs('artifacts', exist_ok=True)
torch.save(encoder.state_dict(), 'artifacts/encoder.pth')
np.save('artifacts/embeddings.npy', embeddings)
np.save('artifacts/clusters.npy', clusters)
print("Training complete. Artifacts saved in ./artifacts.")



import polars as pl
import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import random

# -----------------------
# Configuration
# -----------------------
DATA_PATH = "./data/train.parquet"
FEATURES = [f"feature_{i}" for i in range(79)]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Tabular Augmentations
# -----------------------
class TabularAugment:
    def __init__(self, noise_std=0.05, dropout_prob=0.1):
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob

    def __call__(self, x):
        x = x.copy()
        # Gaussian noise
        noise = self.noise_std * torch.randn_like(torch.tensor(x))
        x += noise.numpy()
        # Feature dropout
        mask = torch.rand(len(x)) > self.dropout_prob
        x = x * mask.numpy()
        return x

# -----------------------
# Iterable Dataset
# -----------------------
class ParquetIterableDataset(IterableDataset):
    def __init__(self, path, features, chunk_size, transform):
        self.path = path
        self.features = features
        self.chunk_size = chunk_size
        self.transform = transform

    def __iter__(self):
        scan = pl.scan_parquet(self.path).select(self.features)
        for batch in scan.iter_chunks(self.chunk_size):  # Use Polars' streaming iterator if available
            arr = batch.to_numpy().astype('float32')
            for x in arr:
                x1 = self.transform(x)
                x2 = self.transform(x)
                yield torch.from_numpy(x1), torch.from_numpy(x2)


# -----------------------
# Model (placeholder)
# -----------------------
class Encoder(nn.Module):
    def __init__(self, input_dim=79, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ProjectionHead(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.proj(x)

# -----------------------
# Loss (NT-Xent)
# -----------------------
def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    representations = torch.cat([z1, z2], dim=0)
    similarity = torch.matmul(representations, representations.T)
    sim_exp = torch.exp(similarity / temperature)
    mask = ~torch.eye(len(similarity), dtype=bool).to(DEVICE)
    sim_exp = sim_exp.masked_select(mask).view(len(similarity), -1)
    positives = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
    positives = torch.cat([positives, positives], dim=0)
    loss = -torch.log(positives / sim_exp.sum(dim=1))
    return loss.mean()

# -----------------------
# Train Loop
# -----------------------
transform = TabularAugment()
dataset = ParquetIterableDataset(DATA_PATH, FEATURES, 50_000, transform)
dataloader = DataLoader(dataset, batch_size=512, num_workers=4, pin_memory=True)

encoder = Encoder().to(DEVICE)
proj_head = ProjectionHead().to(DEVICE)
optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(proj_head.parameters()), lr=1e-3)
scaler = torch.amp.GradScaler(device_type='cuda')

for epoch in range(10):
    for x1, x2 in dataloader:
        x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            z1 = proj_head(encoder(x1))
            z2 = proj_head(encoder(x2))
            loss = nt_xent_loss(z1, z2)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")


