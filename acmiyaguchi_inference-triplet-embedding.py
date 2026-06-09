! pip install pacmap


import os, random
from collections import defaultdict
from typing import Any
from tqdm import tqdm
import pacmap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler


class EmbeddingDataset(Dataset):
    def __init__(self,  metadata: pd.DataFrame):
        self.metadata = metadata
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple:
        row = self.metadata.iloc[idx]
        return row.image_id, torch.from_numpy(row.embeddings).to(self.device)


class ProjectionHead(nn.Module):
    """Simple projection head to transform embeddings."""

    def __init__(self, input_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        # 
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.fc(x)
        z = self.relu(z)
        return z


dino_embeddings = pd.read_parquet("/kaggle/input/preprocess-triplet-embedding/embeddings.parquet")
merged_df = pd.merge(
    pd.read_csv('/kaggle/input/animal-clef-2025/metadata.csv'),
    dino_embeddings, 
    on='image_id',
    how='inner'
).sort_values("image_id")
dataset = EmbeddingDataset(merged_df)

device = "cuda" if torch.cuda.is_available() else "cpu"
head = ProjectionHead(768, 128).to(device)

input_path = "/kaggle/input/train-triplet-embedding/head.pt"
state_dict = torch.load(input_path)
head.load_state_dict(state_dict)


dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

res = []
head.eval()
with torch.no_grad():
    for batch_image_ids, batch_embeddings in tqdm(dataloader):
        out = head(batch_embeddings).detach().cpu().numpy()
        assert out.shape[1] == 128, out.shape
        for image_id, embedding in zip(batch_image_ids, out):
            res.append({
                "image_id": int(image_id),
                "embeddings": embedding
            })

df = pd.DataFrame(res).sort_values("image_id")
display(df.head())
df.to_parquet("embeddings.parquet", index=False)


X = np.stack(dino_embeddings.embeddings)
# 1D embedding to see how the embedding is reshaped
c = pacmap.PaCMAP(n_components=1).fit_transform(X)
g = pacmap.PaCMAP().fit_transform(X)

plt.scatter(g[:,0], g[:,1], s=1, alpha=0.5, c=c)
plt.title("DINO embeddings")
plt.show()


# let's generate a plot with pacmap
X = np.stack(df.embeddings)
g = pacmap.PaCMAP().fit_transform(X)

plt.scatter(g[:,0], g[:,1], s=1, alpha=0.5, c=c)
plt.title("DINO triplet embeddings")
plt.show()

