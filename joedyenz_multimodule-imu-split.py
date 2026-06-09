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


import torch
from torch.utils.data import Dataset

class RotationWindowDataset(Dataset):
    def __init__(
        self,
        df,
        window_size=20,
        stride=5,
        feature_cols=("rot_w", "rot_x", "rot_y", "rot_z"),
        label_col="gesture"
    ):
        self.samples = []

        for seq_id, seq_df in df.groupby("sequence_id"):
            data = seq_df[feature_cols].values
            labels = seq_df[label_col].values

            L = len(seq_df)
            for start in range(0, L - window_size + 1, stride):
                end = start + window_size

                x = data[start:end].T   # (4, T)
                y = labels[start:end]

                # Skip windows with NaNs
                if np.isnan(x).any():
                    continue

                print(y[:5], y.dtype)

                # Majority label (or center label)
                target = np.bincount(y).argmax()

                self.samples.append((
                    torch.tensor(x, dtype=torch.float32),
                    torch.tensor(target, dtype=torch.long)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



import torch
import torch.nn as nn
import torch.nn.functional as F


class AxisCNN(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2),
            #nn.BatchNorm1d(hidden_dim),
            nn.GroupNorm(4, hidden_dim),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            #nn.BatchNorm1d(hidden_dim),
            nn.GroupNorm(4, hidden_dim),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            #nn.BatchNorm1d(hidden_dim),
            nn.GroupNorm(4, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x: (B, 1, T)
        return: (B, hidden_dim)
        """
        x = self.net(x)
        x = x.mean(dim=-1)  # temporal pooling
        return x



class RotationEncoder(nn.Module):
    def __init__(
        self,
        axis_hidden_dim=32,
        fusion_hidden_dim=128,
        k_classes=12
    ):
        super().__init__()

        # One CNN per axis
        self.axis_cnns = nn.ModuleList([
            AxisCNN(axis_hidden_dim) for _ in range(4)
        ])

        fusion_input_dim = 4 * axis_hidden_dim

        # Late fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
        )

        # k-class motion primitive head
        self.motion_head = nn.Linear(fusion_hidden_dim, k_classes)

    def forward(self, x):
        """
        x: (B, 4, T)
        returns:
            logits: (B, k_classes)
            embedding: (B, fusion_hidden_dim)
        """
        axis_features = []

        for i in range(4):
            xi = x[:, i:i+1, :]      # (B, 1, T)
            zi = self.axis_cnns[i](xi)  # (B, axis_hidden_dim)
            axis_features.append(zi)

        z = torch.cat(axis_features, dim=1)  # (B, 4 * axis_hidden_dim)

        embedding = self.fusion_mlp(z)        # (B, fusion_hidden_dim)
        logits = self.motion_head(embedding)  # (B, k_classes)

        return logits, embedding


DATA_DIR = "/kaggle/input/cmi-detect-behavior-with-sensor-data"

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
print(train_df.columns)


from torch.utils.data import DataLoader

labels = train_df["gesture"].unique()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

train_df["gesture_id"] = train_df["gesture"].map(label2id)

from sklearn.model_selection import train_test_split

# Unique sequences
all_sequences = train_df["sequence_id"].unique()

# 80 / 20 split
train_seqs, test_seqs = train_test_split(
    all_sequences,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

train_df_split = train_df[train_df["sequence_id"].isin(train_seqs)]
test_df_split  = train_df[train_df["sequence_id"].isin(test_seqs)]

print("Train sequences:", len(train_seqs))
print("Test sequences:", len(test_seqs))


#dataset = RotationWindowDataset(
#    df=train_df,
#    window_size=20,
#    stride=5,
#    feature_cols = ["rot_w", "rot_x", "rot_y", "rot_z"],
#    label_col="gesture_id"
#)

#loader = DataLoader(
#    dataset,
#    batch_size=32,
#    shuffle=True,
#    num_workers=2,
#    pin_memory=True
#)

train_dataset = RotationWindowDataset(
    df=train_df_split,
    window_size=20,
    stride=5,
    feature_cols=["rot_w", "rot_x", "rot_y", "rot_z"],
    label_col="gesture_id"
)

test_dataset = RotationWindowDataset(
    df=test_df_split,
    window_size=20,
    stride=5,
    feature_cols=["rot_w", "rot_x", "rot_y", "rot_z"],
    label_col="gesture_id"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


#model = RotationEncoder()

device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = train_df["gesture_id"].nunique()

model = RotationEncoder(k_classes=num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train loss: {avg_loss:.4f}")


model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)

        logits, _ = model(x)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.numpy())



print(logits)
print(embedding)


# ACTIVITY

tof_cols = [c for c in train_df.columns if c.startswith("tof")]
tof_data = train_df[tof_cols]

col_var = tof_data.var()
col_std = tof_data.std()

col_var.sort_values(ascending=False).head(10)



# MOTION SENSIVITY

tof_diff = tof_data.diff().abs()
temporal_energy = tof_diff.mean()

temporal_energy.sort_values(ascending=False).head(10)



# CORRELATION

sample = tof_data.sample(n=5000, random_state=42)
corr = sample.corr()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.show()


# CLUSTERS

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# impute
imputer = SimpleImputer(strategy="median")
tof_clean = pd.DataFrame(imputer.fit_transform(tof_data), columns=tof_data.columns)

# extract features
features = pd.DataFrame({
    "std": tof_clean.std(),
    "energy": tof_clean.diff().abs().mean(),
    "mean": tof_clean.mean(),
})

# scale
X = StandardScaler().fit_transform(features)

# cluster
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X)

features["cluster"] = clusters
print(features)


# DIMENSIONALITY

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(StandardScaler().fit_transform(tof_data.dropna()))

explained = np.cumsum(pca.explained_variance_ratio_)

plt.plot(explained)
plt.axhline(0.9, color='r')
plt.show()



# VISUALIZER

top_cols = temporal_energy.sort_values(ascending=False).head(5).index
tof_data[top_cols].iloc[:500].plot()

cluster_cols = features[features.cluster == 1].index
tof_data[cluster_cols[:5]].iloc[:500].plot()



grouped = train_df.groupby("gesture")[tof_cols].mean()

