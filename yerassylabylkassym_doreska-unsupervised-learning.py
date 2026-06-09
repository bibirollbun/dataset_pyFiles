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


sample_submission = pd.read_csv('/kaggle/input/tst-day-2-upsolving/sample_submission.csv')
train = pd.read_csv('/kaggle/input/tst-day-2-upsolving/train.csv')


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 4))

for i, col in enumerate(['gk_diving']):
    plt.subplot(1, 2, i+1)
    sns.histplot(train[col], kde=True, bins=30)
    plt.title(f'{col}')

plt.tight_layout()
plt.show()


gk = train[train['gk_diving'] >= 40]
active = train[train['gk_diving'] < 40]

gk_ids = gk.id.values
active_ids = active.id.values
gk_ids


class FuzzyCMeans:
    def __init__(self, n_clusters=3, m=2.0, max_iter=150, error=1e-5, random_state=None):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state

    def _initialize_membership(self, n_samples):
        rng = np.random.default_rng(self.random_state)
        u = rng.random((n_samples, self.n_clusters))
        u /= np.sum(u, axis=1, keepdims=True)
        return u

    def _update_centers(self, X, u):
        um = u ** self.m
        return (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)

    def _update_membership(self, X, centers):
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        for i, c in enumerate(centers):
            distances[:, i] = np.linalg.norm(X - c, axis=1)

        # Prevent division by zero
        distances = np.fmax(distances, 1e-10)

        exponent = 2 / (self.m - 1)
        inv_distances = distances[:, :, np.newaxis] / distances[:, np.newaxis, :]
        u_new = 1.0 / np.sum(inv_distances ** exponent, axis=2)
        return u_new

    def fit(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        u = self._initialize_membership(n_samples)

        for i in range(self.max_iter):
            u_old = u.copy()
            self.centers = self._update_centers(X, u)
            u = self._update_membership(X, self.centers)

            max_change = np.max(np.abs(u - u_old))
            if max_change < self.error:
                break

        self.u = u
        return self

    def predict(self, X):
        if not hasattr(self, "u"):
            raise Exception("Model not yet fitted.")
        return np.argmax(self.u, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)


from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import mode

X = np.array(active.drop(['gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes'], axis=1))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Autoencoder definition
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# 3. Convert to torch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 4. Train autoencoder
input_dim = X.shape[1]
autoencoder = Autoencoder(input_dim, latent_dim=10)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(50):
    total_loss = 0
    for batch in loader:
        x_batch = batch[0]
        optimizer.zero_grad()
        x_recon, _ = autoencoder(x_batch)
        loss = loss_fn(x_recon, x_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# 5. Get encoded features
with torch.no_grad():
    _, X_encoded = autoencoder(X_tensor)
    X_encoded_np = X_encoded.numpy()

# PCA -- bad

# 6. Clustering with Gaussian Mixture
# gmm = GaussianMixture(n_components=10)
# gmm_labels = gmm.fit_predict(X_encoded_np)


fcm = FuzzyCMeans(n_clusters=9, m=2.0)
fcm.fit(X_encoded_np)

# Get hard labels
labels = fcm.predict(X_encoded_np)


labels += 1

all_df = pd.DataFrame(
    data={
        'id': gk_ids.tolist() + active_ids.tolist(),
        'cluster': [0] * len(gk_ids) + labels.tolist()
    }
)


all_df.to_csv('submission.csv', index=False)
all_df




