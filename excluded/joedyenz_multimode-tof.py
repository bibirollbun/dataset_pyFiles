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
import torch.nn as nn

class ToFAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8, hidden_dim=128):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent



import numpy as np

DATA_DIR = "/kaggle/input/cmi-detect-behavior-with-sensor-data"
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
tof_cols = [c for c in train_df.columns if c.startswith("tof")]
tof_data = train_df[tof_cols]

X = tof_data.values.astype(np.float32)

# Optionally normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Convert to PyTorch tensor
X_tensor = torch.from_numpy(X)

from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=128, shuffle=True)



device = "cuda" if torch.cuda.is_available() else "cpu"
model = ToFAutoencoder(input_dim=X.shape[1], latent_dim=8).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)



EPOCHS = 30

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in loader:
        x = batch[0].to(device)

        reconstruction, latent = model(x)

        loss = criterion(reconstruction, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if torch.isnan(loss):
            print("NaN detected in loss!")
            print("Batch min:", x.min().item())
            print("Batch max:", x.max().item())
            print("Reconstruction min:", reconstruction.min().item())
            print("Reconstruction max:", reconstruction.max().item())
            break

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(loader):.6f}")



model.eval()
with torch.no_grad():
    reconstruction, latent = model(X_tensor.to(device))

latent = latent.cpu().numpy()


