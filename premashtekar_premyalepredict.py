# ===============================
# Yale/UNC-CH - FWI Competition Starter (PyTorch)
# Physics-Guided ML for Velocity Map Prediction
# ===============================

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ===============================
# Custom Dataset for Seismic Data
# ===============================
class SeismicDataset(Dataset):
    def __init__(self, waveforms, labels=None):
        self.waveforms = waveforms  # shape: (N, 1, H, W)
        self.labels = labels        # shape: (N, H, W) or None

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        x = self.waveforms[idx]
        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        return x

# ===============================
# A Simple U-Net Style CNN
# ===============================
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(32, 1, 3, padding=1)  # Final output layer

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.dec1(x2)
        out = self.out(x3)
        return out.squeeze(1)  # Remove channel dimension

# ===============================
# Training Function
# ===============================
def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()  # MAE loss

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}")

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                val_loss += criterion(y_pred, y).item()
            print(f"Val Loss: {val_loss / len(val_loader):.4f}")

# ===============================
# Create Submission CSV
# ===============================
def make_submission(preds, oids, filename="submission.csv"):
    # preds: np.array (N, H, W), oids: list of strings
    rows = []
    for i, pred in enumerate(preds):
        for y in range(pred.shape[0]):
            values = pred[y, 1::2]  # Only odd x columns: x_1, x_3, ..., x_69
            row_id = f"{oids[i]}_y_{y}"
            row = [row_id] + values.tolist()
            rows.append(row)
    columns = ["oid_ypos"] + [f"x_{i}" for i in range(1, 70, 2)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(filename, index=False)
    print(f"Saved submission to {filename}")

# ===============================
# Main Driver Code (Simulated Example)
# ===============================
if __name__ == "__main__":
    # Simulated data (replace with real .npy/.csv data)
    N, H, W = 10, 100, 70  # 10 samples, 100 rows, 70 columns
    waveforms = np.random.randn(N, 1, H, W).astype(np.float32)
    velocities = np.random.uniform(2500, 3500, size=(N, H, W)).astype(np.float32)

    # Split data into training and validation
    train_x, val_x, train_y, val_y = train_test_split(waveforms, velocities, test_size=0.2)

    # Create Dataloaders
    train_dataset = SeismicDataset(train_x, train_y)
    val_dataset = SeismicDataset(val_x, val_y)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    # Initialize model and train
    model = SimpleUNet()
    train_model(model, train_loader, val_loader, num_epochs=5)

    # Predict on new test data (simulated)
    test_waveforms = np.random.randn(2, 1, H, W).astype(np.float32)
    test_dataset = SeismicDataset(test_waveforms)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model.eval()
    preds = []
    with torch.no_grad():
        for x in test_loader:
            pred = model(x)
            preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # Dummy OIDs for submission example
    make_submission(preds, oids=["00001", "00002"])

