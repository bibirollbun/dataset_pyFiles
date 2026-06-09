import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm
import matplotlib.pyplot as plt

import glob
import os
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from colorama import init, Fore, Style
from torch.utils.data import Dataset, DataLoader
from matplotlib.gridspec import GridSpec

init(autoreset=True)


def load_seismic_file(file_path, target_shape=(100, 70)):
   
    data = np.load(file_path).astype(np.float32)

   
    while data.ndim > 3:
        data = data[0]

   
    if data.shape[1:] != target_shape:
        data = data[:, :target_shape[0], :target_shape[1]]

   
    data = (data - data.mean()) / (data.std() + 1e-8)

    return data  # shape: (S, T, R)


class WaveformDataset(Dataset):
    def __init__(self, input_files, target_arrays, transform_input=None, target_shape=(100, 70)):
        self.input_files = input_files
        self.target_arrays = target_arrays
        self.transform_input = transform_input
        self.target_shape = target_shape

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        data = load_seismic_file(self.input_files[idx], self.target_shape)  # (S, T, R)
        if self.transform_input:
            data = self.transform_input(data)

        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, S, T, R)
        data_tensor = data_tensor.permute(0, 3, 1, 2)  # (1, R, S, T)

        target_tensor = torch.tensor(self.target_arrays[idx], dtype=torch.float32)  # (H, W)
        return data_tensor, target_tensor


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.AdaptiveAvgPool3d((1, 100, 70))
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.ConvTranspose3d(8, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)    # -> (B, 16, 1, 100, 70)
        x = self.decoder(x)    # -> (B, 1, 1, 100, 70)
        x = x.squeeze(2)       # -> (B, 1, 100, 70)
        return x.squeeze(1)    # -> (B, 100, 70)


example_file   = "/kaggle/input/waveform-inversion/train_samples/CurveFault_A/seis2_1_0.npy"
example_target = np.zeros((100, 70), dtype=np.float32)
file_list      = [example_file] * 10
target_list    = [example_target] * 10

dataset = WaveformDataset(file_list, target_list)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=2)


device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = UNet3D(in_channels=1, out_channels=1).to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode='min',
                                                 factor=0.5,
                                                 patience=2)


num_epochs = 30
best_val_loss = float("inf")
early_stop_counter = 0
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training")

    for inputs, targets in progress_bar:
        inputs  = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    scheduler.step(avg_val_loss)

    if early_stop_counter >= 5:
        print("Early stopping triggered.")
        break


plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss Curve")
plt.show()


def predict(model, file_paths):
    model.eval()
    predictions = []
    with torch.no_grad():
        for fp in file_paths:
            data = load_seismic_file(fp)
            tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            output = model(tensor)
            predictions.append(output.squeeze().cpu().numpy())
    return predictions

def create_submission(oids, predictions):
    sample_path = '/kaggle/input/waveform-inversion/sample_submission.csv'
    sample_df = pd.read_csv(sample_path)
    id_col = sample_df.columns[0]

    width = predictions[0].shape[1]
    odd_indices = list(range(0, width, 2))

    rows = []
    for oid, pred in zip(oids, predictions):
        for y in range(pred.shape[0]):
            row_id = f"{oid}_y_{y}"
            row = [row_id] + [float(pred[y, x]) for x in odd_indices]
            rows.append(row)

    columns = [id_col] + [f"x_{i}" for i in odd_indices]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv('/kaggle/working/submission.csv', index=False)
    print("Submission saved to /kaggle/working/submission.csv.")


test_files = glob.glob("/kaggle/input/waveform-inversion/test_samples/**/*.npy", recursive=True)
if test_files:
    model.load_state_dict(torch.load("best_model.pth"))
    preds = predict(model, test_files)
    oids = [os.path.splitext(os.path.basename(fp))[0] for fp in test_files]
    create_submission(oids, preds)
else:
    # Fallback to zero-filled submission
    sample_path = '/kaggle/input/waveform-inversion/sample_submission.csv'
    sample_df = pd.read_csv(sample_path)
    sample_df.iloc[:, 1:] = 0.0
    sample_df.to_csv('/kaggle/working/submission.csv', index=False)
    print("Fallback submission written to /kaggle/working/submission.csv.")

