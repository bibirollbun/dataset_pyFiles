import matplotlib.pyplot as plt
import numpy as np

def load_seismic_file(file_path):
    """
    Load a seismic waveform from a .npy file and return a NumPy array.
    Strips any extra leading dimensions so always returns a 3D array: (S, T, R).
    """
    data = np.load(file_path).astype(np.float32)
    while data.ndim > 3:
        data = data[0]
    return data

# Visualization Example: Plot waveform and spectrogram for source=0, receiver=0
file_path = "/kaggle/input/waveform-inversion/train_samples/CurveFault_A/seis2_1_0.npy"
data = load_seismic_file(file_path)  # shape: (S, T, R)

# 1a) Plot waveform
plt.figure()
plt.plot(data[0, :, 0])
plt.title("Seismic Waveform (source=0, receiver=0)")
plt.xlabel("Time step")
plt.ylabel("Amplitude")
plt.show()

# 1b) Plot spectrogram
plt.figure()
plt.specgram(data[0, :, 0])
plt.title("Spectrogram (source=0, receiver=0)")
plt.xlabel("Time step")
plt.ylabel("Frequency bin")
plt.show()



import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ---------------------------
# 1. Data Loading Function
# ---------------------------
def load_seismic_file(file_path):
    """
    Load a seismic waveform from a .npy file and return a 3D NumPy array: (S, T, R).
    Strips any extra leading dimensions.
    """
    data = np.load(file_path).astype(np.float32)
    while data.ndim > 3:
        data = data[0]
    return data

# ---------------------------
# 2. PyTorch Dataset Definition
# ---------------------------
class WaveformDataset(Dataset):
    def __init__(self, input_files, target_arrays, transform_input=None):
        """
        input_files:   list of paths to .npy seismic files
        target_arrays: list of 2D NumPy arrays (H×W) ground truth
        transform_input: optional preprocessing function
        """
        self.input_files = input_files
        self.target_arrays = target_arrays
        self.transform_input = transform_input

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # 1) Load raw data
        data = load_seismic_file(self.input_files[idx])  # shape (S, T, R)
        # 2) Optional transform
        if self.transform_input:
            data = self.transform_input(data)
        # 3) To tensor + add channel dim: (S,T,R) → (1,S,T,R)
        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        # 4) Permute to (C, D, H, W): (1,S,T,R) → (1,R,S,T)
        data_tensor = data_tensor.permute(0, 3, 1, 2)
        # 5) Target tensor (H, W)
        target_tensor = torch.tensor(self.target_arrays[idx], dtype=torch.float32)
        return data_tensor, target_tensor

# ---------------------------
# 3. Sample Data Setup & DataLoader
# ---------------------------
# Prepare only two example files
example_file   = "/kaggle/input/waveform-inversion/train_samples/CurveFault_A/seis2_1_0.npy"
example_target = np.zeros((100, 70), dtype=np.float32)
file_list      = [example_file, example_file]
target_list    = [example_target, example_target]

# Create dataset and data loader
dataset = WaveformDataset(file_list, target_list)
loader  = DataLoader(dataset, batch_size=2, shuffle=True)

# ---------------------------
# 4. Fetch one batch and visualize
# ---------------------------
inputs, targets = next(iter(loader))
# inputs: (B, 1, R, S, T), targets: (B, H, W)

# (a) Plot waveform grid (receivers × [sources × time])
grid = inputs[0, 0].cpu().numpy().reshape(inputs.shape[2], -1)
plt.figure(figsize=(6, 3))
plt.imshow(grid, aspect='auto')
plt.title("Input Waveform Grid")
plt.xlabel("Source × Time")
plt.ylabel("Receiver")
plt.colorbar(label="Amplitude")
plt.show()

# (b) Plot ground-truth velocity map
plt.figure(figsize=(4, 4))
plt.imshow(targets[0].cpu(), aspect='auto')
plt.title("Ground-Truth Velocity Map")
plt.xlabel("X index")
plt.ylabel("Y index")
plt.colorbar(label="Velocity")
plt.show()



import matplotlib.pyplot as plt

# Example: Populate 'losses' with dummy values to demonstrate plotting
# In your actual code, ensure you append avg_loss to 'losses' inside your training loop:
#     losses.append(avg_loss)

losses = [0.8, 0.6, 0.45, 0.3, 0.2]  # replace or extend with your real avg_loss values

# Plot Training Loss Curve
plt.figure()
plt.plot(range(1, len(losses) + 1), losses, marker='o')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Average L1 Loss")
plt.grid(True)
plt.show()



import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 1. Helper function: load .npy seismic file
def load_seismic_file(fp: str) -> np.ndarray:
    """
    Load a seismic waveform from a .npy file and return a 3D NumPy array (S, T, R).
    Strips any extra leading dimensions.
    """
    data = np.load(fp).astype(np.float32)
    while data.ndim > 3:
        data = data[0]
    return data

# 2. Dataset definition for PyTorch
class WaveformDataset(Dataset):
    def __init__(self, files, targets, transform=None):
        """
        files:   list of paths to .npy seismic files
        targets: list of 2D NumPy arrays (H×W) ground-truth velocity maps
        transform: optional preprocessing function
        """
        self.files = files
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load and optionally transform the waveform
        d = load_seismic_file(self.files[idx])
        if self.transform:
            d = self.transform(d)
        # Convert to tensor, add channel dim, then reorder to (C, D, H, W)
        t = torch.tensor(d, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        # Load target velocity map
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return t, y

# 3. Model definition: minimal 3D U-Net
class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        # Encoder: two 3D conv layers + ReLU, then adaptive pooling
        self.enc = nn.Sequential(
            nn.Conv3d(in_ch, 8, 3, padding=1), nn.ReLU(True),
            nn.Conv3d(8, 16, 3, padding=1),    nn.ReLU(True),
            nn.AdaptiveAvgPool3d((1, 100, 70))
        )
        # Decoder: two 3D transpose conv layers + ReLU
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1), nn.ReLU(True),
            nn.ConvTranspose3d(8, out_ch, 3, padding=1)
        )

    def forward(self, x):
        # Encode to bottleneck
        x = self.enc(x)   # → (batch,16,1,100,70)
        # Decode back to spatial dimensions
        x = self.dec(x)   # → (batch,1,1,100,70)
        x = x.squeeze(2)  # remove depth dim → (batch,1,100,70)
        return x.squeeze(1)  # remove channel dim → (batch,100,70)

# 4. Inference function
def predict(model, fps, device):
    """
    Run inference on a list of .npy file paths and return list of 2D NumPy arrays.
    """
    model.eval()
    out = []
    with torch.no_grad():
        for fp in fps:
            # Load and prepare waveform tensor
            d = load_seismic_file(fp)
            t = torch.tensor(d, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            # Model forward pass and convert to NumPy
            o = model(t).squeeze().cpu().numpy()
            out.append(o)
    return out

# 5. Main workflow
def main():
    # Gather training files and create dummy target arrays (replace with real data)
    files   = glob.glob(
        "/kaggle/input/waveform-inversion/train_samples/CurveFault_A/seis2_1_0.npy",
        recursive=True
    )[:10]
    targets = [np.zeros((100, 70), np.float32) for _ in files]
    ds      = WaveformDataset(files, targets)
    loader  = DataLoader(ds, batch_size=2, shuffle=True)

    # Use CPU-only
    device = torch.device("cpu")
    model  = UNet3D().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossfn = nn.L1Loss()
    ckpt   = "best_model.pth"

    # If checkpoint exists, load it; otherwise train one epoch and save
    if os.path.exists(ckpt):
        print(f"Loading checkpoint from {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        print("No checkpoint found → training 1 epoch")
        model.train()
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            opt.zero_grad()
            out = model(inp)
            loss = lossfn(out, tgt)
            loss.backward()
            opt.step()
        torch.save(model.state_dict(), ckpt)
        print(f"Saved checkpoint to {ckpt}")

    # Perform inference on the first sample and visualize
    sample_fp = files[0]
    preds = predict(model, [sample_fp], device)

    plt.figure()
    plt.imshow(preds[0], aspect="auto")
    plt.title("Predicted Velocity Map")
    plt.xlabel("X index")
    plt.ylabel("Y index")
    plt.colorbar(label="Velocity")
    plt.show()

if __name__ == "__main__":
    main()



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob
import os

# ---------------------------
# 1. Data Loading Function
# ---------------------------
def load_seismic_file(file_path):
    """
    Loads a seismic waveform from a .npy file and returns a NumPy array.
    Expected input shape: (S, T, R) or with extra batch dims (e.g., (1, S, T, R)).
    """
    data = np.load(file_path).astype(np.float32)
    # Remove any extra leading dimensions
    while data.ndim > 3:
        data = data[0]
    return data  # shape: (S, T, R)

# ---------------------------
# 2. PyTorch Dataset
# ---------------------------
class WaveformDataset(Dataset):
    def __init__(self, input_files, target_arrays, transform_input=None):
        """
        input_files: list of file paths to seismic .npy files
        target_arrays: list of ground truth arrays with shape (H, W)
        transform_input: optional preprocessing function
        """
        self.input_files = input_files
        self.target_arrays = target_arrays
        self.transform_input = transform_input

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Load and preprocess seismic data
        data = load_seismic_file(self.input_files[idx])  # (S, T, R)
        if self.transform_input:
            data = self.transform_input(data)
        # Convert to PyTorch tensor and add channel dimension
        # (S, T, R) -> (1, S, T, R)
        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        # Permute to (C, D, H, W): (1, S, T, R) -> (1, R, S, T)
        data_tensor = data_tensor.permute(0, 3, 1, 2)
        # Ground truth velocity map tensor
        target_tensor = torch.tensor(self.target_arrays[idx], dtype=torch.float32)  # (H, W)
        return data_tensor, target_tensor

# ---------------------------
# 3. Simplified 3D U-Net Model
# ---------------------------
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels,  8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8,           16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Pool to (1, 100, 70) for consistent spatial dimensions
            nn.AdaptiveAvgPool3d((1, 100, 70))
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(16,  8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(8, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x shape: (B, 1, D, H, W)
        x = self.encoder(x)    # -> (B, 16, 1, 100, 70)
        x = self.decoder(x)    # -> (B, 1, 1, 100, 70)
        x = x.squeeze(2)       # -> (B, 1, 100, 70)
        return x.squeeze(1)    # -> (B, 100, 70)

# ---------------------------
# 4. Sample Data Setup
# ---------------------------
example_file   = "/kaggle/input/waveform-inversion/train_samples/CurveFault_A/seis2_1_0.npy"
example_target = np.zeros((100, 70), dtype=np.float32)
file_list      = [example_file] * 10
target_list    = [example_target] * 10

# Create Dataset and DataLoader
dataset = WaveformDataset(file_list, target_list)
loader  = DataLoader(dataset, batch_size=2, shuffle=True)

# ---------------------------
# 5. Model Training Setup
# ---------------------------
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = UNet3D(in_channels=1, out_channels=1).to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode='min',
                                                 factor=0.5,
                                                 patience=2)

# ---------------------------
# 6. Training Loop
# ---------------------------
num_epochs    = 3
best_val_loss = float("inf")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} Training")
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
    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1}: avg loss = {avg_loss:.4f}")
    scheduler.step(avg_loss)
    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Model improved and saved.")

# ---------------------------
# 7. Inference Function
# ---------------------------
def predict(model, file_paths):
    model.eval()
    predictions = []
    with torch.no_grad():
        for fp in file_paths:
            data = load_seismic_file(fp)
            while data.ndim > 3:
                data = data[0]
            tensor = torch.tensor(data, dtype=torch.float32)
            tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
            output = model(tensor)
            predictions.append(output.squeeze().cpu().numpy())
    return predictions

# ---------------------------
# 8. Submission File Generation
# ---------------------------
def create_submission(oids, predictions):
    """
    Builds a submission.csv matching the sample_submission format.
    oids: list[str] identifiers without extension
    predictions: list[np.ndarray] each of shape (H, W)
    """
    # Load template to get correct column names
    sample_path = '/kaggle/input/waveform-inversion/sample_submission.csv'
    sample_df = pd.read_csv(sample_path)
    id_col = sample_df.columns[0]

    if not predictions:
        raise ValueError("No predictions provided.")
    width = predictions[0].shape[1]
    odd_indices = list(range(0, width, 2))

    rows = []
    for oid, pred in zip(oids, predictions):
        if pred.shape[1] != width:
            raise ValueError(f"Width mismatch for {oid}.")
        for y in range(pred.shape[0]):
            row_id = f"{oid}_y_{y}"
            row = [row_id] + [float(pred[y, x]) for x in odd_indices]
            rows.append(row)

    columns = [id_col] + [f"x_{i}" for i in odd_indices]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv('/kaggle/working/submission.csv', index=False)
    print("Submission saved to /kaggle/working/submission.csv with correct ID column.")

# ---------------------------
# 9. Generate Submission using sample_submission
# ---------------------------
sample_path = '/kaggle/input/waveform-inversion/sample_submission.csv'
sample_df   = pd.read_csv(sample_path)
# Fill with zeros or replace with model predictions
sample_df.iloc[:, 1:] = 0.0
sample_df.to_csv('/kaggle/working/submission.csv', index=False)
print("Sample submission written to /kaggle/working/submission.csv.")



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# ---------------------------
# 1. Data Loading Function
# ---------------------------
def load_seismic_file(file_path):
    """
    Load a seismic waveform from a .npy file and return a NumPy array (S, T, R).
    Strips any extra leading dimensions.
    """
    data = np.load(file_path).astype(np.float32)
    while data.ndim > 3:
        data = data[0]
    return data  # (S, T, R)

# ---------------------------
# 2. Dataset Definition
# ---------------------------
class WaveformDataset(Dataset):
    def __init__(self, input_files, transform_input=None):
        self.input_files = input_files
        self.transform_input = transform_input

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        data = load_seismic_file(self.input_files[idx])  # (S, T, R)
        if self.transform_input:
            data = self.transform_input(data)
        tensor = torch.tensor(data, dtype=torch.float32)  # (S, T, R)
        tensor = tensor.permute(2, 0, 1)                  # (R, S, T)
        return tensor

# ---------------------------
# 3. UNet3D Model
# ---------------------------
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels,  8, 3, padding=1), nn.ReLU(),
            nn.Conv3d(8,           16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,100,70))
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(16,  8, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose3d(8, out_channels, 3, padding=1),
        )

    def forward(self, x):
        # x: (B, 1, R, S, T)
        h = self.encoder(x)    # -> (B,16,1,100,70)
        h = self.decoder(h)    # -> (B,1,1,100,70)
        h = h.squeeze(2)       # -> (B,1,100,70)
        return h.squeeze(1)    # -> (B,100,70)

# ---------------------------
# 4. Forward Solver Stub
# ---------------------------
def forward_wave_solver(vel_map, obs):
    """
    Stub that scales observed data by mean velocity
    vel_map: (B, S, T)
    obs:     (B,1,R,S,T)
    """
    B = vel_map.size(0)
    factor = vel_map.mean(dim=(1,2)).view(B,1,1,1,1)
    return obs * factor

# ---------------------------
# 5. Prediction Helper
# ---------------------------
def predict(model, file_paths, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for fp in file_paths:
            data = load_seismic_file(fp)
            x = torch.tensor(data, dtype=torch.float32)             # (S, T, R)
            x = x.permute(2,0,1).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,R,S,T)
            vel = model(x)                                          # (1,100,70)
            preds.append(vel.squeeze(0).cpu().numpy())
    return preds

# ---------------------------
# 6. Submission Helper
# ---------------------------
def create_submission(oids, predictions, sample_csv, out_csv):
    # Read sample header
    with open(sample_csv) as f:
        header = f.readline().strip().split(',')
    x_cols = header[1:]
    with open(out_csv, 'w') as f:
        f.write(','.join(header) + '\n')
        for oid, pred in zip(oids, predictions):
            for y in range(pred.shape[0]):
                row_id = f"{oid}_y_{y}"
                vals = [f"{pred[y, int(col.split('_')[1])]:.6f}" for col in x_cols]
                f.write(','.join([row_id]+vals) + '\n')
    print(f"Submission saved to {out_csv}")

# ---------------------------
# 7. Main Workflow
# ---------------------------
if __name__ == "__main__":
    # 7.1 Gather and dedup file list
    file_list = glob.glob("/kaggle/input/waveform-inversion/train_samples/**/*.npy", recursive=True)
    unique_files = list(dict.fromkeys(file_list))

    # 7.2 Dataset & DataLoader
    dataset = WaveformDataset(unique_files)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True)

    # 7.3 Model & optimizer
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = UNet3D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 7.4 Training with FWI Loss
    losses = []
    num_epochs = 3
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0
        for waveform in loader:
            obs = waveform.unsqueeze(1).to(device)  # (B,1,R,S,T)
            optimizer.zero_grad()
            vel_pred = model(obs)                   # (B,100,70)
            d_syn    = forward_wave_solver(vel_pred, obs)
            loss     = 0.5 * ((d_syn - obs)**2).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch}: avg FWI loss = {avg_loss:.4e}")

    # 7.5 Plot Training Loss
    plt.figure()
    plt.plot(range(1, num_epochs+1), losses, marker='o')
    plt.title("FWI Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.show()

    # 7.6 Plot Observed vs Synthetic for first receiver
    model.eval()
    with torch.no_grad():
        wf = loader.dataset[0]
        obs = torch.tensor(wf, dtype=torch.float32).permute(1,2,0).unsqueeze(0).unsqueeze(0).to(device)
        vel_pred = model(obs)
        d_syn    = forward_wave_solver(vel_pred, obs)
        obs_np   = obs.squeeze().cpu().numpy()   # (R,S,T)
        syn_np   = d_syn.squeeze().cpu().numpy()

    plt.figure(figsize=(6,4))
    plt.plot(obs_np[0,0,:], label="Observed")
    plt.plot(syn_np[0,0,:], '--', label="Synthetic")
    plt.title("Observed vs Synthetic Trace (Receiver 0)")
    plt.xlabel("Time sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    # 7.7 Inference & Submission
    preds = predict(model, unique_files, device)
    oids  = [os.path.splitext(os.path.basename(fp))[0] for fp in unique_files]
    sample_csv = "/kaggle/input/waveform-inversion/sample_submission.csv"
    out_csv    = "/kaggle/working/submission.csv"
    create_submission(oids, preds, sample_csv, out_csv)

    # 7.8 Plot Predicted Velocity Map from submission.csv
    lines = open(out_csv).read().splitlines()
    header = lines[0].split(',')
    ids    = [line.split(',')[0] for line in lines[1:]]
    vals   = [list(map(float, line.split(',')[1:])) for line in lines[1:]]
    x_idxs = [int(h.split('_')[1]) for h in header[1:]]

    first_oid = ids[0].split('_y_')[0]
    rows = [(ids[i], vals[i]) for i in range(len(ids)) if ids[i].startswith(first_oid + '_y_')]
    y_idxs = [int(rid.split('_y_')[1]) for rid, _ in rows]
    vels   = [v for _, v in rows]

    order = np.argsort(y_idxs)
    grid  = np.array(vels)[order, :]

    plt.figure(figsize=(6,5))
    plt.imshow(grid, aspect='auto', origin='lower',
               extent=[min(x_idxs), max(x_idxs), min(y_idxs), max(y_idxs)])
    plt.title(f"Predicted Velocity Map for {first_oid}")
    plt.xlabel("X index")
    plt.ylabel("Y index")
    plt.colorbar(label="Velocity")
    plt.show()



import matplotlib.pyplot as plt

# --- 1) Plot Training Loss Curve ---
# Suppose you modified your loop to collect avg_loss each epoch:
losses = []  # before training
for epoch in range(1, num_epochs+1):
    model.train()
    total_loss = 0.0
    for obs in loader:
        # ... compute loss ...
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch}: avg FWI loss = {avg_loss:.4e}")

# After training:
plt.figure()
plt.plot(range(1, len(losses)+1), losses, marker='o')
plt.title("FWI Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Average FWI Loss")
plt.grid(True)
plt.show()


# --- 2) Plot Observed vs. Synthetic for One Trace ---
# Pick the first sample & first receiver
model.eval()
with torch.no_grad():
    # Load a single waveform
    fp = file_list[0]
    obs = torch.tensor(load_seismic_file(fp), dtype=torch.float32)  # (S, T, R)
    obs = obs.permute(2,0,1).unsqueeze(0).unsqueeze(0).to(device)    # (1,1,R,S,T)

    # Predict velocity & simulate synthetic
    vel_pred = model(obs)                                          # (1,100,70)
    d_syn    = forward_wave_solver(vel_pred, obs)                  # (1,1,R,100,70)

    # Move to CPU & NumPy
    obs_np   = obs.squeeze().cpu().numpy()     # (R, S, T)
    syn_np   = d_syn.squeeze().cpu().numpy()   # (R, S, T)

# Plot the time‐series of source 0, receiver 0 (for example)
plt.figure(figsize=(8,4))
plt.plot(obs_np[0,0,:], label="Observed")
plt.plot(syn_np[0,0,:], label="Synthetic", linestyle="--")
plt.title("Observed vs Synthetic Trace (Receiver 0)")
plt.xlabel("Time sample")
plt.ylabel("Amplitude")
plt.legend()
plt.show()


