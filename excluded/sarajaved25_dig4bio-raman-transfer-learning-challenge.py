 
# 0. IMPORTS
# =============================================================================
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
import pandas as pd
import numpy as np
import math
import random
import warnings
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CHALLENGE DATASET CLASS
# =============================================================================
class ChallengeDataset(Dataset):
    def __init__(self, X, y=None, is_train=True, mean=None, std=None):
        self.is_train = is_train

        # Standardize the features
        if is_train:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0) + 1e-8
        else:
            self.mean = mean
            self.std = std

        self.X = ((X - self.mean) / self.std).astype(np.float32)
        self.X = self.X[:, np.newaxis, :]  # Add channel dimension for Conv1D

        if is_train:
            self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {'spectrum': torch.tensor(self.X[idx])}
        if self.is_train:
            item['concentration'] = torch.tensor(self.y[idx])
        return item

# =============================================================================
# 2. MODEL ARCHITECTURE (RamanXception)
# =============================================================================
class ResZeroBlock(nn.Module):
    def __init__(self, skip_part, model_part):
        super().__init__()
        self.skip_part = skip_part
        self.model_part = model_part
        self.factor = nn.Parameter(torch.tensor(0.0))

    def forward(self, X):
        return self.skip_part(X) + self.factor * self.model_part(X)

class Identity(nn.Module):
    def forward(self, X):
        return X

class RamanXception(nn.Module):
    def __init__(
        self, spectra_size, initial_channels, entry_channels, num_mid_blocks,
        exit_channels, num_concentrations, fc_dims, fc_dropout,
        activation_function='ELU', dtype=torch.float32, **kwargs,
    ):
        super().__init__()
        activation_fn = getattr(nn, activation_function)
        self.spatial_dimensions = [spectra_size]

        # Initial Convs
        initial_layers = nn.Sequential()
        in_ch = 1
        for i, out_ch in enumerate(initial_channels):
            initial_layers.add_module(
                f'initial_conv_{i}',
                nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, dtype=dtype, bias=False)
            )
            self.spatial_dimensions.append(
                math.floor((self.spatial_dimensions[-1] - 1) / 2 + 1)
            )
            initial_layers.add_module(f'initial_bn_{i}', nn.BatchNorm1d(out_ch, dtype=dtype))
            initial_layers.add_module(f'initial_act_{i}', activation_fn())
            in_ch = out_ch

        # Entry Flow
        entry_flow = nn.Sequential()
        for i, out_ch in enumerate(entry_channels):
            entry_flow.add_module(
                f'entry_flow_{i}',
                ResZeroBlock(
                    skip_part=nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=2, dtype=dtype, bias=False),
                    model_part=nn.Sequential(
                        activation_fn(),
                        nn.Conv1d(in_ch, in_ch, kernel_size=3, groups=in_ch, padding=1, dtype=dtype, bias=False),
                        nn.Conv1d(in_ch, out_ch, kernel_size=1, dtype=dtype, bias=False),
                        nn.BatchNorm1d(out_ch, dtype=dtype),
                        nn.MaxPool1d(3, stride=2, padding=1)
                    )
                )
            )
            self.spatial_dimensions.append(
                math.floor((self.spatial_dimensions[-1] - 1) / 2 + 1)
            )
            in_ch = out_ch

        # Middle Flow
        middle_flow = nn.Sequential()
        for i in range(num_mid_blocks):
            middle_flow.add_module(
                f'middle_flow_{i}',
                ResZeroBlock(
                    skip_part=Identity(),
                    model_part=nn.Sequential(
                        activation_fn(),
                        nn.Conv1d(in_ch, in_ch, kernel_size=3, groups=in_ch, padding=1, dtype=dtype, bias=False),
                        nn.BatchNorm1d(in_ch, dtype=dtype),
                        activation_fn(),
                        nn.Conv1d(in_ch, in_ch, kernel_size=3, groups=in_ch, padding=1, dtype=dtype, bias=False),
                        nn.BatchNorm1d(in_ch, dtype=dtype)
                    )
                )
            )
            self.spatial_dimensions.append(self.spatial_dimensions[-1])

        # Exit Flow
        exit_flow = nn.Sequential()
        for i, (_, out_ch) in enumerate(exit_channels):
            exit_flow.add_module(
                f'exit_flow_{i}',
                ResZeroBlock(
                    skip_part=nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=2, dtype=dtype, bias=False),
                    model_part=nn.Sequential(
                        activation_fn(),
                        nn.Conv1d(in_ch, in_ch, kernel_size=3, groups=in_ch, padding=1, dtype=dtype, bias=False),
                        nn.Conv1d(in_ch, out_ch, kernel_size=1, dtype=dtype, bias=False),
                        nn.BatchNorm1d(out_ch, dtype=dtype),
                        nn.MaxPool1d(3, stride=2, padding=1)
                    )
                )
            )
            self.spatial_dimensions.append(
                math.floor((self.spatial_dimensions[-1] - 1) / 2 + 1)
            )
            in_ch = out_ch

        self.conv_net = nn.Sequential(initial_layers, entry_flow, middle_flow, exit_flow)

        self.fc_input_dim = int(in_ch * self.spatial_dimensions[-1])
        self.fc_net = nn.Sequential()
        in_dim = self.fc_input_dim
        for i, out_dim in enumerate(fc_dims):
            self.fc_net.add_module(f'fc_net_{i}', nn.Linear(in_dim, out_dim, dtype=dtype))
            self.fc_net.add_module(f'fc_relu_{i}', nn.ReLU())
            self.fc_net.add_module(f'fc_dropout_{i}', nn.Dropout(fc_dropout))
            in_dim = out_dim
        self.fc_net.add_module('output_layer', nn.Linear(in_dim, num_concentrations, dtype=dtype))

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.reshape(x, (-1, self.fc_input_dim))
        return self.fc_net(x)

# =============================================================================
# 3. FEATURE ENGINEERING + DATALOADING
# =============================================================================
def load_and_prepare_data(train_path, test_path):
    train_df = pd.read_csv(train_path, dtype=str).iloc[:96].copy()
    spectral_cols = train_df.columns[1:-4]
    for col in spectral_cols:
        train_df[col] = pd.to_numeric(train_df[col].str.replace(r'[\[\]]', '', regex=True), errors='coerce')
    X_train = train_df[spectral_cols].values.astype(np.float32)

    test_df = pd.read_csv(test_path, header=None, dtype=str)
    spectral_cols_test = test_df.columns[1:]
    for col in spectral_cols_test:
        test_df[col] = pd.to_numeric(test_df[col].str.replace(r'[\[\]]', '', regex=True), errors='coerce')
    X_test = test_df[spectral_cols_test].values.astype(np.float32)

    # Smoothing
    kernel = np.ones(5)/5.0
    X_train_smooth = np.array([np.convolve(x, kernel, mode='same') for x in X_train])
    X_test_smooth = np.array([np.convolve(x, kernel, mode='same') for x in X_test])

    # First derivative
    X_train_der = np.gradient(X_train, axis=1)
    X_test_der = np.gradient(X_test, axis=1)

    # Aggregated stats
    def make_stats(raw, der):
        return np.stack([
            raw.mean(axis=1), raw.std(axis=1),
            raw.max(axis=1), raw.min(axis=1),
            der.mean(axis=1), der.std(axis=1),
            der.max(axis=1), der.min(axis=1)
        ], axis=1).astype(np.float32)

    stats_train = make_stats(X_train, X_train_der)
    stats_test = make_stats(X_test, X_test_der)

    # Concatenate features
    X_train_aug = np.hstack([X_train, stats_train])
    X_test_aug = np.hstack([X_test, stats_test])

    # Targets
    target_cols = ['Glucose (g/L)', 'Sodium Acetate (g/L)', 'Magnesium Acetate (g/L)']
    y_train = np.repeat(train_df[target_cols].astype(float).values, 2, axis=0)

    return X_train_aug, y_train, X_test_aug

# =============================================================================
# 4. TRAINING FUNCTIONS
# =============================================================================
def create_training_objects(config):
    config['initial_channels'] = [config['initial_channels'], 2*config['initial_channels']]
    entry = [config['entry_channels_start']]
    for _ in range(config['entry_length']):
        entry.append(int(config['entry_factor'] * entry[-1]))
    config['entry_channels'] = entry

    exit_start = entry[-1]
    exit_c = []
    for _ in range(config['exit_length']):
        mid = int(exit_start * math.sqrt(config['exit_factor']))
        out = int(exit_start * config['exit_factor'])
        exit_c.append((mid, out))
        exit_start = out
    config['exit_channels'] = exit_c
    config['fc_dims'] = [config['fc_dims']]

    model = RamanXception(**config)
    opt = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    sched = ExponentialLR(opt, gamma=config['gamma'])
    return model, opt, sched

def train_model(model, loader, opt, sched, loss_fn, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total = 0
        for b in loader:
            x = b['spectrum'].to(device)
            y = b['concentration'].to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total/len(loader):.6f}")

def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for b in loader:
            p = model(b['spectrum'].to(device))
            preds.append(p.cpu().numpy())
    arr = np.concatenate(preds, axis=0)
    return (arr[0::2] + arr[1::2]) / 2.0

# =============================================================================
# 5. MAIN SCRIPT
# =============================================================================
def main():
    TRAIN_FILE = '/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/transfer_plate.csv'
    TEST_FILE = '/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/96_samples.csv'
    SAMPLE_SUB = '/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/sample_submission.csv'
    SUB_OUT = 'submission.csv'
    EPOCHS = 2500

    X_train, y_train, X_test = load_and_prepare_data(TRAIN_FILE, TEST_FILE)
    train_ds = ChallengeDataset(X_train, y_train, is_train=True)
    train_ld = DataLoader(train_ds, batch_size=model_config['batch_size'], shuffle=True)
    test_ds = ChallengeDataset(X_test, is_train=False, mean=train_ds.mean, std=train_ds.std)
    test_ld = DataLoader(test_ds, batch_size=model_config['batch_size'] * 2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, opt, sched = create_training_objects(model_config)
    model.to(device)
    loss_fn = nn.MSELoss()

    train_model(model, train_ld, opt, sched, loss_fn, device, num_epochs=EPOCHS)

    preds = predict(model, test_ld, device)
    preds[preds < 0] = 0
    sub = pd.read_csv(SAMPLE_SUB)
    sub['Glucose'] = preds[:, 0]
    sub['Sodium Acetate'] = preds[:, 1]
    sub['Magnesium Sulfate'] = preds[:, 2]
    sub.to_csv(SUB_OUT, index=False)
    print("Saved", SUB_OUT)

# =============================================================================
# 6. CONFIG & RUN
# =============================================================================
model_config = {
    'spectra_size': 2056,
    'initial_channels': 8, 'entry_channels_start': 17, 'entry_factor': 1.56925,
    'entry_length': 3, 'exit_factor': 1.56925, 'exit_length': 3,
    'num_mid_blocks': 4, 'fc_dims': 101, 'fc_dropout': 0.11749,
    'learning_rate': 0.001, 'gamma': 0.99217, 'batch_size': 8,
    'num_concentrations': 3, 'activation_function': 'ELU', 'dtype': torch.float32
}

if __name__ == '__main__':
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    main()


