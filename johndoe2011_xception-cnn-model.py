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
# 1. MODEL ARCHITECTURE (FINAL CORRECTED VERSION)
# =============================================================================
class ResZeroBlock(nn.Module):
    def __init__(self, skip_part, model_part):
        super(ResZeroBlock, self).__init__()
        self.skip_part = skip_part
        self.model_part = model_part
        self.factor = nn.parameter.Parameter(torch.tensor(0.))
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
        super(RamanXception, self).__init__()
        activation_fn = getattr(nn, activation_function)
        self.spatial_dimensions = [spectra_size]

        # Initial convolution layers
        initial_layers = nn.Sequential()
        in_ch = 1
        for i, out_ch in enumerate(initial_channels):
            initial_layers.add_module(f'initial_conv_{i}', nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, dtype=dtype, bias=False))
            self.spatial_dimensions.append(math.floor((self.spatial_dimensions[-1] - 1) / 2 + 1))
            initial_layers.add_module(f'initial_bn_{i}', nn.BatchNorm1d(out_ch, dtype=dtype))
            initial_layers.add_module(f'initial_act_{i}', activation_fn())
            in_ch = out_ch

        # Entry flow
        entry_flow = nn.Sequential()
        in_ch = initial_channels[-1]
        for i, out_ch in enumerate(entry_channels):
            entry_flow.add_module(f'entry_flow_{i}', ResZeroBlock(
                skip_part=nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=2, dtype=dtype, bias=False),
                model_part=nn.Sequential(
                    activation_fn(),
                    nn.Conv1d(in_ch, in_ch, kernel_size=3, groups=in_ch, padding=1, dtype=dtype, bias=False),
                    nn.Conv1d(in_ch, out_ch, kernel_size=1, dtype=dtype, bias=False),
                    nn.BatchNorm1d(out_ch, dtype=dtype),
                    nn.MaxPool1d(3, stride=2, padding=1)
                )
            ))
            self.spatial_dimensions.append(math.floor((self.spatial_dimensions[-1] - 1) / 2 + 1))
            in_ch = out_ch

        # Middle flow
        middle_flow = nn.Sequential()
        for i in range(num_mid_blocks):
            middle_flow.add_module(f'middle_flow_{i}', ResZeroBlock(
                skip_part=Identity(),
                model_part=nn.Sequential(
                    activation_fn(),
                    nn.Conv1d(in_ch, in_ch, kernel_size=3, groups=in_ch, padding=1, dtype=dtype, bias=False),
                    nn.BatchNorm1d(in_ch, dtype=dtype),
                    activation_fn(),
                    nn.Conv1d(in_ch, in_ch, kernel_size=3, groups=in_ch, padding=1, dtype=dtype, bias=False),
                    nn.BatchNorm1d(in_ch, dtype=dtype)
                )
            ))
            self.spatial_dimensions.append(self.spatial_dimensions[-1])

        # Exit flow
        exit_flow = nn.Sequential()
        # The exit_channels config gives us (mid_ch, out_ch) but we only need out_ch
        # We will use the standard separable conv structure instead.
        for i, (_, out_ch) in enumerate(exit_channels):
            # THIS BLOCK IS NOW FULLY CORRECTED
            exit_flow.add_module(f'exit_flow_{i}', ResZeroBlock(
                skip_part=nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=2, dtype=dtype, bias=False),
                model_part=nn.Sequential(
                    activation_fn(),
                    # Correct Separable Convolution:
                    # 1. Depthwise conv (in_ch -> in_ch)
                    nn.Conv1d(in_ch, in_ch, kernel_size=3, groups=in_ch, padding=1, dtype=dtype, bias=False),
                    # 2. Pointwise conv (in_ch -> out_ch)
                    nn.Conv1d(in_ch, out_ch, kernel_size=1, dtype=dtype, bias=False),
                    nn.BatchNorm1d(out_ch, dtype=dtype),
                    nn.MaxPool1d(3, stride=2, padding=1)
                )
            ))
            self.spatial_dimensions.append(math.floor((self.spatial_dimensions[-1] - 1) / 2 + 1))
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
# 2. DATA HANDLING (No changes)
# =============================================================================
class ChallengeDataset(Dataset):
    def __init__(self, spectra, concentrations=None, is_train=True, mean=None, std=None):
        self.spectra = torch.tensor(spectra, dtype=torch.float32)
        self.is_train = is_train
        if self.is_train:
            self.concentrations = torch.tensor(concentrations, dtype=torch.float32)
        if mean is None or std is None:
            self.mean = self.spectra.mean()
            self.std = self.spectra.std()
        else:
            self.mean = mean
            self.std = std
        self.spectra = (self.spectra - self.mean) / self.std
    def __len__(self):
        return len(self.spectra)
    def __getitem__(self, idx):
        spectrum = self.spectra[idx].unsqueeze(0)
        if self.is_train:
            return {'spectrum': spectrum, 'concentration': self.concentrations[idx]}
        else:
            return {'spectrum': spectrum}

def load_and_prepare_data(train_path, test_path):
    train_df_raw = pd.read_csv(train_path, dtype=str)
    train_df = train_df_raw.iloc[:96].copy()
    spectral_cols = train_df.columns[1:-4]
    for col in spectral_cols:
        train_df[col] = train_df[col].str.replace('[', '', regex=False).str.replace(']', '', regex=False)
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    X_train_full = train_df[spectral_cols].values
    target_cols = ['Glucose (g/L)', 'Sodium Acetate (g/L)', 'Magnesium Acetate (g/L)']
    y_train_full = train_df[target_cols].astype(float).values
    X_train = X_train_full.reshape(-1, 2048)
    y_train = np.repeat(y_train_full, 2, axis=0)
    test_df = pd.read_csv(test_path, header=None, dtype=str)
    spectral_cols_test = test_df.columns[1:]
    for col in spectral_cols_test:
        test_df[col] = test_df[col].str.replace('[', '', regex=False).str.replace(']', '', regex=False)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    X_test = test_df[spectral_cols_test].values.reshape(-1, 2048)
    return X_train, y_train, X_test

# =============================================================================
# 3. TRAINING AND PREDICTION LOGIC (No changes)
# =============================================================================
def create_training_objects(config):
    config['initial_channels'] = [config['initial_channels'], 2 * config['initial_channels']]
    entry_channels = [config['entry_channels_start']]
    for _ in range(config['entry_length']):
        entry_channels.append(int(config['entry_factor'] * entry_channels[-1]))
    config['entry_channels'] = entry_channels
    exit_channels_start = entry_channels[-1]
    exit_channels = []
    for _ in range(config['exit_length']):
        mid_ch = int(exit_channels_start * math.sqrt(config['exit_factor']))
        out_ch = int(exit_channels_start * config['exit_factor'])
        exit_channels.append((mid_ch, out_ch))
        exit_channels_start = out_ch
    config['exit_channels'] = exit_channels
    config['fc_dims'] = [config['fc_dims']]
    model = RamanXception(**config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ExponentialLR(optimizer, gamma=config['gamma'])
    return model, optimizer, scheduler

def train_model(model, dataloader, optimizer, scheduler, loss_fn, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            spectra = batch['spectrum'].to(device)
            targets = batch['concentration'].to(device)
            optimizer.zero_grad()
            predictions = model(spectra)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

def predict(model, dataloader, device):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            spectra = batch['spectrum'].to(device)
            predictions = model(spectra)
            all_predictions.append(predictions.cpu().numpy())
    predictions_array = np.concatenate(all_predictions, axis=0)
    avg_predictions = (predictions_array[0::2] + predictions_array[1::2]) / 2.0
    return avg_predictions

# =============================================================================
# 4. MAIN EXECUTION SCRIPT (No changes)
# =============================================================================
def main():
    TRAIN_FILE = '/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/transfer_plate.csv'
    TEST_FILE = '/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/96_samples.csv'
    SUBMISSION_FILE = 'submission.csv'
    SAMPLE_SUBMISSION_FILE = '/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/sample_submission.csv'
    NUM_EPOCHS = 2500
    model_config = {
        'initial_channels': 8, 'entry_channels_start': 17, 'channel_factor': 1.5692504144354933,
        'entry_exit_length': 3, 'num_mid_blocks': 4, 'fc_dims': 101, 'fc_dropout': 0.11748964300948816,
        'learning_rate': 0.001, 'gamma': 0.9921697445978254, 'batch_size': 8, 'activation_function': 'ELU',
        'entry_factor': 1.5692504144354933, 'exit_factor': 1.5692504144354933, 'entry_length': 3,
        'exit_length': 3, 'spectra_size': 2048, 'num_concentrations': 3, 'dtype': torch.float32
    }
    print("Loading and preparing data...")
    X_train, y_train, X_test = load_and_prepare_data(TRAIN_FILE, TEST_FILE)
    train_dataset = ChallengeDataset(X_train, y_train, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True)
    test_dataset = ChallengeDataset(X_test, is_train=False, mean=train_dataset.mean, std=train_dataset.std)
    test_loader = DataLoader(test_dataset, batch_size=model_config['batch_size'] * 2, shuffle=False)
    print("Setting up model and training objects...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, optimizer, scheduler = create_training_objects(model_config)
    model.to(device)
    loss_fn = nn.MSELoss()
    print("Starting model training...")
    train_model(model, train_loader, optimizer, scheduler, loss_fn, device, num_epochs=NUM_EPOCHS)
    print("Generating predictions on the test set...")
    final_predictions = predict(model, test_loader, device)
    final_predictions[final_predictions < 0] = 0
    print("Creating submission file...")
    submission_df = pd.read_csv(SAMPLE_SUBMISSION_FILE)
    submission_df['Glucose'] = final_predictions[:, 0]
    submission_df['Sodium Acetate'] = final_predictions[:, 1]
    submission_df['Magnesium Sulfate'] = final_predictions[:, 2]
    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission file saved successfully to '{SUBMISSION_FILE}'!")
    print("Top 5 rows of the submission file:")
    print(submission_df.head())

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main()


