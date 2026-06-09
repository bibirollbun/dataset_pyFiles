!nvidia-smi


import gc
import os
import random
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class CFG:
    seed = 42
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = '/kaggle/input/hms-harmful-brain-activity-classification'
    eeg_path = f'{path}/train_eegs/'
    spec_path = f'{path}/train_spectrograms/'
    eeg_length = 10000 
    batch_size = 32
    num_classes = 6
    learning_rate = 1e-4


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG.seed)


def read_eeg_file(eeg_id):
    eeg_df = pd.read_parquet(f'{CFG.eeg_path}/{eeg_id}.parquet')
    eeg_signal = eeg_df.values.T.flatten()
    if np.any(np.isnan(eeg_signal)) or np.any(np.isinf(eeg_signal)):
        eeg_signal = np.nan_to_num(eeg_signal, nan=0.0, posinf=0.0, neginf=0.0)

    mean_eeg = np.mean(eeg_signal)
    std_eeg = np.std(eeg_signal)

    if std_eeg < 1e-6:
        std_eeg = 1.0 

    downsampled_eeg = eeg_signal[::len(eeg_signal) // CFG.eeg_length][:CFG.eeg_length]
    downsampled_eeg = (downsampled_eeg - mean_eeg) / (std_eeg + 1e-6)

    if np.any(np.isnan(downsampled_eeg)) or np.any(np.isinf(downsampled_eeg)):
        downsampled_eeg = np.nan_to_num(downsampled_eeg, nan=0.0, posinf=0.0, neginf=0.0)

    return downsampled_eeg


def get_eeg_spectrogram(eeg_id):
    eeg_df = pd.read_parquet(f'{CFG.eeg_path}/{eeg_id}.parquet')
    eeg_values = eeg_df.values
    if np.any(np.isnan(eeg_values)) or np.any(np.isinf(eeg_values)):
        eeg_values = np.nan_to_num(eeg_values, nan=0.0, posinf=0.0, neginf=0.0)
    f, t, Sxx = spectrogram(
        x=eeg_values,
        fs=200,
        nperseg=20,
        noverlap=10,
        nfft=256
    )
    if np.any(Sxx < 0):
        Sxx = np.maximum(Sxx, 1e-7)

    Sxx = np.log(Sxx + 1e-6)
    mean_Sxx = np.mean(Sxx)
    std_Sxx = np.std(Sxx)

    if std_Sxx < 1e-6: 
        std_Sxx = 1.0

    Sxx = (Sxx - mean_Sxx) / (std_Sxx + 1e-6)

    if np.any(np.isnan(Sxx)) or np.any(np.isinf(Sxx)):
        Sxx = np.nan_to_num(Sxx, nan=0.0, posinf=0.0, neginf=0.0)

    return Sxx[:100, :100]


def get_2d_eeg(eeg_id):
    eeg_df = pd.read_parquet(f'{CFG.eeg_path}/{eeg_id}.parquet')
    eeg_2d = eeg_df.values.T
    if np.any(np.isnan(eeg_2d)) or np.any(np.isinf(eeg_2d)):
        eeg_2d = np.nan_to_num(eeg_2d, nan=0.0, posinf=0.0, neginf=0.0)
    return eeg_2d[:, :1000]


class EEGDataset(Dataset):
    def __init__(self, df, mode='train'):
        self.df = df
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        eeg_id = row['eeg_id']

        eeg_1d = read_eeg_file(eeg_id)
        eeg_1d = torch.tensor(eeg_1d, dtype=torch.float32)

        eeg_2d = get_2d_eeg(eeg_id)
        eeg_2d = torch.tensor(eeg_2d, dtype=torch.float32).unsqueeze(0)

        labels = row[['seizure_vote', 'gpd_vote', 'lpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].values.astype(np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return eeg_1d, eeg_2d, labels


class EEGNet_1D(nn.Module):
    def __init__(self):
        super(EEGNet_1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.fc = nn.Linear(64 * (CFG.eeg_length // 4), 128)
        self.output = nn.Linear(128, CFG.num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.output(x)
        return x


class EEGNet_2D(nn.Module):
    def __init__(self):
        super(EEGNet_2D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.AdaptiveAvgPool2d((8, 250)) 
        )
        self.fc = nn.Linear(64 * 8 * 250, 128)
        self.output = nn.Linear(128, CFG.num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.output(x)
        return x


class MultimodalNet(nn.Module):
    def __init__(self):
        super(MultimodalNet, self).__init__()
        self.eeg_1d_model = EEGNet_1D()
        self.eeg_2d_model = EEGNet_2D()
        self.fusion = nn.Linear(CFG.num_classes * 2, CFG.num_classes)

    def forward(self, x_1d, x_2d):
        output_1d = self.eeg_1d_model(x_1d)
        output_2d = self.eeg_2d_model(x_2d)
        combined_output = torch.cat([output_1d, output_2d], dim=1)
        final_output = self.fusion(combined_output)
        return final_output


def train_model(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for eeg_1d, eeg_2d, labels in tqdm(train_loader, desc="Training"):
        eeg_1d, eeg_2d, labels = eeg_1d.to(CFG.device), eeg_2d.to(CFG.device), labels.to(CFG.device)

        optimizer.zero_grad()
        outputs = model(eeg_1d, eeg_2d)
        loss = criterion(outputs, labels)
        if torch.isnan(loss):
            return float('nan')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)


def validate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for eeg_1d, eeg_2d, labels in tqdm(val_loader, desc="Validation"):
            eeg_1d, eeg_2d, labels = eeg_1d.to(CFG.device), eeg_2d.to(CFG.device), labels.to(CFG.device)
            outputs = model(eeg_1d, eeg_2d)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
    return total_loss / len(val_loader)


def predict_on_test_data(model):
    test_df = pd.read_csv(f'{CFG.path}/test.csv')
    test_dataset = EEGDataset(test_df, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for eeg_1d, eeg_2d in test_loader:
            eeg_1d, eeg_2d = eeg_1d.to(CFG.device), eeg_2d.to(CFG.device)
            outputs = model(eeg_1d, eeg_2d)
            all_preds.append(outputs.cpu().numpy())
    
    return np.concatenate(all_preds)


def train_model(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_idx, (eeg_1d, eeg_2d, labels) in enumerate(tqdm(train_loader, desc="Training")):
        eeg_1d, eeg_2d, labels = eeg_1d.to(CFG.device), eeg_2d.to(CFG.device), labels.to(CFG.device)
        optimizer.zero_grad()
        outputs = model(eeg_1d, eeg_2d)
        loss = criterion(outputs, labels)

        if torch.isnan(loss):
            return float('nan') 

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def plot_loss_curve(train_losses, val_losses, title):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


train_df = pd.read_csv(f'{CFG.path}/train.csv')
train_split_df, val_split_df = train_test_split(train_df, test_size=0.2, random_state=CFG.seed)


train_dataset = EEGDataset(train_split_df, mode='train')
train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=4)


val_dataset = EEGDataset(val_split_df, mode='val') 
val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=4)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultimodalNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.learning_rate)
criterion = nn.BCEWithLogitsLoss()


best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(CFG.epochs):
    train_loss = train_model(model, train_loader, optimizer, criterion)
    val_loss = validate_model(model, val_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f'  Epoch {epoch + 1}/{CFG.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    if val_loss < best_val_loss:
        print(f'Saved best model!')
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')

print('Training completed!')


plot_loss_curve(train_losses, val_losses, 'Model Training and Validation Loss')


test_predictions = predict_on_test_data(model)
sample_submission_df = pd.read_csv(f'{CFG.path}/sample_submission.csv')

submission_df = pd.DataFrame(test_predictions, columns=sample_submission_df.columns[1:])
submission_df.insert(0, 'eeg_id', sample_submission_df['eeg_id'])

submission_df.to_csv('submission.csv', index=False)
print('submission.csv created successfully!')


del model, optimizer, full_train_dataset, full_train_loader
gc.collect()
torch.cuda.empty_cache()

