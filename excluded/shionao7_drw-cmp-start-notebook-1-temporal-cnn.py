import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm


torch.manual_seed(28)
torch.cuda.manual_seed(28)
np.random.seed(28)
random.seed(28)


train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')


features = [col for col in train.columns if col.startswith('X_')] + ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
label_col = 'label'


scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])


SEQ_LEN = 60  

class CryptoDataset(Dataset):
    def __init__(self, df, seq_len, is_train=True):
        self.features = df[features].values
        self.labels = df[label_col].values if is_train else None
        self.seq_len = seq_len
        self.is_train = is_train

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_len]
        if self.is_train:
            y = self.labels[idx + self.seq_len]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        return torch.tensor(x, dtype=torch.float32)

train_df, val_df = train_test_split(train, test_size=0.1, shuffle=False)
train_dataset = CryptoDataset(train_df, SEQ_LEN, is_train=True)
val_dataset = CryptoDataset(val_df, SEQ_LEN, is_train=True)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)


import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 最終回帰モデル
class TCNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_levels=3):
        super().__init__()
        self.tcn = TCN(input_size=input_size, num_channels=[hidden_size] * num_levels, kernel_size=3, dropout=0.2)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, T, F) -> (B, F, T)
        y = self.tcn(x)        # (B, hidden_size, T)
        out = self.linear(y[:, :, -1])  # 最後の時刻の出力を使う
        return out.squeeze(-1)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=10, patience=5):
    class EarlyStopping:
        def __init__(self, patience=patience, delta=1e-4):
            self.patience = patience
            self.counter = 0
            self.best_loss = None
            self.early_stop = False
            self.delta = delta
            self.best_model_state = None

        def __call__(self, val_loss, model):
            if self.best_loss is None or val_loss < self.best_loss - self.delta:
                self.best_loss = val_loss
                self.counter = 0
                self.best_model_state = model.state_dict()
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    def evaluate(model, dataloader, device):
        model.eval()
        losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                losses.append(loss.item())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        avg_loss = np.mean(losses)
        try:
            from scipy.stats import pearsonr
            corr = pearsonr(all_preds, all_targets)[0]
        except:
            corr = 0.0
        return avg_loss, corr

    early_stopping = EarlyStopping()
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_corr": [],
        "lr": []
    }
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x_batch, y_batch in loop:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        train_loss = total_loss / len(train_loader)
        val_loss, val_corr = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_corr"].append(val_corr)
        history["lr"].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val Corr: {val_corr:.4f} | LR: {history['lr'][-1]:.6f}")

        early_stopping(val_loss, model)
        #if early_stopping.early_stop:
        #    print("Early stopping triggered.")
        #    break

    model.load_state_dict(early_stopping.best_model_state)

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(history["train_loss"], label='Train Loss', color='tab:blue', linestyle='-')
    ax1.plot(history["val_loss"], label='Val Loss', color='tab:blue', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  
    ax2.set_ylabel('Learning Rate / Val Corr', color='tab:orange')
    ax2.plot(history["lr"], label='Learning Rate', color='tab:orange', linestyle='-.')
    ax2.plot(history["val_corr"], label='Val Corr', color='tab:green', linestyle=':')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.legend(loc='upper right')

    plt.title('Training Metrics and Learning Rate')
    plt.show()

    return model, history


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TCNRegressor(input_size=len(features)).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

trained_model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=50, patience=3)


test_dataset = CryptoDataset(test, SEQ_LEN, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

model.eval()
preds = []

with torch.no_grad():
    for x_batch in tqdm(test_loader, desc="Predicting"):
        x_batch = x_batch.to(device)
        outputs = model(x_batch)
        preds.extend(outputs.cpu().numpy())

sample_submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
sample_submission['prediction'] = [0]*SEQ_LEN + preds  
sample_submission.to_csv('submission.csv', index=False)
print(sample_submission.tail())

