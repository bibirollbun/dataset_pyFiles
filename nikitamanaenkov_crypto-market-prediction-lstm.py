import random
import ta
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


def finance_features_chunk(df, cols_chunk):
    features = {}
    for col in cols_chunk:
        features[f'{col}_SMA_10'] = df[col].rolling(window=10).mean()
        features[f'{col}_EMA_10'] = df[col].ewm(span=10, adjust=False).mean()

        sma_10 = features[f'{col}_SMA_10']
        envelope_upper = sma_10 * 1.10
        envelope_lower = sma_10 * 0.90
        envelope_bin = np.where(df[col] > envelope_upper, 1, np.where(df[col] < envelope_lower, -1, 0))
        features[f'{col}_Envelope_bin'] = envelope_bin

        rsi = ta.momentum.RSIIndicator(close=df[col], window=14).rsi()
        features[f'{col}_RSI'] = rsi

        macd = ta.trend.MACD(close=df[col], window_slow=26, window_fast=12, window_sign=9)
        features[f'{col}_MACD'] = macd.macd()
        features[f'{col}_MACD_signal'] = macd.macd_signal()
        features[f'{col}_MACD_diff'] = macd.macd_diff()

    return pd.DataFrame(features, index=df.index).astype('float32')

def finance_features_allX_batch(df, batch_size=1):
    base_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')

    X_cols = [col for col in df.columns if col.startswith('X')]
    result = df[base_cols].copy()  

    for i in range(0, len(X_cols), batch_size):
        chunk = X_cols[i:i+batch_size]
        print(f"Processing columns {i} to {i+len(chunk)-1}")
        features_chunk = finance_features_chunk(df, chunk)
        result = pd.concat([result, features_chunk], axis=1)
        del features_chunk
        gc.collect()
    return result


train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

train_feat = finance_features_allX_batch(train)
test_feat = finance_features_allX_batch(test)



train_feat.dropna(inplace=True)
test_feat.dropna(inplace=True)

features = [col for col in train_feat.columns if col.startswith('X_')] + ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']

scaler = StandardScaler()
train_feat[features] = scaler.fit_transform(train_feat[features])
test_feat[features] = scaler.transform(test_feat[features])


SEQ_LEN = 60  
TARGET_COL = 'label'

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

train_df, val_df = train_test_split(train_feat, test_size=0.1, shuffle=False)

train_dataset = CryptoDataset(train_df, SEQ_LEN, is_train=True)
val_dataset = CryptoDataset(val_df, SEQ_LEN, is_train=True)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  
        out = self.fc(out)
        return out.squeeze()


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
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

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
model = LSTMRegressor(input_size=len(features)).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

trained_model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=50, patience=3)


test_dataset = CryptoDataset(test_feat, SEQ_LEN, is_train=False)
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

