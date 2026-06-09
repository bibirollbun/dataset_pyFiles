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
import warnings
warnings.simplefilter('ignore')


torch.manual_seed(28)
torch.cuda.manual_seed(28)
np.random.seed(28)
random.seed(28)


def add_new_features(df):
    df = df.copy()
    
    df["bid_ask_diff"] = df["bid_qty"] - df["ask_qty"]
    df["buy_sell_ratio"] = df["buy_qty"] / (df["sell_qty"] + 1e-6)
    df["bid_ask_ratio"] = df["bid_qty"] / (df["ask_qty"] + 1e-6)
    df["buy_volume_ratio"] = df["buy_qty"] / (df["volume"] + 1e-6)
    df["sell_volume_ratio"] = df["sell_qty"] / (df["volume"] + 1e-6)

    df["sell_buy_ratio"] = df["sell_qty"] / (df["buy_qty"] + df["sell_qty"] + 1e-9)
    df["buy_sell_diff"] = df["buy_qty"] - df["sell_qty"]
    df["buy_sell_sum"] = df["buy_qty"] + df["sell_qty"]
    df["ask_bid_ratio"] = df["ask_qty"] / (df["bid_qty"] + df["ask_qty"] + 1e-9)
    df["bid_ask_sum"] = df["bid_qty"] + df["ask_qty"]

    df["order_pressure"] = (df["buy_qty"] - df["sell_qty"]) / (df["buy_qty"] + df["sell_qty"] + 1e-6)
    df["quoted_pressure"] = (df["bid_qty"] - df["ask_qty"]) / (df["bid_qty"] + df["ask_qty"] + 1e-6)
    df["execution_ratio"] = (df["buy_qty"] + df["sell_qty"]) / (df["bid_qty"] + df["ask_qty"] + 1e-6)
    df["volume_imbalance"] = (df["buy_qty"] - df["sell_qty"]) / (df["volume"] + 1e-6)
    df["order_book_total"] = df["bid_qty"] + df["ask_qty"]
    df["execution_total"] = df["buy_qty"] + df["sell_qty"]
    df["execution_share"] = df["execution_total"] / (df["order_book_total"] + 1e-6)

    df["log_bid_qty"] = np.log1p(df["bid_qty"])
    df["log_ask_qty"] = np.log1p(df["ask_qty"])
    df["log_buy_qty"] = np.log1p(df["buy_qty"])
    df["log_sell_qty"] = np.log1p(df["sell_qty"])
    df["log_volume"] = np.log1p(df["volume"])

    df["norm_bid_ask_diff"] = (df["bid_qty"] - df["ask_qty"]) / (df["bid_qty"] + df["ask_qty"] + 1e-6)
    df["norm_buy_sell_diff"] = (df["buy_qty"] - df["sell_qty"]) / (df["buy_qty"] + df["sell_qty"] + 1e-6)

    window = 5
    df["buy_qty_ma"] = df["buy_qty"].rolling(window).mean()
    df["sell_qty_ma"] = df["sell_qty"].rolling(window).mean()
    df["momentum_buy_qty"] = df["buy_qty"] - df["buy_qty"].shift(window)
    df["ema_buy_qty"] = df["buy_qty"].ewm(span=window).mean()

    df["roc_buy_qty"] = df["buy_qty"].pct_change(periods=5)
    df["roc_sell_qty"] = df["sell_qty"].pct_change(periods=5)

    df["execution_dominance"] = df["buy_qty"] / (df["sell_qty"] + 1e-6)
    df["liquidity_ratio"] = (df["buy_qty"] + df["sell_qty"]) / (df["bid_qty"] + df["ask_qty"] + 1e-6)

    
    return df


train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

train = add_new_features(train)
test = add_new_features(test)

features = [col for col in train.columns if col.startswith('X_')] + ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
label_col = 'label'

additional_features = [
    "bid_ask_diff", "buy_sell_ratio", "bid_ask_ratio", "buy_volume_ratio", "sell_volume_ratio",
    "sell_buy_ratio", "buy_sell_diff", "buy_sell_sum", "ask_bid_ratio", "bid_ask_sum",
    "order_pressure", "quoted_pressure", "execution_ratio", "volume_imbalance",
    "order_book_total", "execution_total", "execution_share",

    "log_bid_qty", "log_ask_qty", "log_buy_qty", "log_sell_qty", "log_volume",
    "norm_bid_ask_diff", "norm_buy_sell_diff",

    "buy_qty_ma", "sell_qty_ma", "momentum_buy_qty", "ema_buy_qty",
    "roc_buy_qty", "roc_sell_qty",

    "execution_dominance", "liquidity_ratio"
]

features += additional_features

train[features] = train[features].replace([np.inf, -np.inf], np.nan)
test[features] = test[features].replace([np.inf, -np.inf], np.nan)

train[features] = train[features].fillna(0)
test[features] = test[features].fillna(0)

scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])


SEQ_LEN = 30  
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

train_df, val_df = train_test_split(train, test_size=0.1, shuffle=False)

train_dataset = CryptoDataset(train_df, SEQ_LEN, is_train=True)
val_dataset = CryptoDataset(val_df, SEQ_LEN, is_train=True)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)


class GLNRegressor(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size=128):
        super(GLNRegressor, self).__init__()
        self.seq_len = seq_len
        self.input_size = input_size

        self.gate = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid()
        )
        self.linear = nn.Linear(input_size, hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        gate = self.gate(x)                
        lin = self.linear(x)               
        gated_out = gate * lin             
        pooled = gated_out.mean(dim=1)     
        out = self.fc(pooled)              
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
model = GLNRegressor(input_size=len(features), seq_len=SEQ_LEN).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

trained_model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=50, patience=5)


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

