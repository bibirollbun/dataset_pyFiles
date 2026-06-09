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
from scipy.stats import pearsonr

import copy  # To store best model weights

# Seeds
torch.manual_seed(28)
np.random.seed(28)
random.seed(28)



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class FeatureWiseDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.features = df[features].values
        self.labels = df[label_col].values if is_train else None
        self.is_train = is_train

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1) 
        if self.is_train:
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
            return x, y
        return x


class FeatureAttentionModel(nn.Module):
    def __init__(self, num_features, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)

        encoded = self.transformer_encoder(x)
        encoded = self.norm(encoded)
        pooled = encoded[:, 0]
        return self.output_head(pooled).squeeze(-1) 




def plot_predictions(preds, trues, epoch):
    plt.figure(figsize=(10, 4))
    plt.plot(trues[:200], label='True', alpha=0.7)
    plt.plot(preds[:200], label='Pred', alpha=0.7)
    plt.title(f'Predictions vs Ground Truth (Epoch {epoch})')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

class NegativePearsonLoss(nn.Module):
    def forward(self, preds, targets):
        preds = preds - preds.mean(dim=-1, keepdim=True)
        targets = targets - targets.mean(dim=-1, keepdim=True)

        numerator = (preds * targets).sum(dim=-1)
        denominator = torch.norm(preds, dim=-1) * torch.norm(targets, dim=-1)
        correlation = numerator / (denominator + 1e-8)

        return 1 - correlation.mean()

# --- Evaluation ---
def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    losses, preds, trues = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X).squeeze(-1)
            loss = criterion(out, y)
            losses.append(loss.item())
            preds.extend(out.cpu().numpy())
            trues.extend(y.cpu().numpy())
    avg_loss = np.mean(losses)

    preds = np.array(preds)
    trues = np.array(trues)
    if np.std(preds) < 1e-6 or np.std(trues) < 1e-6:
        corr = 0.0
    else:
        corr = pearsonr(preds, trues)[0]

    if return_preds:
        return avg_loss, corr, preds, trues
    return avg_loss, corr


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cuda', save_path='best_model.pt'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    criterion = NegativePearsonLoss()

    best_corr = -float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Training at epoch {epoch}", position=0, leave=True)
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X).squeeze(-1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        val_loss, val_corr, val_preds, val_trues = evaluate(model, val_loader, criterion, device, return_preds=True)
        plot_predictions(val_preds, val_trues, epoch)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {running_loss/len(train_loader):.4f} "
              f"| Val Loss: {val_loss:.4f} | Val Corr: {val_corr:.4f}")

        # Save best model
        if val_corr > best_corr:
            best_corr = val_corr
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)
            print(f"  ✅ New best model saved (val_corr: {val_corr:.4f})")

    # Load best weights before returning
    model.load_state_dict(best_model_wts)
    return model



train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

train = train.replace([np.inf, -np.inf], np.nan)
nan_cols = train.columns[train.isna().any()].tolist() # get list of nan columns

features = [col for col in train.columns if col.startswith('X_')] + ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
label_col = 'label'

scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])

train = train.drop(nan_cols, axis=1)
test = test.drop(nan_cols, axis=1)

print(train.shape)
print(test.shape)


# Sort chronologically, even if it should already
train = train.sort_values('timestamp').reset_index(drop=True)

# Keep last X% as validation
train_frac = 0.9
split_idx = int(len(train) * train_frac)
train_df = train.iloc[:split_idx].copy()
val_df = train.iloc[split_idx:].copy()

train_dataset = FeatureWiseDataset(train_df, is_train=True)
val_dataset = FeatureWiseDataset(val_df, is_train=True)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FeatureAttentionModel(num_features=len(features))
trained_model = train_model(model, train_loader, val_loader, epochs=3, device=device, save_path='best_model.pt')


test_dataset = FeatureWiseDataset(test, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

trained_model.eval()
test_preds = []

with torch.no_grad():
    for x_batch in tqdm(test_loader, desc="Predicting"):
        x_batch = x_batch.to(device)
        outputs = trained_model(x_batch)
        test_preds.extend(outputs.cpu().numpy())

sample_submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
sample_submission['prediction'] = test_preds  # pad the beginning to align sequence
sample_submission.to_csv('submission.csv', index=False)
print(sample_submission.tail())

