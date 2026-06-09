import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import pywt
from copy import deepcopy
from scipy.stats import entropy
from scipy.signal import find_peaks
from scipy.stats import kstest
import warnings

warnings.filterwarnings("ignore")


test_df = pd.read_csv("/kaggle/input/geology-forecast-challenge-open/data/test.csv")
train_df = pd.read_csv("/kaggle/input/geology-forecast-challenge-open/data/train.csv")

input_cols = [str(i) for i in range(-299, 1)]
target_cols = [str(i) for i in range(1, 301)]

imputer = SimpleImputer(strategy="mean")
X_train_raw = imputer.fit_transform(train_df[input_cols])
X_test_raw = imputer.transform(test_df[input_cols])


# https://www.kaggle.com/code/nikitamanaenkov/geology-forecast-challenge-features 

def extract_features(X):
    df = pd.DataFrame(X)
    features = pd.DataFrame()
    X_filled = df.T.interpolate(limit_direction='both').T.fillna(0)
    X_np = X_filled.to_numpy()
    
    features["mean"] = df.mean(axis=1)
    features["std"] = df.std(axis=1)
    features["min"] = df.min(axis=1)
    features["max"] = df.max(axis=1)
    
    second_deriv = np.diff(X_np, n=2, axis=1)
    features["second_deriv_mean"] = second_deriv.mean(axis=1)
    features["second_deriv_std"] = second_deriv.std(axis=1)

    first_deriv = np.diff(X_np, axis=1)
    ratio = np.abs(first_deriv[:, 1:] / (first_deriv[:, :-1] + 1e-6))
    features['change_ratio'] = ratio.mean(axis=1)

    features['slope_mean'] = np.gradient(X_filled, axis=1).mean(axis=1)

    features['sign_changes'] = (np.diff(np.sign(X_filled.values), axis=1) != 0).sum(axis=1)
    
    features['zero_frac'] = (X_filled == 0).mean(axis=1)
    features['neg_frac'] = (X_filled < 0).mean(axis=1)
    
    weights = np.tile(np.arange(X_filled.shape[1]), (X_filled.shape[0], 1))
    features['center_of_mass'] = (X_filled.values * weights).sum(axis=1) / (X_filled.sum(axis=1) + 1e-6)
    
    ks_stats = []
    for row in X_np:
        m, s = row.mean(), row.std()
        stat = kstest((row - m) / (s + 1e-6), 'norm').statistic
        ks_stats.append(stat)
    features['ks_gauss'] = ks_stats
    
    features['auc'] = np.trapz(X_np, axis=1)
    
    return features


def wavelet_features(X):
    features = []
    for row in X:
        coeffs = pywt.wavedec(row, 'db1', level=3)
        flat = np.hstack(coeffs)
        features.append([
            np.mean(flat),
            np.std(flat),
            np.max(flat),
            np.min(flat)
        ])
    return np.array(features)


X_train_features = extract_features(X_train_raw)
X_test_features = extract_features(X_test_raw)

wavelet_feats = wavelet_features(X_train_raw)
X_train = np.hstack([X_train_raw, X_train_features, wavelet_feats])

wavelet_feats_test = wavelet_features(X_test_raw)
X_test = np.hstack([X_test_raw, X_test_features, wavelet_feats_test])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = train_df[target_cols].values

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x.unsqueeze(-1)).unsqueeze(-1)
        return x * scale.squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3, reduction=16):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.se = SEBlock(out_dim, reduction=reduction)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.se(out)
        out = self.dropout(out)
        return out + identity


class GeoNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeoNet, self).__init__()

        self.model = nn.Sequential(
            ResidualBlock(input_dim, 1024, dropout=0.3),
            ResidualBlock(1024, 768, dropout=0.3),
            ResidualBlock(768, 512, dropout=0.25),
            ResidualBlock(512, 384, dropout=0.2),
            ResidualBlock(384, 256, dropout=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)


input_dim = dataset[0][0].shape[0]
output_dim = y_train.shape[1]  
model = GeoNet(input_dim, output_dim)

epochs = 2000
n_splits = 5
patience = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_results = []

train_losses = []
train_r2_scores = []
val_losses_all = []
val_r2_all = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    print(f"\nFold {fold+1}/{n_splits}")
    
    train_subsampler = torch.utils.data.Subset(dataset, train_idx)
    val_subsampler = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subsampler, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=64, shuffle=False)

    model = GeoNet(input_dim, output_dim).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=True)
    
    best_model_wts = deepcopy(model.state_dict())
    best_loss = float('inf')
    trigger_times = 0

    fold_train_losses = []
    fold_train_r2_scores = []
    lrs = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_preds.append(outputs.detach().cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        preds_np = np.vstack(all_preds)
        targets_np = np.vstack(all_targets)
        r2 = r2_score(targets_np, preds_np)

        fold_train_losses.append(epoch_loss)
        fold_train_r2_scores.append(r2)
        scheduler.step(epoch_loss)
        lrs.append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, R2: {r2:.4f}")
        
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            best_model_wts = deepcopy(model.state_dict())
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_wts)
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_preds.append(outputs.cpu().numpy())
            val_targets.append(targets.cpu().numpy())

    val_preds_np = np.vstack(val_preds)
    val_targets_np = np.vstack(val_targets)
    val_r2 = r2_score(val_targets_np, val_preds_np)
    val_loss = np.mean(np.abs(val_preds_np - val_targets_np))

    fold_results.append({
        'fold': fold+1,
        'val_loss': val_loss,
        'val_r2': val_r2
    })

    val_losses_all.append(val_loss)
    val_r2_all.append(val_r2)

    print(f"Fold {fold+1} - Val Loss: {val_loss:.4f}, Val R2: {val_r2:.4f}")

print("\nCross-validation results:")
for res in fold_results:
    print(f"Fold {res['fold']} - Val Loss: {res['val_loss']:.4f}, Val R2: {res['val_r2']:.4f}")



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fold_train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Loss over Epochs")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(fold_train_r2_scores, label="Training R² Score", color='orange')
plt.xlabel("Epoch")
plt.ylabel("R² Score")
plt.title("R² Score over Epochs")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(lrs, label="Learning Rate", color='green')
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.title("Learning Rate over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

residuals = val_preds_np - val_targets_np
plt.figure(figsize=(10, 5))
plt.hist(residuals.flatten(), bins=50, color='gray', edgecolor='black')
plt.title("Distribution of Residuals")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.boxplot(val_losses_all)
plt.title("Validation Loss (per Fold)")

plt.subplot(1, 2, 2)
plt.boxplot(val_r2_all)
plt.title("Validation R² (per Fold)")

plt.tight_layout()
plt.show()


model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'GeoNet.pth')

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)  

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).numpy()

all_realizations = [predictions]
for i in range(9):
    noise = np.random.normal(loc=0, scale=0.5, size=predictions.shape)
    all_realizations.append(predictions + noise)

output_df = pd.DataFrame()
output_df["geology_id"] = test_df["geology_id"]

for i in range(1, 301):
    output_df[str(i)] = all_realizations[0][:, i - 1]

for r in range(1, 10):
    for i in range(1, 301):
        col_name = f"r_{r}_pos_{i}"
        output_df[col_name] = all_realizations[r][:, i - 1]

output_df.to_csv("submission.csv", index=False)
print("Submission saved as submission.csv")
output_df.head()

