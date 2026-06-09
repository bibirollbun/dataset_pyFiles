import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

def preprocess_dataframe(df, is_test=False, train_columns=None):
    df.loc[df["alchemy_category"] == "?", "alchemy_category"] = np.nan
    df = df.drop(columns=['url', 'boilerplate'], errors='ignore')
    df['alchemy_category'] = df['alchemy_category'].fillna('unknown')
    df.replace('?', np.nan, inplace=True)
    df.fillna(0, inplace=True)

    df = pd.get_dummies(df, columns=['alchemy_category'])
    if not is_test:
        labels = df['label'].values.astype(np.float32).reshape(-1, 1)
        df = df.drop(columns=['label'])
    else:
        labels = np.zeros((len(df), 1), dtype=np.float32)

    if is_test and train_columns is not None:
        for col in train_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[train_columns]  # 保证列顺序一致

    features = df.values.astype(np.float32)
    return features, labels, df.columns.tolist()


class WebpageDataset(Dataset):
    def __init__(self, features, labels, urlids=None):
        mu = features.mean(axis=0, keepdims=True)
        sigma = features.std(axis=0, keepdims=True) + 1e-8
        features = (features - mu) / sigma

        self.X = torch.from_numpy(features)
        self.y = torch.from_numpy(labels)
        self.urlids = urlids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            pred_labels = (preds >= 0.5).float()
            correct += (pred_labels == y_batch).sum().item()
    return correct / len(loader.dataset)


def predict_test(model, test_dataset, device):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=64)
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            pred_labels = (preds >= 0.5).int().cpu().numpy()
            all_preds.extend(pred_labels)
    return all_preds


train_path = '/kaggle/input/stumbleupon/train.tsv'
test_path = '/kaggle/input/stumbleupon/test.tsv'
batch_size = 256
lr = 0.01
epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_df_raw = pd.read_csv(train_path, sep='\t')
train_features, train_labels, train_columns = preprocess_dataframe(train_df_raw, is_test=False)

full_dataset = WebpageDataset(train_features, train_labels, urlids=train_df_raw['urlid'].values)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

input_dim = train_features.shape[1]
model = LogisticRegressionModel(input_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

best_acc = 0.0
best_model_path = 'best_model.pth'
for epoch in range(1, epochs + 1):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_acc = evaluate(model, val_loader, device)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f'[Epoch {epoch:02d}] ✅ New best model saved (Val Acc = {val_acc:.4f})')
    else:
        print(f'[Epoch {epoch:02d}] Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}')

print('\n⏳ 加载最佳模型，开始推理 test.tsv...')
model.load_state_dict(torch.load(best_model_path))
test_df_raw = pd.read_csv(test_path, sep='\t')
test_features, test_labels, _ = preprocess_dataframe(test_df_raw, is_test=True, train_columns=train_columns)
test_dataset = WebpageDataset(test_features, test_labels, urlids=test_df_raw['urlid'].values)
pred_y = predict_test(model, test_dataset, device)


df_result = pd.DataFrame({
    'urlid': test_dataset.urlids,
    'label': np.array(pred_y).flatten()
})
df_result.to_csv("submission.csv", index=False)
df_result

