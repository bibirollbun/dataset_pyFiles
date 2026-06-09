import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dask.dataframe as dd
import numpy as np


import warnings
warnings.filterwarnings("ignore")


columns_to_read = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'X21', 'X20', 'X28', 'X863', 'X29', 'X19', 'X27', 'X22', 'X858', 'X219', 'X860', 'X531', 'X287', 'X289', 'X291', 'X293', 'X857', 'X295', 'X598', 'X218',
                    'X297', 'X298', 'X285', 'X300', 'X299', 'X302', 'X26', 'X292', 'X301', 'X294', 'X296', 'X303', 'X283', 'X30', 'X18', 'X465', 'X466', 'X181', 'X288', 'X290', 'label']


df = pd.read_parquet('../input/drw-crypto-market-prediction/train.parquet', columns=columns_to_read)


# df = pd.read_parquet('/content/drive/MyDrive/code/kaggle/DRW_Crypto_Market_Prediction/train.parquet')


# corr_matrix = df.corr()
# target_corr = corr_matrix['label'].drop('label')
# top_features = target_corr.abs().sort_values(ascending=False).head(40)
# selected_columns = top_features.index.tolist()

# df = df[selected_columns + ['label']]


df.shape


df.head()


df.index


df.info()


df.describe()


df.isna().sum()


plt.figure(figsize=(12, 5))
sns.boxplot(data=df[['volume', 'buy_qty', 'sell_qty']])
plt.title("Boxplot of key features for outliers detection")
plt.yscale('log')
plt.show()


features = ['volume', 'buy_qty', 'sell_qty']
plt.figure(figsize=(18, 5))

for i, col in enumerate(features):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df[col], bins=500, kde=True)
    plt.xscale('log')
    plt.title(f'{col} (log scale)')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(2, 1, figsize=(12, 6))
ax = ax.flatten()

sns.boxplot(x=df['label'], ax=ax[0])
sns.histplot(data=df, x='label', kde=True, ax=ax[1])

plt.tight_layout()


sns.boxplot(data=df['label'])
plt.title("Boxplot of target feature for outliers detection")
plt.show()


sns.histplot(df['label'], bins=500, kde=True)
plt.xscale('log')
plt.title('Target feature (log scale)')
plt.xlabel('label')
plt.ylabel('Frequency')
plt.grid(True)


plt.figure(figsize=(20, 4))
plt.plot(df.index, df['label'])
plt.xlabel('Time')
plt.ylabel('Target value')
plt.title('Target value over time')
plt.grid(True)
plt.show()


plt.figure(figsize=(20, 4))
plt.plot(df.index, np.cumsum(df['label']))
plt.xlabel('Time')
plt.ylabel('Target value - cummulative sum')
plt.title('Target value over time')
plt.grid(True)
plt.show()


df[['volume', 'buy_qty', 'sell_qty']].plot(figsize=(20, 6))
plt.title("Crypto market over time")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.show()


df['day_name'] = df.index.day_name()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


avg_label_per_day = df.groupby('day_name')['label'].mean().reindex(weekday_order)

plt.figure(figsize=(10, 5))
sns.lineplot(x=weekday_order, y=avg_label_per_day.values, marker='o')
plt.title('Average target (label) by day of week')
plt.xlabel('Day of week')
plt.ylabel('Average label')
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(18, 15))
heatmap = sns.heatmap(df.corr(numeric_only=True), vmin=-1, vmax=1, annot=False, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12})

plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
      pass

    def fit(self, data, y = None):
      return self

    def transform(self, data):
        # Add lag features for label
        data = data.reset_index()
        try:
            data = data.sort_values("timestamp").reset_index(drop=True)
        except:
            data = data.sort_values("ID").reset_index(drop=True)

        
        for lag in range(1, 6):
            data[f"buy_lag_{lag}"] = data["buy_qty"].shift(lag)
            data[f"sell_lag_{lag}"] = data["sell_qty"].shift(lag)
            data[f"volume_lag_{lag}"] = data["volume"].shift(lag)

        # Add rolling means for lag features
        for lag_col in ['buy_lag_1', 'buy_lag_2', 'buy_lag_3', 'buy_lag_4', 'buy_lag_5',
                       'sell_lag_1', 'sell_lag_2', 'sell_lag_3', 'sell_lag_4', 'sell_lag_5',
                       'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_4', 'volume_lag_5']:
            data[f'{lag_col}_roll_mean'] = data[lag_col].rolling(window=3).mean()

        # Drop null values
        data.dropna(inplace=True)

        return data


# df['dayofweek'] = df.index.dayofweek
# df['day'] = df.index.day
# df['month'] = df.index.month

# df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
# df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)


# df = df.reset_index()
# df = df.sort_values("timestamp").reset_index(drop=True)

# for lag in range(1, 6):
#     df[f"label_lag_{lag}"] = df["label"].shift(lag)


 # for lag_col in ['label_lag_1', 'label_lag_2', 'label_lag_3', 'label_lag_4', 'label_lag_5']:
 #        df[f'{lag_col}_roll_mean'] = df[lag_col].rolling(window=3).mean()


# df.dropna(inplace=True)


# df.columns


class Outliers(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, data, y=None):
        return self

    def transform(self, data):
        try:
            features = data.drop(columns=['label', 'day_name', 'timestamp'])
            target = data['label']

            features_clipped = features.clip(lower=features.quantile(0.01), upper=features.quantile(0.99), axis=1)
        
            data = pd.concat([features_clipped, target], axis=1)

        except:
            features = data.drop(columns=['ID'])
            data = features.clip(lower=features.quantile(0.01), upper=features.quantile(0.99), axis=1)

        return data
        


# features = df.drop(columns=['label', 'day_name', 'hour', 'dayofweek', 'timestamp'])
# target = df['label']

# features_clipped = features.clip(lower=features.quantile(0.01), upper=features.quantile(0.99), axis=1)

# df = pd.concat([features_clipped, target], axis=1)

# print(f"Min value: {df.min().min()}")
# print(f"Max value: {df.max().max()}")


pipeline = Pipeline([
    ('feature_engineering', FeatureEngineer()),
    ('outliers', Outliers())
])


df = pipeline.fit_transform(df)


X = df.drop(columns=['label'])
y = df['label']

print(f"X: {X.shape}")
print(f"Y: {y.shape}")


split_idx = int(len(X) * 0.8)

X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]

X_val = X.iloc[split_idx:]
y_val = y.iloc[split_idx:]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


print(f"Min value: {X_train_scaled.min()}")
print(f"Max value: {X_train_scaled.max()}")


! pip install torchmetrics


from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchmetrics.functional import pearson_corrcoef


class Parameters():
  lr = 0.0001
  batch_size_train = 64
  batch_size_val = 128
  epochs = 30
  input_dim = X.shape[1]
  hidden_dim1 = 128
  hidden_dim2 = 64
  hidden_dim3 = 32
  # hidden_dim4 = 32
  output_dim = 1

params = Parameters()


x_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
x_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)


class CryptoDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


train_dataset = CryptoDataset(x_train_tensor, y_train_tensor)
val_dataset = CryptoDataset(x_val_tensor, y_val_tensor)


train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size_train, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size_val, shuffle=False)


class crypto_MLP(nn.Module):
  def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
    super().__init__()

    self.model = nn.Sequential(
      nn.Linear(input_dim, hidden_dim1),
      nn.BatchNorm1d(hidden_dim1),
      nn.ReLU(),
      nn.Dropout(0.5),

      nn.Linear(hidden_dim1, hidden_dim2),
      nn.BatchNorm1d(hidden_dim2),
      nn.ReLU(),
      nn.Dropout(0.4),

      nn.Linear(hidden_dim2, hidden_dim3),
      nn.BatchNorm1d(hidden_dim3),
      nn.ReLU(),
      nn.Dropout(0.3),

      nn.Linear(hidden_dim3, output_dim)
    )

  def forward(self, x):
    return self.model(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = crypto_MLP(input_dim=params.input_dim, hidden_dim1=params.hidden_dim1, hidden_dim2=params.hidden_dim2, hidden_dim3=params.hidden_dim3, output_dim=params.output_dim).to(device)


loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)


train_losses = []
train_pearsons = []
val_losses = []
val_pearsons = []


for epoch in range(params.epochs):
    model.train()
    running_loss = 0.0
    pearson_total = 0.0
    train_total = 0


    with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}", position=0, leave=True) as progress_bar:
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Forward
            y_pred = model(x).squeeze(1)

            # Calculate loss
            loss = loss_fn(y_pred, y)

            # Backward
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item() * x.size(0)
            pearson = pearson_corrcoef(y_pred, y).item()
            pearson_total += pearson * x.size(0)

            train_total += x.size(0)

            avg_loss = running_loss / train_total
            avg_pearson = pearson_total / train_total

            progress_bar.set_postfix(loss=avg_loss, pearson=avg_pearson)
            progress_bar.update(1)

    running_loss /= train_total
    pearson_total /= train_total

    train_losses.append(running_loss)
    train_pearsons.append(pearson_total)
    print(f"Epoch {epoch+1}: Train Loss = {running_loss:.4f}, Train Pearson correlation coefficient = {pearson_total:.4f}")

    # ---------------- Validation ----------------
    model.eval()
    val_loss = 0.0
    val_pearson_total = 0.0
    val_total = 0

    with torch.no_grad():
        for x, y in val_dataloader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            y_pred = model(x).squeeze(1)

            # Calculate loss
            loss = loss_fn(y_pred, y)
            val_loss += loss.item() * x.size(0)

            # Calculate metrics
            val_pearson = pearson_corrcoef(y_pred, y).item()
            val_pearson_total += pearson * x.size(0)

            val_total += x.size(0)

    val_loss /= val_total
    val_pearson_total /= val_total

    scheduler.step(val_loss)

    val_losses.append(val_loss)
    val_pearsons.append(val_pearson_total)
    print(f"           Val Loss = {val_loss:.4f}, Val Pearson correlation coefficient = {val_pearson_total:.4f}")


# Loss plot
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.show()


# Accuracy plot
plt.plot(train_pearsons, label='Train Pearson correlation coefficient')
plt.plot(val_pearsons, label='Val Pearson correlation coefficient')
plt.xlabel('Epoch')
plt.ylabel('Pearson correlation coefficient')
plt.legend()
plt.title('Pearson correlation coefficient over Epochs')
plt.show()


columns_to_read_test = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'X21', 'X20', 'X28', 'X863', 'X29', 'X19', 'X27', 'X22', 'X858', 'X219', 'X860', 'X531', 'X287', 'X289', 'X291', 'X293', 'X857', 'X295', 'X598', 'X218',
                    'X297', 'X298', 'X285', 'X300', 'X299', 'X302', 'X26', 'X292', 'X301', 'X294', 'X296', 'X303', 'X283', 'X30', 'X18', 'X465', 'X466', 'X181', 'X288', 'X290']


df_test = pd.read_parquet('../input/drw-crypto-market-prediction/test.parquet', columns=columns_to_read_test)


df_test = pipeline.fit_transform(df_test)
scaler = StandardScaler()
df_test_scaled = scaler.fit_transform(df_test)
test_tensor = torch.tensor(df_test_scaled, dtype=torch.float32)


test_dataset = CryptoDataset(test_tensor, y=None)
test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size_val, shuffle=True)


model.eval()
predictions = []

with torch.no_grad():
    for x in test_dataloader:
        x = x.to(device)
        y_pred = model(x)
        predictions.extend(y_pred.cpu().numpy())


submission = pd.DataFrame({
    "ID": list(range(1, len(predictions) + 1)),
    "prediction": predictions
})


submission.head()


submission.to_csv('submission.csv', index=False)

