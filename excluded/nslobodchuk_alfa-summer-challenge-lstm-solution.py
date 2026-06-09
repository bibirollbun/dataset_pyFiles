import pandas as pd
import numpy as np
import gc
import os
from sklearn.model_selection import train_test_split
import copy
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Загрузка данных
df_train = pd.read_parquet("/kaggle/input/alpha-summer-challenge/train.pa")
df_txn   = pd.read_parquet("/kaggle/input/alpha-summer-challenge/df_transaction.pa")

t_min = df_txn.date_time.astype(int).min()
t_max = df_txn.date_time.astype(int).max()


client_aggregates = pd.DataFrame({
    "client_num": np.sort(df_txn.client_num.unique()),
    "mean_amount_by_client_numeric": np.log1p(df_txn.groupby(["client_num"])["amount"].mean().sort_index().values),
    "max_amount_by_client_numeric": np.log1p(df_txn.groupby(["client_num"])["amount"].max().sort_index().values),
    "std_amount_by_client_numeric": np.log1p(df_txn.groupby(["client_num"])["amount"].std().sort_index().values),
    "sum_amount_by_client_numeric": np.log1p(df_txn.groupby(["client_num"])["amount"].sum().sort_index().values),
    "nunique_mcc_code_by_client_numeric": np.log1p(df_txn.groupby(["client_num"])["mcc_code"].nunique().sort_index().values),
    "nunique_merchant_name_by_client_numeric": np.log1p(df_txn.groupby(["client_num"])["merchant_name"].nunique().sort_index().values),
    "min_date_time_by_client_numeric": (df_txn.groupby(["client_num"])["date_time"].min().sort_index().astype(int).values - t_min)/(t_max - t_min),
    "max_date_time_by_client_numeric": (df_txn.groupby(["client_num"])["date_time"].max().sort_index().astype(int).values - t_min)/(t_max - t_min),
})


df = df_txn

df['date_time'] = pd.to_datetime(df['date_time'])

# Map mcc_code
mcc_code_mapping = {v: i+1 for i,v in enumerate(df['mcc_code'].unique())}
df['mcc_code'] = df['mcc_code'].map(mcc_code_mapping)
df['mcc_code'] = df['mcc_code'].astype(int)

# Map merchant_name
print("Mapping merchant_name to integer codes...")
top_999_merchants = df['merchant_name'].value_counts().nlargest(999).index
merchant_mapping = {v: i+1 for i,v in enumerate(top_999_merchants)}
df['merchant_name'] = df['merchant_name'].map(merchant_mapping).fillna(1000).astype(int)

# Add daily column
df['date'] = df['date_time'].dt.floor('D')


agg_amount = df.groupby(['client_num', 'date']).agg(
    amount_sum=('amount', 'sum'),
    amount_count=('amount', 'count'),
    amount_std=('amount', 'std'),
    amount_max=('amount', 'max'),
    amount_min=('amount', 'min'),
    amount_mean=('amount', 'mean'),
    merchant_nunique=('merchant_name', 'nunique'),
    mcc_code_nunique=('mcc_code', 'nunique'),
).reset_index()


def top_n_category_stats(df, category_col, n=5):
    print(f"Computing top-{n} stats for '{category_col}'...")
    
    grouped = df.groupby(['client_num', 'date', category_col])['amount'].agg(['count', 'sum']).reset_index()
    total = df.groupby(['client_num', 'date'])['amount'].sum().rename('total_amount').reset_index()
    grouped = grouped.merge(total, on=['client_num', 'date'])
    
    grouped['ratio'] = grouped['sum'] / grouped['total_amount']
    grouped['rank'] = grouped.groupby(['client_num', 'date'])['count'].rank(method='first', ascending=False)
    
    top_n = grouped[grouped['rank'] <= n]
    
    pivot = top_n.pivot_table(
        index=['client_num', 'date'],
        columns='rank',
        values=['count', 'sum', 'ratio', category_col],
        aggfunc='first'
    )
    pivot.columns = [f'{category_col}_top{int(col[1])}_{col[0]}' for col in pivot.columns]
    pivot = pivot.reset_index()

    # Fill missing values
    for col in pivot.columns:
        if 'ratio' in col or 'sum' in col or 'count' in col:
            pivot[col] = pivot[col].fillna(0)
        elif category_col in col:
            pivot[col] = pivot[col].fillna(0).astype('int32')

    return pivot

top_mcc = top_n_category_stats(df, 'mcc_code', n=5)
top_merchant = top_n_category_stats(df, 'merchant_name', n=5)


features = (
    agg_amount
    .merge(top_mcc, on=['client_num', 'date'], how='left')
    .merge(top_merchant, on=['client_num', 'date'], how='left')
)


all_clients = df['client_num'].unique()
all_dates = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
full_grid = pd.MultiIndex.from_product(
    [all_clients, all_dates],
    names=['client_num', 'date']
).to_frame(index=False)

final_df = full_grid.merge(features, on=['client_num', 'date'], how='left')


numeric_aggregates = [
    'amount_sum', 'amount_count', 'amount_std', 'amount_max',
    'amount_min', 'amount_mean', 'merchant_nunique', 'mcc_code_nunique'
]

# Fill numeric aggregates with 0
final_df[numeric_aggregates] = final_df[numeric_aggregates].fillna(0)

# Fill top-N feature values
for col in final_df.columns:
    if any(stat in col for stat in ['ratio', 'sum', 'count']):
        final_df[col] = final_df[col].fillna(0)
    elif 'mcc_code_top' in col:
        final_df[col] = final_df[col].fillna(0).astype('int32')
    elif 'merchant_name_top' in col: 
        final_df[col] = final_df[col].fillna(0).astype('int32')

# Add time features
final_df['day_of_week'] = final_df['date'].dt.dayofweek
final_df['day_of_year'] = final_df['date'].dt.dayofyear
final_df['month'] = final_df['date'].dt.month
final_df['week'] = final_df['date'].dt.isocalendar().week.astype('int32')

min_date = final_df['date'].min()
max_date = final_df['date'].max()
date_range = (max_date - min_date).days
final_df['timestep_ratio'] = (final_df['date'] - min_date).dt.days / date_range

gc.collect()


final_df[numeric_aggregates] = np.log(1+final_df[numeric_aggregates])
final_df = pd.merge(final_df, client_aggregates, on=["client_num"], how="left")


# Clients in df_txn but not in df_train
clients_to_predict = set(df_txn['client_num'].unique()) - set(df_train['client_num'].unique())

# Stratified split
train, test = train_test_split(
    df_train,
    test_size=0.2,
    stratify=df_train['target'],
    random_state=42
)


numeric_cols, categorical_cols = (['mcc_code_top3_sum',
  'merchant_name_top2_count',
  'mcc_code_top3_count',
  'mcc_code_top2_sum',
  'merchant_name_top5_count',
  'amount_sum',
  'amount_min',
  'amount_max',
  'mcc_code_top3_ratio',
  'merchant_name_top3_sum',
  'std_amount_by_client_numeric',
  'mcc_code_top2_ratio',
  'mcc_code_top5_ratio',
  'merchant_name_top4_ratio',
  'mcc_code_top4_count',
  'merchant_name_top2_sum',
  'nunique_mcc_code_by_client_numeric',
  'amount_count',
  'mcc_code_nunique',
  'amount_mean',
  'max_date_time_by_client_numeric',
  'merchant_name_top4_count',
  'amount_std',
  'mcc_code_top1_sum',
  'mcc_code_top4_sum',
  'mean_amount_by_client_numeric',
  'nunique_merchant_name_by_client_numeric',
  'sum_amount_by_client_numeric',
  'max_amount_by_client_numeric',
  'merchant_name_top5_sum',
  'mcc_code_top1_count',
  'merchant_name_top4_sum',
  'min_date_time_by_client_numeric',
  'merchant_name_top2_ratio',
  'mcc_code_top2_count',
  'merchant_name_top1_ratio',
  'mcc_code_top5_count',
  'mcc_code_top5_sum',
  'merchant_name_top3_ratio',
  'merchant_name_top5_ratio',
  'mcc_code_top4_ratio',
  'merchant_name_top1_sum',
  'merchant_name_top1_count',
  'mcc_code_top1_ratio',
  'merchant_nunique',
  'merchant_name_top3_count',
  'timestep_ratio'],
 ['mcc_code_top1_mcc_code',
  'mcc_code_top2_mcc_code',
  'mcc_code_top3_mcc_code',
  'mcc_code_top4_mcc_code',
  'mcc_code_top5_mcc_code',
  'merchant_name_top1_merchant_name',
  'merchant_name_top2_merchant_name',
  'merchant_name_top3_merchant_name',
  'merchant_name_top4_merchant_name',
  'merchant_name_top5_merchant_name',
  'day_of_week',
  'day_of_year',
  'month',
  'week'])


# Standardize numeric features
scaler = StandardScaler()
final_df[numeric_cols] = scaler.fit_transform(final_df[numeric_cols])


class_weights = {
    0: 1.00, 1: 0.72, 2: 0.52, 3: 0.37,
    4: 0.27, 5: 0.19, 6: 0.14, 7: 0.10
}

def wmae(y_true, y_pred, weights_map):
    weights = np.array([weights_map.get(y, 0.0) for y in y_true])
    abs_errors = np.abs(y_true - y_pred)
    return np.sum(weights * abs_errors) / np.sum(weights)


class WeightedMAELoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, y_pred, y_true):
        weights = torch.tensor(
            [self.class_weights.get(int(y.item()), 0.0) for y in y_true],
            dtype=y_pred.dtype,
            device=y_pred.device
        )
        loss = torch.abs(y_pred - y_true)
        return torch.mean(weights * loss)


class ClientSequenceDataset(Dataset):
    def __init__(self, df, client_targets, seq_len, numeric_cols, categorical_cols):
        self.seq_len = seq_len
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.client_targets = client_targets.set_index('client_num')['target'].to_dict()
        self.client_ids = list(self.client_targets.keys())

        df_sorted = df.sort_values(['client_num', 'date'])
        grouped = df_sorted.groupby('client_num')

        # Precompute fixed-length sequences
        self.sequences = {
            client_id: (
                group[numeric_cols].values.astype(np.float32),
                group[categorical_cols].values.astype(np.int64)
            )
            for client_id, group in grouped if client_id in self.client_targets
        }

    def __len__(self):
        return len(self.client_ids)

    def __getitem__(self, idx):
        client_id = self.client_ids[idx]
        numeric, categorical = self.sequences[client_id]
        target = self.client_targets[client_id]
        return (
            torch.tensor(numeric),
            torch.tensor(categorical),
            torch.tensor(target).float()
        )



class LSTMRegressor(nn.Module):
    def __init__(self, numeric_dim, cat_cardinalities, hidden_size, emb_dim=16, num_layers=3):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=card, embedding_dim=emb_dim)
            for card in cat_cardinalities
        ])

        input_dim = numeric_dim + emb_dim * len(cat_cardinalities)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, numeric_x, cat_x):
        # Embed categorical inputs
        cat_embedded = torch.cat(
            [emb(cat_x[:, :, i]) for i, emb in enumerate(self.embeddings)],
            dim=-1
        )

        # Concatenate numerical and categorical embeddings
        x = torch.cat([numeric_x, cat_embedded], dim=-1)

        # Pass through LSTM
        x, _ = self.lstm(x)

        # Mean pooling over the last 20 timesteps
        x_last20 = x[:, -20:, :]
        pooled = x_last20.mean(dim=1)

        # Output
        return self.fc(pooled).squeeze(1)



SEQ_LEN = 93
BATCH_SIZE = 32
HIDDEN_SIZE = 1024
LEARNING_RATE = 1e-5
NUM_EPOCHS = 100

cat_cardinalities = [int(final_df[col].max()) + 1 for col in categorical_cols]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")

model = LSTMRegressor(
    numeric_dim=len(numeric_cols),
    cat_cardinalities=cat_cardinalities,
    hidden_size=HIDDEN_SIZE
).to(device)

train_dataset = ClientSequenceDataset(final_df, train, SEQ_LEN, numeric_cols, categorical_cols)
test_dataset = ClientSequenceDataset(final_df, test, SEQ_LEN, numeric_cols, categorical_cols)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
criterion = WeightedMAELoss(class_weights).to(device)

gc.collect()


best_val_wmae_raw = float("inf")
best_val_wmae_rounded = float("inf")

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

    for batch_idx, (num_x, cat_x, y) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
        num_x, cat_x, y = num_x.to(device), cat_x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(num_x, cat_x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    preds, actuals = [], []

    with torch.no_grad():
        for num_x, cat_x, y in tqdm(test_loader, desc="Validating"):
            num_x, cat_x, y = num_x.to(device), cat_x.to(device), y.to(device)
            output = model(num_x, cat_x)
            loss = criterion(output, y)
            val_loss += loss.item()
            preds.extend(output.cpu().numpy())
            actuals.extend(y.cpu().numpy())

    val_wmae_raw = wmae(np.array(actuals), np.array(preds), class_weights)
    preds_rounded = np.round(np.clip(preds, 0, 6)).astype(int)
    val_wmae_rounded = wmae(np.array(actuals), preds_rounded, class_weights)

    print(f"Epoch {epoch+1:02d} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(test_loader):.4f} | "
          f"Val WMAE (raw): {val_wmae_raw:.4f} | "
          f"Val WMAE (rounded): {val_wmae_rounded:.4f}")

    print(f"Current LR: {optimizer.param_groups[0]['lr']:.6e}")

    if val_wmae_raw < best_val_wmae_raw:
        best_val_wmae_raw = val_wmae_raw
        torch.save(model.state_dict(), "model_best_raw.pt")
        print("✅ Checkpoint saved (best raw).")

    if val_wmae_rounded < best_val_wmae_rounded:
        best_val_wmae_rounded = val_wmae_rounded
        torch.save(model.state_dict(), "model_best_rounded.pt")
        print("✅ Checkpoint saved (best rounded).")





model.load_state_dict(torch.load('model_best_raw.pt'))


from tqdm import tqdm

model.eval()
val_loss = 0
preds, actuals = [], []

with torch.no_grad():
    val_iterator = tqdm(test_loader, desc="Validating")
    for num_x, cat_x, y in val_iterator:
        num_x, cat_x, y = num_x.to(device), cat_x.to(device), y.to(device)
        output = model(num_x, cat_x)
        loss = criterion(output, y)
        val_loss += loss.item()
        preds.extend(output.cpu().numpy())
        actuals.extend(y.cpu().numpy())

val_wmae_raw = wmae(np.array(actuals), np.array(preds), class_weights)
preds_rounded = np.round(np.clip(preds, 0, 6)).astype(int)
val_wmae_rounded = wmae(np.array(actuals), preds_rounded, class_weights)

print(f"Val Loss: {val_loss/len(test_loader):.4f} | "
      f"Val WMAE (raw): {val_wmae_raw:.4f} | "
      f"Val WMAE (rounded): {val_wmae_rounded:.4f}")



class PredictionDataset(Dataset):
    def __init__(self, df, client_ids, seq_len, numeric_cols, categorical_cols):
        self.seq_len = seq_len
        self.client_ids = client_ids
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.groups = df.sort_values(['client_num', 'date']).groupby('client_num')

    def __len__(self):
        return len(self.client_ids)

    def __getitem__(self, idx):
        client_id = self.client_ids[idx]
        group = self.groups.get_group(client_id)

        numeric = group[self.numeric_cols].values.astype(np.float32)
        categorical = group[self.categorical_cols].values.astype(np.int64)

        return torch.tensor(client_id), torch.tensor(numeric), torch.tensor(categorical)


# ✅ Setup prediction

predict_dataset = PredictionDataset(
    final_df,
    client_ids=sorted(list(clients_to_predict)),
    seq_len=SEQ_LEN,
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols
)
predict_loader = DataLoader(predict_dataset, batch_size=32)

# ✅ Run predictions
model.eval()
results = []

with torch.no_grad():
    for client_ids, num_x, cat_x in predict_loader:
        num_x, cat_x = num_x.to(device), cat_x.to(device)
        preds = model(num_x, cat_x)
        results.extend(zip(client_ids.numpy(), preds.cpu().numpy()))

# ✅ Build DataFrame with results
pred_df = pd.DataFrame(results, columns=["client_num", "prediction"])


pred_df.head()


pred_df["target"] = np.clip(np.round(pred_df["prediction"].astype(float)), 0, 6).astype(int)
pred_df[["client_num", "target"]].to_csv("submission.csv", index=False)

