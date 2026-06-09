import os
import pathlib
import time
import collections
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from metric import score as cmi_score


SEED = 2025
BASE_DIR = pathlib.Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
OUT_DIR = pathlib.Path("/kaggle/working/")
print("output dir:", OUT_DIR)

# data
ID_COL = "sequence_id"
SEQ_COL = "sequence_counter"
FEATURES = [
    "acc_x",
    "acc_y",
    "acc_z",
    "rot_w",
    "rot_x",
    "rot_y",
    "rot_z",
]
TARGET = "gesture"
MAX_LENGTH = 40

# model
NUM_CLASSES = 18
HIDDEN_SIZE = 256
N_LAYERS = 3
N_CLF_LAYERS = 2
BIDIRECTIONAL = True
DROP = 0.15
LABEL_SMOOTHING = 0.1

# trainig
BATCH_SIZE = 16
EPOCHS = 70
LR = 1e-3
LR_MIN = 1e-5
DEVICE = torch.device("cuda")


os.listdir(BASE_DIR)


path_data = BASE_DIR / "train.csv"
header = pd.read_csv(path_data, nrows=0)
columns = header.columns
columns = [x for x in columns if not x.startswith("tof")]
columns = [x for x in columns if not x.startswith("thm")]
columns


data = pd.read_csv(path_data, usecols=columns)
data.shape


data.iloc[0].to_dict()


cat = pd.Categorical(data[TARGET], ordered=True).dtype
data[TARGET] = pd.Categorical(data[TARGET], ordered=True)
data[TARGET] = data[TARGET].cat.codes


cat.categories, cat.ordered


assert NUM_CLASSES == len(cat.categories)
print(f"#{NUM_CLASSES} classes found.")


def add_split(data):
    data["split"] = -1
    df = data[[ID_COL, TARGET]].value_counts().reset_index()[[ID_COL, TARGET]].reset_index(drop=True)
    splitter = StratifiedKFold(shuffle=True, random_state=SEED)
    for i, (__, i_valid) in enumerate(splitter.split(df, y=df[TARGET])):
        ind = df.iloc[i_valid].index
        ids = df.loc[ind][ID_COL].values
        ind = data[data[ID_COL].isin(ids)].index
        data.loc[ind, "split"] = i


add_split(data)


splits = {}
for x in sorted(data["split"].unique()):
    splits[f"split_{x}"] = data[data["split"] == x].groupby(ID_COL)[TARGET].first().value_counts()
pd.DataFrame(splits)


ind_train = data[data["split"] != 0].index
ind_valid = data[data["split"] == 0].index
train = data.loc[ind_train].copy()
valid = data.loc[ind_valid].copy()
len(train), len(valid)


ids_train = train[ID_COL].unique()
ids_valid = valid[ID_COL].unique()
len(ids_train), len(ids_valid), set(ids_valid).isdisjoint(ids_train)


import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


transformer = ColumnTransformer(
    transformers=[
        (
            "pipe",
            Pipeline([("im", SimpleImputer()), ("sc", StandardScaler())]),
            FEATURES,
        )
    ],
    # remainder="passthrough",
    verbose_feature_names_out=False,
)
transformer.set_output(transform="pandas")
transformer.fit(train)


train.loc[11111].to_dict()


train[FEATURES] = transformer.transform(train[FEATURES])
valid[FEATURES] = transformer.transform(valid[FEATURES])
train.loc[11111].to_dict()


joblib.dump(transformer, OUT_DIR / "preprocessing.joblib")


from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class CMIData(Dataset):
    def __init__(self, data, max_length=None):
        super().__init__()
        self.d_data = dict(list(data.groupby(ID_COL)))
        self.features = FEATURES
        self.target = TARGET
        self.max_length = max_length
        self.keys = list(self.d_data)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        df = self.d_data[self.keys[index]]
        if self.max_length is not None:
            df = df.iloc[-self.max_length :]
        return (
            torch.tensor(df[self.features].values.astype(np.float32)),
            torch.tensor(df[self.target].values[-1].astype(np.int64)),
        )


ds = CMIData(train, max_length=10)
len(ds)


x, y = ds[0]
x, y


def collate_fn(batch):
    x_tensors = [item[0] for item in batch]
    y_tensors = [item[1] for item in batch]

    batch_x = pad_sequence(x_tensors, batch_first=False, padding_value=0.0)
    batch_y = torch.tensor(y_tensors)
    # batch_y = pad_sequence(y_tensors, batch_first=False, padding_value=-100)
    mask = pad_sequence(
        [torch.ones_like(x[..., -1]) for x in x_tensors], padding_value=0
    )
    return batch_x, batch_y, mask


ds = CMIData(train, max_length=100)
dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
x, y, mask = next(iter(dl))
x.shape, y.shape, mask.shape


mask[-10:]


class LSTMClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(DROP)
        self.lstm = nn.LSTM(
            input_size=len(FEATURES),
            hidden_size=HIDDEN_SIZE,
            num_layers=N_LAYERS,
            bidirectional=BIDIRECTIONAL,
            dropout=DROP if N_LAYERS > 1 else 0.0,
            batch_first=False,
        )
        output_size = 2 * HIDDEN_SIZE if BIDIRECTIONAL else HIDDEN_SIZE
        clf_layers = []
        for i in range(N_CLF_LAYERS - 1):
            clf_layers.append(nn.Linear(output_size, output_size))
            clf_layers.append(nn.ReLU())
        clf_layers.append(nn.Linear(output_size, NUM_CLASSES))
        self.clf = nn.Sequential(*clf_layers)

    def forward(self, x, mask):
        x, _ = self.lstm(x)
        x = torch.mean(x * mask.unsqueeze(-1), dim=0)
        x = self.drop(x)
        return self.clf(x)


model = LSTMClassifier()
model


with torch.no_grad():
    logits = model(x, mask)
logits


loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
loss_fn


loss_fn(logits, y)


ds_train = CMIData(train, max_length=MAX_LENGTH)
train_loader = DataLoader(
    ds_train, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn, 
    drop_last=True
)
print(f"Training dataset contains: {len(ds_train)}")
ds_valid = CMIData(valid, max_length=MAX_LENGTH)
valid_loader = DataLoader(
    ds_valid,
    shuffle=False,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    drop_last=True,
)
print(f"Valid dataset contains: {len(ds_valid)}")


model = LSTMClassifier().to(DEVICE)
loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, min_lr=LR_MIN
)


def evalute(model, data_loader, loss_fn):
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for x, y, mask in data_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            mask = mask.to(DEVICE)
            logits = model(x, mask)
            loss += loss_fn(logits, y).item()
    model.train()
    return loss / len(data_loader)


def predict(model, data):
    model.eval()
    ds = CMIData(data, max_length=MAX_LENGTH)
    dl = DataLoader(
        ds, shuffle=False, batch_size=BATCH_SIZE, collate_fn=collate_fn, drop_last=False
    )
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y, m in dl:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            m = m.to(DEVICE)
            logits = model(x, m)
            y_pred.append(torch.argmax(logits, dim=-1).cpu().numpy())
            y_true.append(y.cpu().numpy())
    sol = pd.DataFrame(
        {"id": ds.keys, "gesture": cat.categories[np.concatenate(y_true)].values}
    )
    sub = pd.DataFrame(
        {"id": ds.keys, "gesture": cat.categories[np.concatenate(y_pred)].values}
    )
    return sol, sub


sol, sub = predict(model, valid)
cmi_score(sol, sub, "id")


class Monitor:
    def __init__(self):
        self.records = collections.defaultdict(list)

    def add(self, metric_name, epoch, value):
        self.records[metric_name].append({"epoch": epoch, "value": value})
        print(f"Epoch {epoch}/{EPOCHS} - {metric_name}: {value:.4g}")

    @property
    def dataframe(self):
        return pd.DataFrame(
            {
                k: pd.DataFrame(v).rename(columns={"value": k}).set_index("epoch")[k]
                for k, v in self.records.items()
            }
        )


%%time

best_val_loss = float('inf')
best_val_score = float('-inf')

# training loop
monitor = Monitor()
for epoch in tqdm(range(1, EPOCHS + 1)):
    
    # training
    model.train()
    losses = []
    for x, y, mask in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        mask = mask.to(DEVICE)
        logits = model(x, mask)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    train_loss = np.mean(losses)
    monitor.add("train_loss", epoch, train_loss)
    
    # validation
    val_loss = evalute(model, valid_loader, loss_fn)
    monitor.add("val_loss", epoch, val_loss)
    sol, sub = predict(model, valid)
    val_score = cmi_score(sol, sub, "id")
    monitor.add("val_cmi", epoch, val_score)
    scheduler.step(val_loss)
    monitor.add("lr", epoch, optimizer.param_groups[0]['lr'])

    # checkpointing
    if val_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4}")
        path_model = OUT_DIR / "model_best.pt"
        torch.save(model.state_dict(), path_model)
        best_val_loss = val_loss

    # checkpointing
    if val_score > best_val_score:
        print(f"Validation score improved from {best_val_score:.4f} to {val_score:.4}")
        path_model_score = OUT_DIR / "model_best_score.pt"
        torch.save(model.state_dict(), path_model_score)
        best_val_score = val_score


hist = monitor.dataframe
hist.to_csv(OUT_DIR / "history.csv")
hist


hist.loc[hist["val_loss"].idxmin()]


hist.loc[hist["val_cmi"].idxmax()]


fig, axs = plt.subplots(nrows=2, figsize=(12, 8), sharex="col")
plt_kwargs = dict(grid=True, marker='x')
hist[["train_loss", "val_loss"]].plot(ax=axs[0], **plt_kwargs)
hist[["val_cmi"]].plot(ax=axs[1], **plt_kwargs)
plt.show()


model.load_state_dict(torch.load(path_model, weights_only=True))


sol, sub = predict(model, valid)
cmi_score(sol, sub, "id")


model.load_state_dict(torch.load(path_model_score, weights_only=True))


sol, sub = predict(model, valid)
cmi_score(sol, sub, "id")




