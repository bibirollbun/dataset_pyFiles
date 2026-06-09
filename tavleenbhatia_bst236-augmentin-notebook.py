# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import pickle

df = pd.read_csv("/kaggle/input/augdata-train/augmented_train_features.csv")
feature_cols = [col for col in df.columns if col not in ['target_id', 'resid', 'resname', 'Unnamed: 0']]

features_dict_train = {
    tid: group.sort_values("resid")[feature_cols].to_numpy()
    for tid, group in df.groupby("target_id")
}

with open("train_features_dict.pkl", "wb") as f:
    pickle.dump(features_dict_train, f)


df_val = pd.read_csv("/kaggle/input/augdata-val/augmented_validation_features.csv")
val_features_dict = {
    tid: group.sort_values("resid")[feature_cols].to_numpy()
    for tid, group in df_val.groupby("target_id")
}

with open("val_features_dict.pkl", "wb") as f:
    pickle.dump(val_features_dict, f)


df_test = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")


# Explode each sequence into rows per residue
records = []
for _, row in df_test.iterrows():
    seq = row["sequence"]
    for i, res in enumerate(seq):
        records.append({
            "target_id": row["target_id"],
            "resid": i + 1,           # Residue index (1-based)
            "resname": res,           # A, C, G, or U
        })

test_residue_df = pd.DataFrame(records)


# Load original train features
train_df = pd.read_csv("/kaggle/input/augdata-train/augmented_train_features.csv")
feature_cols = [col for col in train_df.columns if col not in ["Unnamed: 0", "target_id", "resid", "resname"]]

# Compute column-wise means
feature_means = train_df[feature_cols].mean()

# Broadcast to each row of the test data
for col in feature_cols:
    test_residue_df[col] = feature_means[col]


test_residue_df.head()


import numpy as np

labels_df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")

def extract_target_id(full_id):
    return "_".join(full_id.split("_")[:2])

labels_df["target_id"] = labels_df["ID"].apply(extract_target_id)

def extract_coords(df):
    coords = []
    i = 1
    while f"x_{i}" in df.columns:
        coords.append(df[[f"x_{i}", f"y_{i}", f"z_{i}"]].to_numpy())
        i += 1
    return np.stack(coords, axis=1)  # shape: [L, 5, 3] if 5 conformations

labels_dict = {
    tid: extract_coords(group.sort_values("resid"))
    for tid, group in labels_df.groupby("target_id")
}

with open("train_labels_dict.pkl", "wb") as f:
    pickle.dump(labels_dict, f)


%pip install pytorch_lightning


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from torch.cuda.amp import autocast

class CleanRNADataset(torch.utils.data.Dataset):
    def __init__(self, features_dict, labels_dict=None):
        self.ids = list(features_dict.keys())
        self.features = features_dict
        self.labels = labels_dict

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        tid = self.ids[idx]
        x = torch.tensor(self.features[tid], dtype=torch.float32)
        sample = {"features": x, "target_id": tid}
        if self.labels:
            y = torch.tensor(self.labels[tid], dtype=torch.float32)
            sample["labels"] = y
        return sample



from torch.utils.data import Dataset

class RNATestDataset(Dataset):
    def __init__(self, df, feature_cols):
        self.groups = list(df.groupby("target_id"))
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        tid, group = self.groups[idx]
        group = group.sort_values("resid")
        features = torch.tensor(group[self.feature_cols].to_numpy(), dtype=torch.float32)
        return {
            "features": features,  # [L, D]
            "target_id": tid,
            "resname": list(group["resname"]),
            "resid": list(group["resid"])
        }



from torch.utils.data import DataLoader

train_ds = CleanRNADataset(features_dict=features_dict_train, labels_dict=labels_dict)
val_ds = CleanRNADataset(features_dict=val_features_dict, labels_dict=None)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)


feature_cols = [col for col in test_residue_df.columns if col not in ["target_id", "resid", "resname"]]
from torch.utils.data import DataLoader


test_ds = RNATestDataset(test_residue_df, feature_cols)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)


import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanMetric
from torch.optim.lr_scheduler import ReduceLROnPlateau

class RNASeqRegressor(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=128, num_structures=5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3)
            ) for _ in range(num_structures)
        ])
        self.loss_tracker = MeanMetric()

    def forward(self, x):
        x, _ = self.lstm(x)              # [B, L, 2H]
        x = self.proj(x)                 # [B, L, H]
        return [head(x) for head in self.heads]  # List of [B, L, 3]

    def calculate_rmsd(self, pred, target):
      return torch.sqrt(torch.mean((pred.float() - target.float()) ** 2))

    def _step(self, batch, mode):
        x = batch['features']                   # [B, L, D]
        y = batch.get('labels')                 # [B, L, S, 3]
        if y is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        preds = self(x)                         # list of [B, L, 3]
        total_loss = 0.0
        valid = 0

        for i, pred in enumerate(preds):
          if i < y.shape[2]:  # y: [B, L_target, num_structures, 3]
            target = y[:, :, i, :]  # [B, L_target, 3]

            # ✨ Truncate to min seq length
            min_len = min(pred.shape[1], target.shape[1])
            pred = pred[:, :min_len, :]       # [B, min_len, 3]
            target = target[:, :min_len, :]   # [B, min_len, 3]

            loss = self.calculate_rmsd(pred, target)
            if torch.isfinite(loss):
                    total_loss += loss
                    valid += 1

        if valid > 0:
            avg_loss = total_loss / valid
        else:
            # Return a dummy differentiable scalar to avoid error
            avg_loss = torch.zeros(1, device=self.device, requires_grad=True).sum()

            # Always log the RMSD as a separate metric
        self.log(f"{mode}_rmsd", avg_loss, prog_bar=False, on_epoch=True, on_step=False, batch_size=x.size(0))
        self.log("val_rmsd", avg_loss, prog_bar=False, on_epoch=True, on_step=False, batch_size=x.size(0))
        # Log loss under Lightning's expected name
        if mode == "train":
            self.log("train_loss", avg_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=x.size(0))
        elif mode == "val":
            self.log("val_loss", avg_loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=x.size(0))

        return avg_loss

    #print("Loss requires grad?", avg_loss.requires_grad)

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch['features'].size(0))

    def on_train_epoch_end(self):
        print("✅ Epoch complete: train_rmsd =", self.trainer.callback_metrics.get("train_rmsd"))

    def on_validation_epoch_end(self):
        print("✅ Epoch complete: val_rmsd =", self.trainer.callback_metrics.get("val_rmsd"))




    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_rmsd"}



from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

input_dim = train_ds[0]['features'].shape[1]
model = RNASeqRegressor(input_dim=input_dim)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, strict=False),
    ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
]

trainer = pl.Trainer(
    max_epochs=25,
    precision=32,  # ✅ Force full precision
    accelerator='auto',
    callbacks=callbacks,
    logger=CSVLogger("logs")
)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)



import pandas as pd

sample_path = "/kaggle/input/stanford-rna-3d-folding/sample_submission.csv"
sample_df = pd.read_csv(sample_path)
print(sample_df.head())



import pandas as pd

# Step 1: Run model predictions and collect coords from all heads
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

all_preds = []

for batch in test_loader:
    features = batch["features"].to(model.device)  # [B, L, D]
    ids = batch["target_id"]  # list of target_ids

    with torch.no_grad():
        outputs = model(features)  # list of 5 tensors: [B, L, 3]

    for i, tid in enumerate(ids):  # loop over batch items
        seq_len = features.shape[1]
        entry = {
            "target_id": tid,
            "resid": list(range(1, seq_len + 1)),  # 1-based indexing
            "resname": ["G"] * seq_len  # placeholder — use real nucleotide if available
        }

        # Gather coords from each head
        for head_idx, head_output in enumerate(outputs):  # loop over 5 heads
            coords = head_output[i].cpu().numpy()  # [L, 3]
            entry.update({
                f"x_{head_idx+1}": coords[:, 0],
                f"y_{head_idx+1}": coords[:, 1],
                f"z_{head_idx+1}": coords[:, 2],
            })

        df = pd.DataFrame(entry)
        df["ID"] = [f"{tid}_{r}" for r in df["resid"]]
        all_preds.append(df)

# Step 2: Combine all sequences into one DataFrame
submission_df = pd.concat(all_preds, ignore_index=True)

# Step 3: Reorder columns to match Kaggle format
coord_cols = []
for i in range(1, 6):
    coord_cols.extend([f"x_{i}", f"y_{i}", f"z_{i}"])

submission_df = submission_df[["ID", "resname", "resid"] + coord_cols]

# Step 4: Save final CSV
submission_df.to_csv("/kaggle/working/submission.csv", index=False)


