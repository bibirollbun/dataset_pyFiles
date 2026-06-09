import os, random, math, time, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor


SEED = 42
pl.seed_everything(SEED, workers=True)


KAGGLE_ROOT = "/kaggle/input/cmi-detect-behavior-with-sensor-data"
TRAIN_CSV   = os.path.join(KAGGLE_ROOT, "train.csv")
TEST_CSV    = os.path.join(KAGGLE_ROOT, "test.csv")
TEST_DEMO   = os.path.join(KAGGLE_ROOT, "test_demographics.csv")


print("Loading train.csv …")
df = pd.read_csv(TRAIN_CSV)
print(f"Loaded {len(df):,} rows.")

le = LabelEncoder()
df["gesture"] = le.fit_transform(df["gesture"].astype(str))
np.save("gesture_classes.npy", le.classes_)

imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]

# preprocess one sequence
def preprocess_sequence(seq_df: pd.DataFrame) -> np.ndarray:
    data = seq_df[imu_cols].copy()
    data = data.ffill().bfill().fillna(0)
    data = StandardScaler().fit_transform(data)
    return data.astype("float32")

# build sequences
print("Building padded sequences …")
seq_groups = df.groupby("sequence_id")
X_list, y_list, lengths = [], [], []
for seq_id, seq_df in seq_groups:
    X_seq = preprocess_sequence(seq_df)
    X_list.append(X_seq)
    lengths.append(X_seq.shape[0])
    y_list.append(seq_df["gesture"].iloc[0])  # one label per sequence

PAD_LEN = int(np.percentile(lengths, 90))
print(f"Pad/trunc length = {PAD_LEN}")
np.save("sequence_maxlen.npy", PAD_LEN)


def pad_to_len(arr: np.ndarray, maxlen: int, dtype="float32") -> np.ndarray:
    """Pad or truncate a (T, C) array along time axis to exactly maxlen."""
    t, c = arr.shape
    if t >= maxlen:
        return arr[:maxlen].astype(dtype, copy=False)
    out = np.zeros((maxlen, c), dtype=dtype)
    out[:t] = arr
    return out

X = np.stack([pad_to_len(a, PAD_LEN) for a in X_list])  # shape (N, L, C)
y = np.array(y_list, dtype="int64")
NUM_CLASSES = len(le.classes_)


# split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)



class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X; self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).transpose(0,1)  # (C, L)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

class GestureDataModule(pl.LightningDataModule):
    def __init__(self, X_train, y_train, X_val, y_val, batch_size=128):
        super().__init__()
        self.X_train, self.y_train = X_train, y_train
        self.X_val,   self.y_val   = X_val,   y_val
        self.batch_size = batch_size
    def setup(self, stage=None):
        self.train_ds = GestureDataset(self.X_train, self.y_train)
        self.val_ds   = GestureDataset(self.X_val,   self.y_val)
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

data_module = GestureDataModule(X_train, y_train, X_val, y_val)



class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.crop = pad * 2
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        res = self.downsample(x)
        out = out[:, :, :-self.crop]  # causal crop
        res = res[:, :, -out.shape[2]:]
        return F.relu(out + res)

class TCN(nn.Module):
    def __init__(self, n_feats, n_classes, channels=[128, 128, 256, 256], k_size=5, dropout=0.2):
        super().__init__()
        layers = []
        in_ch = n_feats
        for i, out_ch in enumerate(channels):
            layers.append(ResidualBlock(in_ch, out_ch, k_size, dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
    def forward(self, x):  # x: (B, C, L)
        x = self.tcn(x)
        x = self.pool(x)
        return self.head(x)




import torchmetrics
from metric import CompetitionMetric

class HierarchicalF1(torchmetrics.Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False)
        self.add_state("y_true", default=[], dist_reduce_fx=None)
        self.add_state("y_pred", default=[], dist_reduce_fx=None)
        self.metric = CompetitionMetric()

    def update(self, preds, targets):
        self.y_pred.extend(preds.detach().cpu().tolist())
        self.y_true.extend(targets.detach().cpu().tolist())

    def compute(self):
        import pandas as pd
        sol = pd.DataFrame({"idx": range(len(self.y_true)),
                            "gesture": [self.metric.all_classes[i] for i in self.y_true]})
        sub = pd.DataFrame({"idx": range(len(self.y_pred)),
                            "gesture": [self.metric.all_classes[i] for i in self.y_pred]})
        return self.metric.calculate_hierarchical_f1(sol, sub)



class GestureLitModule(pl.LightningModule):
    def __init__(
        self,
        n_feats: int,
        n_classes: int,
        lr: float = 1e-3,
        label_smoothing: float = 0.02,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = TCN(n_feats, n_classes, channels=[128, 256, 256], k_size=3, dropout=0.2)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=n_classes
        )
        self.val_acc  = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=n_classes
        )
        self.val_hf1  = HierarchicalF1()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        x, y = batch                           
        logits = self(x)
        loss = self.criterion(logits, y)

        probs = F.softmax(logits, dim=-1)

        if stage == "train":
            self.train_acc.update(probs, y)
        else:
            self.val_acc.update(probs, y)
            self.val_hf1.update(logits.argmax(1), y)

        self.log(f"{stage}_loss", loss,
                 on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        self.log("val_acc",  self.val_acc.compute(),  prog_bar=True)
        self.log("val_hf1",  self.val_hf1.compute(),  prog_bar=True)
        self.val_acc.reset()
        self.val_hf1.reset()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=4, min_lr=1e-6
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_hf1",
            },
        }


checkpt = ModelCheckpoint(
    monitor="val_hf1", mode="max", save_top_k=1, filename="best")
callbacks = [checkpt,
             EarlyStopping(monitor="val_hf1", mode="max", patience=15),
             LearningRateMonitor("epoch")]

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    precision="16-mixed" if torch.cuda.is_available() else 32,
    callbacks=callbacks,
    log_every_n_steps=50,
)

lit_model = GestureLitModule(n_feats=X.shape[2], n_classes=NUM_CLASSES)
trainer.fit(lit_model, data_module)
print("Best ckpt:", checkpt.best_model_path)

# save state_dict for inference (lighter than Lightning ckpt)
best_ckpt = torch.load(checkpt.best_model_path, map_location="cpu")
state_dict = {k.replace("model.",""):v for k,v in best_ckpt["state_dict"].items()}  # strip prefix
torch.save(state_dict, "gesture_tcn_pt.pth")
print("Saved gesture_tcn_pt.pth")



def compute_val_hf1(
    best_ckpt_path: str,
    data_module: pl.LightningDataModule,
    n_feats: int,
    n_classes: int,
    device: torch.device | str | None = None,
) -> float:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    best_model = GestureLitModule.load_from_checkpoint(
        best_ckpt_path,
        n_feats=n_feats,
        n_classes=n_classes,
    ).to(device)
    best_model.eval()

    hf1_metric = HierarchicalF1().to(device)

    with torch.no_grad():
        for x, y in data_module.val_dataloader():
            x, y = x.to(device), y.to(device)
            logits = best_model(x)
            hf1_metric.update(logits.argmax(1), y)

    final_hf1: float = hf1_metric.compute()
    return final_hf1

best_ckpt_path = checkpt.best_model_path  
hf1_val = compute_val_hf1(
    best_ckpt_path=best_ckpt_path,
    data_module=data_module,
    n_feats=X.shape[2],
    n_classes=NUM_CLASSES,
)

print(f"Hierarchical F1 (validation, best ckpt): {hf1_val:.4f}")




import polars as pl

gesture_classes = np.load("gesture_classes.npy", allow_pickle=True)
MAXLEN = int(np.load("sequence_maxlen.npy"))
_infer_model = TCN(n_feats=X.shape[2], n_classes=NUM_CLASSES, channels=[128, 256, 256], k_size=3, dropout=0.2)
_infer_model.load_state_dict(torch.load("gesture_tcn_pt.pth", map_location="cpu"))
_infer_model.eval()

@torch.no_grad()
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    arr = preprocess_sequence(sequence.to_pandas())
    arr = pad_to_len(arr, MAXLEN)
    x = torch.from_numpy(arr).transpose(0,1).unsqueeze(0)
    logits = _infer_model(x)
    idx = logits.argmax(1).item()
    return str(gesture_classes[idx])


import kaggle_evaluation.cmi_inference_server

inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )



# Manual test (only runs outside Kaggle gateway) (credit: https://www.kaggle.com/code/richolson/cmi-2025-1d-cnn-imu-only-baseline)
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print("\nRunning manual test...")
    test_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
    sample_seq_id = test_df['sequence_id'].unique()[0]
    test_seq = test_df[test_df['sequence_id'] == sample_seq_id]
    prediction = predict(pl.DataFrame(test_seq), None)
    print(f"Manual prediction result for sequence_id {sample_seq_id}: {prediction}")

