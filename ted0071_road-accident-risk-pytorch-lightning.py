import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchmetrics as tm
import torch

from time import strftime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from matplotlib.ticker import MaxNLocator
from dataclasses import dataclass
from typing import List, Optional, Tuple

from torch import nn, optim, nn, utils, Tensor
from torch.optim import Adam
from torchmetrics.classification import BinaryAccuracy
from torch.utils.data import DataLoader, random_split, TensorDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pathlib import Path


print(torch.__version__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


@dataclass(frozen=True)
class Config:
    CSV_PATH = '/kaggle/input/playground-series-s5e10/'
    LOG_PATH = '/kaggle/working/lightning_logs'
    BATCH_SIZE = 128
    MAX_EPOCHS = 30
    VAL_SIZE = 0.2
    SEED = 2025 + 7
    LEARNING_RATE = 1e-5
    NUM_WORKERS = 4
    EARLY_STOPPING = 10
    DROPOUT = 0.15
    TARGET  = 'accident_risk'
    INDEX = 'id'
    SUBMISSION_FILE = 'submission.csv'


class AccidentRiskDataModule(pl.LightningDataModule):
    TRAIN_CSV = 'train.csv'
    TEST_CSV = 'test.csv'

    CATEGORICAL_COLS = [
        "road_type", 
        "lighting", 
        "weather", 
        "road_signs_present",
        "public_road", 
        "time_of_day", 
        "holiday", 
        "school_season"
    ]
    NUMERIC_COLS = [
        "num_lanes", 
        "curvature", 
        "speed_limit", 
        "num_reported_accidents"
    ]
    TARGET_COL = "accident_risk"
    INDEX_COL = "id"
    
    def __init__(
        self, 
        csv_path: str, 
        batch_size: int = 32,
        num_workers: int = 0,
        val_size: float = 0.2,
        seed: int = 2025
    ) -> None:
        super().__init__()
        
        self.csv_path = Path(csv_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.seed = seed

        self.transformer: Optional[ColumnTransformer] = None
        self.input_dim: Optional[int] = None

        self.X_train = self.y_train = None
        self.X_val = self.y_val = None
        self.X_test = None
        
        self.transformer = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(
                    handle_unknown="ignore", 
                    drop='if_binary'), 
                 self.CATEGORICAL_COLS),
                ("num", StandardScaler(), self.NUMERIC_COLS),
            ], remainder="drop"
        )
    
    def prepare_data(self) -> None:
        # nothing to do
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        df_train = pd.read_csv(self.csv_path / self.TRAIN_CSV, index_col=self.INDEX_COL)
        df_test = pd.read_csv(self.csv_path / self.TEST_CSV, index_col=self.INDEX_COL)

        X_train, X_val, self.y_train, self.y_val = train_test_split(
            df_train[self.CATEGORICAL_COLS + self.NUMERIC_COLS],
            df_train[self.TARGET_COL],
            test_size=self.val_size, 
            random_state=self.seed
        )

        self.X_train = self.transformer.fit_transform(X_train)
        self.X_val = self.transformer.transform(X_val)
        self.X_test = self.transformer.transform(df_test)
        self.test_ids = df_test.index.values
        self.input_dim = self.X_train.shape[1]

    def train_dataloader(self) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32), 
            torch.from_numpy(self.y_train.values.astype(np.float32))
        )
        
        return DataLoader(
            ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(self.X_val, dtype=torch.float32), 
            torch.from_numpy(self.y_val.values.astype(np.float32))
        )

        return DataLoader(
            ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        ds = TensorDataset(
            torch.tensor(self.X_test, dtype=torch.float32), 
            torch.from_numpy(self.test_ids)
        )

        return DataLoader(
            ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers)


class AccidentRiskRegressor(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.15,
        max_lr: float = 1e-3,
        max_epochs: int = 50,
        eta_min=0.0,
        warmup_epochs=5,
        weight_decay: float = 1e-4
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.loss_fn = nn.MSELoss()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512, bias=False), 
            nn.LeakyReLU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            
            nn.Linear(512, 128, bias=False), 
            nn.LeakyReLU(), 
            nn.LayerNorm(128),
            nn.Dropout(dropout),

            nn.Linear(128, 1)
         )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), 
            lr=self.hparams.max_lr, 
            weight_decay=self.hparams.weight_decay
        )
 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            verbose=True
        )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch"
            }
        }       

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_preds = self(x)

        loss = self.loss_fn(y_preds, y.squeeze(-1))
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_preds = self(x)

        loss = self.loss_fn(y_preds, y.squeeze(-1))
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss


dm = AccidentRiskDataModule(
    csv_path=Config.CSV_PATH,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    val_size=Config.VAL_SIZE,
    seed=Config.SEED
)

dm.setup()


model = AccidentRiskRegressor(
    input_dim=dm.input_dim, 
    max_lr=Config.LEARNING_RATE,
    dropout=Config.DROPOUT
)

callbacks = [
    EarlyStopping(
        monitor="val/loss", 
        patience=Config.EARLY_STOPPING, 
        mode="min"
    ),
    LearningRateMonitor(logging_interval="epoch")
]


run_name = f"accident_{strftime('%Y%m%d-%H%M%S')}"
logger = CSVLogger(Config.LOG_PATH, name=run_name)

trainer = pl.Trainer(
    max_epochs=Config.MAX_EPOCHS,
    accelerator="auto",
    devices="auto",
    precision=32,
    callbacks=callbacks,
    log_every_n_steps=10,
    enable_progress_bar=True,
    num_sanity_val_steps=0,
    logger=logger
)


trainer.fit(model, datamodule=dm)


def get_logs(log_path: str, metrics_csv: str, run_name: str):
    EPOCH_COL = 'epoch'
    METRIC_COLS = [
        EPOCH_COL, 
        'train/loss', 
        'val/loss'
    ]
    
    log_dir =  log_path / run_name / "version_0"

    df = pd.read_csv(log_dir / metrics_csv)
    df = df[df[EPOCH_COL].fillna(-1) >= 0]
    df = df.groupby(EPOCH_COL, as_index=False)[METRIC_COLS].last()

    df[EPOCH_COL] = df[EPOCH_COL].astype(int) + 1
    return df.set_index(EPOCH_COL)

log_df = get_logs(Path(Config.LOG_PATH), "metrics.csv", run_name)


fig, ax = plt.subplots(1, 1, figsize=(8, 4), sharex=True)


ax.plot(log_df.index, log_df["train/loss"], label="Train", marker='o', ms=2)
ax.plot(log_df.index, log_df["val/loss"], label="Val", marker='o', ms=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend()

plt.show()


model.to(device).eval()
val_loader = dm.val_dataloader()


y_true = []
y_pred = []

with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        y_hat = model(x_batch).clamp(0.0, 1.0)

        y_true.extend(y_batch.cpu().numpy())
        y_pred.extend(y_hat.cpu().numpy())


print(f'RMSE: {np.sqrt(mean_squared_error(y_true, y_pred))}')


fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

axes[0].scatter(y_true, y_pred, s=1, alpha=0.6)
axes[0].plot([0, 1], [0, 1], "r-", linewidth=1)
axes[0].set_xlabel("y_true")
axes[0].set_ylabel("y_pred")
axes[0].set_title("y_true vs. y_pred")
axes[0].grid(True, alpha=0.3)

axes[1].hist(y_true, bins=30, alpha=0.3, label="y_true")
axes[1].hist(y_pred, bins=30, alpha=0.3, label="y_pred")
axes[1].set_xlabel("Risk")
axes[1].set_ylabel("Freq")
axes[1].set_title("Histogram y_true / y_pred")
axes[1].legend()
axes[1].grid(True, alpha=0.6)

plt.tight_layout()
plt.show()


model.to(device).eval()
test_loader = dm.test_dataloader()


ids = []
y_pred = []

with torch.no_grad():
    for x_batch, id_batch in test_loader:
        x_batch = x_batch.to(device)
        y_hat = model(x_batch).clamp(0.0, 1.0)

        ids.extend(id_batch.cpu().numpy())
        y_pred.extend(y_hat.cpu().numpy())


submission_data = pd.DataFrame({
    Config.INDEX: ids,
    Config.TARGET: y_pred,
}).set_index(Config.INDEX)

submission_data


submission_data.to_csv(Config.SUBMISSION_FILE)

