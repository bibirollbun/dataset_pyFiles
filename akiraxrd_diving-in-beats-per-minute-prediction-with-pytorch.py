# libraries, globals and configurations

# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, random_split, DataLoader

import warnings
import random
import os
from pathlib import Path
from tqdm.notebook import tqdm

# globals
SEED = 128
BATCH_SIZE = 128
TARGET_COLUMN = "BeatsPerMinute"
StatsDict = dict[str, list[float]]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
input_dir = Path("/kaggle/input")
competition_data_dir = input_dir / "playground-series-s5e9"
extra_data_dir = input_dir / "bpm-prediction-challenge"
submission_file_path = competition_data_dir / "sample_submission.csv"

# enable reproducibility
def set_seed(seed=SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
set_seed()
warnings.simplefilter("ignore")

print(f"Using device: {device}")


def load_dataframes():
    # main data
    competition_train_df = pd.read_csv(competition_data_dir / "train.csv", index_col="id")
    test_df = pd.read_csv(competition_data_dir / "test.csv", index_col="id")

    # extra data
    extra_train_df = pd.read_csv(extra_data_dir / "Train.csv")
    extra_train_df["id"] = np.arange(test_df.index[-1], test_df.index[-1] + len(extra_train_df))
    extra_train_df = extra_train_df.set_index("id")

    # combine main train data with extra train data
    train_df = pd.concat([competition_train_df, extra_train_df], axis=0)

    return train_df, test_df


train_df, test_df = load_dataframes()
train_df.head()


def prepare_dataloaders(dataframe: pd.DataFrame, target_column: str | None = None, batch_size: int = 64):
    dataframe_copy = dataframe.copy()
    feature_columns = [column for column in dataframe.columns
                       if column != target_column]

    # if target column is specified, drop it from the dataframe copy
    if target_column is not None:
        dataframe_copy = dataframe_copy.drop(columns=target_column)

    # min-max normalization
    normalized_dataframe = (dataframe_copy - dataframe_copy.min()) / (dataframe_copy.max() - dataframe_copy.min())
    
    if target_column is None:
        # convert data to tensors
        normalized_features = torch.from_numpy(normalized_dataframe.to_numpy()).to(torch.float32)
        dataset = TensorDataset(normalized_features)
        
        # create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        return dataloader
        
    else:
        # convert data to tensors
        normalized_features = torch.from_numpy(normalized_dataframe[feature_columns].to_numpy()).to(torch.float32)
        targets = torch.from_numpy(dataframe[target_column].to_numpy()).to(torch.float32)
        dataset = TensorDataset(normalized_features, targets)
        
        # divide dataset into datasets for training and validation
        train_dataset_size = int(0.2 * len(dataset))
        validation_dataset_size = len(dataset) - train_dataset_size
        train_dataset, validation_dataset = random_split(dataset, [train_dataset_size, validation_dataset_size])

        # create dataloaders for train and validation datasets
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

        return train_dataloader, validation_dataloader


train_dataloader, validation_dataloader = prepare_dataloaders(train_df, target_column=TARGET_COLUMN, batch_size=BATCH_SIZE)
test_dataloader = prepare_dataloaders(test_df, batch_size=BATCH_SIZE)


class BPMnet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim_factor: int | float = 1.0):
        super().__init__()

        # hard-coded layer dimensions
        layer_dims = (
            in_dim,
            int(512 * hidden_dim_factor),
            int(256 * hidden_dim_factor),
            int(128 * hidden_dim_factor),
        )

        # layer sequence base
        seq = nn.Sequential()

        # add 3 layer blocks to the layer sequence
        for i in range(3):
            seq.add_module(
                f"block_{i}",
                nn.Sequential(
                    nn.Linear(layer_dims[i], layer_dims[i + 1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(layer_dims[i + 1]),
                    nn.Dropout(0.2),
                ),
            )

        # add the output layer
        seq.add_module(
            "regressor",
            nn.Linear(layer_dims[-1], 1),
        )

        # save the layer sequence to the global variable storage
        self.layer_sequence: nn.Module = seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_sequence(x)


def train_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple[nn.Module, StatsDict]:
    
    stats = {
        "loss": [],
        "mae": [],
    }

    model.train()
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        
        logits = model(x).squeeze()

        # calculate loss
        loss = loss_function(logits, y)
        stats["loss"].append(loss.item())

        # calculate metric
        mae_score = F.l1_loss(logits, y)
        stats["mae"].append(mae_score.item())

        # weight tuning
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, stats


def evaluate(
    model: nn.Module, 
    dataloader: DataLoader, 
    loss_function: nn.Module,
    device: torch.device
) -> StatsDict:
    
    stats = {
        "loss": [],
        "mae": [],
    }

    model.eval()
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        
        logits = model(x).squeeze()

        # calculate loss
        loss = loss_function(logits, y)
        stats["loss"].append(loss.item())

        # calculate metric
        mae_score = F.l1_loss(logits, y)
        stats["mae"].append(mae_score.item())

    return stats


def fit(
    model: nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device
) -> tuple[nn.Module, StatsDict]:
    
    history = {
        "train_loss": [],
        "train_mae": [],

        "eval_loss": [],
        "eval_mae": [],
    }

    for epoch in tqdm(range(1, epochs + 1)):
        print(f"Epoch {epoch:3d}/{epochs}")

        model, train_stats = train_epoch(
            model,
            train_dataloader,
            loss_function,
            optimizer,
            device
        )

        eval_stats = evaluate(
            model,
            validation_dataloader,
            loss_function,
            device
        )

        average_train_loss = np.mean(train_stats["loss"])
        average_train_mae = np.mean(train_stats["mae"])
        average_eval_loss = np.mean(eval_stats["loss"])
        average_eval_mae = np.mean(eval_stats["mae"])

        print(f"[TRAIN] Loss: {average_train_loss:<8.3f} MAE score: {average_train_mae:.3f}")
        print(f"[EVAL]  Loss: {average_eval_loss:<8.3f} MAE score: {average_eval_mae:.3f}")
        print()
        
        history["train_loss"].append(average_train_loss)
        history["train_mae"].append(average_train_mae)
        history["eval_loss"].append(average_eval_loss)
        history["eval_mae"].append(average_eval_mae)

    return model, history


class RMSELoss(nn.Module):
    def __init__(self, eps: int = 1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = torch.sqrt(self.mse(logits, targets) + self.eps)
        return loss


learning_rate = 3e-4
eps = 1e-5
epochs = 5

model = BPMnet(
    in_dim=test_df.shape[-1],
    hidden_dim_factor=2.4
)
model.to(device)

loss_function = RMSELoss(eps=eps)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)


model, history = fit(
    model,
    train_dataloader,
    validation_dataloader,
    loss_function,
    optimizer,
    epochs,
    device
)


def infer(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> list[float]:

    predictions: list[float] = []

    model.eval()
    with torch.inference_mode():
        for x in dataloader:
            x = x[0].to(device)
            logits = model(x).squeeze()
            predictions.extend(logits.cpu().tolist())

    return predictions


predictions = infer(model, test_dataloader, device)
predictions[:10]


submission = pd.read_csv(submission_file_path, index_col="id")
submission[TARGET_COLUMN] = predictions

submission.sample(5)


submission.to_csv("submission.csv")

