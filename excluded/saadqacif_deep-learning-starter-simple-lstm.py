import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from collections import Counter

warnings.filterwarnings("ignore")
NUM_WORKERS = 4


DATA_PATH = "/kaggle/input/ventilator-pressure-prediction/"

sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')
df_train = pd.read_csv(DATA_PATH + 'train.csv')
df_test = pd.read_csv(DATA_PATH + 'test.csv')


df = df_train[df_train['breath_id'] < 5].reset_index(drop=True)


df.head()


def plot_sample(sample_id, df):
    df_breath = df[df['breath_id'] == sample_id]
    r, c  = df_breath[['R', 'C']].values[0]

    cols = ['u_in', 'u_out', 'pressure'] if 'pressure' in df.columns else ['u_in', 'u_out']
    
    plt.figure(figsize=(12, 4))
    for col in ['u_in', 'u_out', 'pressure']:
        plt.plot(df_breath['time_step'], df_breath[col], label=col)
        
    plt.legend()
    plt.title(f'Sample {sample_id} - R={r}, C={c}')


for i in df['breath_id'].unique():
    plot_sample(i, df_train)


import torch
from torch.utils.data import Dataset

class VentilatorDataset(Dataset):
    def __init__(self, df):
        if "pressure" not in df.columns:
            df['pressure'] = 0

        self.df = df.groupby('breath_id').agg(list).reset_index()
        
        self.prepare_data()
                
    def __len__(self):
        return self.df.shape[0]
    
    def prepare_data(self):
        self.pressures = np.array(self.df['pressure'].values.tolist())
        
        rs = np.array(self.df['R'].values.tolist())
        cs = np.array(self.df['C'].values.tolist())
        u_ins = np.array(self.df['u_in'].values.tolist())
        
        self.u_outs = np.array(self.df['u_out'].values.tolist())
        
        self.inputs = np.concatenate([
            rs[:, None], 
            cs[:, None], 
            u_ins[:, None], 
            np.cumsum(u_ins, 1)[:, None],
            self.u_outs[:, None]
        ], 1).transpose(0, 2, 1)

    def __getitem__(self, idx):
        data = {
            "input": torch.tensor(self.inputs[idx], dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "p": torch.tensor(self.pressures[idx], dtype=torch.float),
        }
        
        return data


dataset = VentilatorDataset(df)
dataset[0]


import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup

# Enable mixed precision
scaler = torch.cuda.amp.GradScaler()


# Add early stopping callback
class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, save_path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, save_path)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_path):
        torch.save(model.state_dict(), save_path)


def train_and_profile(device, dataset, model, optimizer, scheduler, config):
    """
    Trains and profiles the model on a given device.
    """
    model.to(device)
    model.train()
    start_time = time.time()

    # DataLoader
    train_loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    
    for epoch in range(config.epochs):
        for batch in train_loader:
            inputs = batch['input'].to(device)
            targets = batch['p'].to(device)

            optimizer.zero_grad()
            predictions = model(inputs).squeeze(-1)
            loss = VentilatorLoss()(predictions, targets, batch['u_out'].to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()

    end_time = time.time()
    elapsed_time = end_time - start_time

    return elapsed_time, model



# Training configurations
class Config:
    """
    Parameters used for training
    """
    # General
    seed = 42
    verbose = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights = True

    # k-fold
    k = 5
    selected_folds = [0, 1, 2, 3, 4]

    # Model
    selected_model = 'rnn'
    input_dim = 5
    dense_dim = 512
    lstm_dim = 512
    logit_dim = 512
    num_classes = 1

    # Training
    loss = "L1Loss"  # not used
    optimizer = "Adam"
    batch_size = 128
    epochs = 200

    lr = 1e-3
    warmup_prop = 0

    val_bs = 256
    first_epoch_eval = 0

import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(
        self,
        input_dim=4,
        lstm_dim=256,
        dense_dim=256,
        logit_dim=256,
        num_classes=1,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dense_dim // 2),
            nn.ReLU(),
            nn.Linear(dense_dim // 2, dense_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True)

        self.logits = nn.Sequential(
            nn.Linear(lstm_dim * 2, logit_dim),
            nn.ReLU(),
            nn.Linear(logit_dim, num_classes),
        )

    def forward(self, x):
        features = self.mlp(x)
        features, _ = self.lstm(features)
        pred = self.logits(features)
        return pred


config = Config()
model = RNNModel(
    input_dim=config.input_dim,
    lstm_dim=config.lstm_dim,
    dense_dim=config.dense_dim,
    logit_dim=config.logit_dim,
    num_classes=config.num_classes,
)


# Optimizer and Scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * Config.epochs * len(dataset) / Config.batch_size),
    num_training_steps=int(Config.epochs * len(dataset) / Config.batch_size),
)

class VentilatorLoss(nn.Module):
    """
    Directly optimizes the competition metric
    """
    def __call__(self, preds, y, u_out):
        w = 1 - u_out
        mae = w * (y - preds).abs()
        # Compute the mean of the loss to return a scalar
        mae = mae.sum(-1) / w.sum(-1)
        return mae.mean()  # Add this to ensure a scalar is returned

# Continue with your existing code
cpu_time, cpu_model = train_and_profile('cpu', dataset, model, optimizer, scheduler, Config)
print(f"Training time on CPU: {cpu_time:.2f} seconds")



# Fixing GPU Device Issue
def train_and_profile(device, dataset, model, optimizer, scheduler, config):
    model.to(device)  # Ensure the model is on the correct device
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    loss_fn = VentilatorLoss()
    model.train()
    start_time = time.time()

    for epoch in range(config.epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            # Move batch data to the correct device
            inputs = batch['input'].to(device)
            targets = batch['p'].to(device)
            u_out = batch['u_out'].to(device)

            # Forward pass
            predictions = model(inputs).squeeze(-1)

            # Compute loss
            loss = loss_fn(predictions, targets, u_out)
            loss = loss.mean()  # Ensure loss is scalar

            # Backward pass
            loss.backward()
            optimizer.step()
        scheduler.step()

    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time, model

# GPU Training
if torch.cuda.is_available():
    model = model.to('cuda')  # Ensure the model is moved to GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * Config.epochs * len(dataset) / Config.batch_size),
        num_training_steps=int(Config.epochs * len(dataset) / Config.batch_size),
    )

    gpu_time, gpu_model = train_and_profile('cuda', dataset, model, optimizer, scheduler, Config)
    print(f"Training time on GPU: {gpu_time:.2f} seconds")



# Distributed Training
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for distributed training")
    multi_gpu_model = torch.nn.DataParallel(model)  # Wrap the model for multi-GPU
    multi_gpu_model = multi_gpu_model.to('cuda')  # Ensure the model is on GPU

    optimizer = torch.optim.Adam(multi_gpu_model.parameters(), lr=Config.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * Config.epochs * len(dataset) / Config.batch_size),
        num_training_steps=int(Config.epochs * len(dataset) / Config.batch_size),
    )

    multi_gpu_time, trained_model = train_and_profile(
        'cuda', dataset, multi_gpu_model, optimizer, scheduler, Config
    )
    print(f"Training time on Multi-GPU: {multi_gpu_time:.2f} seconds")
else:
    print("Multi-GPU training is not available. Using single GPU or CPU.")



# Profiling results
results = {
    "CPU Time": cpu_time,
    "GPU Time": gpu_time if torch.cuda.is_available() else None,
    "Multi-GPU Time": multi_gpu_time if torch.cuda.device_count() > 1 else None
}

print("\nProfiling Results:")
for key, value in results.items():
    if value is not None:
        print(f"{key}: {value:.2f} seconds")


# Plot execution times
plt.figure(figsize=(10, 6))
labels = [key for key in results.keys() if results[key] is not None]
times = [results[key] for key in results.keys() if results[key] is not None]
plt.bar(labels, times, color=['blue', 'green', 'orange'])
plt.title('Execution Times for Training Configurations')
plt.ylabel('Time (seconds)')
plt.xlabel('Configuration')
plt.show()


import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(
        self,
        input_dim=4,
        lstm_dim=256,
        dense_dim=256,
        logit_dim=256,
        num_classes=1,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dense_dim // 2),
            nn.ReLU(),
            nn.Linear(dense_dim // 2, dense_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True)

        self.logits = nn.Sequential(
            nn.Linear(lstm_dim * 2, logit_dim),
            nn.ReLU(),
            nn.Linear(logit_dim, num_classes),
        )

    def forward(self, x):
        features = self.mlp(x)
        features, _ = self.lstm(features)
        pred = self.logits(features)
        return pred


import os
import torch
import random
import numpy as np


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (int): Number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def count_parameters(model, all=False):
    """
    Counts the parameters of a model.

    Args:
        model (torch model): Model to count the parameters of.
        all (bool, optional):  Whether to count not trainable parameters. Defaults to False.

    Returns:
        int: Number of parameters.
    """
    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    
def worker_init_fn(worker_id):
    """
    Handles PyTorch x Numpy seeding issues.

    Args:
        worker_id (int): Id of the worker.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    

def save_model_weights(model, filename, verbose=1, cp_folder=""):
    """
    Saves the weights of a PyTorch model.

    Args:
        model (torch model): Model to save the weights of.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        cp_folder (str, optional): Folder to save to. Defaults to "".
    """
    if verbose:
        print(f"\n -> Saving weights to {os.path.join(cp_folder, filename)}\n")
    torch.save(model.state_dict(), os.path.join(cp_folder, filename))


def compute_metric(df, preds):
    """
    Metric for the problem, as I understood it.
    """
    
    y = np.array(df['pressure'].values.tolist())
    w = 1 - np.array(df['u_out'].values.tolist())
    
    assert y.shape == preds.shape and w.shape == y.shape, (y.shape, preds.shape, w.shape)
    
    mae = w * np.abs(y - preds)
    mae = mae.sum() / w.sum()
    
    return mae


class VentilatorLoss(nn.Module):
    """
    Directly optimizes the competition metric
    """
    def __call__(self, preds, y, u_out):
        w = 1 - u_out
        mae = w * (y - preds).abs()
        mae = mae.sum(-1) / w.sum(-1)

        return mae


import gc
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup


def fit(
    model,
    train_dataset,
    val_dataset,
    loss_name="L1Loss",
    optimizer="Adam",
    epochs=50,
    batch_size=32,
    val_bs=32,
    warmup_prop=0.1,
    lr=1e-3,
    num_classes=1,
    verbose=1,
    first_epoch_eval=0,
    device="cuda"
):
    avg_val_loss = 0.

    # Optimizer
    optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=lr)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Loss
#     loss_fct = getattr(torch.nn, loss_name)(reduction="none")
    loss_fct = VentilatorLoss()

    # Scheduler
    num_warmup_steps = int(warmup_prop * epochs * len(train_loader))
    num_training_steps = int(epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    for epoch in range(epochs):
        model.train()
        model.zero_grad()
        start_time = time.time()

        avg_loss = 0
        for data in train_loader:
            pred = model(data['input'].to(device)).squeeze(-1)

            loss = loss_fct(
                pred,
                data['p'].to(device),
                data['u_out'].to(device),
            ).mean()
            loss.backward()
            avg_loss += loss.item() / len(train_loader)

            optimizer.step()
            scheduler.step()

            for param in model.parameters():
                param.grad = None

        model.eval()
        mae, avg_val_loss = 0, 0
        preds = []

        with torch.no_grad():
            for data in val_loader:
                pred = model(data['input'].to(device)).squeeze(-1)

                loss = loss_fct(
                    pred.detach(), 
                    data['p'].to(device),
                    data['u_out'].to(device),
                ).mean()
                avg_val_loss += loss.item() / len(val_loader)

                preds.append(pred.detach().cpu().numpy())
        
        preds = np.concatenate(preds, 0)
        mae = compute_metric(val_dataset.df, preds)

        elapsed_time = time.time() - start_time
        if (epoch + 1) % verbose == 0:
            elapsed_time = elapsed_time * verbose
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1:02d}/{epochs:02d} \t lr={lr:.1e}\t t={elapsed_time:.0f}s \t"
                f"loss={avg_loss:.3f}",
                end="\t",
            )

            if (epoch + 1 >= first_epoch_eval) or (epoch + 1 == epochs):
                print(f"val_loss={avg_val_loss:.3f}\tmae={mae:.3f}")
            else:
                print("")

    del (val_loader, train_loader, loss, data, pred)
    gc.collect()
    torch.cuda.empty_cache()

    return preds



def predict(
    model,
    dataset,
    batch_size=64,
    device="cuda"
):
    """
    Usual torch predict function. Supports sigmoid and softmax activations.
    Args:
        model (torch model): Model to predict with.
        dataset (PathologyDataset): Dataset to predict on.
        batch_size (int, optional): Batch size. Defaults to 64.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        numpy array [len(dataset) x num_classes]: Predictions.
    """
    model.eval()

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )
    
    preds = []
    with torch.no_grad():
        for data in loader:
            pred = model(data['input'].to(device)).squeeze(-1)
            preds.append(pred.detach().cpu().numpy())

    preds = np.concatenate(preds, 0)
    return preds


def train(config, df_train, df_val, df_test, fold):
    """
    Trains and validate a model.

    Args:
        config (Config): Parameters.
        df_train (pandas dataframe): Training metadata.
        df_val (pandas dataframe): Validation metadata.
        df_test (pandas dataframe): Test metadata.
        fold (int): Selected fold.

    Returns:
        np array: Study validation predictions.
    """

    seed_everything(config.seed)

    model = RNNModel(
        input_dim=config.input_dim,
        lstm_dim=config.lstm_dim,
        dense_dim=config.dense_dim,
        logit_dim=config.logit_dim,
        num_classes=config.num_classes,
    ).to(config.device)
    model.zero_grad()

    train_dataset = VentilatorDataset(df_train)
    val_dataset = VentilatorDataset(df_val)
    test_dataset = VentilatorDataset(df_test)

    n_parameters = count_parameters(model)

    print(f"    -> {len(train_dataset)} training breathes")
    print(f"    -> {len(val_dataset)} validation breathes")
    print(f"    -> {n_parameters} trainable parameters\n")

    pred_val = fit(
        model,
        train_dataset,
        val_dataset,
        loss_name=config.loss,
        optimizer=config.optimizer,
        epochs=config.epochs,
        batch_size=config.batch_size,
        val_bs=config.val_bs,
        lr=config.lr,
        warmup_prop=config.warmup_prop,
        verbose=config.verbose,
        first_epoch_eval=config.first_epoch_eval,
        device=config.device,
    )
    
    pred_test = predict(
        model, 
        test_dataset, 
        batch_size=config.val_bs, 
        device=config.device
    )

    if config.save_weights:
        save_model_weights(
            model,
            f"{config.selected_model}_{fold}.pt",
            cp_folder="",
        )

    del (model, train_dataset, val_dataset, test_dataset)
    gc.collect()
    torch.cuda.empty_cache()

    return pred_val, pred_test


from sklearn.model_selection import GroupKFold

def k_fold(config, df, df_test):
    """
    Performs a patient grouped k-fold cross validation.
    """

    pred_oof = np.zeros(len(df))
    preds_test = []
    
    gkf = GroupKFold(n_splits=config.k)
    splits = list(gkf.split(X=df, y=df, groups=df["breath_id"]))

    for i, (train_idx, val_idx) in enumerate(splits):
        if i in config.selected_folds:
            print(f"\n-------------   Fold {i + 1} / {config.k}  -------------\n")

            df_train = df.iloc[train_idx].copy().reset_index(drop=True)
            df_val = df.iloc[val_idx].copy().reset_index(drop=True)

            pred_val, pred_test = train(config, df_train, df_val, df_test, i)
            
            pred_oof[val_idx] = pred_val.flatten()
            preds_test.append(pred_test.flatten())

    print(f'\n -> CV MAE : {compute_metric(df, pred_oof) :.3f}')

    return pred_oof, np.mean(preds_test, 0)


class Config:
    """
    Parameters used for training
    """
    # General
    seed = 42
    verbose = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights = True

    # k-fold
    k = 5
    selected_folds = [0, 1, 2, 3, 4]
    
    # Model
    selected_model = 'rnn'
    input_dim = 5

    dense_dim = 512
    lstm_dim = 512
    logit_dim = 512
    num_classes = 1

    # Training
    loss = "L1Loss"  # not used
    optimizer = "Adam"
    batch_size = 128
    epochs = 200

    lr = 1e-3
    warmup_prop = 0

    val_bs = 256
    first_epoch_eval = 0


pred_oof, pred_test = k_fold(
    Config, 
    df_train,
    df_test,
)


def plot_prediction(sample_id, df):
    df_breath = df[df['breath_id'] == sample_id]

    cols = ['u_in', 'u_out', 'pressure'] if 'pressure' in df.columns else ['u_in', 'u_out']
    
    plt.figure(figsize=(12, 4))
    for col in ['pred', 'pressure', 'u_out']:
        plt.plot(df_breath['time_step'], df_breath[col], label=col)
        
    metric = compute_metric(df_breath, df_breath['pred'])
        
    plt.legend()
    plt.title(f'Sample {sample_id} - MAE={metric:.3f}')


df_train["pred"] = pred_oof


for i in df_train['breath_id'].unique()[:5]:
    plot_prediction(i, df_train)


df_test['pred'] = pred_test

for i in df_test['breath_id'].unique()[:5]:
    plot_prediction(i, df_test)


sub['pressure'] = pred_test
sub.to_csv('submission.csv', index=False)

