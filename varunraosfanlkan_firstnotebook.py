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


# === Imports ===
import os
import random
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm
from scipy.special import logsumexp

from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

import wandb
from google.colab import userdata
from kaggle_secrets import UserSecretsClient



# === Global Configurations ===
warnings.filterwarnings('ignore')
SEED = 42



# === Utility Functions ===
def seed_everything(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeds set with seed {seed}")



def init_wandb(project_name: str = "geology-forecast-challenge", config: dict = None):
    """Initialize Weights & Biases logging."""
    try:
        user_secrets = UserSecretsClient()
        wandb_api_key = user_secrets.get_secret("wandb")
        os.environ['WANDB_API_KEY'] = wandb_api_key
        
        wandb.login(key=wandb_api_key)
        run = wandb.init(
            project=project_name,
            config=config,
            tags=["LSTM", "Geology Forecast Challenge"]
        )
        
        print("âœ… W&B successfully initialized")
        return run

    except Exception as e:
        print(f"â�Œ Error initializing W&B: {e}")
        return None



# === Device Setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"ğŸ”§ Using device: {device}")



# === Load Dataset ===
DATA_PATH = "/kaggle/input/geology-forecast-challenge-open/data"

train = pd.read_csv(f"{DATA_PATH}/train.csv").fillna(0)
test = pd.read_csv(f"{DATA_PATH}/test.csv").fillna(0)
sub = pd.read_csv(f"{DATA_PATH}/sample_submission.csv")

FEATURES = [col for col in test.columns if col != 'geology_id']
TARGETS = [col for col in sub.columns if col != 'geology_id']



# Prepare training solution and subset
solution = train[['geology_id'] + TARGETS].copy()
train_sub = train[['geology_id'] + TARGETS].copy()



# === Model Definition ===
class LSTMForecastModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.GELU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take last time step
        lstm_out = self.layer_norm(lstm_out)

        x = self.activation(self.fc1(lstm_out))
        x = self.dropout(x)
        return self.fc2(x)



# === Dataset Class ===
class GeologyDataset(Dataset):
    def __init__(self, features, targets=None, is_test=False):
        self.features = features
        self.targets = targets
        self.is_test = is_test

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx].reshape(-1, 1)
        if self.is_test:
            return x
        return x, self.targets[idx]




# === Data Preprocessing ===
def preprocess_data(df, feature_cols, target_cols=None, is_test=False):
    X = df[feature_cols].values
    if not is_test and target_cols is not None:
        y = df[target_cols].values
        return X, y
    return X


# === NLL Score Computation ===
def compute_nll_score(solution_df, submission_df, row_id_col='geology_id'):
    solution_df = solution_df.copy()
    submission_df = submission_df.copy()
    solution_df.drop(columns=[row_id_col], inplace=True)
    submission_df.drop(columns=[row_id_col], inplace=True)

    NEGATIVE_PART = -299
    LARGEST_CHUNK = 600
    TOTAL_REALIZATIONS = 10
    INFLATION_SIGMA = 600

    # Precompute inverse diagonal of covariance matrix
    sigma_2 = np.ones(LARGEST_CHUNK + NEGATIVE_PART - 1)
    from_ranges = [1, 61, 245]
    to_ranges = [61, 245, 301]
    slopes = [1.0406, 0.0, 7.8353]
    offsets = [-6.4307, -2.1617, -45.2488]

    for i, (start, end, slope, offset) in enumerate(zip(from_ranges, to_ranges, slopes, offsets)):
        for j in range(start, end):
            sigma_2[j - 1] = np.exp(np.log(j) * slope + offset)

    sigma_2 *= INFLATION_SIGMA
    cov_inv_diag = 1. / sigma_2

    n_rows = solution_df.shape[0]
    n_cols = LARGEST_CHUNK + NEGATIVE_PART - 1
    log_p = np.log(1. / TOTAL_REALIZATIONS)

    solution_arr = np.zeros((n_rows, TOTAL_REALIZATIONS, n_cols))
    submission_arr = np.zeros_like(solution_arr)

    for k in range(TOTAL_REALIZATIONS):
        for i in range(n_cols):
            col = f"r_{k}_pos_{i+1}" if k > 0 else str(i+1)
            solution_arr[:, k, i] = solution_df[col].values
            submission_arr[:, k, i] = submission_df[col].values

    misfit = solution_arr - submission_arr
    inner_prod = np.sum(cov_inv_diag * misfit * misfit, axis=2)
    nll = -logsumexp(log_p - inner_prod, axis=1)

    return nll.mean()



# === Training Function ===
def train_model_with_nll_loss(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for data, target in train_loader:
        data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        output = model(data)

        # Normalize for stability
        target_mean = target.mean(dim=0)
        target_std = target.std(dim=0) + 1e-6

        normalized_output = (output - target_mean) / target_std
        normalized_target = (target - target_mean) / target_std

        loss = F.mse_loss(normalized_output, normalized_target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# === Validation Function ===
def validate_model(model, val_loader, device):
    model.eval()
    val_losses = []
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.float32)
            output = model(data)

            target_mean = target.mean(dim=0)
            target_std = target.std(dim=0) + 1e-6

            normalized_output = (output - target_mean) / target_std
            normalized_target = (target - target_mean) / target_std

            loss = F.mse_loss(normalized_output, normalized_target)

            val_losses.append(loss.item())
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    return (
        np.mean(val_losses),
        np.concatenate(all_preds),
        np.concatenate(all_targets),
    )



def train_and_predict(
    fold_idx, 
    train_index, 
    val_index, 
    X_num, 
    y,
    X_num_test,
    config
):
    fold_config = config.copy()
    fold_config["fold"] = fold_idx
    run = init_wandb(config=fold_config)

    # Prepare data
    X_train, X_val = X_num[train_index], X_num[val_index]
    y_train, y_val = y[train_index], y[val_index]

    train_dataset = GeologyDataset(X_train, y_train)
    val_dataset = GeologyDataset(X_val, y_val)
    test_dataset = GeologyDataset(X_num_test, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    # Model and optimizer
    model = LSTMForecastModel(
        input_size=1,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=len(TARGETS),
        dropout=config['dropout']
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        eps=1e-8
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=1e-6
    )

    best_val_loss = float('inf')
    val_predictions = np.zeros((len(val_index), len(TARGETS)))
    test_predictions = np.zeros((len(X_num_test), len(TARGETS)))

    print(f"\nğŸ”� Training Fold {fold_idx + 1}/{config.get('n_folds', 5)}")

    for epoch in range(config['epochs']):
        train_loss = train_model_with_nll_loss(model, train_loader, optimizer, device)
        val_loss, val_preds, _ = validate_model(model, val_loader, device)
        val_predictions = val_preds
        scheduler.step()

        # Logging
        if run:
            run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]['lr']
            })

        print(f"ğŸ“… Epoch {epoch+1}/{config['epochs']} â€” Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f"model_fold_{fold_idx}.pt"
            torch.save(model.state_dict(), model_path)
            if run:
                run.save(model_path)

    # Predict with best model
    model.load_state_dict(torch.load(f"model_fold_{fold_idx}.pt"))
    model.eval()

    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device, dtype=torch.float32) if not isinstance(batch, list) else batch[0].to(device, dtype=torch.float32)
            output = model(batch)
            test_preds.append(output.cpu().numpy())

    test_predictions = np.concatenate(test_preds)
    train_sub.loc[val_index, TARGETS] = val_predictions

    if run:
        run.finish()

    return test_predictions



config = {
    'model_type': 'LSTM',
    'hidden_size': 1024, 
    'num_layers': 3,
    'dropout': 0.3,
    'learning_rate': 5e-4,
    'weight_decay': 1e-5,
    'batch_size': 128,
    'epochs': 30,
    'seed': SEED,
    'n_folds': 5,  # Optional for printing
}



from sklearn.model_selection import KFold

# === K-Fold Setup ===
folds = 5
kf = KFold(n_splits=folds, random_state=SEED, shuffle=True)

X_num, y = train[FEATURES].values, train[TARGETS].values
X_num_test = test[FEATURES].values



# === Results Tracking ===
test_preds_all_folds = np.zeros((folds, len(test), len(TARGETS)))
val_scores = []

print(f"\nğŸ§ª Starting {folds}-Fold Training...\n")



for fold_idx, (train_index, val_index) in enumerate(kf.split(X_num)):
    test_preds = train_and_predict(
        fold_idx=fold_idx,
        train_index=train_index,
        val_index=val_index,
        X_num=X_num,
        y=y,
        X_num_test=X_num_test,
        config=config
    )

    test_preds_all_folds[fold_idx] = test_preds

    fold_val_preds = train_sub.loc[val_index, ['geology_id'] + TARGETS]
    fold_val_solution = solution.loc[val_index]

    fold_score = compute_nll_score(fold_val_solution, fold_val_preds)
    val_scores.append(fold_score)

    print(f"âœ… Fold {fold_idx + 1}/{folds} â€” Validation NLL Score: {fold_score:.6f}")



# === Aggregated Results ===
avg_val_score = np.mean(val_scores)
print(f"\nğŸ“Š Average Validation NLL Score: {avg_val_score:.6f}")


# === Final Prediction Averaging ===
test_preds_avg = np.mean(test_preds_all_folds, axis=0)


# === Create Submission File ===
submission = sub.copy()
submission[TARGETS] = test_preds_avg
submission.to_csv("submission.csv", index=False)

print("ğŸ“� Submission file 'submission.csv' created successfully!")





