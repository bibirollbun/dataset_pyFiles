import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from scipy.special import logsumexp
from collections import OrderedDict
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

import wandb
from google.colab import userdata

import warnings
warnings.filterwarnings('ignore')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
# seed_everything(SEED)


from kaggle_secrets import UserSecretsClient

def init_wandb(project_name="geology-forecast-challenge", config=None):
    try:
        user_secrets = UserSecretsClient()
        
        wandb_api_key = user_secrets.get_secret("wandb")
        os.environ['WANDB_API_KEY'] = wandb_api_key
        
        wandb.login(key=wandb_api_key)
        
        run = wandb.init(
            project=project_name,
            config=config,
            tags=["LSTM", "Geology Forecast Challenge"],
        )
        
        print("W&B successfully initialized")
        return run
    
    except Exception as e:
        print(f"Error initializing W&B: {str(e)}")
        return None


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


train = pd.read_csv("/kaggle/input/geology-forecast-challenge-open/data/train.csv").fillna(0)
test = pd.read_csv("/kaggle/input/geology-forecast-challenge-open/data/test.csv").fillna(0)
sub = pd.read_csv('/kaggle/input/geology-forecast-challenge-open/data/sample_submission.csv')


FEATURES = [c for c in test.columns if c != 'geology_id']
TARGETS = [c for c in sub.columns if c != 'geology_id']
solution = train[['geology_id'] + TARGETS].copy()
train_sub = train[['geology_id'] + TARGETS].copy()


class LSTMForecastModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        dropout=0.2,
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
        
        lstm_out = lstm_out[:, -1, :]
        
        lstm_out = self.layer_norm(lstm_out)
        
        x = self.fc1(lstm_out)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class GeologyDataset(Dataset):
    def __init__(self, features, targets=None, is_test=False):
        self.features = features
        self.targets = targets
        self.is_test = is_test
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        
        x = x.reshape(-1, 1)
        
        if self.is_test:
            return x
        else:
            y = self.targets[idx]
            return x, y


def preprocess_data(df, feature_cols, target_cols=None, is_test=False):
    X = df[feature_cols].values
    
    if not is_test:
        y = df[target_cols].values
        return X, y
    
    return X


def compute_nll_score(solution, submission, row_id_column_name='geology_id'):
    solution_copy = solution.copy()
    submission_copy = submission.copy()
    
    del solution_copy[row_id_column_name]
    del submission_copy[row_id_column_name]

    NEGATIVE_PART = -299
    LARGEST_CHUNK = 600
    SMALLEST_CHUNK = 350
    TOTAL_REALIZATIONS = 10
    INFLATION_SIGMA = 600
    
    sigma_2 = np.ones((LARGEST_CHUNK+NEGATIVE_PART-1))
    from_ranges = [1, 61, 245]
    to_ranges_excl = [61, 245, 301]
    log_slopes = [1.0406028049510443, 0.0, 7.835345062351012]
    log_offsets = [-6.430669850650689, -2.1617411566043896, -45.24876794412965]

    for growth_mode in range(len(from_ranges)):
        for i in range(from_ranges[growth_mode], to_ranges_excl[growth_mode]):
            sigma_2[i-1] = np.exp(np.log(i)*log_slopes[growth_mode]+log_offsets[growth_mode])

    sigma_2 *= INFLATION_SIGMA
  
    cov_matrix_inv_diag = 1. / sigma_2
    
    num_rows = solution_copy.shape[0]
    num_columns = LARGEST_CHUNK + NEGATIVE_PART - 1
    
    p = 1./TOTAL_REALIZATIONS
    log_p = np.log(p)
    
    solution_arr = np.zeros((num_rows, TOTAL_REALIZATIONS, num_columns))
    submission_arr = np.zeros((num_rows, TOTAL_REALIZATIONS, num_columns))
    
    for k in range(TOTAL_REALIZATIONS):
        for i in range(num_columns):
            if k == 0:
                column_name = str(i+1)
            else:
                column_name = f"r_{k}_pos_{i+1}"
            solution_arr[:, k, i] = solution_copy[column_name].values
            submission_arr[:, k, i] = submission_copy[column_name].values

    misfit = solution_arr - submission_arr
    inner_product_matrix = np.sum(cov_matrix_inv_diag * misfit * misfit, axis=2)
    
    nll = -logsumexp(log_p - inner_product_matrix, axis=1)
    
    return nll.mean()


def train_model_with_nll_loss(model, train_loader, optimizer, device):
    model.train()
    train_losses = []
    
    for data, target in train_loader:
        data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.float32)
        
        optimizer.zero_grad()
        output = model(data)
        
        target_mean = target.mean(dim=0)
        target_std = target.std(dim=0) + 1e-6
        
        normalized_output = (output - target_mean) / target_std
        normalized_target = (target - target_mean) / target_std
        
        loss = F.mse_loss(normalized_output, normalized_target)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_losses.append(loss.item())
    
    return np.mean(train_losses)


def validate_model(model, val_loader, device):
    model.eval()
    val_losses = []
    val_preds = []
    val_targets = []
    
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
            val_preds.append(output.cpu().numpy())
            val_targets.append(target.cpu().numpy())
    
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    
    return np.mean(val_losses), val_preds, val_targets


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
    fold_config.update({"fold": fold_idx})
    
    run = init_wandb(config=fold_config)
    
    X_num_train, X_num_val = X_num[train_index], X_num[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    train_dataset = GeologyDataset(X_num_train, y_train)
    val_dataset = GeologyDataset(X_num_val, y_val)
    test_dataset = GeologyDataset(X_num_test, is_test=True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        pin_memory=True, 
        num_workers=2  
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )
    
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
        eps=1e-8  # Increased stability
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
    
    print(f"Training fold {fold_idx + 1}...")
    for epoch in range(config['epochs']):
        train_loss = train_model_with_nll_loss(model, train_loader, optimizer, device)
        
        val_loss, val_preds, val_targets = validate_model(model, val_loader, device)
        
        val_predictions = val_preds
        
        scheduler.step()

        if run:
            run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f"model_fold_{fold_idx}.pt"
            torch.save(model.state_dict(), model_path)
            if run:
                run.save(model_path)
    
    model.load_state_dict(torch.load(f"model_fold_{fold_idx}.pt"))
    model.eval()
    test_preds = []
    
    with torch.no_grad():
        for data in test_loader:
            if isinstance(data, list):
                data = data[0]  # For test data
            data = data.to(device, dtype=torch.float32)
            output = model(data)
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
}


folds = 5
kf = KFold(n_splits=folds, random_state=SEED, shuffle=True)


X_num, y = train[FEATURES].values, train[TARGETS].values
X_num_test = test[FEATURES].values


test_preds_all_folds = np.zeros((folds, len(test), len(TARGETS)))
val_scores = []


for fold_idx, (train_index, val_index) in enumerate(kf.split(X_num)):
    test_preds = train_and_predict(
        fold_idx, 
        train_index, 
        val_index, 
        X_num, 
        y,
        X_num_test,
        config
    )
    test_preds_all_folds[fold_idx] = test_preds
    
    fold_val_preds = train_sub.loc[val_index, ['geology_id'] + TARGETS]
    fold_val_solution = solution.loc[val_index]
    
    fold_score = compute_nll_score(fold_val_solution, fold_val_preds)
    val_scores.append(fold_score)
    
    print(f"Fold {fold_idx+1} validation NLL score: {fold_score:.6f}")


avg_val_score = np.mean(val_scores)
print(f"Average validation NLL score: {avg_val_score:.6f}")


test_preds_avg = np.mean(test_preds_all_folds, axis=0)


submission = sub.copy()
submission[TARGETS] = test_preds_avg


submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")

