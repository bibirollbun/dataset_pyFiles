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
        pass
        # print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import glob
import uuid
from tqdm import tqdm

np.random.seed(2025)

OUTPUT_FILE = "/kaggle/working/synthetic_train.csv"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def generate_realizations(base, num=10, noise_level=0.5):
    base = base.copy()
    realizations = [base]
    for _ in range(1, num):
        noise = np.random.randn(len(base)) * noise_level
        perturb = np.convolve(noise, np.ones(20)/20, mode='same')  # Smooth noise
        perturbed = base + perturb
        realizations.append(perturbed)
    return np.array(realizations)

def process_file(path, window_size=600, min_valid=610, max_chunks=20):
    df = pd.read_csv(path)
    x_raw = df['VS_APPROX_adjusted'].values
    z_raw = df['HORIZON_Z_adjusted'].values

    # Interpolate on 1-foot grid
    x_new = np.arange(0, x_raw.max(), 1.0)
    f_interp = interp1d(x_raw, z_raw, kind='linear', bounds_error=False, fill_value="extrapolate")
    z_new = f_interp(x_new)

    rows = []
    if len(z_new) < min_valid:
        return rows

    num_chunks = 0
    attempts = 0
    while num_chunks < max_chunks: #  and attempts < 50
        start = np.random.randint(0, len(z_new) - window_size)
        chunk = z_new[start:start + window_size].copy()
        chunk -= chunk[299]  # normalize so Z(0)=0

        # Simulate drilling by hiding part of left context
        if np.random.rand() > 0.6:
            hide_up_to = np.random.randint(0, 250)
        else:
            hide_up_to = 0
        chunk_with_nans = chunk.copy()
        chunk_with_nans[:hide_up_to] = np.nan

        realizations = generate_realizations(chunk[300:], num=1) # 10
        output = {
            'geology_id': str(uuid.uuid4()),
        }
        
        for i in range(300):
            output[str(i - 299)] = chunk_with_nans[i]
        for i in range(300):
            output[str(i + 1)] = realizations[0, i]
        
        # for r in range(1, 10):
        #     for i in range(300):
        #         output[f"r_{r}_pos_{i+1}"] = realizations[r, i]

        rows.append(output)
        num_chunks += 1
        attempts += 1

    return rows

def main():
    all_files = glob.glob("/kaggle/input/geology-forecast-challenge-open/data/train_raw/*.csv")
    all_data = []
    for f in tqdm(all_files):
        all_data.extend(process_file(f, max_chunks=38190//len(all_files)+1))
    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"âœ… Saved {len(df)} synthetic samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import random


import torch.nn.functional as F
from scipy.special import logsumexp

import wandb
from google.colab import userdata

import warnings


# ğŸ”¹ Reproducibility
def seed_everything(seed=2025):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


synth_train_df = pd.read_csv("/kaggle/working/synthetic_train.csv")
synth_train_df = synth_train_df.iloc[:38190,:]
synth_train_df.shape


# 2ï¸�âƒ£ Load Data
train_df = pd.read_csv("/kaggle/input/geology-forecast-challenge-open/data/train.csv")
train_df = train_df.iloc[:,:601]
train_df = pd.concat([train_df, synth_train_df], ignore_index=True)
test_df = pd.read_csv("/kaggle/input/geology-forecast-challenge-open/data/test.csv")
submission_template = pd.read_csv('/kaggle/input/geology-forecast-challenge-open/data/sample_submission.csv')

FEATURES = [col for col in test_df.columns if col != "geology_id"]
REALIZATIONS = [col for col in submission_template.columns if col != "geology_id"]
NUM_REALIZATIONS = len(REALIZATIONS) // 300

print(f"ğŸ”� Detected {NUM_REALIZATIONS} realizations.")


columns_new = ['geology_id']
for r in range(NUM_REALIZATIONS):
    for i in range(-299,301):
        columns_new.append(f'r_{r}_pos_{i}')

train_df_new = pd.DataFrame(
    data=np.concatenate((np.full((train_df.shape[0]//10,1), 'none'), train_df.iloc[:,1:].values.reshape((-1,10*600))), axis=1),
    columns=columns_new
)
train_df_new.shape


train_FEATURES = []
for r in range(NUM_REALIZATIONS):
    for i in range(-299,1):
        train_FEATURES.append(f'r_{r}_pos_{i}')

train_REALIZATIONS = []
for r in range(NUM_REALIZATIONS):
    for i in range(1,301):
        train_REALIZATIONS.append(f'r_{r}_pos_{i}')


train_df = train_df_new
train_df = train_df.iloc[:,1:]
train_df = train_df.astype('float64')
train_df.insert(0, 'geology_id', 'none')
train_df.shape


# ğŸ”§ Replacing nans
train_df.iloc[:, 1:] = train_df.iloc[:, 1:].fillna(0)
test_df.iloc[:, 1:] = test_df.iloc[:, 1:].fillna(0)


class GeologyDataset(Dataset):
    def __init__(self, features, targets=None, realization_ids=None, is_test=False):
        self.features = features
        self.targets = targets
        self.realization_ids = realization_ids
        self.is_test = is_test
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx].reshape(-1, 1)  # [50, 1]

        if self.is_test:
            return x
        else:
            y = self.targets[idx]  # [300]
            rid = self.realization_ids[idx]  # scalar
            return x, y, rid


# ğŸ§  Model
class ParallelLSTMWithAttention(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        dropout=0.2,
        num_realizations=10,
        realization_emb_dim=16,
        use_multihead=True,
        num_heads=4,
        fusion_method='concat'  # options: 'concat', 'add', 'gated'
    ):
        super().__init__()
        self.realization_embedding = nn.Embedding(num_realizations, realization_emb_dim)
        self.fusion_method = fusion_method
        self.hidden_size = hidden_size
        # LSTM Branch
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            ) for i in range(num_layers)
        ])
        self.lstm_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        # Attention Branch
        self.attn_input = nn.Linear(input_size, hidden_size)
        if use_multihead:
            self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        else:
            self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, batch_first=True)
        # Learnable Basis Projection Branch
        self.num_bases = 16
        self.basis_proj = nn.Linear(input_size, self.num_bases)
        self.basis_decoder = nn.Linear(self.num_bases, hidden_size)
        # Fusion
        fusion_input_size = hidden_size * 3 if fusion_method == 'concat' else hidden_size
        if fusion_method == 'gated':
            self.gate_fc = nn.Linear(hidden_size * 2, hidden_size)
            self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(fusion_input_size + realization_emb_dim, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x, realization_ids):
        # LSTM branch
        lstm_out = x
        for lstm, norm in zip(self.lstm_layers, self.lstm_norms):
            residual = lstm_out
            lstm_out, _ = lstm(lstm_out)
            lstm_out = norm(lstm_out + residual)
        lstm_out = lstm_out[:, -1, :]  # Last step
        # Attention branch
        attn_input = self.attn_input(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input)
        attn_out = attn_out.max(dim=1).values
        # Learnable Basis Projection branch
        basis_coeffs = self.basis_proj(x).max(dim=1).values  # [B, num_bases]
        basis_out = self.basis_decoder(basis_coeffs)   # [B, H]
        # Fusion
        if self.fusion_method == 'add':
            fused = lstm_out + attn_out + basis_out
        elif self.fusion_method == 'gated':
            gate_attn = self.sigmoid(self.gate_fc(torch.cat([lstm_out, attn_out], dim=1)))
            fused_attn = gate_attn * lstm_out + (1 - gate_attn) * attn_out
            fused = fused_attn + basis_out
        else:  # concat
            fused = torch.cat([lstm_out, attn_out, basis_out], dim=1)
        realization_emb = self.realization_embedding(realization_ids)
        combined = torch.cat([fused, realization_emb], dim=1)
        x = self.fc1(combined)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


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
    
    for data, target, rid in train_loader:
        data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        rid = rid.to(device)
        output = model(data, rid)
        
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
        for data, target, rid in val_loader:
            data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.float32)
            rid = rid.to(device)
            output = model(data, rid)
            
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


def mixup(data, targets, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = np.random.permutation(len(data))
    return lam * data + (1 - lam) * data[index], lam * targets + (1 - lam) * targets[index]


def train_and_predict(
    fold_idx, 
    train_index, 
    val_index, 
    X_num, 
    y,
    X_num_test,
    config,
    train_sub=None
):
    fold_config = config.copy()
    fold_config.update({"fold": fold_idx})
    
    run = init_wandb(config=fold_config)
    
    # Train data (EXPANDED across 10 realizations)
    X_num_train = X_num[train_index].reshape((-1,300))
    y_train = y[train_index].reshape((-1,300))
    realization_ids_train = np.tile(np.arange(NUM_REALIZATIONS), len(train_index))
    
    # Validation data (USE ONLY REALIZATION 0)
    X_num_val = X_num[val_index][:, :300]
    y_val = y[val_index][:, :300]
    realization_ids_val = np.zeros(len(X_num_val), dtype=int)

    X_num_train, y_train = mixup(X_num_train, y_train, 0.4)
    
    train_dataset = GeologyDataset(X_num_train, y_train, realization_ids_train)
    val_dataset = GeologyDataset(X_num_val, y_val, realization_ids_val)
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

    model = ParallelLSTMWithAttention(
        input_size=1,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=300,
        dropout=config['dropout'],
        num_realizations=NUM_REALIZATIONS,
        realization_emb_dim=16,
        use_multihead=True,
        num_heads=2,
        fusion_method='concat'  # 'concat' or 'add', 'gated'
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
    val_predictions = np.zeros((len(val_index), len(REALIZATIONS)))
    test_predictions = np.zeros((len(X_num_test), len(REALIZATIONS)))

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
            if isinstance(data, list) or isinstance(data, tuple):
                data = data[0]
            data = data.to(device, dtype=torch.float32)
    
            outputs = []
            for r_id in range(NUM_REALIZATIONS):
                realization_ids = torch.full((data.size(0),), r_id, dtype=torch.long, device=device)
                preds = model(data, realization_ids)  # [B, 300]
                outputs.append(preds.cpu().numpy())   # Append [B, 300]
    
            # Stack into shape: [B, 10, 300] â†’ then reshape to [B, 3000]
            outputs = np.stack(outputs, axis=1).reshape(data.size(0), 300 * NUM_REALIZATIONS)
            test_preds.append(outputs)
    
    test_predictions = np.concatenate(test_preds)

    train_sub.loc[val_index, REALIZATIONS[:300]] = val_predictions
    
    if run:
        run.finish()
    
    return test_predictions, train_sub


# SEED = 2025
# config = {
#     'model_type': 'LSTM',
#     'hidden_size': 1024,
#     'num_layers': 3,
#     'dropout': 0.2,
#     'learning_rate': 5e-4,
#     'weight_decay': 1e-5,
#     'batch_size': 256,
#     'epochs': 30,
#     'seed': SEED,
# }

# train_sub = train_df[['geology_id'] + train_REALIZATIONS].copy()
# solution = train_df[['geology_id'] + train_REALIZATIONS].copy()

# folds = 5
# kf = KFold(n_splits=folds, random_state=SEED, shuffle=True)

# X_num, y = train_df[train_FEATURES].values, train_df[train_REALIZATIONS].values # NOTE: We take one realization only train_df.iloc[:,301:601].values
# X_num_test = test_df[FEATURES].values

# test_preds_all_folds = np.zeros((folds, len(test_df), len(REALIZATIONS)))
# val_scores = []

# for fold_idx, (train_index, val_index) in enumerate(kf.split(X_num)):
#     test_preds, train_sub = train_and_predict(
#         fold_idx, 
#         train_index, 
#         val_index, 
#         X_num, 
#         y,
#         X_num_test,
#         config,
#         train_sub
#     )
#     test_preds_all_folds[fold_idx] = test_preds
    
#     fold_val_preds = train_sub.loc[val_index, ['geology_id'] + train_REALIZATIONS]
#     fold_val_solution = solution.loc[val_index]
    
#     fold_score = compute_nll_score(fold_val_solution, fold_val_preds)
#     val_scores.append(fold_score)
    
#     print(f"Fold {fold_idx+1} validation NLL score: {fold_score:.6f}")


# avg_val_score = np.mean(val_scores)
# print(f"Average validation NLL score: {avg_val_score:.6f}")


# # ğŸ’¾ Save Submission
# test_preds_avg = np.mean(test_preds_all_folds, axis=0)
# submission = submission_template.copy()
# submission[REALIZATIONS] = test_preds_avg
# submission.to_csv('submission.csv', index=False)
# print("âœ… Submission saved as submission.csv")


SEED = 2025
config = {
    'model_type': 'LSTM',
    'hidden_size': 1024,
    'num_layers': 3,
    'dropout': 0.2,
    'learning_rate': 5e-4,
    'weight_decay': 1e-5,
    'batch_size': 256,
    'epochs': 30,
    'seed': SEED,
}
FOLDS = 5

X_num_test = test_df[FEATURES].values
test_dataset = GeologyDataset(X_num_test, is_test=True)
test_loader = DataLoader(
    test_dataset, 
    batch_size=config['batch_size'], 
    shuffle=False,
    pin_memory=True,
    num_workers=2
)

weights_path = "/kaggle/input/geologyforecast-model-1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_preds_all_folds = np.zeros((FOLDS, len(test_df), len(REALIZATIONS)))
for fold in range(FOLDS):
    model = ParallelLSTMWithAttention(
        input_size=1,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=300,
        dropout=config['dropout'],
        num_realizations=NUM_REALIZATIONS,
        realization_emb_dim=16,
        use_multihead=True,
        num_heads=2,
        fusion_method='concat'  # 'concat' or 'add', 'gated'
    ).to(device)
    model.load_state_dict(
        torch.load(f"{weights_path}/model_fold_{fold}.pt", map_location=device)
    )
    model.eval()
    test_predictions = np.zeros((len(X_num_test), len(REALIZATIONS)))
    test_preds = []    
    with torch.no_grad():
        for data in test_loader:
            if isinstance(data, list) or isinstance(data, tuple):
                data = data[0]
            data = data.to(device, dtype=torch.float32)
    
            outputs = []
            for r_id in range(NUM_REALIZATIONS):
                realization_ids = torch.full((data.size(0),), r_id, dtype=torch.long, device=device)
                preds = model(data, realization_ids)  # [B, 300]
                outputs.append(preds.cpu().numpy())   # Append [B, 300]
    
            # Stack into shape: [B, 10, 300] â†’ then reshape to [B, 3000]
            outputs = np.stack(outputs, axis=1).reshape(data.size(0), 300 * NUM_REALIZATIONS)
            test_preds.append(outputs)
    
    test_predictions = np.concatenate(test_preds)
    test_preds_all_folds[fold] = test_predictions

test_preds_avg = np.mean(test_preds_all_folds, axis=0)
submission = submission_template.copy()
submission[REALIZATIONS] = test_preds_avg
submission.to_csv('submission.csv', index=False)
print("âœ… Submission saved as submission.csv")


submission.head()

