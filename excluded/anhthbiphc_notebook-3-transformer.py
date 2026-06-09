import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


# Load preprocessed data (replace with your dataset path if needed)
train_df = pd.read_pickle("//kaggle/input/processed-data/processed_train.pkl")
test_df = pd.read_pickle("/kaggle/input/processed-data/processed_test.pkl")

# Convert all numeric columns to float32 to save memory
numeric_cols_train = train_df.select_dtypes(include=['float64', 'int64']).columns
train_df[numeric_cols_train] = train_df[numeric_cols_train].astype('float32')

numeric_cols_test = test_df.select_dtypes(include=['float64', 'int64']).columns
test_df[numeric_cols_test] = test_df[numeric_cols_test].astype('float32')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)


print("Train columns:", train_df.columns.tolist())
print("Test columns:", test_df.columns.tolist())



# ==========================
# Setup & Utilities (OOM-safe)
# ==========================
import time, math, gc, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

# ---------- Target column ----------
TARGET_COL = "behavior"   # use correct label column in train_df

# ---------- Label encoder (train only) ----------
def build_label_encoder(train_df, target_col=TARGET_COL):
    # Ensure labels are all strings
    all_labels = pd.Index(train_df[target_col].astype(str)).unique()
    label2id = {lbl: i for i, lbl in enumerate(sorted(all_labels))}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return label2id, id2label

# Use it
label2id, id2label = build_label_encoder(train_df, TARGET_COL)
N_CLASSES = len(label2id)
print(f"Detected {N_CLASSES} classes.")


# ---------- Feature columns ----------
def get_feature_columns(train_df, test_df, target_col=TARGET_COL):
    train_feats = train_df.drop(columns=[target_col], errors='ignore').select_dtypes(include=[np.number]).columns
    test_feats  = test_df.select_dtypes(include=[np.number]).columns
    common = sorted(set(train_feats).intersection(set(test_feats)))
    if len(common) == 0:
        raise ValueError("No common numeric features between train and test.")
    return common

FEATURE_COLS = get_feature_columns(train_df, test_df, TARGET_COL)
N_FEATURES = len(FEATURE_COLS)
print(f"Using {N_FEATURES} numeric features common to train/test.")

# ==========================
# Positional Encoding (sinusoidal)
# ==========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x):  # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)




# ==========================
# Models
# ==========================
# 1) Vanilla Transformer (tokens=time steps)
class TransformerClassifier(nn.Module):
    def __init__(self, n_features, n_classes, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1, pool='mean'):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool = pool
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):  # x: (B, T, F)
        z = self.input_proj(x)
        z = self.pos(z)
        z = self.encoder(z)
        z = self.norm(z)
        if self.pool == 'cls':
            z = z[:, 0, :]
        else:
            z = z.mean(dim=1)
        return self.head(z)




# ==========================
# Training / Evaluation
# ==========================
model_name = 'vanilla_transformer'

@torch.no_grad()
def macro_f1_from_logits(logits, targets, n_classes=N_CLASSES):
    # logits: (N, C), targets: (N,)
    preds = logits.argmax(dim=1).view(-1).cpu().numpy()
    t = targets.view(-1).cpu().numpy()
    # compute macro-F1 (no sklearn dependency)
    f1s = []
    for c in range(n_classes):
        tp = np.sum((preds == c) & (t == c))
        fp = np.sum((preds == c) & (t != c))
        fn = np.sum((preds != c) & (t == c))
        if tp == 0 and (fp+fn) == 0:
            f1 = 1.0
        else:
            prec = tp / (tp + fp + 1e-12)
            rec  = tp / (tp + fn + 1e-12)
            f1 = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
        f1s.append(f1)
    return float(np.mean(f1s))

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(model, loader, optimizer, scaler, criterion, grad_accum=1):
    model.train()
    total_loss = 0.0
    n_seen = 0
    for step, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type='cuda' if DEVICE=='cuda' else 'cpu', dtype=torch.float16 if DEVICE=='cuda' else torch.bfloat16):
            logits = model(x)
            loss = criterion(logits, y) / grad_accum

        scaler.scale(loss).backward()
        if (step+1) % grad_accum == 0:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum * x.size(0)
        n_seen += x.size(0)

        # small leak guard
        if (step+1) % 200 == 0:
            torch.cuda.empty_cache() if DEVICE=='cuda' else None
    return total_loss / max(n_seen,1)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, n_seen = 0.0, 0
    all_logits, all_targets = [], []
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()*x.size(0)
        n_seen += x.size(0)
        all_logits.append(logits)
        all_targets.append(y)
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    f1 = macro_f1_from_logits(logits, targets, n_classes=N_CLASSES)
    return total_loss / max(n_seen,1), f1

import torch
from torch.utils.data import Dataset

# Ensure this class is defined first
# Make sure SequenceDataset is defined first
class SequenceDataset(Dataset):
    def __init__(self, df, window_size, stride=1, target_col=TARGET_COL, feature_cols=FEATURE_COLS, label2id=label2id):
        # Drop rows with NaN in target
        self.df = df.dropna(subset=[target_col]).reset_index(drop=True)
        self.feature_df = self.df[feature_cols]

        # Map labels safely; unknown labels become -1
        self.target = self.df[target_col].map(label2id).fillna(-1).astype(int).values

        # Remove rows where target is -1 (unknown labels)
        valid_idx = self.target != -1
        self.feature_df = self.feature_df.iloc[valid_idx].reset_index(drop=True)
        self.target = self.target[valid_idx]

        self.window = int(window_size)
        self.stride = int(stride)
        self.n = max((len(self.feature_df) - self.window), 0) // self.stride

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = idx * self.stride
        x = self.feature_df.iloc[i:i+self.window].to_numpy(dtype=np.float32)
        y = self.target[i+self.window]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

class SequenceDatasetTest(Dataset):
    def __init__(self, df, window_size, stride=1, feature_cols=FEATURE_COLS):
        self.df = df.reset_index(drop=True)
        self.feature_df = self.df[feature_cols]

        self.window = int(window_size)
        self.stride = int(stride)
        self.n = max((len(self.feature_df) - self.window), 0) // self.stride

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = idx * self.stride
        x = self.feature_df.iloc[i:i+self.window].to_numpy(dtype=np.float32)
        return torch.from_numpy(x)

# Now your run_experiment function can use SequenceDatasetTest
import os  # Make sure to import os

# ==========================
# Run Experiment (No Checkpoints)
# ==========================
def run_experiment(model_name, model_ctor, window_size=50, stride=5,
                   batch_size=64, epochs=5, lr=1e-3, grad_accum=1, num_workers=0):
    print(f"\n=== {model_name} | window={window_size} stride={stride} ===")
    
    # Initialize the dataset and dataloaders
    train_ds = SequenceDataset(train_df, window_size, stride=stride,
                               target_col=TARGET_COL, feature_cols=FEATURE_COLS, label2id=label2id)
    test_ds  = SequenceDatasetTest(test_df, window_size, stride=stride, feature_cols=FEATURE_COLS)

    if len(train_ds) == 0 or len(test_ds) == 0:
        print("Dataset too small for given window/stride. Skipping.")
        return None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # Initialize the model
    model = model_ctor().to(DEVICE)
    print(f"Params: {count_params(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(DEVICE=='cuda'))

    best_f1, history = -1.0, []
    
    # Training loop
    for ep in range(epochs):
        t1 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, grad_accum=grad_accum)
        
        val_loss, val_f1 = evaluate(model, train_loader, criterion)
        dt = time.time() - t1
        best_f1 = max(best_f1, val_f1)
        history.append({'epoch': ep+1, 'train_loss': train_loss, 'val_loss': val_loss, 'val_macro_f1': val_f1, 'sec': dt})
        print(f"Epoch {ep+1:02d} | train {train_loss:.4f} | val {val_loss:.4f} | F1 {val_f1:.4f} | {dt:.1f}s")
    
    # Return the best results and history
    result = {
        'model': model_name,
        'window': window_size,
        'stride': stride,
        'params': count_params(model),
        'best_macro_f1': round(best_f1, 6),
        'history': history
    }
    return result

# ==========================
# Model registry (easy switch)
# ==========================
def make_transformer():
    return TransformerClassifier(n_features=N_FEATURES, n_classes=N_CLASSES,
                                 d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1)

MODEL_FACTORIES = {
    'vanilla_transformer': make_transformer,}



# ==========================
# Scaling window length experiments (OOM-safe defaults)
# ==========================
WINDOWS = [50]
STRIDE = 5           # reduce overlap to shrink dataset size ~5x
BATCH = 64           # keep small to avoid OOM
EPOCHS = 3           # adjust as you like
LR = 1e-3
GRAD_ACCUM = 2       # set >1 if you must keep BATCH even smaller
WORKERS = 0          # Kaggle often stable with 0 or 2; avoid big numbers on large DataFrames

import pandas as pd
import gc
import time

# Only run Vanilla Transformer
MODELS_TO_RUN = {'vanilla_transformer': MODEL_FACTORIES['vanilla_transformer']}

summary_results = []

for w in WINDOWS:
    for model_name, ctor in MODELS_TO_RUN.items():
        # Measure start time for total train
        start_total = time.time()
        out = run_experiment(
            model_name=model_name,
            model_ctor=ctor,
            window_size=w,
            stride=STRIDE,
            batch_size=BATCH,
            epochs=EPOCHS,
            lr=LR,
            grad_accum=GRAD_ACCUM,
            num_workers=WORKERS
        )
        total_time = time.time() - start_total

        if out is not None:
            # Add inference time measurement (single forward pass)
            model = ctor().to(DEVICE)
            x_sample = next(iter(DataLoader(SequenceDataset(train_df, w, stride=STRIDE,
                                                             target_col=TARGET_COL, 
                                                             feature_cols=FEATURE_COLS, 
                                                             label2id=label2id),
                                             batch_size=1)))
            x_input, _ = x_sample
            x_input = x_input.to(DEVICE)
            torch.cuda.synchronize() if DEVICE=='cuda' else None
            start_infer = time.time()
            _ = model(x_input)
            torch.cuda.synchronize() if DEVICE=='cuda' else None
            inference_time = time.time() - start_infer

            summary_results.append({
                'model_name': out['model'],
                'runtime_id': f"{out['model']}_w{out['window']}_s{out['stride']}",
                'notebook': 'Notebook 3',
                'features_used': 'All',
                'window_size': out['window'],
                'optimizer/solver': 'AdamW',
                'params (M)': out['params'] / 1e6,
                'Binary': 'N/A',
                'Macro': out['best_macro_f1'],
                'Final Score': out['best_macro_f1'],
                'val_acc': out.get('val_acc', None),
                'train_time': total_time,
                'inference_time': inference_time
            })

        # free memory aggressively
        torch.cuda.empty_cache() if DEVICE=='cuda' else None
        gc.collect()

# Create DataFrame
final_summary_df = pd.DataFrame(summary_results)
final_summary_df = final_summary_df.sort_values(['window_size','model_name']).reset_index(drop=True)

# Display final table
pd.set_option('display.max_columns', None)
print("\n=== Final Model Summary ===")
display(final_summary_df)


