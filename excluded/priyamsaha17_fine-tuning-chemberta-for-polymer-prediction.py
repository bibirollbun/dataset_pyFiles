!pip install /kaggle/input/installations-polymer-prediction/rdkit-2025.3.6-cp311-cp311-manylinux_2_28_x86_64.whl


! pip install /kaggle/input/installations-polymer-prediction/huggingface_hub-0.34.5-py3-none-any.whl


! pip install /kaggle/input/installations-polymer-prediction/tokenizers-0.22.0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


!pip install /kaggle/input/installations-polymer-prediction/transformers-4.56.1-py3-none-any.whl


import pandas as pd
import numpy as np

# Competition data
train_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")

# Supplemental datasets
dataset1 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv").rename(columns={"TC_mean": "Tc"})
dataset3 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv")
dataset4 = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv")

# Target list (desired order)
all_targets = ["Tg", "FFV", "Tc", "Density", "Rg"]

# Ensure all targets exist in each dataset
for df in [train_df, dataset1, dataset3, dataset4]:
    for col in all_targets:
        if col not in df.columns:
            df[col] = np.nan

# Combine datasets
train_df = pd.concat([train_df, dataset1, dataset3, dataset4], ignore_index=True)

# Reorder columns: keep SMILES first, then targets
cols = ["SMILES"] + all_targets
train_df = train_df[cols]

# Shuffle and reset index
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)


train_df.head()


train_df.shape


from sklearn.model_selection import train_test_split

# 90% train, 10% validation
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)


train_df.shape


val_df.shape


import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# HuggingFace ChemBERTa tokenizer
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/installations-polymer-prediction/chemberta_tokenizer")

class PolymerDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.targets = ["Tg", "FFV", "Tc", "Density", "Rg"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row["SMILES"]

        # Encode SMILES
        inputs = self.tokenizer(
            smiles,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Extract targets (can be NaN)
        y = torch.tensor(
            row[self.targets].values.astype(float),
            dtype=torch.float
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": y
        }


train_dataset = PolymerDataset(train_df, tokenizer)
val_dataset   = PolymerDataset(val_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=128, shuffle=False)


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# -------------------------
# Helper modules
# -------------------------

class AttentionPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, hidden_states, attention_mask):
        # hidden_states: (B, L, H), attention_mask: (B, L)
        scores = self.attn(hidden_states).squeeze(-1)   # (B, L)
        pad_mask = (attention_mask == 0)

        # Use dtype-appropriate minimum to avoid overflow in float16
        min_val = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(pad_mask, min_val)

        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, L, 1)
        weights = weights.to(hidden_states.dtype)
        pooled = (hidden_states * weights).sum(dim=1)          # (B, H)
        return pooled



class SEBlock(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        mid = max(4, dim // reduction)
        self.fc1 = nn.Linear(dim, mid)
        self.fc2 = nn.Linear(mid, dim)

    def forward(self, x):
        # x: (B, H)
        # apply squeeze-and-excitation on the feature vector directly
        # se: (B, H) -> gating values in (0,1)
        se = torch.relu(self.fc1(x))      # (B, mid)
        se = torch.sigmoid(self.fc2(se))  # (B, H)
        return x * se                      # broadcast multiply (B, H)


class ResidualMLP(nn.Module):
    def __init__(self, hidden_size, hidden_mult=0.5, dropout=0.2):
        super().__init__()
        mid = max(8, int(hidden_size * hidden_mult))
        self.fc1 = nn.Linear(hidden_size, mid)
        self.fc2 = nn.Linear(mid, hidden_size)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        return self.norm(x + out)

# -------------------------
# ChemBERTaRegressor (refactored name)
# -------------------------
class ChemBERTaRegressor(nn.Module):
    def __init__(
        self,
        model_name="/kaggle/input/installations-polymer-prediction/chemberta_model",
        num_targets=5,
        pooling="attn",          # "cls", "mean", or "attn"
        hidden_mult=0.5,
        dropout=0.2,
        use_se=True,
        use_residual_head=True,
        per_target_heads=False,
        use_uncertainty=False
    ):
        super().__init__()
        # load backbone from local path or hub (if internet available)
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.pooling = pooling.lower()
        self.use_se = use_se
        self.use_residual_head = use_residual_head
        self.per_target_heads = per_target_heads
        self.use_uncertainty = use_uncertainty

        if self.pooling == "attn":
            self.attn_pool = AttentionPool(hidden_size)

        if self.use_se:
            self.se = SEBlock(hidden_size, reduction=8)

        if self.use_residual_head:
            self.res_mlp = ResidualMLP(hidden_size, hidden_mult, dropout)

        if self.per_target_heads:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, max(16, int(hidden_size * hidden_mult))),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(max(16, int(hidden_size * hidden_mult)), 1)
                ) for _ in range(num_targets)
            ])
        else:
            head_mid = max(64, int(hidden_size * hidden_mult))
            self.head = nn.Sequential(
                nn.Linear(hidden_size, head_mid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(head_mid),
                nn.Linear(head_mid, max(32, head_mid // 2)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(32, head_mid // 2), num_targets)
            )

        if self.use_uncertainty:
            # learnable log variance per target
            self.log_vars = nn.Parameter(torch.zeros(num_targets))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (B, L, H)

        if self.pooling == "cls":
            pooled = last_hidden[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # (B, L, 1)
            summed = (last_hidden * mask).sum(1)
            denom = mask.sum(1).clamp(min=1e-9)
            pooled = summed / denom
        elif self.pooling == "attn":
            pooled = self.attn_pool(last_hidden, attention_mask)
        else:
            raise ValueError("pooling must be 'cls', 'mean', or 'attn'")

        if self.use_se:
            pooled = self.se(pooled)

        if self.use_residual_head:
            pooled = self.res_mlp(pooled)

        if self.per_target_heads:
            outs = [head(pooled).squeeze(-1) for head in self.heads]  # list of (B,)
            preds = torch.stack(outs, dim=1)                         # (B, T)
        else:
            preds = self.head(pooled)  # (B, T)

        extra = {}
        if self.use_uncertainty:
            extra["log_vars"] = self.log_vars
        return preds, extra


# -------------------------
# Masked MAE with optional uncertainty weighting
# -------------------------
def masked_mae_with_uncertainty(preds, targets, log_vars=None):
    """
    preds: (B, T)
    targets: (B, T) with NaNs for missing
    log_vars: None or parameter tensor shape (T,)
    Returns: scalar loss (torch.Tensor)
    """
    device = preds.device
    mask = ~torch.isnan(targets)             # (B, T)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    abs_diff = torch.abs(preds - torch.where(mask, targets, torch.zeros_like(targets)))
    per_target_sum = (abs_diff * mask.float()).sum(dim=0)        # (T,)
    per_target_count = mask.float().sum(dim=0).clamp(min=1.0)    # avoid div0
    per_target_mae = per_target_sum / per_target_count          # (T,)

    if log_vars is None:
        loss = per_target_mae.mean()
        return loss
    else:
        s = log_vars
        # uncertainty weighting: exp(-s) * mae + s  (then averaged)
        weighted = torch.exp(-s) * per_target_mae + s
        loss = 0.5 * weighted.mean()
        return loss


import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.amp

# -------------------------
# Training loop (fixed)
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"  # use AMP only on CUDA

# instantiate model (point model_name to your local path if needed)
model = ChemBERTaRegressor(
    model_name="/kaggle/input/installations-polymer-prediction/chemberta_model",
    num_targets=5,
    pooling="attn",
    use_se=True,
    use_residual_head=True,
    per_target_heads=False,
    use_uncertainty=True
).to(device)

# Layer-wise optimizer: small LR for backbone, larger for head(s)
backbone_params = list(model.bert.parameters())
head_params = [p for n, p in model.named_parameters() if "bert" not in n]

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 1e-5},
    {"params": head_params,     "lr": 5e-4}
], weight_decay=1e-2)

# Scheduler that reduces LR when val loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    verbose=True
)

# AMP scaler (conditional)
scaler = torch.amp.GradScaler() if use_amp else None

# Training hyperparams
EPOCHS = 100
patience = 7
min_delta = 1e-3   # minimum absolute improvement to reset early stopping
best_val = np.inf
epochs_no_improve = 0

train_losses = []
val_losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    iters = 0
    train_loop = tqdm(train_loader, desc=f"Train Epoch {epoch}/{EPOCHS}", leave=False)
    for batch in train_loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)  # may contain NaNs

        # forward (with/without AMP)
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = out[0] if isinstance(out, (tuple, list)) else out
                extra = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else {}
                log_vars = extra.get("log_vars") if isinstance(extra, dict) else (extra.log_vars if hasattr(extra, "log_vars") else None)
                loss = masked_mae_with_uncertainty(preds, labels, log_vars)
            # backward + step using scaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = out[0] if isinstance(out, (tuple, list)) else out
            extra = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else {}
            log_vars = extra.get("log_vars") if isinstance(extra, dict) else (extra.log_vars if hasattr(extra, "log_vars") else None)
            loss = masked_mae_with_uncertainty(preds, labels, log_vars)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        running_loss += loss.item()
        iters += 1
        train_loop.set_postfix(train_loss=loss.item())

    avg_train_loss = running_loss / max(1, iters)
    train_losses.append(avg_train_loss)

    # ---- Validation ----
    model.eval()
    val_running = 0.0
    val_iters = 0
    val_loop = tqdm(val_loader, desc=f"Val Epoch {epoch}/{EPOCHS}", leave=False)
    with torch.no_grad():
        for batch in val_loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = out[0] if isinstance(out, (tuple, list)) else out
                    extra = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else {}
                    log_vars = extra.get("log_vars") if isinstance(extra, dict) else (extra.log_vars if hasattr(extra, "log_vars") else None)
                    vloss = masked_mae_with_uncertainty(preds, labels, log_vars)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = out[0] if isinstance(out, (tuple, list)) else out
                extra = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else {}
                log_vars = extra.get("log_vars") if isinstance(extra, dict) else (extra.log_vars if hasattr(extra, "log_vars") else None)
                vloss = masked_mae_with_uncertainty(preds, labels, log_vars)

            val_running += vloss.item()
            val_iters += 1
            val_loop.set_postfix(val_loss=vloss.item())

    avg_val_loss = val_running / max(1, val_iters)
    val_losses.append(avg_val_loss)

    # Scheduler step on validation
    scheduler.step(avg_val_loss)

    # Print LR of each param group for clarity
    lrs = [pg['lr'] for pg in optimizer.param_groups]
    print(f"Epoch {epoch} | Train MAE: {avg_train_loss:.6f} | Val MAE: {avg_val_loss:.6f} | LR groups: {lrs}")

    # Early stopping & checkpointing
    if best_val - avg_val_loss > min_delta:
        best_val = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_chemberta_model.pt")
        print(f"  Saved best model (val {best_val:.6f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

# Plot loss curves
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.title("Train & Val MAE")
plt.show()


# Inference / submission cell — KEEP IDS, NO SHUFFLE
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# --- load files (if not already loaded) ---
test_path = "/kaggle/input/neurips-open-polymer-prediction-2025/test.csv"
sample_sub_path = "/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv"

test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

# --- Ensure we have 'id' and reorder to sample_sub exactly (no shuffling) ---
if "id" not in test_df.columns:
    raise ValueError("test_df must contain an 'id' column for inference. "
                     "During training it's OK to drop IDs, but keep them in test.")
# Reindex test_df to match sample_submission order (so row order = required submission order)
# This will also insert any missing IDs as NaN rows (which will raise later)
test_df = test_df.set_index("id").reindex(sample_sub["id"]).reset_index()

# --- Add dummy target columns (so dataset encoding doesn't break) ---
target_cols = ["Tg", "FFV", "Tc", "Density", "Rg"]
for col in target_cols:
    if col not in test_df.columns:
        test_df[col] = np.nan   # use numpy.nan

# --- Build dataset and dataloader WITHOUT shuffling ---
test_dataset = PolymerDataset(test_df, tokenizer)     # your Dataset should read SMILES and return input tensors
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # NO shuffle here!

# --- Run inference ---
model.eval()
preds_list = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = out[0] if isinstance(out, (tuple, list)) else out

        preds_list.append(preds.detach().cpu())

# concatenate predictions
preds_all = torch.cat(preds_list, dim=0).numpy()  # shape should be (N_rows, n_targets) or (N_rows,)

# normalize to 2D (N, T)
if preds_all.ndim == 1:
    preds_all = preds_all[:, None]

n_rows_pred, n_targets_pred = preds_all.shape
expected_target_cols = list(sample_sub.columns[1:])
n_expected = len(expected_target_cols)

# --- Safeguard: row count must match sample submission ---
if n_rows_pred != sample_sub.shape[0]:
    raise RuntimeError(f"Row mismatch: model produced {n_rows_pred} rows but sample_submission requires {sample_sub.shape[0]}. "
                       "Ensure test_df contains all ids and you did NOT shuffle the DataLoader.")

# --- Align target count: trim or pad if needed ---
if n_targets_pred != n_expected:
    if n_targets_pred > n_expected:
        print(f"Warning: model produced {n_targets_pred} targets but submission expects {n_expected}. Trimming extra columns.")
        preds_all = preds_all[:, :n_expected]
    else:
        print(f"Warning: model produced {n_targets_pred} targets but submission expects {n_expected}. Padding with zeros.")
        pad = np.zeros((n_rows_pred, n_expected - n_targets_pred), dtype=preds_all.dtype)
        preds_all = np.concatenate([preds_all, pad], axis=1)

# --- sanitize numeric issues (NaN/Inf) ---
if not np.isfinite(preds_all).all():
    print("Warning: non-finite values found in predictions — replacing with 0.0")
    preds_all = np.nan_to_num(preds_all, nan=0.0, posinf=0.0, neginf=0.0)

# --- Overwrite test_df IN-PLACE to match submission format ---
# drop extra columns that are not part of sample_sub (but keep 'id')
cols_to_drop = [c for c in test_df.columns if c not in sample_sub.columns]
if cols_to_drop:
    test_df.drop(columns=cols_to_drop, inplace=True)

# assign predictions to the target columns in-place
for i, col in enumerate(expected_target_cols):
    test_df[col] = preds_all[:, i]

# reorder columns exactly as sample_sub
test_df = test_df[sample_sub.columns]

# final safety checks
assert list(test_df.columns) == list(sample_sub.columns), "Column names/order mismatch with sample_submission!"
assert test_df.shape == sample_sub.shape, "Shape mismatch with sample_submission!"
assert test_df["id"].equals(sample_sub["id"]), "IDs/order do not match sample_submission (shouldn't happen after reindex)."

# save submission
test_df.to_csv("submission.csv", index=False)
print("✅ Submission saved to submission.csv — shape:", test_df.shape)
print(test_df.head())

