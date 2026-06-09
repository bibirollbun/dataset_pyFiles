import os
from pathlib import Path

def find_dataset_dir(required=("train.csv", "test.csv")) -> Path:
    """Locate the dataset directory in Kaggle or locally, with helpful diagnostics."""
    # 0) Env var overrides
    env_dir = os.getenv("DATASET_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir() and all((p/f).exists() for f in required):
            print(f"ğŸ”§ Using DATASET_DIR override: {p}")
            return p
        else:
            print(f"âš ï¸� DATASET_DIR set but invalid: {p}")

    env_name = os.getenv("DATASET_NAME")
    kaggle_input = Path("/kaggle/input")
    if env_name and kaggle_input.exists():
        cand = kaggle_input / env_name
        if cand.is_dir() and all((cand/f).exists() for f in required):
            print(f"ğŸ”§ Using DATASET_NAME under /kaggle/input: {cand}")
            return cand
        else:
            print(f"âš ï¸� DATASET_NAME set but not found or missing files: {cand}")

    candidates = []

    # 1) Exact files at /kaggle/input (rare, but check)
    if kaggle_input.exists():
        if all((kaggle_input/f).exists() for f in required):
            candidates.append(kaggle_input)

        # 2) Any subdir of /kaggle/input containing required files (non-recursive, then recursive)
        for d in kaggle_input.iterdir():
            if d.is_dir() and all((d/f).exists() for f in required):
                candidates.append(d)

        # 3) Recursive search as a last resort (handles nested or renamed datasets)
        if not candidates:
            trains = list(kaggle_input.rglob(required[0]))
            tests  = set(p.parent for p in kaggle_input.rglob(required[1]))
            for t in trains:
                if t.parent in tests:
                    candidates.append(t.parent)

    # 4) Local working directory fallback
    cwd = Path(".").resolve()
    if all((cwd/f).exists() for f in required):
        candidates.append(cwd)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in candidates:
        if str(c) not in seen:
            seen.add(str(c))
            unique.append(c)

    if not unique:
        # Helpful diagnostics
        print("â�Œ Could not find a folder containing both train.csv and test.csv.")
        if kaggle_input.exists():
            print("ğŸ“� Top-level entries in /kaggle/input:")
            for d in kaggle_input.iterdir():
                print("  -", d.name)
        print("ğŸ”� Searched:", kaggle_input if kaggle_input.exists() else "(no /kaggle/input)")
        print("ğŸ”� Also checked working dir:", cwd)
        raise FileNotFoundError("Could not find dataset directory containing train.csv and test.csv")

    # Pick the most recently modified candidate
    chosen = max(unique, key=lambda p: p.stat().st_mtime if p.exists() else 0)
    # Print a short diagnostic list
    print("âœ… Candidates that contain the required files:")
    for c in unique:
        marker = " (chosen)" if c == chosen else ""
        print(f"  - {c}{marker}")
    return chosen

# Usage:
dataset_dir = find_dataset_dir()
print(f"ğŸ“‚ Using dataset directory: {dataset_dir}")

train_path = dataset_dir / "train.csv"
test_path  = dataset_dir / "test.csv"


# --- Robust tabular -> PyTorch regressor (auto target = right-most col) ---

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Assumes train_path and test_path are already defined by your find_dataset_dir()
# ---------------------------------------------------------------------------

# --- Load ---
df = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# --- Identify columns ---
id_col = df.columns[0]
target_col = df.columns[-1]
print(f"ğŸªª ID column: {id_col}")
print(f"ğŸ�¯ Target column: {target_col}")

# --- Target cleaning: drop rows with NaN/inf in target ---
y_raw = df[target_col].replace([np.inf, -np.inf], np.nan)
keep = y_raw.notna()
dropped = (~keep).sum()
if dropped:
    print(f"âš ï¸� Dropping {dropped} training rows due to invalid target values")
df = df.loc[keep].reset_index(drop=True)

# --- Split features/target ---
X_df = df.drop(columns=[id_col, target_col])
y = df[target_col].astype(float)

# --- Column types ---
cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X_df.select_dtypes(include=[np.number, "bool"]).columns.tolist()

# --- Coerce numerics and scrub infs (so the imputer can work) ---
for c in num_cols:
    X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    X_df[c] = X_df[c].replace([np.inf, -np.inf], np.nan)

# --- Train/val split BEFORE fitting preprocessors (avoid leakage) ---
X_train_df, X_val_df, y_train, y_val = train_test_split(
    X_df, y, test_size=0.2, random_state=42
)

# --- Categorical encoding with 'unknown' handling for val/test ---
label_encoders = {}

def _fit_transform_cat(train_s: pd.Series, val_s: pd.Series, test_s: Optional[pd.Series]):
    le = LabelEncoder()
    train_f = train_s.astype(str).fillna("missing")
    le.fit(train_f.unique())

    # ensure 'unknown' exists (for unseen categories)
    if "unknown" not in le.classes_:
        le.classes_ = np.append(le.classes_, "unknown")

    def _transform(s: pd.Series):
        s2 = s.astype(str).fillna("missing")
        mask = ~np.isin(s2, le.classes_)
        if mask.any():
            s2.loc[mask] = "unknown"
        return le.transform(s2)

    train_t = _transform(train_s)
    val_t   = _transform(val_s)
    test_t  = _transform(test_s) if test_s is not None else None
    return le, train_t, val_t, test_t

X_train_cat, X_val_cat, X_test_cat = [], [], []
for c in cat_cols:
    test_series = df_test[c] if c in df_test.columns else None
    le, tr, va, te = _fit_transform_cat(X_train_df[c], X_val_df[c], test_series)
    label_encoders[c] = le
    X_train_cat.append(tr.reshape(-1, 1))
    X_val_cat.append(va.reshape(-1, 1))
    if te is not None:
        X_test_cat.append(te.reshape(-1, 1))

# --- Numeric: Impute (median) then scale (fit on train only) ---
num_imputer = SimpleImputer(strategy="median")
X_train_num = num_imputer.fit_transform(X_train_df[num_cols]) if num_cols else np.empty((len(X_train_df), 0))
X_val_num   = num_imputer.transform(X_val_df[num_cols])       if num_cols else np.empty((len(X_val_df), 0))
X_test_num  = num_imputer.transform(df_test[num_cols])        if num_cols else np.empty((len(df_test), 0))

num_scaler = StandardScaler()
X_train_num = num_scaler.fit_transform(X_train_num) if num_cols else X_train_num
X_val_num   = num_scaler.transform(X_val_num)       if num_cols else X_val_num
X_test_num  = num_scaler.transform(X_test_num)      if num_cols else X_test_num

# --- Stack feature blocks ---
def _hstack(parts):
    parts = [p for p in parts if p is not None and getattr(p, "size", 0) > 0]
    return np.hstack(parts) if parts else np.empty((len(X_train_df), 0))

X_train = _hstack([X_train_num] + X_train_cat)
X_val   = _hstack([X_val_num]   + X_val_cat)

# Build test features (optional if you need submit preds later)
def _hstack_test(parts):
    parts = [p for p in parts if p is not None and getattr(p, "size", 0) > 0]
    return np.hstack(parts) if parts else np.empty((len(df_test), 0))

X_test = _hstack_test([X_test_num] + X_test_cat)

# --- Final NaN/inf guards ---
def _validate(arr, name):
    bad = np.isnan(arr).sum()
    if bad:
        raise ValueError(f"{name} still contains {bad} NaNs after preprocessing.")
    if np.isinf(arr).any():
        raise ValueError(f"{name} contains inf values after preprocessing.")

_validate(X_train, "X_train")
_validate(X_val,   "X_val")
if np.isnan(y_train).any() or np.isinf(y_train).any():
    raise ValueError("y_train contains NaN/inf.")
if np.isnan(y_val).any() or np.isinf(y_val).any():
    raise ValueError("y_val contains NaN/inf.")

# --- Torch tensors ---
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor   = torch.tensor(X_val,   dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_val_tensor   = torch.tensor(y_val.values,   dtype=torch.float32).view(-1, 1)

# --- Model ---
class RegressionNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.fc(x)

input_dim = X_train.shape[1]
print(f"âœ… Input dimensionality after preprocessing: {input_dim}")
model = RegressionNN(input_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# --- Training ---
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    if not torch.isfinite(loss):
        raise RuntimeError("Loss became non-finite; check preprocessing or reduce LR.")
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor).item()
        print(f"Epoch [{epoch+1}/{epochs}]  train_loss={loss.item():.6f}  val_loss={val_loss:.6f}")

# --- Evaluation ---
model.eval()
with torch.no_grad():
    y_pred_val = model(X_val_tensor).cpu().numpy().ravel()

# Compute metrics
r2 = r2_score(y_val, y_pred_val)
rmse = np.sqrt(np.mean((y_pred_val - y_val.values) ** 2))

print(f"R-squared: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")


# --- Permutation Importance (for PyTorch model) ---
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Ensure model is in evaluation mode
model.eval()

def compute_permutation_importance(model, X_tensor, y_tensor, metric=r2_score):
    """Compute permutation importance for each feature."""
    # Convert tensors to NumPy for metrics
    y_true = y_tensor.detach().cpu().numpy().flatten()
    baseline_pred = model(X_tensor).detach().cpu().numpy().flatten()
    baseline_score = metric(y_true, baseline_pred)

    importances = []
    for i in range(X_tensor.shape[1]):
        X_perm = X_tensor.clone()
        # Shuffle the i-th feature
        X_perm[:, i] = X_perm[:, i][torch.randperm(X_tensor.shape[0])]
        perm_pred = model(X_perm).detach().cpu().numpy().flatten()
        perm_score = metric(y_true, perm_pred)
        importances.append(baseline_score - perm_score)

    return np.array(importances)

# Use validation data for importance (more honest than training)
importances = compute_permutation_importance(model, X_val_tensor, y_val_tensor)

# Retrieve dynamic feature names (without hardcoding column names)
feature_names = X_df.columns.tolist()

# Create DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Display feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance (Permutation, PyTorch Model)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Optional: print top 10
print("\nTop 10 most important features:")
print(feature_importance_df.head(10))



# --- Kaggle submission (uses the same preprocessing + dynamic target name) ---

# Grab test IDs without mutating df_test
test_ids = df_test[id_col].copy()

# X_test was already built earlier from:
# - categorical encoders with 'unknown' handling
# - numeric median imputer + StandardScaler (fit on train only)
# If you need to rebuild, ensure you use the SAME fitted imputers/encoders/scaler.

# Convert to tensor and predict
X_test_final_tensor = torch.tensor(X_test, dtype=torch.float32)

model.eval()
with torch.no_grad():
    submission_preds = model(X_test_final_tensor).cpu().numpy().flatten()

# Create submission with the dynamically detected target column name
submission = pd.DataFrame({
    id_col: test_ids,
    target_col: submission_preds
})

# Save
submission.to_csv("submission.csv", index=False)
print("âœ… Kaggle submission file 'submission.csv' created successfully.")


