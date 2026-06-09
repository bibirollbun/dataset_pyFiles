%%capture
!pip install rtdl_num_embeddings
!pip install tabm


import math
import random
from copy import deepcopy
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
import rtdl_num_embeddings 
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import tabm
import torch
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')


seed = 42
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)

# === Load Data ===
print("Loading data...")
train_df = pd.read_parquet('/kaggle/input/alfa-features-kitchen/SET_2_train_data.parquet')
test_df = pd.read_parquet('/kaggle/input/alfa-features-kitchen/SET_2_test_data.parquet')

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")


# === Custom WMAE metric ===
weights = {0: 1, 1: 0.72, 2: 0.52, 3: 0.37, 4: 0.27, 5: 0.19, 6: 0.14}

def wmae_metric_func(y_true, y_pred):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    abs_errors = np.abs(y_true - y_pred)
    weighted_errors = [weights[int(y)] * err for y, err in zip(y_true, abs_errors)]
    return np.mean(weighted_errors)


# === Prepare dataset for TabM ===
task_type: Literal['multiclass'] = 'multiclass'
n_classes = 7

# Prepare features (exclude target and client_num)
feature_cols = [col for col in train_df.columns if col not in ['target', 'client_num']]
X_num = train_df[feature_cols].values.astype(np.float32)
Y = train_df['target'].values.astype(np.int64)

# Test data features
X_test = test_df[feature_cols].values.astype(np.float32)

print(f"Feature columns: {len(feature_cols)}")
print(f"Target distribution:\n{pd.Series(Y).value_counts().sort_index()}")

# Validate that Y contains values 0 to n_classes-1
assert set(Y.tolist()) == set(range(n_classes)), (
    'Classification labels must form the range [0, 1, ..., n_classes - 1]'
)

# === Split the dataset ===
# Create train/val split for cross-validation
train_idx, val_idx = sklearn.model_selection.train_test_split(
    np.arange(len(Y)), train_size=0.85, random_state=seed, stratify=Y
)

# Prepare data structure
data_numpy = {
    'train': {'x_num': X_num[train_idx], 'y': Y[train_idx]},
    'val': {'x_num': X_num[val_idx], 'y': Y[val_idx]},
    'test': {'x_num': X_test, 'y': np.zeros(len(X_test), dtype=np.int64)}  # Dummy labels for test
}

# No categorical features in our case
cat_cardinalities = []
X_cat = None

n_num_features = X_num.shape[1]

print(f"\nData splits:")
for part, part_data in data_numpy.items():
    for key, value in part_data.items():
        print(f'{part:<5}    {key:<5}    {value.shape!r:<15}    {value.dtype}')

# === Feature preprocessing ===
print("\nApplying feature preprocessing...")

# Check for constant features and remove them
x_num_train_numpy = data_numpy['train']['x_num']

# Find non-constant features
feature_std = np.std(x_num_train_numpy, axis=0)
non_constant_mask = feature_std > 1e-8  # Features with meaningful variance
constant_features = np.sum(~non_constant_mask)

if constant_features > 0:
    print(f"Found {constant_features} constant features, removing them...")
    # Filter out constant features from all datasets
    for part in data_numpy:
        data_numpy[part]['x_num'] = data_numpy[part]['x_num'][:, non_constant_mask]
    
    # Update feature info
    n_num_features = data_numpy['train']['x_num'].shape[1]
    print(f"Features after filtering: {n_num_features}")

# Advanced preprocessing strategy with QuantileTransformer
x_num_train_numpy = data_numpy['train']['x_num']
noise = (
    np.random.default_rng(0)
    .normal(0.0, 1e-5, x_num_train_numpy.shape)
    .astype(x_num_train_numpy.dtype)
)

preprocessing = sklearn.preprocessing.QuantileTransformer(
    n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
    output_distribution='normal',
    subsample=10**9,
).fit(x_num_train_numpy + noise)

# Apply preprocessing to all parts
for part in data_numpy:
    data_numpy[part]['x_num'] = preprocessing.transform(data_numpy[part]['x_num'])

print("Preprocessing completed.")


# === PyTorch setup ===
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Convert data to tensors
data = {
    part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
    for part in data_numpy
}

# AMP settings
amp_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
    if torch.cuda.is_available()
    else None
)
amp_enabled = False and amp_dtype is not None
grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None

compile_model = False

print(f'Device:        {device.type.upper()}')
print(f'AMP:           {amp_enabled}{f" ({amp_dtype})" if amp_enabled else ""}')
print(f'torch.compile: {compile_model}')



# === Model creation ===
print("\nCreating TabM model...")

# Use PiecewiseLinearEmbeddings for best performance
try:
    num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
        rtdl_num_embeddings.compute_bins(data['train']['x_num'], n_bins=48),
        d_embedding=16,
        activation=False,
        version='B',
    )
    print("Using PiecewiseLinearEmbeddings")
except ValueError as e:
    print(f"PiecewiseLinearEmbeddings failed: {e}")
    print("Falling back to PeriodicEmbeddings")
    # Fallback to PeriodicEmbeddings if PiecewiseLinear fails
    num_embeddings = rtdl_num_embeddings.PeriodicEmbeddings(n_num_features, lite=False)

model = tabm.TabM.make(
    n_num_features=n_num_features,
    cat_cardinalities=cat_cardinalities,
    d_out=n_classes,  # 7 classes
    num_embeddings=num_embeddings,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)

if compile_model:
    model = torch.compile(model)
    evaluation_mode = torch.no_grad
else:
    evaluation_mode = torch.inference_mode

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")



# === Training setup ===
share_training_batches = True

@torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)
def apply_model(part: str, idx: Tensor) -> Tensor:
    return model(
        data[part]['x_num'][idx],
        data[part]['x_cat'][idx] if 'x_cat' in data[part] else None,
    ).float()

base_loss_fn = F.cross_entropy

def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
    # TabM produces k predictions. Each of them must be trained separately.
    # Classification: (batch_size, k, n_classes) -> (batch_size * k, n_classes)
    y_pred = y_pred.flatten(0, 1)
    
    if share_training_batches:
        # (batch_size,) -> (batch_size * k,)
        y_true = y_true.repeat_interleave(model.backbone.k)
    else:
        # (batch_size, k) -> (batch_size * k,)
        y_true = y_true.flatten(0, 1)
    
    return base_loss_fn(y_pred, y_true)

@evaluation_mode()
def evaluate(part: str, return_predictions: bool = False) -> float | tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    
    eval_batch_size = 8096
    y_pred_logits = torch.cat([
        apply_model(part, idx)
        for idx in torch.arange(len(data[part]['y']), device=device).split(eval_batch_size)
    ]).cpu().numpy()
    
    # For classification, compute mean in probability space
    y_pred_probs = scipy.special.softmax(y_pred_logits, axis=-1)
    y_pred_mean_probs = y_pred_probs.mean(1)  # Average across k predictions
    y_pred_classes = y_pred_mean_probs.argmax(1)
    
    y_true = data[part]['y'].cpu().numpy()
    
    if part == 'test':
        # For test set, we don't have true labels, so return dummy accuracy
        accuracy = 0.0
        wmae_score = 0.0
    else:
        accuracy = sklearn.metrics.accuracy_score(y_true, y_pred_classes)
        wmae_score = wmae_metric_func(y_true, y_pred_classes)
    
    # Return negative WMAE as score (higher is better for early stopping)
    score = -wmae_score if part != 'test' else accuracy
    
    if return_predictions:
        return score, y_pred_mean_probs, y_pred_classes
    return score

print(f'Validation score before training: {evaluate("val"):.4f}')


# === Training loop ===
print("\nStarting training...")

n_epochs = 1000
train_size = len(train_idx)
batch_size = 512  # Increased batch size for better stability
epoch_size = math.ceil(train_size / batch_size)

epoch = -1
metrics = {'val': -math.inf, 'test': -math.inf}

def make_checkpoint() -> dict[str, Any]:
    return deepcopy({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
    })

best_checkpoint = make_checkpoint()

# Early stopping
patience = 25  # Increased patience for better convergence
remaining_patience = patience

for epoch in range(n_epochs):
    # Training
    batches = (
        torch.randperm(train_size, device=device).split(batch_size)
        if share_training_batches
        else (
            torch.rand((train_size, model.backbone.k), device=device)
            .argsort(dim=0)
            .split(batch_size, dim=0)
        )
    )
    
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    
    for batch_idx in batches:
        optimizer.zero_grad()
        y_batch = data['train']['y'][batch_idx]
        loss = loss_fn(apply_model('train', batch_idx), y_batch)
        
        if grad_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    # Evaluation
    metrics = {'val': evaluate('val')}
    val_score_improved = metrics['val'] > best_checkpoint['metrics']['val']
    
    if epoch % 5 == 0 or val_score_improved:
        print(
            f'{"*" if val_score_improved else " "}'
            f' [epoch] {epoch:<3}'
            f' [loss] {epoch_loss/n_batches:.4f}'
            f' [val] {metrics["val"]:.4f}'
        )
    
    if val_score_improved:
        best_checkpoint = make_checkpoint()
        remaining_patience = patience
    else:
        remaining_patience -= 1
    
    if remaining_patience < 0:
        print(f"Early stopping at epoch {epoch}")
        break

# === Load best model and make final predictions ===
model.load_state_dict(best_checkpoint['model'])

print('\n=== Final Evaluation ===')
print(f'Best epoch: {best_checkpoint["epoch"]}')
print(f'Best validation score: {best_checkpoint["metrics"]["val"]:.4f}')

# Get validation predictions for analysis
val_score, val_probs, val_classes = evaluate('val', return_predictions=True)
val_true = data['val']['y'].cpu().numpy()

print(f"\nValidation Results:")
print(f"WMAE Score: {wmae_metric_func(val_true, val_classes):.6f}")
print(f"Accuracy: {sklearn.metrics.accuracy_score(val_true, val_classes):.4f}")

print("\nValidation Classification Report:")
print(classification_report(val_true, val_classes))

# Get test predictions
_, test_probs, test_classes = evaluate('test', return_predictions=True)

print(f"\nTest predictions distribution:")
print(pd.Series(test_classes).value_counts().sort_index())

# === Save results ===
print("\nSaving results...")

# Main submission file
submission_df = pd.DataFrame({
    'client_num': test_df['client_num'].values,
    'target': test_classes
})

submission_df.to_csv('submission.csv', index=False)
print("Saved submission_tabm.csv")

# Save probabilities for ensemble
probs_df = pd.DataFrame(test_probs, columns=[f'prob_class_{i}' for i in range(n_classes)])
probs_df['client_num'] = test_df['client_num'].values

# Reorder columns to have client_num first
cols = ['client_num'] + [f'prob_class_{i}' for i in range(n_classes)]
probs_df = probs_df[cols]

probs_df.to_csv('test_probabilities_tabm.csv', index=False)
print("Saved test_probabilities_tabm.csv")

# Also save validation probabilities for stacking
val_probs_df = pd.DataFrame(val_probs, columns=[f'prob_class_{i}' for i in range(n_classes)])
val_probs_df['client_num'] = train_df.iloc[val_idx]['client_num'].values
val_probs_df['true_target'] = val_true

cols_val = ['client_num', 'true_target'] + [f'prob_class_{i}' for i in range(n_classes)]
val_probs_df = val_probs_df[cols_val]

val_probs_df.to_csv('val_probabilities_tabm.csv', index=False)
print("Saved val_probabilities_tabm.csv")

print(f"\n=== Summary ===")
print(f"Training completed in {best_checkpoint['epoch']} epochs")
print(f"Final validation WMAE: {wmae_metric_func(val_true, val_classes):.6f}")
print(f"Final validation accuracy: {sklearn.metrics.accuracy_score(val_true, val_classes):.4f}")
print(f"Test predictions saved to submission.csv")
print(f"Test probabilities saved to test_probabilities_tabm.csv")
print(f"Validation probabilities saved to val_probabilities_tabm.csv")


submission_df.head()


probs_df.head()

