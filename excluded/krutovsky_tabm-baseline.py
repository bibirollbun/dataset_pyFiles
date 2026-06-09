!pip install rtdl_num_embeddings tabm --quiet


import math
import random
from copy import deepcopy
from typing import Any, Literal, NamedTuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rtdl_num_embeddings
import scipy.special
import seaborn as sns
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import tabm
import torch
import torch.nn as nn
import torch.optim
from rtdl_num_embeddings import (LinearReLUEmbeddings, PeriodicEmbeddings,
                                 PiecewiseLinearEmbeddings, compute_bins)
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from tabm import TabM
from torch import Tensor


seed = 0
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)
pass


train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

print("ðŸ“ˆ Dataset Information:")
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"Target distribution:\n{train_df['y'].value_counts(normalize=True)}")

# Display basic info
train_df.info()


# Define feature types
categorical_features = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

numerical_features = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]
n_num_features = len(numerical_features)

print("ðŸ”§ Feature Engineering for TabM:")
print(f"Categorical features: {len(categorical_features)}")
print(f"Numerical features: {len(numerical_features)}")

X_num = train_df[numerical_features].to_numpy()
X_cat = train_df[categorical_features].to_numpy()
y = train_df["y"].to_numpy()

all_idx = np.arange(len(y))
trainval_idx, test_idx = train_test_split(all_idx, train_size=0.8)
train_idx, val_idx = train_test_split(trainval_idx, train_size=0.8)
data_numpy = {
    "train": {
        "x_num": X_num[train_idx],
        "x_cat": X_cat[train_idx],
        "y": y[train_idx],
    },
    "val": {
        "x_num": X_num[val_idx],
        "x_cat": X_cat[val_idx],
        "y": y[val_idx],
    },
    "test": {
        "x_num": X_num[test_idx],
        "x_cat": X_cat[test_idx],
        "y": y[test_idx],
    },
    "submission": {
        "x_num": test_df[numerical_features].to_numpy(),
        "x_cat": test_df[categorical_features].to_numpy(),
        "y": np.zeros(len(test_df)),
    }
}

print(f"\nðŸ“Š Data Split:")
for part, part_data in data_numpy.items():
    for key, value in part_data.items():
        print(f'{part:<5}    {key:<5}    {value.shape!r:<10}    {value.dtype}')
        del key, value
    del part, part_data


x_num_train_numpy = data_numpy['train']['x_num']
noise = (
    np.random.default_rng(0)
    .normal(0.0, 1e-5, x_num_train_numpy.shape)
    .astype(np.float32)
)
preprocessing = QuantileTransformer(
    n_quantiles=1000,
    output_distribution='normal',
    subsample=10**9,
).fit(x_num_train_numpy + noise)
del x_num_train_numpy

x_cat_train_numpy = data_numpy['train']['x_cat']
ordinal_encoder = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1,
    dtype=np.int64,
).fit(x_cat_train_numpy)
del x_cat_train_numpy


for part in data_numpy:
    data_numpy[part]['x_num'] = preprocessing.transform(data_numpy[part]['x_num']).astype(np.float32)
    data_numpy[part]['x_cat'] = ordinal_encoder.transform(data_numpy[part]['x_cat'])


data_numpy['train']['x_num'].dtype


Y_train = data_numpy['train']['y'].copy()


train_cat_dataset = data_numpy['train']['x_cat']
cat_cardinalities: np.ndarray = (train_cat_dataset.max(axis=0) + 1).tolist()
cat_cardinalities


# TabM Model Configuration
print("ðŸ¤– Configuring TabM Model...")

n_classes = 2

tabm_model = TabM.make(
    n_num_features=len(numerical_features), cat_cardinalities=cat_cardinalities, d_out=2
)

print("âœ… TabM model configured successfully!")


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

amp_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
    if torch.cuda.is_available()
    else None
)

amp_enabled = False and amp_dtype is not None
grad_scaler = torch.cuda.amp.GradScaler(device) if amp_dtype is torch.float16 else None

compile_model = False

print(f'Device:        {device.type.upper()}')
print(f'AMP:           {amp_enabled}{f" ({amp_dtype})"if amp_enabled else ""}')
print(f'torch.compile: {compile_model}')



data = {
    part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
    for part in data_numpy
}
Y_train = torch.as_tensor(Y_train, device=device)


num_embeddings = None
num_embeddings = LinearReLUEmbeddings(n_num_features)
# num_embeddings = PeriodicEmbeddings(n_num_features, lite=False)


# num_embeddings = PiecewiseLinearEmbeddings(
#     compute_bins(data['train']['x_num'], n_bins=48),
#     d_embedding=16,
#     activation=False,
#     version='B',
# )


model = TabM.make(
    n_num_features=n_num_features,
    cat_cardinalities=cat_cardinalities,
    d_out=1 if n_classes is None else n_classes,
    num_embeddings=num_embeddings,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)
gradient_clipping_norm: Optional[float] = 1.0

if compile_model:
    # NOTE
    # `torch.compile(model, mode="reduce-overhead")` caused issues during training,
    # so the `mode` argument is not used.
    model = torch.compile(model)
    evaluation_mode = torch.no_grad
else:
    evaluation_mode = torch.inference_mode


# A quick reminder: TabM represents an ensemble of k MLPs.
#
# The option below determines if the MLPs are trained
# on the same batches (share_training_batches=True) or
# on different batches. Technically, this option determines:
# - How the loss function is implemented.
# - How the training batches are constructed.
#
# `True` is recommended by default because of better training efficiency.
# On some tasks, `False` may provide better performance.
share_training_batches = True


@torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
def apply_model(part: str, idx: Tensor) -> Tensor:
    return (
        model(
            data[part]["x_num"][idx],
            data[part]["x_cat"][idx],
        )
        .float()
    )


base_loss_fn = nn.functional.cross_entropy


def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
    # TabM produces k predictions. Each of them must be trained separately.

    # Regression:     (batch_size, k)            -> (batch_size * k,)
    # Classification: (batch_size, k, n_classes) -> (batch_size * k, n_classes)
    y_pred = y_pred.flatten(0, 1)

    if share_training_batches:
        # (batch_size,) -> (batch_size * k,)
        y_true = y_true.repeat_interleave(model.backbone.k)
    else:
        # (batch_size, k) -> (batch_size * k,)
        y_true = y_true.flatten(0, 1)

    return base_loss_fn(y_pred, y_true)

def apply_model_on_part(part: str, eval_batch_size = 8096) -> np.ndarray:
    y_pred: np.ndarray = (
        torch.cat(
            [
                apply_model(part, idx)
                for idx in torch.arange(len(data[part]["y"]), device=device).split(
                    eval_batch_size
                )
            ]
        )
        .cpu()
        .numpy()
    )

    return y_pred
    
@evaluation_mode()
def predict_on_part(part: str, eval_batch_size=8096):
    model.eval()
    y_pred = apply_model_on_part(part, eval_batch_size)

    # Compute the mean of the k predictions.
    y_pred = scipy.special.softmax(y_pred, axis=-1)
    y_pred = y_pred.mean(1)

    return y_pred

def evaluate(part: str) -> float:
    model.eval()
    # When using torch.compile, you may need to reduce the evaluation batch size.
    eval_batch_size = 8096
    y_pred = predict_on_part(part)

    y_true = data[part]["y"].cpu().numpy()
    score = roc_auc_score(y_true, y_pred[:, 1])
    return float(score)  # The higher -- the better.


print(f'Test score before training: {evaluate("test"):.4f}')


n_epochs = 1_000_000_000
train_size = len(train_idx)
batch_size = 256
epoch_size = math.ceil(train_size / batch_size)

epoch = -1
metrics = {'val': -math.inf, 'test': -math.inf}


def make_checkpoint() -> dict[str, Any]:
    return deepcopy(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
        }
    )


best_checkpoint = make_checkpoint()

# Early stopping: the training stops if the validation score
# does not improve for more than `patience` consecutive epochs.
patience = 16
remaining_patience = patience

for epoch in range(n_epochs):
    batches = (
        # Create one standard batch sequence.
        torch.randperm(train_size, device=device).split(batch_size)
        if share_training_batches
        # Create k independent batch sequences.
        else (
            torch.rand((train_size, model.backbone.k), device=device)
            .argsort(dim=0)
            .split(batch_size, dim=0)
        )
    )
    for batch_idx in batches:
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(apply_model('train', batch_idx), Y_train[batch_idx])
        if gradient_clipping_norm is not None:
            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), gradient_clipping_norm
            )
        if grad_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            grad_scaler.scale(loss).backward()  # type: ignore
            grad_scaler.step(optimizer)
            grad_scaler.update()

    metrics = {part: evaluate(part) for part in ['val', 'test']}
    val_score_improved = metrics['val'] > best_checkpoint['metrics']['val']

    print(
        f'{"*" if val_score_improved else " "}'
        f' [epoch] {epoch:<3}'
        f' [val] {metrics["val"]:.3f}'
        f' [test] {metrics["test"]:.3f}'
    )

    if val_score_improved:
        best_checkpoint = make_checkpoint()
        remaining_patience = patience
    else:
        remaining_patience -= 1

    if remaining_patience < 0:
        break

# To make final predictions, load the best checkpoint.
model.load_state_dict(best_checkpoint['model'])

print('\n[Summary]')
print(f'best epoch:  {best_checkpoint["epoch"]}')
print(f'val score:  {best_checkpoint["metrics"]["val"]}')
print(f'test score: {best_checkpoint["metrics"]["test"]}')


submission_predictions = predict_on_part("submission")

submission_df = pd.DataFrame(
    {
        "id": test_df["id"],
        "y": submission_predictions[:, 1],
    }
)

submission_df.to_csv("submission.csv", index=False)

