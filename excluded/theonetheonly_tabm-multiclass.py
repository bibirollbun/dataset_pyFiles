!pip install /kaggle/input/tabm-tabular-dl-library/rtdl_num_embeddings-0.0.11-py3-none-any.whl
!pip install /kaggle/input/tabm-tabular-dl-library/tabm-0.0.1.dev0-py3-none-any.whl


import math
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scipy.special
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
from tqdm import tqdm

# Use local tabm_reference from Kaggle input
import sys
sys.path.append("/kaggle/input/tabm-tabular-dl-library")
from tabm_reference import Model, make_parameter_groups
from rtdl_num_embeddings import compute_bins

# Set seed
seed = 0
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)

# Load dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

target_col = "Fertilizer Name"
features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
categorical_features = ["Soil Type", "Crop Type"]
numerical_features = [f for f in features if f not in categorical_features]

# Set task type
task_type = 'multiclass'

# Encode target
classes = sorted(train_df[target_col].unique())
class_to_idx = {cls: i for i, cls in enumerate(classes)}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}
n_classes = len(classes)
train_df["target"] = train_df[target_col].map(class_to_idx)

# Encode categoricals
cat_maps = {col: {v: i for i, v in enumerate(train_df[col].unique())} for col in categorical_features}
for col in categorical_features:
    train_df[col] = train_df[col].map(cat_maps[col])
    test_df[col] = test_df[col].map(lambda x: cat_maps[col].get(x, 0))

cat_cardinalities = [len(cat_maps[col]) for col in categorical_features]

X_cont = train_df[numerical_features].astype(np.float32).values
X_cat = train_df[categorical_features].astype(np.int64).values
Y = train_df["target"].values.astype(np.int64)

train_idx, val_idx = sklearn.model_selection.train_test_split(np.arange(len(Y)), train_size=0.8, stratify=Y)

noise = np.random.default_rng(0).normal(0.0, 1e-5, X_cont[train_idx].shape).astype(np.float32)
qt = sklearn.preprocessing.QuantileTransformer(n_quantiles=100, output_distribution='normal')
qt.fit(X_cont[train_idx] + noise)
X_cont = qt.transform(X_cont)
X_test_cont = qt.transform(test_df[numerical_features].astype(np.float32))
X_test_cat = test_df[categorical_features].astype(np.int64).values

bins = compute_bins(torch.tensor(X_cont[train_idx], dtype=torch.float32))
num_embeddings = {
    'type': 'PiecewiseLinearEmbeddings',
    'd_embedding': 16,
    'activation': False,
    'version': 'B'
}

data_numpy = {
    'train': {'x_cont': X_cont[train_idx], 'x_cat': X_cat[train_idx], 'y': Y[train_idx]},
    'val': {'x_cont': X_cont[val_idx], 'x_cat': X_cat[val_idx], 'y': Y[val_idx]},
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = {k: {kk: torch.tensor(vv, device=device) for kk, vv in v.items()} for k, v in data_numpy.items()}

amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
amp_enabled = False
grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype == torch.float16 else None
compile_model = False

model = Model(
    n_num_features=len(numerical_features),
    cat_cardinalities=cat_cardinalities,
    n_classes=n_classes,
    backbone={'type': 'MLP', 'n_blocks': 1, 'd_block': 128, 'dropout': 0.1},
    bins=bins,
    num_embeddings=num_embeddings,
    arch_type='tabm',
    k=32,
).to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
    model = torch.nn.DataParallel(model)

optimizer = torch.optim.AdamW(make_parameter_groups(model), lr=2e-3, weight_decay=0)

if compile_model:
    model = torch.compile(model)
    evaluation_mode = torch.no_grad
else:
    evaluation_mode = torch.inference_mode

@torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype)
def apply_model(x_cont, x_cat):
    return model(x_cont, x_cat).float()

def loss_fn(y_pred, y_true):
    k = y_pred.shape[-2]
    return F.cross_entropy(y_pred.flatten(0, 1), y_true.repeat_interleave(k))

def mapk(y_true, y_pred, k=3):
    score = 0.0
    for actual, predicted in zip(y_true, y_pred):
        predicted = list(predicted[:k])
        if actual in predicted:
            score += 1.0 / (predicted.index(actual) + 1)
    return score / len(y_true)

@evaluation_mode()
def evaluate():
    model.eval()
    y_pred = torch.cat([
        apply_model(data['val']['x_cont'][i], data['val']['x_cat'][i])
        for i in torch.arange(len(data['val']['y']), device=device).split(8192)
    ])
    y_pred = scipy.special.softmax(y_pred.cpu().numpy(), axis=-1).mean(1)
    y_true = data['val']['y'].cpu().numpy()
    top3_preds = np.argsort(-y_pred, axis=1)[:, :3]
    return mapk(y_true, top3_preds, k=3)

print(f'Device: {device}, AMP: {amp_enabled}, Compile: {compile_model}')
print(f'Val MAP@3 before training: {evaluate():.4f}')

n_epochs = 500
patience = 30
batch_size = 2000
best = {'val': -math.inf, 'epoch': -1, 'weights': None}
remaining_patience = patience

for epoch in range(n_epochs):
    batches = torch.randperm(len(train_idx), device=device).split(batch_size)
    for batch_idx in tqdm(batches, desc=f"Epoch {epoch}"):
        model.train()
        optimizer.zero_grad()
        out = apply_model(data['train']['x_cont'][batch_idx], data['train']['x_cat'][batch_idx])
        loss = loss_fn(out, data['train']['y'][batch_idx])
        if grad_scaler:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()

    val_score = evaluate()
    print(f"Val MAP@3: {val_score:.4f}")
    if val_score > best['val']:
        best = {
            'val': val_score,
            'epoch': epoch,
            'weights': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        }
        remaining_patience = patience
        print("\U0001F31F New best model!")
    else:
        remaining_patience -= 1
    if remaining_patience <= 0:
        break

# Restore best model
if isinstance(model, torch.nn.DataParallel):
    model.module.load_state_dict(best['weights'])
else:
    model.load_state_dict(best['weights'])

print("Best result:", best)

# Inference on CPU to avoid OOM
model = model.module if isinstance(model, torch.nn.DataParallel) else model
model.cpu()
model.eval()

batch_size = 2048
test_tensor_cont = torch.tensor(X_test_cont, dtype=torch.float32)
test_tensor_cat = torch.tensor(X_test_cat, dtype=torch.long)

with torch.no_grad():
    test_preds = torch.cat([
        model(test_tensor_cont[i:i + batch_size], test_tensor_cat[i:i + batch_size]).float()
        for i in range(0, len(test_tensor_cont), batch_size)
    ])

test_probs = scipy.special.softmax(test_preds.numpy(), axis=-1).mean(1)
top3 = np.argsort(-test_probs, axis=1)[:, :3]
top3_labels = [" ".join(idx_to_class[i] for i in row) for row in top3]

sample_submission["Fertilizer Name"] = top3_labels
sample_submission.to_csv("submission.csv", index=False)
print("submission.csv generated.")


