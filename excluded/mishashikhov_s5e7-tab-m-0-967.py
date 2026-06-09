!pip install tabm rtdl-num-embeddings -q


import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
from torch import Tensor

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from copy import deepcopy
import rtdl_num_embeddings
import tabm
import math



train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
original = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


# Basic split of features and targets
X_train = train.drop(['id', 'Personality'], axis=1, errors='ignore')
y_train = train['Personality']
X_test = test.drop(['id'], axis=1, errors='ignore')

# Multiply external dataset (data augmentation)
original_copy = original.copy()
for _ in range(7):  # 1 original + 7 copies = x8 total
    original = pd.concat([original, original_copy], axis=0, ignore_index=True)

X_original = original.drop(['id', 'Personality'], axis=1, errors='ignore')
y_original = original['Personality']



# Detect categorical columns
cat_cols = X_train.select_dtypes(include='object').columns
label_encoders = {}

# Apply label encoding to all categorical features
for col in cat_cols:
    le = LabelEncoder()
    all_data = pd.concat([X_train[col], X_test[col], X_original[col]])
    le.fit(all_data.astype(str))
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    X_original[col] = le.transform(X_original[col].astype(str))
    label_encoders[col] = le

# Encode the target
target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(y_train)
y_original = target_encoder.transform(y_original)



# Combine training and external data
X_all_df = pd.concat([X_train, X_original], axis=0)
feature_names = X_all_df.columns.tolist()

# Convert to float32 and align test columns
X_all = X_all_df.astype(np.float32)
X_test = X_test[feature_names].astype(np.float32)
y_all = np.concatenate([y_train, y_original]).astype(np.int64)

# Replace NaNs and Infs
X_all = np.nan_to_num(X_all, nan=0.0, posinf=1e6, neginf=-1e6)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)


# Holdout 15% of data for final validation
trainval_idx, oof_idx = train_test_split(
    np.arange(len(y_all)), train_size=0.85, stratify=y_all, random_state=42
)

# Further split trainval into train and val
train_idx, val_idx = train_test_split(
    trainval_idx, train_size=0.85, stratify=y_all[trainval_idx], random_state=42
)



data_numpy = {
    'train': {'x_num': X_all[train_idx], 'y': y_all[train_idx]},
    'val': {'x_num': X_all[val_idx], 'y': y_all[val_idx]},
    'oof': {'x_num': X_all[oof_idx], 'y': y_all[oof_idx]},
    'test': {'x_num': X_test, 'y': np.zeros(len(X_test), dtype=np.int64)}
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = {
    part: {k: torch.tensor(v, device=device) for k, v in d.items()}
    for part, d in data_numpy.items()
}



X_bins_input = torch.nan_to_num(data['train']['x_num'], nan=0.0, posinf=1e6, neginf=-1e6)
num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
    rtdl_num_embeddings.compute_bins(X_bins_input, n_bins=32),
    d_embedding=32,
    activation=False,
    version='B'
)


model = tabm.TabM.make(
    n_num_features=X_all.shape[1],
    d_out=len(np.unique(y_all)),
    cat_cardinalities=[],
    num_embeddings=num_embeddings,
    k=4,
    dropout=0.35
).to(device)

optimizer = torch.optim.RMSprop(model.parameters(), lr=3e-3, alpha=0.99)



USE_CLASS_WEIGHTS = False  # Toggle class weights for cross-entropy

if USE_CLASS_WEIGHTS:
    class_weights = compute_class_weight('balanced', classes=np.unique(y_all), y=y_all)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
else:
    class_weights = None

def loss_fn(y_pred: Tensor, y_true: Tensor) -> Tensor:
    y_pred = y_pred.flatten(0, 1)
    y_true = y_true.repeat_interleave(model.backbone.k)
    return F.cross_entropy(y_pred, y_true, weight=class_weights)


@torch.no_grad()
def evaluate(part: str) -> tuple[float, float]:
    model.eval()
    y_pred = []
    for i in torch.arange(len(data[part]['y']), device=device).split(1024):
        out = model(data[part]['x_num'][i])
        out = torch.softmax(out, dim=-1).float().mean(1)
        y_pred.append(out)
    y_pred = torch.cat(y_pred).cpu().numpy()
    y_true = data_numpy[part]['y']
    f1 = f1_score(y_true, y_pred.argmax(1), average='macro')
    acc = accuracy_score(y_true, y_pred.argmax(1))
    return f1, acc



Y_train = data['train']['y']
train_size = len(Y_train)
batch_size = 256
patience = 50
remaining_patience = patience
best_score = -math.inf
best_weights = deepcopy(model.state_dict())

for epoch in range(450):
    model.train()
    total_loss = 0
    for idx in torch.randperm(train_size, device=device).split(batch_size):
        optimizer.zero_grad()
        out = model(data['train']['x_num'][idx])
        loss = loss_fn(out, Y_train[idx])
        if not torch.isfinite(loss):
            print("NaN in loss — breaking.")
            break
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    val_f1, val_acc = evaluate('val')

    print(f"Epoch {epoch:03} | train Loss: {total_loss:.4f} | val Acc: {val_acc:.4f} | val F1: {val_f1:.4f}")
    
    if val_acc > best_score:
        best_score = val_acc
        best_weights = deepcopy(model.state_dict())
        remaining_patience = patience
    else:
        remaining_patience -= 1

    if remaining_patience == 0:
        break


# Load best weights and predict test set
model.load_state_dict(best_weights)
model.eval()

x_test = data['test']['x_num'].float()
final_preds = torch.softmax(model(x_test), dim=-1).float().mean(1)
preds_labels = final_preds.argmax(1).cpu().numpy()

# Evaluate on holdout (honest test)
oof_f1, oof_acc = evaluate('oof')
print(f"FINAL HOLDOUT (OOF) — Acc: {oof_acc:.4f} | F1: {oof_f1:.4f}")

# Create final submission
submission = sample_submission.copy()
submission['Personality'] = target_encoder.inverse_transform(preds_labels)
submission.to_csv('submission_tabm_final.csv', index=False)
print("Submission ready: submission_tabm_final.csv")


