import pickle
import random
import zipfile
import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silencia el warning de tensorflow
from time import time
from itertools import combinations
from collections import defaultdict
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch.optim as optim
# TPU support via PyTorch XLA
# !pip install -q cloud-tpu-client==0.10 torch==2.0.0 \
#     https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-2.0-cp310-cp310-linux_x86_64.whl
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# ── Drop-in replacement for Seq2Pat / DPM / Pat2Feat ─────────────────────────
# Pure Python + NumPy, no external dependencies.

from itertools import combinations
from collections import defaultdict

def _all_subsequences(seq, max_len=4):
    """Yield every unique ordered (non-contiguous) subsequence up to max_len items."""
    seen = set()
    for length in range(1, min(max_len, len(seq)) + 1):
        for indices in combinations(range(len(seq)), length):
            pat = tuple(seq[i] for i in indices)
            if pat not in seen:
                seen.add(pat)
                yield pat


def mine_frequent_patterns(sequences, min_frequency, max_pattern_len=4):
    """Return patterns whose support >= min_frequency (fraction) in sequences."""
    min_count = max(1, int(min_frequency * len(sequences)))
    counts = defaultdict(int)
    for seq in sequences:
        for pat in _all_subsequences(seq, max_len=max_pattern_len):
            counts[pat] += 1
    return [pat for pat, cnt in counts.items() if cnt >= min_count]


def dichotomic_pattern_mining(sequences_pos, sequences_neg,
                               min_freq_pos, min_freq_neg,
                               max_pattern_len=4):
    """Replaces dpm.dichotomic_pattern_mining + DichotomicAggregation."""
    t = time()
    pos_set = set(mine_frequent_patterns(sequences_pos, min_freq_pos, max_pattern_len))
    neg_set = set(mine_frequent_patterns(sequences_neg, min_freq_neg, max_pattern_len))
    print(f'DPM finished! Runtime: {time()-t:.0f} sec')
    result = {
        'union':        list(pos_set | neg_set),
        'pos_only':     list(pos_set - neg_set),
        'neg_only':     list(neg_set - pos_set),
        'intersection': list(pos_set & neg_set),
    }
    for name, pats in result.items():
        print(f'  {name:15s} -> {len(pats):5d} patterns')
    return result


def _is_subsequence(pattern, sequence):
    """True if pattern appears as an ordered subsequence of sequence."""
    it = iter(sequence)
    return all(item in it for item in pattern)


def get_features(sequences, patterns):
    """Replaces Pat2Feat.get_features.
    Returns a DataFrame with a 'sequence' column + one binary column per pattern.
    """
    col_names = [str(p) for p in patterns]
    matrix = np.array(
        [[int(_is_subsequence(pat, seq)) for pat in patterns] for seq in sequences],
        dtype=np.float32
    )
    df_feat = pd.DataFrame(matrix, columns=col_names)
    df_feat.insert(0, 'sequence', sequences)
    return df_feat


DPM_UNION_KEY = 'union'  # alias used in cells below


dir_site='/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/site_dic.pkl'
with open(dir_site, 'rb') as f:
    site_key = pickle.load(f)
df=pd.read_csv('/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/train_sessions.csv')
df.head(5)



def seq_site(df):
    """Return a list of site-id sequences, one per row, dropping NaN values."""
    # FIX: replace slow iterrows with a vectorised approach
    return [
        row.dropna().astype(int).tolist()
        for _, row in df.iterrows()
    ]



site_column = [col for col in df.columns if 'site' in col]


seq_pos=seq_site(df[df['target']==1][site_column])

seq_neg=seq_site(df[df['target']==0][site_column])




num_sqe=len(df)
num_pos=len(df[df['target']==1])

print(f'Number of sequences: {num_sqe}')
print(f'Number of positives: {num_pos}; Number of negatives: {num_sqe-num_pos}')



print(f'The porcentage of positive in the dataset is {100*num_pos/(num_sqe):.2f} %')


_, seq_neg_R, = train_test_split(seq_neg, test_size=0.2, random_state=42)


t = time()

min_frequency_pos = 0.05
min_frequency_neg = 0.05

aggregation_to_patterns = dichotomic_pattern_mining(
    seq_pos, seq_neg_R,
    min_frequency_pos, min_frequency_neg
)


sequences = seq_pos + seq_neg_R

dpm_patterns = aggregation_to_patterns[DPM_UNION_KEY]

encodings = get_features(sequences, dpm_patterns)
encodings.head(3)


labels_aligned = pd.Series(
    [1] * len(seq_pos) + [0] * len(seq_neg_R),
    name='target'
)

X_train, X_test, y_train, y_test = train_test_split(
    encodings, labels_aligned,
    test_size=0.2,
    random_state=42,
    stratify=labels_aligned  # keep class ratio in both splits
)

seq_train  = pad_sequence([torch.tensor(s) for s in X_train['sequence'].tolist()],
                           batch_first=True, padding_value=0)
feat_train = torch.tensor(X_train.iloc[:, 1:].values, dtype=torch.float32)
label_train = torch.tensor(y_train.tolist())

seq_val  = pad_sequence([torch.tensor(s) for s in X_test['sequence'].tolist()],
                         batch_first=True, padding_value=0)
feat_val  = torch.tensor(X_test.iloc[:, 1:].values, dtype=torch.float32)
label_val = torch.tensor(y_test.tolist())



class SequenceDataset(Dataset):
    def __init__(self, sequences, features, labels = None):
        self.sequences = sequences
        self.features = features
        self.labels = labels 

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.sequences[idx], self.features[idx]
        else:
            return self.sequences[idx], self.features[idx], self.labels[idx]



class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_lstm_units, input_len,
                 num_pattern_features, num_classes, layer_nodes=[512]):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, num_lstm_units, batch_first=True)

        self.dense_layers = nn.ModuleList([
            nn.Linear(num_lstm_units + num_pattern_features, layer_nodes[0])
        ])
        for i in range(1, len(layer_nodes)):
            self.dense_layers.append(nn.Linear(layer_nodes[i-1], layer_nodes[i]))

        self.output_layer = nn.Linear(layer_nodes[-1], 1)

    def forward(self, input_lstm, input_pat_feat):
        embedded = self.embedding(input_lstm)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]   # last hidden state

        merged = torch.cat((lstm_out, input_pat_feat), dim=1)

        # FIX: sigmoid in hidden layers causes vanishing gradients → use ReLU
        for layer in self.dense_layers:
            merged = F.relu(layer(merged))

        return self.output_layer(merged)


# Early stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float('inf')
        self.counter    = 0
        self.stop       = False

    def check(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def train_model(model, dataloader_train, dataloader_val,
                criterion, optimizer, scheduler, device,
                num_epochs, early_stopp):
    train_loss, val_loss = [], []
    train_acc,  val_acc  = [], []
    # MpDeviceLoader has no .dataset — capture sizes from the base loaders
    n_train = len(dataloader_train_base.dataset)
    n_val   = len(dataloader_val_base.dataset)

    for epoch in range(num_epochs):
        # ── Training ──────────────────────────────────────────────────────
        model.train()
        running_loss = running_corrects = 0.0

        for batch in dataloader_train:
            seq, feat, label = batch
            seq, feat, label = seq.to(device), feat.to(device), label.to(device)
            label = label.view(-1, 1).float()

            optimizer.zero_grad()
            outputs = model(seq, feat)
            loss    = criterion(outputs, label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # TPU: xm.optimizer_step synchronises gradients across cores
            xm.optimizer_step(optimizer)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            running_loss     += loss.item()
            running_corrects += torch.sum(preds == label.int()).item()

        train_loss.append(running_loss / n_train)
        train_acc.append(running_corrects / n_train)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        running_loss = running_corrects = 0.0

        with torch.no_grad():
            for batch in dataloader_val:
                seq, feat, label = batch
                seq, feat, label = seq.to(device), feat.to(device), label.to(device)
                label   = label.view(-1, 1).float()
                outputs = model(seq, feat)
                loss    = criterion(outputs, label)

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).int()
                running_loss     += loss.item()
                running_corrects += torch.sum(preds == label.int()).item()

        val_loss.append(running_loss / n_val)
        val_acc.append(running_corrects / n_val)

        print(f"[Epoch {epoch+1:02d}/{num_epochs}] "
              f"Train Loss: {train_loss[-1]:.4e} | Train Acc: {train_acc[-1]:.4f} || "
              f"Val Loss: {val_loss[-1]:.4e} | Val Acc: {val_acc[-1]:.4f}")

        scheduler.step()
        early_stopp.check(val_loss[-1])
        if early_stopp.stop:
            print('Early stopping triggered')
            break

    print('Training ended')
    return model, train_loss, train_acc, val_loss, val_acc


batch      = 64   # larger batch is more efficient on TPU
num_epochs = 256

dataset_train = SequenceDataset(seq_train, feat_train, label_train)
dataset_val   = SequenceDataset(seq_val,   feat_val,   label_val)

dataloader_train_base = DataLoader(dataset_train, batch_size=batch, shuffle=True)
dataloader_val_base   = DataLoader(dataset_val,   batch_size=batch, shuffle=False)

vocab_size           = len(site_key) + 1
embedding_dim        = 64
input_len            = seq_train.shape[1]
num_pattern_features = feat_train.shape[1]
num_classes          = 1
num_lstm_units       = 256
layer_nodes          = [256]
patience             = 10
num_pos              = len(seq_pos)
num_neg              = len(seq_neg_R)

model  = LSTM(vocab_size, embedding_dim, num_lstm_units, input_len,
              num_pattern_features, num_classes)

# TPU device via PyTorch XLA
device = xm.xla_device()
model  = model.to(device)

# Wrap DataLoaders so batches are pre-loaded directly onto the TPU
dataloader_train = pl.MpDeviceLoader(dataloader_train_base, device)
dataloader_val   = pl.MpDeviceLoader(dataloader_val_base,   device)

pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer   = torch.optim.Adam(model.parameters(), lr=1e-4,
                                amsgrad=True, weight_decay=1e-5)
scheduler   = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
early_stopp = EarlyStopping(patience=patience, min_delta=1e-5)

model_train, train_loss, train_acc, val_loss, val_acc = train_model(
    model, dataloader_train, dataloader_val,
    criterion, optimizer, scheduler,
    device, num_epochs, early_stopp
)


import matplotlib.pyplot as plt

plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss,   label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()


# Graficar pérdida
plt.plot(train_acc, label='Training Acc')
plt.plot(val_acc, label='Validation Acc')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.legend()
plt.title('Training and Validation Acc')
plt.show()


from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

def test_model(model, test_loader, device,class_names):
    model.eval()
    model.to(device)
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in test_loader:
            seq, feat,label = batch
            seq, feat,label =seq.to(device),feat.to(device),label.to(device)
            label=label.view(-1, 1).float()
            outputs = model(seq, feat)
            probabilities = torch.sigmoid(outputs)  # Convierte logits a probabilidades
            # preds = (probabilities > 0.5).int()  # 0 para falso, 1 para verdadero
            y_true.extend(label.cpu().numpy())
            y_pred.extend(probabilities.cpu().numpy())

    auc = roc_auc_score(y_true, y_pred)
    print(f"Test AUC: {auc:.4f}")
    
    # Usamos zero_division=0 para manejar clases sin predicciones
    print("Classification Report:")
    print(classification_report(y_true, [(i> 0.5) for i in y_pred ], target_names=class_names, digits=4, zero_division=0))

    return y_true,y_pred
class_names=['not Alice', 'Alice']
y_true,y_pred=test_model(model_train, dataloader_val, device, class_names)


test_dir = '/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/test_sessions.csv'

df_test      = pd.read_csv(test_dir)
site_columns = [col for col in df_test.columns if 'site' in col]

seq_test_raw = seq_site(df_test[site_columns])

# Use the same DPM patterns mined during training – never re-mine on test data
encodings_test = get_features(seq_test_raw, dpm_patterns)

seq_test_tensors = [torch.tensor(s) for s in encodings_test['sequence'].tolist()]
feat_test        = torch.tensor(encodings_test.iloc[:, 1:].values, dtype=torch.float32)
seqn             = pad_sequence(seq_test_tensors, batch_first=True, padding_value=0)



batch = 64
dataset_test    = SequenceDataset(seqn, feat_test)
dataloader_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)

submission_list = []
model_train.to(device)
model_train.eval()

with torch.no_grad():
    for batch_data in dataloader_test:
        seq_t, feat_t = batch_data
        seq_t, feat_t = seq_t.to(device), feat_t.to(device)
        # FIX: original used `model` (untrained) instead of `model_train`
        outputs       = model_train(seq_t, feat_t)
        probabilities = torch.sigmoid(outputs)
        submission_list.append(probabilities.cpu().numpy().astype(float))

df_submission = pd.DataFrame()
df_submission['session_id'] = df_test['session_id']
df_submission['target']     = np.concatenate(submission_list, axis=0)

df_submission.to_csv('/kaggle/working/submission.csv', index=False)
print('submission.csv generated successfully!')


df_submission.head()

