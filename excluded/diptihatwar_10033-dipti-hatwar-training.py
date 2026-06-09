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
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Cell 1 - imports and seed
import os
import random
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import log_loss



# set seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  # may slow down
    # torch.backends.cudnn.benchmark = False

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



# Cell 2 - read data and prepare text
train = pd.read_csv('/kaggle/input/creating-folds-lmsys-dipti-10033/train_5folds.csv')  # adjust path if needed
# Ensure expected columns exist
expected_cols = ['prompt','response_a','response_b','label','kfold']
for c in expected_cols:
    if c not in train.columns:
        raise KeyError(f"Expected column '{c}' not found in dataframe. Columns: {train.columns.tolist()}")

# Build single text field combining prompt + both model responses (same as you had)
train['text'] = 'User prompt: ' + train['prompt'].fillna('') +  '\n\nModel A :\n' + train['response_a'].fillna('') +'\n\n--------\n\nModel B:\n'  + train['response_b'].fillna('')
print(train[['id','kfold','label']].head())
print("Sample text preview:\n", train['text'].iloc[4][:400])



# Cell 3 - tokenize and build vocab (single consistent vocab)
# Simple whitespace tokenizer here — replace with HuggingFace tokenizer for best results later.
def simple_tokenize(text):
    return text.strip().split()

# Create tokenized column (we'll create it lazily for memory efficiency)
train['tokens'] = train['text'].apply(simple_tokenize)

# Build vocabulary from training folds only (avoid using validation/test to prevent leakage)
train_tokens_for_vocab = train[train['kfold'] != 0]['tokens']  # choose a fold split strategy; here we use fold 0 as validation example
# You can also use all data for vocab if you prefer; both are common.

# special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for tokens in train_tokens_for_vocab:
    for t in tokens:
        if t not in vocab:
            vocab[t] = len(vocab)

print("Vocab size:", len(vocab))



# Cell 4 - encode helper and Dataset/Collate with lengths for packing
def encode(tokens):
    return torch.tensor([vocab.get(t, vocab[UNK_TOKEN]) for t in tokens], dtype=torch.long)

class TextDataset(Dataset):
    def __init__(self, tokens_list, labels):
        self.tokens = tokens_list
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        encoded = encode(self.tokens[idx])
        label = int(self.labels[idx])
        return encoded, torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    # pad_sequence needs a list of tensors
    padded = pad_sequence(sequences, batch_first=True, padding_value=vocab[PAD_TOKEN])
    return padded, lengths, torch.tensor(labels, dtype=torch.long)



# Cell 5 - model definition
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=1, num_classes=3, bidirectional=True, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers>1 else 0.0)
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        final_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(final_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, lengths):
        x = self.embedding(x)  # (batch, seq_len, embed)
        # pack
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        # h_n shape: (num_layers * num_directions, batch, hidden_dim)
        if self.bidirectional:
            # concat last forward and backward hidden states
            # take last layer's forward and backward
            forward = h_n[-2,:,:]
            backward = h_n[-1,:,:]
            h = torch.cat((forward, backward), dim=1)
        else:
            h = h_n[-1]
        logits = self.fc(h)
        return logits



# Cell 6 - prepare data loaders for fold k (change kfold variable to train other folds)
kfold = 0  # change for other folds in your cross-validation loop
batch_size = 16

train_df = train[train['kfold'] != kfold].reset_index(drop=True)
val_df   = train[train['kfold'] == kfold].reset_index(drop=True)

train_dataset = TextDataset(train_df['tokens'].tolist(), train_df['label'].tolist())
val_dataset   = TextDataset(val_df['tokens'].tolist(), val_df['label'].tolist())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# compute class weights for imbalanced dataset
label_counts = Counter(train_df['label'].tolist())
num_classes = 3
counts = np.array([label_counts.get(i, 0) for i in range(num_classes)], dtype=np.float32)
# avoid divide by zero
counts[counts==0] = 1.0
class_weights = counts.sum() / (num_classes * counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class weights:", class_weights)



# Cell 7 - training + validation loop (one fold). It saves best model by val log_loss and returns OOF probs for this fold.
from math import inf
scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))  # mixed precision

def train_one_fold(train_loader, val_loader, vocab_size, pad_idx=0, epochs=6, lr=3e-4, fold_num=0, out_dir='./models'):
    os.makedirs(out_dir, exist_ok=True)
    model = RNNClassifier(vocab_size=vocab_size, embed_dim=128, hidden_dim=256, num_layers=1, num_classes=num_classes, bidirectional=True, dropout=0.2, pad_idx=pad_idx)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    best_val_loss = inf
    best_epoch = -1

    # prepare arrays for OOF probabilities (for validation set)
    val_preds = np.zeros((len(val_df), num_classes), dtype=np.float32)
    val_targets = val_df['label'].values

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, lengths, y_batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}"):
            X_batch = X_batch.to(device)
            lengths = lengths.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                logits = model(X_batch, lengths)
                loss = criterion(logits, y_batch)

            scaler.scale(loss).backward()
            # gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

        scheduler.step()
        avg_train_loss = np.mean(train_losses)

        # validation
        model.eval()
        val_losses = []
        all_probs = []
        with torch.no_grad():
            for i, (X_batch, lengths, y_batch) in enumerate(tqdm(val_loader, desc="Val")):
                X_batch = X_batch.to(device)
                lengths = lengths.to(device)
                y_batch = y_batch.to(device)
                logits = model(X_batch, lengths)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        avg_val_loss = np.mean(val_losses)
        # stack and fill val_preds
        all_probs = np.vstack(all_probs)
        assert all_probs.shape[0] == len(val_df), f"Validation prediction shape mismatch {all_probs.shape} vs {len(val_df)}"
        val_preds[:] = all_probs

        val_logloss = log_loss(val_targets, val_preds, labels=list(range(num_classes)))
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val LogLoss: {val_logloss:.4f}")

        # save best by val_logloss
        if val_logloss < best_val_loss:
            best_val_loss = val_logloss
            best_epoch = epoch+1
            torch.save(model.state_dict(), os.path.join(out_dir, f"best_model_fold{fold_num}.pth"))
            print(f"Saved best model for fold {fold_num} at epoch {epoch+1} (Val LogLoss {val_logloss:.4f})")

    print(f"Finished training fold {fold_num}, best epoch {best_epoch}, best val logloss {best_val_loss:.4f}")
    return val_preds, val_targets, best_val_loss

# Run training for the example fold
val_preds, val_targets, best_val_loss = train_one_fold(train_loader, val_loader, vocab_size=len(vocab), pad_idx=vocab[PAD_TOKEN], epochs=6, lr=3e-4, fold_num=kfold)
print("Fold val logloss:", best_val_loss)



# Cell 8 - run K-fold loop to get full OOF predictions (you can iterate kfold=0..n-1)
n_splits = train['kfold'].nunique()
oof_probs = np.zeros((len(train), num_classes), dtype=np.float32)
oof_targets = train['label'].values

for fold in range(n_splits):
    print("\n" + "="*40)
    print(f"Starting fold {fold}")
    # build loaders for this fold
    train_df = train[train['kfold'] != fold].reset_index(drop=True)
    val_df   = train[train['kfold'] == fold].reset_index(drop=True)

    train_dataset = TextDataset(train_df['tokens'].tolist(), train_df['label'].tolist())
    val_dataset   = TextDataset(val_df['tokens'].tolist(), val_df['label'].tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # recompute class weights per fold (optional)
    label_counts = Counter(train_df['label'].tolist())
    counts = np.array([label_counts.get(i, 0) for i in range(num_classes)], dtype=np.float32)
    counts[counts==0] = 1.0
    class_weights = torch.tensor((counts.sum() / (num_classes * counts)), dtype=torch.float32).to(device)

    val_preds_fold, val_targets_fold, best_val_loss = train_one_fold(train_loader, val_loader, vocab_size=len(vocab), pad_idx=vocab[PAD_TOKEN], epochs=5, lr=3e-4, fold_num=fold, out_dir='./models')

    # place val_preds into oof array at their original indices
    val_idx_global = train[train['kfold'] == fold].index.values
    oof_probs[val_idx_global] = val_preds_fold

# compute overall log loss
overall_logloss = log_loss(oof_targets, oof_probs, labels=list(range(num_classes)))
print("OOF LogLoss (all folds):", overall_logloss)



# Cell 9 - save OOF predictions and example submission-format CSV
np.save("oof_probs.npy", oof_probs)
oof_df = train[['id']].copy()
oof_df['winner_model_a'] = oof_probs[:,0]
oof_df['winner_model_b'] = oof_probs[:,1]
oof_df['winner_tie']     = oof_probs[:,2]
oof_df.to_csv("oof_preds.csv", index=False)
print("Saved oof_preds.csv and oof_probs.npy")



# Cell 10 - load best model for a fold / inference example (loads best_model_fold0.pth saved earlier)
fold_to_load = 0
model_inf = RNNClassifier(vocab_size=len(vocab), embed_dim=128, hidden_dim=256, num_layers=1, num_classes=num_classes, bidirectional=True, dropout=0.2, pad_idx=vocab[PAD_TOKEN])
model_inf.load_state_dict(torch.load(f"./models/best_model_fold{fold_to_load}.pth", map_location=device))
model_inf.to(device)
model_inf.eval()

import torch.nn.functional as F
def predict_text(text, model=model_inf):
    tokens = simple_tokenize(text)
    encoded = encode(tokens).unsqueeze(0).to(device)
    lengths = torch.tensor([encoded.size(1)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(encoded, lengths)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs

# quick examples
print(predict_text("this movie was amazing"))
print(predict_text("this film was awful"))








