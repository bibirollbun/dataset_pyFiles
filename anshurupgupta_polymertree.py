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
print(os.listdir('/kaggle'))
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle


Symbols_DIR = "/kaggle/input/atomic-symbols/symbols.pkl"
#def preprocess_Data():
#    from rdkit import Chem
#    ptable = Chem.GetPeriodicTable()
#    symbols = [ptable.GetElementSymbol(z) for z in range(1, 118)]
#    with open(Symbols_DIR, "wb") as fp:
#        pickle.dump(symbols, fp)
#preprocess_Data()


embeddings = 128
hidden = 2 * embeddings
numLayers = 1

EPOCH = 250
RandShuffle = True
BatchSize = 64
LearningW = 5e-3
Bidirectional = False

Train_DIR = "/kaggle/input/neurips-open-polymer-prediction-2025/train.csv"
token_DIR = '/kaggle/working/tokens.pkl'
save_DIR = '/kaggle/working/model_weights.pth'


# ========================
# 1. Tokenizer
# ========================
class SmilesRegexTokenizer:
    max_AtomicElements = 118
    def __init__(self):
        with open(Symbols_DIR, "rb") as fp:
            symbols = pickle.load(fp)
        symbols = list(set(symbols))
        two_letter = sorted([s for s in symbols if len(s) == 2], key=lambda x: -len(x))
        one_letter = sorted([s for s in symbols if len(s) == 1])
        atom_regex = "|".join(two_letter + one_letter)
        # SMILES token regex
        self.regex = re.compile(f"({atom_regex}|\\(|\\)|\\.|=|#|-|\\+|\\\\|/|\\*|%\\d\\d?|\\d)")
        self.pad_token_id = 0
        self.vocab = {"[PAD]": 0, "[UNK]": 1}
    
    def tokenize(self, smiles: str):
        tokens = self.regex.findall(smiles)
        tokens = [t for t in tokens if t.strip() != ""]
        return tokens

    def save(self):
        with open(token_DIR, 'wb') as file:
            pickle.dump(self.inv_vocab, file)
    
    def load(self):
        with open(token_DIR, 'rb') as file:
            self.inv_vocab = pickle.load(file)
    
    def build_vocab(self, smiles_list):
        for smi in smiles_list:
            for tok in self.tokenize(smi):
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)
        self.inv_vocab = {i: t for t, i in self.vocab.items()}
    
    def encode(self, smiles: str):
        return [self.vocab.get(tok, self.vocab["[UNK]"]) for tok in self.tokenize(smiles)]
    
    def __len__(self):
        return len(self.vocab)


# ========================
# 2. Dataset
# ========================
class SmilesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: SmilesRegexTokenizer, max_len=256):
        self.smiles = df["SMILES"].tolist()
        self.targets = df[["Tg","FFV","Tc","Density","Rg"]].astype(float).values
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smi = self.smiles[idx]
        token_ids = self.tokenizer.encode(smi)[:self.max_len]
        length = len(token_ids)
        pad_len = self.max_len - length
        token_ids = token_ids + [0] * pad_len
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(length), torch.tensor(self.targets[idx], dtype=torch.float)

def collate_fn(batch):
    tokens, lengths, targets = zip(*batch)
    return torch.stack(tokens), torch.stack(lengths), torch.stack(targets)


# ========================
# 3. LSTM Encoder
# ========================
class MoleculeEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        if self.bidirectional:
            final = torch.cat((h[-2], h[-1]), dim=1)
        else:
            final = h[-1]
        return final  # [B, hidden_dim*2]


# ========================
# 4. Multi-task Model
# ========================
class MoleculeModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_properties=5, num_layers= 2, bidirectional=True):
        super().__init__()
        self.encoder = MoleculeEncoder(vocab_size, embed_dim, hidden_dim, num_layers, bidirectional)
        self.project = nn.Sequential(
            nn.Linear((hidden_dim * 2 if(bidirectional) else hidden_dim), embed_dim),
            nn.LayerNorm(embed_dim),
            
            nn.Linear(embed_dim, (hidden_dim * 2 if(bidirectional) else hidden_dim)),
        )
        self.heads = nn.ModuleList([nn.Linear((hidden_dim * 2 if(bidirectional) else hidden_dim), 1) for _ in range(num_properties)])

    def save(self):
        torch.save(model.state_dict(), save_DIR)
    
    def forward(self, x, lengths):
        rep = self.encoder(x, lengths)
        rep = self.project(rep)
        outputs = [head(rep).squeeze(-1) for head in self.heads]
        return torch.stack(outputs, dim=1)


# ========================
# 5. Masked Loss
# ========================
def masked_loss(pred, target):
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=pred.device)
    return F.mse_loss(pred[mask], target[mask])

# ========================
# 6. Train Example
# ========================
df = pd.read_csv(Train_DIR)
tokenizer = SmilesRegexTokenizer()
tokenizer.build_vocab(df["SMILES"].tolist())

dataset = SmilesDataset(df, tokenizer, max_len=embeddings)
dataloader = DataLoader(dataset, batch_size=BatchSize, shuffle=RandShuffle, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MoleculeModel(len(tokenizer), embed_dim=embeddings, hidden_dim=hidden, num_properties=5, num_layers=numLayers, bidirectional=Bidirectional).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LearningW)

best_loss = np.inf
for epoch in range(EPOCH):
    total_loss = 0
    current_loss = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCH}", unit="batch")
    for tokens, lengths, targets in pbar:
        tokens, lengths, targets = tokens.to(device), lengths.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(tokens, lengths)
        loss = masked_loss(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    if(current_loss < best_loss):
        best_loss = current_loss
        model.save()
    print(f"Epoch {epoch+1}, Loss = {total_loss/len(dataloader):.4f}")


# Load sample submission
sample_sub = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
print(sample_sub.head())

# Load test set
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
print(test_df.head())

# Prepare test SMILES list
X_test = test_df['SMILES'].to_list()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MoleculeModel(len(tokenizer), embed_dim=embeddings, hidden_dim=hidden, num_properties=5, num_layers=numLayers, bidirectional=Bidirectional)
model.load_state_dict(torch.load(save_DIR, map_location=device))
model.to(device)
model.eval()

# Tokenize test SMILES
test_tokens = [tokenizer.encode(smi) for smi in X_test]

# Get sequence lengths before padding
test_lengths = [len(t) for t in test_tokens]

# Pad sequences to max length in test set
max_len = max(test_lengths)
test_tokens = [t + [0] * (max_len - len(t)) for t in test_tokens]

# Convert to tensors
test_tensor = torch.tensor(test_tokens, dtype=torch.long).to(device)
test_lengths = torch.tensor(test_lengths, dtype=torch.long).to(device)

# Forward pass
with torch.no_grad():
    preds = model(test_tensor,test_lengths).cpu().numpy()

# Fill submission
submission_df = sample_sub.copy()
submission_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']] = preds
print(submission_df.head())

# Save for Kaggle
submission_df.to_csv("submission.csv", index=False)
print("All done")

