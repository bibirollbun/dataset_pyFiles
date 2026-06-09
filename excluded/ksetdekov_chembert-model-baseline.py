# Offline‐capable Polymer Baseline Notebook
# -------------------------------------
# 0. Pre‑download the model (from a kaggle model)
# -------------------------------------------------

import os
# --- FORCE OFFLINE MODE ---
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_METRICS_OFFLINE'] = '1'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# 1. Load Data
data_root = '/kaggle/input/neurips-open-polymer-prediction-2025/'
train_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_root, 'test.csv'))

# 2. Pre-trained SMILES tokenizer & model (ChemBERTa)

#  2. Local SMILES MODEL Path
LOCAL_MODEL_DIR = '/kaggle/input/chemberta-zinc-base-v1/transformers/seyonec-chemberta-zinc-base-v1/1/models/ChemBERTa-zinc-base-v1'
# Use local_files_only=True to prevent any internet calls
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
base_model = AutoModel.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)



# 3. Dataset Definition
class PolymerDataset(Dataset):
    def __init__(self, df, tokenizer, targets=None, max_length=512):
        self.smiles = df['SMILES'].tolist()
        self.ids = df['id'].tolist()
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        s = self.smiles[idx]
        enc = self.tokenizer(s, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        item = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0)
        }
        if self.targets is not None:
            item['labels'] = torch.tensor(self.targets[idx], dtype=torch.float)
        return item

# 4. Regression Model
class RegressionModel(nn.Module):
    def __init__(self, base_model, num_targets=5):
        super().__init__()
        self.base = base_model
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(self.base.config.hidden_size, num_targets)

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]  # CLS token pooling
        x = self.dropout(pooled)
        return self.regressor(x)

# 5. Prepare DataLoaders
TARGET_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=42)
train_ds = PolymerDataset(train_data, tokenizer, targets=train_data[TARGET_COLS].values)
val_ds = PolymerDataset(val_data, tokenizer, targets=val_data[TARGET_COLS].values)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

# 6. Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RegressionModel(base_model).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.L1Loss()  # MAE

epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {k: batch[k].to(device) for k in ('input_ids', 'attention_mask')}
        outputs = model(**inputs)
        loss = criterion(outputs, batch['labels'].to(device))
        loss.backward(); optimizer.step()
    # Validation
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: batch[k].to(device) for k in ('input_ids', 'attention_mask')}
            outputs = model(**inputs).cpu().numpy()
            preds.append(outputs)
            trues.append(batch['labels'].numpy())
    preds = np.vstack(preds); trues = np.vstack(trues)
    val_mae = np.mean(np.abs(preds - trues))
    print(f"Epoch {epoch+1} Validation MAE: {val_mae:.4f}")

# 7. Inference on Test Set

test_ds = PolymerDataset(test_df, tokenizer)
test_loader = DataLoader(test_ds, batch_size=8)
model.eval()
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        inputs = {k: batch[k].to(device) for k in ('input_ids', 'attention_mask')}
        outputs = model(**inputs).cpu().numpy()
        all_preds.append(outputs)
all_preds = np.vstack(all_preds)





# 8. Create Submission
submission = pd.DataFrame(all_preds, columns=TARGET_COLS)
submission['id'] = test_df['id']
submission = submission[['id'] + TARGET_COLS]
submission.to_csv('submission.csv', index=False)
print('Done. Submission saved to submission.csv')


