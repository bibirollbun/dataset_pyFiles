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


import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# 1. Imports
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm

# 2. Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 3. Load and split data
df = pd.read_csv("/kaggle/input/iiitb-toxic-comment/train.csv").dropna(subset=["comment_text"])
target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# 4. Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def encode(texts):
    return tokenizer(texts, truncation=True, padding=True, max_length=256)

train_enc = encode(train_df["comment_text"].tolist())
eval_enc = encode(eval_df["comment_text"].tolist())

train_labels = train_df[target_cols].values
eval_labels = eval_df[target_cols].values

# 5. Dataset class
class ToxicDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

train_dataset = ToxicDataset(train_enc, train_labels)
eval_dataset = ToxicDataset(eval_enc, eval_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=8)

# 6. Load BERT
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(target_cols),
    problem_type="multi_label_classification"
).to(device)

# 7. Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * 3,
)

# 8. Training Loop
model.train()
for epoch in range(3):
    print(f"\nEpoch {epoch + 1}")
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item())

# 9. Evaluation
model.eval()
preds, truths = [], []

with torch.no_grad():
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        probs = torch.sigmoid(outputs.logits)
        preds.extend(probs.cpu().numpy())
        truths.extend(batch["labels"].cpu().numpy())

# 10. Thresholding & Report
preds_bin = (np.array(preds) > 0.5).astype(int)
print("\nðŸ“Š Classification Report on 20% Hold-out Set:")
print(classification_report(truths, preds_bin, target_names=target_cols, zero_division=0))






