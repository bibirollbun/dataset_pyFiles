!pip install -q opendatasets


import opendatasets as od


od.download('https://www.kaggle.com/competitions/detecting-generated-scientific-papers')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from pathlib import Path
import os


PATH_TO_DATA = Path('./detecting-generated-scientific-papers/')
INDEX_COL_NAME = 'id'
INPUT_COL_NAME = 'text'
TARGET_COL_NAME = 'fake'

os.listdir(PATH_TO_DATA)


test_df = pd.read_csv(PATH_TO_DATA / "fake_papers_train_part_public_extended.csv", index_col=INDEX_COL_NAME)
train_df = pd.read_csv(PATH_TO_DATA / "fake_papers_test_public_extended.csv", index_col=INDEX_COL_NAME)
sample_sumbission_df = pd.read_csv(PATH_TO_DATA / "sample_submission.csv", index_col=INDEX_COL_NAME)


train_df.head()


test_df.head()


train_df[TARGET_COL_NAME].value_counts()


train_df[INPUT_COL_NAME].apply(lambda s: len(s.split())).describe()


x_train = train_df[INPUT_COL_NAME]
y_train = train_df[TARGET_COL_NAME]


x_test = test_df[INPUT_COL_NAME]
y_test = test_df[TARGET_COL_NAME]


!pip install -q transformers


import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

encoded_train = tokenizer.batch_encode_plus(
    x_train,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors='pt'
)

encoded_test = tokenizer.batch_encode_plus(
    x_test,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors='pt'
)


import torch
from torch.utils.data import TensorDataset, DataLoader

BATCH = 4
NUM_WORKERS = 2

labels_train = torch.tensor(y_train.values, dtype=torch.long)
labels_test  = torch.tensor(y_test.values, dtype=torch.long)

train_dataset = TensorDataset(encoded_train['input_ids'], encoded_train['attention_mask'], labels_train)
test_dataset  = TensorDataset(encoded_test['input_ids'],  encoded_test['attention_mask'],  labels_test)


from sklearn.model_selection import train_test_split


train_idx, val_idx = train_test_split(np.arange(len(train_dataset)), test_size=0.2, random_state=42, stratify=y_train)

def subset_loader(dataset, idxs, batch_size=BATCH, shuffle=True):
    subset = torch.utils.data.Subset(dataset, idxs)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS)

train_loader = subset_loader(train_dataset, train_idx, batch_size=BATCH, shuffle=True)
valid_loader = subset_loader(train_dataset, val_idx, batch_size=BATCH, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

print("Batches: train", len(train_loader), "valid", len(valid_loader), "test", len(test_loader))



import torch.nn as nn
from transformers import BertModel

class BertBinaryClassifier(nn.Module):
    def __init__(self, backbone_name='bert-base-uncased', dropout=0.2):
        super().__init__()
        self.backbone = BertModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:,0,:]
        logits = self.classifier(cls).squeeze(-1)
        return logits

model = BertBinaryClassifier().to(device)



from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report, confusion_matrix

scaler = GradScaler()
criterion = nn.BCEWithLogitsLoss()

optimizer = AdamW([
    {"params": model.backbone.parameters(), "lr": 2e-5},
    {"params": model.classifier.parameters(), "lr": 1e-4}
], weight_decay=0.01)

def eval_loader(loader, model):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for ids, mask, labels in loader:
            ids = ids.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            logits = model(ids, mask)
            probs = torch.sigmoid(logits)
            batch_preds = (probs>0.5).long().cpu().numpy()
            preds.append(batch_preds)
            trues.append(labels.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    f1 = f1_score(trues, preds, average='binary')
    return f1, preds, trues

def predict_texts(texts, tokenizer, model, max_len=512):
    enc = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    ids = enc['input_ids'].to(device)
    mask = enc['attention_mask'].to(device)
    model.eval()
    with torch.no_grad():
        logits = model(ids, mask)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
    return preds, probs



import time

EPOCHS = 5
best_val_f1 = -1.0
SAVE_PATH = "best_bert_binary.pt"

for epoch in range(1, EPOCHS+1):
    model.train()
    t0 = time.time()
    running_loss = 0.0
    for step, batch in enumerate(train_loader, start=1):
        ids, mask, labels = batch
        ids = ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device).float()

        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            logits = model(ids, mask)
            loss = criterion(logits, labels)
            loss = loss / 1.0

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        running_loss += loss.item()

    val_f1, val_preds, val_trues = eval_loader(valid_loader, model)
    print(f"Epoch {epoch} train_loss={running_loss/len(train_loader):.4f} val_f1={val_f1:.4f} time={(time.time()-t0):.0f}s")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), SAVE_PATH)
        print("Saved best model, val_f1=", best_val_f1)



model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
test_f1, test_preds, test_trues = eval_loader(test_loader, model)
print("Test F1:", test_f1)
print("\nClassification report (test):")
print(classification_report(test_trues, test_preds))
print("\nConfusion matrix:")
print(confusion_matrix(test_trues, test_preds))



n_show = 10
sample_texts = x_test[:n_show].tolist()
preds, probs = predict_texts(sample_texts, tokenizer, model, max_len=512)
for i, txt in enumerate(sample_texts):
    print(f"Example {i+1} | Pred: {int(preds[i])} | Prob: {probs[i]:.4f}")
    print(txt[:600].replace("\n"," ") + ("..." if len(txt)>600 else ""))
    print("-"*100)


