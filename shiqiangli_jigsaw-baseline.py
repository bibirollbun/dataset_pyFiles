# import os
# import sys
# sys.path.append(os.path.abspath(".."))

import warnings
from tqdm import tqdm

import polars as pl
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# from src.config import Config

warnings.filterwarnings("ignore")
pl.Config.set_fmt_str_lengths(1000)


class Config:
    RAW_DATA_PATH = r"/kaggle/input/jigsaw-agile-community-rules"
    MODEL_PATH = r"/kaggle/input/bert-base-uncased/transformers/default/1/86b5e0934494bd15c9632b12f734a8a67f723594"


train_data_pl = pl.read_csv(f"{Config.RAW_DATA_PATH}/train.csv")
test_data_pl = pl.read_csv(f"{Config.RAW_DATA_PATH}/test.csv")
train_data_pl.shape, test_data_pl.shape


train_data_pl.head(2)


test_data_pl.head(2)


# 训练和验证数据集的 Dataset
class RuleViolationDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.texts = (
            "[RULE] " + df["rule"] + " [SEP] " +
            "[COMMENT] " + df["body"] + " [SEP] " +
            "[SUB] " + df["subreddit"]
        ).tolist()
        self.labels = df["rule_violation"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Dataset 结构与训练相同
class RuleViolationTestDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.texts = (
            "[RULE] " + df["rule"] + " [SEP] " +
            "[COMMENT] " + df["body"] + " [SEP] " +
            "[SUB] " + df["subreddit"]
        ).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


# 数据集划分
df = train_data_pl.to_pandas()
df = df.dropna(subset=["body", "rule", "subreddit", "rule_violation"])
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["rule_violation"])

# Tokenizer 和 Dataset
tokenizer = BertTokenizer.from_pretrained(Config.MODEL_PATH)
train_dataset = RuleViolationDataset(train_df, tokenizer)
val_dataset = RuleViolationDataset(val_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 模型定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(Config.MODEL_PATH, num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

EPOCHS = 2
for epoch in range(EPOCHS):
    # -------- 训练 --------
    model.train()
    total_loss = 0
    train_probs, train_labels = [], []
    loop = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        probs = torch.softmax(outputs.logits, dim=1)[:, 1].detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()
        train_probs.extend(probs)
        train_labels.extend(labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item())

    train_auc = roc_auc_score(train_labels, train_probs)
    print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(train_loader):.4f}, Train AUC: {train_auc:.4f}")

    # -------- 验证 --------
    model.eval()
    val_probs, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            val_probs.extend(probs)
            val_labels.extend(labels)

    val_auc = roc_auc_score(val_labels, val_probs)
    print(f"[Epoch {epoch+1}] Validation AUC: {val_auc:.4f}")


test_df = test_data_pl.to_pandas()
test_dataset = RuleViolationTestDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16)

# 推理
preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        probabilities = torch.softmax(outputs.logits, dim=1)[:, 1]
        preds.extend(probabilities.cpu().numpy())


submission = pd.DataFrame({
    "row_id": test_df["row_id"],  
    "rule_violation": preds
})
submission.to_csv("submission.csv", index=False)




