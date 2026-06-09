# If needed (often already installed on Kaggle):
# !pip install -q transformers accelerate datasets

import os
import random
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)



# General config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

COMPETITION_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"

TRAIN_PATH = os.path.join(COMPETITION_PATH, "train.csv.zip")
TEST_PATH = os.path.join(COMPETITION_PATH, "test.csv.zip")
TEST_LABELS_PATH = os.path.join(COMPETITION_PATH, "test_labels.csv.zip")  # optional, usually not needed

# Toxic labels in this competition
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Transformer model
MODEL_NAME = "roberta-base"  # you can try bert-base-uncased, roberta-base, etc.

# Training hyperparameters
MAX_LEN = 192
BATCH_SIZE = 32
EPOCHS = 4
LR = 1e-5
WARMUP_RATIO = 0.2
RANDOM_SEED = 42

# For reproducibility
def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()



#Load Data (from .zip CSVs)
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
train_df.head()

#Basic EDA / Label Check
print(train_df[LABEL_COLS].describe())

# Check label imbalance
train_df[LABEL_COLS].mean().sort_values(ascending=False)



train_texts = train_df["comment_text"].fillna(" ").values
train_labels = train_df[LABEL_COLS].values.astype(float)

X_train, X_val, y_train, y_val = train_test_split(
    train_texts,
    train_labels,
    test_size=0.1,
    random_state=RANDOM_SEED,
    stratify=train_df["toxic"]
)

len(X_train), len(X_val)



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class JigsawDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return item

train_dataset = JigsawDataset(X_train, y_train, tokenizer, max_len=MAX_LEN)
val_dataset = JigsawDataset(X_val, y_val, tokenizer, max_len=MAX_LEN)
test_dataset = JigsawDataset(test_df["comment_text"].fillna(" ").values,
                             labels=None,
                             tokenizer=tokenizer,
                             max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



num_labels = len(LABEL_COLS)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

model.classifier.dropout = nn.Dropout(0.3)   # default is 0.1
print(model.classifier)

model.to(DEVICE)

# Optionally inspect number of parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}, trainable: {trainable_params:,}")



# Optimizer
optimizer = AdamW(model.parameters(), lr=LR)

# Scheduler (linear warmup + decay)
num_training_steps = EPOCHS * len(train_loader)
num_warmup_steps = int(WARMUP_RATIO * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_roc_auc(y_true, y_pred_proba):
    """
    y_true: (N, num_labels), 0/1
    y_pred_proba: (N, num_labels), in [0,1]
    """
    scores = []
    for i in range(y_true.shape[1]):
        try:
            score = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
            scores.append(score)
        except ValueError:
            # if a label has only one class in y_true, ROC-AUC is undefined
            pass
    return np.mean(scores) if scores else 0.0

def eval_model(model, dataloader):
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch.get("labels")
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits.detach().cpu().numpy()
            all_logits.append(logits)
            
            if labels is not None:
                all_labels.append(labels.numpy())
    
    all_logits = np.vstack(all_logits)
    if all_labels:
        all_labels = np.vstack(all_labels)
        probs = sigmoid(all_logits)
        roc = compute_roc_auc(all_labels, probs)
    else:
        roc = None
    
    return roc, all_logits



'''
best_val_roc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        train_loss += loss.item()
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        if (step + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    val_roc, _ = eval_model(model, val_loader)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train loss: {avg_train_loss:.4f} | Val ROC-AUC: {val_roc:.6f}")
    
    # Save best model
    if val_roc is not None and val_roc > best_val_roc:
        best_val_roc = val_roc
        torch.save(model.state_dict(), "best_model.pt")
        print("  -> New best model saved.")



# Mixed precision (autocast + GradScaler); Gradient clipping; Early stopping
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

# Option 1: use custom loss with pos_weight for class imbalance
pos_weight = torch.tensor(
    [1.0 / train_df[col].mean() for col in LABEL_COLS],
    dtype=torch.float,
    device=DEVICE
)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

scaler = GradScaler()

best_val_roc = 0.0
patience = 2
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        optimizer.zero_grad()
        
        # ðŸ”¹ Mixed precision forward + loss
        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            loss = loss_fn(logits, labels)

        # ðŸ”¹ Backward with scaler
        scaler.scale(loss).backward()
        
        # ðŸ”¹ Gradient clipping
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        train_loss += loss.item()
        
        if (step + 1) % 200 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Step {step+1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )
    
    avg_train_loss = train_loss / len(train_loader)
    
    # ðŸ”¹ Validation at end of epoch
    val_roc, _ = eval_model(model, val_loader)
    
    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train loss: {avg_train_loss:.4f} | "
        f"Val ROC-AUC: {val_roc:.6f}"
    )
    
    # ðŸ”¹ Early stopping logic
    if val_roc is not None and val_roc > best_val_roc:
        best_val_roc = val_roc
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model.pt")
        print("  -> New best model saved.")
    else:
        epochs_no_improve += 1
        print(f"  -> No improvement for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break



# Load the best model weights (if you saved them)
model.load_state_dict(torch.load("best_model.pt"))
model.to(DEVICE)

test_roc, test_logits = eval_model(model, test_loader)  # test_roc will be None (no labels)
test_probs = sigmoid(test_logits)  # shape: (len(test_df), num_labels)



submission = pd.read_csv(os.path.join(COMPETITION_PATH, "sample_submission.csv.zip"))

print("Submission shape before:", submission.shape)

for i, col in enumerate(LABEL_COLS):
    submission[col] = test_probs[:, i]

submission.head()

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")


