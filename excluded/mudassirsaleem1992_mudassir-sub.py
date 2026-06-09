# first I will download all liabraies and initilaized global variables
!pip -q install transformers datasets accelerate

import os
import random
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    get_cosine_schedule_with_warmup
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# Here I am loading data and splitting the trainin and testing datset
DATA_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"

train_path = os.path.join(DATA_DIR, "train.csv.zip")
test_path = os.path.join(DATA_DIR, "test.csv.zip")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train_df["any_toxic"] = (train_df[label_cols].sum(axis=1) > 0).astype(int)

train_df, valid_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["any_toxic"]
)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)

print("Train shape:", train_df.shape)
print("Valid shape:", valid_df.shape)
print("Test shape:", test_df.shape)
print("Label columns:", label_cols)



# Here I choose roberta as base model and tokenize text
from datasets import Dataset

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

max_length = 320

def tokenize_function(examples):
    return tokenizer(
        examples["comment_text"],
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

train_dataset = Dataset.from_pandas(train_df[["comment_text"] + label_cols], preserve_index=False)
valid_dataset = Dataset.from_pandas(valid_df[["comment_text"] + label_cols], preserve_index=False)
test_dataset = Dataset.from_pandas(test_df[["comment_text"]], preserve_index=False)

train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"] + label_cols)
valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"] + label_cols)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

print(train_dataset)
print(valid_dataset)
print(test_dataset)



# Here I am loading RoBERTa and turn on gradient checkpointing
def add_labels(batch):
    labels = np.stack([batch[c] for c in label_cols], axis=-1)
    batch["labels"] = labels.astype("float32")
    return batch

train_dataset = train_dataset.map(add_labels, batched=True)
valid_dataset = valid_dataset.map(add_labels, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_cols),
    problem_type="multi_label_classification"
)

model.gradient_checkpointing_enable()
model.to(device)

print(model.config.num_labels, "labels")
print("Gradient checkpointing:", model.is_gradient_checkpointing)



# I am setting up optimizer and cosine scheduler with warmup, and FP16 scaler
from torch.utils.data import DataLoader
from torch.optim import AdamW

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

num_epochs = 2
num_training_steps = num_epochs * len(train_loader)
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
loss_fn = torch.nn.BCEWithLogitsLoss()

def evaluate():
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    probs = 1 / (1 + np.exp(-all_logits))
    scores = []
    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) < 2:
            continue
        scores.append(roc_auc_score(all_labels[:, i], probs[:, i]))
    model.train()
    return float(np.mean(scores))

print("Train batches:", len(train_loader))
print("Valid batches:", len(valid_loader))
print("Total training steps:", num_training_steps, "Warmup steps:", num_warmup_steps)



# Actual training loop started
best_roc_auc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()
        if (step + 1) % 200 == 0:
            avg_loss = running_loss / 200
            print(f"Epoch {epoch+1}/{num_epochs} Step {step+1}/{len(train_loader)} Loss {avg_loss:.4f}")
            running_loss = 0.0

    val_roc_auc = evaluate()
    print(f"Epoch {epoch+1} validation ROC-AUC: {val_roc_auc:.5f}")

    if val_roc_auc > best_roc_auc:
        best_roc_auc = val_roc_auc
        torch.save(model.state_dict(), "best_roberta_toxic.pt")
        print("New best model saved with ROC-AUC:", best_roc_auc)

print("Best validation ROC-AUC:", best_roc_auc)



sample_path = os.path.join(DATA_DIR, "sample_submission.csv.zip")
submission = pd.read_csv(sample_path)

model.load_state_dict(torch.load("best_roberta_toxic.pt", map_location=device))
model.to(device)
model.eval()

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

all_probs = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

all_probs = np.concatenate(all_probs, axis=0)

for i, col in enumerate(label_cols):
    submission[col] = all_probs[:, i]

submission.to_csv("submission2.csv", index=False)
print(submission.head())
print("Saved submission2.csv")


