# Cell 0: avoid MessageFactory / protobuf issues
!pip install -q protobuf==3.20.*


# Cell 1: imports

import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


# Cell 2: load data

DATA_PATH = "/kaggle/input/assignment-6"

train_df = pd.read_csv(f"{DATA_PATH}/train.csv")
test_df  = pd.read_csv(f"{DATA_PATH}/test.csv")

label_cols = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

train_df.head()


# Cell 3: split into train / validation

all_texts  = train_df["comment_text"].tolist()
all_labels = train_df[label_cols].values

train_texts, val_texts, train_labels, val_labels = train_test_split(
    all_texts,
    all_labels,
    test_size=0.1,
    random_state=42,
)

# cap training size to keep time under control (e.g. 120k)
max_train_samples = 120000
if len(train_texts) > max_train_samples:
    train_texts  = train_texts[:max_train_samples]
    train_labels = train_labels[:max_train_samples]

len(train_texts), len(val_texts)


# Cell 4: tokenizer

model_name = "roberta-base"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
MAX_LEN = 128


# Cell 5: dataset

class ToxicDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts  = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx]).float(),
        }
        return item


# Cell 6: dataloaders

BATCH_SIZE = 16

train_dataset = ToxicDataset(train_texts, train_labels)
val_dataset   = ToxicDataset(val_texts,   val_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

len(train_dataset), len(val_dataset)


# Cell 7: train roberta-base with warmup and gradient accumulation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_cols),
    problem_type="multi_label_classification",
).to(device)

num_epochs = 2
grad_accum_steps = 2  # effective batch size 32

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

num_training_steps = len(train_loader) * num_epochs // grad_accum_steps
num_warmup_steps   = int(0.1 * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

global_step = 0
print(f"Training on {len(train_dataset)} samples for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    optimizer.zero_grad()
    for step, batch in enumerate(pbar):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        pbar.set_postfix({"loss": loss.item() * grad_accum_steps})

print("Training finished.")


# Cell 7.1: validation ROC-AUC

model.eval()
all_logits = []
all_true   = []

with torch.no_grad():
    for batch in val_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].cpu().numpy()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits.cpu().numpy()
        all_logits.append(logits)
        all_true.append(labels)

all_logits = np.vstack(all_logits)
all_true   = np.vstack(all_true)

val_probs = 1 / (1 + np.exp(-all_logits))
val_auc   = roc_auc_score(all_true, val_probs, average="macro")

print("Validation ROC-AUC:", val_auc)


# Cell 8: test predictions

test_texts = test_df["comment_text"].tolist()

test_dataset = ToxicDataset(
    test_texts,
    np.zeros((len(test_texts), len(label_cols))),
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.eval()
pred_logits = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pred_logits.append(outputs.logits.cpu().numpy())

pred_logits = np.vstack(pred_logits)
pred_probs  = 1 / (1 + np.exp(-pred_logits))


# Cell 9: create submission.csv

submission = pd.DataFrame(pred_probs, columns=label_cols)
submission.insert(0, "id", test_df["id"])

submission.to_csv("submission.csv", index=False)
submission.head()

