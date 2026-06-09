# Cell 0: Fix broken Kaggle image by installing working Transformers version
!pip install transformers==4.40.2 --quiet
!pip install accelerate==0.28.0 --quiet


# Cell 1: Install required libraries
!pip install transformers==4.41.2 datasets accelerate -q


# Cell 2: Load training and test CSV files
import pandas as pd

train_df = pd.read_csv("/kaggle/input/assignment-6-1/train.csv")
test_df  = pd.read_csv("/kaggle/input/assignment-6-1/test.csv")

label_cols = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

train_df = train_df.dropna(subset=["comment_text"])
test_df  = test_df.fillna("")

train_texts = train_df["comment_text"].tolist()
test_texts  = test_df["comment_text"].tolist()


# Cell 3: Load tokenizer and tokenize the training and test texts

from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# tokenize training texts
train_enc = tokenizer(
    train_texts,
    truncation=True,
    padding="max_length",
    max_length=128,
)

# tokenize test texts
test_enc = tokenizer(
    test_texts,
    truncation=True,
    padding="max_length",
    max_length=128,
)

print("Tokenization done. Example input_ids length:", len(train_enc["input_ids"][0]))



# Cell 5: Define dataset class for PyTorch
import torch
from torch.utils.data import Dataset

class ToxicDataset(Dataset):
    def __init__(self, encodings, labels):
        self.enc = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

train_labels = train_df[label_cols].values
train_dataset = ToxicDataset(train_enc, train_labels)


# Cell 6: Create train/validation splits and data loaders

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# labels array from Cell 5
labels_array = train_labels

indices = np.arange(len(labels_array))

# stratify on "any toxic" vs "clean" to balance splits
stratify_target = (labels_array.sum(axis=1) > 0).astype(int)

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.1,
    random_state=42,
    stratify=stratify_target,
)

from torch.utils.data import Subset

train_subset = Subset(train_dataset, train_idx)
val_subset   = Subset(train_dataset, val_idx)

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_subset,   batch_size=64, shuffle=False)

len(train_loader), len(val_loader)


# Cell 7: Create model, optimizer, and scheduler

import torch
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

num_labels = len(label_cols)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="multi_label_classification",
)
model.to(device)

learning_rate = 3e-5
num_epochs = 2

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

criterion = torch.nn.BCEWithLogitsLoss()


# Cell 8: Train the model with a manual training loop

from tqdm.auto import tqdm

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1} finished. Avg train loss: {avg_train_loss:.4f}")


# Cell 9: Evaluate on validation set and print ROC-AUC

import numpy as np
from sklearn.metrics import roc_auc_score

model.eval()
val_loss = 0.0
all_logits = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Validating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        loss = criterion(logits, labels)
        val_loss += loss.item()

        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

avg_val_loss = val_loss / len(val_loader)
logits_array = np.vstack(all_logits)
labels_array = np.vstack(all_labels)

probs = 1 / (1 + np.exp(-logits_array))
val_auc = roc_auc_score(labels_array, probs, average="macro")

print(f"Validation loss: {avg_val_loss:.4f}")
print(f"Validation ROC-AUC: {val_auc:.6f}")


# Cell 10: Run inference on the test set and create submission file

from torch.utils.data import DataLoader

# reuse ToxicDataset from earlier, but labels are dummy zeros
dummy_labels = np.zeros((len(test_texts), len(label_cols)))
test_dataset = ToxicDataset(test_enc, dummy_labels)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()
test_logits = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting on test set"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        test_logits.append(outputs.logits.cpu().numpy())

test_logits = np.vstack(test_logits)
test_probs = 1 / (1 + np.exp(-test_logits))

submission = pd.DataFrame(test_probs, columns=label_cols)
submission.insert(0, "id", test_df["id"])

submission.to_csv("submission.csv", index=False)
submission.head()

