# =========================================================
# MULTI-LABEL DistilBERT FOR JIGSAW COMPETITION
# =========================================================

!pip install transformers accelerate -q

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)

from tqdm.auto import tqdm


# =========================================================
# 1. Load Data (All 6 labels)
# =========================================================
train_df = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
test_df  = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# =========================================================
# 2. Dataset Class for Multi-Label
# =========================================================
class ToxicDataset(Dataset):
    def __init__(self, texts, labels_df=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels_df = labels_df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoded.input_ids.squeeze(),
            "attention_mask": encoded.attention_mask.squeeze(),
        }

        if self.labels_df is not None:
            item["labels"] = torch.tensor(self.labels_df.iloc[idx].values, dtype=torch.float)

        return item


# =========================================================
# 3. Tokenizer & Data Loaders
# =========================================================
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_dataset = ToxicDataset(
    train_df["comment_text"].values,
    train_df[labels],
    tokenizer
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = ToxicDataset(
    test_df["comment_text"].values,
    labels_df=None,
    tokenizer=tokenizer
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# =========================================================
# 4. Model Setup â€” 6 LABELS
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=6,              # IMPORTANT
    problem_type="multi_label_classification"
).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

epochs = 2  # 3 gives higher scores
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


# =========================================================
# 5. Training Loop
# =========================================================
print("\nğŸš€ Starting Multi-Label Training...\n")
model.train()

for epoch in range(epochs):
    print(f"\n===== EPOCH {epoch+1}/{epochs} =====")
    epoch_loss = 0

    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_batch
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        progress.set_postfix({"loss": loss.item()})

    print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.5f}")


# =========================================================
# 6. Inference
# =========================================================
print("\nğŸ”� Running inference...")

model.eval()
predictions = []

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        predictions.append(probs)

        if i % 500 == 0:
            print(f"Inference {i}/{len(test_loader)} batches...")

predictions = np.vstack(predictions)


# =========================================================
# 7. Create Correct Submission File
# =========================================================
submission = pd.DataFrame(predictions, columns=labels)
submission.insert(0, "id", test_df["id"])

submission.to_csv("submission.csv", index=False)

print("\nâœ… Submission ready!  submission.csv")
submission.head()


