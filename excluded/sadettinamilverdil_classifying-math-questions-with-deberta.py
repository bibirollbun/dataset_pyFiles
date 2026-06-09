!pip install -U transformers datasets evaluate scikit-learn


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.model_selection import train_test_split



train=pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")
print("Train",train.shape)
test=pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")
print("Test",test.shape)
sample=pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/sample_submission.csv")
print("Sample Submission",sample.shape)


print(train.isnull().sum())
print("-----------------")
print(test.isnull().sum())


plt.figure(figsize=(10, 5))
sns.countplot(data=train, x='label', palette='viridis')
plt.title("Label Distribution in Training Data")
plt.xlabel("Label (Topic ID)")
plt.ylabel("Number of Questions")
plt.grid(axis='y')
plt.show()

distribution = train['label'].value_counts(normalize=True) * 100  
print(distribution)


train["text_length"] = train["Question"].astype(str).apply(lambda x: len(x.split()))

plt.figure(figsize=(10, 5))
sns.histplot(train["text_length"], bins=30, kde=True, color='teal')
plt.title("Distribution of Question Lengths")
plt.xlabel("Number of Words in Question")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


train_texts = train['Question'].tolist()
train_labels = train['label'].tolist()
test_texts = test['Question'].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.1, stratify=train_labels, random_state=42
)

model_name = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)

class MathDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = MathDataset(train_encodings, train_labels)
val_dataset = MathDataset(val_encodings, val_labels)
test_dataset = MathDataset(test_encodings)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8)
model.to(device)


optimizer = AdamW(model.parameters(), lr=2e-5)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, pin_memory=True)


scaler = GradScaler()
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast():
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_train_loss += loss.item()
        progress_bar.set_postfix(train_loss=loss.item())

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Avg Train Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast():
                outputs = model(**batch)
                loss = outputs.loss

            total_val_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            labels = batch['labels']
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    val_acc = correct / total
    print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")


model.eval()
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())


sample['label'] = all_preds
sample.to_csv("submission.csv", index=False)
sample.head()

