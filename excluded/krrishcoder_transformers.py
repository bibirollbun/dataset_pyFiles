from google.protobuf import message_factory

# Patch for Kaggle submission: dummy GetPrototype
if not hasattr(message_factory.MessageFactory, "GetPrototype"):
    def _dummy(*args, **kwargs):
        return None
    message_factory.MessageFactory.GetPrototype = _dummy










import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW   # <-- correct optimizer import


# ============================================================
# 1) LOAD DATA
# ============================================================

df_train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
df_test  = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

texts = []
labels = []

# ============================================================
# 2) EXTRACT LABELED DATA FROM train.csv
# ============================================================

for _, row in df_train.iterrows():

    # main labeled training text
    texts.append(str(row["body"]))
    labels.append(int(row["rule_violation"]))

    # positive examples
    for col in ["positive_example_1", "positive_example_2"]:
        if pd.notna(row[col]):
            texts.append(str(row[col]))
            labels.append(1)

    # negative examples
    for col in ["negative_example_1", "negative_example_2"]:
        if pd.notna(row[col]):
            texts.append(str(row[col]))
            labels.append(0)

# ============================================================
# 3) EXTRACT LABELED DATA FROM test.csv (examples only)
# ============================================================

for _, row in df_test.iterrows():

    for col in ["positive_example_1", "positive_example_2"]:
        if pd.notna(row[col]):
            texts.append(str(row[col]))
            labels.append(1)

    for col in ["negative_example_1", "negative_example_2"]:
        if pd.notna(row[col]):
            texts.append(str(row[col]))
            labels.append(0)

print("Total samples:", len(texts))

# ============================================================
# 4) TRAIN/VAL SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.15, random_state=42, stratify=labels
)

# ============================================================
# 5) BERT TOKENIZER
# ============================================================

MODEL_NAME = "/kaggle/input/bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = 192     # works well for Jigsaw tasks

# ============================================================
# 6) CUSTOM DATASET
# ============================================================

class TextDataset(Dataset):
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
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

train_ds = TextDataset(X_train, y_train, tokenizer, MAX_LEN)
val_ds   = TextDataset(X_val,   y_val, tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)

# ============================================================
# 7) BERT MODEL
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
).to(device)


optimizer = AdamW(model.parameters(), lr=2e-5)

epochs = 2

# LR scheduler
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# ============================================================
# 8) TRAIN FUNCTION
# ============================================================

def train_one_epoch():
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        total_loss += loss.item()
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return total_loss / len(train_loader)

# ============================================================
# 9) VALIDATION FUNCTION
# ============================================================

def evaluate():
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            labels = batch["labels"]
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            logits = outputs.logits
            
            probs = torch.softmax(logits, dim=-1)[:,1]
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

# ============================================================
# 10) TRAINING LOOP
# ============================================================

from sklearn.metrics import classification_report, roc_auc_score

for epoch in range(epochs):
    print(f"\n===== Epoch {epoch+1}/{epochs} =====")
    tr_loss = train_one_epoch()
    print("Train Loss:", tr_loss)
    
    preds, probs, labels = evaluate()
    print("\nClassification Report:\n", classification_report(labels, preds))
    print("ROC-AUC:", roc_auc_score(labels, probs))

# ============================================================
# 11) INFERENCE ON Kaggle TEST
# ============================================================

test_texts = df_test["body"].astype(str).tolist()
test_ds = TextDataset(test_texts, labels=None, tokenizer=tokenizer, max_len=MAX_LEN)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

test_probs = []

model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        probs = torch.softmax(outputs.logits, dim=-1)[:,1]
        test_probs.extend(probs.cpu().numpy())

# ============================================================
# 12) SAVE SUBMISSION
# ============================================================

sub = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")
sub["rule_violation"] = test_probs
sub.to_csv("submission.csv", index=False)

print("Saved submission_bert.csv!")



sub




