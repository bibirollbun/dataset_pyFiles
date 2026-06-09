import pandas as pd

train=pd.read_csv("/kaggle/input/mercor-ai-detection/train.csv")
test=pd.read_csv("/kaggle/input/mercor-ai-detection/test.csv")


# ==========================================================
# Mercor AI Text Detection - DeBERTa-v3-small (Full Data Training)
# ==========================================================

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from tqdm import tqdm

# ==========================================================
# Load Data
# ==========================================================
# train = pd.read_csv("train.csv")
# test = pd.read_csv("test.csv")

train['answer'] = train['answer'].astype(str)
train['topic'] = train['topic'].astype(str)
test['answer'] = test['answer'].astype(str)
test['topic'] = test['topic'].astype(str)

# ==========================================================
# Parameters
# ==========================================================
MODEL_NAME = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-base"
OUTPUT_DIR = "./deberta_v3_fulltrain"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 50  # reduce if overfitting
LR = 5e-7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# Dataset Class
# ==========================================================
class TextDataset(Dataset):
    def __init__(self, topics, answers, labels=None):
        self.texts = [t + " [SEP] " + a for t, a in zip(topics, answers)]
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# ==========================================================
# Tokenizer
# ==========================================================
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

# ==========================================================
# Prepare Datasets
# ==========================================================
train_dataset = TextDataset(train['topic'].tolist(), train['answer'].tolist(), train['is_cheating'].tolist())
test_dataset = TextDataset(test['topic'].tolist(), test['answer'].tolist())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ==========================================================
# Model, Optimizer, Loss
# ==========================================================
model = DebertaV2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = torch.nn.BCEWithLogitsLoss()

# ==========================================================
# Training Loop
# ==========================================================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device).unsqueeze(1)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = loss_fn(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

# Save final model
final_model_path = os.path.join(OUTPUT_DIR, "final_model.pth")
torch.save(model.state_dict(), final_model_path)
print(f"✅ Final model saved to {final_model_path}")

# ==========================================================
# Test Predictions
# ==========================================================
model.eval()
test_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting on test set"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
        test_preds.extend(torch.sigmoid(outputs).cpu().numpy())

test_preds = np.array(test_preds).flatten()

# ==========================================================
# Submission
# ==========================================================
submission = pd.DataFrame({
    'id': test['id'],
    'is_cheating': test_preds
})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"✅ Submission file saved as {submission_path}")



submission

