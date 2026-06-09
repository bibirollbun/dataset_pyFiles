import pandas as pd

train=pd.read_csv("/kaggle/input/mercor-ai-detection/train.csv")
test=pd.read_csv("/kaggle/input/mercor-ai-detection/test.csv")


# ==========================================================
# Mercor AI Text Detection - DeBERTa-v3-small 5-Fold CV
# With per-epoch AUC tracking, best model saving, and ensemble prediction
# ==========================================================

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ==========================================================
# Load Data
# ==========================================================
# Assuming `train` and `test` DataFrames are already loaded
# train = pd.read_csv("train.csv")
# test = pd.read_csv("test.csv")

train['answer'] = train['answer'].astype(str)
train['topic'] = train['topic'].astype(str)
test['answer'] = test['answer'].astype(str)
test['topic'] = test['topic'].astype(str)

# ==========================================================
# Parameters
# ==========================================================
MODEL_NAME = "/kaggle/input/modernbert/transformers/base/2"
OUTPUT_DIR = "./deberta_v3_checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 12
LR = 1e-5
FOLDS = 5
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
# Cross Validation Setup
# ==========================================================
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

# ==========================================================
# Training Loop with Validation AUC Tracking
# ==========================================================
for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['is_cheating'])):
    print(f"\n========== Fold {fold+1} ==========")

    # Prepare datasets
    X_tr_topics = train['topic'].iloc[train_idx].tolist()
    X_tr_answers = train['answer'].iloc[train_idx].tolist()
    y_tr = train['is_cheating'].iloc[train_idx].tolist()

    X_val_topics = train['topic'].iloc[val_idx].tolist()
    X_val_answers = train['answer'].iloc[val_idx].tolist()
    y_val = train['is_cheating'].iloc[val_idx].tolist()

    train_dataset = TextDataset(X_tr_topics, X_tr_answers, y_tr)
    val_dataset = TextDataset(X_val_topics, X_val_answers, y_val)
    test_dataset = TextDataset(list(test['topic']), list(test['answer']))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Load model
    model = DebertaV2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_auc = 0
    best_model_path = os.path.join(OUTPUT_DIR, f"fold{fold+1}_best_model.pth")

    # Training epochs
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}"):
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

        avg_train_loss = total_loss / len(train_loader)

        # Validation after each epoch
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())

        val_preds = np.array(val_preds).flatten()
        val_auc = roc_auc_score(y_val, val_preds)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val AUC: {val_auc:.4f}")

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"âœ… Best model updated at epoch {epoch+1} (AUC={best_auc:.4f})")

    # Load best model for test prediction
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # OOF predictions
    val_fold_preds = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
            val_fold_preds.extend(torch.sigmoid(outputs).cpu().numpy())

    oof_preds[val_idx] = np.array(val_fold_preds).flatten()
    fold_auc = roc_auc_score(y_val, val_fold_preds)
    print(f"âœ… Fold {fold+1} Best ROC-AUC: {fold_auc:.4f}")

    # Test predictions using best model
    fold_test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
            fold_test_preds.extend(torch.sigmoid(outputs).cpu().numpy())

    test_preds += np.array(fold_test_preds).flatten() / FOLDS  # Ensemble average

# ==========================================================
# Overall OOF Score
# ==========================================================
oof_auc = roc_auc_score(train['is_cheating'], oof_preds)
print(f"\nğŸ�� Overall Out-of-Fold ROC-AUC: {oof_auc:.4f}")

# ==========================================================
# Submission
# ==========================================================
submission = pd.DataFrame({
    'id': test['id'],
    'is_cheating': test_preds
})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"âœ… Submission file saved as {submission_path}")



submission

