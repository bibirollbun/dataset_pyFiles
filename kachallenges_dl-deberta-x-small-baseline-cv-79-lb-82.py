import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import torch.optim as optim
from tqdm.auto import tqdm

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a custom dataset for our text data
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels  # Can be None for test data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Remove batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item



# Load your CSV files
train_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")  # Columns: "Question", "label"
test_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")    # Columns: "id", "Question"

train_texts = train_df["Question"].tolist()
train_labels = train_df["label"].tolist()
test_texts = test_df["Question"].tolist()

# Determine the number of classes
num_classes = len(np.unique(train_labels))

# Initialize the tokenizer (using a BERT-style model)
model_name = "microsoft/deberta-v3-xsmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)



def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Move all tensor data in the batch to the device
        for key in batch:
            batch[key] = batch[key].to(device)
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    true_labels = []
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        for key in batch:
            batch[key] = batch[key].to(device)
        outputs = model(**batch)
        logits = outputs.logits
        batch_preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        preds.extend(batch_preds)
        if "labels" in batch:
            true_labels.extend(batch["labels"].detach().cpu().numpy())
    return np.array(preds), np.array(true_labels)

def predict_probas(model, dataloader, device):
    model.eval()
    all_probs = []
    for batch in tqdm(dataloader, desc="Predicting", leave=False):
        for key in batch:
            batch[key] = batch[key].to(device)
        outputs = model(**batch)
        probs = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)



NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

batch_size = 4
data_collator = DataCollatorWithPadding(tokenizer)

# Arrays for storing OOF predictions and accumulating test probabilities
oof_preds = np.zeros(len(train_texts), dtype=int)
test_probs = np.zeros((len(test_texts), num_classes))
fold_f1_micro = []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
    print(f"===== Fold {fold+1} / {NUM_FOLDS} =====")
    # Prepare fold-specific training and validation data
    X_trn = [train_texts[i] for i in train_idx]
    y_trn = [train_labels[i] for i in train_idx]
    X_val = [train_texts[i] for i in val_idx]
    y_val = [train_labels[i] for i in val_idx]
    
    train_dataset = TextDataset(X_trn, y_trn, tokenizer, max_length=128)
    val_dataset = TextDataset(X_val, y_val, tokenizer, max_length=128)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    
    # Initialize a new model for this fold
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 2
    for epoch in range(num_epochs):
        # Evaluate after each epoch
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_preds, val_true = evaluate(model, val_loader, device)
        epoch_f1 = f1_score(val_true, val_preds, average="micro")
        print(f"Fold {fold+1}, Epoch {epoch+1}, Loss: {train_loss:.4f}, F1 (micro): {epoch_f1:.4f}")
        
    
    # Evaluate on validation set
    val_preds, val_true = evaluate(model, val_loader, device)
    oof_preds[val_idx] = val_preds
    fold_f1 = f1_score(val_true, val_preds, average="micro")
    fold_f1_micro.append(fold_f1)
    print(f"Fold {fold+1} F1 (micro): {fold_f1:.4f}")
    
    # Predict probabilities on the test set
    test_dataset = TextDataset(test_texts, labels=None, tokenizer=tokenizer, max_length=128)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    fold_test_probs = predict_probas(model, test_loader, device)
    test_probs += fold_test_probs

overall_oof_f1 = f1_score(train_labels, oof_preds, average="micro")
print(f"Overall OOF F1 (micro): {overall_oof_f1:.4f}")

# Average the test set probabilities over all folds
test_probs /= NUM_FOLDS
final_test_preds = np.argmax(test_probs, axis=1)



submission = pd.DataFrame({
    "id": test_df["id"],
    "label": final_test_preds
})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")


