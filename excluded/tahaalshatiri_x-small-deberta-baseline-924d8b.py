import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import torch.optim as optim
from tqdm.auto import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, get_cosine_schedule_with_warmup
)
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
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
train_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv") 
new_data = pd.read_csv("/kaggle/input/math-problem-classification-data/train_augmented.csv")
train_df = pd.concat([train_df, new_data]).reset_index(drop=True).sample(frac=1).reset_index(drop=True)
# Columns: "Question", "label"

test_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")  
# Step 1: Load pseudo-labeled data
pseudo_df = pd.read_csv("/kaggle/input/notebooke8bffe972e/submission_ensemble.csv")  # should have columns: "id", "label"

# Step 2: Merge pseudo labels with test questions
pseudo_labeled_test = test_df.merge(pseudo_df, on="id")  # now has columns: id, Question, label

# Step 3: Drop 'id' column to match train format
pseudo_labeled_test = pseudo_labeled_test.drop(columns=["id"])

# Step 4: Concatenate with the original training data
extended_train_df = pd.concat([train_df, pseudo_labeled_test], ignore_index=True)

# (Optional) Check the new shape
print("Original train size:", len(train_df))
print("Pseudo-labeled size:", len(pseudo_labeled_test))
print("New train size:", len(extended_train_df))
train_df =extended_train_df

test_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")    # Columns: "id", "Question"

train_texts = train_df["Question"].tolist()
train_labels = train_df["label"].tolist()
test_texts = test_df["Question"].tolist()

# Determine the number of classes
num_classes = len(np.unique(train_labels))

# Initialize the tokenizer (using a BERT-style model)
model_name = "microsoft/deberta-v3-large"
num_folds = 8
max_length = 256
batch_size = 4  # per device
accumulate_steps = 4  # effective batch size = batch_size * accumulate_steps
num_epochs = 5
learning_rate = 1e-5
weight_decay = 0.01
warmup_ratio = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data ---
train_texts = train_df["Question"].tolist()
train_labels = train_df["label"].tolist()
test_texts = test_df["Question"].tolist()
num_classes = len(np.unique(train_labels))

tokenizer = AutoTokenizer.from_pretrained(model_name)





# (Optional) compute class weights to handle imbalance
class_counts = np.bincount(train_labels)
class_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

data_collator = DataCollatorWithPadding(tokenizer)



# Storage
oof_preds = np.zeros(len(train_texts), dtype=int)
test_probs = np.zeros((len(test_texts), num_classes))
fold_f1_micro = []
fold_f1_scores = []

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler):
    model.train()

    total_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.cuda.amp.autocast():  # <-- autocast for mixed precision
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            logits = outputs.logits
            loss = criterion(logits, batch['labels'])
            loss = loss / accumulate_steps

        scaler.scale(loss).backward()  # <-- scaled backward pass

        if (step + 1) % accumulate_steps == 0:
            scaler.step(optimizer)     # <-- scaled optimizer step
            scaler.update()            # <-- update scaler for next iteration
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulate_steps

    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            logits = outputs.logits
            batch_preds = logits.argmax(dim=1).cpu().numpy()
            preds.extend(batch_preds)
            trues.extend(batch['labels'].cpu().numpy())
    return np.array(preds), np.array(trues)

def predict_probas(model, dataloader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)



train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

# Prepare test dataset
test_dataset = TextDataset(test_texts, None, tokenizer, max_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

# Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

total_steps = len(train_loader) * num_epochs // accumulate_steps
warmup_steps = int(warmup_ratio * total_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# Mixed precision
scaler = torch.cuda.amp.GradScaler()

# Train
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler)
    print(f"Epoch {epoch+1} â€” Train Loss: {train_loss:.4f}")

# Predict on test set
test_probs = predict_probas(model, test_loader)
final_test_preds = np.argmax(test_probs, axis=1)




np.save("test_probs.npy", test_probs)


submission = pd.DataFrame({
    "id": test_df["id"],
    "label": final_test_preds
})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")


