import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, AutoConfig, AutoModel
import torch.optim as optim
import sklearn.utils.class_weight as scikit_class_weight
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels  # Can be None for test data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        # Remove batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item



def preprocess(question):
    question = question.split()
    
    if(len(question) > 256):
        return " ".join(question[:256])
    else: 
        return " ".join(question)

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Move all tensor data in the batch to the device
        for key in batch:
            batch[key] = batch[key].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'])
        loss = loss_fn(outputs.logits, batch['labels'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, scalar = True):
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
    if scalar:
        return f1_score(np.array(true_labels), np.array(preds), average="micro")
    else:
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


train_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")  # Columns: "Question", "label", Len: 10189
exteneded = pd.read_csv("/kaggle/input/math-problem-classification-data/train_augmented.csv")
train_df = pd.concat([train_df, exteneded], axis = 0).sample(frac=1)
train_df['Question'] = train_df['Question'].apply(preprocess)
train_texts = train_df["Question"].tolist()
train_labels = train_df["label"].tolist()
val_texts = train_texts[20000:]
val_labels = train_labels[20000:]
train_texts = train_texts[:20000]
train_labels = train_labels[:20000]

class_list = list(set(train_labels))
class_weight = scikit_class_weight.compute_class_weight(class_weight ='balanced', classes = class_list, y = train_labels)
class_weight = torch.FloatTensor(class_weight).to(device)
loss_fn = CrossEntropyLoss(weight=class_weight) #label_smoothing=0.1


model_name = "/kaggle/input/qwen2.5/transformers/0.5b/1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8)
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
model.config.pad_token_id = tokenizer.all_special_ids[0]


batch_size = 4
num_classes = 8
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
data_collator = DataCollatorWithPadding(tokenizer)

train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=data_collator)

# param_list = list(model.model.parameters())
# total = len(param_list)
for fold, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
    X_trn = [train_texts[i] for i in train_idx]
    y_trn = [train_labels[i] for i in train_idx]
    
    train_dataset_fold = TextDataset(X_trn, y_trn, tokenizer)
    train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    
    # Evaluate after each epoch
    train_loss = train_epoch(model, train_loader_fold, optimizer, loss_fn, device)
    #scheduler.step()
    val_preds, val_true = evaluate(model, val_loader, device, False)
    epoch_f1 = f1_score(val_true, val_preds, average="micro")
    print(f"Epoch {fold+1}, Loss: {train_loss:.4f}, F1 (micro): {epoch_f1:.4f}")


test_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")    # Columns: "id", "Question", Len: 3044
test_df['Question'] = test_df['Question'].apply(preprocess)
test_texts = test_df["Question"].tolist()
test_loader = TextDataset(test_texts, None, tokenizer)
test_loader = DataLoader(test_loader, batch_size=2, shuffle=False, collate_fn=data_collator)
final_test_preds = np.argmax(predict_probas(model, test_loader, device), axis=1)

submission = pd.DataFrame({
    "id": test_df["id"],
    "label": final_test_preds
})

submission.to_csv("submission.csv", index=False)

