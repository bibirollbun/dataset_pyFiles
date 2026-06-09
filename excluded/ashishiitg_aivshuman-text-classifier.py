import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load data
df = pd.read_csv("/kaggle/input/ai-vs-human-content-detection-1000-record-in-2025/ai_human_content_detection_dataset.csv")
texts = df['text_content'].tolist()
labels = df['label'].tolist()

print(f"Device: {device}")
print(f"Rows: {len(df)} Label classes: {set(labels)}")

# Train / val split
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

print(f"Train / Val: {len(train_texts)} {len(val_texts)}")

# 2. Tokenizer & Dataset
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):  # max_len increased
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 3. Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.to(device)

# UNFREEZE all layers so entire model trains
for param in model.parameters():
    param.requires_grad = True

# 4. Optimizer & Weighted loss

# Compute class weights to handle imbalance
class_counts = np.bincount(train_labels)
weights = 1. / torch.tensor(class_counts, dtype=torch.float).to(device)
loss_fn = CrossEntropyLoss(weight=weights)

# Lower learning rate
optimizer = AdamW(model.parameters(), lr=5e-6)

# 5. Training loop

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    losses = []
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    return np.mean(losses)


def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    preds = []
    true_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            
            loss = loss_fn(logits, labels)
            losses.append(loss.item())
            
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = np.mean(losses)
    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds, zero_division=0)
    rec = recall_score(true_labels, preds, zero_division=0)
    f1 = f1_score(true_labels, preds, zero_division=0)
    
    return avg_loss, acc, prec, rec, f1

# 6. Run training

num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_acc, val_prec, val_rec, val_f1 = eval_model(model, val_loader, loss_fn, device)
    
    print(f"Epoch {epoch+1} finished. Avg train loss: {train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}   Acc: {val_acc:.4f}   Prec: {val_prec:.4f}   Rec: {val_rec:.4f}   F1: {val_f1:.4f}")

# Save model
model.save_pretrained("./ai_human_distilbert_manual")
tokenizer.save_pretrained("./ai_human_distilbert_manual")
print("Model saved to ./ai_human_distilbert_manual")



import torch

# Define your label mapping (adjust if your labels are reversed)
label_map = {0: "Human", 1: "AI-generated"}

def predict_text(text, model, tokenizer, device):
    model.eval()
    
    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    
    # Move inputs to device (CPU or GPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Compute probabilities
        probs = torch.softmax(logits, dim=1)
        probs_np = probs.cpu().numpy()[0]
        
        # Get predicted label index
        pred_idx = torch.argmax(probs, dim=1).item()
    
    # Map label index to name
    pred_label = label_map[pred_idx]
    
    print(f"Text: {text}")
    print(f"Prediction: {pred_label}")
    print(f"Probabilities: Human: {probs_np[0]:.4f}, AI-generated: {probs_np[1]:.4f}")

# Example usage
sample_text = "What he gave away and forgot—became everything I built my life on"
predict_text(sample_text, model, tokenizer, device)





