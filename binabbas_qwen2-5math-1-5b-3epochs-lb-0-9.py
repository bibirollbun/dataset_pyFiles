import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import torch.optim as optim
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
from peft import get_peft_model, LoraConfig, TaskType
max_length = 1024  # Maximum length of the input sequences
# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a custom dataset for our text data
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=max_length):
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
train_df = pd.read_csv("data/train.csv")  # Columns: "Question", "label"
test_df = pd.read_csv("data/test.csv")    # Columns: "id", "Question"

import re

def clean(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train_df["Question"] = train_df["Question"].apply(clean)
test_df["Question"] = test_df["Question"].apply(clean)

train_texts = train_df["Question"].tolist()
train_labels = train_df["label"].tolist()
test_texts = test_df["Question"].tolist()

# Determine the number of classes
num_classes = len(np.unique(train_labels))

# Initialize the tokenizer
model_name = "Qwen/Qwen2.5-Math-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)



def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Move input batch to the initial device expected by the model (handled by device_map)
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
    with torch.no_grad(): # Disable gradient calculations for evaluation
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            for key in batch:
                batch[key] = batch[key].to(device)
            outputs = model(**batch)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            preds.extend(batch_preds)
            if "labels" in batch:
                true_labels.extend(batch["labels"].detach().cpu().numpy())

    # Calculate and print classification report
    if true_labels:
        report = classification_report(true_labels, preds, target_names=[f"Class {i}" for i in range(len(np.unique(true_labels)))])
        print(report)

    return np.array(preds), np.array(true_labels)

def predict_probas(model, dataloader, device):
    model.eval()
    all_probs = []
    with torch.no_grad(): # Disable gradient calculations for prediction
        for batch in tqdm(dataloader, desc="Predicting", leave=False):
            for key in batch:
                batch[key] = batch[key].to(device)
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits.to(torch.float32), dim=1).detach().cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)



import gc
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
from sklearn.model_selection import train_test_split

batch_size = 1
data_collator = DataCollatorWithPadding(tokenizer)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels
)

train_dataset = TextDataset(X_train, y_train, tokenizer, max_length=max_length)
val_dataset = TextDataset(X_val, y_val, tokenizer, max_length=max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    device_map="auto",
    num_labels=num_classes,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
)
model.config.pad_token_id = tokenizer.pad_token_id

# LoRA Config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
num_epochs = 3
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_preds, val_true = evaluate(model, val_loader, device)
    epoch_f1 = f1_score(val_true, val_preds, average="micro")
    print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, F1 (micro): {epoch_f1:.4f}")
    scheduler.step()

# Predict probabilities on the test set
test_dataset = TextDataset(test_texts, labels=None, tokenizer=tokenizer, max_length=max_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
test_probs = predict_probas(model, test_loader, device)
final_test_preds = np.argmax(test_probs, axis=1)



submission = pd.DataFrame({
    "id": test_df["id"],
    "label": final_test_preds
})
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")



probs_df = pd.DataFrame(test_probs, columns=[f"class_{i}_prob" for i in range(test_probs.shape[1])])
probs_df.insert(0, "id", test_df["id"])
probs_df.to_csv("test_probs.csv", index=False)
print("Test probabilities saved to test_probs.csv")

