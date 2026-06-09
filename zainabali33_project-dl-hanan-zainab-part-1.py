# Cell 1: Install compatible versions
!pip install -q "protobuf<=3.20.3" \
                "transformers==4.40.0" \
                "datasets==2.19.1" \
                "accelerate==0.30.1" \
                "evaluate"


# Cell 1: Imports

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict

from transformers import AutoTokenizer, DistilBertForSequenceClassification


# Cell 3: Load data, handling .json or .json.zip

import zipfile

DATA_DIR = "/kaggle/input/whats-cooking"

def load_json_maybe_zipped(base_name):
    """
    Try to load base_name.json directly.
    If not found, look for base_name*.zip and read the .json file inside.
    """
    json_path = os.path.join(DATA_DIR, f"{base_name}.json")
    if os.path.exists(json_path):
        # Plain JSON file
        print(f"Loading {json_path}")
        return pd.read_json(json_path)
    
    # Otherwise look for a zip file
    zip_candidates = [f for f in os.listdir(DATA_DIR) 
                      if f.startswith(base_name) and f.endswith(".zip")]
    if not zip_candidates:
        raise FileNotFoundError(f"No {base_name}.json or {base_name}*.zip found in {DATA_DIR}")
    
    zip_path = os.path.join(DATA_DIR, zip_candidates[0])
    print(f"Loading {base_name} from ZIP: {zip_path}")
    
    with zipfile.ZipFile(zip_path, "r") as z:
        # Pick the actual .json file inside, ignore __MACOSX stuff
        json_names = [n for n in z.namelist() 
                      if n.endswith(".json") and "__MACOSX" not in n]
        if not json_names:
            raise ValueError(f"No JSON file found inside {zip_path}")
        json_name = json_names[0]
        print(f"  â†’ Reading inner file: {json_name}")
        with z.open(json_name) as f:
            return pd.read_json(f)

# Use the helper for train & test
train_df = load_json_maybe_zipped("train")
test_df  = load_json_maybe_zipped("test")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

train_df.head()


# Cell 4: Basic info about columns and types

print("Train info:")
print(train_df.info())
print("\nTrain example row:")
print(train_df.iloc[0])

print("\nColumns:", train_df.columns.tolist())
print("\nUnique cuisines:", train_df['cuisine'].nunique())
print("Sample cuisines:", train_df['cuisine'].unique()[:10])


# Cell 5: Cuisine distribution (counts & bar plot)

cuisine_counts = train_df['cuisine'].value_counts().sort_values(ascending=False)
print(cuisine_counts)

plt.figure(figsize=(10, 6))
sns.barplot(x=cuisine_counts.index, y=cuisine_counts.values)
plt.xticks(rotation=45, ha="right")
plt.title("Number of recipes per cuisine")
plt.ylabel("Count")
plt.xlabel("Cuisine")
plt.tight_layout()
plt.show()


# Cell 6: Number of ingredients distribution

train_df['num_ingredients'] = train_df['ingredients'].apply(len)

print("Average number of ingredients:", train_df['num_ingredients'].mean())
print("Min ingredients:", train_df['num_ingredients'].min())
print("Max ingredients:", train_df['num_ingredients'].max())

plt.figure(figsize=(8, 5))
sns.histplot(train_df['num_ingredients'], bins=30, kde=False)
plt.title("Distribution of number of ingredients per recipe")
plt.xlabel("Number of ingredients")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# Cell 7: View some example recipes

for i in range(3):
    row = train_df.sample(1).iloc[0]
    print(f"ID: {row['id']}")
    print(f"Cuisine: {row['cuisine']}")
    print("Ingredients:")
    print(", ".join(row['ingredients']))
    print("-" * 80)


# Cell 8: Create 'text' column and label encode cuisines

# Join ingredients into one string per recipe: "ingredient1, ingredient2, ..."
train_df['text'] = train_df['ingredients'].apply(lambda ings: ", ".join(ings))
test_df['text']  = test_df['ingredients'].apply(lambda ings: ", ".join(ings))

# Create label mapping: cuisine -> id (0 ... num_classes-1)
cuisines = sorted(train_df['cuisine'].unique())
label2id = {c: i for i, c in enumerate(cuisines)}
id2label = {i: c for c, i in label2id.items()}

train_df['label'] = train_df['cuisine'].map(label2id)

num_labels = len(cuisines)
print("Number of labels (cuisines):", num_labels)
print("label2id:", label2id)


# Cell 9: Train/validation split (stratified)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df['label'],
    random_state=42
)

print("Train split shape:", train_df.shape)
print("Val split shape:", val_df.shape)

train_df.head()


# Cell 10: Convert pandas DataFrames to HF Datasets

train_dataset = Dataset.from_pandas(train_df[['text', 'label']].reset_index(drop=True))
val_dataset   = Dataset.from_pandas(val_df[['text', 'label']].reset_index(drop=True))
test_dataset  = Dataset.from_pandas(test_df[['text']].reset_index(drop=True))

raw_datasets = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

raw_datasets


# Cell 11: Load tokenizer (DistilBERT)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = 256  # you can tune this later


# Cell 12 (fixed): tokenize each split and handle labels correctly

from datasets import DatasetDict

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

# Tokenize each split separately
tokenized_train = raw_datasets["train"].map(tokenize_function, batched=True)
tokenized_val   = raw_datasets["validation"].map(tokenize_function, batched=True)
tokenized_test  = raw_datasets["test"].map(tokenize_function, batched=True)

# Rename 'label' -> 'labels' ONLY for train and validation (test has no labels)
tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_val   = tokenized_val.rename_column("label", "labels")

# Remove 'text' column (model doesn't need it)
tokenized_train = tokenized_train.remove_columns(["text"])
tokenized_val   = tokenized_val.remove_columns(["text"])
tokenized_test  = tokenized_test.remove_columns(["text"])

# Remove auto index column if it exists
for ds_name, ds in [("train", tokenized_train), ("validation", tokenized_val), ("test", tokenized_test)]:
    cols = ds.column_names
    if "__index_level_0__" in cols:
        ds = ds.remove_columns(["__index_level_0__"])
    if ds_name == "train":
        tokenized_train = ds
    elif ds_name == "validation":
        tokenized_val = ds
    else:
        tokenized_test = ds

# Rebuild DatasetDict
tokenized_datasets = DatasetDict({
    "train": tokenized_train,
    "validation": tokenized_val,
    "test": tokenized_test,
})

tokenized_datasets


# Cell 13 (fixed): Load DistilBERT for sequence classification

model = DistilBertForSequenceClassification.from_pretrained(
    model_name,           # "distilbert-base-uncased"
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Using device:", device)


# Cell 14: Define accuracy metric for Trainer

import evaluate
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)


# Cell 15: Prepare PyTorch DataLoaders with progress printing in mind

from torch.utils.data import DataLoader

batch_size = 16 
epochs = 5        

# Make HF datasets return PyTorch tensors
train_torch = tokenized_datasets["train"].with_format("torch")
val_torch   = tokenized_datasets["validation"].with_format("torch")

train_loader = DataLoader(train_torch, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_torch, batch_size=batch_size, shuffle=False)

len(train_loader), len(val_loader)


# ğŸ”� Replace your optimizer/scheduler cell with this

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

learning_rate = 5e-5   # was 2e-5 â†’ try a bit higher
epochs = 5             # you can keep 5 for now

optimizer = AdamW(model.parameters(), lr=learning_rate)

total_steps = len(train_loader) * epochs

# No warmup (sometimes small datasets + warmup = too gentle)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps,
)


from tqdm.auto import tqdm
import torch

def evaluate(model, dataloader):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_steps_epoch = len(train_loader)

    print(f"\n===== Epoch {epoch+1}/{epochs} =====")

    for step, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch+1}"), start=1):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()

        # ğŸ”¹ gradient clipping â€“ helps stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        total_correct += (preds == batch["labels"]).sum().item()
        total_samples += batch["labels"].size(0)

        if step % 100 == 0 or step == total_steps_epoch:
            avg_loss_so_far = total_loss / step
            train_acc_so_far = total_correct / total_samples
            print(
                f"  Step {step}/{total_steps_epoch} - "
                f"avg train loss: {avg_loss_so_far:.4f}, "
                f"avg train acc: {train_acc_so_far:.4f}"
            )

    # ğŸ”¹ end-of-epoch validation
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1} finished. Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")

print("\nTraining complete.")


# Make Kaggle submission.csv from test set

from torch.utils.data import DataLoader

# 1) Create a PyTorch dataset for test
test_torch = tokenized_datasets["test"].with_format("torch")

test_loader = DataLoader(test_torch, batch_size=32, shuffle=False)

model.eval()
all_pred_ids = []

with torch.no_grad():
    for batch in test_loader:
        # batch has input_ids, attention_mask (no labels)
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        all_pred_ids.extend(preds.cpu().numpy())

# 2) Map predicted IDs to cuisine names
pred_cuisines = [id2label[int(i)] for i in all_pred_ids]

# 3) Build submission DataFrame (id + cuisine)
submission = pd.DataFrame({
    "id": test_df["id"],
    "cuisine": pred_cuisines
})

# Save to CSV in working directory
submission.to_csv("submission.csv", index=False)
submission.head()


# New Cell 18: Save the fine-tuned model

output_dir = "./distilbert-whatscooking-manual"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Model and tokenizer saved to", output_dir)


# Cell A: Build PyTorch DataLoaders from tokenized datasets

from torch.utils.data import DataLoader

train_torch = tokenized_train.with_format("torch")
val_torch   = tokenized_val.with_format("torch")

batch_size = 16
epochs = 4  # you can tune

train_loader = DataLoader(train_torch, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_torch, batch_size=32, shuffle=False)

len(train_loader), len(val_loader)



# Cell B: Optimizer + LR scheduler for BERT

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

learning_rate = 2e-5

optimizer = AdamW(model.parameters(), lr=learning_rate)

total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
    num_training_steps=total_steps,
)



# Cell C: Manual training loop with progress + validation accuracy

from tqdm.auto import tqdm
import torch

def evaluate(model, dataloader):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_steps_epoch = len(train_loader)

    print(f"\n===== Epoch {epoch+1}/{epochs} =====")

    for step, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch+1}"), start=1):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        total_correct += (preds == batch["labels"]).sum().item()
        total_samples += batch["labels"].size(0)

        if step % 100 == 0 or step == total_steps_epoch:
            avg_loss_so_far = total_loss / step
            train_acc_so_far = total_correct / total_samples
            print(
                f"  Step {step}/{total_steps_epoch} - "
                f"avg train loss: {avg_loss_so_far:.4f}, "
                f"avg train acc: {train_acc_so_far:.4f}"
            )

    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1} finished. Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")

print("\nTraining complete.")



# Cell D: Save fine-tuned BERT model

output_dir = "./bert-whatscooking"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Model and tokenizer saved to", output_dir)


# Make Kaggle submission.csv from test set

from torch.utils.data import DataLoader

# 1) Create a PyTorch dataset for test
test_torch = tokenized_datasets["test"].with_format("torch")

test_loader = DataLoader(test_torch, batch_size=32, shuffle=False)

model.eval()
all_pred_ids = []

with torch.no_grad():
    for batch in test_loader:
        # batch has input_ids, attention_mask (no labels)
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        all_pred_ids.extend(preds.cpu().numpy())

# 2) Map predicted IDs to cuisine names
pred_cuisines = [id2label[int(i)] for i in all_pred_ids]

# 3) Build submission DataFrame (id + cuisine)
submission = pd.DataFrame({
    "id": test_df["id"],
    "cuisine": pred_cuisines
})

# Save to CSV in working directory
submission.to_csv("submission.csv", index=False)
submission.head()


from transformers import AutoTokenizer, DistilBertForSequenceClassification

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = 256   # was 128 before, bump this



from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

batch_size = 32       # if GPU fits; else keep 16
epochs = 8            # weâ€™ll do early stopping, so 8 is okay
learning_rate = 3e-5  # sweet spot for DistilBERT usually

train_torch = tokenized_train.with_format("torch")
val_torch   = tokenized_val.with_format("torch")

from torch.utils.data import DataLoader
train_loader = DataLoader(train_torch, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_torch, batch_size=64, shuffle=False)

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
    num_training_steps=total_steps,
)



from tqdm.auto import tqdm
import torch

def evaluate(model, dataloader):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

best_val_acc = 0.0
best_state_dict = None
patience = 2          # stop if no improvement for 2 epochs
patience_counter = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_steps_epoch = len(train_loader)

    print(f"\n===== Epoch {epoch+1}/{epochs} =====")

    for step, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch+1}"), start=1):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        total_correct += (preds == batch["labels"]).sum().item()
        total_samples += batch["labels"].size(0)

        if step % 100 == 0 or step == total_steps_epoch:
            avg_loss_so_far = total_loss / step
            train_acc_so_far = total_correct / total_samples
            print(
                f"  Step {step}/{total_steps_epoch} - "
                f"avg train loss: {avg_loss_so_far:.4f}, "
                f"avg train acc: {train_acc_so_far:.4f}"
            )

    # end-of-epoch validation
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1} finished. Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")

    # early stopping logic
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
        print(f"ğŸŒŸ New best val accuracy: {best_val_acc:.4f}")
    else:
        patience_counter += 1
        print(f"No improvement. Patience {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

print("\nTraining complete. Best val accuracy:", best_val_acc)



if best_state_dict is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    print("Loaded best model weights (based on val accuracy).")


# FULL DistilBERT pipeline in one cell:
# - load data
# - preprocess & split
# - tokenize
# - train DistilBERT with early stopping
# - make submission.csv

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

from transformers import AutoTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup

# -------------------------
# 1. Load data (handles .json or .json.zip with __MACOSX)
# -------------------------

DATA_DIR = "/kaggle/input/whats-cooking"

import zipfile

def load_json_maybe_zipped(base_name):
    json_path = os.path.join(DATA_DIR, f"{base_name}.json")
    if os.path.exists(json_path):
        print(f"Loading {json_path}")
        return pd.read_json(json_path)
    zip_candidates = [f for f in os.listdir(DATA_DIR) if f.startswith(base_name) and f.endswith(".zip")]
    if not zip_candidates:
        raise FileNotFoundError(f"No {base_name}.json or {base_name}*.zip found in {DATA_DIR}")
    zip_path = os.path.join(DATA_DIR, zip_candidates[0])
    print(f"Loading {base_name} from ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        json_names = [n for n in z.namelist() if n.endswith(".json") and "__MACOSX" not in n]
        if not json_names:
            raise ValueError(f"No JSON file found inside {zip_path}")
        json_name = json_names[0]
        print(f"  â†’ Reading inner file: {json_name}")
        with z.open(json_name) as f:
            return pd.read_json(f)

train_df = load_json_maybe_zipped("train")
test_df  = load_json_maybe_zipped("test")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print(train_df.head())

# -------------------------
# 2. Preprocessing & label encoding
# -------------------------

# join ingredients into a single string
train_df["text"] = train_df["ingredients"].apply(lambda ings: ", ".join(ings))
test_df["text"]  = test_df["ingredients"].apply(lambda ings: ", ".join(ings))

# label mapping
cuisines = sorted(train_df["cuisine"].unique())
label2id = {c: i for i, c in enumerate(cuisines)}
id2label = {i: c for c, i in label2id.items()}
num_labels = len(cuisines)

train_df["label"] = train_df["cuisine"].map(label2id)

print("Number of cuisines:", num_labels)

# -------------------------
# 3. Train/val split
# -------------------------

train_df, val_df = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df["label"],
    random_state=42,
)

print("Train split:", train_df.shape)
print("Val split:", val_df.shape)

# -------------------------
# 4. HF Dataset + tokenization
# -------------------------

raw_train = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
raw_val   = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True))
raw_test  = Dataset.from_pandas(test_df[["text"]].reset_index(drop=True))

raw_datasets = DatasetDict({
    "train": raw_train,
    "validation": raw_val,
    "test": raw_test,
})

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = 256   # larger than 128 to capture more ingredients

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

tokenized_train = raw_datasets["train"].map(tokenize_function, batched=True)
tokenized_val   = raw_datasets["validation"].map(tokenize_function, batched=True)
tokenized_test  = raw_datasets["test"].map(tokenize_function, batched=True)

tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_val   = tokenized_val.rename_column("label", "labels")

tokenized_train = tokenized_train.remove_columns(["text"])
tokenized_val   = tokenized_val.remove_columns(["text"])
tokenized_test  = tokenized_test.remove_columns(["text"])

for ds_name, ds in [("train", tokenized_train), ("validation", tokenized_val), ("test", tokenized_test)]:
    cols = ds.column_names
    if "__index_level_0__" in cols:
        ds = ds.remove_columns(["__index_level_0__"])
    if ds_name == "train":
        tokenized_train = ds
    elif ds_name == "validation":
        tokenized_val = ds
    else:
        tokenized_test = ds

# -------------------------
# 5. Model, DataLoaders
# -------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)
model.to(device)

train_torch = tokenized_train.with_format("torch")
val_torch   = tokenized_val.with_format("torch")

batch_size = 32
epochs = 8               # we'll use early stopping
learning_rate = 3e-5

train_loader = DataLoader(train_torch, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_torch, batch_size=64, shuffle=False)

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

# -------------------------
# 6. Training loop with early stopping
# -------------------------

def evaluate(model, dataloader):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item() * batch["labels"].size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

best_val_acc = 0.0
best_state_dict = None
patience = 2
patience_counter = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_steps_epoch = len(train_loader)

    print(f"\n===== Epoch {epoch+1}/{epochs} =====")

    for step, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch+1}"), start=1):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        total_correct += (preds == batch["labels"]).sum().item()
        total_samples += batch["labels"].size(0)

        if step % 100 == 0 or step == total_steps_epoch:
            avg_loss_so_far = total_loss / step
            train_acc_so_far = total_correct / total_samples
            print(
                f"  Step {step}/{total_steps_epoch} - "
                f"avg train loss: {avg_loss_so_far:.4f}, "
                f"avg train acc: {train_acc_so_far:.4f}"
            )

    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1} finished. Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
        print(f"ğŸŒŸ New best val accuracy: {best_val_acc:.4f}")
    else:
        patience_counter += 1
        print(f"No improvement. Patience {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

print("\nTraining complete. Best val accuracy:", best_val_acc)

# load best weights
if best_state_dict is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    print("Loaded best model weights.")

# -------------------------
# 7. Inference on test + submission.csv
# -------------------------

test_torch = tokenized_test.with_format("torch")
test_loader = DataLoader(test_torch, batch_size=64, shuffle=False)

model.eval()
all_pred_ids = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        all_pred_ids.extend(preds.cpu().numpy())

pred_cuisines = [id2label[int(i)] for i in all_pred_ids]

submission = pd.DataFrame({
    "id": test_df["id"],
    "cuisine": pred_cuisines
})

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
print(submission.head())

