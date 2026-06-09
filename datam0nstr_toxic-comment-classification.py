# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install -q --upgrade \
    transformers \
    tokenizers \
    datasets \
    evaluate \
    accelerate \
    torchmetrics

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from torch import nn
from torch.optim import AdamW


train = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
test = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")
sample_submission = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")
test_labels = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()



# Count each toxic category
label_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

# Count how many 1s each column has
category_counts = train[label_cols].sum().sort_values(ascending=False)
print("âœ… Count of each label (how many comments have this label):")
print(category_counts)

# Total number of positive labels across all columns
total_positive_labels = category_counts.sum()
print("\nTotal number of label instances (sum of all 6 columns):", total_positive_labels)

# Number of comments that have at least one label
rows_with_any_label = (train[label_cols].any(axis=1)).sum()
print("Number of comments with at least one label:", rows_with_any_label)

# Number of comments with no label at all
rows_with_no_label = len(train) - rows_with_any_label
print("Number of comments with no label:", rows_with_no_label)

# (Optional) Distribution of how many labels each comment has (0â€“6)
label_combo_counts = train[label_cols].sum(axis=1).value_counts().sort_index()
print("\nHow many labels each comment typically has (0â€“6):")
print(label_combo_counts)



label_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

# 1) Show dtype of each label column
print(train[label_cols].dtypes)

# 2) Unique values per label (first 50 unique entries) â€” good to spot non-binary tokens
for c in label_cols:
    vals = train[c].dropna().astype(str).str.strip().unique()[:50]
    print(f"\n{c} unique examples (up to 50):\n", vals)

# 3) Count NaNs per label
print("\nNaN counts per label:")
print(train[label_cols].isna().sum())

# 4) Count values other than '0' or '1' (string or numeric)
for c in label_cols:
    bad = train[~train[c].isin([0,1]) & ~train[c].isin(['0','1'])][c]
    print(f"{c} â€” non 0/1 sample count:", len(bad))
    if len(bad) > 0:
        print("Example non-binary values:", bad.astype(str).unique()[:20])

# 5) Rows where labels sum > 6 or < 0 (impossible) â€” shows unexpected numeric values
sums = train[label_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
print("\nRows with label-sum < 0 or > 6 (should be none):", ((sums<0) | (sums>6)).sum())

# 6) Distribution of #labels per row (read-only)
print("\nDistribution (0..6) of number of positive labels per row:")
print(train[label_cols].apply(lambda s: pd.to_numeric(s, errors='coerce')).sum(axis=1).value_counts().sort_index())

# 7) Duplicate checking by 'id' (if id exists) and by comment text
if 'id' in train.columns:
    print("\nDuplicate id count:", train['id'].duplicated().sum())
    print("Unique duplicate id examples (up to 10):", train[train['id'].duplicated(keep=False)]['id'].unique()[:10])
if 'comment_text' in train.columns:
    print("\nDuplicate comment_text count:", train['comment_text'].duplicated().sum())

# 8) Show sample rows that are labeled but look suspicious (e.g., non-binary)
mask_nonbinary = False
for c in label_cols:
    mask_nonbinary |= ~train[c].isin([0,1]) & ~train[c].isin(['0','1'])
if mask_nonbinary.any():
    print("\nExample rows with non-binary label values (first 10):")
    print(train[mask_nonbinary].head(10)[['id','comment_text'] + label_cols])
else:
    print("\nNo non-binary label values found in the subset checked.")

# 9) Check leading/trailing whitespace or invisible chars in label columns
# Show rows where stringified label differs from stripped version
def has_whitespace_issue(col):
    s = train[col].astype(str)
    return (s != s.str.strip()).any()
for c in label_cols:
    print(f"{c} has leading/trailing whitespace?:", has_whitespace_issue(c))



# Label columns
label_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
y = train[label_cols]



import re

# Define cleaning patterns (same as found during EDA)
url_pattern = r"http\S+|www\.\S+"
html_pattern = r"<.*?>"
emoji_pattern = r"[\U00010000-\U0010ffff]"       # captures emojis and symbols
non_ascii_pattern = r"[^\x00-\x7F]+"             # captures non-ASCII chars
multi_space_pattern = r"\s+"                     # normalize spaces

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(url_pattern, " ", text)
    text = re.sub(html_pattern, " ", text)
    text = re.sub(emoji_pattern, " ", text)
    text = re.sub(non_ascii_pattern, " ", text)
    text = re.sub(multi_space_pattern, " ", text).strip()
    return text

# Apply cleaning to train and test sets
train["comment"] = train["comment_text"].apply(clean_text)
test["comment"] = test["comment_text"].apply(clean_text)

# Quick preview to ensure it worked
train[["comment_text", "comment"]].head(5)


import random

for label in label_cols:
    sample_row = train[train[label] == 1].sample(1, random_state=random.randint(1,100))
    print(f"ğŸ”¹ Category: {label}")
    print(sample_row['comment_text'].values[0])
    print("-" * 100)


test_df = test.drop(columns=['comment_text','id'])
test_df


import pandas as pd

# Load the dataset
train = train.drop(columns=['id','comment_text' ])

train


from sklearn.model_selection import train_test_split

# 1. Remove rows with missing or empty cleaned text
before = len(train)
train = train[train["comment"].notna() & (train["comment"].str.strip() != "")]
after = len(train)
print(f"Removed {before - after} rows with missing or empty cleaned text.")

# 2. Remove duplicate comments (based on comment)
before = len(train)
train = train.drop_duplicates(subset=["comment"])
after = len(train)
print(f"Removed {before - after} duplicate rows.")

# 3. Trainâ€“Validation split (80â€“20), stratified on "toxic" label
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train_df, val_df = train_test_split(
    train,
    test_size=0.2,
    random_state=42,
    stratify=train["toxic"] if "toxic" in train.columns else None
)

print("\nFinal sizes:")
print("Train:", train_df.shape)
print("Validation:", val_df.shape)

# Optional: Print label distribution in train and val sets
if all(col in train.columns for col in LABELS):
    print("\nTrain set label distribution:")
    print(train_df[LABELS].sum())
    print("\nValidation set label distribution:")
    print(val_df[LABELS].sum())


train_df



val_df



test_df


# ====================================================
# KAGGLE MULTI-GPU SETUP
# ====================================================
from accelerate import Accelerator
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW  # âœ… Import from torch.optim instead
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn

# ====================================================
# CONFIG
# ====================================================
MODEL_PATH = "microsoft/mdeberta-v3-base"
NUM_LABELS = 6
EPOCHS = 6
BATCH_SIZE = 16  # Per GPU, effective = 32
LR = 2e-5
MAX_LEN = 256
SEED = 42
PATIENCE = 2
MIN_DELTA = 0.0005

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

torch.manual_seed(SEED)
np.random.seed(SEED)

# ====================================================
# CHECK GPU AVAILABILITY
# ====================================================
print(f"ğŸ”§ GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ====================================================
# TOKENIZER
# ====================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ====================================================
# DATASET CLASS
# ====================================================
class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256, is_test=False):
        self.texts = df["comment"].tolist()
        self.is_test = is_test
        self.tokenizer = tokenizer
        self.max_len = max_len
        if not is_test:
            self.labels = df[LABELS].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}

        if not self.is_test:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


# ====================================================
# CREATE DATASETS
# ====================================================
train_dataset = ToxicDataset(train_df, tokenizer, max_len=MAX_LEN)
val_dataset = ToxicDataset(val_df, tokenizer, max_len=MAX_LEN)
test_dataset = ToxicDataset(test_df, tokenizer, max_len=MAX_LEN, is_test=True)

data_collator = DataCollatorWithPadding(tokenizer)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=data_collator,
    num_workers=2,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    collate_fn=data_collator,
    num_workers=2,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    collate_fn=data_collator,
    num_workers=2,
    pin_memory=True
)

# ====================================================
# MODEL
# ====================================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"
)

# ====================================================
# WEIGHTED LOSS
# ====================================================
label_sum = train_df[LABELS].sum(axis=0)
label_freq = label_sum / len(train_df)
class_weights = (1 / (label_freq + 1e-6))
class_weights = class_weights / class_weights.sum() * NUM_LABELS
class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float)

print("âœ… Class Weights:", class_weights_tensor)

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, logits, targets):
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, weight=self.weights.to(logits.device)
        )
        return loss

criterion = WeightedBCEWithLogitsLoss(class_weights_tensor)

# ====================================================
# OPTIMIZER & SCHEDULER
# ====================================================
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)  # âœ… Now from torch.optim
num_training_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps,
)

# ====================================================
# âœ… ACCELERATOR - AUTOMATICALLY USES BOTH GPUs
# ====================================================
accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=1
)

# Prepare everything
train_loader, val_loader, test_loader, model, optimizer, scheduler = accelerator.prepare(
    train_loader, val_loader, test_loader, model, optimizer, scheduler
)

print(f"âœ… Training on {accelerator.num_processes} GPU(s)")
print(f"âœ… Effective batch size: {BATCH_SIZE * accelerator.num_processes}")

# ====================================================
# TRAINING LOOP WITH EARLY STOPPING
# ====================================================
best_auc = 0.0
patience_counter = 0
best_model_path = "best_model"

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    if accelerator.is_main_process:
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    else:
        progress = train_loader

    for batch in progress:
        with accelerator.accumulate(model):
            outputs = model(**batch)
            logits = outputs.logits
            loss = criterion(logits, batch["labels"])
            
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        if accelerator.is_main_process:
            progress.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")
        print(f"Train Loss: {avg_train_loss:.4f}")

    # ====================================================
    # VALIDATION
    # ====================================================
    model.eval()
    preds, refs = [], []
    val_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(**batch)
            logits = outputs.logits
            
            loss = criterion(logits, batch["labels"])
            val_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            all_probs = accelerator.gather(probs)
            all_labels = accelerator.gather(batch["labels"])
            
            preds.append(all_probs.cpu().numpy())
            refs.append(all_labels.cpu().numpy())

    preds = np.concatenate(preds)
    refs = np.concatenate(refs)
    
    if accelerator.is_main_process:
        preds = preds[:len(val_dataset)]
        refs = refs[:len(val_dataset)]
        
        avg_val_loss = val_loss / len(val_loader)
        
        from sklearn.metrics import roc_auc_score
        
        valid_aucs = []
        print("\nPer-class AUC:")
        for i, label in enumerate(LABELS):
            if len(np.unique(refs[:, i])) > 1:
                auc_score = roc_auc_score(refs[:, i], preds[:, i])
                valid_aucs.append(auc_score)
                print(f"  {label:15s}: {auc_score:.4f}")
        
        macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0
        print(f"\n{'='*60}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val AUC (macro): {macro_auc:.4f}")
        print(f"{'='*60}\n")
        
        if macro_auc > best_auc + MIN_DELTA:
            best_auc = macro_auc
            patience_counter = 0
            
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                best_model_path,
                save_function=accelerator.save
            )
            tokenizer.save_pretrained(best_model_path)
            print(f"âœ… Best model saved! AUC: {best_auc:.4f}\n")
        else:
            patience_counter += 1
            print(f"âš ï¸�  No improvement. Patience: {patience_counter}/{PATIENCE}\n")
        
        if patience_counter >= PATIENCE:
            print(f"ğŸ›‘ Early stopping triggered after {epoch+1} epochs")
            break
    
    accelerator.wait_for_everyone()

if accelerator.is_main_process:
    print(f"\n{'='*60}")
    print(f"âœ… Training Complete!")
    print(f"{'='*60}")
    print(f"Best Validation AUC: {best_auc:.4f}")
    print(f"Model saved at: {best_model_path}")
    print(f"{'='*60}\n")


# ====================================================
# TEST INFERENCE
# ====================================================
model.eval()
test_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        outputs = model(**batch)
        logits = torch.sigmoid(outputs.logits)
        test_preds.append(accelerator.gather(logits).detach().cpu().numpy())

test_preds = np.concatenate(test_preds)
submission = pd.DataFrame(test_preds, columns=label_cols)
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv saved!")

