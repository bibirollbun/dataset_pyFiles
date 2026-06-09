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


bhili = pd.read_csv("/kaggle/input/mm-lo-so-2025/bhili-train.csv")
gondi = pd.read_csv("/kaggle/input/mm-lo-so-2025/gondi-train.csv")
mundari = pd.read_csv("/kaggle/input/mm-lo-so-2025/mundari-train.csv")
santali = pd.read_csv("/kaggle/input/mm-lo-so-2025/santali-train.csv")
test = pd.read_csv("/kaggle/input/mm-lo-so-2025/test.csv")


# Step 1a: Clean and prepare Bhili
bhili_clean = bhili.dropna(subset=["Hindi", "Bhili"]).copy()
bhili_clean.rename(columns={"Unnamed: 0": "id"}, inplace=True)

# Step 1b: Clean and prepare Gondi
gondi_clean = gondi.dropna(subset=["Hindi", "Gondi"]).copy()
gondi_clean.rename(columns={"Unnamed: 0": "id"}, inplace=True)

# Step 1c: Clean and prepare Mundari
mundari_clean = mundari.dropna(subset=["Hindi", "Mundari"]).copy()
mundari_clean.rename(columns={"Unnamed: 0": "id"}, inplace=True)

# Step 1d: Clean and prepare Santali
santali_clean = santali.dropna(subset=["English", "Santali"]).copy()
santali_clean.rename(columns={"Unnamed: 0": "id"}, inplace=True)



# Step 2: Remove duplicates in each dataset

bhili_clean = bhili_clean.drop_duplicates(subset=["Hindi", "Bhili"])
gondi_clean = gondi_clean.drop_duplicates(subset=["Hindi", "Gondi"])
mundari_clean = mundari_clean.drop_duplicates(subset=["Hindi", "Mundari"])
santali_clean = santali_clean.drop_duplicates(subset=["English", "Santali"])

# Print resulting shapes
print("Bhili:", bhili_clean.shape)
print("Gondi:", gondi_clean.shape)
print("Mundari:", mundari_clean.shape)
print("Santali:", santali_clean.shape)



# Step 3: Build unified dataset

train_rows = []

# ----- Bhili (Hindi <-> Bhili) -----
for _, r in bhili_clean.iterrows():
    # forward: Hindi → Bhili
    train_rows.append({
        "source_lang": "Hindi",
        "source_text": r["Hindi"],
        "target_lang": "Bhilli",
        "target_text": r["Bhili"],
    })
    # reverse: Bhili → Hindi
    train_rows.append({
        "source_lang": "Bhilli",
        "source_text": r["Bhili"],
        "target_lang": "Hindi",
        "target_text": r["Hindi"],
    })

# ----- Gondi (Hindi <-> Gondi) -----
for _, r in gondi_clean.iterrows():
    train_rows.append({
        "source_lang": "Hindi",
        "source_text": r["Hindi"],
        "target_lang": "Gondi",
        "target_text": r["Gondi"],
    })
    train_rows.append({
        "source_lang": "Gondi",
        "source_text": r["Gondi"],
        "target_lang": "Hindi",
        "target_text": r["Hindi"],
    })

# ----- Mundari (Hindi <-> Mundari) -----
for _, r in mundari_clean.iterrows():
    train_rows.append({
        "source_lang": "Hindi",
        "source_text": r["Hindi"],
        "target_lang": "Mundari",
        "target_text": r["Mundari"],
    })
    train_rows.append({
        "source_lang": "Mundari",
        "source_text": r["Mundari"],
        "target_lang": "Hindi",
        "target_text": r["Hindi"],
    })

# ----- Santali (English <-> Santali) -----
for _, r in santali_clean.iterrows():
    train_rows.append({
        "source_lang": "English",
        "source_text": r["English"],
        "target_lang": "Santali",
        "target_text": r["Santali"],
    })
    train_rows.append({
        "source_lang": "Santali",
        "source_text": r["Santali"],
        "target_lang": "English",
        "target_text": r["English"],
    })

import pandas as pd
train_df = pd.DataFrame(train_rows)
print(train_df.shape)
train_df.head()



# Step 4 — Add language tags for mT5 style training

def make_tag(lang):
    return f"<to_{lang.lower()}>"

train_df["tag"] = train_df["target_lang"].apply(make_tag)

# mT5 input format:  "<to_language>  actual sentence"
train_df["input_text"] = train_df["tag"] + " " + train_df["source_text"]
train_df["label_text"] = train_df["target_text"]

# Inspect
train_df.head()
print("Total training rows:", len(train_df))



from sklearn.model_selection import train_test_split

# Stratified split by target language
train_df, val_df = train_test_split(
    train_df,
    test_size=0.05,
    random_state=42,
    stratify=train_df["target_lang"]
)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print(val_df["target_lang"].value_counts())



# Step 6 — Load mT5-small tokenizer and add language tags
from transformers import MT5Tokenizer

tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

special_tokens = [
    "<to_hindi>",
    "<to_bhilli>",
    "<to_mundari>",
    "<to_gondi>",
    "<to_english>",
    "<to_santali>"
]

tokenizer.add_tokens(special_tokens)

print("Tokenizer size:", len(tokenizer))



# Step 7 — Build PyTorch dataset for mT5
import torch
from torch.utils.data import Dataset, DataLoader

class MT5Dataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        source = row["input_text"]
        target = row["label_text"]

        # Encode
        source_enc = self.tokenizer(
            source,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_enc = self.tokenizer(
            target,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_enc.input_ids.squeeze()
        labels[labels == tokenizer.pad_token_id] = -100   # ignore pad tokens

        return {
            "input_ids": source_enc.input_ids.squeeze(),
            "attention_mask": source_enc.attention_mask.squeeze(),
            "labels": labels,
        }

# Create datasets
train_dataset = MT5Dataset(train_df, tokenizer, max_len=128)
val_dataset   = MT5Dataset(val_df, tokenizer, max_len=128)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False)

print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))



# Step 8 — Load model
from transformers import MT5ForConditionalGeneration
import torch

# Define device FIRST
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Then load and move model
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
model.resize_token_embeddings(len(tokenizer))
model.to(device)


!pip install bitsandbytes
print("lets go")


# =========================
# MEMORY-OPTIMIZED Training Code for mT5
# =========================
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn as nn
from tqdm import tqdm
import torch
import gc

# =========================
# Hyperparameters - REDUCED for memory
# =========================
epochs = 1
lr = 1e-4
warmup_ratio = 0.1
accum_steps = 32  # DOUBLED from 16 to reduce memory per step
max_grad_norm = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# CRITICAL MEMORY OPTIMIZATIONS
# =========================
# 1. Enable gradient checkpointing
model.gradient_checkpointing_enable()
print("✓ Gradient checkpointing enabled")

# 2. Clear CUDA cache
torch.cuda.empty_cache()
gc.collect()

model.train()
model.to(device)

# =========================
# Check and resize embeddings if needed
# =========================
if len(tokenizer) != model.config.vocab_size:
    print(f"Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

# =========================
# Optimizer & Scheduler - Use 8-bit optimizer
# =========================
try:
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(), 
        lr=lr, 
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    print("✓ Using 8-bit AdamW optimizer (saves ~50% memory)")
except ImportError:
    print("⚠ bitsandbytes not available, using regular AdamW")
    optimizer = AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    )

total_steps = len(train_loader) * epochs // accum_steps
warmup_steps = int(total_steps * warmup_ratio)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# =========================
# Training loop with memory management
# =========================
def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    valid_batches = 0
    loop = tqdm(train_loader, leave=True)
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(loop):
        try:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Mask padding tokens in labels
            labels = labels.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            
            # Check if we have valid labels
            num_valid_labels = (labels != -100).sum()
            if num_valid_labels == 0:
                continue
            
            # Create decoder_input_ids
            decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
            )
            loss = outputs.loss / accum_steps
            
            # Sanity check
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n❌ Invalid loss at batch {batch_idx}: {loss.item()}")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
            
            # Backward pass
            loss.backward()
            
            # Check for NaN in gradients
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN gradient in {name}")
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"Skipping batch {batch_idx} due to NaN gradients")
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
            
            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # CRITICAL: Clear cache after optimizer step
                torch.cuda.empty_cache()
            
            total_loss += loss.item() * accum_steps
            valid_batches += 1
            
            # Memory monitoring
            if batch_idx % 100 == 0:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                loop.set_description(f"Epoch {epoch} | Mem: {allocated:.1f}/{reserved:.1f}GB")
            else:
                loop.set_description(f"Epoch {epoch}")
            
            loop.set_postfix(loss=loss.item() * accum_steps, valid=valid_batches)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n❌ OOM at batch {batch_idx}")
                print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
                print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
                
                # Emergency cleanup
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                print(f"\n❌ Runtime error at batch {batch_idx}: {e}")
                optimizer.zero_grad()
                continue
    
    if valid_batches == 0:
        print("❌ NO VALID BATCHES PROCESSED!")
        return float('nan')
    
    return total_loss / valid_batches

# =========================
# Validation loop with memory management
# =========================
def validate():
    model.eval()
    total_loss = 0
    valid_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                # Mask padding
                labels = labels.clone()
                labels[labels == tokenizer.pad_token_id] = -100
                
                if (labels != -100).sum() == 0:
                    continue
                
                # Create decoder_input_ids
                decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels
                )
                
                if not torch.isnan(outputs.loss):
                    total_loss += outputs.loss.item()
                    valid_batches += 1
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                raise
    
    return total_loss / max(valid_batches, 1)

# =========================
# Train
# =========================
print("\nMemory-Optimized Training Configuration:")
print(f"Total batches: {len(train_loader)}")
print(f"Accumulation steps: {accum_steps}")
print(f"Effective batch size: {train_loader.batch_size * accum_steps}")
print(f"Total optimization steps: {total_steps}")
print(f"Initial memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

train_loss = train_one_epoch(1)
print(f"\nTrain loss: {train_loss}")

if not torch.isnan(torch.tensor(train_loss)):
    val_loss = validate()
    print(f"Val loss: {val_loss}")


print("we can start the next steps")


# small cell
save_dir = "/kaggle/working/mt5_checkpoint_epoch1"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print("Saved to", save_dir)



print("Starting epoch 2...")

train_loss = train_one_epoch(2)
print("Epoch 2 train loss:", train_loss)

val_loss = validate()
print("Epoch 2 val loss:", val_loss)

# Save checkpoint if val improved
save_dir2 = "/kaggle/working/mt5_checkpoint_epoch2"
model.save_pretrained(save_dir2)
tokenizer.save_pretrained(save_dir2)
print("Saved epoch 2 checkpoint to:", save_dir2)



print("ready for model testing ?")


# ========================================================
# FULL INFERENCE + SUBMISSION PIPELINE FOR mT5
# ========================================================
import torch
import pandas as pd
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================================================
# STEP 1 — LOAD CHECKPOINT (CHANGE PATH!)
# ========================================================
model_path = "/kaggle/working/mt5_checkpoint_epoch2" 

tokenizer = MT5Tokenizer.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path)
model.to(device)
model.eval()

print("✅ Model loaded:", model_path)

# ========================================================
# STEP 2 — LOAD TEST SET
# ========================================================
test_path = "/kaggle/input/mm-lo-so-2025/test.csv"
test = pd.read_csv(test_path)

# Fix column name
if "Target Lang " in test.columns:
    test.rename(columns={"Target Lang ": "Target Lang"}, inplace=True)

print("✅ Test rows:", len(test))

# ========================================================
# STEP 3 — BUILD INPUT PROMPTS
# ========================================================
lang_to_tag = {
    "Hindi": "<to_hindi>",
    "Bhilli": "<to_bhilli>",
    "Mundari": "<to_mundari>",
    "Gondi": "<to_gondi>",
    "English": "<to_english>",
    "Santali": "<to_santali>"
}

def build_input(source_lang, target_lang, text):
    prefix = lang_to_tag[target_lang]
    return f"{prefix} {source_lang}: {text}"

inputs = [
    build_input(row["Source Lang"], row["Target Lang"], row["Source Sentence"])
    for _, row in test.iterrows()
]

print("✅ Example input:", inputs[0])

# ========================================================
# STEP 4 — GENERATION FUNCTION
# ========================================================
def generate_batch(texts, batch_size=32, max_len=64):
    outputs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating"):
        batch = texts[i:i+batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_length=max_len,
                num_beams=4,
                early_stopping=True
            )

        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        outputs.extend(decoded)

    return outputs

# ========================================================
# STEP 5 — RUN INFERENCE
# ========================================================
preds = generate_batch(inputs, batch_size=32, max_len=64)
print("✅ Predictions generated:", len(preds))

# ========================================================
# STEP 6 — BUILD SUBMISSION FILE
# ========================================================
submission = test.copy()
submission["Target Sentence"] = preds
submission["Target Sentence"] = submission["Target Sentence"].replace("", ".")

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("✅ Submission saved to:", submission_path)



print("ok")

