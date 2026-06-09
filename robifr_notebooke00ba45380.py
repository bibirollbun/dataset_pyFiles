# ============================================
# CONFIGURATION
# ============================================
MODEL_NAME = "distilbert-base-uncased"
EPOCHS = 2
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 32
LEARNING_RATE = 3e-5
DATA_PATH = "/kaggle/input/c/jigsaw-unintended-bias-in-toxicity-classification/all_data.csv"
SAVE_PATH = "/kaggle/working/toxic_model"
SEED = 42

# ============================================
# SUPPRESS WARNINGS
# ============================================
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Suppress TensorFlow and CUDA warnings
import sys
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

# ============================================
# IMPORTS
# ============================================
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "transformers", "datasets", "accelerate", "tqdm", "-q"], 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from tqdm import tqdm

# ============================================
# SETUP
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================
# LOAD DATA - FULL VERSION WITH ALL COLUMNS
# ============================================
print("Loading Jigsaw dataset with all columns...")

# Load all columns first to see what we have
df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {len(df.columns)}")
print("\nFirst few column names:")
for i, col in enumerate(df.columns[:20]):
    print(f"  {i+1}. {col}")
if len(df.columns) > 20:
    print(f"  ... and {len(df.columns)-20} more columns")

# The Jigsaw dataset has multiple toxicity annotations:
# - 'toxicity': Overall toxicity score (0-1)
# - 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat'
# - Many identity columns for bias analysis

print("\n" + "="*60)
print("DATA PREPARATION")
print("="*60)

# Create binary label from toxicity score (>= 0.5 means toxic)
df['label'] = (df['toxicity'] >= 0.5).astype(int)

# Let's see the distribution
print(f"\nLabel distribution:")
print(f"Non-toxic (label=0): {(df['label'] == 0).sum():,} samples")
print(f"Toxic (label=1): {(df['label'] == 1).sum():,} samples")
print(f"Toxicity rate: {df['label'].mean()*100:.2f}%")

# Check for missing values
print(f"\nMissing values in comment_text: {df['comment_text'].isnull().sum()}")
df = df.dropna(subset=['comment_text', 'toxicity'])

# ============================================
# ANALYZE BIAS IN THE DATA
# ============================================
print("\n" + "="*60)
print("BIAS ANALYSIS BY IDENTITY GROUP")
print("="*60)

# Identity columns in Jigsaw dataset
identity_columns = [
    'male', 'female', 'transgender', 'other_gender',
    'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation',
    'christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist', 'other_religion',
    'black', 'white', 'asian', 'latino', 'other_race_or_ethnicity',
    'physical_disability', 'intellectual_or_learning_disability',
    'psychiatric_or_mental_illness', 'other_disability'
]

# Count how many comments mention each identity
print("\nComments mentioning each identity (>0.5 probability):")
for col in identity_columns:
    if col in df.columns:
        count = (df[col] > 0.5).sum()
        if count > 0:
            toxic_rate = df[df[col] > 0.5]['label'].mean() * 100
            print(f"  {col}: {count:,} comments ({toxic_rate:.1f}% toxic)")

# ============================================
# SPLIT DATA
# ============================================
# Use the 'split' column if available, otherwise do random split
if 'split' in df.columns:
    print("\nUsing existing 'split' column for train/val split...")
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'test'].copy()  # or 'val' depending on dataset
    
    # If no test split, split train further
    if len(val_df) == 0:
        train_df, val_df = train_test_split(
            train_df[['comment_text', 'label']], 
            test_size=0.1, 
            random_state=SEED, 
            stratify=train_df['label']
        )
    else:
        train_df = train_df[['comment_text', 'label']]
        val_df = val_df[['comment_text', 'label']]
else:
    print("\nSplitting data randomly (80/20)...")
    train_df, val_df = train_test_split(
        df[['comment_text', 'label']], 
        test_size=0.2, 
        random_state=SEED, 
        stratify=df['label']
    )

print(f"\nTraining set: {len(train_df):,} samples")
print(f"Validation set: {len(val_df):,} samples")

# ============================================
# LOAD MODEL
# ============================================
print("\nLoading model...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

# ============================================
# DATASET - OPTIMIZED FOR JIGSAW
# ============================================
class JigsawToxicDataset(Dataset):
    def __init__(self, texts, labels):
        # Store texts as Python list to save memory
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize with all special tokens needed
        enc = tokenizer(
            text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ============================================
# CREATE DATALOADERS
# ============================================
print("Creating datasets...")
train_ds = JigsawToxicDataset(train_df["comment_text"], train_df["label"])
val_ds = JigsawToxicDataset(val_df["comment_text"], val_df["label"])

train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False)

print(f"\nTraining batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# ============================================
# TRAINING WITH BIAS AWARENESS
# ============================================
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

print("\n" + "="*60)
print("TRAINING STARTING")
print("="*60)

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{EPOCHS}", leave=True)
    
    for batch in train_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        train_loss += loss.item()
        train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validate
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    val_bar = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{EPOCHS}", leave=True)
    
    with torch.no_grad():
        for batch in val_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            
            correct += (preds == batch["labels"]).sum().item()
            total += len(preds)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            
            current_acc = correct / total if total > 0 else 0
            val_bar.set_postfix({'acc': f'{current_acc:.4f}'})
    
    val_acc = correct / total
    
    # Calculate F1 score for better evaluation
    from sklearn.metrics import f1_score
    val_f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Accuracy: {val_acc:.4f}")
    print(f"  Val F1 Score: {val_f1:.4f}")

# ============================================
# SAVE MODEL
# ============================================
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

os.makedirs(SAVE_PATH, exist_ok=True)
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print(f"Model saved to: {SAVE_PATH}")
print(f"Files saved:")
print(f"  - {SAVE_PATH}/config.json")
print(f"  - {SAVE_PATH}/pytorch_model.bin")
print(f"  - {SAVE_PATH}/tokenizer files")


--------------------------------------------------

































