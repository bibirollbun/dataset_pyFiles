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


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import gc
import os

# --- 1. Configuration ---
class Config:
    MODEL_PATH = "/kaggle/input/m/gauripatil2299/deberta-v3-base/transformers/default/1/deberta-v3-base"
    TRAIN_FILE = "/kaggle/input/jigsaw-agile-community-rules/train.csv"
    TEST_FILE = "/kaggle/input/jigsaw-agile-community-rules/test.csv"
    SUBMISSION_FILE = "/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv"

    # Memory optimization settings
    BATCH_SIZE = 2                    # Reduced batch size 
    GRADIENT_ACCUMULATION_STEPS = 4   # Simulate batch_size of 8
    MAX_LENGTH = 256                  # Reduced from 512 to save memory
    USE_AMP = True                    # Use automatic mixed precision
    
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    
    # Text formatting
    PROMPT_TEMPLATE = """
    Comment: {body}
    [SEP]
    Subreddit: {subreddit}
    Rule: {rule}
    Positive Example (Violates Rule): {positive_example}
    Negative Example (Does Not Violate Rule): {negative_example}
    """

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(Config.SEED)

# --- 2. Data Loading and Preprocessing ---
print("Loading data...")
train_df = pd.read_csv(Config.TRAIN_FILE)
test_df = pd.read_csv(Config.TEST_FILE)
submission_df = pd.read_csv(Config.SUBMISSION_FILE)

# Combine positive and negative examples for a richer context. 
# We'll just use the first one for simplicity in this baseline.
train_df['positive_example'] = train_df['positive_example_1']
train_df['negative_example'] = train_df['negative_example_1']
test_df['positive_example'] = test_df['positive_example_1']
test_df['negative_example'] = test_df['negative_example_1']

def create_input_text(row):
    """Formats the input text according to the template."""
    return Config.PROMPT_TEMPLATE.format(
        body=str(row['body']),
        subreddit=str(row['subreddit']),
        rule=str(row['rule']),
        positive_example=str(row['positive_example']),
        negative_example=str(row['negative_example'])
    ).replace('nan', 'Not available.') # Handle potential NaNs in examples

print("Preprocessing text...")
train_df['input_text'] = train_df.apply(create_input_text, axis=1)
test_df['input_text'] = test_df.apply(create_input_text, axis=1)

# --- 3. Tokenizer and PyTorch Dataset ---
tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)

class CommentDataset(Dataset):
    def __init__(self, df, tokenizer, is_train=True):
        self.df = df
        self.texts = df['input_text'].values
        self.tokenizer = tokenizer
        self.is_train = is_train
        if self.is_train:
            self.labels = df['rule_violation'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.is_train:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

# --- 4. Training Loop ---
def train_loop(model, train_loader, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss = 0
    
    # Track steps for gradient accumulation
    steps = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Implement mixed precision training with torch.amp
        if Config.USE_AMP:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / Config.GRADIENT_ACCUMULATION_STEPS  # Normalize loss for gradient accumulation
            
            # Scale loss and backprop
            scaler.scale(loss).backward()
            
            # Step for gradient accumulation
            steps += 1
            if steps % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                # Unscale gradients, clip, optimize, and step scheduler
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / Config.GRADIENT_ACCUMULATION_STEPS  # Normalize loss for gradient accumulation
            
            # Regular backprop
            loss.backward()
            
            # Step for gradient accumulation
            steps += 1
            if steps % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * Config.GRADIENT_ACCUMULATION_STEPS  # De-normalize for reporting
        progress_bar.set_postfix(loss=loss.item() * Config.GRADIENT_ACCUMULATION_STEPS)
        
        # Free up memory
        del input_ids, attention_mask, labels, outputs, loss
        if steps % 8 == 0:  # Every 8 steps, clear CUDA cache
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

# --- 5. Inference Function ---
def inference(model, test_loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Predicting", leave=False)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Use mixed precision for inference as well
            if Config.USE_AMP:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            
            # For binary classification, we use the probability of class 1 (violation)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            predictions.extend(probs)
            
            # Free up memory
            del input_ids, attention_mask, outputs, logits
            torch.cuda.empty_cache()
            
    return predictions

# --- 6. Main Execution ---
print("Starting training and prediction process...")

# Prepare data loaders
train_dataset = CommentDataset(train_df, tokenizer, is_train=True)
test_dataset = CommentDataset(test_df, tokenizer, is_train=False)

# Reduce num_workers for less memory pressure
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE * 2, shuffle=False, num_workers=1)

# Initialize model
model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_PATH, num_labels=2)

# Memory optimization: Offload model to CPU if needed
if torch.cuda.is_available():
    # Move model to GPU
    model.to(Config.DEVICE)
else:
    print("CUDA not available, using CPU")

# Initialize mixed precision scaler
scaler = torch.cuda.amp.GradScaler() if Config.USE_AMP else None

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
# Calculate effective batch size (batch_size * gradient_accumulation_steps)
effective_batch_size = Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS
# Calculate total steps with gradient accumulation
num_training_steps = (len(train_loader) // Config.GRADIENT_ACCUMULATION_STEPS) * Config.EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Print memory usage before training
if torch.cuda.is_available():
    print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Training
for epoch in range(Config.EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{Config.EPOCHS} ---")
    avg_loss = train_loop(model, train_loader, optimizer, scheduler, Config.DEVICE, scaler)
    print(f"Average Training Loss: {avg_loss:.4f}")
    
    # Print memory usage after each epoch
    if torch.cuda.is_available():
        print(f"GPU memory after epoch {epoch+1}: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Force garbage collection between epochs
    gc.collect()
    torch.cuda.empty_cache()

# Inference
print("\n--- Generating Predictions ---")
final_predictions = inference(model, test_loader, Config.DEVICE)

# Create submission file
submission_df['rule_violation'] = final_predictions
submission_df.to_csv("submission.csv", index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print(submission_df.head())

# Clean up memory
del model, train_loader, test_loader, train_dataset, test_dataset
gc.collect()
torch.cuda.empty_cache()


submission_df.head()


submission_df.shape




