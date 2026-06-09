import os 
import pandas as pd
import numpy as np
import re
import string
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
import time
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.cuda.amp import autocast, GradScaler  # For mixed precision


# ==============================================================================
# PHASE 1: Data Loading & Initial Exploration
# ==============================================================================
df_train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
df_test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
df_sample_submission = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')

print(f"Training data shape: {df_train.shape}")
print(f"Test data shape: {df_test.shape}")


# ==============================================================================
# PHASE 2.1: Text Cleaning and Normalization
# ==============================================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '_URL_', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

text_columns = ['body', 'rule', 'positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2']
for col in text_columns:
    df_train[col] = df_train[col].apply(clean_text)
    df_test[col] = df_test[col].apply(clean_text)

print("\n--- Sample of Cleaned Training Data ---")
print(df_train[text_columns].head())


# ==============================================================================
# PHASE 2.2: Tokenization and Sequence Preparation (Offline Version)
# ==============================================================================
# Check available datasets and find BERT model
import os
print("Available input directories:")
for item in os.listdir('/kaggle/input/'):
    print(f"  - {item}")

# Try common BERT dataset names in Kaggle
possible_bert_paths = [
    '/kaggle/input/bert-base-uncased',
    '/kaggle/input/bert-base-uncased-pytorch',
    '/kaggle/input/transformers-bert-base-uncased',
    '/kaggle/input/bert-base-uncased-huggingface',
    '/kaggle/input/huggingface-bert-base-uncased'
]

BERT_DIR = None
for path in possible_bert_paths:
    if os.path.exists(path):
        BERT_DIR = path
        print(f"Found BERT model at: {BERT_DIR}")
        # Check contents
        print("Contents:")
        for item in os.listdir(BERT_DIR):
            print(f"  - {item}")
        break

if BERT_DIR is None:
    print("BERT model not found in expected locations!")
    print("Please ensure you have added the BERT model as a dataset to your Kaggle notebook.")
    print("You can use the 'huggingface/transformers' dataset or download BERT manually.")
    # Fallback to online loading (will work if internet is available)
    print("Attempting to use online model (may fail if internet is disabled)...")
    BERT_DIR = 'bert-base-uncased'

# Load tokenizer with better error handling
try:
    print(f"Loading tokenizer from: {BERT_DIR}")
    if BERT_DIR.startswith('/kaggle/input/'):
        # For local files, try different approaches
        tokenizer = BertTokenizerFast.from_pretrained(BERT_DIR, local_files_only=True)
    else:
        # For online model
        tokenizer = BertTokenizerFast.from_pretrained(BERT_DIR)
    print("Tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Trying alternative approach...")
    try:
        # Try loading from the transformers dataset if available
        transformers_path = '/kaggle/input/transformers-bert-base-uncased'
        if os.path.exists(transformers_path):
            tokenizer = BertTokenizerFast.from_pretrained(transformers_path, local_files_only=True)
            BERT_DIR = transformers_path
            print(f"Successfully loaded from: {BERT_DIR}")
        else:
            raise Exception("No valid BERT model found")
    except Exception as e2:
        print(f"Failed to load tokenizer: {e2}")
        print("Please add a BERT model dataset to your Kaggle notebook.")
        raise
MAX_LEN = 512

def encode_full_context(body, rule, pos_ex1, pos_ex2, neg_ex1, neg_ex2):
    full_text = f"{body} [SEP] {rule} [SEP] pos ex: {pos_ex1} [SEP] pos ex: {pos_ex2} [SEP] neg ex: {neg_ex1} [SEP] neg ex: {neg_ex2}"
    encoded = tokenizer.encode_plus(
        full_text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors='np'
    )
    return encoded['input_ids'][0], encoded['attention_mask'][0], encoded['token_type_ids'][0]

# Prepare arrays
train_input_ids, train_attention_masks, train_token_type_ids = [], [], []
test_input_ids, test_attention_masks, test_token_type_ids = [], [], []

print("\nEncoding training data...")
for i, row in tqdm(df_train.iterrows(), total=len(df_train)):
    input_ids, attention_mask, token_type_ids = encode_full_context(
        row['body'], row['rule'], row['positive_example_1'],
        row['positive_example_2'], row['negative_example_1'],
        row['negative_example_2']
    )
    train_input_ids.append(input_ids)
    train_attention_masks.append(attention_mask)
    train_token_type_ids.append(token_type_ids)

print("\nEncoding test data...")
for i, row in tqdm(df_test.iterrows(), total=len(df_test)):
    input_ids, attention_mask, token_type_ids = encode_full_context(
        row['body'], row['rule'], row['positive_example_1'],
        row['positive_example_2'], row['negative_example_1'],
        row['negative_example_2']
    )
    test_input_ids.append(input_ids)
    test_attention_masks.append(attention_mask)
    test_token_type_ids.append(token_type_ids)

# Convert to numpy arrays
train_input_ids = np.array(train_input_ids)
train_attention_masks = np.array(train_attention_masks)
train_token_type_ids = np.array(train_token_type_ids)
train_labels = df_train['rule_violation'].values

test_input_ids = np.array(test_input_ids)
test_attention_masks = np.array(test_attention_masks)
test_token_type_ids = np.array(test_token_type_ids)

print(f"\nTraining Input IDs shape: {train_input_ids.shape}")
print(f"Test Input IDs shape: {test_input_ids.shape}")


# ==============================================================================
# PHASE 3: Model Building, 5-Fold Training with Gradient Accumulation, FP16 & Auto-Resume
# ==============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Optimized parameters for GPU P100
MAX_EPOCHS = 15  # Reduced for Kaggle time limits
BATCH_SIZE = 8   # Optimized for P100 memory
GRAD_ACCUM_STEPS = 2
patience = 3
MAX_TRAIN_TIME = 4 * 60 * 60  # 4 hours limit for Kaggle

# Prepare test dataloader
test_input_ids_t = torch.tensor(test_input_ids, dtype=torch.long)
test_attention_masks_t = torch.tensor(test_attention_masks, dtype=torch.long)
test_token_type_ids_t = torch.tensor(test_token_type_ids, dtype=torch.long)
test_dataset = TensorDataset(test_input_ids_t, test_attention_masks_t, test_token_type_ids_t)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=16)

fold_test_preds = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Function to load checkpoint if exists (Kaggle working directory)
def load_checkpoint(checkpoint_path, model, optimizer):
    start_epoch = 1
    best_val_loss = float('inf')
    patience_counter = 0
    epoch_times = []
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path} ...")
        try:
            # Fix for PyTorch 2.6+ security issue
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            patience_counter = checkpoint['patience_counter']
            epoch_times = checkpoint.get('epoch_times', [])
            print(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting fresh training...")
    
    return start_epoch, best_val_loss, patience_counter, epoch_times

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

for fold, (train_idx, val_idx) in enumerate(skf.split(train_input_ids, train_labels)):
    print(f"\n========== Fold {fold+1} ==========")

    # Prepare fold data
    fold_train_inputs = torch.tensor(train_input_ids[train_idx], dtype=torch.long)
    fold_train_masks = torch.tensor(train_attention_masks[train_idx], dtype=torch.long)
    fold_train_token_types = torch.tensor(train_token_type_ids[train_idx], dtype=torch.long)
    fold_train_labels = torch.tensor(train_labels[train_idx], dtype=torch.long)

    fold_val_inputs = torch.tensor(train_input_ids[val_idx], dtype=torch.long)
    fold_val_masks = torch.tensor(train_attention_masks[val_idx], dtype=torch.long)
    fold_val_token_types = torch.tensor(train_token_type_ids[val_idx], dtype=torch.long)
    fold_val_labels = torch.tensor(train_labels[val_idx], dtype=torch.long)

    # Create data loaders
    train_data = TensorDataset(fold_train_inputs, fold_train_masks, fold_train_token_types, fold_train_labels)
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE)

    val_data = TensorDataset(fold_val_inputs, fold_val_masks, fold_val_token_types, fold_val_labels)
    val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=BATCH_SIZE)

    # Load model from offline directory
    try:
        if BERT_DIR.startswith('/kaggle/input/'):
            model = BertForSequenceClassification.from_pretrained(
                BERT_DIR,
                local_files_only=True,
                num_labels=2,
                output_attentions=False,
                output_hidden_states=False
            )
        else:
            model = BertForSequenceClassification.from_pretrained(
                BERT_DIR,
                num_labels=2,
                output_attentions=False,
                output_hidden_states=False
            )
        print(f"Model loaded successfully from: {BERT_DIR}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative model loading...")
        # Try without local_files_only if the path seems problematic
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )
    model.to(device)
    
    # Optimizer and scaler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8, weight_decay=0.01)
    scaler = GradScaler()

    # Checkpoint path in Kaggle working directory
    checkpoint_path = f'/kaggle/working/best_bert_model_fold{fold+1}.pth'
    start_epoch, best_val_loss, patience_counter, epoch_times = load_checkpoint(checkpoint_path, model, optimizer)

    epoch_i = start_epoch - 1
    total_elapsed = sum(epoch_times)
    training_start_time = time.time()

    # Training loop
    while True:
        epoch_i += 1
        print(f'\n-------- Epoch {epoch_i} --------')
        model.train()
        start_time_epoch = time.time()
        total_train_loss = 0

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Fold {fold+1} Training")):
            b_input_ids = batch[0].to(device, non_blocking=True)
            b_attention_mask = batch[1].to(device, non_blocking=True)
            b_token_type_ids = batch[2].to(device, non_blocking=True)
            b_labels = batch[3].to(device, non_blocking=True)

            with autocast():
                outputs = model(b_input_ids, attention_mask=b_attention_mask, 
                              token_type_ids=b_token_type_ids, labels=b_labels)
                loss = outputs.loss / GRAD_ACCUM_STEPS
            
            scaler.scale(loss).backward()
            total_train_loss += loss.item() * GRAD_ACCUM_STEPS

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_dataloader)
        epoch_duration = time.time() - start_time_epoch
        epoch_times.append(epoch_duration)
        total_elapsed = sum(epoch_times)
        
        print(f"Avg train loss: {avg_train_loss:.4f}, epoch time: {epoch_duration/60:.2f} mins")

        # ========= Validation =========
        model.eval()
        all_preds = []
        all_labels = []
        total_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Fold {fold+1} Validation"):
                b_input_ids = batch[0].to(device, non_blocking=True)
                b_attention_mask = batch[1].to(device, non_blocking=True)
                b_token_type_ids = batch[2].to(device, non_blocking=True)
                b_labels = batch[3].to(device, non_blocking=True)

                with autocast():
                    outputs = model(b_input_ids, attention_mask=b_attention_mask, 
                                  token_type_ids=b_token_type_ids, labels=b_labels)
                    loss = outputs.loss
                    total_val_loss += loss.item()
                    probs = torch.softmax(outputs.logits, dim=1)[:,1]
                    all_preds.extend(probs.cpu().numpy())
                    all_labels.extend(b_labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_roc_auc = roc_auc_score(all_labels, all_preds)
        print(f"Validation loss: {avg_val_loss:.4f}, ROC-AUC: {val_roc_auc:.4f}")

        # Save checkpoint if improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            try:
                torch.save({
                    'epoch': epoch_i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                    'epoch_times': epoch_times,
                    'val_roc_auc': val_roc_auc
                }, checkpoint_path)
                print(f"Saved checkpoint with validation loss: {best_val_loss:.4f}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")

        # Check stopping conditions
        current_total_time = time.time() - training_start_time
        if current_total_time >= MAX_TRAIN_TIME:
            print(f"Reached training time limit ({MAX_TRAIN_TIME/3600:.1f} hours).")
            break
        if patience_counter >= patience:
            print(f"Early stopping after {patience} epochs without improvement.")
            break
        if epoch_i >= MAX_EPOCHS:
            print(f"Reached maximum epochs ({MAX_EPOCHS}).")
            break

    # Load best model for inference
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model with validation loss: {checkpoint['best_val_loss']:.4f}")
        except Exception as e:
            print(f"Error loading best model: {e}")
            print("Using current model state...")

    # Generate predictions for test set
    model.eval()
    fold_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc=f"Fold {fold+1} Test Prediction"):
            b_input_ids = batch[0].to(device, non_blocking=True)
            b_attention_mask = batch[1].to(device, non_blocking=True)
            b_token_type_ids = batch[2].to(device, non_blocking=True)
            
            with autocast():
                outputs = model(b_input_ids, attention_mask=b_attention_mask, 
                              token_type_ids=b_token_type_ids)
                probs = torch.softmax(outputs.logits, dim=1)[:,1]
                fold_probs.append(probs.cpu().numpy())
    
    fold_probs = np.concatenate(fold_probs, axis=0)
    fold_test_preds.append(fold_probs)
    
    # Clear memory
    del model, optimizer, scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Fold {fold+1} completed. Predictions shape: {fold_probs.shape}")

# Average predictions across folds
print("\nAveraging predictions across all folds...")
final_test_preds = np.mean(fold_test_preds, axis=0)
print(f"Final predictions shape: {final_test_preds.shape}")
print(f"Prediction range: [{final_test_preds.min():.4f}, {final_test_preds.max():.4f}]")



# ==============================================================================
# PHASE 4: Submission
# ==============================================================================
submission = pd.DataFrame({
    'row_id': df_test['row_id'].values,
    'rule_violation': final_test_preds
})

submission_path = '/kaggle/working/submission.csv'
submission.to_csv(submission_path, index=False)

print(f"\nSaved submission to {submission_path}")
print("Submission preview:")
print(submission.head())
print(f"\nSubmission shape: {submission.shape}")
print("Training completed successfully!")

