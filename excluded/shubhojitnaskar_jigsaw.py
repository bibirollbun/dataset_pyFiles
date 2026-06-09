# INSTALL OFFLINE PACKAGES
!pip install /kaggle/input/jigsaw-package/pip_packages/transformers-4.55.0-py3-none-any.whl -q


# # =========================================================================
# # 1. IMPORTS
# # =========================================================================
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torch.optim import AdamW
# from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import os

# # Suppress Hugging Face tokenizer parallelism warning
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # =========================================================================
# # 2. HYPERPARAMETER CONFIGURATION
# # =========================================================================
# class Config:
#     # --- File Paths ---
#     OFFLINE_MODEL_PATH = "/kaggle/input/jigsaw-package/bert_base_uncased_offline/"
#     TRAIN_CSV_PATH = "/kaggle/input/jigsaw-agile-community-rules/train.csv"
#     OUTPUT_DIR = "./"  # Directory to save models and plots

#     # --- Model & Training Parameters ---
#     MODEL_NAME = "bert-base-uncased"
#     N_SPLITS = 5          # Number of folds for cross-validation
#     EPOCHS = 3            # Number of epochs to train for each fold
#     MAX_LEN = 128         # Max sequence length for tokenizer
#     BATCH_SIZE = 16       # Batch size
#     LR = 2e-5             # Learning Rate
#     RANDOM_STATE = 42     # Random state for reproducibility

# # Create output directory if it doesn't exist
# if not os.path.exists(Config.OUTPUT_DIR):
#     os.makedirs(Config.OUTPUT_DIR)

# # =========================================================================
# # 3. DATASET CLASS
# # =========================================================================
# class CommentDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_len):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = str(self.texts[idx])
#         label = self.labels[idx]
        
#         encoding = self.tokenizer(
#             text,
#             truncation=True,
#             padding='max_length',
#             max_length=self.max_len,
#             return_tensors='pt'
#         )
        
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'labels': torch.tensor(label, dtype=torch.float)
#         }

# # =========================================================================
# # 4. EVALUATION FUNCTION (Corrected)
# # =========================================================================
# def evaluate_model(model, dataloader, device, loss_fn):
#     model.eval()
#     total_loss = 0
#     all_labels, all_preds = [], []
#     all_probs = []  # <-- FIX 1: Initialize list to store all probabilities

#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device).unsqueeze(1)

#             outputs = model(input_ids, attention_mask=attention_mask)
#             loss = loss_fn(outputs.logits, labels)
#             total_loss += loss.item()

#             # Get probabilities and predictions for the current batch
#             probs = torch.sigmoid(outputs.logits).cpu().numpy().flatten()
#             preds = (probs >= 0.5).astype(int)
            
#             all_probs.extend(probs) # <-- FIX 2: Accumulate probabilities from each batch
#             all_preds.extend(preds)
#             all_labels.extend(labels.cpu().numpy().flatten())

#     avg_loss = total_loss / len(dataloader)
#     accuracy = accuracy_score(all_labels, all_preds)
#     f1 = f1_score(all_labels, all_preds, zero_division=0)
    
#     # <-- FIX 3: Use the accumulated probabilities (all_probs) for AUC calculation
#     auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5

#     return avg_loss, accuracy, f1, auc


# # =========================================================================
# # 5. MAIN TRAINING SCRIPT
# # =========================================================================
# # --- Load Data ---
# train_df = pd.read_csv(Config.TRAIN_CSV_PATH)
# train_df['text'] = train_df['body'] + " [RULE] " + train_df['rule']

# # --- Load Tokenizer ---
# tokenizer = BertTokenizer.from_pretrained(Config.OFFLINE_MODEL_PATH)

# # --- Initialize K-Fold ---
# skf = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.RANDOM_STATE)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")


# # --- Start Cross-Validation Loop ---
# for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['rule_violation'])):
#     print(f"\n{'='*20} FOLD {fold+1} / {Config.N_SPLITS} {'='*20}")

#     # --- Prepare Data for current fold ---
#     train_fold_df = train_df.iloc[train_idx]
#     val_fold_df = train_df.iloc[val_idx]

#     train_dataset = CommentDataset(
#         texts=train_fold_df['text'].values,
#         labels=train_fold_df['rule_violation'].values,
#         tokenizer=tokenizer,
#         max_len=Config.MAX_LEN
#     )
#     val_dataset = CommentDataset(
#         texts=val_fold_df['text'].values,
#         labels=val_fold_df['rule_violation'].values,
#         tokenizer=tokenizer,
#         max_len=Config.MAX_LEN
#     )

#     # Use a separate loader for evaluating on training data without shuffling
#     train_eval_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
#     train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

#     # --- Initialize Model, Optimizer, Loss for current fold ---
#     model = BertForSequenceClassification.from_pretrained(Config.OFFLINE_MODEL_PATH, num_labels=1)
#     model.to(device)

#     optimizer = AdamW(model.parameters(), lr=Config.LR)
#     total_steps = len(train_loader) * Config.EPOCHS
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer, num_warmup_steps=0, num_training_steps=total_steps
#     )
#     loss_fn = torch.nn.BCEWithLogitsLoss()

#     # --- History Tracking for Plots ---
#     history = {'train_loss': [], 'val_loss': []}

#     # --- Epoch Loop for current fold ---
#     for epoch in range(Config.EPOCHS):
#         print(f"\n--- Epoch {epoch+1}/{Config.EPOCHS} ---")
        
#         # --- Training Phase ---
#         model.train()
#         epoch_train_loss = 0
#         for batch in tqdm(train_loader, desc="Training"):
#             optimizer.zero_grad()
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device).unsqueeze(1)

#             outputs = model(input_ids, attention_mask=attention_mask)
#             loss = loss_fn(outputs.logits, labels)
            
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#             epoch_train_loss += loss.item()
        
#         avg_epoch_train_loss = epoch_train_loss / len(train_loader)
#         history['train_loss'].append(avg_epoch_train_loss)

#         # --- Evaluation Phase ---
#         print("Evaluating...")
#         # Evaluate on Training Data
#         _, train_acc, train_f1, train_auc = evaluate_model(model, train_eval_loader, device, loss_fn)
#         # Evaluate on Validation Data
#         val_loss, val_acc, val_f1, val_auc = evaluate_model(model, val_loader, device, loss_fn)
#         history['val_loss'].append(val_loss)

#         # Print Epoch Results
#         print(f"Epoch {epoch+1} Summary:")
#         print(f"  Train Loss: {avg_epoch_train_loss:.4f} | Val Loss: {val_loss:.4f}")
#         print(f"  Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Train AUC: {train_auc:.4f}")
#         print(f"  Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f} | Val AUC:   {val_auc:.4f}")

#     # --- Save the model for the current fold ---
#     model_path = os.path.join(Config.OUTPUT_DIR, f"bert_fold_{fold}.pth")
#     torch.save(model.state_dict(), model_path)
#     print(f"\nModel for Fold {fold+1} saved to {model_path}")

#     # --- Plot Training & Validation Loss for the current fold ---
#     plt.figure(figsize=(10, 5))
#     plt.plot(history['train_loss'], label='Train Loss')
#     plt.plot(history['val_loss'], label='Validation Loss')
#     plt.title(f'Loss vs. Epochs for Fold {fold+1}')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# print("\nK-Fold Training Finished!")


# =========================================================================
# 1. IMPORTS
# =========================================================================
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import numpy as np
import os

print("Libraries imported.")

# =========================================================================
# 2. DEFINE PATHS AND PARAMETERS
# =========================================================================
class Config:
    # --- File Paths ---
    OFFLINE_MODEL_PATH = "/kaggle/input/jigsaw-package/bert_base_uncased_offline/"
    TEST_CSV_PATH = "/kaggle/input/jigsaw-agile-community-rules/test.csv"
    
    # --- IMPORTANT: Path to the directory where your k-fold models are saved ---
    MODEL_DIR = "/kaggle/input/jigsaw-model/" 

    # --- Model & Dataloader Parameters ---
    N_SPLITS = 5          # Should match the number of folds used during training
    MAX_LEN = 128         # Should match the max_len used during training
    BATCH_SIZE = 16       # Can be adjusted for inference speed/memory

print("Paths and parameters defined.")

# =========================================================================
# 3. DATASET CLASS (Unchanged)
# =========================================================================
class CommentDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in enc.items()}
        return item

print("Dataset class defined.")

# =========================================================================
# 4. LOAD TOKENIZER AND PREPARE TEST DATA (Unchanged)
# =========================================================================
tokenizer = BertTokenizer.from_pretrained(Config.OFFLINE_MODEL_PATH)
test_df = pd.read_csv(Config.TEST_CSV_PATH)

# Preprocess the test data
test_df['text'] = test_df['body'] + " [RULE] " + test_df['rule']
test_texts = test_df['text'].tolist()

test_dataset = CommentDataset(texts=test_texts, tokenizer=tokenizer, max_len=Config.MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

print("Tokenizer loaded and test data prepared.")

# =========================================================================
# 5. LOAD MODELS AND RUN ENSEMBLE INFERENCE
# =========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get paths to all saved model folds
model_paths = [os.path.join(Config.MODEL_DIR, f"bert_fold_{i}.pth") for i in range(Config.N_SPLITS)]
all_fold_probs = []

# --- Loop through each model fold for inference ---
for fold, path in enumerate(model_paths):
    print(f"\n--- Predicting with Fold {fold+1}/{Config.N_SPLITS} ---")
    
    # Load the base model structure
    model = BertForSequenceClassification.from_pretrained(Config.OFFLINE_MODEL_PATH, num_labels=1)
    
    # Load the fine-tuned weights for the current fold
    try:
        model.load_state_dict(torch.load(path, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {path}. Make sure the MODEL_DIR is correct.")
        print("Skipping this fold.")
        continue # Skip to the next fold if a model file is missing
        
    model.to(device)
    model.eval()

    # Store probabilities for the current fold
    current_fold_probs = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Predicting Fold {fold+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).cpu().numpy().flatten()
            current_fold_probs.extend(probs)
    
    all_fold_probs.append(current_fold_probs)

print("\nEnsemble inference finished.")

# =========================================================================
# 6. AVERAGE PREDICTIONS AND CREATE SUBMISSION FILE
# =========================================================================
if not all_fold_probs:
    print("Could not generate predictions as no model files were found. Exiting.")
else:
    # Convert the list of fold predictions into a numpy array
    predictions_array = np.array(all_fold_probs)

    # Average the predictions across all folds (axis=0)
    final_probs = np.mean(predictions_array, axis=0)

    # Create the submission DataFrame
    submission_df = pd.DataFrame({
        'row_id': test_df['row_id'],
        'rule_violation': final_probs
    })

    # Save the final submission file
    submission_df.to_csv('submission.csv', index=False)

    print("\nSubmission file 'submission.csv' created successfully!")
    print("This file contains the averaged predictions from all model folds.")
    print("Top 5 rows of the submission file:")
    print(submission_df.head())

