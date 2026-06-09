import shutil

# Rename (or copy) submission_input.csv → submission.csv
shutil.copy("/kaggle/input/submission-input-csv/submission.csv", "submission.csv")



# ============================================
# CELL 1: Install Required Libraries
# ============================================
# Uncomment nếu chưa cài đặt
# !pip install transformers datasets accelerate scikit-learn torch

# ============================================
# CELL 2: Import Libraries
# ============================================
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import gc
import os

print("All libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================
# CELL 3: Configuration
# ============================================
MODEL_NAME = 'microsoft/deberta-v3-base'

N_SPLITS = 5      # Number of folds
MAX_LENGTH = 160  # Max length of prompt
BATCH_SIZE = 8    # Giảm xuống 4 hoặc 2 nếu máy yếu
EPOCHS = 3        # Number of epochs
LR = 2e-5         # Learning rate

# ĐỔI ĐƯỜNG DẪN FILE Ở ĐÂY
TRAIN_PATH = "./train.csv"  # Đường dẫn đến file train.csv
TEST_PATH = "./test.csv"    # Đường dẫn đến file test.csv

print(f"Configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  K-Folds: {N_SPLITS}")
print(f"  Max Length: {MAX_LENGTH}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning Rate: {LR}")

# ============================================
# CELL 4: Load Data
# ============================================
try:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print("✓ Data loaded successfully!")
    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape: {test_df.shape}")
    print(f"  Train columns: {train_df.columns.tolist()}")
    print(f"  Test columns: {test_df.columns.tolist()}")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    print(f"Make sure files exist at:")
    print(f"  {os.path.abspath(TRAIN_PATH)}")
    print(f"  {os.path.abspath(TEST_PATH)}")
    raise

# ============================================
# CELL 5: Prepare Text Column
# ============================================
# Check if 'text' column exists, if not, look for 'prompt' or similar
if 'text' not in train_df.columns:
    if 'prompt' in train_df.columns:
        train_df['text'] = train_df['prompt']
        test_df['text'] = test_df['prompt']
        print("✓ Using 'prompt' column as 'text'")
    else:
        print(f"✗ Available columns: {train_df.columns.tolist()}")
        raise ValueError("Cannot find text/prompt column!")
else:
    print("✓ 'text' column found")

# Convert string labels to integers
label_mapping = {'benign': 0, 'jailbreak': 1}
train_df['label'] = train_df['label'].map(label_mapping)

# Check label distribution
print(f"\nLabel distribution:")
print(train_df['label'].value_counts())
print(f"Label mapping: {label_mapping}")

# ============================================
# CELL 6: Load Tokenizer
# ============================================
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("✓ Tokenizer loaded successfully!")

# ============================================
# CELL 7: Define Helper Functions
# ============================================
def tokenize_function(examples):
    """Tokenize text data"""
    return tokenizer(
        examples['text'], 
        padding='max_length', 
        truncation=True, 
        max_length=MAX_LENGTH
    )

def compute_metrics(eval_pred):
    """Compute ROC-AUC metric"""
    logits, labels = eval_pred
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    auc = roc_auc_score(labels, probs[:, 1])
    return {"roc_auc": auc}

print("✓ Helper functions defined!")

# ============================================
# CELL 8: Prepare Test Data
# ============================================
print("Tokenizing test data...")
test_dataset = Dataset.from_pandas(test_df[['text']])
tokenized_test_ds = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
print(f"✓ Test data tokenized! Shape: {len(tokenized_test_ds)}")

# ============================================
# CELL 9: K-Fold Cross-Validation Training
# ============================================
print(f"\n{'='*60}")
print(f"STARTING K-FOLD TRAINING WITH {N_SPLITS} FOLDS")
print(f"{'='*60}\n")

# Storage for predictions and scores
test_predictions = [] 
oof_scores = []

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Split based on label
for fold, (train_index, val_index) in enumerate(skf.split(train_df, train_df['label'])):
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold+1}/{N_SPLITS}")
    print(f"{'='*60}")
    
    # Split data for this fold
    train_fold_df = train_df.iloc[train_index].reset_index(drop=True)
    val_fold_df = train_df.iloc[val_index].reset_index(drop=True)
    
    print(f"Train size: {len(train_fold_df)}, Val size: {len(val_fold_df)}")
    
    # Convert to Dataset - only keep necessary columns
    train_dataset = Dataset.from_pandas(train_fold_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_fold_df[['text', 'label']])
    
    # Tokenize
    print(f"Tokenizing fold {fold+1}...")
    tokenized_train_ds = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    tokenized_val_ds = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    # Rename 'label' to 'labels' for Hugging Face Trainer
    tokenized_train_ds = tokenized_train_ds.rename_column('label', 'labels')
    tokenized_val_ds = tokenized_val_ds.rename_column('label', 'labels')
    
    # Set format
    tokenized_train_ds.set_format('torch')
    tokenized_val_ds.set_format('torch')
    
    # Load NEW model for each fold
    print(f"Loading model for fold {fold+1}...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=f"./results_fold_{fold+1}",
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),  # Tự động tắt nếu không có GPU
        report_to="none",
        save_total_limit=1,
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    # Train
    print(f"Training fold {fold+1}...")
    trainer.train()
    
    # Get best validation score
    eval_results = trainer.evaluate()
    best_score = eval_results['eval_roc_auc']
    print(f"✓ Fold {fold+1}: Best Val ROC-AUC = {best_score:.4f}")
    oof_scores.append(best_score)
    
    # Predict on test set
    print(f"Predicting on test set for fold {fold+1}...")
    predictions = trainer.predict(tokenized_test_ds)
    test_predictions.append(predictions.predictions)
    
    # Clean up memory
    del model, trainer, tokenized_train_ds, tokenized_val_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n{'='*60}")
print("✓ ALL FOLDS COMPLETED!")
print(f"{'='*60}")

# ============================================
# CELL 10: Evaluation Results
# ============================================
print(f"\n{'='*60}")
print("CROSS-VALIDATION RESULTS")
print(f"{'='*60}")
print(f"\nMean OOF ROC-AUC: {np.mean(oof_scores):.4f} ± {np.std(oof_scores):.4f}")
print(f"\nIndividual Fold Scores:")
for i, score in enumerate(oof_scores, 1):
    print(f"  Fold {i}: {score:.4f}")

# ============================================
# CELL 11: Create Submission File
# ============================================
print(f"\n{'='*60}")
print("CREATING SUBMISSION FILE")
print(f"{'='*60}")

# Average predictions (logits) from all models
avg_logits = np.mean(np.array(test_predictions), axis=0)

# Convert average logits to probabilities
probs = torch.softmax(torch.from_numpy(avg_logits), dim=-1).numpy()
final_test_preds = probs[:, 1]  # Get probability of class 1 (jailbreak)

# Check the correct column name for ID
id_col = 'Id' if 'Id' in test_df.columns else 'id'

# Create submission dataframe
submission_df = pd.DataFrame({
    id_col: test_df[id_col],
    'target': final_test_preds
})

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

print("\n✓ SUCCESS! Created 'submission.csv'")
print(f"File saved at: {os.path.abspath('submission.csv')}")
print(f"\nSubmission preview:")
print(submission_df.head(10))
print(f"\nTarget statistics:")
print(f"  Min:    {final_test_preds.min():.4f}")
print(f"  Max:    {final_test_preds.max():.4f}")
print(f"  Mean:   {final_test_preds.mean():.4f}")
print(f"  Median: {np.median(final_test_preds):.4f}")
print(f"\n{'='*60}")
print("Ready to submit to Kaggle!")
print(f"{'='*60}")

