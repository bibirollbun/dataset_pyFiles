# --- 1. Configuration ---
class CFG:
    debug = False
    model_name = 'microsoft/deberta-v3-base'
    learning_rate = 1e-5
    batch_size = 8  # Adjust based on GPU memory
    num_epochs = 3
    num_classes = 65  # From your previous notebook
    n_splits = 5
    random_state = 42
    output_dir = 'exp002_deberta_baseline_output'


# --- 2. Setup & Imports ---
print("ðŸ”¹ Installing necessary libraries...")
!pip install -q transformers datasets evaluate accelerate

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 3. Load Data & Preprocessing ---
print("\nðŸ”¹ Loading and preparing data...")
try:
    df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
    submission_df = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv')
except FileNotFoundError:
    print("Running locally, ensure you have the competition data.")
    # Add your local data paths here if needed
    df = pd.read_csv('train.csv') # Example local path

# === SMOKE TEST MODIFICATION ===
if CFG.debug:
    print("ðŸ”¥ RUNNING IN DEBUG MODE ON A SMALL SUBSET (400 samples) ðŸ”¥")
    df = df.sample(n=400, random_state=CFG.random_state).reset_index(drop=True)

# Create target and text features
df['Misconception'] = df['Misconception'].fillna('NA').astype(str)
df['target'] = df['Category'] + ':' + df['Misconception']

le = LabelEncoder()
df['target_encoded'] = le.fit_transform(df['target'])

df['full_text'] = "question: " + df['QuestionText'].fillna('') + \
                  " [SEP] mc_answer: " + df['MC_Answer'].fillna('') + \
                  " [SEP] explanation: " + df['StudentExplanation'].fillna('')

# Convert to Hugging Face Dataset
full_dataset = Dataset.from_pandas(df)
print("Data loaded and prepared.")
print("-" * 50)


# --- 4. Tokenizer ---
print(f"ðŸ”¹ Loading tokenizer: {CFG.model_name}...")
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

def tokenize_function(examples):
    return tokenizer(examples['full_text'], padding='max_length', truncation=True, max_length=512)

print("Tokenizer loaded.")
print("-" * 50)



# --- 5. Custom MAP@3 Metric Function ---
def map_at_3(y_true, y_pred_proba):
    top_3_preds = np.argsort(-y_pred_proba, axis=1)[:, :3]
    avg_precisions = []
    for i in range(len(y_true)):
        true_label = y_true[i]
        top_3 = top_3_preds[i]
        if true_label in top_3:
            rank = np.where(top_3 == true_label)[0][0] + 1
            avg_precisions.append(1 / rank)
        else:
            avg_precisions.append(0)
    return np.mean(avg_precisions)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Apply softmax to convert logits to probabilities
    probabilities = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    score = map_at_3(labels, probabilities)
    return {'map_at_3': score}

print("Metric function defined.")
print("-" * 50)


# --- 6. Cross-Validation Training Loop ---
print("ðŸ”¹ Starting 5-fold cross-validation for DeBERTa model...")
skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.random_state)

oof_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['target_encoded'])):
    print(f"\n===== Fold {fold} =====")

    # Split data for this fold
    train_data = full_dataset.select(train_idx)
    val_data = full_dataset.select(val_idx)

    # Tokenize the datasets
    train_dataset = train_data.map(tokenize_function, batched=True)
    val_dataset = val_data.map(tokenize_function, batched=True)

    # Rename target column for the trainer
    train_dataset = train_dataset.rename_column("target_encoded", "labels")
    val_dataset = val_dataset.rename_column("target_encoded", "labels")

    # Define the model for this fold
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG.model_name,
        num_labels=CFG.num_classes
    ).to(device)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f"{CFG.output_dir}/fold_{fold}",
        learning_rate=CFG.learning_rate,
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.batch_size,
        num_train_epochs=CFG.num_epochs,
        weight_decay=0.01,
        eval_strategy="epoch", # Evaluate at the end of each epoch
        eval_steps=5 if CFG.debug else 500, # Quick evaluation
        save_strategy="epoch",       # Save a checkpoint at the end of each epoch
        save_steps=5 if CFG.debug else 500,
        max_steps=10 if CFG.debug else -1,
        load_best_model_at_end=True, # Load the best model based on the metric
        metric_for_best_model="map_at_3",
        save_total_limit=1,
        greater_is_better=True,
        report_to="none", # Disable wandb/tensorboard logging for simplicity
        fp16=True if device.type == 'cuda' else False # Enable mixed precision on GPU
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("Training model...")
    trainer.train()

    # Predict on validation set to get final fold score
    print("Evaluating on validation set...")
    predictions = trainer.predict(val_dataset)
    fold_score = predictions.metrics['test_map_at_3']
    oof_scores.append(fold_score)
    print(f"âœ… Fold {fold} MAP@3 Score: {fold_score:.4f}")

    # Clean up GPU memory
    del model, trainer
    torch.cuda.empty_cache()

    if CFG.debug:
        print("Debug mode: stopping after one fold.")
        break

# --- 7. Final Results ---
print("\n" + "="*50)
print("Cross-validation complete.")
print(f"Scores for each fold: {[round(s, 4) for s in oof_scores]}")
print(f"ðŸ“ˆ Average CV MAP@3 Score: {np.mean(oof_scores):.4f}")
print("="*50)




