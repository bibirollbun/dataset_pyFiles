import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

# --- CONFIGURATION ---
MODEL_NAME = '/kaggle/input/huggingfacedebertav3variants/deberta-v3-base'
EPOCHS = 2
MAX_LEN = 256
BATCH_SIZE = 8 


# --- LOAD DATA ---
df_train = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
df_test = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

df_train = df_train.rename(columns={'rule_violation': 'label'})

print(f"Train shape: {df_train.shape}")
df_train.head()


# --- FORMAT INPUT ---
def format_input(row):
    """Creates a descriptive prompt for the model."""
    return f"SUBREDDIT: {row.subreddit}\nRULE: {row.rule}\nCOMMENT: {row.body}"

df_train['text'] = df_train.apply(format_input, axis=1)
df_test['text'] = df_test.apply(format_input, axis=1)

print("Example of a formatted input prompt:")
print(df_train['text'].iloc[0])


# --- 4. Tokenization and Dataset Creation (Corrected) ---

train_df, val_df = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train['label'])

# Convert to Hugging Face Dataset
COLS = ['text','label']
train_ds = Dataset.from_pandas(train_df[COLS])
val_ds = Dataset.from_pandas(val_df[COLS])
test_ds = Dataset.from_pandas(df_test[['text']])

# Load tokenizer and tokenize datasets
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Set format for PyTorch
columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)
test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])


# --- TRAIN ---
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Define the metric we want to compute (AUC)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = softmax(logits, axis=1)[:, 1]
    auc = roc_auc_score(labels, probs)
    return {"auc": auc}

# Define training arguments
training_args = TrainingArguments(
    output_dir = "./deberta-output",
    num_train_epochs = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size = BATCH_SIZE * 2,
    learning_rate = 2e-5,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="auc",  # Use AUC to find the best model
    greater_is_better=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()


print("Making predictions on the test set...")
predictions = trainer.predict(test_ds)

probs = softmax(predictions.predictions, axis=1)[:, 1]

print("Creating submission file...")
submission_df = pd.DataFrame({
    'row_id': df_test['row_id'],
    'rule_violation': probs
})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
submission_df.head()

