import pandas as pd
import re
import numpy as np
import torch
import torch.nn.functional as F
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


# Disable wandb to prevent hanging
os.environ['WANDB_DISABLED'] = 'true'


# Load competition dataset
train_comp_df = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/train_essays.csv')

# Load DAIGT-V2 external dataset
daigt_df = pd.read_csv('/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv')
daigt_df = daigt_df[daigt_df['RDizzl3_seven'] == True][['text', 'label']].rename(columns={'label': 'generated'})

# Merge datasets
train_df = pd.concat([train_comp_df[['text', 'generated']], daigt_df], ignore_index=True)


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

train_df['text'] = train_df['text'].apply(clean_text)

# Deduplicate to reduce data leakage and overfitting
train_df = train_df.drop_duplicates(subset=['text']).reset_index(drop=True)
print(train_df['generated'].value_counts())  # Check balance

# Check unique samples count
print(f"Total unique texts: {train_df['text'].nunique()} / {len(train_df)}")


# Split into train and validation sets (80-20)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'].tolist(), train_df['generated'].tolist(), test_size=0.2, stratify=train_df['generated']
)

# Check train/val overlap
train_set = set(train_texts)
val_set = set(val_texts)
overlap = train_set.intersection(val_set)
print(f"Train/Val overlap: {len(overlap)} samples")


# Create Hugging Face Dataset
train_dataset = Dataset.from_pandas(pd.DataFrame({'text': train_texts, 'generated': train_labels}))
val_dataset = Dataset.from_pandas(pd.DataFrame({'text': val_texts, 'generated': val_labels}))


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Define tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

# Apply tokenization
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Rename label column to 'labels'
tokenized_train = tokenized_train.rename_column("generated", "labels")
tokenized_val = tokenized_val.rename_column("generated", "labels")

# Set format to torch
tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_val.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Print tokenized sample (check dtype)
print("Tokenized Train Sample:")
print(tokenized_train[0])
print(f"Labels dtype: {tokenized_train[0]['labels'].dtype}")  # Should be torch.int64


# Load model (num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define compute_metrics function (using softmax)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = F.softmax(torch.tensor(logits), dim=-1)[:, 1].cpu().numpy()
    auc = roc_auc_score(labels, probs)
    return {'roc_auc': auc}

# TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.1,  # Increase to reduce overfitting
    logging_dir='./logs',
    logging_steps=10,
    eval_steps=500,
    save_steps=500,
    load_best_model_at_end=True,
    eval_strategy='steps',
    metric_for_best_model='roc_auc',
    greater_is_better=True,
    report_to="none"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics
)


# Start training
trainer.train()


# Evaluate AUC-ROC
eval_results = trainer.evaluate()
print(f"Validation ROC-AUC: {eval_results['eval_roc_auc']:.3f}")


# Load test data and predict
test_df = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
test_df['text'] = test_df['text'].apply(clean_text)
test_dataset = Dataset.from_pandas(test_df[['text']])
tokenized_test = test_dataset.map(tokenize_function, batched=True)
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask'])

# Predict
test_preds = trainer.predict(tokenized_test).predictions
print(f"Test preds shape: {test_preds.shape}")  # Debug

# Compute probabilities (using softmax)
test_probs = F.softmax(torch.tensor(test_preds), dim=-1)[:, 1].numpy()

# Save submission file
submission = pd.DataFrame({'id': test_df['id'], 'generated': test_probs})
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")

