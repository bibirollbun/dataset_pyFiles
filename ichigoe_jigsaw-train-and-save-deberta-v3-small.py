%%time

import random
import os
import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *
import re
import torch
from transformers import AutoTokenizer, AutoConfig
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import gc
import shutil
import warnings 
warnings.filterwarnings('ignore')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

version = 1
base_model_name = "microsoft/deberta-v3-small"
EPOCHS = 4
MAX_LEN = 488
SEED = 42
CLASSES = 1

# Set output directory to kaggle working directory
DIR = "/kaggle/working"
os.makedirs(DIR, exist_ok=True)
set_seed(SEED)

# Environment variables
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Clear GPU cache
torch.cuda.empty_cache()

print(f"Using model: {base_model_name}")


%%time

# Load training data
train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")

# Initialize tokenizer from base model
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

def make_prompt(row):
    """Create structured prompt for few-shot learning approach"""
    return f"""[RULE]: {row['rule']}
[SUBREDDIT]: {row['subreddit']}

[COMMENT]: {row['body']}

[POSITIVE EXAMPLES]:
1. {row['positive_example_1']}
2. {row['positive_example_2']}

[NEGATIVE EXAMPLES]:
1. {row['negative_example_1']}
2. {row['negative_example_2']}

[QUESTION]: Does the comment violate the rule?
[ANSWER]:"""
    
# Apply prompt formatting to training data
train['text'] = train.apply(make_prompt, axis=1)

# Split data into train and validation sets
train_, val_ = train_test_split(train, test_size=0.2, random_state=42)
train_["label"] = train_["rule_violation"].astype(float)
val_["label"] = val_["rule_violation"].astype(float)

# Create datasets for training
features_cols = ['text', 'label']
train_ds = Dataset.from_pandas(train_[features_cols])
val_ds = Dataset.from_pandas(val_[features_cols])

def tokenize(batch):
    """Tokenize text with padding and truncation"""
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)
    
# Apply tokenization
train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Set format for PyTorch
columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)


%%time

def compute_column_auc(eval_pred):
    """Compute AUC metric for evaluation"""
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))

    if probs.ndim == 1 or probs.shape[1] == 1:
        auc = roc_auc_score(labels, probs)
        return {"auc": auc}

    aucs = []
    for i in range(probs.shape[1]):
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
        except ValueError:
            auc = 0.5
        aucs.append(auc)
    return {"mean_column_auc": np.mean(aucs)}

# Initialize model from base pretrained model
model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name,
    num_labels=CLASSES,
)

# Configure training arguments
training_args = TrainingArguments(
    output_dir=DIR,
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    logging_steps=50,
    save_total_limit=1,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=7,
    learning_rate=2e-5,
    num_train_epochs=EPOCHS,
    gradient_accumulation_steps=1, 
    load_best_model_at_end=True,
    metric_for_best_model="auc",
    greater_is_better=True,
    fp16=True,   
    bf16=False,  
    report_to="none",
    logging_dir="./logs",
    seed=SEED,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_column_auc,
)


%%time

# Train the model
print("Starting model training...")
trainer.train()
print("Training completed!")


torch.cuda.empty_cache()
gc.collect()


%%time

# Evaluate the trained model
results = trainer.evaluate()
print("Mean_column_AUC:", results['eval_auc'])


%%time

# Save the trained model to /kaggle/working directory
print("Saving model and tokenizer...")

# Save model (this will create model.safetensors and config.json)
trainer.save_model(DIR)

# Save tokenizer (this will create tokenizer files)
tokenizer.save_pretrained(DIR)

# Clean up checkpoint directories
print("Cleaning up checkpoint directories...")
for item in os.listdir(DIR):
    item_path = os.path.join(DIR, item)
    if os.path.isdir(item_path) and item.startswith('checkpoint-'):
        shutil.rmtree(item_path)
        print(f"  Removed: {item}")

# List all saved files
saved_files = os.listdir(DIR)
print("Files saved in /kaggle/working:")
for file in sorted(saved_files):
    if os.path.isfile(os.path.join(DIR, file)):  # Only show files, not directories
        print(f"  {file}")

print("Model and tokenizer saved successfully!")

