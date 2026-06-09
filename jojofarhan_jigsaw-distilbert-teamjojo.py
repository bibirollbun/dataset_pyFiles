# Jigsaw - Agile Community Rules Classification Notebook

import transformers
from packaging import version
transformers_version = version.parse(transformers.__version__)
from transformers import (
    AutoTokenizer,  # Still using AutoTokenizer for flexibility
    AutoModelForSequenceClassification,  # Still using AutoModelForSequenceClassification
    Trainer,
    TrainingArguments
)
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os



print(transformers.__version__)




os.environ["WANDB_DISABLED"] = "true"

# Configurations
MODEL_NAME = "/kaggle/input/distilbert-base-model-uncased/distilbert-base-uncased"  # Changed from /kaggle/input/roberta-base
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 3

# Load data
train_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

# Combine rule and examples into prompt
def build_prompt(row):
    prompt = f"Rule: {row['rule']}\n"
    prompt += f"Positive 1: {row['positive_example_1']}\n"
    prompt += f"Positive 2: {row['positive_example_2']}\n"
    prompt += f"Negative 1: {row['negative_example_1']}\n"
    prompt += f"Negative 2: {row['negative_example_2']}\n"
    prompt += f"Comment: {row['body']}"
    return prompt

train_df['prompt'] = train_df.apply(build_prompt, axis=1)

# Tokenizer class
class JigsawDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # Loads DistilBertTokenizer
        self.encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        self.labels = labels.values if labels is not None else None

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# Split and create datasets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['prompt'],
    train_df['rule_violation'],
    stratify=train_df['rule_violation'],
    test_size=0.2,
    random_state=42
)

train_dataset = JigsawDataset(train_texts, train_labels)
val_dataset = JigsawDataset(val_texts, val_labels)

# Model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)  # Loads DistilBertForSequenceClassification

# Metrics
def compute_metrics(pred):
    preds = pred.predictions.ravel()
    labels = pred.label_ids.ravel()
    return {"roc_auc": roc_auc_score(labels, preds)}

# Note: You may see a warning like:
# "Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']"
# This is expected because the model is not pre-trained for sequence classification and will be fine-tuned.

print(transformers.__version__)  # Should still be 4.52.4 or compatible




# Trainer setup
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    report_to="none"  # Disable wandb
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)



# Train
trainer.train()



# Inference on test set
def build_test_prompt(row):
    return f"Rule: {row['rule']}\nComment: {row['body']}"

test_df['prompt'] = test_df.apply(build_test_prompt, axis=1)
test_dataset = JigsawDataset(test_df['prompt'])
preds = trainer.predict(test_dataset)
test_df['rule_violation'] = torch.sigmoid(torch.tensor(preds.predictions)).numpy().ravel()

# Submission
submission = test_df[['row_id', 'rule_violation']]
submission.to_csv("submission.csv", index=False)

# Check for the file
print("Files in current directory:", os.listdir())

# Preview the submission
submission = pd.read_csv("submission.csv")
submission.head(10)

