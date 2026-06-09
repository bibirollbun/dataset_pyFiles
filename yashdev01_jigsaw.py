from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
import torch
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Silence warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load datasets
train_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')

# Prepare text and label
train_df['text'] = train_df['body'] + '[SEP]' + train_df['rule'] + '[SEP]' + train_df['subreddit']
train_df['label'] = train_df['rule_violation'].astype(int)

test_df['text'] = test_df['body'] + '[SEP]' + test_df['rule'] + '[SEP]' + test_df['subreddit']

# Train/val split
train_data, val_data = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=42)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("/kaggle/input/bert-base-uncased/pytorch/default/1/bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

# Convert to HF datasets
train_dataset = Dataset.from_pandas(train_data[['text', 'label']])
val_dataset = Dataset.from_pandas(val_data[['text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['text']])

# Apply tokenization
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Load model
model = BertForSequenceClassification.from_pretrained("/kaggle/input/bert-base-uncased/pytorch/default/1/bert-base-uncased", num_labels=2).to(device)

# Metric function
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    metric_for_best_model="eval_accuracy",  
    greater_is_better=True,       
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    learning_rate=0.0001,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=25,
    report_to="none",             
    disable_tqdm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate
print("Validation Metrics:", trainer.evaluate())

# Predict on test set
preds = trainer.predict(test_dataset)
probabilities = torch.nn.functional.softmax(torch.tensor(preds.predictions), dim=1)[:, 1].numpy()

# Prepare submission
submission = pd.DataFrame({
    "row_id": test_df["row_id"],
    "rule_violation": probabilities.round(4)  # round to 4 decimals
})

# Save to CSV
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission saved to submission.csv")


submission




