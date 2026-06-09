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


 # imdb_tinybert_full_run.py
# Requirements:
# pip install -U transformers datasets scikit-learn matplotlib seaborn torch packaging

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from packaging import version

print("transformers version:", transformers.__version__)

# Step 1: Load the IMDb dataset (500 samples for quick training)
raw = load_dataset("imdb")
dataset = raw["train"].shuffle(seed=42).select(range(500)).train_test_split(test_size=0.2)

# Step 2: Load TinyBERT tokenizer and model
model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Step 3: Tokenize the dataset
def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Ensure label column exists
if "label" not in tokenized_dataset["train"].column_names:
    raise ValueError("Expected a 'label' column in the dataset")

# Set format for Trainer (PyTorch tensors)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Step 4: Define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds)}

# Step 5: Set training arguments (version-aware)
if version.parse(transformers.__version__) >= version.parse("4.3.0"):
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",      # ⬅️ FIXED: Changed 'evaluation_strategy' to 'eval_strategy'
        save_strategy="no",         # do not save intermediate checkpoints (or use "epoch")
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./logs",
        logging_steps=10,
        seed=42,
        remove_unused_columns=False,
        report_to="none",           # ⬅️ ADDED: Prevents unwanted WandB login interruption
    )
else:
    print("Old transformers version detected — using fallback TrainingArguments without evaluation_strategy/save_strategy")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./logs",
        logging_steps=10,
        seed=42,
        remove_unused_columns=False,
        report_to="none",           # ⬅️ ADDED: Prevents unwanted WandB login interruption
    )

# Step 6: Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

# Step 7: Train the model
trainer.train()

# Step 8: Make predictions on the test set
predictions = trainer.predict(tokenized_dataset["test"])
y_true = predictions.label_ids
y_pred = predictions.predictions.argmax(axis=-1)

# Step 9: Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ Test Accuracy: {accuracy:.4f}\n")

# Step 10: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Step 11: Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))

# Step 12: Plot the confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - TinyBERT IMDb")
plt.tight_layout()
plt.show()




