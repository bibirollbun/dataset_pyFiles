
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



#Observing the dataset

import pandas as pd

train_path = '/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv'
test_path = '/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train.rename(columns={"Question": "question"}, inplace=True)
test.rename(columns={"Question": "question"}, inplace=True)

print('first 5 rows of the training data')
train.head()



#Installing transformers and datasets
!pip install -q transformers datasets


# Import libraries
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from datasets import Dataset

# Step 1: Prepare your original full training data (pandas DataFrame)
# Assume your labeled data is in a dataframe called `train` with 'question' and 'label' columns
train_df, eval_df = train_test_split(train, test_size=0.1, random_state=42, stratify=train['label'])

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
test_dataset = Dataset.from_pandas(test)  # Unlabeled test data from Kaggle

# Step 2: Tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["question"], padding="max_length", truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)


import transformers
print(transformers.__version__)



!pip install -U transformers



from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch

# Step 3: Load model and send to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=8)
model.to(device)

# Step 4: Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": acc, "f1": f1}

# Step 5: Training arguments (manual eval â€” no evaluation_strategy)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    save_steps=500,
    do_train=True,
    do_eval=False,           # ðŸ‘ˆ Disable built-in eval
    report_to="none"
)

# Step 6: Trainer (no eval_dataset since we eval manually)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
)

# Step 7: Train
trainer.train()

# Step 8: Manually Evaluate on validation set
outputs = trainer.predict(tokenized_eval_dataset)
predictions = np.argmax(outputs.predictions, axis=1)
labels = outputs.label_ids

# Step 9: Calculate metrics
acc = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions, average="weighted")
print(f"Manual Evaluation Metrics:\n  Accuracy: {acc:.4f}\n  F1 Score: {f1:.4f}")



# Step 8: Predict on unlabeled test set
predictions = trainer.predict(tokenized_test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)

# Step 9: Prepare submission
test["label"] = pred_labels
submission = test[["id", "label"]]  # Assuming your test set has an 'id' column
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

