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


import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# Set environment variable to disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration
MODEL_CHECKPOINT = "bert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5  # Optimal learning rate for BERT
WEIGHT_DECAY = 0.01
OUTPUT_DIR = "/kaggle/working/bert-math-classifier"
TRAIN_FILE = "/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv"
TEST_FILE = "/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv"
SUBMISSION_FILE = "/kaggle/working/submission.csv"

# Labels
id2label = {
    0: "Algebra",
    1: "Geometry and Trigonometry",
    2: "Calculus and Analysis",
    3: "Probability and Statistics",
    4: "Number Theory",
    5: "Combinatorics and Discrete Math",
    6: "Linear Algebra",
    7: "Abstract Algebra and Topology"
}
label2id = {v: k for k, v in id2label.items()}
NUM_LABELS = len(id2label)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Using model: {MODEL_CHECKPOINT}")
print(f"Number of labels: {NUM_LABELS}")

# Load data
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# Text cleaning
def clean_text(text):
    return text.lower().strip()

train_df['Question'] = train_df['Question'].apply(clean_text)
test_df['Question'] = test_df['Question'].apply(clean_text)

# Train/validation split
train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=42)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["Question"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

# Create datasets
from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Class weights for imbalanced classes
class_weight = torch.tensor([1 / (count + 1e-5) for count in train_df['label'].value_counts().sort_index()])
class_weight /= class_weight.sum()

# Initialize model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metric computation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    logging_dir=f"{OUTPUT_DIR}/logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

# Custom trainer with weighted loss
from torch.nn import CrossEntropyLoss
from transformers import Trainer

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Initialize trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    class_weights=class_weight,
)

# Start training
print("Starting training...")
train_result = trainer.train()
print("Training completed.")

# Evaluate
print("Evaluating...")
eval_metrics = trainer.evaluate()
print(f"Validation Accuracy: {eval_metrics['eval_accuracy']*100:.2f}%")

# Predict on test set
print("Predicting...")
predictions = trainer.predict(tokenized_test)
preds = predictions.predictions.argmax(axis=-1)

# Generate submission
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'label': preds
})
submission_df.to_csv(SUBMISSION_FILE, index=False)
print(f"Submission saved to {SUBMISSION_FILE}")

