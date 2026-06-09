# Imports Section
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.metrics import f1_score
import os
import logging
import warnings


# Suppress Hugging Face and tokenizer-related warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Core configuration
MODEL_NAME = "microsoft/deberta-v3-small"
EPOCHS = 3
BATCH_SIZE = 32
LR = 8e-5
MAX_LEN = 256

print("Config loaded. Model:", MODEL_NAME)


# Load dataset files
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
sample_sub = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

# Handle missing text values and prepare input columns
train_df["text"] = train_df["text"].fillna("")
test_df["text"] = test_df["text"].fillna("")
train_df["input"] = train_df["text"]
test_df["input"] = test_df["text"]

# Rename target to 'labels' for Hugging Face compatibility
train_df = train_df.rename(columns={"target": "labels"})

# Output checks
train_df.describe(include='object')


# Converts Kaggle dataframe to HuggingFace dataset using import
train_ds = Dataset.from_pandas(train_df[["input", "labels"]])
test_ds = Dataset.from_pandas(test_df[["input"]])

print("Train dataset (example format):")
train_ds[1]


# Tokeniser helper function
def tokenize(batch):
    return tokenizer(batch["input"], truncation=True, max_length=MAX_LEN)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Show tokenisation breakdown for a sample input
raw_text = train_ds[1]["input"]
tokens = tokenizer.tokenize(raw_text)

print(f"Example text:\n {raw_text}")
print(f"\nTokenised into: {tokens}")

# Then continue with mapping
train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)



# Split off 20% of training data for validation
split_ds = train_ds.train_test_split(test_size=0.2, seed=42)
print("Train size:", len(split_ds["train"]))
print("Validation size:", len(split_ds["test"]))


# Load pretrained model with two output classes
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Automatically pad sequences in each batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Define custom evaluation metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"f1": f1_score(labels, preds)}


# === TRAINING ARGS ===
training_args = TrainingArguments(
    output_dir="output",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="no",
    weight_decay=0.01,
    warmup_ratio=0.1,
    report_to="none",
    fp16=True,
    disable_tqdm=False
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_ds["train"],
    eval_dataset=split_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train();


# Run predictions on the test dataset
preds = trainer.predict(test_ds).predictions
class_preds = np.argmax(preds, axis=1)

print("First 5 predictions:", class_preds[:5])


# Add predictions to the sample submission
sample_sub["target"] = class_preds

# Save predictions with custom filename
base = MODEL_NAME.split("/")[-1]
sample_sub.to_csv(f"submission_{base}_ep{EPOCHS}.csv", index=False)
sample_sub.to_csv("submission.csv", index=False)

print(sample_sub.head())


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Get validation predictions
val_preds = trainer.predict(split_ds['test'])
val_labels = val_preds.label_ids
val_classes = np.argmax(val_preds.predictions, axis=1)

# Print classification report
print("Validation Classification Report:")
print(classification_report(val_labels, val_classes))

cm = confusion_matrix(val_labels, val_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Disaster", "Disaster"])
disp.plot(cmap='Blues')


val_df = pd.DataFrame(split_ds["test"])
val_df["predicted"] = val_classes
val_df["actual"] = val_labels
val_df["conf"] = np.max(val_preds.predictions, axis=1)

# Top correct predictions (most confident)
correct = val_df[val_df.predicted == val_df.actual].sort_values("conf", ascending=False).head(5)
print("Most Confident Correct Predictions:")
print(correct[["input", "actual", "predicted", "conf"]])

# Top incorrect predictions (most confident errors)
incorrect = val_df[val_df.predicted != val_df.actual].sort_values("conf", ascending=False).head(5)
print("Most Confident Wrong Predictions:")
print(incorrect[["input", "actual", "predicted", "conf"]])




