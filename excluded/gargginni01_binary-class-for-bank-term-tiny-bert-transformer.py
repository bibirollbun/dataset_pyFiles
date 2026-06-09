%pip install evaluate


import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch

# -----------------------
# 1. Load Data
# -----------------------
# data_dir = Path("/kaggle/input/playground-series-s5e8")
# train_path = data_dir / "train.csv"
# test_path = data_dir / "test.csv"
# sample_path = data_dir / "sample_submission.csv"
out_path = "/kaggle/working/submission.csv"

train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

TARGET = "y"
ID_COL = "id"
FEATURES = [col for col in train.columns if col not in [ID_COL, TARGET]]

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Features:", FEATURES)

# -----------------------
# 2. Prepare Text Feature
# -----------------------
# Concatenate features into one text column (since HuggingFace works with text input)
train["text"] = train[FEATURES].astype(str).agg(" ".join, axis=1)
test["text"] = test[FEATURES].astype(str).agg(" ".join, axis=1)

# Train/Validation split
train_df, valid_df = train_test_split(train, test_size=0.2, stratify=train[TARGET], random_state=42)

# Convert to HuggingFace Dataset
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df[["text", TARGET]]).rename_column(TARGET, "labels"),
    "validation": Dataset.from_pandas(valid_df[["text", TARGET]]).rename_column(TARGET, "labels"),
    "test": Dataset.from_pandas(test[["text", ID_COL]])
})
print(dataset["train"])
# -----------------------
# 3. Tokenizer
# -----------------------
# MODEL = "distilbert-base-uncased"
# MODEL = "huawei-noah/TinyBERT_General_4L_312D"
MODEL = "ginnigarg/binary-classification-kaggle-tiny-bert"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# Rename target to "labels" for Trainer
# Using map
# train_ds = train_ds.map(lambda x: {"text_length": len(x["input_ids"])})
# dataset["train"] = dataset["train"].rename_column(TARGET, "labels")
# dataset["validation"] = dataset["validation"].rename_column(TARGET, "labels")

# Set format for PyTorch
dataset['train'].set_format("torch", columns=["input_ids", "attention_mask", "labels"], output_all_columns=False)
dataset['validation'].set_format("torch", columns=["input_ids", "attention_mask", "labels"], output_all_columns=False)


# -----------------------
# 4. Model
# -----------------------
num_labels = len(train[TARGET].unique())  # should be 2 for binary classification
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels)

BATCH_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to GPU
model.to(device)
model.eval()

# -----------------------
# 5. Training Setup
# -----------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=15,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
)

# Metrics
from evaluate import load
accuracy = load("accuracy")
roc_auc = load("roc_auc")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=preds, references=labels)
    auc = roc_auc.compute(prediction_scores=logits[:,1], references=labels)
    return {"accuracy": acc["accuracy"], "roc_auc": auc["roc_auc"]}

# -----------------------
# 6. Trainer
# -----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# -----------------------
# 7. Train
# -----------------------
trainer.train()

# -----------------------
# 8. Predictions on Test
# -----------------------

test_texts = test["text"].tolist()
all_probs = []

with torch.no_grad():
    for i in range(0, len(test_texts), BATCH_SIZE):
        batch_texts = test_texts[i:i+BATCH_SIZE]
        batch_encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        # Move batch tensors to GPU
        batch_encodings = {k: v.to(device) for k, v in batch_encodings.items()}

        outputs = model(**batch_encodings)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())  # move back to CPU for storage

# Convert to numpy array
all_probs = np.array(all_probs)
        
# -----------------------
# 9. Save Submission
# -----------------------
submission = pd.DataFrame({
    "id": test[ID_COL],
    "y": (all_probs > 0.5).astype(int)  # threshold at 0.5
})
submission.to_csv(out_path, index=False)

print("✅ Saved submission.csv")


