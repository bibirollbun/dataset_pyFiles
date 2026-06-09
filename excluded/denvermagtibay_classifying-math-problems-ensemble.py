# =======================
# ğŸš€ Setup Environment
# =======================
import torch
import numpy as np
import pandas as pd
import re
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from datasets import Dataset
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Disable WandB logging
os.environ["WANDB_DISABLED"] = "true"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =======================
# ğŸš€ Config
# =======================
SEED = 42
MODELS = ["microsoft/deberta-v3-base", "roberta-large"]
NUM_LABELS = 8
MAX_LENGTH = 384
BATCH_SIZE = 4
EPOCHS = 3
MINORITY_CLASSES = [6, 7]  # Linear Algebra and Abstract/Topology


# =======================
# ğŸ“Œ Load and Clean Data
# =======================
train_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")
test_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\+\-\*/=^()., ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

train_df['cleaned'] = train_df['Question'].apply(preprocess)
test_df['cleaned'] = test_df['Question'].apply(preprocess)


# =======================
# ğŸ”„ Balance Minority Classes
# =======================
def balance_classes(train_df):
    minority_classes = train_df[train_df['label'].isin(MINORITY_CLASSES)]
    majority_classes = train_df[~train_df['label'].isin(MINORITY_CLASSES)]

    # Oversample the minority classes to balance
    minority_upsampled = resample(minority_classes, 
                                  replace=True,
                                  n_samples=200,
                                  random_state=SEED)

    train_df_balanced = pd.concat([majority_classes, minority_upsampled])
    return train_df_balanced.sample(frac=1).reset_index(drop=True)

train_df = balance_classes(train_df)


# =======================
# ğŸ”„ Tokenization
# =======================
tokenizer = AutoTokenizer.from_pretrained(MODELS[0])

def tokenize_function(example):
    return tokenizer(
        example["cleaned"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

train_dataset = Dataset.from_pandas(train_df[['cleaned', 'label']])
train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.rename_column("label", "labels")
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# =======================
# ğŸ“� Prepare Test Set
# =======================
test_dataset = Dataset.from_pandas(test_df[['cleaned']])
test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# =======================
# ğŸš€ Train and Analyze
# =======================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
labels = train_df["label"].values
all_fold_logits = []

# Learning Rate Scheduler
def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, labels)):
    print(f"\nğŸŸ¦ Fold {fold + 1} Training")
    
    train_split = train_dataset.select(train_idx.tolist())
    val_split = train_dataset.select(val_idx.tolist())

    # âœ… Use the correct model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODELS[0], num_labels=NUM_LABELS
    ).to(device)

    args = TrainingArguments(
        output_dir=f"./deberta-fold-{fold}",
        do_train=True,
        do_eval=True,
        learning_rate=3e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir=f"./logs-{fold}",
        seed=SEED,
        logging_steps=50,
        save_strategy="no",
        fp16=True
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"f1": f1_score(labels, preds, average="micro")}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_split,
        eval_dataset=val_split,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    trainer.train()
    eval_result = trainer.evaluate()
    print(f"âœ… Fold {fold + 1} F1-micro: {eval_result['eval_f1']:.4f}")

    # Predict on test set and store logits
    test_predictions = trainer.predict(test_dataset)
    print(f"Logits shape for fold {fold + 1}: {test_predictions.predictions.shape}")

    all_fold_logits.append(test_predictions.predictions)



# =======================
# ğŸ“� Submission Generation
# =======================
all_fold_logits = np.array(all_fold_logits)
avg_logits = np.mean(all_fold_logits, axis=0)
final_preds = np.argmax(avg_logits, axis=1)

submission = pd.DataFrame({
    "id": test_df.index,
    "label": final_preds
})
submission.to_csv("submission.csv", index=False)
print("âœ… Saved submission.csv â€” Ready to upload!")

