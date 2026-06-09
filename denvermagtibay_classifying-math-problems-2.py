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
from sklearn.metrics import f1_score
from datasets import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Disable WandB logging
os.environ["WANDB_DISABLED"] = "true"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Config
SEED = 42
MODEL_NAME = "microsoft/deberta-v3-base"
NUM_LABELS = 8
MAX_LENGTH = 384
BATCH_SIZE = 4
EPOCHS = 3

# Load data
train_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")
test_df = pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")

# Clean text
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\+\-\*/=^()., ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

train_df['cleaned'] = train_df['Question'].apply(preprocess)
test_df['cleaned'] = test_df['Question'].apply(preprocess)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(example):
    return tokenizer(
        example["cleaned"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

# Convert to HF dataset
train_dataset = Dataset.from_pandas(train_df[['cleaned', 'label']])
train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.rename_column("label", "labels")
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# Prepare test set
test_dataset = Dataset.from_pandas(test_df[['cleaned']])
test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
labels = train_df["label"].values
all_fold_logits = []

# Train and predict on test set for each fold
for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, labels)):
    print(f"\nğŸŸ¦ Fold {fold + 1} Training")

    train_split = train_dataset.select(train_idx.tolist())
    val_split = train_dataset.select(val_idx.tolist())

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    ).to(device)

    args = TrainingArguments(
        output_dir=f"./deberta-fold-{fold}",
        do_train=True,
        do_eval=True,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir=f"./logs-{fold}",
        seed=SEED,
        logging_steps=50,
        save_strategy="no"
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

    from sklearn.metrics import classification_report

    # Predict on validation set
    val_preds = trainer.predict(val_split)
    val_logits = val_preds.predictions
    val_preds_labels = np.argmax(val_logits, axis=1)
    val_true_labels = val_split["labels"]

    # ğŸ“Š Per-Class Error Analysis
    print(f"\nğŸ“Š Classification Report for Fold {fold + 1}:\n")
    print(classification_report(
        val_true_labels,
        val_preds_labels,
        digits=4,
        target_names=[
            "Algebra",
            "Geometry/Trig",
            "Calculus",
            "Probability",
            "Number Theory",
            "Combinatorics",
            "Linear Algebra",
            "Abstract/Topology"
        ]
    ))

    # ğŸ“‰ Confusion Matrix
    cm = confusion_matrix(val_true_labels, val_preds_labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Algebra", "Geometry/Trig", "Calculus", "Probability",
            "Number Theory", "Combinatorics", "Linear Algebra", "Abstract/Topology"
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix â€“ Fold {fold + 1}")
    plt.show()

    # ğŸ•µï¸� Show some misclassified examples
    print("\nâ�Œ Sample Misclassifications:")
    val_df = train_df.iloc[val_idx].copy()
    val_df["true_label"] = val_true_labels
    val_df["pred_label"] = val_preds_labels
    val_df_errors = val_df[val_df["true_label"] != val_df["pred_label"]]

    # Show 5 random mistakes
    for i, row in val_df_errors.sample(min(5, len(val_df_errors)), random_state=SEED).iterrows():
        print(f"\nğŸ”¹ Question: {row['Question']}")
        print(f"   âœ… True: {row['true_label']} | â�Œ Predicted: {row['pred_label']}")


    # Predict on test set and store logits
    test_predictions = trainer.predict(test_dataset)
    all_fold_logits.append(test_predictions.predictions)


# Average logits from all folds
avg_logits = np.mean(all_fold_logits, axis=0)
final_preds = np.argmax(avg_logits, axis=1)

# Create submission
submission = pd.DataFrame({
    "id": test_df.index,
    "label": final_preds
})
submission.to_csv("submission.csv", index=False)
print("âœ… Saved submission.csv â€” Ready to upload!")

