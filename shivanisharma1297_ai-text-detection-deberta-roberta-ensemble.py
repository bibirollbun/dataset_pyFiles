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

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output 
# when you create a version using "Save & Run All". 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session.

# ====================================================
# Verify Environment
# ====================================================
import torch, transformers, datasets

print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Datasets:", datasets.__version__)

# ====================================================
# Mercor AI Text Detection - DeBERTa + RoBERTa Ensemble
# ====================================================
import gc
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

DATA_PATH = "/kaggle/input/mercor-ai-detection"
SEED = 42
MAX_LEN = 384
EPOCHS = 7
LR = 2e-5
BATCH_SIZE = 8

gc.collect()
torch.cuda.empty_cache()


def load_and_prepare():
    train = pd.read_csv(f"{DATA_PATH}/train.csv")
    test = pd.read_csv(f"{DATA_PATH}/test.csv")
    train["text"] = train["topic"].astype(str) + " " + train["answer"].astype(str)
    test["text"] = test["topic"].astype(str) + " " + test["answer"].astype(str)
    tr, val = train_test_split(train, test_size=0.2, stratify=train["is_cheating"], random_state=SEED)
    return tr, val, test


def tokenize(tokenizer, ds):
    def fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN)
    return ds.map(fn, batched=True)


def train_and_predict(model_name, train_df, val_df, test_df):
    print(f"\nğŸ”¹ Training {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    train_ds = tokenize(tokenizer, train_ds)
    val_ds = tokenize(tokenizer, val_ds)
    test_ds = tokenize(tokenizer, test_ds)

    train_ds = train_ds.rename_column("is_cheating", "labels")
    val_ds = val_ds.rename_column("is_cheating", "labels")

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format("torch", columns=cols)
    val_ds.set_format("torch", columns=cols)
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "id"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.config.use_cache = False  # required when using gradient checkpointing
# model.gradient_checkpointing_enable()  # Uncomment later if you want to experiment


    def compute_metrics(p):
        logits, labels = p
        preds = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        return {"roc_auc": roc_auc_score(labels, preds)}

    args = TrainingArguments(
        output_dir=f"./results_{model_name.split('/')[-1]}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        metric_for_best_model="roc_auc",
        load_best_model_at_end=True,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        logging_strategy="epoch",
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    preds = trainer.predict(val_ds)
    print("Validation ROC-AUC:", preds.metrics["test_roc_auc"])

    test_logits = trainer.predict(test_ds).predictions
    test_preds = torch.softmax(torch.tensor(test_logits), dim=1)[:, 1].numpy()

    # Free GPU memory before next model
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return test_preds, preds.metrics["test_roc_auc"]


train_df, val_df, test_df = load_and_prepare()

deberta_preds, deb_auc = train_and_predict("microsoft/deberta-v3-small", train_df, val_df, test_df)
roberta_preds, rob_auc = train_and_predict("roberta-base", train_df, val_df, test_df)

final_preds = (deberta_preds + roberta_preds) / 2

submission = pd.DataFrame({"id": test_df["id"], "is_cheating": final_preds})
submission.to_csv("submission.csv", index=False)

print("\nâœ… Ensemble complete")
print(f"DeBERTa ROC-AUC: {deb_auc:.4f} | RoBERTa ROC-AUC: {rob_auc:.4f}")
print("Saved submission.csv for upload ğŸš€")



import pandas as pd

sub = pd.read_csv("submission.csv")
sub.head()



sub.info()
print(f"\nRows: {len(sub)} | Columns: {list(sub.columns)}")



sample = pd.read_csv("/kaggle/input/mercor-ai-detection/sample_submission.csv")
print("Sample shape:", sample.shape)
print("Submission shape:", sub.shape)

assert list(sub.columns) == list(sample.columns), "âš ï¸� Column names don't match sample_submission.csv!"
print("âœ… Column names verified")


