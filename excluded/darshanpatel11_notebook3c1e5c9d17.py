import zipfile
from glob import glob
all_zip_files = glob("/kaggle/input/text-normalization-challenge-english-language/*.zip")
for f in all_zip_files:
    with zipfile.ZipFile(f, "r") as zip_file:
        zip_file.extractall(path='/kaggle/working/')


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text Normalization – T5‑small fine‑tuning with 5‑fold CV
Produces submission.csv in the current directory.
"""

import os
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import KFold

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

# --------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------
DATA_DIR = "/kaggle/working/"#DATA_DIR = "./data"
TRAIN_PATH = os.path.join(DATA_DIR, "en_train.csv")
TEST_PATH = os.path.join(DATA_DIR, "en_test_2.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "en_sample_submission_2.csv")

print("Loading training data ...")
train_df = pd.read_csv(
    TRAIN_PATH, usecols=["sentence_id", "token_id", "class", "before", "after"]
)
train_df["id"] = train_df.index.tolist()

print("Loading test data ...")
test_df = pd.read_csv(TEST_PATH, usecols=["sentence_id", "token_id", "before"])
test_df["id"] = test_df.index.tolist()
test_df["class"] = None
# --------------------------------------------------------------
# 2. Build source / target strings
# --------------------------------------------------------------
train_df["source"] = (
    train_df["class"].astype(str) + " " + train_df["before"].astype(str)
)
train_df["target"] = train_df["after"].astype(str)

test_df["source"] = (
    test_df["class"].fillna("PLAIN").astype(str) + " " + test_df["before"].astype(str)
)
# note: test file does NOT contain the class column; we fall back to "PLAIN"

# --------------------------------------------------------------
# 3. Tokenizer & constants
# --------------------------------------------------------------
tokenizer = T5TokenizerFast.from_pretrained("t5-small")
MAX_INPUT_LEN = 64
MAX_TARGET_LEN = 64
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------
# 4. Dataset class
# --------------------------------------------------------------
class NormDataset(Dataset):
    def __init__(
        self, df, tokenizer, max_input_len=64, max_target_len=64, is_train=True
    ):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.is_train = is_train
        self.sources = df["source"].tolist()
        if is_train:
            self.targets = df["target"].tolist()
        else:
            self.targets = None

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        src = self.sources[idx]
        enc = self.tokenizer(
            src,
            truncation=True,
            padding="max_length",
            max_length=self.max_input_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.is_train:
            tgt = self.targets[idx]
            dec = self.tokenizer(
                tgt,
                truncation=True,
                padding="max_length",
                max_length=self.max_target_len,
                return_tensors="pt",
            )
            item["labels"] = dec["input_ids"].squeeze(0)
            # Replace padding token id's of the labels by -100 as per HF convention
            item["labels"][item["labels"] == tokenizer.pad_token_id] = -100
        return item


# --------------------------------------------------------------
# 5. Helper: compute exact‑match accuracy on a DataLoader
# --------------------------------------------------------------
def compute_accuracy(model, dataloader, tokenizer):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_TARGET_LEN,
                num_beams=3,
                early_stopping=True,
            )
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            refs = tokenizer.batch_decode(
                batch["labels"]
                .cpu()
                .masked_fill(batch["labels"] == -100, tokenizer.pad_token_id),
                skip_special_tokens=True,
            )
            for p, r in zip(preds, refs):
                total += 1
                if p.strip() == r.strip():
                    correct += 1
    return correct / total if total > 0 else 0.0


# --------------------------------------------------------------
# 6. 5‑fold cross validation
# --------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

indices = np.arange(len(train_df))
for fold, (train_idx, val_idx) in enumerate(kf.split(indices), 1):
    print(f"\n=== Fold {fold} ===")
    train_split = train_df.iloc[train_idx].reset_index(drop=True)
    val_split = train_df.iloc[val_idx].reset_index(drop=True)

    train_dataset = NormDataset(train_split, tokenizer, is_train=True)
    val_dataset = NormDataset(val_split, tokenizer, is_train=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=None)

    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.to(DEVICE)

    # training arguments – short run (1 epoch) with early stopping
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./t5_fold_{fold}",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        save_strategy="no",
        num_train_epochs=1,
        learning_rate=5e-4,
        weight_decay=0.01,
        predict_with_generate=False,
        fp16=torch.cuda.is_available(),
        logging_steps=500,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # ------- Validation accuracy (exact match) -------
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )
    acc = compute_accuracy(model, val_loader, tokenizer)
    print(f"Fold {fold} token‑accuracy: {acc:.6f}")
    fold_accuracies.append(acc)

mean_acc = np.mean(fold_accuracies)
print(f"\n=== 5‑Fold Mean Token Accuracy: {mean_acc:.6f} ===")

# --------------------------------------------------------------
# 7. Train on full data
# --------------------------------------------------------------
print("\nTraining final model on the full training set ...")
full_dataset = NormDataset(train_df, tokenizer, is_train=True)

model_full = T5ForConditionalGeneration.from_pretrained("t5-small")
model_full.to(DEVICE)

training_args_full = Seq2SeqTrainingArguments(
    output_dir="./t5_full",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    learning_rate=5e-4,
    weight_decay=0.01,
    logging_steps=1000,
    fp16=torch.cuda.is_available(),
    report_to=[],
)

trainer_full = Seq2SeqTrainer(
    model=model_full,
    args=training_args_full,
    train_dataset=full_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=None),
)

trainer_full.train()

# --------------------------------------------------------------
# 8. Predict on test set
# --------------------------------------------------------------
print("\nGenerating predictions for test set ...")
test_dataset = NormDataset(test_df, tokenizer, is_train=False)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=DataCollatorForSeq2Seq(tokenizer, model=None),
)

model_full.eval()
predictions = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test Predict"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        gen_ids = model_full.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=MAX_TARGET_LEN,
            num_beams=3,
            early_stopping=True,
        )
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        predictions.extend([p.strip() for p in preds])

# --------------------------------------------------------------
# 9. Save submission
# --------------------------------------------------------------
submission = pd.DataFrame({"id": test_df["id"], "after": predictions})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"\nSubmission saved to {submission_path}")




