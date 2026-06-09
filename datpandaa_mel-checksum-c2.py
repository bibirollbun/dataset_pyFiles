!pip install -U transformers


# Config and seeding
import os
import random
import numpy as np
import torch
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path("/kaggle/input/rmit-hackathon-2025")
DATA_DIR = BASE_DIR
DEVICE



# Load data and preprocess
import pandas as pd
from sklearn.model_selection import train_test_split

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

for df in (train_df, test_df):
    if "text" in df.columns:
        df["text"] = df["text"].fillna("").astype(str)

label2id = {"benign": 0, "jailbreak": 1}
id2label = {v: k for k, v in label2id.items()}
train_df["label"] = train_df["label"].map(label2id).astype(int)

train_df, val_df = train_test_split(
    train_df, test_size=0.1, random_state=SEED, stratify=train_df["label"]
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_df.shape, val_df.shape



# Simple data augmentation (EDA-lite: deletion/swap/insertion)
from typing import List
import re


def tokenize_to_words(text: str) -> List[str]:
    return text.split()


def random_deletion(words: List[str], drop_prob: float = 0.1) -> List[str]:
    if len(words) <= 1:
        return words
    kept = [w for w in words if random.random() > drop_prob]
    if not kept:
        kept = [random.choice(words)]
    return kept


def random_swap(words: List[str], n_swaps: int = 1) -> List[str]:
    if len(words) < 2:
        return words
    words = words.copy()
    for _ in range(n_swaps):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return words


def random_insertion(words: List[str], n_inserts: int = 1) -> List[str]:
    if not words:
        return words
    words = words.copy()
    for _ in range(n_inserts):
        w = random.choice(words)
        pos = random.randint(0, len(words))
        words.insert(pos, w)
    return words


def augment_text_once(text: str) -> str:
    words = tokenize_to_words(text)
    if not words:
        return text
    op = random.choice(["delete", "swap", "insert"])
    if op == "delete":
        aug = random_deletion(words, drop_prob=0.1)
    elif op == "swap":
        aug = random_swap(words, n_swaps=1)
    else:
        aug = random_insertion(words, n_inserts=1)
    return " ".join(aug)


# Create one augmented sample per training row and append to train set
aug_texts = [augment_text_once(t) for t in train_df["text"].tolist()]
aug_df = pd.DataFrame({"text": aug_texts, "label": train_df["label"].values})
train_df = pd.concat([train_df, aug_df], ignore_index=True)
train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print(f"Augmented train size: {len(train_df)} (added {len(aug_df)} augmented rows)")



# Tokenizer with snapshot fallback
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

MAX_LENGTH = 256


def build_tokenizer(model_name: str):
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        print(f"Tokenizer remote load failed for {model_name}: {e}\nFalling back to local snapshot...")
        local_dir = snapshot_download(repo_id=model_name)
        tok = AutoTokenizer.from_pretrained(local_dir, use_fast=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if hasattr(tok, "eos_token") else tok.sep_token
    return tok

roberta_name = "roberta-base"
roberta_tokenizer = build_tokenizer(roberta_name)
print("Tokenizer ready.")



# Torch dataset
import torch
from torch.utils.data import Dataset


def tokenize_df(df, tokenizer):
    return tokenizer(
        list(df["text"].values),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        enc = tokenize_df(df, tokenizer)
        self.enc = enc
        self.labels = torch.tensor(df["label"].values, dtype=torch.long) if "label" in df.columns else None
    def __len__(self):
        return self.enc["input_ids"].size(0)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item

train_dataset = TextDataset(train_df, roberta_tokenizer)
val_dataset = TextDataset(val_df, roberta_tokenizer)
len(train_dataset), len(val_dataset)



# Model + Trainer with snapshot fallback
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback

num_labels = 2
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        roberta_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    )
except Exception as e:
    print(f"Model remote load failed for {roberta_name}: {e}\nFalling back to local snapshot...")
    local_dir = snapshot_download(repo_id=roberta_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        local_dir, num_labels=num_labels, id2label=id2label, label2id=label2id, local_files_only=True
    )

if hasattr(model, "gradient_checkpointing_enable"):
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

collator = DataCollatorWithPadding(tokenizer=roberta_tokenizer)

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.tensor(logits).softmax(dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs[:, 1]),
    }

args = TrainingArguments(
    output_dir=str("bert_roberta_output"),
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model='roc_auc',
    greater_is_better=True,
    report_to=["none"],
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=roberta_tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0)],
)

print("Trainer ready.")



# Train and evaluate
train_result = trainer.train()
val_metrics = trainer.evaluate()
val_metrics



# Inference on test and write submission
from tqdm.auto import tqdm
model.eval()

enc = roberta_tokenizer(
    list(test_df["text"].values),
    padding="max_length",
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)

probs_list = []
with torch.no_grad():
    for i in tqdm(range(0, enc["input_ids"].size(0), 256)):
        batch = {k: v[i:i+256].to(DEVICE) for k, v in enc.items()}
        outputs = model(**batch)
        p = torch.softmax(outputs.logits, dim=1)[:, 1]
        probs_list.append(p.cpu())

probs = torch.cat(probs_list).numpy()

submission = pd.DataFrame({
    "Id": test_df["Id"].values,
    "TARGET": probs,
})
sub_path = BASE_DIR / "submission.csv"
submission.to_csv("submission.csv", index=False)
print(f"Wrote {sub_path}")


