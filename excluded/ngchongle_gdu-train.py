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


# Optional: install dependencies in fresh environments
# %pip install -q transformers datasets scikit-learn torch accelerate
!pip install -qU transformers


from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


def set_environment(cache_dir: Path, seed: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def read_data(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if "label" not in train_df.columns:
        raise ValueError("Expected a `label` column in train.csv.")
    return train_df, test_df


def build_label_maps(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    ordered_labels = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(ordered_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def make_dataset(texts: List[str], labels: List[int] | None = None) -> Dataset:
    data_dict: Dict[str, List] = {"text": texts}
    if labels is not None:
        data_dict["labels"] = labels
    return Dataset.from_dict(data_dict)


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int, with_labels: bool) -> Dataset:
    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(_tokenize, batched=True)
    columns = ["input_ids", "attention_mask"]
    if with_labels:
        columns.append("labels")
    tokenized.set_format(type="torch", columns=columns)
    return tokenized


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def compute_metrics(prediction_output, positive_index: int) -> Dict[str, float]:
    logits = prediction_output.predictions
    labels = prediction_output.label_ids
    if labels is None:
        raise ValueError("Labels are required to compute metrics.")
    probs = softmax(logits)
    preds = np.argmax(logits, axis=1)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }
    if probs.shape[1] >= 2:
        metrics["roc_auc"] = roc_auc_score(labels, probs[:, positive_index])
    return metrics


@dataclass
class Config:
    model_name: str = "microsoft/deberta-v3-large"
    num_epochs: int = 16
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_length: int = 256
    seed: int = 42
    cache_dir: Path = Path("hf-cache")
    output_dir: Path = Path("out-deberta-full")
    train_path: Path = Path("/kaggle/input/rmit-hackathon-2025/train.csv")
    test_path: Path = Path("/kaggle/input/rmit-hackathon-2025/test.csv")
    sample_submission_path: Path = Path("/kaggle/input/rmit-hackathon-2025/sample_submission.csv")
    submission_path: Path = Path("submission.csv")
    positive_label: str = "jailbreak"
    device_preference: str = "auto"  # "auto" uses CUDA when available


CFG = Config()
set_environment(CFG.cache_dir, CFG.seed)

device = torch.device("cuda") if CFG.device_preference == "auto" and torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")


train_df, test_df = read_data(CFG.train_path, CFG.test_path)
label2id, id2label = build_label_maps(train_df["label"].tolist())
train_df = train_df.assign(label_id=train_df["label"].map(label2id))

positive_label = CFG.positive_label if CFG.positive_label in label2id else sorted(label2id.keys())[-1]
positive_index = label2id[positive_label]
print(f"Positive class: {positive_label} (index {positive_index})")

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)

train_dataset = make_dataset(
    texts=train_df["text"].tolist(),
    labels=train_df["label_id"].tolist(),
)
test_dataset = make_dataset(texts=test_df["text"].tolist())

tokenized_train = tokenize_dataset(train_dataset, tokenizer, CFG.max_length, with_labels=True)
tokenized_test = tokenize_dataset(test_dataset, tokenizer, CFG.max_length, with_labels=False)
print(tokenized_train)


model = AutoModelForSequenceClassification.from_pretrained(
    CFG.model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)
model.to(device)

training_output_dir = CFG.output_dir / "full_training"
training_output_dir.mkdir(parents=True, exist_ok=True)
training_args = TrainingArguments(
    output_dir=str(training_output_dir),
    num_train_epochs=CFG.num_epochs,
    per_device_train_batch_size=CFG.batch_size,
    per_device_eval_batch_size=CFG.batch_size,
    learning_rate=CFG.learning_rate,
    weight_decay=CFG.weight_decay,
    eval_strategy="no",
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="no",
    seed=CFG.seed,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
)

trainer.train()
print("Finished full-data training.")

checkpoint_dir = training_output_dir / "model_final"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(checkpoint_dir))
tokenizer.save_pretrained(str(checkpoint_dir))
print(f"Saved final model checkpoint to {checkpoint_dir}")


train_predictions = trainer.predict(tokenized_train)
train_metrics = compute_metrics(train_predictions, positive_index)

print("Training-set metrics:")
for metric_name, metric_value in train_metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")


print("Generating test predictions...")
test_preds = trainer.predict(tokenized_test)
test_probs = softmax(test_preds.predictions)

sample_submission = pd.read_csv(CFG.sample_submission_path)
if "TARGET" not in sample_submission.columns:
    raise ValueError("Expected a `TARGET` column in sample_submission.csv.")

sample_submission["TARGET"] = test_probs[:, positive_index]
submission = sample_submission

# BASE_DIR = Path.cwd()
submission.to_csv('submissionGDU.csv', index=False)
submission.head()




