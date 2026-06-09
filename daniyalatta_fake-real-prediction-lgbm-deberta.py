!pip install evaluate


#!/usr/bin/env python3
# deberta_ensemble_5heads.py
"""
Ensemble of 5 DeBERTa models with different pooling heads:
1. Mean Pooling
2. Max Pooling  
3. CLS Token
4. Attention Pooling
5. Concatenated Pooling (CLS + Mean + Max)

Combines predictions using:
1. Simple averaging
2. Majority voting
3. Optuna-optimized weights
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import logging
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import optuna
from typing import List, Dict, Tuple

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset
import evaluate




# ---------------- CONFIG ----------------
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LEN = 512
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 3
N_FOLDS = 5
OUTPUT_DIR = "hf_run_deberta_ensemble"
DATA_DIR = "/kaggle/input/fake-or-real-the-impostor-hunt/data"
SEED = 42
BASE_LR = 2e-5
WARMUP_RATIO = 0.06
LR_SCHEDULER_TYPE = "cosine"
WEIGHT_DECAY = 0.01
DATALOADER_NUM_WORKERS = 4
LABEL_SMOOTHING = 0.05

# Model types
MODEL_TYPES = ["mean_pool", "max_pool", "cls_token", "attention_pool", "concat_pool"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# ---------------- Data helpers (same as before) ----------------
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def normalize_folder_name(raw_id: str) -> str:
    raw = str(raw_id)
    if raw.startswith("article_"):
        return raw
    if raw.isdigit():
        return f"article_{int(raw):04d}"
    return raw

def numeric_id_from_folder(aid: str) -> str:
    if aid.startswith("article_"):
        return aid.replace("article_", "")
    return aid

def load_train_pairs(data_dir: str) -> pd.DataFrame:
    train_csv = Path(data_dir) / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found at {train_csv}")
    df_gt = pd.read_csv(train_csv)
    rows = []
    for _, row in df_gt.iterrows():
        raw_id = str(row["id"])
        folder_name = normalize_folder_name(raw_id)
        folder = Path(data_dir) / "train" / folder_name
        if not folder.exists() or not folder.is_dir():
            logger.warning("Missing train folder for id '%s' -> expected %s", raw_id, folder)
            continue
        real_idx = int(row["real_text_id"])
        for idx in (1, 2):
            file_path = folder / f"file_{idx}.txt"
            if not file_path.exists():
                logger.warning("Missing file: %s", file_path)
                continue
            text = read_text(file_path)
            label = 1 if idx == real_idx else 0
            rows.append({"id": folder_name, "file_idx": idx, "text": text, "label": label})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No training examples found. Check data layout.")
    return df

def load_test_pairs(data_dir: str) -> pd.DataFrame:
    test_folder = Path(data_dir) / "test"
    rows = []
    if not test_folder.exists():
        logger.warning("Test folder not found: %s", test_folder)
        return pd.DataFrame(rows)
    for aid_folder in sorted(test_folder.iterdir()):
        name = aid_folder.name
        if name.startswith(".") or not aid_folder.is_dir():
            continue
        folder_name = name
        if name.isdigit():
            folder_name = f"article_{int(name):04d}"
        for idx in (1, 2):
            file_path = aid_folder / f"file_{idx}.txt"
            if not file_path.exists():
                logger.warning("Missing file: %s", file_path)
                continue
            text = read_text(file_path)
            rows.append({"id": folder_name, "file_idx": idx, "text": text})
    return pd.DataFrame(rows)

def prepare_hf_dataset(df: pd.DataFrame, tokenizer, max_len=512, is_train=True):
    ds = Dataset.from_pandas(df.reset_index(drop=True))
    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=max_len)
    ds = ds.map(tokenize, batched=False, load_from_cache_file=True)
    if is_train and "label" in ds.column_names:
        ds = ds.rename_column("label", "labels")
    cols = ["input_ids", "attention_mask"]
    if "token_type_ids" in ds.column_names:
        cols.append("token_type_ids")
    if is_train:
        cols.append("labels")
    ds.set_format(type="torch", columns=cols)
    return ds

def compute_balanced_weights(labels, n_classes=2):
    labels = np.array(labels, dtype=int)
    counts = np.bincount(labels, minlength=n_classes)
    total = labels.size
    weights = []
    for c in range(n_classes):
        if counts[c] == 0:
            weights.append(0.0)
        else:
            weights.append(float(total / (n_classes * counts[c])))
    return weights




# ---------------- In-memory best callback ----------------
class InMemoryBestCallback(TrainerCallback):
    def __init__(self, metric_name="eval_accuracy"):
        self.metric_name = metric_name
        self.best = -float("inf")
        self.best_state = None
        self.best_step = None

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics", None)
        trainer = kwargs.get("trainer", None)
        if metrics is None:
            return
        val = None
        for key in (self.metric_name, "accuracy", "eval_accuracy", "eval_acc"):
            if key in metrics:
                val = metrics[key]
                break
        if val is None:
            return
        model = kwargs.get("model", None) or (trainer.model if trainer is not None else None)
        if model is None:
            return
        if float(val) > self.best:
            self.best = float(val)
            self.best_step = int(state.global_step) if hasattr(state, "global_step") else None
            sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.best_state = sd
            logger.info(f"[InMemoryBest] new best {self.best:.6f} at step {self.best_step}")




# ---------------- Weighted Trainer ----------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, label_smoothing=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = float(label_smoothing)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs_cp = inputs.copy()
        labels = inputs_cp.pop("labels", None)
        if labels is None:
            labels = inputs_cp.pop("label", None)

        outputs = model(**inputs_cp)
        logits = None
        if isinstance(outputs, SequenceClassifierOutput):
            logits = outputs.logits
        elif isinstance(outputs, dict):
            logits = outputs.get("logits", None)
        else:
            logits = getattr(outputs, "logits", None)

        if labels is None:
            loss = outputs.loss if hasattr(outputs, "loss") else None
        else:
            device = logits.device
            if self.class_weights is not None and len(self.class_weights) == model.config.num_labels:
                w = torch.tensor(self.class_weights, dtype=torch.float32, device=device)
            else:
                w = None
            try:
                loss_fct = nn.CrossEntropyLoss(weight=w, label_smoothing=self.label_smoothing)
                loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            except TypeError:
                if self.label_smoothing and self.label_smoothing > 0.0:
                    n_classes = logits.size(-1)
                    with torch.no_grad():
                        smooth = self.label_smoothing
                        off_value = smooth / (n_classes - 1)
                        on_value = 1.0 - smooth
                        labels_onehot = torch.full_like(logits, off_value).scatter_(1, labels.unsqueeze(1), on_value)
                    log_prob = F.log_softmax(logits, dim=-1)
                    loss = -(labels_onehot * log_prob).sum(dim=-1).mean()
                else:
                    loss_fct = nn.CrossEntropyLoss(weight=w)
                    loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss




# ---------------- 5 Different Classifier Heads ----------------

class MeanPoolClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_labels: int = 2, dropout: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_name, num_labels=num_labels)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name, config=self.config)
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.label_smoothing = float(label_smoothing)

    def mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-9)
        return summed / lengths

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, return_dict=True, **kwargs):
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        last_hidden = enc_out.last_hidden_state
        pooled = self.mean_pool(last_hidden, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            try:
                loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            except TypeError:
                if self.label_smoothing and self.label_smoothing > 0.0:
                    n_classes = logits.size(-1)
                    with torch.no_grad():
                        smooth = self.label_smoothing
                        off_value = smooth / (n_classes - 1)
                        on_value = 1.0 - smooth
                        labels_onehot = torch.full_like(logits, off_value).scatter_(1, labels.unsqueeze(1), on_value)
                    log_prob = F.log_softmax(logits, dim=-1)
                    loss = -(labels_onehot * log_prob).sum(dim=-1).mean()
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if return_dict:
            return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=enc_out.hidden_states if hasattr(enc_out, "hidden_states") else None, attentions=enc_out.attentions if hasattr(enc_out, "attentions") else None)
        output = (logits,) + enc_out.to_tuple()[1:]
        return ((loss,) + output) if loss is not None else output

class MaxPoolClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_labels: int = 2, dropout: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_name, num_labels=num_labels)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name, config=self.config)
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.label_smoothing = float(label_smoothing)

    def max_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        masked_hidden = last_hidden_state.masked_fill(mask == 0, -float('inf'))
        pooled = torch.max(masked_hidden, dim=1)[0]
        return pooled

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, return_dict=True, **kwargs):
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        last_hidden = enc_out.last_hidden_state
        pooled = self.max_pool(last_hidden, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            try:
                loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            except TypeError:
                if self.label_smoothing and self.label_smoothing > 0.0:
                    n_classes = logits.size(-1)
                    with torch.no_grad():
                        smooth = self.label_smoothing
                        off_value = smooth / (n_classes - 1)
                        on_value = 1.0 - smooth
                        labels_onehot = torch.full_like(logits, off_value).scatter_(1, labels.unsqueeze(1), on_value)
                    log_prob = F.log_softmax(logits, dim=-1)
                    loss = -(labels_onehot * log_prob).sum(dim=-1).mean()
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if return_dict:
            return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=enc_out.hidden_states if hasattr(enc_out, "hidden_states") else None, attentions=enc_out.attentions if hasattr(enc_out, "attentions") else None)
        output = (logits,) + enc_out.to_tuple()[1:]
        return ((loss,) + output) if loss is not None else output

class CLSTokenClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_labels: int = 2, dropout: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_name, num_labels=num_labels)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name, config=self.config)
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.label_smoothing = float(label_smoothing)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, return_dict=True, **kwargs):
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        last_hidden = enc_out.last_hidden_state
        # Use CLS token (first token)
        cls_output = last_hidden[:, 0, :]
        pooled = self.dropout(cls_output)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            try:
                loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            except TypeError:
                if self.label_smoothing and self.label_smoothing > 0.0:
                    n_classes = logits.size(-1)
                    with torch.no_grad():
                        smooth = self.label_smoothing
                        off_value = smooth / (n_classes - 1)
                        on_value = 1.0 - smooth
                        labels_onehot = torch.full_like(logits, off_value).scatter_(1, labels.unsqueeze(1), on_value)
                    log_prob = F.log_softmax(logits, dim=-1)
                    loss = -(labels_onehot * log_prob).sum(dim=-1).mean()
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if return_dict:
            return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=enc_out.hidden_states if hasattr(enc_out, "hidden_states") else None, attentions=enc_out.attentions if hasattr(enc_out, "attentions") else None)
        output = (logits,) + enc_out.to_tuple()[1:]
        return ((loss,) + output) if loss is not None else output

class AttentionPoolClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_labels: int = 2, dropout: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_name, num_labels=num_labels)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name, config=self.config)
        hidden_size = self.config.hidden_size
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.label_smoothing = float(label_smoothing)

    def attention_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Compute attention weights
        attention_weights = self.attention(last_hidden_state).squeeze(-1)  # [batch, seq_len]
        
        # Mask out padding tokens
        mask = attention_mask.to(attention_weights.dtype)
        attention_weights = attention_weights.masked_fill(mask == 0, -float('inf'))
        
        # Apply softmax
        attention_weights = F.softmax(attention_weights, dim=1)  # [batch, seq_len]
        
        # Weighted sum
        pooled = torch.sum(last_hidden_state * attention_weights.unsqueeze(-1), dim=1)  # [batch, hidden_size]
        return pooled

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, return_dict=True, **kwargs):
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        last_hidden = enc_out.last_hidden_state
        pooled = self.attention_pool(last_hidden, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            try:
                loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            except TypeError:
                if self.label_smoothing and self.label_smoothing > 0.0:
                    n_classes = logits.size(-1)
                    with torch.no_grad():
                        smooth = self.label_smoothing
                        off_value = smooth / (n_classes - 1)
                        on_value = 1.0 - smooth
                        labels_onehot = torch.full_like(logits, off_value).scatter_(1, labels.unsqueeze(1), on_value)
                    log_prob = F.log_softmax(logits, dim=-1)
                    loss = -(labels_onehot * log_prob).sum(dim=-1).mean()
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if return_dict:
            return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=enc_out.hidden_states if hasattr(enc_out, "hidden_states") else None, attentions=enc_out.attentions if hasattr(enc_out, "attentions") else None)
        output = (logits,) + enc_out.to_tuple()[1:]
        return ((loss,) + output) if loss is not None else output

class ConcatPoolClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_labels: int = 2, dropout: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_name, num_labels=num_labels)
        self.encoder = AutoModel.from_pretrained(pretrained_model_name, config=self.config)
        hidden_size = self.config.hidden_size
        # Concatenate CLS + Mean + Max = 3 * hidden_size
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_size * 3, num_labels)
        self.label_smoothing = float(label_smoothing)

    def mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-9)
        return summed / lengths

    def max_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        masked_hidden = last_hidden_state.masked_fill(mask == 0, -float('inf'))
        pooled = torch.max(masked_hidden, dim=1)[0]
        return pooled

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, return_dict=True, **kwargs):
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        last_hidden = enc_out.last_hidden_state
        
        # Get all three representations
        cls_output = last_hidden[:, 0, :]  # CLS token
        mean_output = self.mean_pool(last_hidden, attention_mask)  # Mean pooling
        max_output = self.max_pool(last_hidden, attention_mask)  # Max pooling
        
        # Concatenate all three
        pooled = torch.cat([cls_output, mean_output, max_output], dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            try:
                loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            except TypeError:
                if self.label_smoothing and self.label_smoothing > 0.0:
                    n_classes = logits.size(-1)
                    with torch.no_grad():
                        smooth = self.label_smoothing
                        off_value = smooth / (n_classes - 1)
                        on_value = 1.0 - smooth
                        labels_onehot = torch.full_like(logits, off_value).scatter_(1, labels.unsqueeze(1), on_value)
                    log_prob = F.log_softmax(logits, dim=-1)
                    loss = -(labels_onehot * log_prob).sum(dim=-1).mean()
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if return_dict:
            return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=enc_out.hidden_states if hasattr(enc_out, "hidden_states") else None, attentions=enc_out.attentions if hasattr(enc_out, "attentions") else None)
        output = (logits,) + enc_out.to_tuple()[1:]
        return ((loss,) + output) if loss is not None else output






# ---------------- Model Factory ----------------
def create_model(model_type: str, pretrained_model_name: str, num_labels: int = 2, dropout: float = 0.1, label_smoothing: float = 0.0):
    if model_type == "mean_pool":
        return MeanPoolClassifier(pretrained_model_name, num_labels, dropout, label_smoothing)
    elif model_type == "max_pool":
        return MaxPoolClassifier(pretrained_model_name, num_labels, dropout, label_smoothing)
    elif model_type == "cls_token":
        return CLSTokenClassifier(pretrained_model_name, num_labels, dropout, label_smoothing)
    elif model_type == "attention_pool":
        return AttentionPoolClassifier(pretrained_model_name, num_labels, dropout, label_smoothing)
    elif model_type == "concat_pool":
        return ConcatPoolClassifier(pretrained_model_name, num_labels, dropout, label_smoothing)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------- Training function ----------------
def train_and_eval_fold(fold_id, model_type, train_df, val_df, test_df, tokenizer, class_weights=None):
    logger.info("Fold %d [%s]: train_files=%d val_files=%d", fold_id, model_type, len(train_df), len(val_df))
    train_ds = prepare_hf_dataset(train_df, tokenizer, max_len=MAX_LEN, is_train=True)
    val_ds = prepare_hf_dataset(val_df, tokenizer, max_len=MAX_LEN, is_train=True)

    model = create_model(model_type, MODEL_NAME, num_labels=2, dropout=0.1, label_smoothing=LABEL_SMOOTHING)
    try:
        if hasattr(model.encoder, "gradient_checkpointing_enable"):
            model.encoder.gradient_checkpointing_enable()
    except Exception:
        pass

    torch.backends.cudnn.benchmark = True

    effective_batch = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    num_update_steps_per_epoch = math.ceil(len(train_ds) / effective_batch)
    max_train_steps = num_update_steps_per_epoch * EPOCHS
    warmup_steps = int(max_train_steps * WARMUP_RATIO)

    has_cuda = torch.cuda.is_available()
    bf16_supported = False
    try:
        bf16_supported = torch.cuda.is_bf16_supported()
    except Exception:
        bf16_supported = False

    args = TrainingArguments(
        output_dir=str(Path(OUTPUT_DIR) / f"{model_type}_fold_{fold_id}"),
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",
        save_strategy="no",
        save_total_limit=0,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=SEED + fold_id,
        fp16=has_cuda and not bf16_supported,
        bf16=(has_cuda and bf16_supported),
        logging_strategy="epoch",
        learning_rate=BASE_LR,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_steps=warmup_steps,
        report_to="none",
        weight_decay=WEIGHT_DECAY,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=has_cuda,
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(pred):
        logits = pred.predictions
        if logits is None:
            return {"accuracy": None}
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim == 1 or logits.shape[-1] == 1:
            preds = logits.astype(int)
        else:
            preds = np.argmax(logits, axis=1)
        labels = pred.label_ids if hasattr(pred, "label_ids") else None
        if labels is None:
            return {"accuracy": None}
        return {"accuracy": float(accuracy_score(labels, preds))}

    inmem_cb = InMemoryBestCallback()

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        label_smoothing=LABEL_SMOOTHING,
        callbacks=[inmem_cb],
    )

    trainer.train()

    # Restore best state in-memory
    if inmem_cb.best_state is not None:
        logger.info(f"[Fold {fold_id} {model_type}] Restoring in-memory best model (acc={inmem_cb.best:.6f})")
        trainer.model.load_state_dict(inmem_cb.best_state, strict=False)
        trainer.model.to(trainer.args.device)
    else:
        logger.info(f"[Fold {fold_id} {model_type}] No in-memory best found; using final model weights")

    # Validation predictions
    val_hf = prepare_hf_dataset(val_df, tokenizer, max_len=MAX_LEN, is_train=False)
    out = trainer.predict(val_hf)
    logits = out.predictions
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if logits is None:
        raise RuntimeError(f"Fold {fold_id} {model_type}: no predictions for validation")
    if logits.ndim > 1 and logits.shape[1] == 2:
        probs_val = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    elif logits.ndim == 1:
        probs_val = 1.0 / (1.0 + np.exp(-logits))
    else:
        raise RuntimeError("Unexpected logits shape for val: %s" % str(logits.shape))

    oof_rows = []
    for i, prob in enumerate(probs_val):
        row = val_df.iloc[i]
        oof_rows.append({"id": row["id"], "file_idx": int(row["file_idx"]), "prob_real": float(prob), "label": int(row["label"])})

    # Test predictions
    test_probs = None
    if not test_df.empty:
        test_hf = prepare_hf_dataset(test_df, tokenizer, max_len=MAX_LEN, is_train=False)
        tout = trainer.predict(test_hf)
        tlogits = tout.predictions
        if isinstance(tlogits, (tuple, list)):
            tlogits = tlogits[0]
        if tlogits is None:
            raise RuntimeError(f"Fold {fold_id} {model_type}: no predictions for test")
        if tlogits.ndim > 1 and tlogits.shape[1] == 2:
            test_probs = torch.softmax(torch.tensor(tlogits), dim=1).numpy()[:, 1]
        elif tlogits.ndim == 1:
            test_probs = 1.0 / (1.0 + np.exp(-tlogits))
        else:
            raise RuntimeError("Unexpected logits shape for test: %s" % str(tlogits.shape))

    return pd.DataFrame(oof_rows), test_probs





# ---------------- Memory cleanup ----------------
def cleanup_memory():
    """Clean up GPU and CPU memory"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()



# ---------------- Ensemble methods ----------------
def calculate_per_article_accuracy(oof_df: pd.DataFrame) -> float:
    """Calculate per-article accuracy by choosing file with highest probability per article"""
    per_article_labels = []
    for aid, g in oof_df.groupby("id"):
        chosen = g.loc[g["prob_real"].idxmax()]
        per_article_labels.append(int(chosen["label"]))
    return np.mean(per_article_labels)

def ensemble_average(oof_dicts: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Simple averaging ensemble"""
    # Get base structure from first model
    first_key = list(oof_dicts.keys())[0]
    result_df = oof_dicts[first_key][["id", "file_idx", "label"]].copy()
    
    # Average probabilities across all models
    prob_sum = np.zeros(len(result_df))
    for model_type, df in oof_dicts.items():
        prob_sum += df["prob_real"].values
    
    result_df["prob_real"] = prob_sum / len(oof_dicts)
    return result_df

def ensemble_voting(oof_dicts: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Majority voting ensemble (each model votes 0/1, majority wins)"""
    first_key = list(oof_dicts.keys())[0]
    result_df = oof_dicts[first_key][["id", "file_idx", "label"]].copy()
    
    # Get votes from each model (0 or 1)
    votes = np.zeros((len(result_df), len(oof_dicts)))
    for i, (model_type, df) in enumerate(oof_dicts.items()):
        votes[:, i] = (df["prob_real"].values > 0.5).astype(int)
    
    # Majority vote
    majority_votes = (votes.sum(axis=1) > len(oof_dicts) / 2).astype(int)
    result_df["prob_real"] = majority_votes.astype(float)
    return result_df

def optuna_optimize_weights(oof_dicts: Dict[str, pd.DataFrame], n_trials: int = 100) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Optimize ensemble weights using Optuna"""
    model_names = list(oof_dicts.keys())
    
    def objective(trial):
        # Suggest weights that sum to 1
        weights = []
        for i, model_name in enumerate(model_names[:-1]):
            weight = trial.suggest_float(f"weight_{model_name}", 0.0, 1.0)
            weights.append(weight)
        
        # Last weight is constrained to make sum = 1
        last_weight = 1.0 - sum(weights)
        if last_weight < 0:
            return -1.0  # Invalid weights
        weights.append(last_weight)
        
        # Create weighted ensemble
        first_key = model_names[0]
        result_df = oof_dicts[first_key][["id", "file_idx", "label"]].copy()
        
        weighted_prob = np.zeros(len(result_df))
        for i, (model_name, df) in enumerate(oof_dicts.items()):
            weighted_prob += weights[i] * df["prob_real"].values
        
        result_df["prob_real"] = weighted_prob
        
        # Calculate per-article accuracy
        accuracy = calculate_per_article_accuracy(result_df)
        return accuracy
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best weights
    best_weights = {}
    for i, model_name in enumerate(model_names[:-1]):
        best_weights[model_name] = study.best_params[f"weight_{model_name}"]
    best_weights[model_names[-1]] = 1.0 - sum(best_weights.values())
    
    # Create best ensemble
    first_key = model_names[0]
    result_df = oof_dicts[first_key][["id", "file_idx", "label"]].copy()
    
    weighted_prob = np.zeros(len(result_df))
    for model_name, df in oof_dicts.items():
        weighted_prob += best_weights[model_name] * df["prob_real"].values
    
    result_df["prob_real"] = weighted_prob
    
    logger.info(f"Optuna best weights: {best_weights}")
    logger.info(f"Optuna best score: {study.best_value:.6f}")
    
    return best_weights, result_df

def create_test_ensemble(test_probs_dict: Dict[str, np.ndarray], weights: Dict[str, float], test_df: pd.DataFrame) -> pd.DataFrame:
    """Create test ensemble predictions using given weights"""
    result_df = test_df.copy()
    
    weighted_prob = np.zeros(len(result_df))
    for model_name, probs in test_probs_dict.items():
        weighted_prob += weights[model_name] * probs
    
    result_df["prob_real"] = weighted_prob
    return result_df




# ---------------- Main training loop ----------------
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_pairs = load_train_pairs(DATA_DIR)
    test_pairs = load_test_pairs(DATA_DIR)
    logger.info("Train files: %d | Test files: %d", len(train_pairs), len(test_pairs))

    # Build article-level ids and stratify labels
    train_csv = pd.read_csv(Path(DATA_DIR) / "train.csv")
    folder_ids = [normalize_folder_name(i) for i in train_csv["id"].astype(str).tolist()]
    strat_labels = train_csv["real_text_id"].astype(int).tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    cls_weights = compute_balanced_weights(train_pairs["label"].astype(int).tolist(), n_classes=2)
    logger.info("Computed class weights: %s", cls_weights)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    # Storage for all model predictions
    all_oof_preds = {model_type: [] for model_type in MODEL_TYPES}
    all_test_preds = {model_type: [] for model_type in MODEL_TYPES}

    # Train each model type
    for model_type in MODEL_TYPES:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_type.upper()} models")
        logger.info(f"{'='*50}")
        
        oof_list = []
        test_accum = []
        
        for fold_id, (tr_idx, va_idx) in enumerate(skf.split(folder_ids, strat_labels), start=1):
            train_folders = [folder_ids[i] for i in tr_idx]
            val_folders = [folder_ids[i] for i in va_idx]
            train_df = train_pairs[train_pairs["id"].isin(train_folders)].reset_index(drop=True)
            val_df = train_pairs[train_pairs["id"].isin(val_folders)].reset_index(drop=True)
            
            oof_fold, test_probs = train_and_eval_fold(
                fold_id, model_type, train_df, val_df, test_pairs, tokenizer, class_weights=cls_weights
            )
            oof_list.append(oof_fold)
            if test_probs is not None:
                test_accum.append(test_probs)

        # Combine OOF predictions
        oof_all = pd.concat(oof_list, ignore_index=True)
        all_oof_preds[model_type] = oof_all
        
        # Combine test predictions
        if len(test_accum) > 0:
            avg_test_probs = np.mean(np.vstack(test_accum), axis=0)
            all_test_preds[model_type] = avg_test_probs
        
        # Calculate individual model performance
        per_article_acc = calculate_per_article_accuracy(oof_all)
        logger.info(f"{model_type} per-article accuracy: {per_article_acc:.6f}")
        
        # Clean up memory after each model type
        cleanup_memory()
        logger.info(f"Memory cleanup completed for {model_type}")

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Save individual model OOF predictions
    for model_type, oof_df in all_oof_preds.items():
        oof_df.to_csv(Path(OUTPUT_DIR) / f"oof_{model_type}.csv", index=False)

    # Ensemble methods
    logger.info(f"\n{'='*50}")
    logger.info("ENSEMBLE RESULTS")
    logger.info(f"{'='*50}")
    
    # 1. Simple averaging
    ensemble_avg = ensemble_average(all_oof_preds)
    avg_accuracy = calculate_per_article_accuracy(ensemble_avg)
    logger.info(f"Simple Averaging Ensemble: {avg_accuracy:.6f}")
    ensemble_avg.to_csv(Path(OUTPUT_DIR) / "oof_ensemble_average.csv", index=False)
    
    # 2. Majority voting
    ensemble_vote = ensemble_voting(all_oof_preds)
    vote_accuracy = calculate_per_article_accuracy(ensemble_vote)
    logger.info(f"Majority Voting Ensemble: {vote_accuracy:.6f}")
    ensemble_vote.to_csv(Path(OUTPUT_DIR) / "oof_ensemble_voting.csv", index=False)
    
    # 3. Optuna optimized weights
    logger.info("Optimizing ensemble weights with Optuna...")
    best_weights, ensemble_optuna = optuna_optimize_weights(all_oof_preds, n_trials=200)
    optuna_accuracy = calculate_per_article_accuracy(ensemble_optuna)
    logger.info(f"Optuna Optimized Ensemble: {optuna_accuracy:.6f}")
    ensemble_optuna.to_csv(Path(OUTPUT_DIR) / "oof_ensemble_optuna.csv", index=False)
    
    # Generate test submissions for all ensemble methods
    if len(all_test_preds) == len(MODEL_TYPES) and all(len(probs) > 0 for probs in all_test_preds.values()):
        logger.info("\nGenerating test submissions...")
        
        # Average ensemble submission
        avg_weights = {model_type: 1.0/len(MODEL_TYPES) for model_type in MODEL_TYPES}
        test_avg = create_test_ensemble(all_test_preds, avg_weights, test_pairs)
        submission_avg = []
        for aid, g in test_avg.groupby("id"):
            chosen = g.loc[g["prob_real"].idxmax()]
            submission_avg.append({"id": numeric_id_from_folder(aid), "real_text_id": int(chosen["file_idx"])})
        sub_avg_df = pd.DataFrame(submission_avg)
        try:
            sub_avg_df["id_int"] = sub_avg_df["id"].astype(int)
            sub_avg_df = sub_avg_df.sort_values("id_int").drop(columns=["id_int"])
        except Exception:
            sub_avg_df = sub_avg_df.sort_values("id")
        sub_avg_df.to_csv(Path(OUTPUT_DIR) / "submission_average.csv", index=False)
        
        # Voting ensemble submission
        test_voting = test_pairs.copy()
        # Get votes from each model (0 or 1)
        votes = np.zeros((len(test_voting), len(MODEL_TYPES)))
        for i, (model_type, probs) in enumerate(all_test_preds.items()):
            votes[:, i] = (probs > 0.5).astype(int)
        # Majority vote
        majority_votes = (votes.sum(axis=1) > len(MODEL_TYPES) / 2).astype(int)
        test_voting["prob_real"] = majority_votes.astype(float)
        
        submission_voting = []
        for aid, g in test_voting.groupby("id"):
            chosen = g.loc[g["prob_real"].idxmax()]
            submission_voting.append({"id": numeric_id_from_folder(aid), "real_text_id": int(chosen["file_idx"])})
        sub_voting_df = pd.DataFrame(submission_voting)
        try:
            sub_voting_df["id_int"] = sub_voting_df["id"].astype(int)
            sub_voting_df = sub_voting_df.sort_values("id_int").drop(columns=["id_int"])
        except Exception:
            sub_voting_df = sub_voting_df.sort_values("id")
        sub_voting_df.to_csv(Path(OUTPUT_DIR) / "submission_voting.csv", index=False)
        
        # Optuna ensemble submission
        test_optuna = create_test_ensemble(all_test_preds, best_weights, test_pairs)
        submission_optuna = []
        for aid, g in test_optuna.groupby("id"):
            chosen = g.loc[g["prob_real"].idxmax()]
            submission_optuna.append({"id": numeric_id_from_folder(aid), "real_text_id": int(chosen["file_idx"])})
        sub_optuna_df = pd.DataFrame(submission_optuna)
        try:
            sub_optuna_df["id_int"] = sub_optuna_df["id"].astype(int)
            sub_optuna_df = sub_optuna_df.sort_values("id_int").drop(columns=["id_int"])
        except Exception:
            sub_optuna_df = sub_optuna_df.sort_values("id")
        sub_optuna_df.to_csv(Path(OUTPUT_DIR) / "submission_optuna.csv", index=False)
        
        logger.info("Saved submissions: submission_average.csv, submission_voting.csv, submission_optuna.csv")
    
    # Final summary
    logger.info(f"\n{'='*50}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*50}")
    for model_type in MODEL_TYPES:
        acc = calculate_per_article_accuracy(all_oof_preds[model_type])
        logger.info(f"{model_type:15s}: {acc:.6f}")
    logger.info(f"{'Simple Average':15s}: {avg_accuracy:.6f}")
    logger.info(f"{'Majority Vote':15s}: {vote_accuracy:.6f}")
    logger.info(f"{'Optuna Optimized':15s}: {optuna_accuracy:.6f}")




if __name__ == "__main__":
    main()




