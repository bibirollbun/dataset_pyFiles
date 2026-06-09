# ============================================================
# 1. Install
# ============================================================
!pip install -q transformers accelerate datasets

# ============================================================
# 2. Imports
# ============================================================
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from transformers import (
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ============================================================
# 3. Load data (.csv or .csv.zip)
# ============================================================
DATA_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"

def load_csv(fname: str) -> pd.DataFrame:
    if fname in os.listdir(DATA_DIR):
        return pd.read_csv(os.path.join(DATA_DIR, fname))
    if fname + ".zip" in os.listdir(DATA_DIR):
        return pd.read_csv(os.path.join(DATA_DIR, fname + ".zip"))
    raise FileNotFoundError(f"{fname}(.zip) not found in {DATA_DIR}")

train_df = load_csv("train.csv")
test_df  = load_csv("test.csv")

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train_df["comment_text"] = train_df["comment_text"].fillna("")
test_df["comment_text"]  = test_df["comment_text"].fillna("")

# Stratification helper
train_df["label_sum"] = train_df[label_cols].sum(axis=1)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=42,
    stratify=train_df["label_sum"]
)

train_df = train_df.drop(columns=["label_sum"])
val_df   = val_df.drop(columns=["label_sum"])

print("Train:", len(train_df), "| Val:", len(val_df))

# ============================================================
# 4. Tokenizer
# ============================================================
MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = 320   # longer sequences for more accuracy

# ============================================================
# 5. Dataset
# ============================================================
class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, label_cols=None, train=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_cols = label_cols
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["comment_text"]

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding=False,
        )

        item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}
        if self.train:
            labels = torch.tensor(row[self.label_cols].values.astype("float32"))
            item["labels"] = labels
        return item

train_ds = ToxicDataset(train_df, tokenizer, MAX_LEN, label_cols, train=True)
val_ds   = ToxicDataset(val_df, tokenizer, MAX_LEN, label_cols, train=True)
test_ds  = ToxicDataset(test_df, tokenizer, MAX_LEN, train=False)

collator = DataCollatorWithPadding(tokenizer)

# ============================================================
# 6. Class imbalance (pos_weight)
# ============================================================
pos_freq = train_df[label_cols].mean().values
pos_weight = torch.tensor((1 - pos_freq) / pos_freq, dtype=torch.float32).to(DEVICE)
print("pos_weight:", pos_weight)

# ============================================================
# 7. Custom Roberta model with multi-sample dropout
#    (Higher accuracy than plain head)
# ============================================================
class RobertaMultiSample(nn.Module):
    def __init__(self, model_name, num_labels=6):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.msd = nn.ModuleList([nn.Dropout(0.5) for _ in range(4)])
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token representation
        x = outputs.last_hidden_state[:, 0]  # [batch, hidden_size]

        logits = 0
        for dp in self.msd:
            logits = logits + self.classifier(dp(x))
        logits = logits / len(self.msd)

        if labels is not None:
            loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

model = RobertaMultiSample(MODEL_NAME, num_labels=len(label_cols)).to(DEVICE)

# Force everything on single GPU (important if Kaggle still gives more than one)
if torch.cuda.device_count() > 1:
    print("Multiple GPUs detected, but using cuda:0 only for stability.")
    model.to("cuda:0")

# ============================================================
# 8. Metrics (mean ROC-AUC)
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    aucs = []
    for i in range(len(label_cols)):
        try:
            aucs.append(roc_auc_score(labels[:, i], probs[:, i]))
        except Exception:
            pass
    return {"mean_roc_auc": float(np.mean(aucs))}

# ============================================================
# 9. TrainingArguments (compatible eval key)
# ============================================================
try:
    TrainingArguments(output_dir="./x", evaluation_strategy="epoch")
    eval_key = "evaluation_strategy"
except TypeError:
    eval_key = "eval_strategy"

ta_kwargs = {
    "output_dir": "./model_roberta_hq",
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 2,   # effective batch = 32
    "learning_rate": 1.5e-5,
    "num_train_epochs": 4,
    "warmup_ratio": 0.10,
    "weight_decay": 0.05,
    "fp16": True if DEVICE == "cuda" else False,
    "logging_steps": 250,
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "mean_roc_auc",
    "greater_is_better": True,
    "report_to": "none",
}
ta_kwargs[eval_key] = "epoch"

training_args = TrainingArguments(**ta_kwargs)

# ============================================================
# 10. Trainer (standard, loss is computed inside model)
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# ============================================================
# 11. Train
# ============================================================
trainer.train()
metrics = trainer.evaluate()
print("Validation metrics:", metrics)

# ============================================================
# 12. Predict on test + submission
# ============================================================
preds = trainer.predict(test_ds)
probs = 1 / (1 + np.exp(-preds.predictions))

submission = pd.DataFrame(probs, columns=label_cols)
submission.insert(0, "id", test_df["id"])
submission.to_csv("submission.csv", index=False)

print("submission.csv created!")
submission.head()


