# === Setup: installs, imports, seed ===
!pip -q install -U transformers datasets evaluate scikit-learn torchmetrics timm
!pip install --upgrade protobuf==3.20.3

import os, random, zipfile
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from transformers.modeling_outputs import SequenceClassifierOutput

try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

INPUT_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"
WORK_DIR = "/kaggle/working"
os.makedirs(WORK_DIR, exist_ok=True)

for zname in ["train.csv.zip", "test.csv.zip", "sample_submission.csv.zip"]:
    zpath = os.path.join(INPUT_DIR, zname)
    if os.path.exists(zpath):
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(WORK_DIR)

for f in ["train.csv", "test.csv", "sample_submission.csv"]:
    fp = os.path.join(WORK_DIR, f)
    assert os.path.exists(fp), f"Missing extracted file: {fp}"
print("Data ready:", [os.path.join(WORK_DIR, f) for f in ["train.csv","test.csv","sample_submission.csv"]])

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train_df = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(WORK_DIR, "test.csv"))
train_df["comment_text"] = train_df["comment_text"].fillna("")
test_df["comment_text"]  = test_df["comment_text"].fillna("")

train_df["label_sum"] = train_df[LABELS].sum(axis=1)
train_df, valid_df = train_test_split(
    train_df, test_size=0.1, random_state=SEED, stratify=train_df["label_sum"]
)
train_df = train_df.drop(columns=["label_sum"])
valid_df = valid_df.drop(columns=["label_sum"])

MODEL_NAME = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

pos_freq = train_df[LABELS].mean().values
pos_freq = np.clip(pos_freq, 1e-6, 1.0)
pos_weight = torch.tensor((1.0 - pos_freq) / pos_freq, dtype=torch.float)

MAX_LEN = 320

class JigsawDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=MAX_LEN, is_train=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row["comment_text"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if self.is_train:
            item["labels"] = torch.tensor(row[LABELS].values.astype(np.float32))
        return item

train_ds = JigsawDataset(train_df, tokenizer, is_train=True)
valid_ds = JigsawDataset(valid_df, tokenizer, is_train=True)
test_ds  = JigsawDataset(test_df,  tokenizer, is_train=False)

class MultiLabelModel(nn.Module):
    def __init__(self, model_name, num_labels=6, pos_weight=None):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.ones(num_labels))

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        logits = outputs.logits
        loss = None
        if labels is not None:
            bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            loss = bce(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiLabelModel(MODEL_NAME, num_labels=len(LABELS), pos_weight=pos_weight.to(device))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    roc_list = []
    for i in range(len(LABELS)):
        try:
            roc_list.append(roc_auc_score(labels[:, i], probs[:, i]))
        except ValueError:
            pass
    roc_auc = float(np.mean(roc_list)) if roc_list else 0.0
    preds_bin = (probs >= 0.5).astype(int)
    accs = [accuracy_score(labels[:, i], preds_bin[:, i]) for i in range(len(LABELS))]
    return {"roc_auc": roc_auc, "accuracy_t0.5": float(np.mean(accs))}

EPOCHS = 2
BS = 16
LR = 2e-5

eval_key = "evaluation_strategy"
try:
    TrainingArguments(output_dir="tmp", eval_strategy="steps")
    eval_key = "eval_strategy"
except TypeError:
    eval_key = "evaluation_strategy"

ta_kwargs = {
    "output_dir": os.path.join(WORK_DIR, "outputs"),
    "per_device_train_batch_size": BS,
    "per_device_eval_batch_size": BS,
    "num_train_epochs": EPOCHS,
    "learning_rate": LR,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "logging_steps": 100,
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "roc_auc",
    "greater_is_better": True,
    "fp16": False,
    "report_to": [],
    "seed": SEED
}
ta_kwargs[eval_key] = "epoch"

training_args = TrainingArguments(**ta_kwargs)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

train_result = trainer.train()
eval_result = trainer.evaluate()
print("Validation metrics:", eval_result)

pred = trainer.predict(test_ds, metric_key_prefix="predict")
pred_logits = pred.predictions
pred_probs = 1 / (1 + np.exp(-pred_logits))

sub = pd.DataFrame(pred_probs, columns=LABELS)
sub.insert(0, "id", test_df["id"])
sub_path = os.path.join(WORK_DIR, "submission.csv")
sub.to_csv(sub_path, index=False)
print("Saved submission to:", sub_path)

save_dir = os.path.join(WORK_DIR, "model_roberta_large_final")
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
print("Saved model to:", save_dir)


