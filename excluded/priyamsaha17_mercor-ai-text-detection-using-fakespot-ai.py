# full script: accuracy-focused, 20 epochs, tqdm epoch progress, improved head + MC-dropout
import os, random, math, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    logging as transformers_logging
)
from datasets import Dataset

# -------------------- Settings --------------------
DATA_DIR = "/kaggle/input/mercor-ai-detection"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")
SEED = 42

MODEL_NAME = "fakespot-ai/roberta-base-ai-text-detection-v1"
MAX_LEN = 512
PER_DEVICE_TRAIN_BATCH = 32
PER_DEVICE_EVAL_BATCH  = 4
GRADIENT_ACCUMULATION_STEPS = 1
NUM_EPOCHS = 20                # changed to 20
LEARNING_RATE = 2e-5
FP16 = False
OUTPUT_DIR = "./finetuned_roberta_improved_ann"
PLOT_DIR = "./plots"
os.makedirs(PLOT_DIR, exist_ok=True)

ANN_INNER_DIM = 1024
ANN_DROPOUT = 0.3
USE_MEAN_POOLING = False

FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.0

# seed
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
seed_everything()

transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- Helpers --------------------
def clean_text_field(s):
    if pd.isna(s):
        return ""
    return " ".join(str(s).strip().replace("\r", " ").replace("\n", " ").split())

def make_input_string(topic, answer):
    t = clean_text_field(topic)
    a = clean_text_field(answer)
    return f"TOPIC: {t}\n\nANSWER: {a}"

def mean_pooling(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

# -------------------- Load data --------------------
df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

required = {"topic", "answer", "is_cheating"}
if not required.issubset(df.columns):
    raise ValueError(f"train.csv must contain columns: {required}")
if "id" not in test_df.columns:
    raise ValueError("test.csv must contain column 'id'")

df["text"] = df.apply(lambda r: make_input_string(r["topic"], r["answer"]), axis=1)
df["label"] = df["is_cheating"].astype(int)
test_df["text"] = test_df.apply(lambda r: make_input_string(r.get("topic",""), r.get("answer","")), axis=1)

# upsample minority class 0
counts = df["label"].value_counts().to_dict()
print("Original label counts:", counts)
target_label = 0
majority_count = df["label"].value_counts().max()
minor_count = df[df["label"] == target_label].shape[0]
if minor_count == 0:
    raise ValueError(f"No samples with label {target_label}, cannot upsample.")
if minor_count < majority_count:
    needed = majority_count - minor_count
    minority_df = df[df["label"] == target_label]
    extra = minority_df.sample(n=needed, replace=True, random_state=SEED)
    df_ups = pd.concat([df, extra], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"Upsampled label={target_label}: +{needed} samples -> new counts {df_ups['label'].value_counts().to_dict()}")
else:
    df_ups = df.copy()
    print("No upsampling needed.")

# train/val split
train_df, val_df = train_test_split(
    df_ups[["text", "label"]],
    test_size=0.20,
    random_state=SEED,
    stratify=df_ups["label"]
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
print("Train label counts:", train_df["label"].value_counts().to_dict())

train_ds = Dataset.from_pandas(train_df)
val_ds   = Dataset.from_pandas(val_df)
test_ds  = Dataset.from_pandas(test_df[["id","text"]].rename(columns={"id":"orig_id"}))

# -------------------- Tokenizer + base model --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
config = AutoConfig.from_pretrained(MODEL_NAME)
config.num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

# improved head
hidden_size = model.config.hidden_size

class ImprovedANNHeadV2(nn.Module):
    def __init__(self, hidden_size, inner_dim=ANN_INNER_DIM, dropout_prob=ANN_DROPOUT, use_batchnorm=True, hard_threshold=None):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.hard_threshold = hard_threshold

        self.proj1 = weight_norm(nn.Linear(hidden_size, inner_dim))
        self.bn1 = nn.BatchNorm1d(inner_dim) if use_batchnorm else nn.Identity()
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_prob)

        self.proj2 = weight_norm(nn.Linear(inner_dim, inner_dim // 2))
        self.bn2 = nn.BatchNorm1d(inner_dim // 2) if use_batchnorm else nn.Identity()
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout_prob)

        self.res_proj = weight_norm(nn.Linear(hidden_size, inner_dim // 2))
        self.gate = nn.Linear(inner_dim // 2, inner_dim // 2)
        self.out = weight_norm(nn.Linear(inner_dim // 2, 2))

        nn.init.constant_(self.out.bias, 0.0)

    def forward(self, features, attention_mask=None, mc_dropout=False):
        if features.dim() == 3:
            if USE_MEAN_POOLING and attention_mask is not None:
                pooled = mean_pooling(features, attention_mask)
            else:
                pooled = features[:, 0, :]
        else:
            pooled = features

        x = self.proj1(pooled)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x) if (self.training or mc_dropout) else x

        x = self.proj2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x) if (self.training or mc_dropout) else x

        res = self.res_proj(pooled)
        gate = torch.sigmoid(self.gate(res))
        fused = x * (1.0 - gate) + res * gate

        logits = self.out(fused)

        if (not self.training) and (self.hard_threshold is not None):
            probs = F.softmax(logits, dim=1)[:, 1]
            eps_pos = 1.0 - 1e-6
            eps_neg = 1e-6
            hard_mask = (probs >= float(self.hard_threshold)).to(dtype=probs.dtype)
            snapped = hard_mask * eps_pos + (1.0 - hard_mask) * eps_neg
            single_logit = torch.log(snapped / (1.0 - snapped))
            logits = torch.stack([-single_logit, single_logit], dim=1)

        return logits

model.classifier = ImprovedANNHeadV2(hidden_size, inner_dim=ANN_INNER_DIM, dropout_prob=ANN_DROPOUT, use_batchnorm=True)

# defensive
try:
    model.config.use_cache = False
    if hasattr(model, "model") and hasattr(model.model, "config"):
        model.model.config.use_cache = False
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------- Tokenize --------------------
def tokenize_batch(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

print("Tokenizing train dataset...")
train_ds = train_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
print("Tokenizing val dataset...")
val_ds = val_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
print("Tokenizing test dataset...")
test_ds = test_ds.map(lambda b: tokenizer(b["text"], padding="max_length", truncation=True, max_length=MAX_LEN), batched=True, remove_columns=["text"])

train_ds = train_ds.rename_column("label", "labels")
val_ds   = val_ds.rename_column("label", "labels")

train_ds.set_format(type="torch")
val_ds.set_format(type="torch")
test_ds.set_format(type="torch")

data_collator = DataCollatorWithPadding(tokenizer)

# -------------------- Trainer subclass with focal loss --------------------
class MyTrainer(Trainer):
    def __init__(self, focal_gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        forward_inputs = {k:v for k,v in inputs.items() if k != "labels"}
        outputs = model(**forward_inputs)
        logits = outputs.get("logits")

        if self.label_smoothing and self.label_smoothing > 0.0:
            n_classes = logits.size(-1)
            with torch.no_grad():
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(self.label_smoothing / (n_classes - 1))
                true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)
            log_prob = F.log_softmax(logits, dim=1)
            ce_loss = -(true_dist * log_prob).sum(dim=1).mean()
        else:
            ce_loss = F.cross_entropy(logits, labels)

        if self.focal_gamma and self.focal_gamma > 0.0:
            probs = F.softmax(logits, dim=1)
            pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
            focal_term = (1 - pt) ** self.focal_gamma
            focal_loss = -focal_term * torch.log(pt)
            focal_loss = focal_loss.mean()
            loss = focal_loss
        else:
            loss = ce_loss

        return (loss, outputs) if return_outputs else loss

# -------------------- Metrics: focus on accuracy --------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    # still return roc_auc if desired (guarded)
    try:
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:,1]
        roc = float(roc_auc_score(labels, probs))
    except Exception:
        roc = 0.5
    return {"accuracy": float(acc), "roc_auc": roc}

# -------------------- Tqdm callback for per-epoch display --------------------
class TqdmEpochCallback(TrainerCallback):
    def __init__(self):
        self.bar = None
        self.last_epoch = 0

    def on_train_begin(self, args, state, control, **kwargs):
        total = int(args.num_train_epochs) if args.num_train_epochs is not None else None
        self.bar = tqdm(total=total, desc="Training", unit="epoch")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # update on evaluation logs that include eval_accuracy
        if "eval_accuracy" in logs:
            epoch_val = logs.get("epoch", state.epoch or 0.0)
            try:
                epoch_int = int(math.floor(epoch_val))
            except Exception:
                epoch_int = int(epoch_val) if epoch_val is not None else self.last_epoch + 1
            # avoid double increment of bar
            steps_to_inc = max(0, epoch_int - self.last_epoch)
            if steps_to_inc > 0:
                self.bar.update(steps_to_inc)
                self.last_epoch = epoch_int
            acc = logs.get("eval_accuracy")
            loss = logs.get("eval_loss", logs.get("loss", 0.0))
            self.bar.set_description(f"Epoch {epoch_int} Acc {acc:.4f} Loss {loss:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.bar:
            self.bar.close()

# -------------------- Training args: focus on accuracy --------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",   # focus on accuracy
    greater_is_better=True,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.05,
    fp16=FP16,
    logging_dir="./logs",
    logging_steps=50,
    seed=SEED,
    report_to=[],
    warmup_ratio=0.06,
    lr_scheduler_type="cosine",
)

early_stopping = EarlyStoppingCallback(early_stopping_patience=3)  # stop if accuracy doesn't improve
tqdm_cb = TqdmEpochCallback()

trainer = MyTrainer(
    model=model,
    args=training_args,
    focal_gamma=FOCAL_GAMMA,
    label_smoothing=LABEL_SMOOTHING,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping, tqdm_cb],
)

# -------------------- Train --------------------
print("Starting training (accuracy primary)...")
t0 = time.time()
train_result = trainer.train()
t1 = time.time()
duration_min = (t1 - t0) / 60.0
result_dict = getattr(train_result, "__dict__", None) or str(train_result)
print(f"Training done in {duration_min:.2f} minutes. Train result: {result_dict}")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# -------------------- Evaluate & Visualize --------------------
eval_result = trainer.evaluate(eval_dataset=val_ds)
print("Evaluation result:", eval_result)

preds_out = trainer.predict(val_ds)
logits = preds_out.predictions
probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:,1]
labels = preds_out.label_ids
fpr, tpr, _ = roc_curve(labels, probs)
roc_auc_val = auc(fpr, tpr)
print(f"Validation ROC-AUC: {roc_auc_val:.4f}")

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_val:.4f})')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Validation ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, "val_roc_curve.png"))
plt.show()

precision, recall, _ = precision_recall_curve(labels, probs)
plt.figure(figsize=(7,6))
plt.plot(recall, precision, label="PR curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Validation Precision-Recall Curve")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, "val_pr_curve.png"))
plt.show()

preds_bin = (np.argmax(logits, axis=1)).astype(int)
cm = confusion_matrix(labels, preds_bin)
disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
disp.plot(cmap="Blues")
plt.title("Validation Confusion Matrix")
plt.savefig(os.path.join(PLOT_DIR, "val_confusion_matrix.png"))
plt.show()

# save val preds
val_out = val_df.copy()
val_out["pred_prob_ai"] = probs
val_out["pred_label"] = preds_bin
val_out.to_csv("val_predictions_with_probs.csv", index=False)
print("Saved val_predictions_with_probs.csv")

# -------------------- MC Dropout predict helper --------------------
def mc_dropout_predict(trainer, dataset, n_samples=12):
    model = trainer.model
    model.train()  # enable dropout
    device = next(model.parameters()).device
    loader = trainer.get_eval_dataloader(dataset)
    collected = []
    for _ in range(n_samples):
        probs_i = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs.logits
                p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                probs_i.append(p)
        probs_i = np.concatenate(probs_i, axis=0)
        collected.append(probs_i)
    collected = np.stack(collected, axis=0)
    mean_probs = collected.mean(axis=0)
    std_probs = collected.std(axis=0)
    model.eval()
    return mean_probs, std_probs

# -------------------- Predict on test set --------------------
print("Predicting on test set using MC-dropout ensemble...")
THRESHOLD = 0.5

if hasattr(model, "classifier"):
    model.classifier.hard_threshold = None

mean_probs, std_probs = mc_dropout_predict(trainer, test_ds, n_samples=12)
test_classes = (mean_probs >= THRESHOLD).astype(int)

ids = test_df["id"].tolist()
if len(ids) != len(test_classes):
    if "orig_id" in test_ds.column_names:
        ids = list(test_ds["orig_id"])
    else:
        ids = list(range(len(test_classes)))

submission_df = pd.DataFrame({
    "id": ids,
    "is_cheating": test_classes,
    "prob_ai": mean_probs,
    "prob_std": std_probs
})
submission_df.to_csv("submission_mc_dropout.csv", index=False)
print(f"Saved submission_mc_dropout.csv with {len(submission_df)} rows (mc_samples=12)")

# optional head-snapped submission
if hasattr(model, "classifier"):
    model.classifier.hard_threshold = THRESHOLD
    head_preds = trainer.predict(test_ds)
    head_logits = head_preds.predictions
    head_classes = np.argmax(head_logits, axis=1).astype(int)
    submission_snapped = pd.DataFrame({"id": ids, "is_cheating": head_classes})
    submission_snapped.to_csv("submission_head_snapped.csv", index=False)
    print("Saved submission_head_snapped.csv")

print("Done. Plots saved in", PLOT_DIR)

