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



# RoBERTa-base 5-fold CV for Toxic Comment Classification

# ------------------------- SECTION 0: RUNTIME GUARDS -------------------------
import os, sys, math, random, gc, inspect

os.environ["TRANSFORMERS_NO_TF"]   = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"]   = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

# ------------------------- SECTION 1: IMPORTS -------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

import torch
from torch import nn

try:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
    EarlyStoppingCallback, DataCollatorWithPadding
)

# ------------------------- SECTION 2: CONFIG -------------------------
# detect dataset folder automatically (should just be one)
INPUT_ROOT = "/kaggle/input"
data_folders = [
    d for d in os.listdir(INPUT_ROOT)
    if os.path.isdir(os.path.join(INPUT_ROOT, d))
]
print("Input folders:", data_folders)
assert len(data_folders) >= 1, "No input dataset folder found!"
DATA_DIR = os.path.join(INPUT_ROOT, data_folders[0])

MODEL_NAME = "roberta-base"
LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

MAX_LEN   = 192       # shorten for speed
EPOCHS    = 1         # can set to 2 for even better score
LR        = 2e-5
BS_TRAIN  = 16
BS_EVAL   = 64
WARMUP_RATIO     = 0.06
WEIGHT_DECAY     = 0.01
GRAD_ACCUM_STEPS = 2
N_FOLDS  = 5
USE_CV   = True
RNG_SEED = 42

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RNG_SEED)

os.makedirs("plots", exist_ok=True)

# ------------------------- SECTION 3: UTILITIES -------------------------
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\t", " ").replace("\r", " ")
    s = " ".join(s.split())
    return s

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick_path(name: str) -> str:
    """
    Returns either DATA_DIR/name.csv or DATA_DIR/name.csv.zip (whichever exists).
    """
    p1 = os.path.join(DATA_DIR, f"{name}.csv")
    p2 = os.path.join(DATA_DIR, f"{name}.csv.zip")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"Could not find {name}.csv or {name}.csv.zip in {DATA_DIR}")

# ------------------------- SECTION 4: LOAD DATA -------------------------
train_path  = pick_path("train")
test_path   = pick_path("test")
sample_path = pick_path("sample_submission")

print("train_path :", train_path)
print("test_path  :", test_path)
print("sample_path:", sample_path)

train_df  = pd.read_csv(train_path)
test_df   = pd.read_csv(test_path)
sample_df = pd.read_csv(sample_path)

train_df["text"] = train_df["comment_text"].map(clean_text)
test_df["text"]  = test_df["comment_text"].map(clean_text)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

# ------------------------- SECTION 5: TOKENIZER & QUICK EDA -------------------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)

# Label prevalence
counts = train_df[LABELS].sum().sort_values(ascending=False)
plt.figure(figsize=(6,4))
counts.plot(kind="bar")
plt.title("Label prevalence (train)")
plt.ylabel("positives")
plt.tight_layout()
plt.savefig("plots/label_prevalence.png")
plt.show()

# Token length distribution for first 20k examples
encoded = tok(
    train_df["text"].tolist()[:20000],
    truncation=True,
    max_length=MAX_LEN,
    padding=False,
)
lengths = [len(ids) for ids in encoded["input_ids"]]
plt.figure(figsize=(6,4))
plt.hist(lengths, bins=50)
plt.title("Token length histogram (first 20k)")
plt.xlabel("length (tokens)")
plt.ylabel("count")
plt.tight_layout()
plt.savefig("plots/token_length_hist.png")
plt.show()

# ------------------------- SECTION 6: HF DATASET HELPERS -------------------------
NUM_PROC = max(1, (os.cpu_count() or 2) // 2)

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=MAX_LEN)

def df_to_hfds(df: pd.DataFrame) -> Dataset:
    d = Dataset.from_pandas(df[["text"] + LABELS], preserve_index=False)
    d = d.map(tokenize, batched=True, remove_columns=["text"], num_proc=NUM_PROC)

    # rename labels to avoid confusion then pack into a single "labels" field
    rename_map = {lab: f"lab_{lab}" for lab in LABELS}
    d = d.rename_columns(rename_map)

    def pack(ex):
        ex["labels"] = [float(ex[f"lab_{lab}"]) for lab in LABELS]
        return ex

    d = d.map(pack, num_proc=NUM_PROC)
    d = d.remove_columns(list(rename_map.values()))
    return d

def test_to_hfds(df: pd.DataFrame) -> Dataset:
    d = Dataset.from_pandas(df[["text"]], preserve_index=False)
    d = d.map(tokenize, batched=True, remove_columns=["text"], num_proc=NUM_PROC)
    return d

# ------------------------- SECTION 7: CUSTOM TRAINER + METRICS -------------------------
class BCETrainer(Trainer):
    """
    Trainer that uses BCEWithLogitsLoss with optional pos_weight.
    Accepts **kwargs in compute_loss to be safe with HF internals
    (e.g. num_items_in_batch).
    """
    def __init__(self, pos_weight=None, **kwargs):
        super().__init__(**kwargs)
        self._pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss(
            pos_weight=self._pos_weight.to(logits.device) if self._pos_weight is not None else None
        )
        loss = loss_fct(logits, labels.to(logits.device))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = sigmoid(logits)
    per_label_auc = []

    for j, lab in enumerate(LABELS):
        y_true = labels[:, j]
        y_pred = probs[:, j]
        if len(np.unique(y_true)) < 2:
            per_label_auc.append(np.nan)
        else:
            per_label_auc.append(roc_auc_score(y_true, y_pred))

    mean_auc = float(np.nanmean(per_label_auc))
    out = {"mean_auc": mean_auc}
    for lab, auc in zip(LABELS, per_label_auc):
        out[f"auc_{lab}"] = float(auc) if not math.isnan(auc) else -1.0
    return out

# ------------------------- SECTION 8: TRAININGARGUMENTS HELPERS -------------------------
EVAL_STEPS = 500
LOG_STEPS  = 100
SAVE_STEPS = 500

def make_training_args(output_dir: str) -> TrainingArguments:
    base = dict(
        output_dir=output_dir,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=BS_TRAIN,
        per_device_eval_batch_size=BS_EVAL,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="mean_auc",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=LOG_STEPS,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        save_total_limit=2,
    )
    try:
        return TrainingArguments(evaluation_strategy="steps", save_strategy="steps", **base)
    except TypeError:
        # older Transformers on Kaggle
        return TrainingArguments(eval_strategy="steps", save_strategy="steps", **base)

def make_infer_args(output_dir: str) -> TrainingArguments:
    base = dict(
        output_dir=output_dir,
        per_device_eval_batch_size=BS_EVAL,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )
    try:
        return TrainingArguments(evaluation_strategy="no", save_strategy="no", **base)
    except TypeError:
        return TrainingArguments(eval_strategy="no", save_strategy="no", **base)

# ------------------------- SECTION 9: ONE-FOLD TRAIN FUNCTION -------------------------
def train_one_fold(trn_df: pd.DataFrame, val_df: pd.DataFrame, fold_name: str):
    set_seed(RNG_SEED)

    # class imbalance weighting
    y = trn_df[LABELS].values
    pos = y.sum(axis=0)
    neg = len(trn_df) - pos
    pos_weight = torch.tensor(
        np.maximum(neg / np.maximum(pos, 1), 1.0),
        dtype=torch.float32
    )

    train_ds = df_to_hfds(trn_df)
    val_ds   = df_to_hfds(val_df)

    cfg = AutoConfig.from_pretrained(MODEL_NAME, num_labels=len(LABELS))
    cfg.problem_type = "multi_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=cfg)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    args = make_training_args(output_dir=f"roberta-{fold_name}")

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        pos_weight=pos_weight,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    # HF 4.40+ uses processing_class; older versions expect tokenizer
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok

    trainer = BCETrainer(**trainer_kwargs)

    trainer.train()

    # Log history for plots
    loghist = pd.DataFrame(trainer.state.log_history)
    loghist.to_csv(f"logs_{fold_name}.csv", index=False)

    # Validation predictions
    val_logits = trainer.predict(val_ds).predictions
    val_probs  = sigmoid(val_logits)

    # --- ROC curves ---
    plt.figure(figsize=(10,7)); any_line = False
    for j, lab in enumerate(LABELS):
        y_true = val_df[lab].values
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, val_probs[:, j])
        plt.plot(fpr, tpr, label=lab); any_line = True
    if any_line:
        plt.plot([0,1],[0,1],"--",color="black")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"ROC curves — {fold_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/roc_{fold_name}.png")
        plt.show()
    else:
        plt.close()

    # --- PR curves ---
    plt.figure(figsize=(10,7)); any_line = False
    for j, lab in enumerate(LABELS):
        y_true = val_df[lab].values
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, val_probs[:, j])
        ap = average_precision_score(y_true, val_probs[:, j])
        plt.plot(recall, precision, label=f"{lab} (AP={ap:.3f})"); any_line = True
    if any_line:
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Precision–Recall — {fold_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/pr_{fold_name}.png")
        plt.show()
    else:
        plt.close()

    # --- Validation AUC vs step ---
    try:
        curve = loghist.dropna(subset=["eval_mean_auc"])
        plt.figure(figsize=(6,4))
        plt.plot(curve["step"], curve["eval_mean_auc"], marker="o")
        plt.xlabel("step"); plt.ylabel("mean AUC")
        plt.title(f"Validation mean AUC — {fold_name}")
        plt.tight_layout()
        plt.savefig(f"plots/val_auc_{fold_name}.png")
        plt.show()
    except Exception:
        pass

    # --- Training loss vs step ---
    try:
        loss_curve = loghist.dropna(subset=["loss"])
        plt.figure(figsize=(6,4))
        plt.plot(loss_curve["step"], loss_curve["loss"])
        plt.xlabel("step"); plt.ylabel("loss")
        plt.title(f"Training loss — {fold_name}")
        plt.tight_layout()
        plt.savefig(f"plots/train_loss_{fold_name}.png")
        plt.show()
    except Exception:
        pass

    # Simple pipeline diagram
    fig, ax = plt.subplots(figsize=(8,2.8))
    ax.axis("off")
    boxes = [
        ("raw text", 0.05),
        ("Tokenizer", 0.25),
        ("RoBERTa base", 0.50),
        ("[CLS]/Pooling", 0.75),
        ("6 logits → Sigmoid", 0.92),
    ]
    for label, x in boxes:
        ax.text(x, 0.5, label, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black"))
    for i in range(len(boxes)-1):
        ax.annotate("", xy=(boxes[i+1][1]-0.04, 0.5),
                    xytext=(boxes[i][1]+0.07, 0.5),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
    plt.tight_layout()
    plt.savefig(f"plots/pipeline_{fold_name}.png")
    plt.show()

    best_ckpt = trainer.state.best_model_checkpoint or args.output_dir

    del model, trainer, train_ds, val_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_probs, best_ckpt

# ------------------------- SECTION 10: CV TRAINING + INFERENCE -------------------------
has_any = (train_df[LABELS].sum(axis=1) > 0).astype(int)

test_ds = test_to_hfds(test_df)
test_pred_accum = np.zeros((len(test_df), len(LABELS)), dtype=np.float32)
oof_rows = []
fold_auc_rows = []

if USE_CV:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)

    for fold_idx, (idx_tr, idx_val) in enumerate(skf.split(train_df, has_any), start=1):
        fold_name = f"fold{fold_idx}"
        print(f"\n======== {fold_name} / {N_FOLDS} ========")
        trn_df = train_df.iloc[idx_tr].reset_index(drop=True)
        val_df = train_df.iloc[idx_val].reset_index(drop=True)

        val_probs, best_ckpt = train_one_fold(trn_df, val_df, fold_name=fold_name)

        # per-label AUCs on this fold
        per_label_auc = []
        for j, lab in enumerate(LABELS):
            y_true = val_df[lab].values
            if len(np.unique(y_true)) < 2:
                per_label_auc.append(np.nan)
            else:
                per_label_auc.append(roc_auc_score(y_true, val_probs[:, j]))
        mean_auc = float(np.nanmean(per_label_auc))
        print(f"{fold_name} mean AUC: {mean_auc:.6f}")
        fold_auc_rows.append(dict(
            fold=fold_name,
            **{lab: float(a) if not np.isnan(a) else np.nan
               for lab, a in zip(LABELS, per_label_auc)}
        ))

        # test predictions from the best checkpoint
        cfg = AutoConfig.from_pretrained(MODEL_NAME, num_labels=len(LABELS))
        cfg.problem_type = "multi_label_classification"
        model = AutoModelForSequenceClassification.from_pretrained(best_ckpt, config=cfg)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        infer_args = make_infer_args(output_dir=best_ckpt)

        infer_kwargs = dict(model=model, args=infer_args, data_collator=collator)
        if "processing_class" in inspect.signature(Trainer.__init__).parameters:
            infer_kwargs["processing_class"] = tok
        else:
            infer_kwargs["tokenizer"] = tok
        infer_trainer = Trainer(**infer_kwargs)

        test_logits = infer_trainer.predict(test_ds).predictions
        test_pred_accum += sigmoid(test_logits)

        # OOF storage
        oof = pd.DataFrame({
            "id": val_df["id"].values,
            "fold": fold_idx
        })
        for j, lab in enumerate(LABELS):
            oof[lab] = val_probs[:, j]
        oof_rows.append(oof)

        del model, infer_trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_probs = test_pred_accum / N_FOLDS
    oof_all = pd.concat(oof_rows, axis=0, ignore_index=True)
    oof_all.to_csv("oof_validation_probs.csv", index=False)

    # grouped bar chart of AUC per label per fold
    auc_df = pd.DataFrame(fold_auc_rows)
    melt_rows = []
    for _, row in auc_df.iterrows():
        for lab in LABELS:
            melt_rows.append((row["fold"], lab, row[lab]))
    long_auc = pd.DataFrame(melt_rows, columns=["fold", "label", "auc"])

    plt.figure(figsize=(10,5))
    folds = long_auc["fold"].unique().tolist()
    x = np.arange(len(LABELS))
    width = 0.8 / len(folds)
    for i, f in enumerate(folds):
        vals = long_auc[long_auc["fold"] == f]["auc"].values
        plt.bar(x + i*width, vals, width, label=f)
    plt.xticks(x + (len(folds)-1)*width/2, LABELS)
    plt.ylabel("AUC")
    plt.title("AUC by label across folds")
    plt.ylim(0.94, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/auc_by_label_grouped.png")
    plt.show()

else:
    # single train/val split (if you ever want it)
    trn_df, val_df = train_test_split(
        train_df, test_size=0.10, random_state=RNG_SEED, stratify=has_any
    )
    val_probs, best_ckpt = train_one_fold(trn_df, val_df, fold_name="single")

    cfg = AutoConfig.from_pretrained(MODEL_NAME, num_labels=len(LABELS))
    cfg.problem_type = "multi_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(best_ckpt, config=cfg)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    infer_args = make_infer_args(output_dir=best_ckpt)
    infer_kwargs = dict(model=model, args=infer_args, data_collator=collator)
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        infer_kwargs["processing_class"] = tok
    else:
        infer_kwargs["tokenizer"] = tok
    infer_trainer = Trainer(**infer_kwargs)

    test_logits = infer_trainer.predict(test_ds).predictions
    test_probs = sigmoid(test_logits)

    oof = pd.DataFrame({"id": val_df["id"].values, "fold": 1})
    for j, lab in enumerate(LABELS):
        oof[lab] = val_probs[:, j]
    oof.to_csv("oof_validation_probs.csv", index=False)

# ------------------------- SECTION 11: SUBMISSION -------------------------
# start from sample submission so column order matches competition exactly
sub = sample_df.copy()
if "id" not in sub.columns:
    sub.insert(0, "id", test_df["id"].values)

for i, lab in enumerate(LABELS):
    sub[lab] = test_probs[:, i]

sub.to_csv("submission.csv", index=False)
print("Saved submission.csv with columns:", list(sub.columns))

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


