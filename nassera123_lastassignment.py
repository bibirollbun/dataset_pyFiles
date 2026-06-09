# ============================================
# DistilBERT Strong Baseline for Jigsaw Toxic Classification
# - Fine-tuning DistilBERT on multi-label task
# - Cosine LR + warmup, Early stopping
# - 5-fold CV ensembling, pos_weight for imbalance
# - Saves & shows figures under plots/*.png
# Output: submission.csv
# ============================================

# Phase 1: Runtime Guards and Configuration ğŸš€
# -------------------- Runtime guards (must run BEFORE any HF imports) --------------------
import os, sys, importlib, inspect, math, random, gc
os.environ["TRANSFORMERS_NO_TF"]    = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"]    = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]    = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# Kaggle stability tweaks
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

# -------------------- Imports --------------------
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

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
    Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorWithPadding
)

# -------------------- Config --------------------
DATA_DIR    = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"
MODEL_NAME = "distilbert-base-uncased"       
MAX_LEN    = 256
EPOCHS     = 4
LR         = 2.5e-5
BS_TRAIN   = 16
BS_EVAL    = 64
WARMUP_RATIO    = 0.10
WEIGHT_DECAY    = 0.01
GRAD_ACCUM_STEPS = 2
USE_CV     = True
N_FOLDS    = 5
LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
RNG_SEED = 42

random.seed(RNG_SEED); np.random.seed(RNG_SEED); torch.manual_seed(RNG_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RNG_SEED)

os.makedirs("plots", exist_ok=True)

# Phase 2: Utility Functions ğŸ› ï¸�
# -------------------- Utils --------------------
def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.replace("\t", " ").replace("\r", " ")
    s = " ".join(s.split())
    return s

def sigmoid(x): return 1. / (1. + np.exp(-x))

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# Phase 3: Data Loading and Preprocessing ğŸ’¾
# -------------------- Load data --------------------
TRAIN_CSV  = f"{DATA_DIR}/train.csv.zip"
TEST_CSV    = f"{DATA_DIR}/test.csv.zip"
SAMPLE_CSV = f"{DATA_DIR}/sample_submission.csv.zip"

train_df  = pd.read_csv(TRAIN_CSV)
test_df    = pd.read_csv(TEST_CSV)
sample_df = pd.read_csv(SAMPLE_CSV)

train_df["text"] = train_df["comment_text"].map(clean_text)
test_df["text"]  = test_df["comment_text"].map(clean_text)

# -------------------- Tokenizer / quick EDA figs --------------------
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8) # Pad to multiple of 8 for Tensor Cores

# 1) Label prevalence
counts = train_df[LABELS].sum().sort_values(ascending=False)
plt.figure(figsize=(6,4)); counts.plot(kind="bar")
plt.title("Label prevalence (train)"); plt.ylabel("positives")
plt.tight_layout(); plt.savefig("plots/label_prevalence.png"); plt.show()


# 2) Token lengths (first 20k)
encoded = tok(train_df["text"].tolist()[:20000], truncation=True, max_length=MAX_LEN, padding=False)
lengths = [len(ids) for ids in encoded["input_ids"]]
plt.figure(figsize=(6,4)); plt.hist(lengths, bins=50)
plt.title("Token length histogram (first 20k samples)")
plt.xlabel("length (tokens)"); plt.ylabel("count")
plt.tight_layout(); plt.savefig("plots/token_length_hist.png"); plt.show()


# -------------------- HF Datasets helpers --------------------
NUM_PROC = max(1, (os.cpu_count() or 2) // 2)

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=MAX_LEN)

def df_to_hfds(df):
    d = Dataset.from_pandas(df[["text"] + LABELS], preserve_index=False)
    d = d.map(tokenize, batched=True, remove_columns=["text"], num_proc=NUM_PROC)
    rename = {lab: f"lab_{lab}" for lab in LABELS}
    d = d.rename_columns(rename)
    def pack(ex):
        ex["labels"] = [float(ex[f"lab_{lab}"]) for lab in LABELS]
        return ex
    d = d.map(pack, num_proc=NUM_PROC)
    d = d.remove_columns(list(rename.values()))
    return d

def test_to_hfds(df):
    d = Dataset.from_pandas(df[["text"]], preserve_index=False)
    d = d.map(tokenize, batched=True, remove_columns=["text"], num_proc=NUM_PROC)
    return d

# Phase 4: Model, Training, and Metrics Setup ğŸ§ 
# -------------------- Custom Trainer --------------------
class BCETrainer(Trainer):
    def __init__(self, pos_weight=None, **kw):
        super().__init__(**kw)
        self._pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        # Custom BCEWithLogitsLoss to handle class imbalance
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=self._pos_weight.to(logits.device) if self._pos_weight is not None else None
        )
        loss = loss_fn(logits, labels.to(logits.device))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
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

# ------- Training/eval argument helpers -------
EVAL_STEPS = 500
LOG_STEPS  = 100
SAVE_STEPS = 500

def make_training_args(output_dir):
    base_kwargs = dict(
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
        return TrainingArguments(evaluation_strategy="steps", save_strategy="steps", **base_kwargs)
    except TypeError:
        return TrainingArguments(eval_strategy="steps", save_strategy="steps", **base_kwargs)

def make_infer_args(output_dir):
    base_kwargs = dict(
        output_dir=output_dir,
        per_device_eval_batch_size=BS_EVAL,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True
    )
    try:
        return TrainingArguments(evaluation_strategy="no", save_strategy="no", **base_kwargs)
    except TypeError:
        return TrainingArguments(eval_strategy="no", save_strategy="no", **base_kwargs)

# Phase 5: Training Execution Helpers ğŸ�ƒ
# -------------------- Train helpers --------------------
def train_once(trn_df, val_df, fold_name="single"):
    set_seed(RNG_SEED)

    # class imbalance weights calculation
    y = trn_df[LABELS].values
    pos = y.sum(axis=0)
    neg = len(trn_df) - pos
    pos_weight = torch.tensor(np.maximum(neg / np.maximum(pos, 1), 1.0), dtype=torch.float32)

    train_ds = df_to_hfds(trn_df)
    val_ds    = df_to_hfds(val_df)

    mcfg  = AutoConfig.from_pretrained(MODEL_NAME, num_labels=len(LABELS))
    mcfg.problem_type = "multi_label_classification"
    mdl   = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=mcfg)
    if hasattr(mdl, "gradient_checkpointing_enable"): mdl.gradient_checkpointing_enable()
    if hasattr(mdl.config, "use_cache"): mdl.config.use_cache = False

    args = make_training_args(output_dir=f"tox-{fold_name}")

    trainer_kwargs = dict(
        model=mdl,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        pos_weight=pos_weight,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    # Handle Trainer initialization parameters across HF versions
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok

    trainer = BCETrainer(**trainer_kwargs)
    trainer.train()

    # logs
    loghist = pd.DataFrame(trainer.state.log_history)
    loghist.to_csv(f"logs_{fold_name}.csv", index=False)

    # Validation predictions
    val_logits = trainer.predict(val_ds).predictions
    val_probs  = sigmoid(val_logits)

    # ---- Figures for this fold ----
    # ROC
    plt.figure(figsize=(10,7)); did_any = False
    for j, lab in enumerate(LABELS):
        y_true = val_df[lab].values
        if len(np.unique(y_true)) < 2: continue
        fpr, tpr, _ = roc_curve(y_true, val_probs[:, j])
        plt.plot(fpr, tpr, label=f"{lab}"); did_any = True
    if did_any:
        plt.plot([0,1],[0,1],"--", lw=1, color="black")
        plt.title(f"ROC curves â€” {fold_name}")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()
        plt.tight_layout(); plt.savefig(f"plots/roc_{fold_name}.png"); plt.show()
    else:
        plt.close()
    

    # PR
    plt.figure(figsize=(10,7)); did_any = False
    for j, lab in enumerate(LABELS):
        y_true = val_df[lab].values
        if len(np.unique(y_true)) < 2: continue
        precision, recall, _ = precision_recall_curve(y_true, val_probs[:, j])
        ap = average_precision_score(y_true, val_probs[:, j])
        plt.plot(recall, precision, label=f"{lab} (AP={ap:.3f})"); did_any = True
    if did_any:
        plt.title(f"Precisionâ€“Recall curves â€” {fold_name}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
        plt.tight_layout(); plt.savefig(f"plots/pr_{fold_name}.png"); plt.show()
    else:
        plt.close()
    

    # val AUC curve
    try:
        curve = loghist.dropna(subset=["eval_mean_auc"])
        plt.figure(figsize=(6,4)); plt.plot(curve["step"], curve["eval_mean_auc"], marker="o")
        plt.title(f"Validation mean AUC â€” {fold_name}")
        plt.xlabel("step"); plt.ylabel("AUC")
        plt.tight_layout(); plt.savefig(f"plots/val_auc_curve_{fold_name}.png"); plt.show()
    except Exception:
        pass
    

    # training loss curve
    try:
        loss_curve = loghist.dropna(subset=["loss"])
        plt.figure(figsize=(6,4)); plt.plot(loss_curve["step"], loss_curve["loss"], alpha=0.9)
        plt.title(f"Training loss â€” {fold_name}")
        plt.xlabel("step"); plt.ylabel("loss")
        plt.tight_layout(); plt.savefig(f"plots/train_loss_{fold_name}.png"); plt.show()
    except Exception:
        pass
    

    # pipeline diagram
    fig, ax = plt.subplots(figsize=(8,2.8)); ax.axis("off")
    boxes = [("Text", 0.05), ("Tokenizer (WordPiece)", 0.25), (f"{MODEL_NAME}", 0.50), ("[CLS]/Pooling", 0.75), ("6 logits \u2192 Sigmoid", 0.90)]
    for label, x in boxes:
        ax.text(x, 0.5, label, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black"))
    for i in range(len(boxes)-1):
        ax.annotate("", xy=(boxes[i+1][1]-0.05, 0.5), xytext=(boxes[i][1]+0.06, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    plt.tight_layout(); plt.savefig(f"plots/pipeline_{fold_name}.png"); plt.show()
    

    best_ckpt = trainer.state.best_model_checkpoint or args.output_dir
    del mdl; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return val_probs, best_ckpt

# Phase 6: Cross-Validation and Inference ğŸ“Š
# -------------------- Train/Eval --------------------
has_any = (train_df[LABELS].sum(axis=1) > 0).astype(int)

test_ds = test_to_hfds(test_df)
test_pred_accum = np.zeros((len(test_df), len(LABELS)))
oof_meta = []
fold_auc_rows = []

if USE_CV:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
    for k, (idx_tr, idx_va) in enumerate(skf.split(train_df, has_any)):
        fold = f"fold{k+1}"
        print(f"\n {fold} / {N_FOLDS} ")
        trn_df, val_df = train_df.iloc[idx_tr].reset_index(drop=True), train_df.iloc[idx_va].reset_index(drop=True)

        val_probs, best_ckpt = train_once(trn_df, val_df, fold_name=fold)

        # fold AUCs
        per_label = []
        for j, lab in enumerate(LABELS):
            y_true = val_df[lab].values
            if len(np.unique(y_true)) < 2: per_label.append(np.nan)
            else: per_label.append(roc_auc_score(y_true, val_probs[:, j]))
        mean_auc = float(np.nanmean(per_label))
        print(f"{fold} mean AUC: {mean_auc:.5f}")

        fold_auc_rows.append(dict(fold=fold, **{lab: float(a) if not np.isnan(a) else np.nan
                                                  for lab, a in zip(LABELS, per_label)}))

        # test predictions from best checkpoint (use inference-only args)
        mcfg  = AutoConfig.from_pretrained(MODEL_NAME, num_labels=len(LABELS))
        mcfg.problem_type = "multi_label_classification"
        mdl = AutoModelForSequenceClassification.from_pretrained(best_ckpt, config=mcfg)
        if hasattr(mdl.config, "use_cache"): mdl.config.use_cache = False
        eval_args = make_infer_args(output_dir=best_ckpt)

        init_kwargs = dict(model=mdl, args=eval_args, data_collator=collator)
        if "processing_class" in inspect.signature(Trainer.__init__).parameters:
            init_kwargs["processing_class"] = tok
        else:
            init_kwargs["tokenizer"] = tok
        eval_trainer = Trainer(**init_kwargs)

        test_logits = eval_trainer.predict(test_ds).predictions
        test_pred_accum += sigmoid(test_logits)

        # OOF save
        oof = pd.DataFrame({"id": val_df.index if "id" not in val_df.columns else val_df["id"]})
        for j, lab in enumerate(LABELS): oof[lab] = val_probs[:, j]
        oof["fold"] = k+1; oof_meta.append(oof)

        del mdl, eval_trainer; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    test_probs = test_pred_accum / N_FOLDS
    oof_all = pd.concat(oof_meta, axis=0, ignore_index=True)
    oof_all.to_csv("oof_validation_probs.csv", index=False)

    # ---- Grouped bar chart: AUC by label across folds ----
    auc_df = pd.DataFrame(fold_auc_rows)
    long_rows = []
    for _, row in auc_df.iterrows():
        for lab in LABELS:
            long_rows.append((row["fold"], lab, row[lab]))
    g = pd.DataFrame(long_rows, columns=["fold", "label", "auc"])
    plt.figure(figsize=(10,5))
    folds = g["fold"].unique().tolist(); labs = LABELS
    x = np.arange(len(labs)); width = 0.8 / len(folds)
    for i, f in enumerate(folds):
        vals = g[g["fold"]==f]["auc"].values
        plt.bar(x + i*width, vals, width, label=f)
    plt.xticks(x + (len(folds)-1)*width/2, labs)
    plt.ylim(0.94, 1.00)
    plt.ylabel("AUC"); plt.title("AUC by label across folds"); plt.legend()
    plt.tight_layout(); plt.savefig("plots/auc_by_label_grouped.png"); plt.show()
    

else:
    trn_df, val_df = train_test_split(train_df, test_size=0.10, random_state=RNG_SEED, stratify=has_any)
    val_probs, best_ckpt = train_once(trn_df, val_df, fold_name="single")

    # single-model inference
    mcfg  = AutoConfig.from_pretrained(MODEL_NAME, num_labels=len(LABELS))
    mcfg.problem_type = "multi_label_classification"
    mdl = AutoModelForSequenceClassification.from_pretrained(best_ckpt, config=mcfg)
    if hasattr(mdl.config, "use_cache"): mdl.config.use_cache = False
    eval_args = make_infer_args(output_dir=best_ckpt)

    init_kwargs = dict(model=mdl, args=eval_args, data_collator=collator)
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        init_kwargs["processing_class"] = tok
    else:
        init_kwargs["tokenizer"] = tok
    eval_trainer = Trainer(**init_kwargs)

    test_logits = eval_trainer.predict(test_ds).predictions
    test_probs  = sigmoid(test_logits)

    # OOF
    oof = pd.DataFrame({"id": val_df.index if "id" not in val_df.columns else val_df["id"]})
    for j, lab in enumerate(LABELS): oof[lab] = val_probs[:, j]
    oof["fold"] = 1
    oof.to_csv("oof_validation_probs.csv", index=False)

# Phase 7: Final Submission and Cleanup âœ…
# -------------------- Submission --------------------
sub = pd.DataFrame({"id": test_df["id"]})
for i, col in enumerate(LABELS):
    sub[col] = test_probs[:, i]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with columns:", sub.columns.tolist())

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Expected Execution Time: ~4 hours (5 folds * 2 epochs)

