# =======================
# Full fixed notebook (offline-safe, submission-ready)
# - Designed so you can "Run all" locally with the sample test.csv (if present)
# - When you Submit to the competition, Kaggle will re-run this notebook and provide the full test.csv
# =======================

# 0) Force offline transformers/datasets to avoid accidental internet usage
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 1) Imports & env
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# -----------------------
# 2) Files & read train only (we don't require test.csv locally)
# -----------------------
TRAIN_CSV = "/kaggle/input/jigsaw-agile-community-rules/train.csv"
TEST_CSV  = "/kaggle/input/jigsaw-agile-community-rules/test.csv"  # may be a small sample locally

assert Path(TRAIN_CSV).exists(), "train.csv missing under /kaggle/input/jigsaw-agile-community-rules/"
train_df = pd.read_csv(TRAIN_CSV)

print("Train columns:", list(train_df.columns))
print("Label counts:\n", train_df['rule_violation'].value_counts())

# -----------------------
# 3) Build text inputs
# -----------------------
RULE_COL = 'rule'
BODY_COL = 'body'
train_df['input_text'] = "Rule: " + train_df[RULE_COL].fillna('').astype(str) + " [SEP] Comment: " + train_df[BODY_COL].fillna('').astype(str)
train_df['rule_violation'] = train_df['rule_violation'].astype(int)

# -----------------------
# 4) Tokenizer / Hyperparams and model path discovery
# -----------------------
CANDIDATE_PATHS = [
    "/kaggle/input/microsoft-deberta-v3-base",
    "/kaggle/input/hy7haseeb-microsoft-deberta-v3-base",
    "/kaggle/input/deberta-v3-base",
]

INPUT_ROOT = "/kaggle/input"
if os.path.isdir(INPUT_ROOT):
    for name in os.listdir(INPUT_ROOT):
        lname = name.lower()
        if "deberta" in lname or "microsoft-deberta" in lname or "microsoft_deberta" in lname:
            candidate = os.path.join(INPUT_ROOT, name)
            if candidate not in CANDIDATE_PATHS:
                CANDIDATE_PATHS.append(candidate)

def find_model_path(candidates):
    for c in candidates:
        if not c:
            continue
        if os.path.isdir(c):
            files = set(os.listdir(c))
            if "config.json" in files and ("pytorch_model.bin" in files or "tokenizer_config.json" in files or "tokenizer.json" in files):
                return c
            for sub in files:
                subp = os.path.join(c, sub)
                if os.path.isdir(subp):
                    subfiles = set(os.listdir(subp))
                    if "config.json" in subfiles and ("pytorch_model.bin" in subfiles or "tokenizer_config.json" in subfiles):
                        return subp
    return None

MODEL_PATH = find_model_path(CANDIDATE_PATHS)
if MODEL_PATH is None:
    raise FileNotFoundError(
        "Could not find a local DeBERTa model under /kaggle/input.\n"
        "Attach the folder that contains config.json & pytorch_model.bin via 'Add data' in the notebook."
    )

print("Using MODEL_PATH =", MODEL_PATH)
print("Files in MODEL_PATH:", os.listdir(MODEL_PATH)[:50])

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, local_files_only=True)

# hyperparams
MAX_LEN = 384
BATCH_SIZE = 6
GRAD_ACCUM = 2
LR = 2e-5
EPOCHS = 5
WARMUP_RATIO = 0.06
SEED = 42

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------
# 5) Build HF train dataset and tokenize
# -----------------------
hf_train = Dataset.from_pandas(
    train_df[['input_text','rule_violation']].rename(columns={'rule_violation':'labels'}).reset_index(drop=True)
)

def preprocess_tokenize(examples):
    texts = examples['input_text']
    tok = tokenizer(texts, truncation=True, padding=False, max_length=MAX_LEN)
    if 'labels' in examples:
        tok['labels'] = examples['labels']
    return tok

hf_train = hf_train.map(preprocess_tokenize, batched=True, remove_columns=['input_text'])
hf_train.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

data_collator = DataCollatorWithPadding(tokenizer)

# -----------------------
# 6) Model builder
# -----------------------
def build_model():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2, local_files_only=True)

# -----------------------
# 7) 5-fold CV training
# -----------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
labels = train_df['rule_violation'].values
oof_preds = np.zeros(len(labels))
fold_aucs = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f"\n--- Fold {fold+1}/{skf.n_splits} ---")
    train_split = hf_train.select(list(tr_idx))
    val_split   = hf_train.select(list(val_idx))

    model = build_model()

    training_args = TrainingArguments(
        output_dir=f"./deberta_fold{fold}",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        warmup_ratio=WARMUP_RATIO,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_roc_auc",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        seed=SEED,
    )

    def compute_metrics(eval_pred):
        logits, labels_eval = eval_pred
        probs = softmax(logits, axis=1)[:,1]
        auc = float(roc_auc_score(labels_eval, probs))
        return {"eval_roc_auc": auc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=val_split,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    val_out = trainer.predict(val_split)
    val_logits = val_out.predictions
    if isinstance(val_logits, tuple):
        val_logits = val_logits[0]
    val_probs = softmax(val_logits, axis=1)[:,1]
    oof_preds[val_idx] = val_probs
    auc = roc_auc_score(labels[val_idx], val_probs)
    fold_aucs.append(float(auc))
    print(f"Fold {fold+1} AUC: {auc:.5f}")

print("CV fold AUCs:", fold_aucs)
print("Mean CV AUC:", float(np.mean(fold_aucs)))

# -----------------------
# 8) Retrain on full dataset
# -----------------------
model_final = build_model()
final_args = TrainingArguments(
    output_dir="./deberta_single",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    warmup_ratio=WARMUP_RATIO,
    fp16=torch.cuda.is_available(),
    save_strategy="no",
    logging_steps=100,
    report_to="none",
    seed=SEED,
)

trainer_final = Trainer(
    model=model_final,
    args=final_args,
    train_dataset=hf_train,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer_final.train()
trainer_final.save_model("./deberta_single")
tokenizer.save_pretrained("./deberta_single")

# -----------------------
# 9) Predict on test and save submission
# -----------------------
# Behavior:
#  - If TEST_CSV exists locally (sample or full), use it and produce submission.csv so the Notebook Version has output.
#  - If TEST_CSV is missing locally, attempt to find any test.csv under /kaggle/input and use it.
#  - If still missing, print a clear message and create a minimal placeholder submission.csv (so the Notebook Version is submittable).
#    When you actually Submit the notebook to the competition, Kaggle will re-run it and the full test.csv will be present.

def find_any_test_csv():
    # look for any file named test.csv under /kaggle/input
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if f.lower() == "test.csv":
                return os.path.join(root, f)
    return None

test_path_to_use = None
if Path(TEST_CSV).exists():
    test_path_to_use = TEST_CSV
else:
    found = find_any_test_csv()
    if found:
        test_path_to_use = found

if test_path_to_use is not None:
    test_df = pd.read_csv(test_path_to_use)
    print(f"Loaded test.csv with {len(test_df)} rows from {test_path_to_use}")
    if len(test_df) < 100:
        print("WARNING: that looks like the small sample test.csv (<=100 rows).")
        print("When you submit this notebook to the competition, Kaggle will re-run it and provide the full test.csv.")
    # Build hf_test and tokenize
    test_df['input_text'] = "Rule: " + test_df[RULE_COL].fillna('').astype(str)  + " [SEP] Comment: " + test_df[BODY_COL].fillna('').astype(str)
    hf_test  = Dataset.from_pandas(test_df[['input_text']].reset_index(drop=True))
    hf_test  = hf_test.map(preprocess_tokenize, batched=True, remove_columns=['input_text'])
    hf_test.set_format(type='torch', columns=['input_ids','attention_mask'])

    test_out = trainer_final.predict(hf_test)
    test_logits = test_out.predictions
    if isinstance(test_logits, tuple):
        test_logits = test_logits[0]
    test_probs = softmax(test_logits, axis=1)[:,1]

    out_path = "/kaggle/working/submission.csv"
    row_id_col = 'row_id' if 'row_id' in test_df.columns else None
    row_ids = test_df[row_id_col].values if row_id_col else np.arange(len(test_df))

    out_df = pd.DataFrame({"row_id": row_ids, "rule_violation": test_probs})
    out_df.to_csv(out_path, index=False)
    print("Saved submission.csv ->", out_path, "rows:", len(out_df))
else:
    # No test.csv found locally - create a clear placeholder so the Notebook Version has a submission file
    print("WARNING: No test.csv found locally under /kaggle/input.")
    print("When you submit this notebook to the competition, Kaggle will re-run it and provide the full test.csv.")
    # Create a tiny placeholder submission.csv (helps Notebook Version be saved with an output file).
    out_path = "/kaggle/working/submission.csv"
    placeholder = pd.DataFrame({"row_id":[0], "rule_violation":[0.0]})
    placeholder.to_csv(out_path, index=False)
    print("Saved placeholder submission.csv ->", out_path, "rows:", len(placeholder))
    print("IMPORTANT: This placeholder is only so the saved Notebook Version has a submission.csv. Do NOT submit this placeholder version.")
    print("Instead, save this notebook version and then use 'Submit to competition' — Kaggle will re-run this notebook and generate a full submission from the hidden test set.")

print("Files in /kaggle/working:", os.listdir('/kaggle/working')[:50])





