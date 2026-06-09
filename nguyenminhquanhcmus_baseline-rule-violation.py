
# =========================
# CONFIG: Pick your mode
# =========================
MODE = 3  # 1=TFIDF+LR, 2=Bi-Encoder, 3=Cross-Encoder

# Common paths (Kaggle: keep as ".")
TRAIN_PATH = "/kaggle/input/jigsaw-agile-community-rules/train.csv"
TEST_PATH  = "/kaggle/input/jigsaw-agile-community-rules/test.csv"
SAMPLE_SUB = "/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv"
SUBMIT_OUT = "/kaggle/working/submission.csv"

# Reproducibility
SEED = 42

# Bi-Encoder configs (MODE=2)
BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cross-Encoder configs (MODE=3)
CE_INPUT         = "/kaggle/input/robertabase/roberta-base"
CE_MODEL         = "roberta-base"           # try "microsoft/deberta-v3-base" later
MAX_LENGTH       = 384
LR               = 2e-5
EPOCHS           = 3
BATCH_SIZE       = 16
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.06
FP16             = True                     # set False if fp16 not available
EVAL_STEPS       = 200
SAVE_BEST        = True




# Install packages if needed (Kaggle usually has them pre-installed).
# Uncomment in Kaggle if missing, or leave as-is if environment is ready.
# !pip -q install scikit-learn pandas numpy torch torchvision torchaudio -U
# !pip -q install transformers datasets accelerate -U
# !pip -q install sentence-transformers -U




import os, random, math, gc, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def read_data(train_path, test_path, sample_path):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    sample   = pd.read_csv(sample_path)
    # Basic checks
    assert "row_id" in test_df.columns, "test.csv must have 'row_id'"
    assert "body" in train_df.columns and "rule" in train_df.columns, "train needs 'body' & 'rule'"
    assert "rule_violation" in train_df.columns, "train needs 'rule_violation'"
    return train_df, test_df, sample

def build_text_row(row, include_examples=True, include_subreddit=True, max_chars_each_example=300):
    # Flexible template to concatenate useful fields
    parts = []
    parts.append(f"Rule: {str(row.get('rule',''))}")
    if include_subreddit:
        parts.append(f"Subreddit: {str(row.get('subreddit',''))}")
    if include_examples:
        for k in ['positive_example_1','positive_example_2','negative_example_1','negative_example_2']:
            if k in row and pd.notna(row[k]) and len(str(row[k]))>0:
                ex = str(row[k])[:max_chars_each_example]
                parts.append(f"{k}: {ex}")
    parts.append(f"Comment: {str(row.get('body',''))}")
    return " \n ".join(parts)





# ============================
# MODE 1: TF-IDF + LogReg
# ============================
def run_tfidf_lr(train_df, test_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    # Build text
    train_texts = train_df.apply(lambda r: build_text_row(r, include_examples=False), axis=1)
    test_texts  = test_df.apply(lambda r: build_text_row(r, include_examples=False), axis=1)

    y = train_df["rule_violation"].astype(int).values

    X_train, X_valid, y_train, y_valid = train_test_split(train_texts, y, test_size=0.2, random_state=SEED, stratify=y)

    vec = TfidfVectorizer(ngram_range=(1,2), max_features=30000, min_df=2)
    Xtr = vec.fit_transform(X_train)
    Xva = vec.transform(X_valid)

    clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced", n_jobs=None)
    clf.fit(Xtr, y_train)

    va_prob = clf.predict_proba(Xva)[:,1]
    auc = roc_auc_score(y_valid, va_prob)
    print(f"[TFIDF+LR] Valid AUC = {auc:.5f}")

    Xtst = vec.transform(test_texts)
    tst_prob = clf.predict_proba(Xtst)[:,1]
    return tst_prob, auc




# ===========================================
# MODE 2: SentenceTransformer Bi-Encoder + LR
# ===========================================
def run_bi_encoder(train_df, test_df, model_name):
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    model = SentenceTransformer(model_name)

    # Encode body & rule separately; combine with elementwise product
    train_body = train_df["body"].astype(str).tolist()
    train_rule = train_df["rule"].astype(str).tolist()
    test_body  = test_df["body"].astype(str).tolist()
    test_rule  = test_df["rule"].astype(str).tolist()

    print("Encoding train body...")
    body_tr = model.encode(train_body, batch_size=128, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    print("Encoding train rule...")
    rule_tr = model.encode(train_rule, batch_size=128, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    X = np.concatenate([body_tr, rule_tr, body_tr*rule_tr, np.abs(body_tr-rule_tr)], axis=1)
    y = train_df["rule_violation"].astype(int).values

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    va_prob = clf.predict_proba(X_va)[:,1]
    auc = roc_auc_score(y_va, va_prob)
    print(f"[Bi-Encoder] Valid AUC = {auc:.5f}")

    # Test
    print("Encoding test body/rule...")
    body_te = model.encode(test_body, batch_size=128, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    rule_te = model.encode(test_rule, batch_size=128, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    X_te = np.concatenate([body_te, rule_te, body_te*rule_te, np.abs(body_te-rule_te)], axis=1)

    tst_prob = clf.predict_proba(X_te)[:,1]
    return tst_prob, auc




# ==================================
# MODE 3: Cross-Encoder (Transformers)
# ==================================
def run_cross_encoder(train_df, test_df):
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(CE_INPUT, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(CE_INPUT, local_files_only=True,num_labels=1,                      # ép về 1 logit
                                                                    ignore_mismatched_sizes=True )      # chấp nhận re-init head)


    def to_text(row):
        return build_text_row(row, include_examples=True, include_subreddit=True, max_chars_each_example=200)

    train_df = train_df.copy()
    train_df["text"] = train_df.apply(to_text, axis=1)
    test_df  = test_df.copy()
    test_df["text"]  = test_df.apply(to_text, axis=1)

    tr, va = train_test_split(train_df, test_size=0.2, random_state=SEED, stratify=train_df["rule_violation"])

    class TxtDS(Dataset):
        def __init__(self, df, is_train=True):
            self.df = df.reset_index(drop=True)
            self.is_train = is_train
        def __len__(self): return len(self.df)
        def __getitem__(self, i):
            r = self.df.iloc[i]
            item = tokenizer(
                r["text"],
                truncation=True, padding="max_length", max_length=MAX_LENGTH
            )
            if self.is_train:
                item["labels"] = np.array([r["rule_violation"]], dtype=np.float32)
            return {k: torch.tensor(v) for k,v in item.items()}

    ds_tr = TxtDS(tr, True)
    ds_va = TxtDS(va, True)
    ds_te = TxtDS(test_df, False)

    model.to(device)

    args = TrainingArguments(
        output_dir="./ce_out",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="steps",
        save_strategy="steps" if SAVE_BEST else "no",
        save_total_limit=1,
        logging_steps=200,
        eval_steps=EVAL_STEPS,
        save_steps=EVAL_STEPS,        # quan trọng: match với eval_steps
        load_best_model_at_end=SAVE_BEST,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        warmup_ratio=WARMUP_RATIO,
        fp16=bool(FP16),
        report_to="none"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1/(1+np.exp(-logits.reshape(-1)))
        auc = roc_auc_score(labels, probs)
        return {"roc_auc": auc}

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    # Validation AUC
    va_pred = trainer.predict(ds_va).predictions.reshape(-1)
    va_prob = 1/(1+np.exp(-va_pred))
    auc = roc_auc_score(va["rule_violation"].values, va_prob)
    print(f"[Cross-Encoder] Valid AUC = {auc:.5f}")

    # Test inference
    te_logits = trainer.predict(ds_te).predictions.reshape(-1)
    te_prob = 1/(1+np.exp(-te_logits))
    return te_prob, auc




# ============== DRIVER ==============
set_seed(SEED)
train_df, test_df, sample = read_data(TRAIN_PATH, TEST_PATH, SAMPLE_SUB)

print("Train shape:", train_df.shape, "| Test shape:", test_df.shape)
print("Columns:", list(train_df.columns))

if MODE == 1:
    preds, auc = run_tfidf_lr(train_df, test_df)
elif MODE == 2:
    preds, auc = run_bi_encoder(train_df, test_df, BI_ENCODER_MODEL)
elif MODE == 3:
    preds, auc = run_cross_encoder(train_df, test_df)
else:
    raise ValueError("MODE must be 1, 2, or 3.")

# Build submission
sub = pd.DataFrame({
    "row_id": test_df["row_id"].astype(int),
    "rule_violation": np.clip(preds, 0.0, 1.0).astype(float)
})
sub.to_csv(SUBMIT_OUT, index=False)
print("Saved:", SUBMIT_OUT)
sub.head()


