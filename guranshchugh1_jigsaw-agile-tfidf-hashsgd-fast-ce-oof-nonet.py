# %% [code]
import os, gc, re, json, math, random, string
import numpy as np
import pandas as pd

from typing import List, Tuple, Dict, Iterable
from dataclasses import dataclass

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize as sk_normalize
from scipy import sparse
import joblib

import torch
from torch.utils.data import Dataset

try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
except:
    pass


# %% [markdown]
# # Jigsaw Agile Rules — Phase 0–1 (Data Hygiene, CV, Baselines, Embeddings)
# - Normalize + template
# - Near-duplicate removal (char-3gram Jaccard)
# - 5-fold stratified + 2 domain-shift folds
# - Rule clustering (KMeans) via E5 embeddings (fallback: TF-IDF)
# - Baseline: TF-IDF -> Logistic Regression (+ Platt)
# - Cheap neural baseline: deberta-v3-base cross-encoder (1–2 epochs)
# - Embedding store: E5 (comment/rule/examples)


SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

COMP_DIR = "/kaggle/input/jigsaw-agile-community-rules"
assert os.path.exists(COMP_DIR), "Attach the competition dataset!"

OUT_DIR = "/kaggle/working/artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

# Attach your local model datasets (no internet at runtime)
# MODEL_DIR_DEBERTA = "/kaggle/input/hf-deberta-v3-base"   # <- change to your dataset name/path
# MODEL_DIR_E5      = "/kaggle/input/hf-e5-base"           # <- change to your dataset name/path

train = pd.read_csv(os.path.join(COMP_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(COMP_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(COMP_DIR, "sample_submission.csv"))

print(train.head(2))
print(train.isna().mean())


# ## Phase 0.1 — Normalize + Template + keep normalized forms for later use

# %% [code]
URL_RE = re.compile(r'https?://\S+|www\.\S+')
USER_RE = re.compile(r'u/[A-Za-z0-9_\-]+|@[A-Za-z0-9_]+')
EMAIL_RE= re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
EMOJI_RE= re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)

def normalize_text(s: str) -> str:
    if not isinstance(s, str) or not s:
        return ""
    s = s.strip()
    # mask first (so placeholders aren't lowercased weirdly)
    s = URL_RE.sub(" <URL> ", s)
    s = USER_RE.sub(" <USER> ", s)
    s = EMAIL_RE.sub(" <EMAIL> ", s)
    s = EMOJI_RE.sub(" <EMOJI> ", s)
    # normalize whitespace + lowercase
    s = re.sub(r'\s+', ' ', s).lower()
    return s

def build_template(df: pd.DataFrame, clip_map=None) -> pd.Series:
    if clip_map is None:
        clip_map = dict(rule=400, positive_example_1=400, positive_example_2=400,
                        negative_example_1=400, negative_example_2=400, subreddit=100, body=1200)
    def clip(x, n=400):
        return (x[:n] if isinstance(x, str) else "") if n else (x if isinstance(x, str) else "")
    cols = ["rule","positive_example_1","positive_example_2","negative_example_1","negative_example_2","subreddit","body"]
    norm = {}
    for c in cols:
        norm[c] = df[c].fillna("").astype(str).map(normalize_text).map(lambda t: clip(t, clip_map.get(c, None)))
    parts = []
    parts.append("[RULE] " + norm["rule"])
    parts.append("[POS1] " + norm["positive_example_1"])
    parts.append("[POS2] " + norm["positive_example_2"])
    parts.append("[NEG1] " + norm["negative_example_1"])
    parts.append("[NEG2] " + norm["negative_example_2"])
    parts.append("[SUBREDDIT] " + norm["subreddit"])
    parts.append("[COMMENT] " + norm["body"])
    return pd.concat(parts, axis=1).agg("\n".join, axis=1)

train["_text"] = build_template(train)
test["_text"]  = build_template(test)

# Keep normalized columns for later use (embeddings)
for c in ["rule","positive_example_1","positive_example_2","negative_example_1","negative_example_2","subreddit","body"]:
    train[f"norm_{c}"] = train[c].fillna("").astype(str).map(normalize_text)
    test[f"norm_{c}"]  = test[c].fillna("").astype(str).map(normalize_text)


# ## Phase 0.2 — Near-duplicate removal (char-3gram Jaccard ≥ 0.9)
# Pure-Python approximate; buckets by length & first 12 chars to avoid O(N^2).

# %% [code]
def char_ngrams(s: str, n=3) -> set:
    s = s or ""
    if len(s) < n:
        return {s}
    return {s[i:i+n] for i in range(len(s)-n+1)}

def jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def dedupe_by_jaccard(df: pd.DataFrame, text_col="_text", threshold=0.9) -> pd.DataFrame:
    buckets = {}
    keep = np.ones(len(df), dtype=bool)
    ngrams_cache = {}
    for i, t in enumerate(df[text_col].fillna("").astype(str).values):
        key = (len(t)//50, (t[:12] if t else ""))
        ngrams = char_ngrams(t, 3)
        ngrams_cache[i] = ngrams
        if key not in buckets: buckets[key] = []
        # compare inside bucket only
        dup_found = False
        for j in buckets[key]:
            sim = jaccard(ngrams, ngrams_cache[j])
            if sim >= threshold:
                keep[i] = False
                dup_found = True
                break
        if not dup_found:
            buckets[key].append(i)
    return df.loc[keep].reset_index(drop=True)

print("Train before dedupe:", len(train))
train = dedupe_by_jaccard(train, "_text", 0.9)
print("Train after  dedupe:", len(train))


# Phase 0.3 — CV: 5-fold stratified + 2 domain-shift folds
# 5-fold stratified by target
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold = np.full(len(train), -1, dtype=int)
for k, (tr, va) in enumerate(skf.split(train, train["rule_violation"].values.astype(int))):
    fold[va] = k
train["fold"] = fold

# 2 domain-shift folds: group subreddits by frequency into 2 buckets
sub_counts = train["subreddit"].value_counts()
subs_sorted = list(sub_counts.index)

group_a, group_b = [], []
tot_a, tot_b = 0, 0
for s in subs_sorted:
    c = sub_counts[s]
    if tot_a <= tot_b:
        group_a.append(s); tot_a += c
    else:
        group_b.append(s); tot_b += c

def mark_shift_fold(df):
    # shift_fold 0: hold out A, 1: hold out B
    sf = np.zeros(len(df), dtype=int)
    mask_a = df["subreddit"].isin(group_a).values
    sf[mask_a] = 1  # if in A, then this row will be in validation for shift_fold 0
    # We'll store as which fold the row belongs to when held-out:
    # For inference later we use them to evaluate; for training they just annotate.
    return sf

train["shift_fold"] = mark_shift_fold(train)

print(train[["fold","shift_fold"]].head())



# ## Phase 0.4 — Rule clustering (KMeans)
# Prefer E5 embeddings; fallback to TF-IDF if not available.

MODEL_DIR_DEBERTA = "/kaggle/input/deberta-v3-base"   # <- change to your dataset name/path
MODEL_DIR_E5      = "/kaggle/input/e5-base-v2"        # <- change to your dataset name/path

def try_load_e5():
    ok = os.path.exists(MODEL_DIR_E5)
    if not ok:
        return None, None
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_DIR_E5, use_fast=True)
        mdl = AutoModel.from_pretrained(MODEL_DIR_E5)
        mdl.eval()
        return tok, mdl
    except Exception as e:
        print("E5 load failed, falling back ->", e)
        return None, None

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / attention_mask.sum(1).clamp(min=1e-9).unsqueeze(-1)

def encode_texts(texts: List[str], tok, mdl, batch=128, device=None):
    """E5-style encoding with 'passage:' prefix + L2 norm."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(device)
    embs = []
    for i in range(0, len(texts), batch):
        raw = [t if isinstance(t, str) else "" for t in texts[i:i+batch]]
        batch_text = ["passage: " + t for t in raw]   # <-- important for E5
        with torch.no_grad():
            enc = tok(batch_text, padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(device)
            last = mdl(**enc).last_hidden_state
            e = mean_pool(last, enc["attention_mask"])
            e = torch.nn.functional.normalize(e, p=2, dim=1)
            embs.append(e.cpu().numpy().astype(np.float32))
    return np.vstack(embs)

# ---------- Build rule text robustly ----------
rules_text_norm = (train["norm_rule"].fillna("").astype(str).tolist() + 
                   test["norm_rule"].fillna("").astype(str).tolist())

if len(set(rules_text_norm)) < 5:
    print(f"[Rule clustering] Too few unique normalized rules ({len(set(rules_text_norm))}). Falling back to raw 'rule'.")
    rules_text = (train["rule"].fillna("").astype(str).tolist() +
                  test["rule"].fillna("").astype(str).tolist())
else:
    rules_text = rules_text_norm

# ---------- Encode (E5 preferred; TF-IDF+SVD fallback) ----------
e5_tok, e5_mdl = try_load_e5()

if e5_tok is not None:
    RULE_EMB = encode_texts(rules_text, e5_tok, e5_mdl, batch=256)
else:
    print("Using TF-IDF fallback for rule clustering.")
    tfidf_rule = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1, max_features=200_000)
    RULE_EMB = tfidf_rule.fit_transform(rules_text)
    from sklearn.decomposition import TruncatedSVD
    RULE_EMB = TruncatedSVD(n_components=256, random_state=SEED).fit_transform(RULE_EMB).astype(np.float32)

# ---------- Split back ----------
n_train = len(train)
rule_emb_train = RULE_EMB[:n_train]
rule_emb_test  = RULE_EMB[n_train:]

# ---------- Choose K based on exact uniqueness to avoid warnings ----------
def n_unique_rows_exact(X):
    # exact uniqueness on dense float32 embeddings
    return np.unique(X, axis=0).shape[0]

uniq_train = n_unique_rows_exact(rule_emb_train)
if uniq_train <= 1:
    print("[Rule clustering] Only one unique rule vector in train; assigning cluster 0 to all.")
    train["rule_cluster"] = 0
    test["rule_cluster"]  = 0
else:
    K_target = 10  # between 8–12
    K_eff = min(K_target, uniq_train)
    if K_eff < K_target:
        print(f"[Rule clustering] Reducing K from {K_target} -> {K_eff} (distinct train vectors: {uniq_train}).")

    km = KMeans(n_clusters=K_eff, random_state=SEED, n_init=10)
    train["rule_cluster"] = km.fit_predict(rule_emb_train)
    test["rule_cluster"]  = km.predict(rule_emb_test)

# ---------- Save ----------
train.to_parquet(os.path.join(OUT_DIR, "train_phase01.parquet"), index=False)
test.to_parquet(os.path.join(OUT_DIR, "test_phase01.parquet"), index=False)


# =========================
# code piece 7 (FAST, GPU-aware, ETA)
# =========================

# Assumes you already loaded: train, test, sample_sub, SEED
# and created OUT_DIR (or we create a default below).
import os, warnings, gc, numpy as np
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ---- env + logging hygiene ----
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings(
    "ignore",
    message="The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option"
)
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

# ---- basic checks / setup ----
assert "_text" in train.columns, "Missing `_text` column. Build the tagged template earlier."
assert "rule_violation" in train.columns, "Missing target `rule_violation` in train."
if "OUT_DIR" not in globals():
    OUT_DIR = "/kaggle/working/artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Phase 1.1 — FAST baseline: HashingVectorizer + SGD (3 folds)
# ============================================================
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

N_FEATURES = 2**20        # 1,048,576 hashed features
FOLDS_TFIDF = 3           # speed; switch to 5 when locking CV
USE_WORD = True           # set False for char-only (even faster)
DO_CALIB = False          # turn on later if you need calibrated probs

char_vec = HashingVectorizer(
    analyzer="char", ngram_range=(3, 5),
    n_features=N_FEATURES, alternate_sign=False, norm="l2"
)
Xc_tr = char_vec.transform(train["_text"])
Xc_te = char_vec.transform(test["_text"])

if USE_WORD:
    word_vec = HashingVectorizer(
        analyzer="word", ngram_range=(1, 2),
        n_features=N_FEATURES, alternate_sign=False, norm="l2",
        token_pattern=r"(?u)\b\w+\b", lowercase=True
    )
    Xw_tr = word_vec.transform(train["_text"])
    Xw_te = word_vec.transform(test["_text"])
    X_train = sparse.hstack([Xc_tr, Xw_tr], format="csr").astype(np.float32)
    X_test  = sparse.hstack([Xc_te, Xw_te], format="csr").astype(np.float32)
else:
    X_train = Xc_tr.astype(np.float32)
    X_test  = Xc_te.astype(np.float32)

y = train["rule_violation"].astype(int).values

skf_fast = StratifiedKFold(n_splits=FOLDS_TFIDF, shuffle=True, random_state=SEED)
oof = np.zeros(len(train), dtype=float)
test_pred = np.zeros(len(test), dtype=float)

for f, (tr, va) in enumerate(skf_fast.split(X_train, y)):
    Xt, Xv = X_train[tr], X_train[va]
    yt, yv = y[tr], y[va]
    clf = SGDClassifier(
        loss="log_loss", alpha=1e-5,
        max_iter=30, tol=1e-3,
        n_jobs=-1, random_state=SEED
    )
    clf.fit(Xt, yt)
    oof[va] = clf.predict_proba(Xv)[:, 1]
    print(f"[Hashing-SGD] Fold {f} AUC: {roc_auc_score(yv, oof[va]):.6f}")
    test_pred += clf.predict_proba(X_test)[:, 1] / skf_fast.n_splits

base_auc = roc_auc_score(y, oof)
print(f"[Hashing-SGD] OOF AUC: {base_auc:.6f}")

if DO_CALIB:
    from sklearn.linear_model import LogisticRegression as PlattLR
    platt = PlattLR(solver="lbfgs", max_iter=500)
    platt.fit(oof.reshape(-1,1), y)
    oof = platt.predict_proba(oof.reshape(-1,1))[:,1]
    test_pred = platt.predict_proba(test_pred.reshape(-1,1))[:,1]
    print(f"[Hashing-SGD] Calibrated OOF AUC: {roc_auc_score(y, oof):.6f}")
    import joblib; joblib.dump(platt, os.path.join(OUT_DIR, "platt_calibrator.pkl"))

np.save(os.path.join(OUT_DIR, "tfidf_oof.npy"),  oof)
np.save(os.path.join(OUT_DIR, "tfidf_test.npy"), test_pred)

sub = sample_sub.copy()
sub["rule_violation"] = test_pred.clip(0,1)
sub.to_csv(os.path.join(OUT_DIR, "submission_tfidf.csv"), index=False)
print("Wrote:", os.path.join(OUT_DIR, "submission_tfidf.csv"))

# ===================================================================
# Phase 1.2 — FAST DeBERTa baseline (GPU-fed, pretokenized, HF-compat)
# ===================================================================
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)

MODEL_DIR_DEBERTA = globals().get("MODEL_DIR_DEBERTA", "/kaggle/input/deberta-v3-base")
HAS_DEBERTA = os.path.exists(MODEL_DIR_DEBERTA)
USE_GPU = torch.cuda.is_available()
print("DeBERTa available?", HAS_DEBERTA, "| CUDA:", USE_GPU)

if HAS_DEBERTA and USE_GPU:
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except: pass

    # speed knobs
    FOLDS_CE = 3
    MAX_LEN = 128
    PER_DEV_BS = 24        # try 24; if OOM, back down to 16
    GRAD_ACCUM = 1
    LR = 1e-3              # head-only training
    MAX_STEPS = 500
    LOG_STEPS = 100
    NUM_WORKERS = 2

    # ---- 1) Pre-tokenize once (FAST tokenizer + parallelism ON just for this step) ----
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR_DEBERTA, use_fast=True)
    train_texts = train["_text"].astype(str).tolist()
    test_texts  = test["_text"].astype(str).tolist()
    tr_enc = tok(train_texts, truncation=True, padding=False, max_length=MAX_LEN)
    te_enc = tok(test_texts,  truncation=True, padding=False, max_length=MAX_LEN)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    collator = DataCollatorWithPadding(tokenizer=tok)

    class PreTokenizedDataset(Dataset):
        def __init__(self, indices, encodings, labels=None):
            self.idx = np.asarray(indices)
            self.enc = encodings
            self.labels = labels
        def __len__(self): return len(self.idx)
        def __getitem__(self, i):
            j = int(self.idx[i])
            item = {k: torch.tensor(v[j]) for k, v in self.enc.items() if k in ("input_ids","attention_mask","token_type_ids")}
            if self.labels is not None:
                item["labels"] = torch.tensor(float(self.labels[j]), dtype=torch.float)
            return item

    def build_model_frozen():
        m = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR_DEBERTA, num_labels=1)
        # freeze encoder; train only pooler+classifier
        for n,p in m.named_parameters():
            if n.startswith("deberta.encoder.") or n.startswith("deberta.embeddings."):
                p.requires_grad = False
        if hasattr(m, "gradient_checkpointing_enable"): m.gradient_checkpointing_enable()
        if hasattr(m.config, "use_cache"): m.config.use_cache = False
        return m

    def infer(model, dataset, bs=64):
        model.eval(); device = "cuda"; model.to(device)
        loader = DataLoader(dataset, batch_size=bs, shuffle=False, collate_fn=collator,
                            num_workers=NUM_WORKERS, pin_memory=True)
        preds=[]
        with torch.no_grad():
            for batch in loader:
                batch.pop("labels", None)
                for k,v in list(batch.items()):
                    if hasattr(v, "to"): batch[k]=v.to(device, non_blocking=True)
                logits = model(**batch).logits.view(-1)
                preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.concatenate(preds)

    # ---- tiny ETA helper (measures your steps/sec) ----
    import time
    def estimate_steps_per_sec(model_builder, dataset, bs, grad_accum=1, warmup=20, measure=60):
        device = "cuda"; model = model_builder().to(device); model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True,
                            collate_fn=collator, num_workers=NUM_WORKERS,
                            pin_memory=True, drop_last=True)
        it = iter(loader)
        # warmup
        for _ in range(min(warmup, len(loader))):
            b = next(it); 
            for k,v in list(b.items()):
                if hasattr(v, "to"): b[k]=v.to(device, non_blocking=True)
            loss = model(**b).loss
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        # measure
        t0, steps = time.time(), 0
        for _ in range(min(measure, len(loader))):
            try: b = next(it)
            except StopIteration: break
            for k,v in list(b.items()):
                if hasattr(v, "to"): b[k]=v.to(device, non_blocking=True)
            loss = model(**b).loss
            loss.backward()
            if ((_+1) % grad_accum) == 0:
                opt.step(); opt.zero_grad(set_to_none=True)
            steps += 1
        torch.cuda.synchronize()
        dt = max(1e-6, time.time()-t0)
        sps = steps / dt / max(1, grad_accum)
        del model, opt, loader; torch.cuda.empty_cache(); gc.collect()
        return sps

    def print_fold_eta(model_builder, dataset, max_steps, bs, grad_accum=1):
        sps = estimate_steps_per_sec(model_builder, dataset, bs, grad_accum)
        if sps > 0:
            mins = (max_steps / sps) / 60.0
            print(f"[ETA] ~{mins:.1f} min for {max_steps} steps (≈{sps:.2f} steps/sec)")
        else:
            print("[ETA] Unable to measure steps/sec.")

    # ---- 5-fold split (3 folds for speed) ----
    y_ce = train["rule_violation"].astype(int).values
    skf = StratifiedKFold(n_splits=FOLDS_CE, shuffle=True, random_state=SEED)
    oof_ce = np.zeros(len(train), dtype=float)
    test_ce = np.zeros(len(test), dtype=float)

    for f,(tr_idx, va_idx) in enumerate(skf.split(train, y_ce)):
        torch.cuda.empty_cache(); gc.collect()
        ds_tr = PreTokenizedDataset(tr_idx, tr_enc, labels=y_ce)
        ds_va = PreTokenizedDataset(va_idx, tr_enc, labels=y_ce)
        ds_te = PreTokenizedDataset(np.arange(len(test)), te_enc, labels=None)

        # print ETA measured on your GPU
        print_fold_eta(build_model_frozen, ds_tr, MAX_STEPS, PER_DEV_BS, GRAD_ACCUM)

        args = TrainingArguments(
            output_dir=os.path.join(OUT_DIR, f"ce_fast_fold{f}"),
            learning_rate=LR,
            per_device_train_batch_size=PER_DEV_BS,
            gradient_accumulation_steps=GRAD_ACCUM,
            max_steps=MAX_STEPS,
            num_train_epochs=1,  # ignored after hitting max_steps
            fp16=True,
            logging_steps=LOG_STEPS,
            seed=SEED,
            dataloader_pin_memory=True,
            dataloader_num_workers=NUM_WORKERS,
        )

        model = build_model_frozen()

        # newer HF prefers processing_class; older HF needs tokenizer=
        try:
            from transformers import Trainer
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=ds_tr,
                data_collator=collator,
                processing_class=tok,
            )
        except TypeError:
            from transformers import Trainer
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=ds_tr,
                data_collator=collator,
                tokenizer=tok,
            )

        print(f"[CE fast] Fold {f} training up to {MAX_STEPS} steps… (BS={PER_DEV_BS}, L={MAX_LEN})")
        trainer.train()

        # full val + test inference
        oof_fold = infer(trainer.model, ds_va, bs=64)
        oof_ce[va_idx] = oof_fold
        print(f"[CE fast] Fold {f} OOF AUC: {roc_auc_score(y_ce[va_idx], oof_fold):.6f}")

        test_ce += infer(trainer.model, ds_te, bs=64) / skf.n_splits

        del trainer, model; torch.cuda.empty_cache(); gc.collect()

    ce_auc = roc_auc_score(y_ce, oof_ce)
    print(f"[CE fast] OOF AUC: {ce_auc:.6f}")
    np.save(os.path.join(OUT_DIR, "ce_oof.npy"),  oof_ce)
    np.save(os.path.join(OUT_DIR, "ce_test.npy"), test_ce)

else:
    print("CUDA not available or DeBERTa weights missing — skipping CE baseline. Hashing-SGD submission is ready.")


# =========================
# code piece 8 (FAST — robust paths, CUDA-safe workers, ctx auto-build, pretokenized test, AMP v2)
# =========================

import os, gc, math, random, re, warnings
import numpy as np
import pandas as pd
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from transformers.utils import logging as hf_logging

# ---- env + logging hygiene (match piece 7) ----
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings(
    "ignore",
    message="The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option"
)
hf_logging.set_verbosity_error()

# -------------------------------
# Paths & config
# -------------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda":
    print("CUDA not available — skipping Model A fast trainer (use Hashing-SGD submission for now).")

# Auto-pick competition directory (robust)
COMP_DIR_CANDIDATES = [
    "/kaggle/input/jigsaw-agile-community-rules",
    "/kaggle/input/jigsaw-agile-community-rules-classification",
]
def pick_comp_dir(cands):
    for d in cands:
        if os.path.exists(os.path.join(d, "sample_submission.csv")):
            return d
        if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(os.path.join(d, "test.csv")):
            return d
    raise FileNotFoundError("Could not find competition folder. Checked: " + ", ".join(cands))
COMP_DIR = pick_comp_dir(COMP_DIR_CANDIDATES)
print("Using COMP_DIR:", COMP_DIR)

ART_DIR_CANDIDATES = [
    "/kaggle/working/artifacts_fast",
    "/kaggle/working/artifacts",
]
OUT_DIR   = "/kaggle/working/modelA_debv3l_fast"
os.makedirs(OUT_DIR, exist_ok=True)

# Attach your large model dataset (no internet). Fallback to base if large missing.
PREF_MODEL_L = "/kaggle/input/deberta-v3-large"
PREF_MODEL_B = "/kaggle/input/deberta-v3-base"
MODEL_DIR_DEBERTA_L = PREF_MODEL_L if os.path.exists(PREF_MODEL_L) else PREF_MODEL_B
print("Using model from:", MODEL_DIR_DEBERTA_L)

# FAST toggles (relax later for score)
FAST_N_FOLDS          = 3
FAST_MAX_LEN          = 256
FAST_EPOCHS           = 1
FAST_TRAIN_BATCH      = 8
FAST_GRAD_ACCUM       = 2
FAST_MAX_STEPS_EPOCH  = 1200
USE_R_DROP            = False
R_DROP_LAMBDA         = 0.3
USE_FGM               = False
FGM_EPS               = 0.5
USE_E5_NEG            = False    # keep False for fastest runs

LR = 1e-5
WD = 0.01
WARMUP_FRAC = 0.06

torch.backends.cuda.matmul.allow_tf32 = True
try: torch.set_float32_matmul_precision("high")
except: pass

# -------------------------------
# Load processed Phase 0–1 data if present, else build minimally
# -------------------------------
def normalize_text(s: str) -> str:
    if not isinstance(s, str) or not s: return ""
    s = s.strip()
    s = re.sub(r'https?://\S+|www\.\S+', ' <URL> ', s)
    s = re.sub(r'u/[A-Za-z0-9_\-]+|@[A-Za-z0-9_]+', ' <USER> ', s)
    s = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' <EMAIL> ', s)
    s = re.sub("[\U00010000-\U0010ffff]", " <EMOJI> ", s)
    s = re.sub(r'\s+', ' ', s).lower()
    return s

def load_phase01_or_minimal():
    phase_paths = []
    for base in ART_DIR_CANDIDATES:
        phase_paths.append((os.path.join(base, "train_phase01.parquet"),
                            os.path.join(base, "test_phase01.parquet")))
    for tr_pq, te_pq in phase_paths:
        if os.path.exists(tr_pq) and os.path.exists(te_pq):
            tr = pd.read_parquet(tr_pq); te = pd.read_parquet(te_pq)
            return tr, te, True
    tr = pd.read_csv(os.path.join(COMP_DIR, "train.csv"))
    te = pd.read_csv(os.path.join(COMP_DIR, "test.csv"))
    for c in ["rule","positive_example_1","positive_example_2","negative_example_1","negative_example_2","subreddit","body"]:
        tr[f"norm_{c}"] = tr[c].fillna("").astype(str).map(normalize_text)
        te[f"norm_{c}"] = te[c].fillna("").astype(str).map(normalize_text)
    return tr, te, False

train, test, had_phase01 = load_phase01_or_minimal()

# --- Ensure rule_context/comment (and folds) exist, regardless of source ---
def _text_col(df, colname):
    if colname in df.columns:
        s = df[colname].fillna("").astype(str)
        if not colname.startswith("norm_"):
            return s.map(normalize_text)
        return s
    return pd.Series([""] * len(df))

def ensure_ctx_comment_and_folds(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    rule   = _text_col(df, "norm_rule")               if "norm_rule" in df.columns else _text_col(df, "rule")
    pos1   = _text_col(df, "norm_positive_example_1") if "norm_positive_example_1" in df.columns else _text_col(df, "positive_example_1")
    pos2   = _text_col(df, "norm_positive_example_2") if "norm_positive_example_2" in df.columns else _text_col(df, "positive_example_2")
    neg1   = _text_col(df, "norm_negative_example_1") if "norm_negative_example_1" in df.columns else _text_col(df, "negative_example_1")
    neg2   = _text_col(df, "norm_negative_example_2") if "norm_negative_example_2" in df.columns else _text_col(df, "negative_example_2")
    subr   = _text_col(df, "norm_subreddit")          if "norm_subreddit" in df.columns else _text_col(df, "subreddit")
    body   = _text_col(df, "norm_body")               if "norm_body" in df.columns else _text_col(df, "body")

    if "rule_context" not in df.columns:
        df["rule_context"] = (
            "[RULE] " + rule + "\n" +
            "[POS1] " + pos1 + "\n" +
            "[POS2] " + pos2 + "\n" +
            "[NEG1] " + neg1 + "\n" +
            "[NEG2] " + neg2 + "\n" +
            "[SUBREDDIT] " + subr
        )
    if "comment" not in df.columns:
        df["comment"] = "[COMMENT] " + body

    if is_train and "fold" not in df.columns and "rule_violation" in df.columns:
        y = df["rule_violation"].astype(int).values
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold = np.full(len(df), -1, dtype=int)
        for k, (_, va) in enumerate(skf.split(df, y)):
            fold[va] = k
        df["fold"] = fold

    return df

train = ensure_ctx_comment_and_folds(train, is_train=True)
test  = ensure_ctx_comment_and_folds(test,  is_train=False)
print("Columns now present:",
      "rule_context" in train.columns,
      "comment" in train.columns,
      "fold" in train.columns)

# -------------------------------
# Optional E5-based hard negatives (guarded by switch)
# -------------------------------
neg_pool = None
if USE_E5_NEG:
    EMB_TRAIN_PATH = os.path.join(ART_DIR_CANDIDATES[0], "emb_train_e5.npz")
    if os.path.exists(EMB_TRAIN_PATH):
        try:
            emb_train = np.load(EMB_TRAIN_PATH)
            fields = ["norm_rule","norm_positive_example_1","norm_positive_example_2",
                      "norm_negative_example_1","norm_negative_example_2","norm_subreddit"]
            comps = [emb_train[f] for f in fields if f in emb_train]
            rule_ctx_emb = np.mean(np.stack(comps, axis=0), axis=0)
            nbrs = NearestNeighbors(n_neighbors=51, metric="cosine", n_jobs=-1)
            nbrs.fit(rule_ctx_emb)
            _, inds = nbrs.kneighbors(rule_ctx_emb, n_neighbors=51, return_distance=True)
            neg_pool = [[int(j) for j in inds[i,1:].tolist()] for i in range(len(train))]
            print("Using E5 hard negatives.")
        except Exception as e:
            print("E5 negatives disabled due to error:", e)
            neg_pool = None
    else:
        print("E5 embeddings not found; falling back to in-batch negatives.")
else:
    print("E5 negatives skipped (FAST mode). Using in-batch swaps for negatives.")

# -------------------------------
# Dataset + collate with on-the-fly negatives (CPU-only in workers)
# -------------------------------
class IndexDataset(Dataset):
    def __init__(self, indices: List[int]): self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, i): return self.indices[i]

def make_pair_text(rule_ctx, comment): return rule_ctx + "\n" + comment

class PairBatcher:
    def __init__(self, tokenizer, n_neg=1, max_len=FAST_MAX_LEN):
        self.tok = tokenizer
        self.n_neg = n_neg
        self.max_len = max_len
    def __call__(self, batch_indices):
        pos_texts, neg_texts = [], []
        for idx in batch_indices:
            rc = train.iloc[idx]["rule_context"]
            cm = train.iloc[idx]["comment"]
            pos_texts.append(make_pair_text(rc, cm))
        if neg_pool is not None:
            for idx in batch_indices:
                cm = train.iloc[idx]["comment"]
                j = random.choice(neg_pool[idx])
                rc_neg = train.iloc[j]["rule_context"]
                neg_texts.append(make_pair_text(rc_neg, cm))
        else:
            rc_list = [train.iloc[i]["rule_context"] for i in batch_indices]
            rc_shift = rc_list[1:] + rc_list[:1]
            for i_idx, rc_neg in zip(batch_indices, rc_shift):
                cm = train.iloc[i_idx]["comment"]
                neg_texts.append(make_pair_text(rc_neg, cm))
        texts = pos_texts + neg_texts
        labels_np = np.array([1]*len(pos_texts) + [0]*len(neg_texts), dtype=np.float32)
        # keep on CPU (workers cannot touch CUDA)
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        labels = torch.from_numpy(labels_np)  # CPU tensor
        return enc, labels

# R-Drop & FGM (optional)
def rdrop_kl(p, q):
    p1 = torch.sigmoid(p).clamp(1e-6, 1-1e-6)
    q1 = torch.sigmoid(q).clamp(1e-6, 1-1e-6)
    P = torch.stack([p1, 1-p1], dim=-1)
    Q = torch.stack([q1, 1-q1], dim=-1)
    kl1 = torch.sum(P * (P.log() - Q.log()), dim=-1)
    kl2 = torch.sum(Q * (Q.log() - P.log()), dim=-1)
    return (kl1 + kl2).mean() * 0.5

class FGM:
    def __init__(self, model, emb_name='embeddings.word_embeddings', epsilon=FGM_EPS):
        self.model = model; self.epsilon = epsilon; self.backup = {}; self.emb_name = emb_name
    def attack(self):
        for n,p in self.model.named_parameters():
            if p.requires_grad and self.emb_name in n and p.grad is not None:
                self.backup[n] = p.data.clone()
                norm = torch.norm(p.grad)
                if norm != 0:
                    p.data.add_(self.epsilon * p.grad / norm)
    def restore(self):
        for n,p in self.model.named_parameters():
            if n in self.backup: p.data = self.backup[n]
        self.backup = {}

# -------------------------------
# Tokenizer & folds
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_DEBERTA_L, use_fast=True)

all_folds = sorted(train["fold"].unique().tolist())
folds_to_run = all_folds[:FAST_N_FOLDS] if FAST_N_FOLDS < len(all_folds) else all_folds
print("Running folds:", folds_to_run)

oof = np.zeros(len(train), dtype=float)
test_pred = np.zeros(len(test), dtype=float)

# Pre-tokenize test pair strings once (for faster inference)
test_texts = [make_pair_text(rc, cm) for rc, cm in zip(test["rule_context"], test["comment"])]
test_enc = tokenizer(test_texts, padding=True, truncation=True, max_length=FAST_MAX_LEN, return_tensors="pt")
# move test enc to device in main process
for k in list(test_enc.keys()):
    test_enc[k] = test_enc[k].to(DEVICE)

def infer_on_preenc(model, enc, batch=32):
    preds = []
    model.eval(); model.to(DEVICE)
    with torch.no_grad():
        for i in range(0, enc["input_ids"].shape[0], batch):
            batch_slice = {k: v[i:i+batch] for k, v in enc.items()}
            logits = model(**batch_slice).logits.view(-1)
            preds.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(preds)

# -------------------------------
# Train per fold (FAST)
# -------------------------------
if DEVICE == "cuda":
    for f in folds_to_run:
        print(f"\n========== FOLD {f} ==========")
        tr_idx = np.where(train["fold"].values != f)[0]
        va_idx = np.where(train["fold"].values == f)[0]

        ds_tr = IndexDataset(tr_idx.tolist())
        collate = PairBatcher(tokenizer, n_neg=1, max_len=FAST_MAX_LEN)
        dl_tr = DataLoader(
            ds_tr, batch_size=FAST_TRAIN_BATCH, shuffle=True, drop_last=True,
            collate_fn=collate, num_workers=2, pin_memory=True
        )

        va_texts = [make_pair_text(train.iloc[i]["rule_context"], train.iloc[i]["comment"]) for i in va_idx]
        va_labels = train.iloc[va_idx]["rule_violation"].values.astype(int)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR_DEBERTA_L, num_labels=1).to(DEVICE)

        optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        total_steps = FAST_EPOCHS * min(FAST_MAX_STEPS_EPOCH, math.ceil(len(dl_tr) / FAST_GRAD_ACCUM))
        warmup_steps = int(WARMUP_FRAC * total_steps)
        sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        # AMP v2 (no deprecation warnings)
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        autocast_ctx = torch.amp.autocast('cuda', enabled=True)

        bce = nn.BCEWithLogitsLoss()
        fgm = FGM(model) if USE_FGM else None

        best_auc = -1; best_state = None

        for epoch in range(1, FAST_EPOCHS+1):
            model.train(); epoch_loss = 0.0; step = 0
            for batch_i, (enc, labels) in enumerate(dl_tr, 1):
                # move to GPU here (main process)
                for k, v in list(enc.items()):
                    enc[k] = v.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                with autocast_ctx:
                    logits1 = model(**enc).logits.view(-1)
                    loss = bce(logits1, labels)
                    if USE_R_DROP:
                        logits2 = model(**enc).logits.view(-1)
                        loss = loss + R_DROP_LAMBDA * rdrop_kl(logits1, logits2)

                scaler.scale(loss / FAST_GRAD_ACCUM).backward()

                do_step = (batch_i % FAST_GRAD_ACCUM) == 0
                if do_step:
                    if fgm is not None:
                        fgm.attack()
                        with autocast_ctx:
                            loss_adv = bce(model(**enc).logits.view(-1), labels) * 0.5
                        scaler.scale(loss_adv).backward()
                        fgm.restore()

                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    scaler.step(optim); scaler.update()
                    optim.zero_grad(set_to_none=True); sched.step()
                    step += 1

                epoch_loss += loss.item()

                if step >= FAST_MAX_STEPS_EPOCH:
                    break

                if batch_i % 200 == 0:
                    print(f"Epoch {epoch} | step {step} | loss {epoch_loss/max(1,batch_i):.4f}")

            # quick eval on full fold val (after training epoch)
            model.eval()
            va_pred_epoch = []
            with torch.no_grad():
                for i in range(0, len(va_texts), 32):
                    encv = tokenizer(va_texts[i:i+32], padding=True, truncation=True,
                                     max_length=FAST_MAX_LEN, return_tensors="pt")
                    for k, v in list(encv.items()):
                        encv[k] = v.to(DEVICE, non_blocking=True)
                    logits = model(**encv).logits.view(-1)
                    va_pred_epoch.append(torch.sigmoid(logits).detach().cpu().numpy())
            va_pred_epoch = np.concatenate(va_pred_epoch)
            fold_auc = roc_auc_score(va_labels, va_pred_epoch)
            print(f"Epoch {epoch} FOLD {f} AUC: {fold_auc:.6f} | train_loss={epoch_loss/max(1,batch_i):.4f}")

            if fold_auc > best_auc:
                best_auc = fold_auc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        # restore best and recompute val for OOF
        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        va_pred = []
        with torch.no_grad():
            for i in range(0, len(va_texts), 32):
                encv = tokenizer(va_texts[i:i+32], padding=True, truncation=True,
                                 max_length=FAST_MAX_LEN, return_tensors="pt")
                for k, v in list(encv.items()):
                    encv[k] = v.to(DEVICE, non_blocking=True)
                logits = model(**encv).logits.view(-1)
                va_pred.append(torch.sigmoid(logits).detach().cpu().numpy())
        va_pred = np.concatenate(va_pred)

        # store OOF
        oof[va_idx] = va_pred
        print(f"FOLD {f} best AUC: {roc_auc_score(va_labels, oof[va_idx]):.6f}")

        # test preds (accumulate) — uses pretokenized test for speed
        test_pred += infer_on_preenc(model, test_enc, batch=32) / len(folds_to_run)

        del model; torch.cuda.empty_cache(); gc.collect()

    # Report
    valid_mask = oof > 0
    final_auc = roc_auc_score(train.loc[valid_mask, "rule_violation"].astype(int), oof[valid_mask])
    print(f"\n==== Model A (FAST) OOF AUC (on trained folds): {final_auc:.6f} ====")

    # Save predictions
    np.save(os.path.join(OUT_DIR, "modelA_oof_fast.npy"),  oof)
    np.save(os.path.join(OUT_DIR, "modelA_test_fast.npy"), test_pred)

    # Optional: quick calibration using Platt from Phase 1 hashing baseline (if saved)
    platt_path = os.path.join(ART_DIR_CANDIDATES[0], "platt_calibrator.pkl")
    if os.path.exists(platt_path):
        import joblib
        platt = joblib.load(platt_path)
        test_cal = platt.predict_proba(test_pred.reshape(-1,1))[:,1]
    else:
        test_cal = test_pred

    # Robust submission: use sample_submission if present, else build from test
    ss_path = None
    for d in COMP_DIR_CANDIDATES + [COMP_DIR]:
        cand = os.path.join(d, "sample_submission.csv")
        if os.path.exists(cand):
            ss_path = cand
            break

    if ss_path:
        sample_sub = pd.read_csv(ss_path)
        if "row_id" in sample_sub.columns and "row_id" in test.columns:
            # align order with test if needed
            sample_sub = test[["row_id"]].merge(sample_sub.drop(columns=["rule_violation"], errors="ignore"),
                                                on="row_id", how="left")
    else:
        if "row_id" not in test.columns:
            raise RuntimeError("Cannot build submission: `row_id` not found in test.")
        sample_sub = pd.DataFrame({"row_id": test["row_id"]})

    sample_sub["rule_violation"] = test_cal.clip(0,1)
    out_path = os.path.join(OUT_DIR, "submission_modelA_fast.csv")
    sample_sub.to_csv(out_path, index=False)
    print("Saved:", out_path)


# Choose ONE of these two:

# 1) Recommended: Hashing-SGD baseline (better OOF)
import shutil, os
src = "/kaggle/working/artifacts/submission_tfidf.csv"
dst = "/kaggle/working/submission.csv"
shutil.copy(src, dst)
print("Submission ready at:", dst)

# 2) Fast CE (if you want that instead)
# import shutil
# shutil.copy("/kaggle/working/modelA_debv3l_fast/submission_modelA_fast.csv",
#             "/kaggle/working/submission.csv")
# print("Submission ready at: /kaggle/working/submission.csv")




