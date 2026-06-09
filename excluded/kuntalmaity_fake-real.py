import os
import numpy as np 
import pandas as pd 
import os, re, unicodedata, glob
import re, unicodedata
import os, re, glob
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
!pip -q install sentence-transformers --upgrade

import os, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.linear_model import LogisticRegression





for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"
# Read CSV
train_df = pd.read_csv(train_path)


# Quick look at data
print("Shape:", train_df.shape)
print("Columns:", train_df.columns.tolist())
print("Null counts:\n", train_df.isnull().sum())
train_df.head()


TRAIN_DIR = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"

article_dirs = sorted([p for p in glob.glob(os.path.join(TRAIN_DIR, "article_*")) if os.path.isdir(p)])
print("Found article dirs:", len(article_dirs))
print("Sample dirs:", article_dirs[:5])


# Purpose: Map each id in train.csv to the corresponding article folder
def article_id_from_path(p):
    m = re.search(r"article_(\d+)$", p)
    return int(m.group(1)) if m else None

dir_index = {article_id_from_path(p): p for p in article_dirs}

path_by_id = {}
missing = []
for pid in train_df["id"].tolist():
    if pid in dir_index:
        path_by_id[pid] = dir_index[pid]
    else:
        candidate = os.path.join(TRAIN_DIR, f"article_{pid:04d}")
        if os.path.isdir(candidate):
            path_by_id[pid] = candidate
        else:
            missing.append(pid)

print("Mapped:", len(path_by_id), "Missing:", len(missing))
if missing:
    print("Missing examples:", missing[:10])


# Purpose: Define helpers to clean and read text files consistently
ZERO_WIDTH = "".join(["\u200b","\u200c","\u200d","\ufeff","\u2060","\u180e","\u2028","\u2029"])
ZW_RE = re.compile("[" + re.escape(ZERO_WIDTH) + "]")

def clean_text(s: str) -> str:
    if not isinstance(s, str): 
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = ZW_RE.sub("", s)                       # remove zero-width chars
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t\f\v]+", " ", s)          # collapse runs of spaces/tabs
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)     # tidy spaces around newlines
    return s.strip()

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return clean_text(f.read())



# Purpose: Read file_1.txt and file_2.txt for each pair id and set label (0 if file_1 is real, else 1)
rows = []
for _, r in train_df.iterrows():
    pid        = int(r["id"])
    real_tid   = int(r["real_text_id"])
    folder     = path_by_id[pid]
    file_1     = os.path.join(folder, "file_1.txt")
    file_2     = os.path.join(folder, "file_2.txt")
    if not (os.path.isfile(file_1) and os.path.isfile(file_2)):
        continue
    t1 = read_text_file(file_1)
    t2 = read_text_file(file_2)
    label = 0 if real_tid == 1 else 1
    rows.append({"id": pid, "text_1": t1, "text_2": t2, "label": label})

pairs_df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
print("pairs_df shape:", pairs_df.shape)
pairs_df.head(10)



# Purpose: Drop empty texts (if any) and compute simple stats
before = pairs_df.shape[0]
pairs_df = pairs_df[(pairs_df["text_1"].str.len() > 0) & (pairs_df["text_2"].str.len() > 0)].reset_index(drop=True)
after = pairs_df.shape[0]
print(f"Removed {before - after} empty-text pairs")

pairs_df["len_1"] = pairs_df["text_1"].str.split().apply(len)
pairs_df["len_2"] = pairs_df["text_2"].str.split().apply(len)
print(pairs_df[["len_1","len_2"]].describe().round(2))
print("Label distribution:\n", pairs_df["label"].value_counts())



# Purpose: Build one-row-per-text dataset with ground-truth flag y (1=real, 0=fake)
a = pairs_df[["id","text_1","label"]].rename(columns={"id":"pair_id"})
a["text"] = a["text_1"]; a["y"] = (a["label"] == 0).astype(int)

b = pairs_df[["id","text_2","label"]].rename(columns={"id":"pair_id"})
b["text"] = b["text_2"]; b["y"] = (b["label"] == 1).astype(int)

long_df = pd.concat([a[["pair_id","text","y"]], b[["pair_id","text","y"]]], ignore_index=True)
print("long_df shape:", long_df.shape)
long_df.head(3)



# Purpose: Persist cleaned datasets so later cells can load them directly
os.makedirs("/kaggle/working/clean", exist_ok=True)
pairs_df[["id","text_1","text_2","label"]].to_csv("/kaggle/working/clean/train_pairs_clean.csv", index=False)
long_df.to_csv("/kaggle/working/clean/train_long_clean.csv", index=False)
print("Saved:",
      "/kaggle/working/clean/train_pairs_clean.csv",
      "/kaggle/working/clean/train_long_clean.csv", sep="\n")



# Purpose: Visualize basic length distributions for a sanity check
import matplotlib.pyplot as plt

plt.figure()
pairs_df["len_1"].plot(kind="hist", bins=30, alpha=0.7)
plt.title("Text 1 length (words)")
plt.xlabel("words")
plt.show()

plt.figure()
pairs_df["len_2"].plot(kind="hist", bins=30, alpha=0.7)
plt.title("Text 2 length (words)")
plt.xlabel("words")
plt.show()



# Purpose: Load the cleaned per-text dataset produced earlier
clean_long_path = "/kaggle/working/clean/train_long_clean.csv"
long_df = pd.read_csv(clean_long_path)

# Expect: columns = ['pair_id','text','y']  where y=1 (real), y=0 (fake)
print(long_df.shape, long_df.columns.tolist())
long_df.head(3)


# Purpose: Convert per-text probabilities to pairwise predictions and compute accuracy
def pairwise_accuracy_from_probs(long_df: pd.DataFrame, probs: np.ndarray) -> float:
    df = long_df.copy()
    df["p"] = probs  # probability that this text is REAL
    
    # each pair has exactly two rows; pick the higher prob within each pair
    chosen = (
        df.sort_values(["pair_id","p"], ascending=[True, False])
          .groupby("pair_id")
          .head(1)[["pair_id","y"]]
          .rename(columns={"y":"is_real_chosen"})
          .reset_index(drop=True)
    )
    # if top row is real (y=1), it's a correct pick
    return float(chosen["is_real_chosen"].mean())


# Purpose: Vectorize text with a robust TF-IDF setup (char + word n-grams)
# (char_wb helps with misspellings; words capture semantics)
word_vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=250_000)
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, max_features=250_000)

X_word = word_vec.fit_transform(long_df["text"])
X_char = char_vec.fit_transform(long_df["text"])
X = sparse.hstack([X_word, X_char]).tocsr()

y = long_df["y"].values
groups = long_df["pair_id"].values  # to keep pairs in the same fold

print("Feature matrix:", X.shape)


# Purpose: Run CV with group-wise splitting (no leakage across a pair), collect OOF probs
try:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splitter = sgkf.split(X, y, groups=groups)
except Exception:
    # Fallback if StratifiedGroupKFold not available: GroupKFold (not stratified)
    splitter = GroupKFold(n_splits=5).split(X, y, groups=groups)

oof = np.zeros(len(long_df), dtype=float)
fold_models = []

for fold, (tr_idx, va_idx) in enumerate(splitter, 1):
    clf = LogisticRegression(
        C=2.0,           # a bit stronger than default
        penalty="l2",
        solver="liblinear",  # stable for sparse
        max_iter=200
    )
    clf.fit(X[tr_idx], y[tr_idx])
    oof[va_idx] = clf.predict_proba(X[va_idx])[:,1]
    fold_models.append(clf)
    acc = pairwise_accuracy_from_probs(long_df.iloc[va_idx].reset_index(drop=True),
                                       oof[va_idx])
    print(f"Fold {fold} pairwise ACC: {acc:.4f}")

cv_acc = pairwise_accuracy_from_probs(long_df, oof)
print(f"\nOOF Pairwise Accuracy: {cv_acc:.4f}")



# Purpose: Fit the final Logistic Regression on ALL training data
final_clf = LogisticRegression(
    C=2.0, penalty="l2", solver="liblinear", max_iter=200
)
final_clf.fit(X, y)
print("Final model trained on full data.")



# Purpose: Read test pairs (id, text_1, text_2) from folder structure
TEST_DIR = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"

def article_id_from_path(p):
    m = re.search(r"article_(\d+)$", p)
    return int(m.group(1)) if m else None

def read_text_clean(p):
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        t = f.read()
    # light clean compatible with earlier train cleaning
    t = t.replace("\r\n","\n").replace("\r","\n")
    t = re.sub(r"[ \t\f\v]+"," ", t)
    t = re.sub(r"[ \t]*\n[ \t]*","\n", t)
    return t.strip()

test_rows = []
for adir in sorted([p for p in glob.glob(os.path.join(TEST_DIR, "article_*")) if os.path.isdir(p)]):
    pid = article_id_from_path(adir)
    f1 = os.path.join(adir, "file_1.txt")
    f2 = os.path.join(adir, "file_2.txt")
    if not (os.path.isfile(f1) and os.path.isfile(f2)): 
        continue
    t1 = read_text_clean(f1)
    t2 = read_text_clean(f2)
    test_rows.append({"id": pid, "text_1": t1, "text_2": t2})

test_df = pd.DataFrame(test_rows).sort_values("id").reset_index(drop=True)
print("test_df shape:", test_df.shape)
test_df.head(10)



# Purpose: Transform test texts with the SAME vectorizers and predict final labels
# Build per-text frame matching training 'long_df' shape
test_long = pd.concat([
    pd.DataFrame({"pair_id": test_df["id"], "text": test_df["text_1"], "slot": 1}),
    pd.DataFrame({"pair_id": test_df["id"], "text": test_df["text_2"], "slot": 2})
], ignore_index=True)

Xw_te = word_vec.transform(test_long["text"])
Xc_te = char_vec.transform(test_long["text"])
X_te = sparse.hstack([Xw_te, Xc_te]).tocsr()

probs = final_clf.predict_proba(X_te)[:,1]
test_long["p"] = probs

# pick the higher prob within each pair; map back to label (0 if file_1 chosen else 1)
chosen = (test_long.sort_values(["pair_id","p"], ascending=[True, False])
                   .groupby("pair_id").head(1)[["pair_id","slot"]])

submission = chosen.copy()
submission["label"] = (submission["slot"] == 2).astype(int)
submission = submission[["pair_id","label"]].rename(columns={"pair_id":"id"}).sort_values("id").reset_index(drop=True)

print(submission.head())



# Purpose: Save the current Logistic Regression submission to disk
submission_path = "/kaggle/working/submission_lr.csv"
submission.to_csv(submission_path, index=False)
print("Saved:", submission_path)



# splitter (keep texts from the same pair together)
try:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splitter = sgkf.split(X, y, groups=groups)
except Exception:
    splitter = GroupKFold(n_splits=5).split(X, y, groups=groups)

oof_svm = np.zeros(len(long_df), dtype=float)
svm_models = []

for fold, (tr_idx, va_idx) in enumerate(splitter, 1):
    # LinearSVC + Platt scaling to get probabilities
    base = LinearSVC(C=1.0)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(X[tr_idx], y[tr_idx])
    oof_svm[va_idx] = clf.predict_proba(X[va_idx])[:,1]
    svm_models.append(clf)
    acc = pairwise_accuracy_from_probs(long_df.iloc[va_idx].reset_index(drop=True), oof_svm[va_idx])
    print(f"SVM Fold {fold} pairwise ACC: {acc:.4f}")

svm_oof_acc = pairwise_accuracy_from_probs(long_df, oof_svm)
print(f"\nSVM OOF Pairwise Accuracy: {svm_oof_acc:.4f}")

# Train final SVM on all data
base_final = LinearSVC(C=1.0)
svm_final = CalibratedClassifierCV(base_final, method="sigmoid", cv=5)
svm_final.fit(X, y)
print("Final SVM trained.")


# Purpose: Make a second submission using SVM only
probs_svm = svm_final.predict_proba(X_te)[:,1]
test_long_svm = test_long.copy()
test_long_svm["p"] = probs_svm

chosen_svm = (test_long_svm.sort_values(["pair_id","p"], ascending=[True, False])
                           .groupby("pair_id").head(1)[["pair_id","slot"]])

submission_svm = chosen_svm.copy()
submission_svm["label"] = (submission_svm["slot"] == 2).astype(int)
submission_svm = submission_svm[["pair_id","label"]].rename(columns={"pair_id":"id"}).sort_values("id").reset_index(drop=True)

submission_svm_path = "/kaggle/working/submission_svm.csv"
submission_svm.to_csv(submission_svm_path, index=False)
print("Saved:", submission_svm_path)



# Purpose: Average LR and SVM probabilities per text, then pick higher per pair
probs_lr  = probs                      # from  LR prediction cell
probs_svm = probs_svm                  # from SVM prediction cell

probs_ens = 0.5 * probs_lr + 0.5 * probs_svm
test_long_ens = test_long.copy()
test_long_ens["p"] = probs_ens

chosen_ens = (test_long_ens.sort_values(["pair_id","p"], ascending=[True, False])
                             .groupby("pair_id").head(1)[["pair_id","slot"]])

submission_ens = chosen_ens.copy()
submission_ens["label"] = (submission_ens["slot"] == 2).astype(int)
submission_ens = submission_ens[["pair_id","label"]].rename(columns={"pair_id":"id"}).sort_values("id").reset_index(drop=True)

submission_ens_path = "/kaggle/working/submission_ensemble.csv"
submission_ens.to_csv(submission_ens_path, index=False)
print("Saved:", submission_ens_path)



# Purpose: Inspect decision confidence per pair (margin between top/bottom probs)
def pair_margins_from_probs(long_df, probs):
    d = long_df.copy()
    d["p"] = probs
    g = d.groupby("pair_id")["p"].agg(["min","max"])
    g["margin"] = g["max"] - g["min"]
    return g.sort_values("margin")

margins_lr  = pair_margins_from_probs(long_df, oof)
margins_svm = pair_margins_from_probs(long_df, oof_svm)

print("Lowest-margin pairs (LR):")
print(margins_lr.head(10))
print("\nLowest-margin pairs (SVM):")
print(margins_svm.head(10))



# Purpose: Load the melted training data and initialize embedder
long_df = pd.read_csv("/kaggle/working/clean/train_long_clean.csv")  # ['pair_id','text','y']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast & strong

embedder = SentenceTransformer(EMB_MODEL_NAME, device=DEVICE)
embedder.max_seq_length = 512  # truncate safely for long docs
print("Device:", DEVICE, "| Model:", EMB_MODEL_NAME)
print(long_df.shape, long_df.columns.tolist())



# Purpose: Define a reusable function to embed a list/series of texts in batches
def sbert_encode(texts, batch_size=64, normalize=True):
    embs = embedder.encode(
        texts.tolist() if isinstance(texts, pd.Series) else texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=normalize,  # cosine-friendly
    )
    return embs



# Purpose: Embed the training texts once, cache to disk for speed
train_emb_path = "/kaggle/working/sbert_train_emb.npy"

if os.path.exists(train_emb_path):
    X_emb = np.load(train_emb_path)
    print("Loaded cached train embeddings:", X_emb.shape)
else:
    X_emb = sbert_encode(long_df["text"], batch_size=64, normalize=True)
    np.save(train_emb_path, X_emb)
    print("Computed & cached train embeddings:", X_emb.shape)

y = long_df["y"].values
groups = long_df["pair_id"].values



# Purpose: Cross-validate a simple classifier on SBERT embeddings and compute OOF pairwise accuracy
def pairwise_accuracy_from_probs(long_chunk: pd.DataFrame, probs: np.ndarray) -> float:
    df = long_chunk.copy()
    df["p"] = probs
    top = (df.sort_values(["pair_id","p"], ascending=[True, False])
             .groupby("pair_id").head(1)[["pair_id","y"]])
    return float(top["y"].mean())

try:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = sgkf.split(X_emb, y, groups=groups)
except Exception:
    splits = GroupKFold(n_splits=5).split(X_emb, y, groups=groups)

oof_sbert = np.zeros(len(long_df), dtype=float)

for fold, (tr, va) in enumerate(splits, 1):
    clf = LogisticRegression(
        C=2.0, solver="lbfgs", penalty="l2", max_iter=1000, n_jobs=-1
    )
    clf.fit(X_emb[tr], y[tr])
    oof_sbert[va] = clf.predict_proba(X_emb[va])[:,1]
    acc = pairwise_accuracy_from_probs(long_df.iloc[va].reset_index(drop=True), oof_sbert[va])
    print(f"SBERT fold {fold} pairwise ACC: {acc:.4f}")

oof_acc = pairwise_accuracy_from_probs(long_df, oof_sbert)
print(f"\nSBERT OOF pairwise ACC: {oof_acc:.4f}")



# Purpose: Fit final classifier on all SBERT embeddings
sbert_clf = LogisticRegression(C=2.0, solver="lbfgs", penalty="l2", max_iter=1000, n_jobs=-1)
sbert_clf.fit(X_emb, y)
print("Final SBERT classifier trained.")



# Purpose: Build per-text test frame (slot=1/2) from the already-loaded test_df (or rebuild if needed)
import glob, re

TEST_DIR = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"

def article_id_from_path(p):
    m = re.search(r"article_(\d+)$", p)
    return int(m.group(1)) if m else None

def read_text_clean(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        t = f.read()
    t = t.replace("\r\n","\n").replace("\r","\n")
    t = re.sub(r"[ \t\f\v]+"," ", t)
    t = re.sub(r"[ \t]*\n[ \t]*","\n", t)
    return t.strip()

# Build if not present already
if "test_df" not in globals():
    rows = []
    for adir in sorted([p for p in glob.glob(os.path.join(TEST_DIR, "article_*")) if os.path.isdir(p)]):
        pid = article_id_from_path(adir)
        f1, f2 = os.path.join(adir, "file_1.txt"), os.path.join(adir, "file_2.txt")
        if os.path.isfile(f1) and os.path.isfile(f2):
            rows.append({"id": pid, "text_1": read_text_clean(f1), "text_2": read_text_clean(f2)})
    test_df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    print("Built test_df:", test_df.shape)

test_long_sbert = pd.concat([
    pd.DataFrame({"pair_id": test_df["id"], "text": test_df["text_1"], "slot": 1}),
    pd.DataFrame({"pair_id": test_df["id"], "text": test_df["text_2"], "slot": 2}),
], ignore_index=True)
print("test_long_sbert:", test_long_sbert.shape)



# Purpose: Compute (or load) SBERT test embeddings and get per-text probabilities
test_emb_path = "/kaggle/working/sbert_test_emb.npy"

if os.path.exists(test_emb_path):
    X_te_emb = np.load(test_emb_path)
    print("Loaded cached test embeddings:", X_te_emb.shape)
else:
    X_te_emb = sbert_encode(test_long_sbert["text"], batch_size=64, normalize=True)
    np.save(test_emb_path, X_te_emb)
    print("Computed & cached test embeddings:", X_te_emb.shape)

probs_sbert = sbert_clf.predict_proba(X_te_emb)[:,1]
test_long_sbert["p"] = probs_sbert



# Purpose: Choose higher-prob text per pair, map to Kaggle label, and save
chosen = (test_long_sbert.sort_values(["pair_id","p"], ascending=[True, False])
                       .groupby("pair_id")
                       .head(1)[["pair_id","slot"]])

submission_sbert = chosen.copy()
submission_sbert["label"] = (submission_sbert["slot"] == 2).astype(int)
submission_sbert = (submission_sbert[["pair_id","label"]]
                    .rename(columns={"pair_id":"id"})
                    .sort_values("id")
                    .reset_index(drop=True))

path_sbert = "/kaggle/working/submission_sbert.csv"
submission_sbert.to_csv(path_sbert, index=False)
print("Saved:", path_sbert)
print(submission_sbert.head())



# Purpose: If you already have per-text probabilities from TF-IDF models (probs from LR/SVM),
#          combine them with SBERT probabilities via a simple average.

# Preconditions:
# - `test_long` with column 'p' from LR (TF-IDF), i.e., probs for each text row in test_long (slot 1/2).
# - Or `probs` variable holding LR probs aligned with test_long's rows.

if "test_long" in globals():
    # Align rows by (pair_id, slot)
    tl = test_long[["pair_id","slot"]].copy().reset_index(drop=True)
    tl["p_lr"] = probs  # from your LR TF-IDF prediction cell

    ts = test_long_sbert[["pair_id","slot","p"]].rename(columns={"p":"p_sbert"}).reset_index(drop=True)

    ens = tl.merge(ts, on=["pair_id","slot"], how="inner")
    ens["p_ens"] = 0.5*ens["p_lr"] + 0.5*ens["p_sbert"]

    ens_pred = (ens.sort_values(["pair_id","p_ens"], ascending=[True, False])
                   .groupby("pair_id").head(1)[["pair_id","slot"]])
    submission_ens = ens_pred.copy()
    submission_ens["label"] = (submission_ens["slot"] == 2).astype(int)
    submission_ens = (submission_ens[["pair_id","label"]]
                      .rename(columns={"pair_id":"id"})
                      .sort_values("id")
                      .reset_index(drop=True))
    path_ens = "/kaggle/working/submission_sbert_tfidf_ens.csv"
    submission_ens.to_csv(path_ens, index=False)
    print("Saved:", path_ens)
else:
    print("Note: define `test_long` + `probs` (LR TF-IDF) before running the ensemble cell.")



# Purpose: From any per-text probabilities you already have (e.g., LR/SVM/SBERT/ensemble),
#          pick the higher-prob text per *training* pair and compute accuracy vs real_text_id.

# Inputs expected:
#   - long_df: ['pair_id','text','y']  (already loaded)
#   - oof_best: numpy array of shape (len(long_df),) with per-text P(real) on TRAIN
#               (e.g., use your best OOF probs: TF-IDF LR, SVM, SBERT, or their blend)

# If you want to blend OOFs, do it here:
# oof_best = 0.6 * oof_sbert + 0.4 * oof   # example (SBERT + TF-IDF LR)

assert 'long_df' in globals(), "long_df not found"
assert 'oof' in globals() or 'oof_sbert' in globals() or 'oof_svm' in globals(), "Provide per-text OOF probs"

# Pick the best available OOF vector
oof_candidates = [v for v in [globals().get('oof'), globals().get('oof_sbert'), globals().get('oof_svm')] if v is not None]
if len(oof_candidates) == 1:
    oof_best = oof_candidates[0]
else:
    # Simple robust average if multiple exist
    oof_best = sum(oof_candidates) / len(oof_candidates)

# Chosen slot per pair_id using TRAIN OOF probs
tmp = long_df.copy()
tmp['p'] = oof_best
chosen_train = (tmp.sort_values(['pair_id','p'], ascending=[True, False])
                  .groupby('pair_id').head(1)[['pair_id']]
                  .assign(pred_real_text_id=lambda d: 1)  # placeholder; we'll fix below
               )

# We need to know which row (slot=1 or slot=2). Rebuild slots from original pairs_df
pairs_train = pd.read_csv("/kaggle/working/clean/train_pairs_clean.csv")  # id, text_1, text_2, label
left = tmp.reset_index(drop=True).copy()
# Recreate slot by matching text to text_1/text_2 (safe because lengths differ).
def which_slot(row):
    # quick heuristic: check which text matches better length-wise (exact match fastest if equal)
    t = row['text']
    pid = row['pair_id']
    p = pairs_train.loc[pairs_train['id']==pid].iloc[0]
    return 1 if len(t)==len(p['text_1']) else 2
# Faster: we annotated slots when building test_long; let's re-create for train:
long_train_slots = pd.concat([
    pd.DataFrame({'pair_id': pairs_train['id'], 'slot': 1, 'text': pairs_train['text_1']}),
    pd.DataFrame({'pair_id': pairs_train['id'], 'slot': 2, 'text': pairs_train['text_2']}),
], ignore_index=True)

tmp2 = tmp.merge(long_train_slots[['pair_id','text','slot']], on=['pair_id','text'], how='left')
chosen_train = (tmp2.sort_values(['pair_id','p'], ascending=[True, False])
                   .groupby('pair_id').head(1)[['pair_id','slot']])
chosen_train['pred_real_text_id'] = chosen_train['slot']  # 1 or 2
print(chosen_train.head())



# Purpose: Compute pairwise accuracy on TRAIN using the 1/2 encoding directly
gt = pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv")[['id','real_text_id']]
chk = chosen_train.merge(gt, left_on='pair_id', right_on='id', how='inner')
acc_train_id12 = (chk['pred_real_text_id'] == chk['real_text_id']).mean()
print(f"Sanity check â€” TRAIN accuracy using (1/2) encoding: {acc_train_id12:.4f}")



# Purpose: Create both possible encodings to eliminate label-format mismatch.

# 1) id + real_text_id in {1,2}
sub_id12 = (chosen_train[['pair_id','pred_real_text_id']]
            .rename(columns={'pair_id':'id','pred_real_text_id':'real_text_id'})
            .sort_values('id').reset_index(drop=True))
sub_id12.to_csv("/kaggle/working/submission_id12.csv", index=False)

# 2) id + label in {0,1}  (0 => file_1 real, 1 => file_2 real)
sub_01 = chosen_train[['pair_id','slot']].copy()
sub_01['label'] = (sub_01['slot']==2).astype(int)
sub_01 = sub_01[['pair_id','label']].rename(columns={'pair_id':'id'}).sort_values('id').reset_index(drop=True)
sub_01.to_csv("/kaggle/working/submission_01.csv", index=False)

print("Saved both:")
print("/kaggle/working/submission_id12.csv")
print("/kaggle/working/submission_01.csv")






# Purpose: Initialize Sentence-BERT embedder
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMB_MODEL, device=DEVICE)
embedder.max_seq_length = 512
print("Device:", DEVICE, "| Model:", EMB_MODEL)



# Purpose: Load per-text training dataset created during cleaning
long_df = pd.read_csv("/kaggle/working/clean/train_long_clean.csv")  # pair_id, text, y
y = long_df["y"].values
groups = long_df["pair_id"].values
print(long_df.shape, long_df.columns.tolist())



# Purpose: Split long docs into overlapping word-chunks for SBERT encoding
def chunk_text_words(text, max_words=350, stride=300):
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+max_words]))
        if i + max_words >= len(words): break
        i += stride
    return chunks

def sbert_encode_chunked(texts, batch_size=32, normalize=True, max_words=350, stride=300):
    out = []
    for t in texts:
        ch = chunk_text_words(t, max_words=max_words, stride=stride)
        embs = embedder.encode(ch, batch_size=batch_size, convert_to_numpy=True,
                               show_progress_bar=False, normalize_embeddings=normalize)
        out.append(embs.mean(axis=0))
    return np.vstack(out)



# Purpose: Encode training texts with chunking
X_emb = sbert_encode_chunked(long_df["text"], batch_size=32, normalize=True)
print("Train embeddings shape:", X_emb.shape)



# Purpose: Cross-validate SBERT-chunk + Logistic Regression on train data
try:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = sgkf.split(X_emb, y, groups=groups)
except Exception:
    splits = GroupKFold(n_splits=5).split(X_emb, y, groups=groups)

oof = np.zeros(len(long_df))
for fold, (tr, va) in enumerate(splits, 1):
    clf = LogisticRegression(C=2.0, solver="lbfgs", penalty="l2", max_iter=1000, n_jobs=-1)
    clf.fit(X_emb[tr], y[tr])
    oof[va] = clf.predict_proba(X_emb[va])[:,1]
    acc = (long_df.iloc[va].assign(p=oof[va])
           .sort_values(['pair_id','p'], ascending=[True, False])
           .groupby('pair_id').head(1)['y'].mean())
    print(f"Fold {fold} Pairwise ACC: {acc:.4f}")

acc_total = (long_df.assign(p=oof)
             .sort_values(['pair_id','p'], ascending=[True, False])
             .groupby('pair_id').head(1)['y'].mean())
print("OOF Pairwise Accuracy:", round(acc_total, 4))



# Purpose: Fit final classifier on ALL SBERT embeddings
final_clf = LogisticRegression(C=2.0, solver="lbfgs", penalty="l2", max_iter=1000, n_jobs=-1)
final_clf.fit(X_emb, y)
print("Final SBERT-chunk classifier trained.")



# Purpose: Read all test article folders into a dataframe with id, text_1, text_2
TEST_DIR = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"

def article_id_from_path(p):
    m = re.search(r"article_(\d+)$", p)
    return int(m.group(1)) if m else None

def read_text_clean(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f: t = f.read()
    t = t.replace("\r\n","\n").replace("\r","\n")
    t = re.sub(r"[ \t\f\v]+"," ", t)
    t = re.sub(r"[ \t]*\n[ \t]*","\n", t)
    return t.strip()

rows=[]
for adir in sorted([p for p in glob.glob(os.path.join(TEST_DIR,"article_*")) if os.path.isdir(p)]):
    pid = article_id_from_path(adir)
    f1, f2 = os.path.join(adir,"file_1.txt"), os.path.join(adir,"file_2.txt")
    if os.path.isfile(f1) and os.path.isfile(f2):
        rows.append({"id":pid,"text_1":read_text_clean(f1),"text_2":read_text_clean(f2)})
test_df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
print("test_df shape:", test_df.shape)



# Purpose: Reshape test into per-text format (slot=1 for file_1, slot=2 for file_2)
test_long = pd.concat([
    pd.DataFrame({"pair_id": test_df["id"], "text": test_df["text_1"], "slot": 1}),
    pd.DataFrame({"pair_id": test_df["id"], "text": test_df["text_2"], "slot": 2}),
], ignore_index=True)
print("test_long shape:", test_long.shape)



# Purpose: Embed test texts with chunking and predict probabilities
X_te_emb = sbert_encode_chunked(test_long["text"], batch_size=32, normalize=True)
probs = final_clf.predict_proba(X_te_emb)[:,1]
test_long["p"] = probs



# Purpose: Pick the higher prob text in each pair and output (id, real_text_id âˆˆ {1,2})
chosen = (test_long.sort_values(["pair_id","p"], ascending=[True, False])
                    .groupby("pair_id").head(1)[["pair_id","slot"]])
submission = (chosen.rename(columns={"pair_id":"id"})
                    .assign(real_text_id=lambda d: d["slot"])
                    .loc[:,["id","real_text_id"]]
                    .sort_values("id").reset_index(drop=True))
print(submission.head())



# Purpose: Save final SBERT-chunk submission
submission_path = "/kaggle/working/submission_chunk_id12.csv"
submission.to_csv(submission_path, index=False)
print("Saved:", submission_path)


