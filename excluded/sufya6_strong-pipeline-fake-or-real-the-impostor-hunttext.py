
# == Setup & imports
import os, re, math, random, string, unicodedata, statistics, gc, time
from collections import Counter
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

# tqdm optional
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

SEED = 42
random.seed(SEED); np.random.seed(SEED)

# Paths (Kaggle layout)
KAGGLE_TRAIN_DIR = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
KAGGLE_TEST_DIR  = "/kaggle/input/fake-or-real-the-impostor-hunt/data/test"
KAGGLE_TRAIN_CSV = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"

LOCAL_DATA_DIR = None  # set this if running locally, e.g. r"C:\\data\\fake-or-real-impostor-hunt\\data"




# == Data loading helpers
def read_texts_from_dir(dir_path):
    data = []
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(dir_path)
    for folder_name in sorted(os.listdir(dir_path)):
        folder_path = os.path.join(dir_path, folder_name)
        if os.path.isdir(folder_path):
            try:
                with open(os.path.join(folder_path, "file_1.txt"), "r", encoding="utf-8") as f1:
                    t1 = f1.read().strip()
                with open(os.path.join(folder_path, "file_2.txt"), "r", encoding="utf-8") as f2:
                    t2 = f2.read().strip()
                # extract trailing number as id (robust)
                m = re.findall(r"(\\d+)$", folder_name)
                idx = int(m[0]) if m else len(data)
                data.append((idx, t1, t2))
            except Exception as e:
                print("WARN", folder_name, e)
    df = pd.DataFrame(data, columns=["id","file_1","file_2"]).set_index("id").sort_index()
    return df

def load_competition_data():
    if os.path.isdir(KAGGLE_TRAIN_DIR) and os.path.isdir(KAGGLE_TEST_DIR):
        train_dir, test_dir, train_csv = KAGGLE_TRAIN_DIR, KAGGLE_TEST_DIR, KAGGLE_TRAIN_CSV
    elif LOCAL_DATA_DIR is not None:
        train_dir = os.path.join(LOCAL_DATA_DIR, "train")
        test_dir  = os.path.join(LOCAL_DATA_DIR, "test")
        train_csv = os.path.join(LOCAL_DATA_DIR, "train.csv")
    else:
        raise RuntimeError("No data found. Set LOCAL_DATA_DIR if not on Kaggle.")
    print("Loading:", train_dir, test_dir, train_csv)
    df_train = read_texts_from_dir(train_dir)
    df_test  = read_texts_from_dir(test_dir)
    df_train_gt = pd.read_csv(train_csv).set_index("id").sort_index()
    assert len(df_train) == len(df_train_gt), "train pairs vs labels mismatch"
    return df_train, df_train_gt, df_test

df_train, df_train_gt, df_test = load_competition_data()
print("Train pairs:", len(df_train), "Test pairs:", len(df_test))
display(df_train.head())
display(df_train_gt.head())




# == Feature engineering (interpretable signals)
_WORD_RE = re.compile(r"[A-Za-zÃ€-Ã¿]+", re.UNICODE)

def is_latin_char(c):
    try:
        return "LATIN" in unicodedata.name(c)
    except Exception:
        return False

def proportion_english_by_chunks(text, n=10):
    # safe wrapper around langdetect (if available). We'll use a token-chunk approach.
    try:
        from langdetect import detect
    except Exception:
        detect = None
    toks = re.split(r"\\s+", (text or "").strip())
    if not toks:
        return 0.0
    chunks = [" ".join(toks[i:i+n]) for i in range(0, len(toks), n)]
    english = 0
    for ch in chunks:
        if detect:
            try:
                if detect(ch) == "en":
                    english += 1
            except Exception:
                pass
    return english / max(1, len(chunks))

def char_script_ratios(text):
    if not text:
        return 0.0, 0.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0, 0.0
    latin = sum(1 for c in chars if is_latin_char(c))
    ascii_ = sum(1 for c in chars if ord(c) < 128)
    total = len(chars)
    return latin/total, ascii_/total

def char_entropy(text):
    if not text:
        return 0.0
    cnt = Counter(text)
    total = sum(cnt.values())
    return -sum((v/total) * math.log(v/total, 2) for v in cnt.values() if v>0)

def vocab_richness(text):
    words = _WORD_RE.findall((text or "").lower())
    if not words:
        return 0.0
    return len(set(words))/len(words)

def avg_word_len(text):
    words = _WORD_RE.findall(text or "")
    if not words:
        return 0.0
    return statistics.mean(len(w) for w in words)

def extract_features(text):
    latin, ascii_ = char_script_ratios(text)
    return {
        "len_chars": len(text or ""),
        "len_words": len(re.split(r"\\s+", (text or "").strip())),
        "english_prop": proportion_english_by_chunks(text),
        "lat_ratio": latin,
        "ascii_ratio": ascii_,
        "entropy": char_entropy(text),
        "vocab_richness": vocab_richness(text),
        "avg_word_len": avg_word_len(text),
        "punct_ratio": sum(1 for c in (text or "") if c in string.punctuation) / max(1, sum(1 for c in (text or "") if not c.isspace())),
        "digit_ratio": sum(1 for c in (text or "") if c.isdigit()) / max(1, sum(1 for c in (text or "") if not c.isspace())),
        "uppercase_ratio": sum(1 for c in (text or "") if c.isupper()) / max(1, sum(1 for c in (text or "") if c.isalpha())),
    }

# build feature frames (one row per text)
def build_feature_df(df_pairs):
    rows = []
    for pid, row in tqdm(df_pairs.iterrows(), total=len(df_pairs)):
        for which in (1,2):
            txt = row[f"file_{which}"]
            feats = extract_features(txt)
            feats.update({"pair_id": pid, "which": which, "text": txt})
            rows.append(feats)
    return pd.DataFrame(rows)

train_feat_df = build_feature_df(df_train)
test_feat_df  = build_feature_df(df_test)
display(train_feat_df.head())



# == TF-IDF (word + char) + Logistic Regression pipeline
# We'll train a single-text classifier (y=1 if text is real). At prediction time, pick the text with higher "real" prob per pair.

# Make single-text training/test frame
def make_single_text_frame(df_pairs, labels=None):
    rows = []
    for pid, row in df_pairs.iterrows():
        real = None
        if labels is not None and pid in labels.index:
            real = labels.loc[pid, "real_text_id"]

        for which in (1,2):
            txt = row[f"file_{which}"]
            if real is None:   # test set â†’ no ground truth
                y = None
            else:
                y = 1 if which == real else 0
            rows.append({"pair_id": pid, "which": which, "text": txt, "y": y})
    return pd.DataFrame(rows)

# Build train and test frames
df_train_single = make_single_text_frame(df_train, df_train_gt)
df_test_single  = make_single_text_frame(df_test)   # no labels

# Vectorizers
word_vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000, strip_accents="unicode")
char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, max_features=50000, strip_accents="unicode")

# Fit-transform train
X_word = word_vec.fit_transform(df_train_single["text"].values)
X_char = char_vec.fit_transform(df_train_single["text"].values)

from scipy.sparse import hstack
X_full = hstack([X_word, X_char])
y_full = df_train_single["y"].values.astype(int)  # ensure numeric

# Quick fit (full-train) to get a sense
SEED = 42
clf = LogisticRegression(C=4.0, max_iter=400, solver="liblinear", random_state=SEED)
clf.fit(X_full, y_full)
train_preds = clf.predict(X_full)
print("Quick full-train accuracy (single-text):", accuracy_score(y_full, train_preds))

# Pair-level evaluation on train (choose higher probability in each pair)
proba = clf.predict_proba(X_full)[:,1]
df_train_single["proba"] = proba
best = df_train_single.sort_values(["pair_id","proba"], ascending=[True, False]).groupby("pair_id").head(1)
pair_preds = best["which"].tolist()
pair_true  = df_train_gt.loc[best["pair_id"], "real_text_id"].tolist()
print("Pair-level accuracy (full-train):", accuracy_score(pair_true, pair_preds))

# Later we can transform test data with vectorizers and run predictions




# == Grouped CV for reliable estimate (char+word TF-IDF)
gkf = GroupKFold(n_splits=5)
pairs = df_train_single["pair_id"].values
X = X_full
y = y_full
oof_proba = np.zeros(len(y), dtype=float)
pair_accs = []

for fold, (tr, va) in enumerate(gkf.split(X, y, groups=pairs), 1):
    print("-- Fold", fold)
    X_tr = X[tr]
    X_va = X[va]
    y_tr = y[tr]
    model = LogisticRegression(C=4.0, max_iter=400, solver="liblinear", random_state=SEED)
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_va)[:,1]
    oof_proba[va] = p
    # pair-level
    va_df = df_train_single.iloc[va].copy()
    va_df["proba"] = p
    best = va_df.sort_values(["pair_id","proba"], ascending=[True, False]).groupby("pair_id").head(1)
    pred = best["which"].tolist()
    true = df_train_gt.loc[best["pair_id"], "real_text_id"].tolist()
    acc = accuracy_score(true, pred)
    pair_accs.append(acc)
    print("Fold pair-accuracy:", acc)

print("Mean pair-level CV acc:", np.mean(pair_accs), "std:", np.std(pair_accs))




# == Train a small feature-based model and CV it (same GroupKFold strategy)
feat_cols = ["len_chars","len_words","english_prop","lat_ratio","ascii_ratio","entropy","vocab_richness","avg_word_len","punct_ratio","digit_ratio","uppercase_ratio"]
Xf = train_feat_df[feat_cols].values
yf = train_feat_df["which"].apply(lambda w: 1 if ((w==1) and (df_train_gt.loc[train_feat_df.loc[train_feat_df.index[0],"pair_id"],"real_text_id"]==1)) else 0) if False else None
# The proper way: build a feature single-text frame aligned with df_train_single
train_feat_single = pd.DataFrame(train_feat_df[feat_cols].values, columns=feat_cols)
train_feat_single["pair_id"] = train_feat_df["pair_id"].values
train_feat_single["which"] = train_feat_df["which"].values
# Map y from df_train_single (they are aligned order: pair_id, which)
# We will reconstruct y aligned to train_feat_single:
y_map = {}
tmp = df_train_single[["pair_id","which","y"]].copy()
for _, r in tmp.iterrows():
    y_map[(r["pair_id"], r["which"])] = r["y"]
train_feat_single["y"] = train_feat_single.apply(lambda r: y_map[(r["pair_id"], r["which"])], axis=1)

from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
oof_feat = np.zeros(len(train_feat_single), dtype=float)
pair_accs_feat = []
for fold, (tr, va) in enumerate(gkf.split(train_feat_single[feat_cols].values, train_feat_single["y"].values, groups=train_feat_single["pair_id"].values), 1):
    print("Feat fold", fold)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_feat_single.loc[tr, feat_cols].values)
    Xva = scaler.transform(train_feat_single.loc[va, feat_cols].values)
    clf_f = LogisticRegression(C=2.0, max_iter=400, solver="liblinear", random_state=SEED)
    clf_f.fit(Xtr, train_feat_single.loc[tr, "y"].values)
    p = clf_f.predict_proba(Xva)[:,1]
    oof_feat[va] = p
    # pair-level
    va_df = train_feat_single.iloc[va].copy()
    va_df["proba"] = p
    best = va_df.sort_values(["pair_id","proba"], ascending=[True, False]).groupby("pair_id").head(1)
    pred = best["which"].tolist()
    true = df_train_gt.loc[best["pair_id"], "real_text_id"].tolist()
    acc = accuracy_score(true, pred)
    pair_accs_feat.append(acc)
    print("Feat pair-acc:", acc)

print("Mean feature model pair-acc:", np.mean(pair_accs_feat))




# == Simple ensemble: average TF-IDF prob + feature prob (aligned by single-text rows)
# We have oof_proba (tfidf) and oof_feat (feature model) aligned to df_train_single and train_feat_single respectively.
# To keep things simple, we'll re-train models on full train and produce test probs, then average.

# Fit TF-IDF on full train (again) and get test probs
from scipy.sparse import hstack
word_vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000, strip_accents="unicode")
char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, max_features=50000, strip_accents="unicode")
Xw = word_vec.fit_transform(df_train_single["text"].values)
Xc = char_vec.fit_transform(df_train_single["text"].values)
X_train_full = hstack([Xw, Xc])
clf_full = LogisticRegression(C=4.0, max_iter=400, solver="liblinear", random_state=SEED)
clf_full.fit(X_train_full, df_train_single["y"].values)

# Test transforms
Xw_test = word_vec.transform(df_test_single["text"].values)
Xc_test = char_vec.transform(df_test_single["text"].values)
X_test_full = hstack([Xw_test, Xc_test])
proba_tfidf_test = clf_full.predict_proba(X_test_full)[:,1]

# Feature model on full train
feat_cols = ["len_chars","len_words","english_prop","lat_ratio","ascii_ratio","entropy","vocab_richness","avg_word_len","punct_ratio","digit_ratio","uppercase_ratio"]
scaler_final = StandardScaler()
Xf_full = scaler_final.fit_transform(train_feat_single[feat_cols].values)
clf_feat_full = LogisticRegression(C=2.0, max_iter=400, solver="liblinear", random_state=SEED).fit(Xf_full, train_feat_single["y"].values)

# Feature test set (aligned to df_test_single order)
test_feat_single = pd.DataFrame(test_feat_df[feat_cols].values, columns=feat_cols)
test_feat_single["pair_id"] = test_feat_df["pair_id"].values
test_feat_single["which"] = test_feat_df["which"].values
Xf_test = scaler_final.transform(test_feat_single[feat_cols].values)
proba_feat_test = clf_feat_full.predict_proba(Xf_test)[:,1]

# Ensemble
proba_ens_test = 0.55*proba_tfidf_test + 0.45*proba_feat_test  # slight weight to TF-IDF

# Choose best per pair
test_pred_df = df_test_single.copy()
test_pred_df["proba"] = proba_ens_test
best = test_pred_df.sort_values(["pair_id","proba"], ascending=[True, False]).groupby("pair_id").head(1)
submission = best[["pair_id","which"]].rename(columns={"pair_id":"id","which":"real_text_id"}).sort_values("id")
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", submission.shape)
display(submission.head())



# OPTIONAL: Hugging Face fine-tune (robust to different transformers versions)
# Run only if internet is enabled and you want to fine-tune.
# Install if you want (may be optional on Kaggle):
# !pip install -q transformers datasets evaluate accelerate

import inspect
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

model_name = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(model_name)

# Prepare train dataframe: drop rows without labels and make 'labels' column (Trainer expects 'labels')
train_df = df_train_single.dropna(subset=["y"]).copy()
train_df["labels"] = train_df["y"].astype(int)

# Build HF datasets
ds_train = Dataset.from_pandas(train_df[["text","labels"]])
ds_test  = Dataset.from_pandas(df_test_single[["text"]])  # test has no labels

# Tokenizer function
def tokenize_fn(batch):
    return tok(batch["text"], truncation=True, padding="max_length", max_length=256)

# Map tokenization (batched)
ds_train = ds_train.map(tokenize_fn, batched=True, remove_columns=["text"])
ds_test  = ds_test.map(tokenize_fn, batched=True, remove_columns=["text"])

# Make sure datasets use torch tensors (Trainer works best)
ds_train.set_format(type="torch")
ds_test.set_format(type="torch")

# Train/validation split
ds_train = ds_train.train_test_split(test_size=0.1, seed=SEED)

# Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Build TrainingArguments kwargs only with supported keys
ta_kwargs = dict(
    output_dir="hf_out",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    save_strategy="no",
    fp16=False,
)
sig = inspect.signature(TrainingArguments)
if "evaluation_strategy" in sig.parameters:
    ta_kwargs["evaluation_strategy"] = "epoch"
if "logging_strategy" in sig.parameters:
    ta_kwargs["logging_strategy"] = "epoch"
# add other safe keys only if present (example)
if "load_best_model_at_end" in sig.parameters:
    ta_kwargs["load_best_model_at_end"] = False

args = TrainingArguments(**ta_kwargs)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == labels).mean()}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_train["train"],
    eval_dataset=ds_train["test"],
    tokenizer=tok,
    compute_metrics=compute_metrics,
)

trainer.train()

# Predict on test set
preds_out = trainer.predict(ds_test)
logits = preds_out.predictions
# convert logits -> probabilities robustly
try:
    # multi-class logits -> softmax
    from scipy.special import softmax
    probs = softmax(logits, axis=1)[:, 1]
except Exception:
    # fallback: if logits shape (N,1) -> sigmoid
    if logits.ndim == 2 and logits.shape[1] == 1:
        probs = 1 / (1 + np.exp(-logits.ravel()))
    else:
        # last-resort: take argmax and convert to 0/1 prob (not ideal)
        preds_arg = np.argmax(logits, axis=1)
        probs = preds_arg.astype(float)

# Map probs back to df_test_single order and create submission
test_pred_df = df_test_single.copy().reset_index(drop=True)
if len(probs) != len(test_pred_df):
    raise RuntimeError(f"Length mismatch: probs={len(probs)} vs test rows={len(test_pred_df)}")

test_pred_df["proba"] = probs
best = (test_pred_df.sort_values(["pair_id","proba"], ascending=[True, False])
                       .groupby("pair_id").head(1))
submission = best[["pair_id","which"]].rename(columns={"pair_id":"id","which":"real_text_id"}).sort_values("id")
submission.to_csv("submission_hf.csv", index=False)
print("Saved submission_hf.csv with shape:", submission.shape)
display(submission.head())


