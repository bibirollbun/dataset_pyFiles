# Full pipeline: Fine-tune Fakespot + Desklib, Benford+engineered features, VAL-based stacking ensemble, save final submission.
# NOTE: heavy. Adjust MAX_LEN_*, batch sizes, and epochs if OOM.

# ---------- Environment & installs ----------
import os, gc, sys, math, time, random
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from transformers import (
    AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel,
    AutoModelForSequenceClassification, Trainer, TrainingArguments,
    DataCollatorWithPadding, EarlyStoppingCallback
)

# ---------- Repro & device ----------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------- Paths & model ids ----------
DATA_DIR = "/kaggle/input/mercor-ai-detection"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")

FS_MODEL   = "fakespot-ai/roberta-base-ai-text-detection-v1"
DSK_MODEL  = "desklib/ai-text-detector-v1.01"

OUTPUT_DIR = "./ensemble_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- T4-friendly hyperparameters (adjust if OOM) ----------
MAX_LEN_FS  = 256
MAX_LEN_DSK = 256

BATCH_FS = 1
ACCUM_FS = 8
BATCH_DSK = 1
ACCUM_DSK = 8

EPOCHS_FS = 4
EPOCHS_DSK = 4

LR = 2e-5
FP16 = True

# ---------- helpers ----------
def free_cuda(names=[]):
    g = globals()
    for n in names:
        if n in g:
            try:
                del g[n]
            except Exception:
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def clean_text(s):
    if pd.isna(s): return ""
    return " ".join(str(s).strip().replace("\r"," ").replace("\n"," ").split())

def make_input(topic, answer):
    return f"TOPIC: {clean_text(topic)}\n\nANSWER: {clean_text(answer)}"

def plot_dist(scores, labels, title):
    df = pd.DataFrame({"score": scores, "label": labels})
    plt.figure(figsize=(7,3.5))
    sns.histplot(data=df, x="score", hue="label", bins=60, stat="density", common_norm=False, kde=True)
    plt.title(title); plt.tight_layout(); plt.show()

# ---------- Load data ----------
print("Loading CSVs...")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)
required = {"topic","answer","is_cheating"}
if not required.issubset(train_df.columns):
    raise ValueError(f"train.csv must contain columns {required}")
if "id" not in test_df.columns:
    raise ValueError("test.csv must contain 'id' column")

train_df["text"] = train_df.apply(lambda r: make_input(r["topic"], r["answer"]), axis=1)
train_df["label"] = train_df["is_cheating"].astype(int)
test_df["text"] = test_df.apply(lambda r: make_input(r.get("topic",""), r.get("answer","")), axis=1)

# Upsample minority label 0 to match majority (keeps your previous behavior)
orig_counts = train_df["label"].value_counts().to_dict()
print("Original class counts:", orig_counts)
minor_label = 0
minor_df = train_df[train_df["label"]==minor_label]
if len(minor_df) == 0:
    raise ValueError("No samples with label 0 to upsample.")
major_count = train_df["label"].value_counts().max()
if len(minor_df) < major_count:
    extra = minor_df.sample(n=(major_count - len(minor_df)), replace=True, random_state=SEED)
    train_df = pd.concat([train_df, extra], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
print("After upsample class counts:", train_df["label"].value_counts().to_dict())

# Train/val split stratified
train_part, val_part = train_test_split(train_df, test_size=0.20, random_state=SEED, stratify=train_df["label"])
print("Train rows:", len(train_part), "Val rows:", len(val_part))

# =========================
# 1) FAKESPOT full-encoder
# =========================
print("\n=== Fine-tune Fakespot (full encoder) ===")
tokenizer_fs = AutoTokenizer.from_pretrained(FS_MODEL, use_fast=True)
config_fs = AutoConfig.from_pretrained(FS_MODEL, num_labels=2)
model_fs = AutoModelForSequenceClassification.from_pretrained(FS_MODEL, config=config_fs)

# modest ANN head
hidden_size = model_fs.config.hidden_size
class ANNHead(nn.Module):
    def __init__(self, hidden_size, inner_dim=512, dropout_prob=0.25):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, inner_dim)
        self.act = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(inner_dim, max(128, inner_dim//2))
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.out = nn.Linear(max(128, inner_dim//2), 2)
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.constant_(self.fc1.bias,0.0)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.constant_(self.fc2.bias,0.0)
        nn.init.xavier_uniform_(self.out.weight); nn.init.constant_(self.out.bias,0.0)
    def forward(self, x):
        if x.dim() == 3:
            x = x[:,0,:]
        x = self.fc1(x); x = self.act(x); x = self.dropout1(x)
        x = self.fc2(x); x = self.act2(x); x = self.dropout2(x)
        return self.out(x)

model_fs.classifier = ANNHead(hidden_size, inner_dim=512)
model_fs.config.use_cache = False
try:
    model_fs.base_model.gradient_checkpointing_enable()
except Exception:
    pass
model_fs.to(device)

def tokenize_fs(batch):
    return tokenizer_fs(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN_FS)

fs_train_ds = Dataset.from_pandas(train_part[["text","label"]].reset_index(drop=True)).map(tokenize_fs, batched=True, remove_columns=["text"])
fs_val_ds   = Dataset.from_pandas(val_part[["text","label"]].reset_index(drop=True)).map(tokenize_fs, batched=True, remove_columns=["text"])
fs_train_ds = fs_train_ds.rename_column("label","labels"); fs_val_ds = fs_val_ds.rename_column("label","labels")
fs_train_ds.set_format(type="torch"); fs_val_ds.set_format(type="torch")

def compute_metrics_fs(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:,1]
    return {"roc_auc": roc_auc_score(labels, probs)}

training_args_fs = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR,"fakespot_full"),
    per_device_train_batch_size=BATCH_FS,
    per_device_eval_batch_size=BATCH_FS,
    gradient_accumulation_steps=ACCUM_FS,
    num_train_epochs=EPOCHS_FS,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    learning_rate=LR,
    weight_decay=0.01,
    fp16=FP16,
    seed=SEED,
    logging_steps=50,
    save_total_limit=2,
    report_to=[]
)

trainer_fs = Trainer(
    model=model_fs,
    args=training_args_fs,
    train_dataset=fs_train_ds,
    eval_dataset=fs_val_ds,
    tokenizer=tokenizer_fs,
    data_collator=DataCollatorWithPadding(tokenizer_fs),
    compute_metrics=compute_metrics_fs,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("Training Fakespot...")
trainer_fs.train()
trainer_fs.save_model(os.path.join(OUTPUT_DIR,"fakespot_full_best"))
tokenizer_fs.save_pretrained(os.path.join(OUTPUT_DIR,"fakespot_full_best"))

# predict on validation and test
val_fs_ds_full = Dataset.from_pandas(val_part[["text","label"]].reset_index(drop=True)).map(tokenize_fs, batched=True, remove_columns=["text"]).rename_column("label","labels")
val_fs_ds_full.set_format(type="torch")
test_ds_fs = Dataset.from_pandas(test_df[["id","text"]].rename(columns={"id":"orig_id"})).map(tokenize_fs, batched=True, remove_columns=["text"])
test_ds_fs.set_format(type="torch")

print("Predicting with Fakespot on VAL and TEST...")
preds_val_fs = trainer_fs.predict(val_fs_ds_full).predictions
probs_val_fs = torch.softmax(torch.from_numpy(preds_val_fs), dim=1).numpy()[:,1]
preds_test_fs = trainer_fs.predict(test_ds_fs).predictions
probs_test_fs = torch.softmax(torch.from_numpy(preds_test_fs), dim=1).numpy()[:,1]

# free larger objects related to Fakespot model/trainer to release memory (we'll keep probs)
free_cuda(['trainer_fs','model_fs'])
print("Freed Fakespot trainer/model from memory.")

# =========================
# 2) DESKLIB full-encoder
# =========================
print("\n=== Fine-tune Desklib (full encoder) ===")
tokenizer_dsk = AutoTokenizer.from_pretrained(DSK_MODEL, use_fast=True)
config_dsk = AutoConfig.from_pretrained(DSK_MODEL)

class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig
    def __init__(self, config):
        super().__init__(config)
        # Note: using AutoModel.from_config here (random init but fine-tuned)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())
        if loss is not None:
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

model_dsk = DesklibAIDetectionModel.from_pretrained(DSK_MODEL, config=config_dsk)
model_dsk.to(device)

def tokenize_dsk(batch):
    return tokenizer_dsk(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN_DSK)

dsk_train_ds = Dataset.from_pandas(train_part[["text","label"]].reset_index(drop=True)).map(tokenize_dsk, batched=True, remove_columns=["text"])
dsk_val_ds   = Dataset.from_pandas(val_part[["text","label"]].reset_index(drop=True)).map(tokenize_dsk, batched=True, remove_columns=["text"])
dsk_train_ds = dsk_train_ds.rename_column("label","labels"); dsk_val_ds = dsk_val_ds.rename_column("label","labels")
dsk_train_ds.set_format(type="torch"); dsk_val_ds.set_format(type="torch")

def compute_metrics_dsk(eval_pred):
    logits, labels = eval_pred
    arr = logits
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
    if arr.ndim==2 and arr.shape[1]==1:
        probs = torch.sigmoid(arr.squeeze()).numpy()
    else:
        probs = torch.sigmoid(arr).numpy()
    return {"roc_auc": roc_auc_score(labels, probs)}

training_args_dsk = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR,"desklib_full"),
    per_device_train_batch_size=BATCH_DSK,
    per_device_eval_batch_size=BATCH_DSK,
    gradient_accumulation_steps=ACCUM_DSK,
    num_train_epochs=EPOCHS_DSK,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    learning_rate=LR,
    weight_decay=0.01,
    fp16=FP16,
    seed=SEED,
    logging_steps=50,
    save_total_limit=2,
    report_to=[]
)

trainer_dsk = Trainer(
    model=model_dsk,
    args=training_args_dsk,
    train_dataset=dsk_train_ds,
    eval_dataset=dsk_val_ds,
    tokenizer=tokenizer_dsk,
    data_collator=DataCollatorWithPadding(tokenizer_dsk),
    compute_metrics=compute_metrics_dsk,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("Training Desklib...")
trainer_dsk.train()
trainer_dsk.save_model(os.path.join(OUTPUT_DIR,"desklib_full_best"))
tokenizer_dsk.save_pretrained(os.path.join(OUTPUT_DIR,"desklib_full_best"))

# Predict VAL & TEST with Desklib
val_dsk_ds_full = Dataset.from_pandas(val_part[["text","label"]].reset_index(drop=True)).map(tokenize_dsk, batched=True, remove_columns=["text"]).rename_column("label","labels")
val_dsk_ds_full.set_format(type="torch")
test_ds_dsk = Dataset.from_pandas(test_df[["id","text"]].rename(columns={"id":"orig_id"})).map(tokenize_dsk, batched=True, remove_columns=["text"])
test_ds_dsk.set_format(type="torch")

print("Predicting with Desklib on VAL and TEST...")
preds_val_dsk = trainer_dsk.predict(val_dsk_ds_full).predictions
preds_val_dsk = np.asarray(preds_val_dsk)
if preds_val_dsk.ndim==2 and preds_val_dsk.shape[1]==1:
    probs_val_dsk = torch.sigmoid(torch.from_numpy(preds_val_dsk).squeeze()).numpy()
else:
    probs_val_dsk = torch.sigmoid(torch.from_numpy(preds_val_dsk)).numpy()

preds_test_dsk = trainer_dsk.predict(test_ds_dsk).predictions
preds_test_dsk = np.asarray(preds_test_dsk)
if preds_test_dsk.ndim==2 and preds_test_dsk.shape[1]==1:
    probs_test_dsk = torch.sigmoid(torch.from_numpy(preds_test_dsk).squeeze()).numpy()
else:
    probs_test_dsk = torch.sigmoid(torch.from_numpy(preds_test_dsk)).numpy()

free_cuda(['trainer_dsk','model_dsk'])
print("Freed Desklib trainer/model from memory.")

# ================
# 3) ENGINEER FEATURES
# ================
print("\n=== Building Benford + enhanced features ===")

class BenfordAnalyzer:
    def __init__(self):
        self.benford_dist = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
    @staticmethod
    def extract_first_digit(value):
        if value <= 0: return 1
        return int(str(int(abs(value)))[0])
    def calculate_benford_metrics(self, sequence):
        if len(sequence) == 0:
            return 0.0, 0.0, 0.0
        if len(sequence) > 2000:
            sequence = np.random.choice(sequence, 2000, replace=False)
        first_digits = np.array([self.extract_first_digit(x) for x in sequence if x > 0])
        if len(first_digits) == 0:
            return 0.0, 0.0, 0.0
        observed_dist = np.zeros(9)
        for digit in range(1,10):
            observed_dist[digit-1] = np.sum(first_digits == digit) / len(first_digits)
        expected_counts = self.benford_dist * len(first_digits)
        observed_counts = observed_dist * len(first_digits)
        chi_square = np.sum((observed_counts - expected_counts)**2 / (expected_counts + 1e-8))
        kl_div = np.sum(observed_dist * np.log(observed_dist / (self.benford_dist + 1e-8) + 1e-8))
        entropy = -np.sum(observed_dist * np.log(observed_dist + 1e-8))
        return float(chi_square), float(kl_div), float(entropy)
    def extract_benford_features(self, df, text_col='answer'):
        rows = []
        for text in df[text_col].fillna("").astype(str):
            words = text.split()
            word_lengths = np.array([len(w) for w in words])
            char_counts = np.array([len(w) for w in words])
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            sentence_lengths = np.array([len(s.split()) for s in sentences]) if len(sentences)>0 else np.array([])
            punct_positions = np.array([i for i,c in enumerate(text) if c in '.,;:!?'])
            chi_word, kl_word, ent_word = self.calculate_benford_metrics(word_lengths)
            chi_char, kl_char, ent_char = self.calculate_benford_metrics(char_counts)
            chi_sent, kl_sent, ent_sent = self.calculate_benford_metrics(sentence_lengths)
            chi_punct, kl_punct, ent_punct = self.calculate_benford_metrics(punct_positions)
            rows.append({
                'benford_chi_word': chi_word,
                'benford_kl_word': kl_word,
                'benford_entropy_word': ent_word,
                'benford_chi_char': chi_char,
                'benford_kl_char': kl_char,
                'benford_entropy_char': ent_char,
                'benford_chi_sent': chi_sent,
                'benford_kl_sent': kl_sent,
                'benford_entropy_sent': ent_sent,
                'benford_chi_punct': chi_punct,
                'benford_kl_punct': kl_punct,
                'benford_entropy_punct': ent_punct,
            })
        return pd.DataFrame(rows)

def enhanced_ai_features(df, text_col='answer'):
    features = pd.DataFrame(index=df.index)
    text_series = df[text_col].fillna("").astype(str)
    features['text_length'] = text_series.str.len()
    features['word_count'] = text_series.str.split().str.len().fillna(0).astype(int)
    features['avg_word_length'] = features['text_length'] / (features['word_count'] + 1)
    features['sentence_count'] = text_series.str.count(r'[.!?]+')
    features['avg_sentence_length'] = features['word_count'] / (features['sentence_count'] + 1)
    features['sentence_length_variance'] = text_series.apply(
        lambda x: np.var([len(s.split()) for s in re.split(r'[.!?]+', str(x)) if s.strip()]) if len(re.split(r'[.!?]+', str(x)))>1 else 0
    )
    features['comma_count'] = text_series.str.count(',')
    features['period_count'] = text_series.str.count(r'\.')
    features['unique_words'] = text_series.apply(lambda x: len(set(str(x).lower().split())))
    features['ttr'] = features['unique_words'] / (features['word_count'] + 1)
    ai_connectors = ['in conclusion', 'in summary', 'furthermore', 'moreover', 'additionally', 'however', 'therefore', 'thus']
    features['ai_connector_density'] = text_series.apply(lambda x: sum(1 for p in ai_connectors if p in x.lower()) / (len(str(x).split())+1))
    features['total_punctuation'] = features['comma_count'] + features['period_count']
    features['punctuation_ratio'] = features['total_punctuation'] / (features['text_length'] + 1)
    features['repeated_words'] = text_series.apply(lambda x: len([w for w,c in pd.Series(str(x).lower().split()).value_counts().items() if c>1]))
    features['capital_letters'] = text_series.apply(lambda x: sum(1 for c in str(x) if c.isupper()))
    features['capital_ratio'] = features['capital_letters'] / (features['text_length'] + 1)
    if 'topic' in df.columns:
        topic_dummies = pd.get_dummies(df['topic'].fillna(''), prefix='topic')
        features = pd.concat([features, topic_dummies], axis=1)
    return features.fillna(0.0)

benford = BenfordAnalyzer()
print("Computing features on train_part/val_part/test...")
train_meta_feats = pd.concat([enhanced_ai_features(train_part), benford.extract_benford_features(train_part)], axis=1).reset_index(drop=True).fillna(0.0)
val_meta_feats   = pd.concat([enhanced_ai_features(val_part),   benford.extract_benford_features(val_part)], axis=1).reset_index(drop=True).fillna(0.0)
test_meta_feats  = pd.concat([enhanced_ai_features(test_df),    benford.extract_benford_features(test_df)], axis=1).reset_index(drop=True).fillna(0.0)

# Choose a compact subset to avoid overfitting at meta-level (toggle if you want all)
use_all_meta = False
if not use_all_meta:
    keep_cols = [
        'text_length','word_count','avg_word_length','sentence_count','avg_sentence_length',
        'sentence_length_variance','punctuation_ratio','unique_words','ttr','ai_connector_density',
        'benford_chi_word','benford_kl_word','benford_entropy_word',
        'benford_chi_char','benford_kl_char'
    ]
    keep_cols = [c for c in keep_cols if c in train_meta_feats.columns]
    train_meta_reduced = train_meta_feats[keep_cols].copy()
    val_meta_reduced   = val_meta_feats[keep_cols].copy()
    test_meta_reduced  = test_meta_feats[keep_cols].copy()
else:
    train_meta_reduced = train_meta_feats.copy()
    val_meta_reduced   = val_meta_feats.copy()
    test_meta_reduced  = test_meta_feats.copy()

print("Meta features prepared. Count:", train_meta_reduced.shape[1])

# =========================
# 4) BUILD META-FEATURES (VAL-only meta training)
# =========================
print("\n=== Building meta-features and training VAL-based stacker ===")

# --- Robust shape alignment to avoid ValueError during stacking ---
n_val = len(probs_val_fs)
n_test = len(probs_test_fs)

if val_meta_reduced.shape[0] != n_val:
    print(f"Warning: VAL meta rows ({val_meta_reduced.shape[0]}) != VAL probs ({n_val}). Aligning by truncation.")
    val_meta_used = val_meta_reduced.iloc[:n_val].reset_index(drop=True)
else:
    val_meta_used = val_meta_reduced.reset_index(drop=True)

if test_meta_reduced.shape[0] != n_test:
    print(f"Warning: TEST meta rows ({test_meta_reduced.shape[0]}) != TEST probs ({n_test}). Aligning by truncation.")
    test_meta_used = test_meta_reduced.iloc[:n_test].reset_index(drop=True)
else:
    test_meta_used = test_meta_reduced.reset_index(drop=True)

# Combine model probs (VAL) + meta features (VAL) to build meta-training set
X_val_meta = np.column_stack([
    probs_val_fs.reshape(-1, 1),
    probs_val_dsk.reshape(-1, 1),
    val_meta_used.values,
])

X_test_meta = np.column_stack([
    probs_test_fs.reshape(-1, 1),
    probs_test_dsk.reshape(-1, 1),
    test_meta_used.values,
])

y_val = val_part['label'].astype(int).values

# Baseline per-model AUCs on VAL
auc_val_fs = roc_auc_score(y_val, probs_val_fs)
auc_val_dsk = roc_auc_score(y_val, probs_val_dsk)
print(f"VAL AUCs — Fakespot: {auc_val_fs:.4f}, Desklib: {auc_val_dsk:.4f}")

# Train logistic stacking meta-model on VAL meta-features ONLY
meta_clf = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=2000)
meta_clf.fit(X_val_meta, y_val)
meta_val_raw = meta_clf.predict_proba(X_val_meta)[:,1]
auc_meta_val = roc_auc_score(y_val, meta_val_raw)
print(f"Stacking (LogReg) VAL AUC: {auc_meta_val:.4f}")

# Optional isotonic calibration on VAL
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(meta_val_raw, y_val)
meta_val_iso = iso.predict(meta_val_raw)
auc_meta_val_iso = roc_auc_score(y_val, meta_val_iso)
print(f"Stacking + isotonic VAL AUC: {auc_meta_val_iso:.4f}")

# Choose best (raw vs iso)
if auc_meta_val_iso >= auc_meta_val:
    selected_mode = "stack_logreg_iso"
    print("Selected stacking with isotonic calibration (better VAL AUC).")
else:
    selected_mode = "stack_logreg"
    print("Selected stacking raw (better VAL AUC).")

# Produce final test probabilities using selected meta-model
if selected_mode == "stack_logreg_iso":
    test_meta_raw = meta_clf.predict_proba(X_test_meta)[:,1]
    final_test_probs = iso.predict(test_meta_raw)
else:
    final_test_probs = meta_clf.predict_proba(X_test_meta)[:,1]

final_test_probs = np.clip(final_test_probs, 0.0, 1.0)

# Save final ensembled submission (only this file required)
submission = pd.DataFrame({"id": test_df["id"].tolist(), "is_cheating": final_test_probs})
out_name = os.path.join(OUTPUT_DIR, f"final_submission_stack_val_{selected_mode}.csv")
submission.to_csv(out_name, index=False)
print(f"\nSaved final submission to: {out_name}")

# Save per-model test probs for inspection (optional)
pd.DataFrame({"id": test_df["id"].tolist(), "fakespot": probs_test_fs}).to_csv(os.path.join(OUTPUT_DIR,"test_probs_fakespot.csv"), index=False)
pd.DataFrame({"id": test_df["id"].tolist(), "desklib": probs_test_dsk}).to_csv(os.path.join(OUTPUT_DIR,"test_probs_desklib.csv"), index=False)

# Quick diagnostics
print("\nFinal ensemble selection:", selected_mode)
print("VAL AUCs summary:")
print(f"  Fakespot on VAL: {auc_val_fs:.4f}")
print(f"  Desklib  on VAL: {auc_val_dsk:.4f}")
print(f"  Selected stack VAL AUC (selected): {max(auc_meta_val, auc_meta_val_iso):.4f}")

# Optional: display distribution plots
try:
    y_val_full = val_part['label'].astype(int).values
    plot_dist(probs_val_fs, y_val_full, f"Fakespot VAL dist (AUC={auc_val_fs:.4f})")
    plot_dist(probs_val_dsk, y_val_full, f"Desklib VAL dist (AUC={auc_val_dsk:.4f})")
    if selected_mode == "stack_logreg_iso":
        plot_dist(iso.predict(meta_val_raw), y_val_full, f"Stack (iso) VAL dist (AUC={auc_meta_val_iso:.4f})")
    else:
        plot_dist(meta_val_raw, y_val_full, f"Stack VAL dist (AUC={auc_meta_val:.4f})")
except Exception:
    pass

print("\nAll done. Only final ensembled submission saved (plus per-model debug files).")

