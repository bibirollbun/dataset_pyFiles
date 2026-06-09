! pip install transformers==4.57.1



import os
import gc
import re
import warnings
import logging
from tabulate import tabulate

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.sparse import hstack, csr_matrix

import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    RobertaTokenizer,
    RobertaModel,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from tqdm import tqdm

import time
from datetime import timedelta
import datasets

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ["LIGHTGBM_VERBOSE"] = "-1"
os.environ["LIGHTGBM_LOGGING_LEVEL"] = "fatal"
os.environ["LGBM_VERBOSE"] = "-1"

logging.getLogger("lightgbm").setLevel(logging.CRITICAL + 1)
logging.getLogger("lightgbm").handlers.clear()
logging.getLogger("lightgbm").propagate = False
logging.getLogger().setLevel(logging.CRITICAL)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


train=pd.read_csv("/kaggle/input/mercor-ai-detection/train.csv")
test=pd.read_csv("/kaggle/input/mercor-ai-detection/test.csv")
train.head()


train["is_cheating"].value_counts()


# =============================================
# CELL 2: Clean Text + TF-IDF (Word + Char)
# =============================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    return text

# Combine topic + answer
train['full_text'] = train['topic'] + ' [SEP] ' + train['answer'].apply(clean_text)
test['full_text'] = test['topic'] + ' [SEP] ' + test['answer'].apply(clean_text)

X = train['full_text']
y = train['is_cheating']
X_test = test['full_text']

# Word TF-IDF (1-3 grams)
word_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    stop_words='english',
    sublinear_tf=True
)
X_word = word_vectorizer.fit_transform(X)
X_test_word = word_vectorizer.transform(X_test)

# Char TF-IDF (2-5 grams)
char_vectorizer = TfidfVectorizer(
    max_features=4000,
    ngram_range=(2, 5),
    analyzer='char',
    lowercase=True
)
X_char = char_vectorizer.fit_transform(X)
X_test_char = char_vectorizer.transform(X_test)

# Combine features
X_combined = hstack([X_word, X_char])
X_test_combined = hstack([X_test_word, X_test_char])

print(f"Feature matrix: {X_combined.shape}")


# =============================================
# CELL 3: 10-Fold CV with 5 Models + Progress Tracking
# =============================================


n_folds = 20
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# OOF & Test prediction holders (only the 5 kept models)
lr_oof   = np.zeros(len(X));   mnb_oof = np.zeros(len(X))
cnb_oof  = np.zeros(len(X));   svm_oof = np.zeros(len(X))
lgb_oof  = np.zeros(len(X))

lr_test  = np.zeros(len(test)); mnb_test = np.zeros(len(test))
cnb_test = np.zeros(len(test)); svm_test = np.zeros(len(test))
lgb_test = np.zeros(len(test))

print(f"Starting {n_folds}-fold CV with 5 models (LR, MNB, CNB, SVM, LGB)...\n")

start_time = time.time()
fold_times = []

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_combined, y)):
    fold_start = time.time()
    
    X_tr, X_val      = X_combined[trn_idx], X_combined[val_idx]
    y_tr, y_val      = y.iloc[trn_idx], y.iloc[val_idx]
    X_tr_word, X_val_word = X_word[trn_idx], X_word[val_idx]
    
    # ---------- 1. Logistic Regression ----------
    lr = LogisticRegression(class_weight='balanced', max_iter=2000,
                            C=1.0, n_jobs=-1, random_state=42)
    lr.fit(X_tr, y_tr)
    lr_oof[val_idx] = lr.predict_proba(X_val)[:, 1]
    lr_test += lr.predict_proba(X_test_combined)[:, 1] / n_folds
    
    # ---------- 2. Multinomial NB ----------
    mnb = MultinomialNB(alpha=0.05)
    mnb.fit(X_tr_word, y_tr)
    mnb_oof[val_idx] = mnb.predict_proba(X_val_word)[:, 1]
    mnb_test += mnb.predict_proba(X_test_word)[:, 1] / n_folds
    
    # ---------- 3. Complement NB ----------
    cnb = ComplementNB(alpha=0.05)
    cnb.fit(X_tr_word, y_tr)
    cnb_oof[val_idx] = cnb.predict_proba(X_val_word)[:, 1]
    cnb_test += cnb.predict_proba(X_test_word)[:, 1] / n_folds
    
    # ---------- 4. Linear SVM (SGD) ----------
    svm = SGDClassifier(loss='modified_huber', class_weight='balanced',
                        max_iter=2000, learning_rate='adaptive',
                        eta0=0.1, random_state=42)
    svm.fit(X_tr, y_tr)
    svm_oof[val_idx] = svm.predict_proba(X_val)[:, 1]
    svm_test += svm.predict_proba(X_test_combined)[:, 1] / n_folds
    
    # ---------- 5. LightGBM ----------
    lgb_model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05,
                                   max_depth=6, subsample=0.8,
                                   colsample_bytree=0.8,
                                   class_weight='balanced',
                                   n_jobs=-1, metric='auc',
                                   verbose=-1, random_state=42)
    lgb_model.fit(X_tr, y_tr)
    lgb_oof[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    lgb_test += lgb_model.predict_proba(X_test_combined)[:, 1] / n_folds
    
    # ---------- Timing & Progress ----------
    fold_time = time.time() - fold_start
    fold_times.append(fold_time)
    avg_fold_time = np.mean(fold_times)
    remaining = n_folds - (fold + 1)
    eta_sec   = remaining * avg_fold_time
    eta_str   = str(timedelta(seconds=int(eta_sec)))
    
    prog = int(20 * (fold + 1) / n_folds)
    bar  = "█" * prog + " " * (20 - prog)
    
    print(f"Fold {fold+1:2d}/{n_folds} | Time: {fold_time:.1f}s | "
          f"ETA: {eta_str} | [{bar}] {100*(fold+1)/n_folds:5.1f}%")

# ---------- Ensemble OOF ----------
ensemble_oof = (0.30*lr_oof + 0.15*mnb_oof + 0.15*cnb_oof +
                0.20*svm_oof + 0.20*lgb_oof)

total_time = time.time() - start_time
print(f'\nFINAL {n_folds}-FOLD CV ROC-AUC: {roc_auc_score(y, ensemble_oof):.6f}')
print(f"Total CV Time: {str(timedelta(seconds=int(total_time)))}")


# =============================================
# CELL 4: Final Training on Full Data + Submission (5 Models)
# =============================================
print("\nTraining final 5 models on full dataset...")

# ---------- 1. Logistic Regression ----------
lr_full = LogisticRegression(
    class_weight='balanced', max_iter=2000, C=1.0,
    n_jobs=-1, random_state=42
)
lr_full.fit(X_combined, y)
lr_pred = lr_full.predict_proba(X_test_combined)[:, 1]

# ---------- 2. Multinomial NB ----------
mnb_full = MultinomialNB(alpha=0.05)
mnb_full.fit(X_word, y)
mnb_pred = mnb_full.predict_proba(X_test_word)[:, 1]

# ---------- 3. Complement NB ----------
cnb_full = ComplementNB(alpha=0.05)
cnb_full.fit(X_word, y)
cnb_pred = cnb_full.predict_proba(X_test_word)[:, 1]

# ---------- 4. Linear SVM (SGD) ----------
svm_full = SGDClassifier(
    loss='modified_huber', class_weight='balanced',
    max_iter=2000, learning_rate='adaptive', eta0=0.1,
    random_state=42
)
svm_full.fit(X_combined, y)
svm_pred = svm_full.predict_proba(X_test_combined)[:, 1]

# ---------- 5. LightGBM ----------
lgb_full = lgb.LGBMClassifier(
    n_estimators=2000, learning_rate=0.05,
    max_depth=6, subsample=0.8, colsample_bytree=0.8,
    class_weight='balanced', n_jobs=-1,
    metric='auc', verbose=-1, random_state=42
)
lgb_full.fit(X_combined, y)
lgb_pred = lgb_full.predict_proba(X_test_combined)[:, 1]

# ---------- Final Ensemble (weights sum to 1.0) ----------
final_probs = (
    0.30 * lr_pred +   # Strongest on sparse TF-IDF
    0.15 * mnb_pred +
    0.15 * cnb_pred +
    0.20 * svm_pred +
    0.20 * lgb_pred
)

# Clip for safety (avoid log(0) in evaluation)
final_probs = np.clip(final_probs, 1e-5, 1 - 1e-5)

# ---------- Create Submission ----------
submission = pd.DataFrame({
    'id': test['id'],
    'is_cheating': final_probs
})

print("\nSUBMISSION PREVIEW:")
print(submission.head())
print(f"\nMin prob: {final_probs.min():.6f} | Max prob: {final_probs.max():.6f} | Mean prob: {final_probs.mean():.6f}")

submission.to_csv('ml_submission.csv', index=False)
print("\nsubmission.csv SAVED! Ready for upload.")


# ------------------- 1. Load Data -------------------
train = pd.read_csv("/kaggle/input/mercor-ai-detection/train.csv")
test = pd.read_csv("/kaggle/input/mercor-ai-detection/test.csv")
sub_ids = test["id"].copy()

print(f"Train: {train.shape} | Test: {test.shape} | Cheating ratio: {train.is_cheating.mean():.4f}")

# ------------------- 2. Create Text -------------------
train["text"] = train["topic"] + " [SEP] " + train["answer"].fillna("")
test["text"] = test["topic"] + " [SEP] " + test["answer"].fillna("")

train = train[["text", "is_cheating"]]
test = test[["text"]]

# ------------------- 3. NO-LEAK SPLIT (Critical!) -------------------
def make_key(text, label):
    clean = re.sub(r'\s+', '', text.lower()[:240])
    return f"{clean}_{int(label)}"

train["key"] = train.apply(lambda row: make_key(row["text"], row["is_cheating"]), axis=1)
unique_keys = train["key"].unique()
np.random.seed(42)
val_keys = np.random.choice(unique_keys, size=int(0.2 * len(unique_keys)), replace=False)

val_mask = train["key"].isin(val_keys)
train_df = train[~val_mask].reset_index(drop=True)
val_df = train[val_mask].reset_index(drop=True)

print(f"Train size: {len(train_df)} | Val size: {len(val_df)} | Val cheating: {val_df.is_cheating.mean():.4f}")

# ------------------- 4. Tokenization -------------------
MODEL = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_attention_mask=True
    )

# Convert to Dataset
train_ds = Dataset.from_pandas(train_df[["text", "is_cheating"]])
val_ds = Dataset.from_pandas(val_df[["text", "is_cheating"]])
test_ds = Dataset.from_pandas(test)

# Rename label
train_ds = train_ds.rename_column("is_cheating", "labels")
val_ds = val_ds.rename_column("is_cheating", "labels")

# Tokenize
train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
test_ds = test_ds.map(tokenize_fn, batched=True, remove_columns=["text"])


train_ds = train_ds.cast_column("labels", datasets.Value("float32"))
val_ds = val_ds.cast_column("labels", datasets.Value("float32"))

# Set torch format
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask"])

# ------------------- 5. Model & Training -------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)

args = TrainingArguments(
    output_dir="mercor_deberta",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=True,
    logging_steps=20,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="auc",
    greater_is_better=True,
    report_to="none",
    seed=42,
    dataloader_num_workers=2,
)

def compute_auc(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy().ravel()
    return {"auc": roc_auc_score(labels, probs)}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_auc,
)

# ------------------- 6. TRAIN -------------------
print("STARTING TRAINING...")
trainer.train()
print("TRAINING DONE!")

# ------------------- 7. VALIDATION AUC (REAL) -------------------
val_pred = trainer.predict(val_ds)
val_logits = val_pred.predictions
val_labels = val_pred.label_ids
val_probs = torch.sigmoid(torch.tensor(val_logits)).numpy().ravel()
val_auc = roc_auc_score(val_labels, val_probs)
print(f"REAL VALIDATION AUC: {val_auc:.6f}")

# ------------------- 8. TEST PREDICTION -------------------
test_pred = trainer.predict(test_ds)
test_logits = test_pred.predictions
probs = torch.sigmoid(torch.tensor(test_logits)).numpy().ravel()

# ------------------- 9. SUBMISSION -------------------
submission = pd.DataFrame({
    "id": sub_ids,
    "is_cheating": probs.clip(1e-7, 1 - 1e-7)  # Prevent 0.0 or 1.0
})
submission.to_csv("hf_submission.csv", index=False)
print("submission.csv SAVED!")
display(submission.head(10))
print(f"Mean prob: {probs.mean():.4f} | Min: {probs.min():.6f} | Max: {probs.max():.6f}")



train=pd.read_csv("/kaggle/input/mercor-ai-detection/train.csv")
print(f"Train Data Shape: {train.shape}")
print(f"Train Data INFO: {train.info()}")
print(f"Train Data Check NULL: {train.isnull().sum()}")

train.drop(columns=["id"],axis=1,inplace=True)
train["text"]=train["topic"]+' [SEP] '+train["answer"]
train.drop(columns=["topic","answer"],axis=1,inplace=True)
print("\n")
print("Display Train DATA:\n")
display(train.head())
print("#"*150)

test=pd.read_csv("/kaggle/input/mercor-ai-detection/test.csv")
Id=test["id"]
test.drop(columns=["id"],axis=1,inplace=True)
test["text"]=test["topic"]+' [SEP] '+test["answer"]
test.drop(columns=["answer","topic"],axis=1,inplace=True)
print(f"Test Data Shape: {test.shape}")
print(f"Test Data INFO: {test.info()}")
print(f"Test Data Check NULL: {test.isnull().sum()}")
print("\n")
print("Display Test DATA:\n")
display(test.head())
print("#"*150)


def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'\n+', ' ', text)  # Normalize newlines
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = text.lower().strip()
    return text
    
train["text"]=train["text"].apply(clean_text)
    
test["text"]=test["text"].apply(clean_text)


tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased', use_fast=True)

def tokenize(batch):
    return tokenizer(batch["text"],truncation=True,padding="max_length",max_length=512,return_attention_mask=True)


hf_train = Dataset.from_pandas(train[["text", "is_cheating"]].astype({"is_cheating": "float32"}))
hf_train = hf_train.rename_column("is_cheating", "labels")

hf_train = hf_train.map(tokenize, batched=True, remove_columns=["text"])
hf_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(np.zeros(len(hf_train)), hf_train["labels"]))
train_ds = hf_train.select(train_idx)
val_ds = hf_train.select(val_idx)

model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=1)
model.config.pad_token_id = tokenizer.pad_token_id



training_args = TrainingArguments(
    output_dir="./roberta_out",
    overwrite_output_dir=True,
    num_train_epochs=30,
    per_device_train_batch_size=4,      # small data → small batch
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=10,
    weight_decay=0.01,
    fp16=True,
    logging_steps=1,                    # see every step
    eval_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="auc",
    greater_is_better=True,
    report_to="none",
    seed=42,
    
    save_total_limit=1,           # ← Only keep best model
    load_best_model_at_end=True

)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy().ravel()
    return {"auc": roc_auc_score(labels, probs)}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("STARTING TRAINING...")
trainer.train()
print("TRAINING DONE!")


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch

# Get predictions on validation set
val_preds_output = trainer.predict(val_ds)
val_logits = torch.from_numpy(val_preds_output.predictions).squeeze()
val_labels = val_preds_output.label_ids

# Apply sigmoid since it's binary
val_probs = torch.sigmoid(val_logits).numpy()
val_preds = (val_probs >= 0.5).astype(int)

# Metrics
accuracy = accuracy_score(val_labels, val_preds)
f1 = f1_score(val_labels, val_preds)
precision = precision_score(val_labels, val_preds)
recall = recall_score(val_labels, val_preds)
roc_auc = roc_auc_score(val_labels, val_probs)

print(f"Accuracy:  {accuracy:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")



# Tokenize test
test_hf = Dataset.from_pandas(test[["text"]])
test_hf = test_hf.map(tokenize, batched=True, remove_columns=["text"])
test_hf.set_format("torch", columns=["input_ids", "attention_mask"])

# Predict
pred_result = trainer.predict(test_hf)
probs = torch.sigmoid(torch.from_numpy(pred_result.predictions)).numpy().ravel()

# Submission
submission = pd.DataFrame({"id": Id,"is_cheating": probs})
submission.to_csv("dl_submission.csv", index=False)
print("submission.csv saved!")
submission.head()




