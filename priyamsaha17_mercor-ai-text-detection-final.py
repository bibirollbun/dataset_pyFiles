# Full pipeline: Fine-tune Fakespot + Desklib (full-encoder), build Benford+engineered features + TF-IDF,
# include transformer probabilities as features, train Stratified K-Fold ML ensemble (LGB/XGB/Cat/LR),
# compute OOF ensemble, optimize weights, optional pseudo-labeling & calibration, save final submission.
# NOTE: heavy. Adjust MAX_LEN, batch sizes, accum steps, and epochs if OOM.

# ---------- 0) ENV & IMPORTS ----------
import os, gc, sys, time, random, re
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from datasets import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
)

# ---------- 0.5) Repro & device ----------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------- 1) PATHS & MODEL IDS ----------
DATA_DIR = "/kaggle/input/mercor-ai-detection"    # adjust if different
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")

FS_MODEL   = "fakespot-ai/roberta-base-ai-text-detection-v1"
DSK_MODEL  = "desklib/ai-text-detector-v1.01"

OUTPUT_DIR = "./ensemble_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 2) HYPERPARAMETERS ----------
# Token lengths & T4-friendly sizes
MAX_LEN_FS  = 256
MAX_LEN_DSK = 256

# Transformer training
BATCH_FS = 1
ACCUM_FS = 8
EPOCHS_FS = 4

BATCH_DSK = 1
ACCUM_DSK = 8
EPOCHS_DSK = 4

LR = 2e-5
FP16 = True

# ML stacking
N_SPLITS = 10

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

# ---------- 3) LOAD DATA ----------
print("Loading data...")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

required = {"topic","answer","is_cheating"}
if not required.issubset(train_df.columns):
    raise ValueError(f"train.csv must contain columns {required}")
if "id" not in test_df.columns:
    raise ValueError("test.csv must contain 'id' column")

# Build text inputs for transformers
train_df["text"] = train_df.apply(lambda r: make_input(r["topic"], r["answer"]), axis=1)
train_df["label"] = train_df["is_cheating"].astype(int)
test_df["text"] = test_df.apply(lambda r: make_input(r.get("topic",""), r.get("answer","")), axis=1)

# keep a copy of the original train for engineered features + stacking later
train_df_orig = train_df.copy()

# Optional: upsample minority label 0 to match majority like earlier (if desired)
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

# We'll fine-tune FS and DSK on this upsampled train_df, but for stacking we want features aligned
# with the ORIGINAL train_df_orig (so transformer probs must be computed on train_df_orig).

# ---------- 4) TRAIN / FINE-TUNE FAKESPOT ----------
print("\n=== Fine-tune Fakespot (full encoder) ===")
tokenizer_fs = AutoTokenizer.from_pretrained(FS_MODEL, use_fast=True)
config_fs = AutoConfig.from_pretrained(FS_MODEL, num_labels=2)
model_fs = AutoModelForSequenceClassification.from_pretrained(FS_MODEL, config=config_fs)

# Replace classifier head with small ANN (as you used)
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

# create dataset objects on upsampled train_df for training
fs_train_ds = Dataset.from_pandas(
    train_df[["text","label"]].reset_index(drop=True)
).map(tokenize_fs, batched=True, remove_columns=["text"])
fs_train_ds = fs_train_ds.rename_column("label","labels")
fs_train_ds.set_format(type="torch")

# small validation split from upsampled train for transformer fine-tuning stability
tt_train, tt_val = train_test_split(
    train_df, test_size=0.05, random_state=SEED, stratify=train_df["label"]
)
fs_tt_ds = Dataset.from_pandas(
    tt_train[["text","label"]].reset_index(drop=True)
).map(tokenize_fs, batched=True, remove_columns=["text"]).rename_column("label","labels")
fs_tt_ds.set_format(type="torch")
fs_val_ds = Dataset.from_pandas(
    tt_val[["text","label"]].reset_index(drop=True)
).map(tokenize_fs, batched=True, remove_columns=["text"]).rename_column("label","labels")
fs_val_ds.set_format(type="torch")

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
    eval_steps=2000,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    learning_rate=LR,
    weight_decay=0.01,
    fp16=FP16,
    seed=SEED,
    logging_steps=100,
    save_total_limit=2,
    report_to=[]
)

trainer_fs = Trainer(
    model=model_fs,
    args=training_args_fs,
    train_dataset=fs_tt_ds,
    eval_dataset=fs_val_ds,
    tokenizer=tokenizer_fs,
    data_collator=DataCollatorWithPadding(tokenizer_fs),
    compute_metrics=compute_metrics_fs,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("Training Fakespot (may take a while)...")
trainer_fs.train()
trainer_fs.save_model(os.path.join(OUTPUT_DIR,"fakespot_full_best"))
tokenizer_fs.save_pretrained(os.path.join(OUTPUT_DIR,"fakespot_full_best"))

# Predict full-train & test with Fakespot (probabilities)
# IMPORTANT: use ORIGINAL train_df_orig so shapes match TF-IDF & engineered features
print("Predicting Fakespot probabilities for ALL train and TEST...")
full_train_ds_fs = Dataset.from_pandas(
    train_df_orig[["text","label"]].reset_index(drop=True)
).map(tokenize_fs, batched=True, remove_columns=["text"])
full_train_ds_fs = full_train_ds_fs.rename_column("label","labels")
full_train_ds_fs.set_format(type="torch")

test_ds_fs = Dataset.from_pandas(
    test_df[["id","text"]].rename(columns={"id":"orig_id"})
).map(tokenize_fs, batched=True, remove_columns=["text"])
test_ds_fs.set_format(type="torch")

preds_train_fs = trainer_fs.predict(full_train_ds_fs).predictions
probs_train_fs = torch.softmax(torch.from_numpy(preds_train_fs), dim=1).numpy()[:,1]
preds_test_fs = trainer_fs.predict(test_ds_fs).predictions
probs_test_fs = torch.softmax(torch.from_numpy(preds_test_fs), dim=1).numpy()[:,1]

# free trainer/model to conserve memory
free_cuda(['trainer_fs','model_fs'])
print("Fakespot done.")

# ---------- 5) FINE-TUNE DESKLIB ----------
print("\n=== Fine-tune Desklib (full encoder) ===")
tokenizer_dsk = AutoTokenizer.from_pretrained(DSK_MODEL, use_fast=True)
config_dsk = AutoConfig.from_pretrained(DSK_MODEL)

class DesklibAIDetectionModel(nn.Module):
    # We'll create a wrapper that loads AutoModel and a classifier similar to earlier approach
    def __init__(self, config):
        super().__init__()
        self.transformer = AutoModelForSequenceClassification.from_config(config) if False else None
        from transformers import AutoModel
        self.base = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)
        self.config = config

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
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

from transformers import PreTrainedModel, AutoModel
class DesklibPretrained(PreTrainedModel):
    config_class = AutoConfig
    def __init__(self, config):
        super().__init__(config)
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

# load model weights (trusting the repo)
model_dsk = DesklibPretrained.from_pretrained(DSK_MODEL, config=config_dsk)
model_dsk.to(device)

def tokenize_dsk(batch):
    return tokenizer_dsk(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN_DSK)

# create small internal val split for DSK fine-tune (on upsampled train_df)
tt_train2, tt_val2 = train_test_split(
    train_df, test_size=0.05, random_state=SEED, stratify=train_df["label"]
)
dsk_tt_ds = Dataset.from_pandas(
    tt_train2[["text","label"]].reset_index(drop=True)
).map(tokenize_dsk, batched=True, remove_columns=["text"]).rename_column("label","labels")
dsk_tt_ds.set_format(type="torch")
dsk_val_ds = Dataset.from_pandas(
    tt_val2[["text","label"]].reset_index(drop=True)
).map(tokenize_dsk, batched=True, remove_columns=["text"]).rename_column("label","labels")
dsk_val_ds.set_format(type="torch")

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
    eval_steps=2000,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    learning_rate=LR,
    weight_decay=0.01,
    fp16=FP16,
    seed=SEED,
    logging_steps=100,
    save_total_limit=2,
    report_to=[]
)

trainer_dsk = Trainer(
    model=model_dsk,
    args=training_args_dsk,
    train_dataset=dsk_tt_ds,
    eval_dataset=dsk_val_ds,
    tokenizer=tokenizer_dsk,
    data_collator=DataCollatorWithPadding(tokenizer_dsk),
    compute_metrics=compute_metrics_dsk,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("Training Desklib (may take a while)...")
trainer_dsk.train()
trainer_dsk.save_model(os.path.join(OUTPUT_DIR,"desklib_full_best"))
tokenizer_dsk.save_pretrained(os.path.join(OUTPUT_DIR,"desklib_full_best"))

# Predict full-train & test with Desklib
# IMPORTANT: use ORIGINAL train_df_orig so shapes match TF-IDF & engineered features
print("Predicting Desklib probabilities for ALL train and TEST...")
full_train_ds_dsk = Dataset.from_pandas(
    train_df_orig[["text","label"]].reset_index(drop=True)
).map(tokenize_dsk, batched=True, remove_columns=["text"])
full_train_ds_dsk = full_train_ds_dsk.rename_column("label","labels")
full_train_ds_dsk.set_format(type="torch")

test_ds_dsk = Dataset.from_pandas(
    test_df[["id","text"]].rename(columns={"id":"orig_id"})
).map(tokenize_dsk, batched=True, remove_columns=["text"])
test_ds_dsk.set_format(type="torch")

preds_train_dsk_out = trainer_dsk.predict(full_train_ds_dsk)
preds_train_dsk = preds_train_dsk_out.predictions
if isinstance(preds_train_dsk, tuple):
    preds_train_dsk = preds_train_dsk[0]
preds_train_dsk = np.asarray(preds_train_dsk)
if preds_train_dsk.ndim==2 and preds_train_dsk.shape[1]==1:
    probs_train_dsk = torch.sigmoid(torch.from_numpy(preds_train_dsk).squeeze()).numpy()
else:
    probs_train_dsk = torch.sigmoid(torch.from_numpy(preds_train_dsk)).numpy()

preds_test_dsk_out = trainer_dsk.predict(test_ds_dsk)
preds_test_dsk = np.asarray(preds_test_dsk_out.predictions)
if preds_test_dsk.ndim==2 and preds_test_dsk.shape[1]==1:
    probs_test_dsk = torch.sigmoid(torch.from_numpy(preds_test_dsk).squeeze()).numpy()
else:
    probs_test_dsk = torch.sigmoid(torch.from_numpy(preds_test_dsk)).numpy()

free_cuda(['trainer_dsk','model_dsk'])
print("Desklib done.")

# ---------- 6) ENGINEER FEATURES (Benford + enhanced) ----------
print("\n=== Engineering Benford + enhanced features ===")

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

def enhanced_ai_features(df):
    features = pd.DataFrame()
    text_series = df['answer'].fillna("").astype(str)
    features['text_length'] = text_series.str.len()
    features['word_count'] = text_series.str.split().str.len().fillna(0).astype(int)
    features['avg_word_length'] = features['text_length'] / (features['word_count'] + 1)
    features['sentence_count'] = text_series.str.count(r'[.!?]+')
    features['avg_sentence_length'] = features['word_count'] / (features['sentence_count'] + 1)
    features['sentence_length_variance'] = text_series.apply(
        lambda x: np.var([len(s.split()) for s in re.split(r'[.!?]+', str(x)) if s.strip()]) if len(re.split(r'[.!?]+', str(x)))>1 else 0
    )
    features['max_sentence_length'] = text_series.apply(lambda x: max([len(s.split()) for s in re.split(r'[.!?]+', str(x)) if s.strip()] + [0]))
    features['min_sentence_length'] = text_series.apply(lambda x: min([len(s.split()) for s in re.split(r'[.!?]+', str(x)) if s.strip()] + [999]))
    features['comma_count'] = text_series.str.count(',')
    features['semicolon_count'] = text_series.str.count(';')
    features['colon_count'] = text_series.str.count(':')
    features['exclamation_count'] = text_series.str.count('!')
    features['question_count'] = text_series.str.count(r'\?')
    features['period_count'] = text_series.str.count(r'\.')
    features['quote_count'] = text_series.str.count('"')
    features['apostrophe_count'] = text_series.str.count("'")
    features['dash_count'] = text_series.str.count('-')
    features['ellipsis_count'] = text_series.str.count(r'\.\.\.')
    features['parentheses_count'] = text_series.str.count(r'[\(\)]')
    features['total_punctuation'] = (features['comma_count'] + features['semicolon_count'] + features['colon_count'] + features['exclamation_count'] + features['question_count'] + features['period_count'])
    features['punctuation_ratio'] = features['total_punctuation'] / (features['text_length'] + 1)
    features['punctuation_diversity'] = text_series.apply(lambda x: len(set([c for c in str(x) if c in '.,;:!?"\'-()[]{}'])))
    features['unique_words'] = text_series.apply(lambda x: len(set(str(x).lower().split())))
    features['ttr'] = features['unique_words'] / (features['word_count'] + 1)
    features['unique_word_ratio'] = features['unique_words'] / (features['word_count'] + 1)
    features['yules_k'] = text_series.apply(lambda x: 10000 * (sum([freq**2 for freq in pd.Series(str(x).lower().split()).value_counts().values]) - len(str(x).split())) / (len(str(x).split())**2) if len(str(x).split()) > 0 else 0)
    features['hapax_legomena'] = text_series.apply(lambda x: sum(1 for word, count in pd.Series(str(x).lower().split()).value_counts().items() if count == 1))
    features['dis_legomena'] = text_series.apply(lambda x: sum(1 for word, count in pd.Series(str(x).lower().split()).value_counts().items() if count == 2))
    features['hapax_ratio'] = features['hapax_legomena'] / (features['word_count'] + 1)
    ai_connectors = ['in conclusion', 'in summary', 'furthermore', 'moreover', 'additionally', 'however', 'therefore', 'thus', 'consequently', 'as a result', 'on the other hand', 'for instance', 'for example', 'it is important to note', 'it is worth noting', 'that being said', 'in other words', 'specifically', 'namely']
    features['ai_connector_density'] = text_series.apply(lambda x: sum(1 for phrase in ai_connectors if phrase in str(x).lower()) / (len(str(x).split()) + 1))
    formal_words = ['utilize', 'facilitate', 'implement', 'methodology', 'paradigm', 'leverage', 'robust', 'optimal', 'enhance', 'demonstrate']
    features['formal_word_ratio'] = text_series.apply(lambda x: sum(1 for word in formal_words if word in str(x).lower()) / (len(str(x).split()) + 1))
    passive_indicators = ['is made', 'was made', 'is given', 'was given', 'is shown', 'was shown', 'is considered', 'was considered', 'by the']
    features['passive_voice_ratio'] = text_series.apply(lambda x: sum(1 for phrase in passive_indicators if phrase in str(x).lower()) / (len(str(x).split()) + 1))
    sentence_starters = ['the', 'this', 'it', 'in', 'on', 'as', 'when', 'while', 'although']
    features['repetitive_starts'] = text_series.apply(lambda x: len(set([s.split()[0].lower() if s.split() else '' for s in re.split(r'[.!?]+', str(x)) if s.strip()])) / (len([s for s in re.split(r'[.!?]+', str(x)) if s.strip()]) + 1))
    features['hapax_dis_ratio'] = text_series.apply(lambda x: (sum(1 for count in pd.Series(str(x).lower().split()).value_counts().values if count == 1) + sum(1 for count in pd.Series(str(x).lower().split()).value_counts().values if count == 2)) / len(str(x).split()) if len(str(x).split()) > 0 else 0)
    features['subordinate_ratio'] = text_series.str.count(r'\b(that|which|who|when|where|while|although|because|if)\b') / (features['word_count'] + 1)
    try:
        from textblob import TextBlob
        features['sentiment_polarity'] = text_series.apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        features['sentiment_subjectivity'] = text_series.apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    except Exception:
        features['sentiment_polarity'] = 0
        features['sentiment_subjectivity'] = 0
    def flesch_reading_ease(text):
        sentences = len([s for s in re.split(r'[.!?]+', str(text)) if s.strip()])
        words = len(str(text).split())
        syllables = sum([len(re.findall(r'[aeiouy]+', word.lower())) for word in str(text).split()])
        if sentences > 0 and words > 0:
            return 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        return 0
    features['flesch_reading_ease'] = text_series.apply(flesch_reading_ease)
    features['max_word_length'] = text_series.apply(lambda x: max([len(w) for w in str(x).split()] + [0]))
    features['min_word_length'] = text_series.apply(lambda x: min([len(w) for w in str(x).split()] + [999]))
    features['word_length_std'] = text_series.apply(lambda x: np.std([len(w) for w in str(x).split()]) if len(str(x).split()) > 1 else 0)
    features['very_short_words'] = text_series.apply(lambda x: sum(1 for w in str(x).split() if len(w) <= 2))
    features['short_words'] = text_series.apply(lambda x: sum(1 for w in str(x).split() if 3 <= len(w) <= 4))
    features['medium_words'] = text_series.apply(lambda x: sum(1 for w in str(x).split() if 5 <= len(w) <= 7))
    features['long_words'] = text_series.apply(lambda x: sum(1 for w in str(x).split() if 8 <= len(w) <= 10))
    features['very_long_words'] = text_series.apply(lambda x: sum(1 for w in str(x).split() if len(w) > 10))
    features['short_word_ratio'] = features['short_words'] / (features['word_count'] + 1)
    features['long_word_ratio'] = (features['long_words'] + features['very_long_words']) / (features['word_count'] + 1)
    features['capital_letters'] = text_series.apply(lambda x: sum(1 for c in str(x) if c.isupper()))
    features['capital_ratio'] = features['capital_letters'] / (features['text_length'] + 1)
    features['all_caps_words'] = text_series.apply(lambda x: sum(1 for w in str(x).split() if w.isupper() and len(w) > 1))
    features['title_case_words'] = text_series.apply(lambda x: sum(1 for w in str(x).split() if w.istitle()))
    features['title_case_ratio'] = features['title_case_words'] / (features['word_count'] + 1)
    features['digit_count'] = text_series.str.count(r'\d')
    features['digit_ratio'] = features['digit_count'] / (features['text_length'] + 1)
    features['special_char_count'] = text_series.apply(lambda x: sum(1 for c in str(x) if not c.isalnum() and not c.isspace()))
    features['paragraph_count'] = text_series.apply(lambda x: len([p for p in str(x).split('\n\n') if p.strip()]))
    features['avg_paragraph_length'] = features['word_count'] / (features['paragraph_count'] + 1)
    features['repeated_words'] = text_series.apply(lambda x: len([w for w,c in pd.Series(str(x).lower().split()).value_counts().items() if c > 1]))
    features['max_word_repetition'] = text_series.apply(lambda x: pd.Series(str(x).lower().split()).value_counts().max() if len(str(x).split()) > 0 else 0)
    features['consecutive_duplicates'] = text_series.apply(lambda x: sum(1 for i in range(len(str(x).split())-1) if str(x).split()[i].lower() == str(x).split()[i+1].lower()))
    features['word_length_uniformity'] = text_series.apply(lambda x: 1 / (np.std([len(w) for w in str(x).split()]) + 0.1) if len(str(x).split()) > 1 else 0)
    features['sentence_length_uniformity'] = text_series.apply(lambda x: 1 / (np.std([len(s.split()) for s in re.split(r'[.!?]+', str(x)) if s.strip()]) + 0.1) if len([s for s in re.split(r'[.!?]+', str(x)) if s.strip()]) > 1 else 0)
    features['burstiness'] = features['sentence_length_variance'] / (features['avg_sentence_length'] + 1)
    topic_dummies = pd.get_dummies(df['topic'].fillna(''), prefix='topic') if 'topic' in df.columns else pd.DataFrame(index=df.index)
    return pd.concat([features, topic_dummies], axis=1).fillna(0.0)

benford = BenfordAnalyzer()
print("Computing train/test engineered features (this will take time)...")
train_features = enhanced_ai_features(train_df_orig)
test_features  = enhanced_ai_features(test_df)
train_benford  = benford.extract_benford_features(train_df_orig)
test_benford   = benford.extract_benford_features(test_df)
train_features = pd.concat([train_features, train_benford], axis=1).reset_index(drop=True).fillna(0.0)
test_features  = pd.concat([test_features, test_benford], axis=1).reset_index(drop=True).fillna(0.0)

# Align test columns to train columns
test_features = test_features.reindex(columns=train_features.columns, fill_value=0.0)

print("Engineered features shape:", train_features.shape)

# ---------- 7) TF-IDF features ----------
print("\nCreating TF-IDF features (char & word & custom)...")
def enhanced_text_features_simple(train_df, test_df):
    char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=2000, min_df=2, max_df=0.9, sublinear_tf=True)
    word_vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=3000, min_df=2, max_df=0.85, sublinear_tf=True, stop_words='english')
    ai_pattern_words = ['conclusion', 'summary', 'however', 'therefore', 'moreover', 'furthermore', 'additionally', 'importantly', 'notably']
    custom_vectorizer = TfidfVectorizer(vocabulary=ai_pattern_words, binary=True)
    train_char = char_vectorizer.fit_transform(train_df['answer'])
    test_char  = char_vectorizer.transform(test_df['answer'])
    train_word = word_vectorizer.fit_transform(train_df['answer'])
    test_word  = word_vectorizer.transform(test_df['answer'])
    train_custom = custom_vectorizer.fit_transform(train_df['answer'])
    test_custom  = custom_vectorizer.transform(test_df['answer'])
    return train_char, test_char, train_word, test_word, train_custom, test_custom

train_char, test_char, train_word, test_word, train_custom, test_custom = enhanced_text_features_simple(train_df_orig, test_df)

# Combine dense features (careful: memory heavy)
print("Converting sparse TF-IDF matrices to dense arrays (may use a lot of RAM)...")
X_train_tfidf = np.hstack([train_char.toarray(), train_word.toarray(), train_custom.toarray(), train_features.values])
X_test_tfidf  = np.hstack([test_char.toarray(),  test_word.toarray(),  test_custom.toarray(),  test_features.values])
print("TF-IDF + engineered feature shapes:", X_train_tfidf.shape, X_test_tfidf.shape)

# ---------- 8) Add transformer model probabilities as features ----------
# probs_train_fs, probs_train_dsk now correspond to ORIGINAL train_df_orig order
print("\nAdding Fakespot/Desklib probabilities as meta-features for ML models...")
X_train_full = np.column_stack([probs_train_fs, probs_train_dsk, X_train_tfidf])
X_test_full  = np.column_stack([probs_test_fs,  probs_test_dsk,  X_test_tfidf])
y_train = train_df_orig['is_cheating'].astype(int).values

print("Final ML feature shapes:", X_train_full.shape, X_test_full.shape)

# ---------- 9) ML ensemble: Stratified K-Fold training (OOF) ----------
print("\n=== Training ML models with Stratified K-Fold (producing OOF preds) ===")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_preds = {
    'lgb': np.zeros(len(X_train_full)),
    'xgb': np.zeros(len(X_train_full)),
    'cat': np.zeros(len(X_train_full)),
    'lr':  np.zeros(len(X_train_full))
}
test_preds = {k: np.zeros(X_test_full.shape[0]) for k in oof_preds.keys()}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_full, y_train)):
    print(f"\nFold {fold+1}/{N_SPLITS}")
    X_tr, X_val = X_train_full[tr_idx], X_train_full[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.008, max_depth=7, num_leaves=63,
        min_child_samples=5, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, min_split_gain=0.01,
        random_state=SEED+fold, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
    oof_preds['lgb'][val_idx] = lgb_model.predict_proba(X_val)[:,1]
    test_preds['lgb'] += lgb_model.predict_proba(X_test_full)[:,1] / N_SPLITS

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.008, max_depth=6, min_child_weight=1,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.1,
        gamma=0, random_state=SEED+fold, eval_metric='auc', tree_method='hist', n_jobs=-1
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_preds['xgb'][val_idx] = xgb_model.predict_proba(X_val)[:,1]
    test_preds['xgb'] += xgb_model.predict_proba(X_test_full)[:,1] / N_SPLITS

    # CatBoost
    cat_model = CatBoostClassifier(
        iterations=1500, learning_rate=0.02, depth=6, l2_leaf_reg=3,
        random_seed=SEED+fold, verbose=0
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    oof_preds['cat'][val_idx] = cat_model.predict_proba(X_val)[:,1]
    test_preds['cat'] += cat_model.predict_proba(X_test_full)[:,1] / N_SPLITS

    # Logistic Regression
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    lr_model = LogisticRegression(C=0.3, max_iter=1000, random_state=SEED+fold, solver='saga', n_jobs=-1)
    lr_model.fit(X_tr_s, y_tr)
    oof_preds['lr'][val_idx] = lr_model.predict_proba(X_val_s)[:,1]
    test_preds['lr'] += lr_model.predict_proba(scaler.transform(X_test_full))[:,1] / N_SPLITS

    # Fold metrics
    lgb_score = roc_auc_score(y_val, oof_preds['lgb'][val_idx])
    xgb_score = roc_auc_score(y_val, oof_preds['xgb'][val_idx])
    cat_score = roc_auc_score(y_val, oof_preds['cat'][val_idx])
    lr_score  = roc_auc_score(y_val, oof_preds['lr'][val_idx])
    print(f" Fold results — LGB: {lgb_score:.4f}, XGB: {xgb_score:.4f}, CAT: {cat_score:.4f}, LR: {lr_score:.4f}")

# ---------- 10) Ensemble optimization on OOF (constrained weights) ----------
print("\n=== Ensemble weight optimization on OOF predictions ===")
from scipy.optimize import minimize

def objective(weights, preds_dict, y):
    combined = np.zeros_like(y, dtype=float)
    for w, k in zip(weights, preds_dict.keys()):
        combined += w * preds_dict[k]
    return -roc_auc_score(y, combined)

keys = list(oof_preds.keys())
x0 = np.array([1/len(keys)]*len(keys))
bounds = [(0.0,1.0)]*len(keys)
cons = ({'type':'eq','fun': lambda w: np.sum(w)-1.0})
res = minimize(objective, x0, args=(oof_preds, y_train), method='SLSQP', bounds=bounds, constraints=cons)
if res.success:
    w_opt = res.x
    print("Optimized weights:", dict(zip(keys, w_opt)))
else:
    print("Optimization failed; using equal weights.")
    w_opt = x0

# Compute OOF ensemble with optimized weights
oof_combined = np.zeros_like(y_train, dtype=float)
test_combined = np.zeros(X_test_full.shape[0], dtype=float)
for w, k in zip(w_opt, keys):
    oof_combined += w * oof_preds[k]
    test_combined += w * test_preds[k]
oof_auc = roc_auc_score(y_train, oof_combined)
print(f"OOF ensemble AUC (optimized weights): {oof_auc:.6f}")

# ---------- 11) Stacking meta-learner on OOF predictions (meta-features = oof preds) ----------
print("\n=== Training stacking meta-learner (Logistic) on OOF preds ===")
X_meta = np.column_stack([oof_preds[k] for k in keys])
X_test_meta = np.column_stack([test_preds[k] for k in keys])
meta_clf = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=2000)
meta_clf.fit(X_meta, y_train)
meta_oof_pred = meta_clf.predict_proba(X_meta)[:,1]
meta_test_pred = meta_clf.predict_proba(X_test_meta)[:,1]
meta_oof_auc = roc_auc_score(y_train, meta_oof_pred)
print(f"Meta Logistic OOF AUC: {meta_oof_auc:.6f}")

# Optionally calibrate meta outputs with isotonic (use OOF)
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(meta_oof_pred, y_train)
meta_oof_iso = iso.predict(meta_oof_pred)
meta_oof_iso_auc = roc_auc_score(y_train, meta_oof_iso)
print(f"Meta (isotonic) OOF AUC: {meta_oof_iso_auc:.6f}")

# Choose best final ensemble (opt weights vs meta)
candidates = {
    "opt_weight": oof_auc,
    "meta_logistic": meta_oof_auc,
    "meta_iso": meta_oof_iso_auc
}
best_method = max(candidates, key=candidates.get)
print("Candidate OOF AUCs:", candidates)
print("Selected final ensemble method:", best_method)

if best_method == "opt_weight":
    final_test_probs = test_combined
    chosen_desc = f"optimized_weights_{dict(zip(keys, w_opt))}"
elif best_method == "meta_logistic":
    final_test_probs = meta_test_pred
    chosen_desc = "meta_logistic"
else:
    final_test_probs = iso.predict(meta_test_pred)
    chosen_desc = "meta_logistic_isotonic"

# ---------- 12) Smart pseudo-labeling (optional) ----------
def smart_pseudo_labeling(X_train, X_test, y_train, initial_pred, high_thresh=0.98, low_thresh=0.02):
    high_mask = (initial_pred >= high_thresh) | (initial_pred <= low_thresh)
    if high_mask.sum() < len(initial_pred) * 0.3:
        high_thresh, low_thresh = 0.95, 0.05
        high_mask = (initial_pred >= high_thresh) | (initial_pred <= low_thresh)
    idxs = np.where(high_mask)[0]
    if len(idxs)==0:
        return X_train, y_train, None, 0
    pseudo_X = X_test[idxs]
    pseudo_y = (initial_pred[idxs] > 0.5).astype(int)
    confidence_weights = np.where(initial_pred[idxs] > 0.5, initial_pred[idxs], 1-initial_pred[idxs])
    X_comb = np.vstack([X_train, pseudo_X])
    y_comb = np.concatenate([y_train, pseudo_y])
    sample_weights = np.concatenate([np.ones(len(y_train)), confidence_weights*0.5])
    return X_comb, y_comb, sample_weights, len(idxs)

X_combined, y_combined, sample_weights, n_pseudo = smart_pseudo_labeling(X_train_full, X_test_full, y_train, final_test_probs)
if n_pseudo > 0:
    print(f"Using {n_pseudo} pseudo-labeled samples (retraining small LGB)...")
    lgb_pseudo = lgb.LGBMClassifier(
        n_estimators=1500, learning_rate=0.01, max_depth=7,
        num_leaves=63, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, n_jobs=-1
    )
    lgb_pseudo.fit(X_combined, y_combined, sample_weight=sample_weights)
    test_pseudo = lgb_pseudo.predict_proba(X_test_full)[:,1]
    blend_ratio = min(0.3, n_pseudo / X_test_full.shape[0])
    final_test_probs = final_test_probs * (1 - blend_ratio) + test_pseudo * blend_ratio
    print(f"Pseudo-label blend ratio: {blend_ratio:.3f}")

# ---------- 13) Topic-specific calibration ----------
def topic_specific_calibration(preds, test_df, train_df, strength=0.1, min_samples=5):
    stats = train_df.groupby('topic')['is_cheating'].agg(['mean','count']).reset_index()
    stats = stats[stats['count'] > min_samples]
    preds_adj = preds.copy()
    for _, row in stats.iterrows():
        topic = row['topic']; mean = row['mean']
        mask = test_df['topic'] == topic
        if mask.sum() == 0: continue
        if mean > 0.7:
            preds_adj[mask] = preds_adj[mask] * (1 - strength) + np.clip(preds_adj[mask] + 0.1,0,1) * strength
        elif mean < 0.3:
            preds_adj[mask] = preds_adj[mask] * (1 - strength) + np.clip(preds_adj[mask] - 0.1,0,1) * strength
    return preds_adj

test_calibrated = topic_specific_calibration(final_test_probs, test_df, train_df_orig)

# ---------- 14) Advanced isotonic calibration (trained on OOF ensemble) ----------
iso_final = IsotonicRegression(out_of_bounds='clip')
if best_method == "opt_weight":
    oof_for_iso = oof_combined
elif best_method == "meta_logistic":
    oof_for_iso = meta_oof_pred
else:
    oof_for_iso = meta_oof_iso

iso_final.fit(oof_for_iso, y_train)
test_final_calibrated = iso_final.transform(test_calibrated)
test_final_calibrated = np.clip(test_final_calibrated, 0.001, 0.999)
test_final = 0.97 * test_final_calibrated + 0.03 * final_test_probs

# ---------- 15) Save submissions ----------
submission_main = pd.DataFrame({"id": test_df['id'], "is_cheating": test_final})
out_main = os.path.join(OUTPUT_DIR, "submission_final.csv")
submission_main.to_csv(out_main, index=False)

submission_cons = pd.DataFrame({"id": test_df['id'], "is_cheating": test_final_calibrated})
submission_cons.to_csv(os.path.join(OUTPUT_DIR, "submission_conservative.csv"), index=False)

submission_aggr = pd.DataFrame({"id": test_df['id'], "is_cheating": np.power(test_final, 0.9)})
submission_aggr.to_csv(os.path.join(OUTPUT_DIR, "submission_aggressive.csv"), index=False)

# Save per-model test probs for debugging
pd.DataFrame({"id": test_df['id'], "fakespot": probs_test_fs}).to_csv(os.path.join(OUTPUT_DIR,"test_probs_fakespot.csv"), index=False)
pd.DataFrame({"id": test_df['id'], "desklib":  probs_test_dsk}).to_csv(os.path.join(OUTPUT_DIR,"test_probs_desklib.csv"), index=False)

print("\nSaved final submissions to", OUTPUT_DIR)
print("Selected ensemble method:", chosen_desc)
print("OOF CV AUC (selected):", candidates[best_method])

