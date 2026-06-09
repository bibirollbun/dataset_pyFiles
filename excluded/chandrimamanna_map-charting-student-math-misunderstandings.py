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


## Environment / Installs (guarded for offline commit)
import socket

def _online(host="pypi.org"):
    try:
        socket.gethostbyname(host)
        return True
    except Exception:
        return False

if _online():
    print("Online: you may install/upgrade packages if you wish.")
    # Example (only if you are developing with Internet ON):
    # !pip install -q transformers datasets lightgbm sentence-transformers
else:
    print("Offline commit: skipping pip installs (use preinstalled libs or attach wheels).")




import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sentence_transformers import CrossEncoder, InputExample

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)



# Offline + local model paths
import os, torch

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["WANDB_DISABLED"] = "true"

BASE_MODEL_DIR   = "/kaggle/input/distilroberta-base"   # <- use your exact dataset name
RERANK_MODEL_DIR = "/kaggle/input/ms-marco-minilm-l-6-v2"

assert os.path.isdir(BASE_MODEL_DIR),   f"Missing: {BASE_MODEL_DIR} (did you Add data?)"
assert os.path.isdir(RERANK_MODEL_DIR), f"Missing: {RERANK_MODEL_DIR} (did you Add data?)"
print("Using local models:", BASE_MODEL_DIR, " | ", RERANK_MODEL_DIR)



import pandas as pd

# Load raw data
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test  = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')

# Build label mapping
train['label'] = train['Category'] + ':' + train['Misconception']
unique_labels = train['label'].unique().tolist()
label2id = {lab: idx for idx, lab in enumerate(unique_labels)}
id2label = {idx: lab for lab, idx in label2id.items()}
train['label_id'] = train['label'].map(label2id)
y = train['label_id'].values

# Identify the text column dynamically
print("All train columns:", train.columns.tolist())
obj_cols = train.select_dtypes(include='object').columns.tolist()
exclude = {'Category','Misconception','label'}
candidates = [c for c in obj_cols if c not in exclude]
if 'student_response' in candidates:
    TEXT_COL = 'student_response'
elif candidates:
    TEXT_COL = candidates[0]
else:
    raise ValueError(f"No text column found. Object cols: {obj_cols}")
print(f"Using text column: {TEXT_COL}")



def map3_score(y_true, y_probs):
    top3 = np.argsort(-y_probs, axis=1)[:, :3]
    scores = []
    for true, preds in zip(y_true, top3):
        if true in preds:
            scores.append(1.0 / (list(preds).index(true) + 1))
        else:
            scores.append(0.0)
    return np.mean(scores)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import torch

# 1. TF–IDF vectorization
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X = vectorizer.fit_transform(train[TEXT_COL])

# 2. 5-fold stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_map3, baseline_acc = [], []

for tr_idx, va_idx in skf.split(X, y):
    clf = LogisticRegression(solver='saga', n_jobs=-1, max_iter=1000)
    clf.fit(X[tr_idx], y[tr_idx])
    probs = clf.predict_proba(X[va_idx])
    baseline_map3.append(map3_score(y[va_idx], probs))
    baseline_acc.append(accuracy_score(y[va_idx], np.argmax(probs, axis=1)))

print(f"Baseline MAP@3   : {np.mean(baseline_map3):.4f}")
print(f"Baseline Accuracy: {np.mean(baseline_acc):.4f}")

# Clear GPU cache (if used later)
torch.cuda.empty_cache()





skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = BASE_MODEL_DIR  # use local snapshot
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

dataset = Dataset.from_pandas(train[[TEXT_COL, 'label_id']])
def tok(ex): 
    return tokenizer(ex[TEXT_COL], max_length=128, truncation=True, padding='max_length')
dataset = dataset.map(tok, batched=True).rename_column('label_id','labels')
dataset.set_format('torch', ['input_ids','attention_mask','labels'])

oof = np.zeros((len(train), len(unique_labels)))
scores = []
for fold, (tr, va) in enumerate(skf.split(train, y)):
    dtr, dva = dataset.select(tr), dataset.select(va)
    model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(unique_labels), local_files_only=True
    ).to(device)
    args = TrainingArguments(
        output_dir=f'./dr{fold}',
        seed=SEED,
        report_to='none',
        num_train_epochs=2,
        per_device_train_batch_size=32,
        learning_rate=2e-5,
        fp16=True,
        save_strategy='no',
        logging_steps=100
    )
    trainer = Trainer(model, args, train_dataset=dtr)
    trainer.train()
    model.save_pretrained(f'./dr{fold}')
    tokenizer.save_pretrained(f'./dr{fold}')
    preds = trainer.predict(dva).predictions
    p = torch.softmax(torch.tensor(preds), 1).numpy()
    oof[va] = p
    sc = map3_score(train['label_id'].iloc[va], p)
    scores.append(sc)
    print(f"Fold {fold} MAP@3: {sc:.4f}")
    torch.cuda.empty_cache()

print(f"DistilRoBERTa CV MAP@3: {np.mean(scores):.4f}")

        




import numpy as np
import pandas as pd
import torch
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer

# reload models and tokenizer
models, tokenizers = [], []
for fold in range(3):
    m = AutoModelForSequenceClassification.from_pretrained(f'./dr{fold}', num_labels=len(unique_labels)).to(device)
    t = AutoTokenizer.from_pretrained(f'./dr{fold}')
    models.append(m)
    tokenizers.append(t)
# prepare test
test_ds = Dataset.from_pandas(test)
def tok_test(ex): return tokenizers[0](ex[TEXT_COL], truncation=True, padding='max_length', max_length=128)
test_ds = test_ds.map(tok_test, batched=True)
test_ds.set_format(type='torch', columns=['input_ids','attention_mask'])
# inference
probs_list = []
for m in models:
    tmp_args = TrainingArguments(output_dir='./tmp', save_strategy='no', report_to='none')
    pred = Trainer(model=m, args=tmp_args).predict(test_ds)
    probs = torch.softmax(torch.tensor(pred.predictions), dim=1).numpy()
    probs_list.append(probs)
en = np.mean(probs_list, axis=0)
# pseudo-labels
mask = en.max(axis=1) > 0.995
if mask.sum() == 0:
    print("No pseudo-labels; using original train set.")
    aug_df = train[[TEXT_COL,'label_id']].copy()
else:
    pseudo_df = pd.DataFrame({TEXT_COL: test.loc[mask, TEXT_COL].values,
                              'label_id': en[mask].argmax(axis=1)})
    aug_df = pd.concat([train[[TEXT_COL,'label_id']], pseudo_df], ignore_index=True)
# build augmented dataset
df_aug = Dataset.from_pandas(aug_df)
def tok_aug(ex): return tokenizers[0](ex[TEXT_COL], truncation=True, padding='max_length', max_length=128)
df_aug = df_aug.map(tok_aug, batched=True).rename_column('label_id','labels')
df_aug.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
# retrain
aug_model = AutoModelForSequenceClassification.from_pretrained('./dr0', num_labels=len(unique_labels)).to(device)
aug_args = TrainingArguments(output_dir='./aug', seed=SEED, report_to='none',
                              num_train_epochs=1, per_device_train_batch_size=32,
                              learning_rate=2e-5, fp16=True, save_strategy='no')
Trainer(aug_model, aug_args, train_dataset=df_aug).train()
torch.cuda.empty_cache()
print(f"Retrain done. Aug size={len(aug_df)}")




 # 1) TF–IDF + meta-features
# Reduce TF–IDF dimension for speed
from sklearn.model_selection import StratifiedKFold

tf = vectorizer.transform(train[TEXT_COL])
mt = pd.DataFrame({
    'len': train[TEXT_COL].str.len(),
    'dig': train[TEXT_COL].str.count(r"\d")
})
Xm = np.hstack([tf[:, :500].toarray(), mt.values])  # only 500 TF-IDF features

# 2) Prepare placeholder for LightGBM OOF
moof = np.zeros((len(train), len(unique_labels)))

# 3) 3-fold LightGBM stacking with parallel jobs and fewer trees
skf_stack = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
for tr_idx, va_idx in skf_stack.split(Xm, y):
    gb = lgb.LGBMClassifier(n_estimators=50, n_jobs=-1)
    gb.fit(Xm[tr_idx], y[tr_idx])
    probs_fold = gb.predict_proba(Xm[va_idx])
    full_probs = np.zeros((len(va_idx), len(unique_labels)))
    for i, cls in enumerate(gb.classes_):
        full_probs[:, cls] = probs_fold[:, i]
    moof[va_idx] = full_probs

# 4) Train final LightGBM model on all data for stacking
gb_full = lgb.LGBMClassifier(n_estimators=50, n_jobs=-1).fit(Xm, y)

# 5) Meta-learner on concatenated OOFs with fewer estimators
stack_input = np.hstack([oof, moof])
meta_learner = GradientBoostingClassifier(n_estimators=25).fit(stack_input, y)



## Reranker Training (HF Trainer, Local Weights)
import numpy as np, pandas as pd, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Build pair examples from top-5 OOF candidates
top5 = np.argsort(-oof, axis=1)[:, :5]
rows = []
text_series = train[TEXT_COL].astype(str).fillna("")
label_ids   = train['label_id'].astype(int).values
for i, txt in enumerate(text_series):
    true_id = int(label_ids[i])
    for c in top5[i]:
        rows.append({"sentence1": txt, "sentence2": str(id2label[int(c)]), "label": int(c == true_id)})

rerank_df = pd.DataFrame(rows)
rerank_ds = Dataset.from_pandas(rerank_df, preserve_index=False)

tokenizer_r = AutoTokenizer.from_pretrained(RERANK_MODEL_DIR, local_files_only=True)

def tokenize_pairs(batch):
    t = tokenizer_r(batch["sentence1"], batch["sentence2"], padding="max_length", truncation=True, max_length=128)
    t["labels"] = [float(x) for x in batch["label"]]  # regression-style head (num_labels=1)
    return t

rerank_ds = rerank_ds.map(tokenize_pairs, batched=True, remove_columns=rerank_ds.column_names)
rerank_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

model_r = AutoModelForSequenceClassification.from_pretrained(
    RERANK_MODEL_DIR, num_labels=1, local_files_only=True
).to(device)

args_r = TrainingArguments(
    output_dir="./reranker",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    learning_rate=3e-5,
    fp16=torch.cuda.is_available(),
    save_strategy="no",
    logging_steps=100,
    report_to="none",
)

Trainer(model=model_r, args=args_r, train_dataset=rerank_ds).train()
torch.cuda.empty_cache()


## Batched Final Reranking & Submission
import numpy as np, pandas as pd, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Pick test text column safely
if 'TEXT_COL' in globals() and TEXT_COL in test.columns:
    TEXT_COL_TEST = TEXT_COL
else:
    obj_cols_test = test.select_dtypes(include='object').columns.tolist()
    resp_like = [c for c in obj_cols_test if 'response' in c.lower()]
    TEXT_COL_TEST = resp_like[0] if resp_like else next((c for c in obj_cols_test if c != 'row_id'), obj_cols_test[0])
print(f"[Final] Using test text column: {TEXT_COL_TEST}")

test_text = test[TEXT_COL_TEST].astype(str).fillna("")

# Load base fold models saved in fine-tuning
base_models = []
for fold in range(3):
    m = AutoModelForSequenceClassification.from_pretrained(
        f'./dr{fold}', num_labels=len(unique_labels)
    ).to(device)
    base_models.append(m)
tokenizer_base = AutoTokenizer.from_pretrained('./dr0')

# Tokenize test for base models
def tok_base(batch):
    return tokenizer_base(batch[TEXT_COL_TEST], padding='max_length', truncation=True, max_length=128)

test_ds = Dataset.from_pandas(pd.DataFrame({TEXT_COL_TEST: test_text}), preserve_index=False)
test_ds = test_ds.map(tok_base, batched=True, remove_columns=[TEXT_COL_TEST])
test_ds.set_format(type='torch', columns=['input_ids','attention_mask'])

# Ensemble base models
probs_list = []
infer_args = TrainingArguments(
    output_dir='./infer_base',
    per_device_eval_batch_size=128,
    fp16=torch.cuda.is_available(),
    report_to='none',
    save_strategy='no'
)
for m in base_models:
    pred = Trainer(model=m, args=infer_args).predict(test_ds)
    probs = torch.softmax(torch.tensor(pred.predictions), dim=1).numpy()
    probs_list.append(probs)
base_probs = np.mean(probs_list, axis=0)

# Top-5 per test row
top5 = np.argsort(-base_probs, axis=1)[:, :5]

# Ensure reranker available (trained model_r/tokenizer_r or fallback to local snapshot)
try:
    _ = model_r; _ = tokenizer_r
except NameError:
    tokenizer_r = AutoTokenizer.from_pretrained(RERANK_MODEL_DIR, local_files_only=True)
    model_r = AutoModelForSequenceClassification.from_pretrained(
        RERANK_MODEL_DIR, num_labels=1, local_files_only=True
    ).to(device)

# Pair dataset for reranker
pairs = []
for i, txt in enumerate(test_text):
    for c in top5[i]:
        pairs.append({'s1': txt, 's2': str(id2label[int(c)]), 'row_idx': i, 'cand_id': int(c)})
pairs_df = pd.DataFrame(pairs)

def tok_pairs(batch):
    t = tokenizer_r(batch['s1'], batch['s2'], padding='max_length', truncation=True, max_length=128)
    t['row_idx'] = batch['row_idx']
    t['cand_id'] = batch['cand_id']
    return t

pairs_ds = Dataset.from_pandas(pairs_df, preserve_index=False)
pairs_ds = pairs_ds.map(tok_pairs, batched=True, remove_columns=['s1','s2'])
pairs_ds.set_format(type='torch', columns=['input_ids','attention_mask','row_idx','cand_id'])

# Reranker inference
rerank_args = TrainingArguments(
    output_dir='./infer_rerank',
    per_device_eval_batch_size=128,
    fp16=torch.cuda.is_available(),
    report_to='none',
    save_strategy='no'
)
pred = Trainer(model=model_r, args=rerank_args).predict(pairs_ds)
scores = torch.tensor(pred.predictions).squeeze(-1).cpu().numpy()

# Take top-3 per row by reranker score
out_strings, ptr = [], 0
for i in range(len(test)):
    five_scores = scores[ptr:ptr+5]; five_cands = top5[i]; ptr += 5
    order = np.argsort(-five_scores)[:3]
    best3 = [five_cands[j] for j in order]
    out_strings.append(' '.join(str(id2label[int(k)]) for k in best3))

submission = pd.DataFrame({'row_id': test['row_id'], 'Category:Misconception': out_strings})
submission.to_csv('submission.csv', index=False)
print("Saved submission.csv:", submission.shape)
torch.cuda.empty_cache()


## Submission Sanity Check
import pandas as pd
sub = pd.read_csv('submission.csv')
print(sub.head(10))
print("\nRows:", len(sub), " Null preds:", sub['Category:Misconception'].isna().sum())
assert {'row_id','Category:Misconception'}.issubset(sub.columns), "Bad submission columns."
assert sub.shape[0] == test.shape[0], "Submission rows != test rows."
assert sub['Category:Misconception'].str.split().apply(len).between(1,3).all(), "Each row must have 1–3 labels."
print("\nSubmission looks valid. Ready to submit.")


