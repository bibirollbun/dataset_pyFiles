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


# ================================
# MINDMATE EMOTION MODEL 
# ================================

!pip install -q transformers datasets evaluate accelerate

import os, re, json, random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
from datasets import Dataset, DatasetDict
from transformers import *
import evaluate

# seeds
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
set_seed(SEED)

# ---------------------------
# AUTO-DETECT DATASET
# ---------------------------
input_root = Path("/kaggle/input")
DATA_PATH = None

if input_root.exists():
    candidates=[]
    for p in input_root.rglob("*.csv"):
        name=p.name.lower()
        if "emotion" in name or "goemotion" in name or "go-emotion" in name:
            candidates.append(p)
    if not candidates:
        candidates = list(input_root.rglob("*.csv"))
    if candidates:
        DATA_PATH=str(candidates[0])
        print("Using CSV:", DATA_PATH)
    else:
        raise Exception("No CSV file found under /kaggle/input")
else:
    raise Exception("/kaggle/input missing.")

df = pd.read_csv(DATA_PATH)
print("Columns:", df.columns.tolist())
display(df.head())

# ---------------------------
# DETECT TEXT COLUMN
# ---------------------------
text_col=None
for c in df.columns:
    if c.lower() in ["text","sentence","comment","utterance"]:
        text_col=c; break

if text_col is None:
    # fallback: longest string column
    string_cols=[c for c in df.columns if df[c].dtype=="object"]
    lens={c:df[c].astype(str).map(len).mean() for c in string_cols}
    text_col=max(lens,key=lens.get)

print("TEXT COLUMN =", text_col)

# ---------------------------
# DETECT LABEL COLUMN
# ---------------------------
label_col=None
for c in df.columns:
    if c!=text_col and c.lower() in ["label","labels","emotion","emotions","label_ids"]:
        label_col=c; break

if label_col is None:
    others=[c for c in df.columns if c!=text_col]
    label_col=others[-1]

print("LABEL COLUMN =", label_col)

work=df[[text_col,label_col]].rename(columns={text_col:"text",label_col:"labels"}).copy()

# ---------------------------
# PARSE LABELS
# ---------------------------
def parse_label_item(x):
    if pd.isnull(x): return []
    if isinstance(x,list): return x
    s=str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        inside=s[1:-1].strip()
        if not inside: return []
        parts=[p.strip().strip("\"'") for p in inside.split(",")]
        out=[]
        for p in parts:
            if p.isdigit(): out.append(int(p))
            else: out.append(p)
        return out
    if re.fullmatch(r"(\d+\s+)+\d+", s):
        return [int(p) for p in s.split()]
    if re.fullmatch(r"(\d+,)+\d+", s):
        return [int(p) for p in s.split(",")]
    if s.isdigit(): return [int(s)]
    if "," in s:
        parts=[p.strip() for p in s.split(",")]
        return [int(p) if p.isdigit() else p for p in parts]
    return [s]

work["labels_parsed"]=work["labels"].apply(parse_label_item)

# ---------------------------
# BUILD LABEL SPACE
# ---------------------------
all_items=[i for sub in work["labels_parsed"] for i in sub]
label_is_str=any(isinstance(x,str) and not x.isdigit() for x in all_items)

if label_is_str:
    uniq=sorted(list(set(all_items)))
    label2id={l:i for i,l in enumerate(uniq)}
    id2label={i:l for l,i in label2id.items()}
else:
    uniq_int=[int(x) for x in all_items]
    max_id=max(uniq_int)
    label2id={i:i for i in range(max_id+1)}
    id2label={i:str(i) for i in range(max_id+1)}

num_labels=len(label2id)
print("NUM LABELS =", num_labels)

def labels_to_ids(lst):
    out=[]
    for x in lst:
        if isinstance(x,str) and x in label2id:
            out.append(label2id[x])
        elif str(x).isdigit():
            out.append(int(x))
    return sorted(list(set(out)))

def to_multi(ids):
    arr=[0]*num_labels
    for i in ids:
        if 0<=i<num_labels: arr[i]=1
    return arr

work["label_ids"]=work["labels_parsed"].apply(labels_to_ids)
work["multihot"]=work["label_ids"].apply(to_multi)

# remove empty labels
work=work[work["label_ids"].map(len)>0].reset_index(drop=True)

# ---------------------------
# TRAIN/VAL SPLIT
# ---------------------------
strat=work["label_ids"].apply(lambda x:x[0])
train_df,val_df=train_test_split(work,test_size=0.12,random_state=SEED,stratify=strat)

hf_train=Dataset.from_pandas(train_df[["text","multihot"]].rename(columns={"multihot":"labels"}))
hf_val=Dataset.from_pandas(val_df[["text","multihot"]].rename(columns={"multihot":"labels"}))

dataset=DatasetDict({"train":hf_train,"validation":hf_val})

# ---------------------------
# TOKENIZE
# ---------------------------
MODEL_NAME="distilbert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(ex):
    enc=tokenizer(ex["text"],truncation=True,padding=False,max_length=128)
    enc["labels"]=ex["labels"]
    return enc

tokenized=dataset.map(preprocess,batched=True)
data_collator=DataCollatorWithPadding(tokenizer=tokenizer,return_tensors="pt")

# ---------------------------
# MODEL + METRICS
# ---------------------------
model=AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

f1=evaluate.load("f1")

def compute_metrics(eval_pred):
    logits,labels=eval_pred
    probs=torch.sigmoid(torch.tensor(logits))
    preds=(probs>=0.5).int().numpy()
    labels=np.array(labels)
    micro=f1.compute(predictions=preds,references=labels,average="micro")["f1"]
    macro=f1.compute(predictions=preds,references=labels,average="macro")["f1"]
    return {"micro_f1":micro,"macro_f1":macro}

# ---------------------------
# TRAIN
# ---------------------------
training_args=TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    greater_is_better=True,
    fp16=True if torch.cuda.is_available() else False
)

trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# ---------------------------
# SAVE MODEL
# ---------------------------
OUT="/kaggle/working/mindmate_text_emotion_model"
model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)
print("Saved to:", OUT)

# ---------------------------
# INFERENCE DEMO
# ---------------------------
pipe=pipeline("text-classification",model=OUT,tokenizer=OUT,function_to_apply="sigmoid",top_k=None)

def test_sent(s):
    out=pipe(s)[0]
    pairs=[]
    for item in out:
        m=re.search(r"(\d+)$",item["label"])
        idx=int(m.group(1)) if m else None
        name=id2label.get(idx,item["label"])
        pairs.append((name,item["score"]))
    pairs=sorted(pairs,key=lambda x:x[1],reverse=True)
    print("\nTEXT:",s)
    for name,score in pairs[:8]:
        print(f"  {name}: {score:.3f}")

test_sent("I feel nervous about my exams tomorrow.")
test_sent("I feel hopeless and I want to give up.")
test_sent("I am so happy and excited today!")


