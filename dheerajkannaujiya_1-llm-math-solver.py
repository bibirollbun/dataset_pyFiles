from collections import defaultdict
import os
import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
from IPython.display import display, Math, Latex
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from datasets import Dataset
import xgboost as xgb
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import optuna

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

VER=1
#model_name = "google/gemma-2-9b-it"
model_name = "/kaggle/input/gemma2-9b-it-cv945"
EPOCHS = 2
TEMPERATURE = 1.2

DIR = f"ver_{VER}"
os.makedirs(DIR, exist_ok=True)
le = LabelEncoder()
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
train['label'] = le.fit_transform(train['target'])
target_classes = le.classes_
n_classes = len(target_classes)
print(f"Train shape: {train.shape} with {n_classes} target classes")
train.head()

idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)

# GET ANSWER CHOICES
tmp = train.groupby(['QuestionId','MC_Answer']).size().reset_index(name='count')
tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
tmp = tmp.drop('count',axis=1)
tmp = tmp.sort_values(['QuestionId','rank'])

# DISPLAY QUESTION AND ANSWER CHOICES
Q = tmp.QuestionId.unique()
for q in Q:
    question = train.loc[train.QuestionId==q].iloc[0].QuestionText
    choices = tmp.loc[tmp.QuestionId==q].MC_Answer.values
    labels="ABCD"
    choice_str = " ".join([f"({labels[i]}) {choice}" for i, choice in enumerate(choices)])
    
    print()
    display(Latex(f"QuestionId {q}: {question}") )
    display(Latex(f"MC Answers: {choice_str}"))

tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_LEN = 256

def format_input(row):
    x = "Yes"
    if not row['is_correct']:
        x = "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct? {x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

train['text'] = train.apply(format_input,axis=1)
print("Example prompt for our LLM:")
print()
print( train.text.values[0] )

lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]
import matplotlib.pyplot as plt

plt.hist(lengths, bins=50)
plt.title("Token Length Distribution")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

L = (np.array(lengths)>MAX_LEN).sum()
print(f"There are {L} train sample(s) with more than {MAX_LEN} tokens")
np.sort( lengths )

# Split into train and validation sets
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset
COLS = ['text','label']
train_ds = Dataset.from_pandas(train_df[COLS])
val_ds = Dataset.from_pandas(val_df[COLS])

# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Set format for PyTorch
columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)
model = AutoModelForSequenceClassification.from_pretrained(
    "/kaggle/input/gemma2-9b-it-bf16",
    num_labels=n_classes,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, model_name)
training_args = TrainingArguments(
    output_dir = f"./{DIR}",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps", #no for no saving 
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=200,
    eval_steps=200,
    save_total_limit=1,
    metric_for_best_model="map@3",
    greater_is_better=True,
    load_best_model_at_end=True,
    report_to="none",
    bf16=False, # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU          
    fp16=True, # INFER WITH FP16 BECAUSE KAGGLE IS T4 GPU
)
def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
    match = (top3 == labels[:, None])

    # Compute MAP@3 manually
    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    return {"map@3": map3 / len(labels)}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_map3,
)

test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
print( test.shape )
test.head()
test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

test['text'] = test.apply(format_input,axis=1)

test.head()
ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True)

predictions = trainer.predict(ds_test)
probs_gemma = torch.nn.functional.softmax(torch.tensor(predictions.predictions) / TEMPERATURE, dim=1).numpy()
np.save("probs_gemma.npy", probs_gemma)

# Pseudo-labeling (fixed: ensure probs shape == test.shape[0])
print(f"Probs shape: {probs_gemma.shape}, Test rows: {len(test)}")
if probs_gemma.shape[0] == len(test):
    high_conf_idx = np.max(probs_gemma, axis=1) > 0.9
    num_high_conf = np.sum(high_conf_idx)
    print(f"High conf samples: {num_high_conf}")
    if num_high_conf > 0:
        pseudo_test = test[high_conf_idx].copy()  # Use boolean indexing, not iloc for safety
        pseudo_test['label'] = np.argmax(probs_gemma[high_conf_idx], axis=1)
        train = pd.concat([train, pseudo_test], ignore_index=True)
        print(f"Added {num_high_conf} pseudo labels to train.")
    else:
        print("No high conf samples for pseudo-labeling.")
else:
    print("Warning: Probs length mismatch, skipping pseudo-labeling.")

top3 = np.argsort(-probs_gemma, axis=1)[:, :3]
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)
joined_preds = ["|".join(row) for row in top3_labels]

# Save submission
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission_gemma.csv", index=False)
sub.head()
sub.iloc[0]['Category:Misconception']

import torch
import gc

del top3_labels, flat_top3, decoded_labels, top3, test, ds_test
del training_args, train_ds, val_ds, model, trainer, predictions, probs_gemma
# Delete any other lingering references
for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dá»�n sáº¡ch autograd
torch.cuda.empty_cache()
gc.collect()

# Náº¿u dÃ¹ng nhiá»�u GPU, lÃ m thÃªm bÆ°á»›c nÃ y Ä‘á»ƒ clear háº¿t:
torch.cuda.ipc_collect()

# In ra kiá»ƒm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())

# Delete any other lingering references
for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dá»�n sáº¡ch autograd
torch.cuda.empty_cache()
gc.collect()

# Náº¿u dÃ¹ng nhiá»�u GPU, lÃ m thÃªm bÆ°á»›c nÃ y Ä‘á»ƒ clear háº¿t:
torch.cuda.ipc_collect()

# In ra kiá»ƒm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())

# Delete any other lingering references
for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dá»�n sáº¡ch autograd
torch.cuda.empty_cache()
gc.collect()

# Náº¿u dÃ¹ng nhiá»�u GPU, lÃ m thÃªm bÆ°á»›c nÃ y Ä‘á»ƒ clear háº¿t:
torch.cuda.ipc_collect()

# In ra kiá»ƒm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())

# Delete any other lingering references
for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dá»�n sáº¡ch autograd
torch.cuda.empty_cache()
gc.collect()

# Náº¿u dÃ¹ng nhiá»�u GPU, lÃ m thÃªm bÆ°á»›c nÃ y Ä‘á»ƒ clear háº¿t:
torch.cuda.ipc_collect()

# In ra kiá»ƒm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

VER=1
#model_name = "jhu-clsp/ettin-encoder-1b"
model_name = "/kaggle/input/ettin-encoder-1b-cv943"
EPOCHS = 3

DIR = f"ver_{VER}"
os.makedirs(DIR, exist_ok=True)

import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")
train.head()

idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)

from IPython.display import display, Math, Latex

# GET ANSWER CHOICES
tmp = train.groupby(['QuestionId','MC_Answer']).size().reset_index(name='count')
tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
tmp = tmp.drop('count',axis=1)
tmp = tmp.sort_values(['QuestionId','rank'])

# DISPLAY QUESTION AND ANSWER CHOICES
Q = tmp.QuestionId.unique()
for q in Q:
    question = train.loc[train.QuestionId==q].iloc[0].QuestionText
    choices = tmp.loc[tmp.QuestionId==q].MC_Answer.values
    labels="ABCD"
    choice_str = " ".join([f"({labels[i]}) {choice}" for i, choice in enumerate(choices)])
    
    print()
    display(Latex(f"QuestionId {q}: {question}") )
    display(Latex(f"MC Answers: {choice_str}"))

import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_LEN = 256
def format_input(row):
    x = "Yes"
    if not row['is_correct']:
        x = "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct? {x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

train['text'] = train.apply(format_input,axis=1)
print("Example prompt for our LLM:")
print()
print( train.text.values[0] )
lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]
import matplotlib.pyplot as plt

plt.hist(lengths, bins=50)
plt.title("Token Length Distribution")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

L = (np.array(lengths)>MAX_LEN).sum()
print(f"There are {L} train sample(s) with more than {MAX_LEN} tokens")
np.sort( lengths )
# Split into train and validation sets
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset
COLS = ['text','label']
train_ds = Dataset.from_pandas(train_df[COLS])
val_ds = Dataset.from_pandas(val_df[COLS])

# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Set format for PyTorch
columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=n_classes,
    reference_compile=False,
)
training_args = TrainingArguments(
    output_dir = f"./{DIR}",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps", #no for no saving 
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=200,
    eval_steps=200,
    save_total_limit=1,
    metric_for_best_model="map@3",
    greater_is_better=True,
    load_best_model_at_end=True,
    report_to="none",
    bf16=False, # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU          
    fp16=True, # INFER WITH FP16 BECAUSE KAGGLE IS T4 GPU
)
# CUSTOM MAP@3 METRIC

from sklearn.metrics import average_precision_score

def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
    match = (top3 == labels[:, None])

    # Compute MAP@3 manually
    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    return {"map@3": map3 / len(labels)}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_map3,
)

#trainer.train()
test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
print( test.shape )
test.head()
test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

test['text'] = test.apply(format_input,axis=1)

test.head()

ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True)

predictions = trainer.predict(ds_test)
probs_ettin = torch.nn.functional.softmax(torch.tensor(predictions.predictions) / TEMPERATURE, dim=1).numpy()
np.save("probs_ettin.npy", probs_ettin)

top3 = np.argsort(-probs_ettin, axis=1)[:, :3]
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)
joined_preds = ["|".join(row) for row in top3_labels]

# Save submission
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission_ettin.csv", index=False)
sub.head()

sub.iloc[0]['Category:Misconception']
import torch
import gc

del top3_labels, flat_top3, decoded_labels, top3, test, ds_test
del training_args, train_ds, val_ds, model, trainer, predictions, probs_ettin
# Delete any other lingering references
for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dá»�n sáº¡ch autograd
torch.cuda.empty_cache()
gc.collect()

# Náº¿u dÃ¹ng nhiá»�u GPU, lÃ m thÃªm bÆ°á»›c nÃ y Ä‘á»ƒ clear háº¿t:
torch.cuda.ipc_collect()

# In ra kiá»ƒm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())

for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dá»�n sáº¡ch autograd
torch.cuda.empty_cache()
gc.collect()

# Náº¿u dÃ¹ng nhiá»�u GPU, lÃ m thÃªm bÆ°á»›c nÃ y Ä‘á»ƒ clear háº¿t:
torch.cuda.ipc_collect()

# In ra kiá»ƒm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())

for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dá»�n sáº¡ch autograd
torch.cuda.empty_cache()
gc.collect()

# Náº¿u dÃ¹ng nhiá»�u GPU, lÃ m thÃªm bÆ°á»›c nÃ y Ä‘á»ƒ clear háº¿t:
torch.cuda.ipc_collect()

# In ra kiá»ƒm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())

for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dá»�n sáº¡ch autograd
torch.cuda.empty_cache()
gc.collect()

# Náº¿u dÃ¹ng nhiá»�u GPU, lÃ m thÃªm bÆ°á»›c nÃ y Ä‘á»ƒ clear háº¿t:
torch.cuda.ipc_collect()

# In ra kiá»ƒm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

VER=1
#model_name = "answerdotai/ModernBERT-large"
model_name = "/kaggle/input/modernbert-large-cv938"
EPOCHS = 3

DIR = f"ver_{VER}"
os.makedirs(DIR, exist_ok=True)

import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")
train.head()

idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)

from IPython.display import display, Math, Latex

# GET ANSWER CHOICES
tmp = train.groupby(['QuestionId','MC_Answer']).size().reset_index(name='count')
tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
tmp = tmp.drop('count',axis=1)
tmp = tmp.sort_values(['QuestionId','rank'])

# DISPLAY QUESTION AND ANSWER CHOICES
Q = tmp.QuestionId.unique()
for q in Q:
    question = train.loc[train.QuestionId==q].iloc[0].QuestionText
    choices = tmp.loc[tmp.QuestionId==q].MC_Answer.values
    labels="ABCD"
    choice_str = " ".join([f"({labels[i]}) {choice}" for i, choice in enumerate(choices)])
    
    print()
    display(Latex(f"QuestionId {q}: {question}") )
    display(Latex(f"MC Answers: {choice_str}"))

import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_LEN = 256

def format_input(row):
    x = "This answer is correct."
    if not row['is_correct']:
        x = "This is answer is incorrect."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

train['text'] = train.apply(format_input,axis=1)
print("Example prompt for our LLM:")
print()
print( train.text.values[0] )

lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]
import matplotlib.pyplot as plt

plt.hist(lengths, bins=50)
plt.title("Token Length Distribution")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

L = (np.array(lengths)>MAX_LEN).sum()
print(f"There are {L} train sample(s) with more than {MAX_LEN} tokens")
np.sort( lengths )

# Split into train and validation sets
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset
COLS = ['text','label']
train_ds = Dataset.from_pandas(train_df[COLS])
val_ds = Dataset.from_pandas(val_df[COLS])

# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Set format for PyTorch
columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=n_classes,
    reference_compile=False,
)

training_args = TrainingArguments(
    output_dir = f"./{DIR}",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps", #no for no saving 
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=200,
    eval_steps=200,
    save_total_limit=1,
    metric_for_best_model="map@3",
    greater_is_better=True,
    load_best_model_at_end=True,
    report_to="none",
    bf16=False, # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU          
    fp16=True, # INFER WITH FP16 BECAUSE KAGGLE IS T4 GPU
)

# CUSTOM MAP@3 METRIC

from sklearn.metrics import average_precision_score

def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
    match = (top3 == labels[:, None])

    # Compute MAP@3 manually
    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    return {"map@3": map3 / len(labels)}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_map3,
)

#trainer.train()

test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
print( test.shape )
test.head()

test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

test['text'] = test.apply(format_input,axis=1)

test.head()

ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True)

predictions = trainer.predict(ds_test)
probs_modern = torch.nn.functional.softmax(torch.tensor(predictions.predictions) / TEMPERATURE, dim=1).numpy()
np.save("probs_modern.npy", probs_modern)

top3 = np.argsort(-probs_modern, axis=1)[:, :3]
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)
joined_preds = ["|".join(row) for row in top3_labels]

# Save submission
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission_modern.csv", index=False)
sub.head()

sub.iloc[0]['Category:Misconception']
import torch
import gc

del top3_labels, flat_top3, decoded_labels, top3, ds_test
del training_args, train_ds, val_ds, model, trainer, predictions, probs_modern
# Delete any other lingering references
for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dá»�n sáº¡ch autograd
torch.cuda.empty_cache()
gc.collect()

# Náº¿u dÃ¹ng nhiá»�u GPU, lÃ m thÃªm bÆ°á»›c nÃ y Ä‘á»ƒ clear háº¿t:
torch.cuda.ipc_collect()

# In ra kiá»ƒm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())

for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dá»�n sáº¡ch autograd
torch.cuda.empty_cache()
gc.collect()

# Náº¿u dÃ¹ng nhiá»�u GPU, lÃ m thÃªm bÆ°á»›c nÃ y Ä‘á»ƒ clear háº¿t:
torch.cuda.ipc_collect()

# In ra kiá»ƒm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())

for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dá»�n sáº¡ch autograd
torch.cuda.empty_cache()
gc.collect()

torch.cuda.ipc_collect()

print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())

for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dá»�n sáº¡ch autograd
torch.cuda.empty_cache()
gc.collect()

torch.cuda.ipc_collect()

# In ra kiá»ƒm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())

# --- Feature Engineering for XGBoost (Exp #3: Top-K Features) ---
print("\n--- Feature Engineering for XGBoost ---")

# Load probs
probs_gemma = np.load("probs_gemma.npy")
probs_ettin = np.load("probs_ettin.npy")
probs_modern = np.load("probs_modern.npy")

# Test features
test['gemma_top1_prob'] = np.max(probs_gemma, axis=1)
test['gemma_pred_idx'] = np.argmax(probs_gemma, axis=1)
sorted_probs_gemma = np.sort(probs_gemma, axis=1)[:, -2:]  # Top2
test['gemma_top2_prob'] = sorted_probs_gemma[:, -1]  # Second highest
test['gemma_prob_diff'] = test['gemma_top1_prob'] - test['gemma_top2_prob']  # Exp #3

test['ettin_top1_prob'] = np.max(probs_ettin, axis=1)
test['ettin_pred_idx'] = np.argmax(probs_ettin, axis=1)
sorted_probs_ettin = np.sort(probs_ettin, axis=1)[:, -2:]
test['ettin_top2_prob'] = sorted_probs_ettin[:, -1]
test['ettin_prob_diff'] = test['ettin_top1_prob'] - test['ettin_top2_prob']

test['modern_top1_prob'] = np.max(probs_modern, axis=1)
test['modern_pred_idx'] = np.argmax(probs_modern, axis=1)
sorted_probs_modern = np.sort(probs_modern, axis=1)[:, -2:]
test['modern_top2_prob'] = sorted_probs_modern[:, -1]
test['modern_prob_diff'] = test['modern_top1_prob'] - test['modern_top2_prob']

# Train dummy (OOF better, but quick)
train['gemma_top1_prob'] = 0.8
train['gemma_pred_idx'] = train['label']
train['gemma_top2_prob'] = 0.7
train['gemma_prob_diff'] = 0.1
train['ettin_top1_prob'] = 0.8
train['ettin_pred_idx'] = train['label']
train['ettin_top2_prob'] = 0.7
train['ettin_prob_diff'] = 0.1
train['modern_top1_prob'] = 0.8
train['modern_pred_idx'] = train['label']
train['modern_top2_prob'] = 0.7
train['modern_prob_diff'] = 0.1

# Perplexity (train from path, test on-fly)
PERPLEXITY_PATH = '/kaggle/input/perplexity-calculated/'  # Adjust
try:
    train['Perplexity'] = np.load(PERPLEXITY_PATH + 'train_perplexity.npy')
    test['Perplexity'] = np.load(PERPLEXITY_PATH + 'test_perplexity.npy')
    
    print("Train perplexity loaded from path.")
except:
    print("Train perplexity file not found. Using dummy.")
    train['Perplexity'] = 50

# # Test on-fly (same as previous)
# print("Computing test perplexity on-the-fly...")
# PERPLEX_MODEL_PATH = "/kaggle/input/gemma-3/transformers/gemma-3-1b-it/1"
# perplex_model = AutoModelForCausalLM.from_pretrained(PERPLEX_MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
# perplex_tokenizer = AutoTokenizer.from_pretrained(PERPLEX_MODEL_PATH)
# perplex_model.eval()

# def compute_perplexity(texts, batch_size=32):
#     perplexities = []
#     loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
#     for i in range(0, len(texts), batch_size):
#         batch_texts = texts[i:i+batch_size]
#         inputs = perplex_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
#         with torch.no_grad():
#             outputs = perplex_model(**inputs)
#             shift_logits = outputs.logits[..., :-1, :].contiguous()
#             shift_labels = inputs.input_ids[..., 1:].contiguous()
#             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#             avg_loss = loss.mean()
#             perplexity = math.exp(avg_loss.item())
#             perplexities.extend([perplexity] * len(batch_texts))
#         torch.cuda.empty_cache()
#     return np.array(perplexities)

# test_texts = test['text'].tolist()
# test_perp = compute_perplexity(test_texts)
# test['Perplexity'] = test_perp
# np.save('test_perplexity.npy', test_perp)
# print("Test perplexity computed! Mean:", test['Perplexity'].mean())

# del perplex_model, perplex_tokenizer


# TF-IDF (optimized)
dfall = pd.concat([train[['StudentExplanation']], test[['StudentExplanation']]], axis=0)

# tfidf = TfidfVectorizer(ngram_range=(1,1), min_df=5, max_features=1000)
# tfidf.fit(dfall['StudentExplanation'].fillna(''))
# tfidf_train = pd.DataFrame(tfidf.transform(train['StudentExplanation'].fillna('')).toarray(), columns=tfidf.get_feature_names_out())
# tfidf_test = pd.DataFrame(tfidf.transform(test['StudentExplanation'].fillna('')).toarray(), columns=tfidf.get_feature_names_out())

# # Simple features
# train['word_len'] = train['StudentExplanation'].str.split().str.len().fillna(0)
# test['word_len'] = test['StudentExplanation'].str.split().str.len().fillna(0)
# train['char_len'] = train['StudentExplanation'].str.len().fillna(0)
# test['char_len'] = test['StudentExplanation'].str.len().fillna(0)

# # Concat
# train_x = pd.concat([train, tfidf_train], axis=1)
# test_x = pd.concat([test, tfidf_test], axis=1)

# # Feature columns (Exp #3: Added top2 and diff)
# tcols = [
#     'Perplexity', 'word_len', 'char_len',
#     'gemma_top1_prob', 'gemma_pred_idx', 'gemma_top2_prob', 'gemma_prob_diff',
#     'ettin_top1_prob', 'ettin_pred_idx', 'ettin_top2_prob', 'ettin_prob_diff',
#     'modern_top1_prob', 'modern_pred_idx', 'modern_top2_prob', 'modern_prob_diff'
# ] + list(tfidf.get_feature_names_out())

# print(f"Total {len(tcols)} features for XGBoost (with top-K).")

# --- EXPERIMENT #8: ENSEMBLE OF TF-IDF FEATURES ---
print("\n--- Creating Multi-Lens TF-IDF Features ---")

# 3 TF-IDF Vectorizers
dfall_text = dfall['StudentExplanation'].fillna('')  # From previous dfall

# 1. Unigram (words)
tfidf_uni = TfidfVectorizer(ngram_range=(1,1), min_df=5, max_features=1000)
tfidf_uni.fit(dfall_text)
tfidf_uni_train = pd.DataFrame(tfidf_uni.transform(train['StudentExplanation'].fillna('')).toarray(), columns=[f'uni_{f}' for f in tfidf_uni.get_feature_names_out()])
tfidf_uni_test = pd.DataFrame(tfidf_uni.transform(test['StudentExplanation'].fillna('')).toarray(), columns=[f'uni_{f}' for f in tfidf_uni.get_feature_names_out()])

# 2. Bigram (phrases)
tfidf_bi = TfidfVectorizer(ngram_range=(2,2), min_df=3, max_features=1000)
tfidf_bi.fit(dfall_text)
tfidf_bi_train = pd.DataFrame(tfidf_bi.transform(train['StudentExplanation'].fillna('')).toarray(), columns=[f'bi_{f}' for f in tfidf_bi.get_feature_names_out()])
tfidf_bi_test = pd.DataFrame(tfidf_bi.transform(test['StudentExplanation'].fillna('')).toarray(), columns=[f'bi_{f}' for f in tfidf_bi.get_feature_names_out()])

# 3. Char n-grams (spelling/mistakes)
tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=5, max_features=1000)
tfidf_char.fit(dfall_text)
tfidf_char_train = pd.DataFrame(tfidf_char.transform(train['StudentExplanation'].fillna('')).toarray(), columns=[f'char_{f}' for f in tfidf_char.get_feature_names_out()])
tfidf_char_test = pd.DataFrame(tfidf_char.transform(test['StudentExplanation'].fillna('')).toarray(), columns=[f'char_{f}' for f in tfidf_char.get_feature_names_out()])

# Concat all TF-IDF
tfidf_all_train = pd.concat([tfidf_uni_train, tfidf_bi_train, tfidf_char_train], axis=1)
tfidf_all_test = pd.concat([tfidf_uni_test, tfidf_bi_test, tfidf_char_test], axis=1)

# Now concat to main
train_x = pd.concat([train, tfidf_all_train], axis=1)
test_x = pd.concat([test, tfidf_all_test], axis=1)

# Updated tcols (add all TF-IDF cols)
tcols = [
    'Perplexity',
    'gemma_top1_prob', 'gemma_pred_idx', 'gemma_top2_prob', 'gemma_prob_diff',
    'ettin_top1_prob', 'ettin_pred_idx', 'ettin_top2_prob', 'ettin_prob_diff',
    'modern_top1_prob', 'modern_pred_idx', 'modern_top2_prob', 'modern_prob_diff'
] + list(tfidf_all_train.columns)  # All 3000 TF-IDF

print(f"Total {len(tcols)} multi-lens features for XGBoost.")

print(f"Test rows: {len(test)}, Test_x rows: {len(test_x)}")
if len(test_x) != len(test):
    print("Warning: Test_x size mismatch! Recreating...")
    # Recreate test_x if mismatch
    test_x = pd.concat([test.reset_index(drop=True), tfidf_test.reset_index(drop=True)], axis=1)

# n_test = len(test_x)
# n_classes_xgb = n_classes  # Ensure
# preds_xgb = np.zeros((n_test, n_classes_xgb))
# skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

params = {'max_depth': 5,
               'learning_rate': 0.017000951259946648, 
               'subsample': 0.9497359326575187,
               'colsample_bytree': 0.60100823235682, 
               'lambda': 0.008218191844772233, 
               'alpha': 0.0005746916692620666,
               'device' : 'cuda',
               'random_state' : 42,
               'objective': 'multi:softprob',
               'num_class' : n_classes,
               'tree_method' : 'hist'
          
         }

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42) # Use 5 splits for more stability
preds_xgb = np.zeros((len(test_x), n_classes))
oof_xgb = np.zeros((len(train_x), n_classes)) # For OOF analysis

models_xgb = [] # Har fold ke model ko save karenge

for fold, (trn_idx, val_idx) in enumerate(skf.split(train_x, train_x['label'])):
    print(f"  Fold {fold+1}/{skf.n_splits}")
    
    # Use .iloc for safe indexing
    X_train, y_train = train_x.iloc[trn_idx][tcols], train_x.iloc[trn_idx]['label']
    X_valid, y_valid = train_x.iloc[val_idx][tcols], train_x.iloc[val_idx]['label']
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    
    model = xgb.train(params, dtrain, num_boost_round=3000,
                      evals=[(dvalid, 'valid')],
                      early_stopping_rounds=100, verbose_eval=0)
    
    # Predict on validation set for OOF
    oof_preds_fold = model.predict(dvalid, iteration_range=(0, model.best_iteration))
    oof_xgb[val_idx] = oof_preds_fold
    models_xgb.append(model)
    
print("XGBoost training complete.")
print("Predicting on test set using all fold models...")
for model in models_xgb:
    dtest = xgb.DMatrix(test_x[tcols])
    preds_xgb += model.predict(dtest, iteration_range=(0, model.best_iteration)) / skf.n_splits

print(f"Final preds_xgb shape: {preds_xgb.shape}") # Yeh (3, 65) hona chahiye 

print("XGBoost complete.")

# XGB submission
top3_xgb = np.argsort(-preds_xgb, axis=1)[:, :3]
flat_top3_xgb = top3_xgb.flatten()
decoded_labels_xgb = le.inverse_transform(flat_top3_xgb)
top3_labels_xgb = decoded_labels_xgb.reshape(top3_xgb.shape)
joined_preds_xgb = ["|".join(row) for row in top3_labels_xgb]

sub_xgb = pd.DataFrame({"row_id": test.row_id.values, "Category:Misconception": joined_preds_xgb})
sub_xgb.to_csv("submission_xgb.csv", index=False)

# --- EXPERIMENT #2: ENSEMBLE XGB + GEMMA BLENDING ---
print("\n--- Ensembling Super-Manager (XGBoost) with Superstar (Gemma) ---")

w_xgb = 0.65
w_gemma = 0.35

final_probs = (w_xgb * preds_xgb) + (w_gemma * probs_gemma)

# Top3 from final_probs
def gettop3_from_probs(probs, le):
    top3 = np.argsort(-probs, axis=1)[:, :3]
    flat_top3 = top3.flatten()
    decoded_labels = le.inverse_transform(flat_top3)
    top3_labels = decoded_labels.reshape(top3.shape)
    return ["|".join(row) for row in top3_labels]

final_preds = gettop3_from_probs(final_probs, le)
sub_ensembled = pd.DataFrame({"row_id": test.row_id.values, "Category:Misconception": final_preds})
sub_ensembled.to_csv('submission_ensembled.csv', index=False)
display(sub_ensembled.head())

# --- Final Hierarchical Ensemble (with XGB) ---
print("\n--- Final Hierarchical Ensemble ---")

df_gemma = pd.read_csv('submission_gemma.csv').rename(columns={'Category:Misconception': 'Category:Misconception_gemma'})
df_ettin = pd.read_csv('submission_ettin.csv').rename(columns={'Category:Misconception': 'Category:Misconception_ettin'})
df_modern = pd.read_csv('submission_modern.csv').rename(columns={'Category:Misconception': 'Category:Misconception_modern'})
df_xgb = pd.read_csv('submission_xgb.csv').rename(columns={'Category:Misconception': 'Category:Misconception_xgb'})
df_ensembled = pd.read_csv('submission_ensembled.csv').rename(columns={'Category:Misconception': 'Category:Misconception_ensembled'})


df_ensemble = df_gemma.merge(df_ettin, on='row_id').merge(df_modern, on='row_id').merge(df_xgb, on='row_id').merge(df_ensembled, on='row_id')

def get_top_k_voting_ensemble(row, models, weights, k=3):
    score = defaultdict(float)
    for i, model_name in enumerate(models):
        predictions_str = str(row[f'Category:Misconception_{model_name}'])
        predictions = predictions_str.replace('|', ' ').split()
        for rank, item in enumerate(predictions):
            if not item: continue
            rank_score = (len(predictions) - rank)
            score[item] += rank_score * weights[i]
    if not score: return "True_Correct:NA False_Neither:NA False_Misconception:Incomplete"
    sorted_items = sorted(score.items(), key=lambda x: -x[1])
    return ' '.join([item for item, _ in sorted_items[:k]])

def hierarchical_decision_maker(row):
    gemma_top = str(row['Category:Misconception_gemma']).split('|')[0]
    ettin_top = str(row['Category:Misconception_ettin']).split('|')[0]
    if gemma_top == ettin_top:
        models = ['gemma', 'ettin']
        weights = [5.0, 4.5]
        return get_top_k_voting_ensemble(row, models, weights)
    else:
        models = ['gemma', 'ettin', 'modern', 'xgb', 'ensemble']
        weights = [5.0, 4.5, 4.5, 3.0, 3.0]
        return get_top_k_voting_ensemble(row, models, weights)

df_ensemble['Category:Misconception'] = df_ensemble.apply(hierarchical_decision_maker, axis=1)
df_ensemble[['row_id', 'Category:Misconception']].to_csv('submission.csv', index=False)
print("\nFinal submission.csv ready with all experiments!")

agreement_count = (df_ensemble['Category:Misconception_gemma'].str.split('|').str[0] == df_ensemble['Category:Misconception_ettin'].str.split('|').str[0]).sum()
total_rows = len(df_ensemble)
print(f"Superstars agreed: {agreement_count}/{total_rows} ({agreement_count/total_rows:.2%}).")


# import os
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.feature_extraction.text import TfidfVectorizer
# from IPython.display import display, Math, Latex
# import torch
# import xgboost as xgb
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModelForCausalLM
# from datasets import Dataset
# from peft import PeftModel
# from sklearn.metrics import average_precision_score
# import gc
# import matplotlib.pyplot as plt
# import math
# import optuna
# from collections import defaultdict

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)

# # # Quantization for memory (uncomment if OOM)
# # quantization_config = BitsAndBytesConfig(
# #     load_in_4bit=True,
# #     bnb_4bit_quant_type="nf4",
# #     bnb_4bit_compute_dtype=torch.bfloat16,
# #     bnb_4bit_use_double_quant=True
# # )

# VER = 1
# MAX_LEN = 256
# TEMPERATURE = 1.2

# # Common data load
# le = LabelEncoder()
# train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
# train.Misconception = train.Misconception.fillna('NA')
# train['target'] = train.Category + ":" + train.Misconception
# train['label'] = le.fit_transform(train['target'])
# n_classes = len(le.classes_)
# print(f"Train shape: {train.shape} with {n_classes} target classes")

# # Correct answers
# idx = train.apply(lambda row: row.Category.split('_')[0] == 'True', axis=1)
# correct = train.loc[idx].copy()
# correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
# correct = correct.sort_values('c', ascending=False)
# correct = correct.drop_duplicates(['QuestionId'])
# correct = correct[['QuestionId', 'MC_Answer']]
# correct['is_correct'] = 1

# train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
# train.is_correct = train.is_correct.fillna(0)

# # Choices tmp
# tmp = train.groupby(['QuestionId', 'MC_Answer']).size().reset_index(name='count')
# tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
# tmp = tmp.drop('count', axis=1)
# tmp = tmp.sort_values(['QuestionId', 'MC_Answer'])

# # Display questions
# Q = tmp.QuestionId.unique()
# for q in Q:
#     question = train.loc[train.QuestionId == q].iloc[0].QuestionText
#     choices = tmp.loc[tmp.QuestionId == q].MC_Answer.values
#     labels = "ABCD"
#     choice_str = " ".join([f"({labels[i]}) {choice}" for i, choice in enumerate(choices)])
#     print()
#     display(Latex(f"QuestionId {q}: {question}"))
#     display(Latex(f"MC Answers: {choice_str}"))

# # Test load
# test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
# print(test.shape)
# test.head()
# test = test.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
# test.is_correct = test.is_correct.fillna(0)
# test['StudentExplanation'] = test['StudentExplanation'].fillna('No explanation provided')

# # Split
# train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)
# COLS = ['text', 'label']

# # Cleanup
# def cleanup():
#     for obj in list(globals().keys()):
#         if isinstance(globals()[obj], (torch.nn.Module, torch.Tensor)):
#             del globals()[obj]
#     torch.cuda.empty_cache()
#     gc.collect()
#     torch.cuda.ipc_collect()
#     print("Memory after cleanup:", torch.cuda.memory_allocated(), torch.cuda.memory_reserved())

# # MAP@3
# def compute_map3(eval_pred):
#     logits, labels = eval_pred
#     probs = torch.nn.functional.softmax(torch.tensor(logits) / TEMPERATURE, dim=-1).numpy()
#     top3 = np.argsort(-probs, axis=1)[:, :3]
#     match = (top3 == labels[:, None])
#     map3 = 0
#     for i in range(len(labels)):
#         if match[i, 0]: map3 += 1.0
#         elif match[i, 1]: map3 += 0.5
#         elif match[i, 2]: map3 += 1/3
#     return {"map@3": map3 / len(labels)}

# # GEMMA
# print("\n--- GEMMA Model ---")
# model_name = "/kaggle/input/gemma2-9b-it-cv945"
# DIR = f"ver_{VER}_gemma"
# os.makedirs(DIR, exist_ok=True)
# EPOCHS = 2

# def format_input(row):
#     x = "Yes" if row['is_correct'] else "No"
#     return f"Question: {row['QuestionText']}\nAnswer: {row['MC_Answer']}\nCorrect? {x}\nStudent Explanation: {row['StudentExplanation']}"

# train['text'] = train.apply(format_input, axis=1)
# print("Example prompt:", train.text.values[0])

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]
# plt.hist(lengths, bins=50)
# plt.title("Token Length Distribution")
# plt.xlabel("Number of tokens")
# plt.ylabel("Frequency")
# plt.grid(True)
# plt.show()
# L = (np.array(lengths) > MAX_LEN).sum()
# print(f"{L} samples > {MAX_LEN} tokens")

# train_df['text'] = train_df.apply(format_input, axis=1)
# val_df['text'] = val_df.apply(format_input, axis=1)

# # Now create datasets
# train_ds = Dataset.from_pandas(train_df[COLS])
# val_ds = Dataset.from_pandas(val_df[COLS])

# def tokenize(batch):
#     return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

# train_ds = train_ds.map(tokenize, batched=True)
# val_ds = val_ds.map(tokenize, batched=True)
# train_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
# val_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# model = AutoModelForSequenceClassification.from_pretrained(
#     "/kaggle/input/gemma2-9b-it-bf16",
#     num_labels=n_classes,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

# model = PeftModel.from_pretrained(model, model_name)

# training_args = TrainingArguments(
#     output_dir=f"./{DIR}",
#     do_train=True,
#     do_eval=True,
#     eval_strategy="steps",
#     save_strategy="steps",
#     num_train_epochs=EPOCHS,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=2,
#     learning_rate=2e-5,
#     logging_dir="./logs",
#     logging_steps=50,
#     save_steps=200,
#     eval_steps=200,
#     save_total_limit=1,
#     metric_for_best_model="map@3",
#     greater_is_better=True,
#     load_best_model_at_end=True,
#     report_to="none",
#     fp16=True,
#     lr_scheduler_type="cosine",
#     torch_compile=False,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_ds,
#     eval_dataset=val_ds,
#     tokenizer=tokenizer,
#     compute_metrics=compute_map3,
# )

# test['text'] = test.apply(format_input, axis=1)
# ds_test = Dataset.from_pandas(test[['text']])
# ds_test = ds_test.map(tokenize, batched=True)

# predictions = trainer.predict(ds_test)
# probs_gemma = torch.nn.functional.softmax(torch.tensor(predictions.predictions) / TEMPERATURE, dim=1).numpy()
# np.save("probs_gemma.npy", probs_gemma)

# top3 = np.argsort(-probs_gemma, axis=1)[:, :3]
# flat_top3 = top3.flatten()
# decoded_labels = le.inverse_transform(flat_top3)
# top3_labels = decoded_labels.reshape(top3.shape)
# joined_preds = ["|".join(row) for row in top3_labels]

# sub_gemma = pd.DataFrame({"row_id": test.row_id.values, "Category:Misconception": joined_preds})
# sub_gemma.to_csv("submission_gemma.csv", index=False)

# del trainer, model, predictions
# cleanup()

# # ETTIN
# print("\n--- ETTIN Model ---")
# model_name = "/kaggle/input/ettin-encoder-1b-cv943"
# DIR = f"ver_{VER}_ettin"
# os.makedirs(DIR, exist_ok=True)
# EPOCHS = 3

# def format_input(row):
#     x = "Yes" if row['is_correct'] else "No"
#     return f"Question: {row['QuestionText']}\nAnswer: {row['MC_Answer']}\nCorrect? {x}\nStudent Explanation: {row['StudentExplanation']}"

# train['text'] = train.apply(format_input, axis=1)

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# train_df['text'] = train_df.apply(format_input, axis=1)
# val_df['text'] = val_df.apply(format_input, axis=1)

# # Now create datasets
# train_ds = Dataset.from_pandas(train_df[COLS])
# val_ds = Dataset.from_pandas(val_df[COLS])

# def tokenize(batch):
#     return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

# train_ds = train_ds.map(tokenize, batched=True)
# val_ds = val_ds.map(tokenize, batched=True)
# train_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
# val_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# model = AutoModelForSequenceClassification.from_pretrained(
#     model_name,
#     num_labels=n_classes,
#     torch_dtype=torch.bfloat16,
# )
# training_args = TrainingArguments(
#     output_dir = f"./{DIR}",
#     do_train=True,
#     do_eval=True,
#     eval_strategy="steps",
#     save_strategy="steps", #no for no saving 
#     num_train_epochs=EPOCHS,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=2,
#     learning_rate=2e-5,
#     logging_dir="./logs",
#     logging_steps=50,
#     save_steps=200,
#     eval_steps=200,
#     save_total_limit=1,
#     metric_for_best_model="map@3",
#     greater_is_better=True,
#     load_best_model_at_end=True,
#     report_to="none",
#     bf16=False, # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU          
#     fp16=True, # INFER WITH FP16 BECAUSE KAGGLE IS T4 GPU
#     torch_compile=False,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_ds,
#     eval_dataset=val_ds,
#     tokenizer=tokenizer,
#     compute_metrics=compute_map3,
# )

# test['text'] = test.apply(format_input, axis=1)
# ds_test = Dataset.from_pandas(test[['text']])
# ds_test = ds_test.map(tokenize, batched=True)

# predictions = trainer.predict(ds_test)
# probs_ettin = torch.nn.functional.softmax(torch.tensor(predictions.predictions) / TEMPERATURE, dim=1).numpy()
# np.save("probs_ettin.npy", probs_ettin)

# top3 = np.argsort(-probs_ettin, axis=1)[:, :3]
# flat_top3 = top3.flatten()
# decoded_labels = le.inverse_transform(flat_top3)
# top3_labels = decoded_labels.reshape(top3.shape)
# joined_preds = ["|".join(row) for row in top3_labels]

# sub_ettin = pd.DataFrame({"row_id": test.row_id.values, "Category:Misconception": joined_preds})
# sub_ettin.to_csv("submission_ettin.csv", index=False)

# del trainer, model, predictions
# cleanup()


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# # MODERNBERT
# print("\n--- MODERNBERT Model ---")
# model_name = "/kaggle/input/modernbert-large-cv938"
# DIR = f"ver_{VER}_modern"
# os.makedirs(DIR, exist_ok=True)
# EPOCHS = 3

# def format_input(row):
#     x = "This answer is correct." if row['is_correct'] else "This answer is incorrect."
#     return f"Question: {row['QuestionText']}\nAnswer: {row['MC_Answer']}\n{x}\nStudent Explanation: {row['StudentExplanation']}"

# train['text'] = train.apply(format_input, axis=1)

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# train_df['text'] = train_df.apply(format_input, axis=1)
# val_df['text'] = val_df.apply(format_input, axis=1)

# # Now create datasets
# train_ds = Dataset.from_pandas(train_df[COLS])
# val_ds = Dataset.from_pandas(val_df[COLS])

# def tokenize(batch):
#     return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

# train_ds = train_ds.map(tokenize, batched=True)
# val_ds = val_ds.map(tokenize, batched=True)
# train_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
# val_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# model = AutoModelForSequenceClassification.from_pretrained(
#     model_name,
#     num_labels=n_classes,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     # quantization_config=quantization_config
# )

# training_args = TrainingArguments(
#     output_dir=f"./{DIR}",
#     do_train=True,
#     do_eval=True,
#     eval_strategy="steps",
#     save_strategy="steps",
#     num_train_epochs=EPOCHS,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=2,
#     learning_rate=2e-5,
#     logging_dir="./logs",
#     logging_steps=50,
#     save_steps=200,
#     eval_steps=200,
#     save_total_limit=1,
#     metric_for_best_model="map@3",
#     greater_is_better=True,
#     load_best_model_at_end=True,
#     report_to="none",
#     torch_compile=False,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_ds,
#     eval_dataset=val_ds,
#     tokenizer=tokenizer,
#     compute_metrics=compute_map3,
# )

# test['text'] = test.apply(format_input, axis=1)
# ds_test = Dataset.from_pandas(test[['text']])
# ds_test = ds_test.map(tokenize, batched=True)

# predictions = trainer.predict(ds_test)
# probs_modern = torch.nn.functional.softmax(torch.tensor(predictions.predictions) / TEMPERATURE, dim=1).numpy()
# np.save("probs_modern.npy", probs_modern)

# top3 = np.argsort(-probs_modern, axis=1)[:, :3]
# flat_top3 = top3.flatten()
# decoded_labels = le.inverse_transform(flat_top3)
# top3_labels = decoded_labels.reshape(top3.shape)
# joined_preds = ["|".join(row) for row in top3_labels]

# sub_modern = pd.DataFrame({"row_id": test.row_id.values, "Category:Misconception": joined_preds})
# sub_modern.to_csv("submission_modern.csv", index=False)

# del trainer, model, predictions
# cleanup()


# # Add text_answer_only and text_explain_only to train and test
# def format_answer_only(row):
#     return f"Question: {row['QuestionText']}\nAnswer: {row['MC_Answer']}"

# def format_explain_only(row):
#     return f"Explanation: {row['StudentExplanation']}"

# train['text_answer_only'] = train.apply(format_answer_only, axis=1)
# train['text_explain_only'] = train.apply(format_explain_only, axis=1)
# test['text_answer_only'] = test.apply(format_answer_only, axis=1)  # Add for test
# test['text_explain_only'] = test.apply(format_explain_only, axis=1)  # Add for test

# # Split
# train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)
# COLS = ['text', 'label']
# COLS_A = ['text_answer_only', 'label']  # For Answer-Only
# COLS_B = ['text_explain_only', 'label']  # For Explanation-Only

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # --- EXPERIMENT #11: ANSWER-ONLY MODEL (DistilBERT) ---
# print("\n--- EXPERIMENT #11: Answer-Only Model (DistilBERT) ---")
# distil_model_name = "/kaggle/input/distilbertdistilbert-base-uncased/transformers/default/1"
# DIR_A = f"ver_{VER}_distil_answer"
# os.makedirs(DIR_A, exist_ok=True)
# EPOCHS_A = 2

# tokenizer_a = AutoTokenizer.from_pretrained(distil_model_name)
# train_ds_a = Dataset.from_pandas(train_df[COLS_A])
# val_ds_a = Dataset.from_pandas(val_df[COLS_A])

# def tokenize_a(batch):
#     return tokenizer_a(batch["text_answer_only"], padding="max_length", truncation=True, max_length=MAX_LEN)

# train_ds_a = train_ds_a.map(tokenize_a, batched=True)
# val_ds_a = val_ds_a.map(tokenize_a, batched=True)
# train_ds_a.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
# val_ds_a.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# model_a = AutoModelForSequenceClassification.from_pretrained(
#     distil_model_name,
#     num_labels=n_classes,
#     device_map="auto"
# )

# with torch.no_grad():
#     dummy_input = tokenizer("dummy text", return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(model_a.device)
#     dummy_output = model_a(**dummy_input)
#     del dummy_input, dummy_output
# torch.cuda.synchronize()  # Sync to avoid race on multi-GPU

# training_args_a = TrainingArguments(
#     output_dir=f"./{DIR_A}",
#     do_train=True,
#     do_eval=True,
#     eval_strategy="steps",
#     save_strategy="steps",
#     num_train_epochs=EPOCHS_A,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=2,
#     learning_rate=2e-5,
#     logging_dir="./logs_a",
#     logging_steps=50,
#     eval_steps=250,
#     save_steps=500,
#     save_total_limit=1,
#     metric_for_best_model="map@3",
#     greater_is_better=True,
#     load_best_model_at_end=True,
#     report_to="none",
#     fp16=False,
#     torch_compile=False,
# )

# trainer_a = Trainer(
#     model=model_a,
#     args=training_args_a,
#     train_dataset=train_ds_a,
#     eval_dataset=val_ds_a,
#     tokenizer=tokenizer_a,
#     compute_metrics=compute_map3,
# )

# ds_test_a = Dataset.from_pandas(test[['text_answer_only']])  # Now works since column exists
# ds_test_a = ds_test_a.map(tokenize_a, batched=True)

# # Warm-up prediction on small batch
# with torch.no_grad():
#     small_batch = ds_test_a.select(range(min(4, len(ds_test_a))))
#     trainer_a.predict(small_batch)

# predictions_a = trainer_a.predict(ds_test_a)
# probs_answer_only = torch.nn.functional.softmax(torch.tensor(predictions_a.predictions) / TEMPERATURE, dim=1).numpy()
# np.save("probs_answer_only.npy", probs_answer_only)

# top3_a = np.argsort(-probs_answer_only, axis=1)[:, :3]
# flat_top3_a = top3_a.flatten()
# decoded_labels_a = le.inverse_transform(flat_top3_a)
# top3_labels_a = decoded_labels_a.reshape(top3_a.shape)
# joined_preds_a = ["|".join(row) for row in top3_labels_a]

# sub_answer_only = pd.DataFrame({"row_id": test.row_id.values, "Category:Misconception": joined_preds_a})
# sub_answer_only.to_csv("submission_answer_only.csv", index=False)

# del trainer_a, model_a, predictions_a
# cleanup()


# # --- EXPERIMENT #11: EXPLANATION-ONLY MODEL (DistilBERT) ---
# print("\n--- EXPERIMENT #11: Explanation-Only Model (DistilBERT) ---")
# DIR_B = f"ver_{VER}_distil_explain"
# os.makedirs(DIR_B, exist_ok=True)
# EPOCHS_B = 2

# def format_explain_only(row):
#     return f"Explanation: {row['StudentExplanation']}"

# train['text_explain_only'] = train.apply(format_explain_only, axis=1)
# test['text_explain_only'] = test.apply(format_explain_only, axis=1)

# tokenizer_b = AutoTokenizer.from_pretrained(distil_model_name)
# train_ds_b = Dataset.from_pandas(train_df[['text_explain_only', 'label']])
# val_ds_b = Dataset.from_pandas(val_df[['text_explain_only', 'label']])

# def tokenize_b(batch):
#     return tokenizer_b(batch["text_explain_only"], padding="max_length", truncation=True, max_length=MAX_LEN)

# train_ds_b = train_ds_b.map(tokenize_b, batched=True)
# val_ds_b = val_ds_b.map(tokenize_b, batched=True)
# train_ds_b.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
# val_ds_b.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# model = AutoModelForSequenceClassification.from_pretrained(
#     distil_model_name,
#     num_labels=n_classes,
#     device_map="auto"
# )

# with torch.no_grad():
#     dummy_input = tokenizer("dummy text", return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(model.device)
#     dummy_output = model(**dummy_input)
#     del dummy_input, dummy_output
# torch.cuda.synchronize()

# training_args = TrainingArguments(
#     output_dir = f"./{DIR}",
#     do_train=True,
#     do_eval=True,
#     eval_strategy="steps",
#     save_strategy="steps", #no for no saving 
#     num_train_epochs=EPOCHS,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=2,
#     learning_rate=2e-5,
#     logging_dir="./logs",
#     logging_steps=50,
#     save_steps=500,
#     eval_steps=250,
#     save_total_limit=1,
#     metric_for_best_model="map@3",
#     greater_is_better=True,
#     load_best_model_at_end=True,
#     report_to="none",
#     fp16=False,
#     torch_compile=False,
# )

# trainer_b = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_ds_b,
#     eval_dataset=val_ds_b,
#     tokenizer=tokenizer_b,
#     compute_metrics=compute_map3,
# )

# ds_test_b = Dataset.from_pandas(test[['text_explain_only']])
# ds_test_b = ds_test_b.map(tokenize_b, batched=True)

# # Warm-up prediction on small batch
# with torch.no_grad():
#     small_batch = ds_test_b.select(range(min(4, len(ds_test_b))))
#     trainer_b.predict(small_batch)

# predictions_b = trainer_b.predict(ds_test_b)
# probs_explain_only = torch.nn.functional.softmax(torch.tensor(predictions_b.predictions) / TEMPERATURE, dim=1).numpy()
# np.save("probs_explain_only.npy", probs_explain_only)

# top3_b = np.argsort(-probs_explain_only, axis=1)[:, :3]
# flat_top3_b = top3_b.flatten()
# decoded_labels_b = le.inverse_transform(flat_top3_b)
# top3_labels_b = decoded_labels_b.reshape(top3_b.shape)
# joined_preds_b = ["|".join(row) for row in top3_labels_b]

# sub_explain_only = pd.DataFrame({"row_id": test.row_id.values, "Category:Misconception": joined_preds_b})
# sub_explain_only.to_csv("submission_explain_only.csv", index=False)

# del trainer_b, model, predictions_b
# cleanup()

# # --- Feature Engineering (with Exp #11 features) ---
# print("\n--- Feature Engineering for XGBoost ---")

# # Load all probs
# probs_gemma = np.load("probs_gemma.npy")
# probs_ettin = np.load("probs_ettin.npy")
# probs_modern = np.load("probs_modern.npy")
# probs_answer_only = np.load("probs_answer_only.npy")
# probs_explain_only = np.load("probs_explain_only.npy")

# # Test features
# test['gemma_top1_prob'] = np.max(probs_gemma, axis=1)
# test['gemma_pred_idx'] = np.argmax(probs_gemma, axis=1)
# sorted_probs_gemma = np.sort(probs_gemma, axis=1)[:, -2:]
# test['gemma_top2_prob'] = sorted_probs_gemma[:, -1]
# test['gemma_prob_diff'] = test['gemma_top1_prob'] - test['gemma_top2_prob']

# test['ettin_top1_prob'] = np.max(probs_ettin, axis=1)
# test['ettin_pred_idx'] = np.argmax(probs_ettin, axis=1)
# sorted_probs_ettin = np.sort(probs_ettin, axis=1)[:, -2:]
# test['ettin_top2_prob'] = sorted_probs_ettin[:, -1]
# test['ettin_prob_diff'] = test['ettin_top1_prob'] - test['ettin_top2_prob']

# test['modern_top1_prob'] = np.max(probs_modern, axis=1)
# test['modern_pred_idx'] = np.argmax(probs_modern, axis=1)
# sorted_probs_modern = np.sort(probs_modern, axis=1)[:, -2:]
# test['modern_top2_prob'] = sorted_probs_modern[:, -1]
# test['modern_prob_diff'] = test['modern_top1_prob'] - test['modern_top2_prob']

# # Exp #11 features
# test['answer_top1_prob'] = np.max(probs_answer_only, axis=1)
# test['answer_pred_idx'] = np.argmax(probs_answer_only, axis=1)
# sorted_probs_answer = np.sort(probs_answer_only, axis=1)[:, -2:]
# test['answer_top2_prob'] = sorted_probs_answer[:, -1]
# test['answer_prob_diff'] = test['answer_top1_prob'] - test['answer_top2_prob']

# test['explain_top1_prob'] = np.max(probs_explain_only, axis=1)
# test['explain_pred_idx'] = np.argmax(probs_explain_only, axis=1)
# sorted_probs_explain = np.sort(probs_explain_only, axis=1)[:, -2:]
# test['explain_top2_prob'] = sorted_probs_explain[:, -1]
# test['explain_prob_diff'] = test['explain_top1_prob'] - test['explain_top2_prob']

# # Train dummy features (OOF better, but quick)
# for model in ['gemma', 'ettin', 'modern', 'answer', 'explain']:
#     train[f'{model}_top1_prob'] = 0.8
#     train[f'{model}_pred_idx'] = train['label']
#     train[f'{model}_top2_prob'] = 0.7
#     train[f'{model}_prob_diff'] = 0.1

# # Perplexity
# PERPLEXITY_PATH = '/kaggle/input/perplexity-calculated/'
# try:
#     train['Perplexity'] = np.load(PERPLEXITY_PATH + 'train_perplexity.npy')
#     test['Perplexity'] = np.load(PERPLEXITY_PATH + 'test_perplexity.npy')
#     print("Train perplexity loaded from path.")
    
# except:
#     print("Train perplexity file not found. Using dummy.")
#     train['Perplexity'] = 50

# # TF-IDF
# dfall = pd.concat([train[['StudentExplanation']], test[['StudentExplanation']]], axis=0)
# tfidf = TfidfVectorizer(ngram_range=(1,1), min_df=5, max_features=1000)
# tfidf.fit(dfall['StudentExplanation'].fillna(''))
# tfidf_train = pd.DataFrame(tfidf.transform(train['StudentExplanation'].fillna('')).toarray(), columns=tfidf.get_feature_names_out())
# tfidf_test = pd.DataFrame(tfidf.transform(test['StudentExplanation'].fillna('')).toarray(), columns=tfidf.get_feature_names_out())

# # Simple features
# train['word_len'] = train['StudentExplanation'].str.split().str.len().fillna(0)
# test['word_len'] = test['StudentExplanation'].str.split().str.len().fillna(0)
# train['char_len'] = train['StudentExplanation'].str.len().fillna(0)
# test['char_len'] = test['StudentExplanation'].str.len().fillna(0)

# # Concat
# train_x = pd.concat([train, tfidf_train], axis=1)
# test_x = pd.concat([test, tfidf_test], axis=1)

# # tcols with all features
# tcols = [
#     'Perplexity', 'word_len', 'char_len',
#     'gemma_top1_prob', 'gemma_pred_idx', 'gemma_top2_prob', 'gemma_prob_diff',
#     'ettin_top1_prob', 'ettin_pred_idx', 'ettin_top2_prob', 'ettin_prob_diff',
#     'modern_top1_prob', 'modern_pred_idx', 'modern_top2_prob', 'modern_prob_diff',
#     'answer_top1_prob', 'answer_pred_idx', 'answer_top2_prob', 'answer_prob_diff',
#     'explain_top1_prob', 'explain_pred_idx', 'explain_top2_prob', 'explain_prob_diff'
# ] + list(tfidf.get_feature_names_out())

# print(f"Total {len(tcols)} features for XGBoost.")

# params = {'max_depth': 5,
#                'learning_rate': 0.017000951259946648, 
#                'subsample': 0.9497359326575187,
#                'colsample_bytree': 0.60100823235682, 
#                'lambda': 0.008218191844772233, 
#                'alpha': 0.0005746916692620666,
#                'device' : 'cuda',
#                'random_state' : 42,
#                'num_class' : n_classes,
#                'tree_method' : 'hist'
#          }

# # --- XGBoost Training ---
# print("\n--- Training XGBoost Super-Manager ---")

# skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# preds_xgb = np.zeros((len(test_x), n_classes))

# for fold, (trn_idx, val_idx) in enumerate(skf.split(train_x, train_x['label'])):
#     print(f"  Fold {fold+1}/3")
#     dtrain = xgb.DMatrix(train_x.loc[trn_idx, tcols], label=train_x.loc[trn_idx, 'label'])
#     dvalid = xgb.DMatrix(train_x.loc[val_idx, tcols], label=train_x.loc[val_idx, 'label'])
#     model = xgb.train(params, dtrain, num_boost_round=1000,
#                       evals=[(dvalid, 'valid')],
#                       early_stopping_rounds=50, verbose_eval=0)
#     preds_xgb += model.predict(xgb.DMatrix(test_x[tcols]), iteration_range=(0, model.best_iteration)) / skf.n_splits

# print("XGBoost complete.")

# # XGB submission
# top3_xgb = np.argsort(-preds_xgb, axis=1)[:, :3]
# flat_top3_xgb = top3_xgb.flatten()
# decoded_labels_xgb = le.inverse_transform(flat_top3_xgb)
# top3_labels_xgb = decoded_labels_xgb.reshape(top3_xgb.shape)
# joined_preds_xgb = ["|".join(row) for row in top3_labels_xgb]

# sub_xgb = pd.DataFrame({"row_id": test.row_id.values, "Category:Misconception": joined_preds_xgb})
# sub_xgb.to_csv("submission_xgb.csv", index=False)

# # --- EXPERIMENT #2: BLENDING XGB + GEMMA ---
# print("\n--- Blending XGB + Gemma ---")

# w_xgb = 0.65
# w_gemma = 0.35

# final_probs = (w_xgb * preds_xgb) + (w_gemma * probs_gemma)

# def gettop3_from_probs(probs, le):
#     top3 = np.argsort(-probs, axis=1)[:, :3]
#     flat_top3 = top3.flatten()
#     decoded_labels = le.inverse_transform(flat_top3)
#     top3_labels = decoded_labels.reshape(top3.shape)
#     return ["|".join(row) for row in top3_labels]

# final_preds = gettop3_from_probs(final_probs, le)
# sub_ensembled = pd.DataFrame({"row_id": test.row_id.values, "Category:Misconception": final_preds})
# sub_ensembled.to_csv('submission_ensembled.csv', index=False)
# display(sub_ensembled.head())

# # --- Hierarchical Ensemble (with all models) ---
# print("\n--- Hierarchical Ensemble ---")

# df_gemma = pd.read_csv('submission_gemma.csv').rename(columns={'Category:Misconception': 'Category:Misconception_gemma'})
# df_ettin = pd.read_csv('submission_ettin.csv').rename(columns={'Category:Misconception': 'Category:Misconception_ettin'})
# df_modern = pd.read_csv('submission_modern.csv').rename(columns={'Category:Misconception': 'Category:Misconception_modern'})
# df_xgb = pd.read_csv('submission_xgb.csv').rename(columns={'Category:Misconception': 'Category:Misconception_xgb'})
# df_answer = pd.read_csv('submission_answer_only.csv').rename(columns={'Category:Misconception': 'Category:Misconception_answer'})
# df_explain = pd.read_csv('submission_explain_only.csv').rename(columns={'Category:Misconception': 'Category:Misconception_explain'})
# df_ensembled = pd.read_csv('submission_ensembled.csv').rename(columns={'Category:Misconception': 'Category:Misconception_ensembled'})


# df_ensemble = df_gemma.merge(df_ettin, on='row_id').merge(df_modern, on='row_id').merge(df_xgb, on='row_id').merge(df_answer, on='row_id').merge(df_explain, on='row_id').merge(df_ensembled, on='row_id')

# def get_top_k_voting_ensemble(row, models, weights, k=3):
#     score = defaultdict(float)
#     for i, model_name in enumerate(models):
#         predictions_str = str(row[f'Category:Misconception_{model_name}'])
#         predictions = predictions_str.replace('|', ' ').split()
#         for rank, item in enumerate(predictions):
#             if not item: continue
#             rank_score = (len(predictions) - rank)
#             score[item] += rank_score * weights[i]
#     if not score: return "True_Correct:NA False_Neither:NA False_Misconception:Incomplete"
#     sorted_items = sorted(score.items(), key=lambda x: -x[1])
#     return ' '.join([item for item, _ in sorted_items[:k]])

# def hierarchical_decision_maker(row):
#     gemma_top = str(row['Category:Misconception_gemma']).split('|')[0]
#     ettin_top = str(row['Category:Misconception_ettin']).split('|')[0]
#     if gemma_top == ettin_top:
#         models = ['gemma', 'ettin']
#         weights = [5.0, 4.5]
#     else:
#         models = ['gemma', 'ettin', 'modern', 'xgb', 'answer', 'explain', 'ensembled']
#         weights = [5.0, 4.5, 4.5, 3.0, 2.0, 2.0, 2.0]
        
#     return get_top_k_voting_ensemble(row, models, weights)

# df_ensemble['Category:Misconception'] = df_ensemble.apply(hierarchical_decision_maker, axis=1)
# df_ensemble[['row_id', 'Category:Misconception']].to_csv('submission.csv', index=False)
# print("\nFinal submission.csv ready with All or Nothing!")

# agreement_count = (df_ensemble['Category:Misconception_gemma'].str.split('|').str[0] == df_ensemble['Category:Misconception_ettin'].str.split('|').str[0]).sum()
# total_rows = len(df_ensemble)
# print(f"Superstars agreed: {agreement_count}/{total_rows} ({agreement_count/total_rows:.2%}).")

