import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

VER=1
#model_name = "google/gemma-2-9b-it"
model_name = "/kaggle/input/gemma2-9b-it-cv945"
EPOCHS = 2

DIR = f"ver_{VER}"
os.makedirs(DIR, exist_ok=True)


import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder

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
    "/kaggle/input/gemma2-9b-it-bf16",
    num_labels=n_classes,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


from peft import PeftModel
model = PeftModel.from_pretrained(model, model_name)


training_args = TrainingArguments(
    output_dir = f"./{DIR}",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps", #no for no saving 
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
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


#trainer.save_model(f"ver_{VER}")      
#tokenizer.save_pretrained(f"ver_{VER}")


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
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()


# Get top 3 predicted class indices
top3 = np.argsort(-probs, axis=1)[:, :]   # shape: [num_samples, 3]

# Decode numeric class indices to original string labels
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)

# Join 3 labels per row with space
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
del training_args, train_ds, val_ds, model, trainer, predictions, probs
# Delete any other lingering references
for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


# Delete any other lingering references
for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


# Delete any other lingering references
for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


# Delete any other lingering references
for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
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
    per_device_train_batch_size=16*2,
    per_device_eval_batch_size=32*2,
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
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()


# Get top 3 predicted class indices
top3 = np.argsort(-probs, axis=1)[:, :]   # shape: [num_samples, 3]

# Decode numeric class indices to original string labels
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)

# Join 3 labels per row with space
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
del training_args, train_ds, val_ds, model, trainer, predictions, probs
# Delete any other lingering references
for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
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
    per_device_train_batch_size=16*2,
    per_device_eval_batch_size=32*2,
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
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()


# Get top 3 predicted class indices
top3 = np.argsort(-probs, axis=1)[:, :]   # shape: [num_samples, 3]

# Decode numeric class indices to original string labels
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels = decoded_labels.reshape(top3.shape)

# Join 3 labels per row with space
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

del top3_labels, flat_top3, decoded_labels, top3, test, ds_test
del training_args, train_ds, val_ds, model, trainer, predictions, probs
# Delete any other lingering references
for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

# Dọn sạch autograd
torch.cuda.empty_cache()
gc.collect()

# Nếu dùng nhiều GPU, làm thêm bước này để clear hết:
torch.cuda.ipc_collect()

# In ra kiểm tra
print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


df=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
dt=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
samp=pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")


df.info()


import numpy as np
import pandas as pd
import re
import nltk
import warnings
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from scipy import sparse
import xgboost as xgb
from lightgbm import LGBMClassifier

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
warnings.filterwarnings('ignore')


import sys
import gc

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from datasets import Dataset
from tqdm.auto import tqdm
tqdm.pandas()
from transformers import PreTrainedTokenizerFast

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

LOWERCASE = False
VOCAB_SIZE = 14000000

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModel

# === Load trước các mô hình NLP ngoài ===
embed_model = SentenceTransformer('/kaggle/input/sentencetransformersallminilml6v2')

deberta_tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/deberta-xsmall-map/ver_1/best")
deberta_model = AutoModel.from_pretrained("/kaggle/input/deberta-xsmall-map/ver_1/best")
deberta_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deberta_model.to(device)

deberta_small_tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/debertav3base-map/ver_1/best")
deberta_small_model = AutoModel.from_pretrained("/kaggle/input/debertav3base-map/ver_1/best")
deberta_small_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deberta_small_model.to(device)

deberta_large_tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/experiment-no-1/ver_1/best")
deberta_large_model = AutoModel.from_pretrained("/kaggle/input/experiment-no-1/ver_1/best")
deberta_large_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deberta_large_model.to(device)

roberta_tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/roberta-base-map/ver_1/best")
roberta_model = AutoModel.from_pretrained("/kaggle/input/roberta-base-map/ver_1/best")
roberta_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
roberta_model.to(device)

roberta_large_tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/roberta-large-map/best")
roberta_large_model = AutoModel.from_pretrained("/kaggle/input/roberta-large-map/best")
roberta_large_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
roberta_large_model.to(device)

print('Done load model!')
# MAP@3 metric
def map3(target_list, pred_list):
    score = 0.
    for t, p in zip(target_list, pred_list):
        if t == p[0]: score += 1.
        elif len(p) > 1 and t == p[1]: score += 1/2
        elif len(p) > 2 and t == p[2]: score += 1/3
    return score / len(target_list)

def advanced_clean(text):
    text = re.sub(r'(\d+)\s*/\s*(\d+)', r'FRAC_\1_\2', text)
    text = re.sub(r'\\frac\{([^\}]+)\}\{([^\}]+)\}', r'FRAC_\1_\2', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s_]', '', text)
    return text.strip().lower()

def extract_math_features(text):
    features = {
        'frac_count': len(re.findall(r'FRAC_\d+_\d+|\\frac', text)),
        'number_count': len(re.findall(r'\b\d+\b', text)),
        'operator_count': len(re.findall(r'[\+\-\*\/\=]', text))
    }
    return features

def fast_lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

# Tập các từ khóa toán học thường gặp
math_keywords = {'add', 'subtract', 'multiply', 'divide', 'fraction', 'equation', 
                 'solve', 'equal', 'product', 'sum', 'difference', 'denominator', 'numerator'}

def has_math_keywords(text):
    tokens = text.lower().split()
    return int(any(word in tokens for word in math_keywords))

def add_similarity_features(df):
    """Tính độ tương đồng cosine giữa lời giải thích và câu hỏi"""
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    texts = df['StudentExplanation'].fillna('') + df['QuestionText'].fillna('')
    tfidf_matrix = tfidf.fit_transform(texts)
    
    expl_matrix = tfidf.transform(df['StudentExplanation'].fillna(''))
    ques_matrix = tfidf.transform(df['QuestionText'].fillna(''))

    sim_scores = (expl_matrix.multiply(ques_matrix)).sum(axis=1).A1
    df['explanation_question_sim'] = sim_scores
    return df

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
           torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

def get_deberta_embedding(texts, model, model_tokenizer, batch_size=64):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        encoded_input = model_tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        all_embeddings.append(embeddings.cpu())  # đưa về CPU nếu cần concat
    return torch.cat(all_embeddings).cpu().numpy()


def format_input(row):
    x = "This answer is correct."
    if not row['is_correct']:
        x = "This answer is incorrect."

    extra = (
        f"Additional Info: "
        f"The explanation has {row['explanation_len']} characters "
        f"and includes {row['mc_frac_count']} fraction(s)."
    )

    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}\n"
        f"{extra}"
    )

def create_features(df):
    df['mc_answer_len'] = df['MC_Answer'].astype(str).str.len()
    df['explanation_len'] = df['StudentExplanation'].astype(str).str.len()
    df['question_len'] = df['QuestionText'].astype(str).str.len()
    df['explanation_to_question_ratio'] = df['explanation_len'] / (df['question_len'] + 1)

    for col in ['QuestionText', 'MC_Answer']:
        math_features = df[col].apply(extract_math_features).apply(pd.Series)
        prefix = 'mc_' if col == 'MC_Answer' else ''
        math_features.columns = [f'{prefix}{c}' for c in math_features.columns]
        df = pd.concat([df, math_features], axis=1)

    # ==== Đặc trưng từ vựng ====
    df['explanation_word_count'] = df['StudentExplanation'].astype(str).apply(lambda x: len(x.split()))
    df['explanation_avg_word_len'] = df['StudentExplanation'].astype(str).apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.strip() else 0
    )
    df['explanation_has_math_terms'] = df['StudentExplanation'].astype(str).apply(has_math_keywords)

    # ==== Dấu câu ====
    df['explanation_punctuation_count'] = df['StudentExplanation'].astype(str).apply(
        lambda x: len(re.findall(r'[\.!?]', x))
    )

    # ==== Độ tương đồng cosine giữa giải thích và câu hỏi ====
    df = add_similarity_features(df)

    # === Thêm các feature Semantic Mismatch ===
    print("Encoding sentence embeddings...")
    combined = (
        "Question: " + df["QuestionText"].astype(str) + 
        " Answer: " + df["MC_Answer"].astype(str) + 
        " Explanation: " + df["StudentExplanation"].astype(str)
    ).tolist()

    # Lấy embedding dạng numpy (n_samples, 384)
    embeddings = embed_model.encode(combined, show_progress_bar=True, batch_size=64)
    emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    # === Embedding từ E5 và GTE ===
    print("Encoding sentence embeddings from deberta xsmall...")
    
    deberta_embeddings = get_deberta_embedding(combined, deberta_model, deberta_tokenizer, batch_size=64)
    deberta_df = pd.DataFrame(deberta_embeddings, columns=[f"deberta_emb_{i}" for i in range(deberta_embeddings.shape[1])])
    df = pd.concat([df.reset_index(drop=True), deberta_df], axis=1)

    # === Embedding từ E5 và GTE ===
    print("Encoding sentence embeddings from deberta base...")
    
    deberta_embeddings = get_deberta_embedding(combined, deberta_small_model, deberta_small_tokenizer, batch_size=32)
    deberta_df = pd.DataFrame(deberta_embeddings, columns=[f"deberta_small_emb_{i}" for i in range(deberta_embeddings.shape[1])])
    df = pd.concat([df.reset_index(drop=True), deberta_df], axis=1)

    print("Encoding sentence embeddings from roberta base...")
    
    roberta_embeddings = get_deberta_embedding(combined, roberta_model, roberta_tokenizer, batch_size=64)
    roberta_df = pd.DataFrame(roberta_embeddings, columns=[f"roberta_emb_{i}" for i in range(roberta_embeddings.shape[1])])
    df = pd.concat([df.reset_index(drop=True), roberta_df], axis=1)

    print("Encoding sentence embeddings from deberta large...")
    df['large_text'] = df.apply(format_input, axis = 1)
    deberta_embeddings = get_deberta_embedding(df['large_text'].tolist(), deberta_large_model, deberta_large_tokenizer, batch_size=16)
    deberta_df = pd.DataFrame(deberta_embeddings, columns=[f"deberta_large_emb_{i}" for i in range(deberta_embeddings.shape[1])])
    df = pd.concat([df.reset_index(drop=True), deberta_df], axis=1)
    df = df.drop('large_text', axis = 1)

    print("Encoding sentence embeddings from roberta large...")
    
    roberta_embeddings = get_deberta_embedding(combined, roberta_large_model, roberta_large_tokenizer, batch_size=16)
    roberta_df = pd.DataFrame(roberta_embeddings, columns=[f"roberta_large_emb_{i}" for i in range(roberta_embeddings.shape[1])])
    df = pd.concat([df.reset_index(drop=True), roberta_df], axis=1)
    
    return df

# Load data
train = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
sample = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv")

train['Misconception'] = train['Misconception'].fillna('NA')
train['target_cat'] = train['Category'] + ':' + train['Misconception']
train = train.sort_values('target_cat').reset_index(drop=True)


idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)
test = test.merge(correct, on=['QuestionId','MC_Answer'], how='left')
test.is_correct = test.is_correct.fillna(0)

le = LabelEncoder()
train['target_encoded'] = le.fit_transform(train['target_cat'])
target_classes = le.classes_
n_classes = len(target_classes)

train = create_features(train)
test = create_features(test)

train['combined_text'] = "Question: " + train['QuestionText'] + " Answer: " + train['MC_Answer'] + " Explanation: " + train['StudentExplanation']
test['combined_text'] = "Question: " + test['QuestionText'] + " Answer: " + test['MC_Answer'] + " Explanation: " + test['StudentExplanation']

train['cleaned_text'] = train['combined_text'].apply(advanced_clean).apply(fast_lemmatize)
test['cleaned_text'] = test['combined_text'].apply(advanced_clean).apply(fast_lemmatize)

# Creating Byte-Pair Encoding tokenizer
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
# Adding normalization and pre_tokenizer
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
# Adding special tokens and creating trainer instance
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
# Creating huggingface dataset object
dataset = Dataset.from_pandas(test[['cleaned_text']])
def train_corp_iter(): 
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["cleaned_text"]
raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
tokenized_texts_test = []

for text in tqdm(test['cleaned_text'].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(text))

tokenized_texts_train = []

for text in tqdm(train['cleaned_text'].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))

# TF-IDF
tfidf_word = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=5000)
tfidf_word.fit(pd.concat([train['cleaned_text'], test['cleaned_text']]))

train_tfidf_word = tfidf_word.transform(train['cleaned_text'])
test_tfidf_word = tfidf_word.transform(test['cleaned_text'])

tfidf_expl = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=3000)
tfidf_expl.fit(pd.concat([train['StudentExplanation'], test['StudentExplanation']]))
train_tfidf_expl = tfidf_expl.transform(train['StudentExplanation'])
test_tfidf_expl = tfidf_expl.transform(test['StudentExplanation'])

char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=3000)
char_tfidf.fit(pd.concat([train['cleaned_text'], test['cleaned_text']]))
train_char = char_tfidf.transform(train['cleaned_text'])
test_char = char_tfidf.transform(test['cleaned_text'])

def dummy(text):
    return text

vectorizer = TfidfVectorizer(ngram_range=(1, 3), lowercase=False, sublinear_tf=True, 
                            analyzer = 'word',
                            tokenizer = dummy,
                            preprocessor = dummy,
                            token_pattern = None, strip_accents='unicode', max_features=1000
                            )

X_train_tfidf = vectorizer.fit_transform(tokenized_texts_train)
X_test_tfidf = vectorizer.transform(tokenized_texts_test)

del vectorizer
gc.collect()

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
    
del deberta_small_model, deberta_small_tokenizer
clear_gpu_memory()

del deberta_large_model, deberta_large_tokenizer
clear_gpu_memory()

del deberta_model, deberta_tokenizer
clear_gpu_memory()

del roberta_model, roberta_tokenizer
clear_gpu_memory()

del roberta_large_model, roberta_large_tokenizer
clear_gpu_memory()

# Numeric features
numeric_cols = [
    'mc_answer_len', 'explanation_len', 'question_len',
    'explanation_to_question_ratio', 'frac_count', 'number_count',
    'operator_count', 'mc_frac_count', 'mc_number_count', 'mc_operator_count',
    'explanation_word_count', 'explanation_avg_word_len', 'explanation_has_math_terms',
    'explanation_punctuation_count', 'explanation_question_sim', 'is_correct'] \
    + [f"emb_{i}" for i in range(384)] + [f"deberta_emb_{i}" for i in range(384)] + [f"deberta_small_emb_{i}" for i in range(768)] + [f"roberta_emb_{i}" for i in range(768)] + [f"deberta_large_emb_{i}" for i in range(1024)] + [f"roberta_large_emb_{i}" for i in range(1024)]
X_numeric = sparse.csr_matrix(train[numeric_cols].fillna(0).values)
X_numeric_test = sparse.csr_matrix(test[numeric_cols].fillna(0).values)

X_train = sparse.hstack([train_tfidf_word, train_tfidf_expl, train_char, X_numeric, X_train_tfidf])
X_test = sparse.hstack([test_tfidf_word, test_tfidf_expl, test_char, X_numeric_test, X_test_tfidf])

y = train['target_encoded'].values

# XGBoost KFold
oof_preds = np.zeros((len(train), n_classes))
test_preds = np.zeros((len(test), n_classes))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

params = {
    'objective': 'multi:softprob',
    'num_class': n_classes,
    'eval_metric': 'mlogloss',
    'max_depth': 10,
    'learning_rate': 0.05,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'random_state': 42
}

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y)):
    print(f"Fold {fold+1}")
    dtrain = xgb.DMatrix(X_train[trn_idx], label=y[trn_idx])
    dvalid = xgb.DMatrix(X_train[val_idx], label=y[val_idx])

    model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dvalid, 'valid')], early_stopping_rounds=50, verbose_eval=50)
    oof_preds[val_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration))
    test_preds += model.predict(xgb.DMatrix(X_test), iteration_range=(0, model.best_iteration)) / skf.n_splits

# MAP@3
oof_top3 = np.argsort(-oof_preds, axis=1)[:, :3]
oof_labels = [[le.inverse_transform([i])[0] for i in row] for row in oof_top3]
y_true = train['target_cat'].tolist()
map_score = map3(y_true, oof_labels)
print(f"\nValidation MAP@3: {map_score:.4f}")

# Prepare submission
top3_test = np.argsort(-test_preds, axis=1)[:, :]
preds = ['|'.join([le.inverse_transform([i])[0] for i in row]) for row in top3_test]
sample['Category:Misconception'] = preds
sample.to_csv("submission_tree.csv", index=False)
print("Saved submission.csv")


sample.iloc[0]['Category:Misconception']


from collections import defaultdict

def get_top_k_ensemble(l1, l2, l3, l4, k=3):
    list1, list2, list3, list4 = l1.split('|'), l2.split('|'), l3.split('|'), l4.split('|')
    weights = [4, 4, 1, 1]  # độ tin cậy: list1 > list2 > list3 > list4
    lists = [list1, list2, list3, list4]
    score = defaultdict(int)

    for i, lst in enumerate(lists):
        weight = weights[i]
        for rank, item in enumerate(lst):
            score[item] += (len(lst) - rank) * weight

    # Sắp xếp theo điểm giảm dần
    sorted_items = sorted(score.items(), key=lambda x: -x[1])
    return ' '.join([item for item, _ in sorted_items[:k]])

list1 = 'a|b|d|f'
list2 = 'b|c|a|e'
list3 = 'c|e|b'
list4 = 'f|a'

print(get_top_k_ensemble(list1, list2, list3, list4, k=3))


df1 = pd.read_csv('submission_gemma.csv').rename(columns = {'Category:Misconception':'Category:Misconception_gemma'})
df2 = pd.read_csv('submission_ettin.csv').rename(columns = {'Category:Misconception':'Category:Misconception_ettin'})
df3 = pd.read_csv('submission_modern.csv').rename(columns = {'Category:Misconception':'Category:Misconception_modern'})
df4 = pd.read_csv('submission_tree.csv').rename(columns = {'Category:Misconception':'Category:Misconception_tree'})


df = pd.merge(df1, df2, on = 'row_id', how = 'inner')
df = pd.merge(df, df3, on = 'row_id', how = 'inner')
df = pd.merge(df, df4, on = 'row_id', how = 'inner')

df['Category:Misconception'] = df.apply(lambda x: get_top_k_ensemble(x['Category:Misconception_gemma'], x['Category:Misconception_ettin'], x['Category:Misconception_modern'], x['Category:Misconception_tree']), axis = 1)
df[['row_id', 'Category:Misconception']].to_csv('submission.csv', index = False)
pd.read_csv('submission.csv')




