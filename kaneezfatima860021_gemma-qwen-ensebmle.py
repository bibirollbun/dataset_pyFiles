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


import os  

# Set which GPUs to use (here: GPU 0 and 1).  
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Versioning variable for experiment tracking  
VER = 1  

# Choose model name or path:  
# Uncomment below if using HuggingFace model hub  

# Using local Kaggle dataset path instead  
model_name = "/kaggle/input/gemma2-9b-it-cv945"  

# Number of training epochs  
EPOCHS = 2  

# Create an output directory for this version  
DIR = f"ver_{VER}"  
os.makedirs(DIR, exist_ok=True)  



import pandas as pd  
import numpy as np  
from sklearn.preprocessing import LabelEncoder  

# Initialize label encoder for target variable encoding
le = LabelEncoder()

# Load training dataset
train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')

# Handle missing values in 'Misconception' by replacing NaN with 'NA'
train["Misconception"] = train["Misconception"].fillna("NA")

# Create a combined target string (Category + Misconception)
train["target"] = train["Category"] + ":" + train["Misconception"]

# Encode target labels into numeric form
train["label"] = le.fit_transform(train["target"])

# Store classes and total number of unique target labels
target_classes = le.classes_
n_classes = len(target_classes)

# Print dataset shape and number of classes
print(f"Train shape: {train.shape} with {n_classes} target classes")

# Display first few rows for quick inspection
train.head()



# Identify rows where Category starts with 'True'
idx = train.apply(lambda row: row["Category"].split("_")[0], axis=1) == "True"

# Subset data with these rows
correct = train.loc[idx].copy()

# Count how many times each (QuestionId, MC_Answer) pair appears
correct["c"] = correct.groupby(["QuestionId", "MC_Answer"])["MC_Answer"].transform("count")

# Sort by count in descending order (most common answers first)
correct = correct.sort_values("c", ascending=False)

correct = correct.drop_duplicates(["QuestionId"])

correct = correct[["QuestionId", "MC_Answer"]]

correct["is_correct"] = 1

train = train.merge(correct, on=["QuestionId", "MC_Answer"], how="left")

# Fill missing values with 0 (not correct)
train["is_correct"] = train["is_correct"].fillna(0)



from IPython.display import display, Latex  



# Count occurrences of each (QuestionId, MC_Answer) pair
tmp = train.groupby(["QuestionId", "MC_Answer"]).size().reset_index(name="count")

# Assign rank to answers within each question based on frequency
# rank starts at 0 (most common answer = rank 0)
tmp["rank"] = (
    tmp.groupby("QuestionId")["count"]
       .rank(method="dense", ascending=False)
       .astype(int) - 1
)

# Drop raw counts (no longer needed)
tmp = tmp.drop("count", axis=1)

# Sort by QuestionId and rank for readability
tmp = tmp.sort_values(["QuestionId", "rank"])

# Unique question IDs
Q = tmp.QuestionId.unique()

for q in Q:
    # Extract question text (first occurrence per ID)
    question = train.loc[train.QuestionId == q].iloc[0].QuestionText
    
    # Get all answer choices for this question
    choices = tmp.loc[tmp.QuestionId == q].MC_Answer.values
    
    # Label choices A, B, C, D...
    labels = "ABCD"
    choice_str = " ".join(
        [f"({labels[i]}) {choice}" for i, choice in enumerate(choices)]
    )
    
    print()  # spacing between outputs
    display(Latex(f"QuestionId {q}: {question}"))
    display(Latex(f"MC Answers: {choice_str}"))



import torch  
from transformers import AutoTokenizer  
from sklearn.model_selection import train_test_split  
from datasets import Dataset  
import numpy as np  

# Load tokenizer for the chosen model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Max sequence length for tokenization
# (truncate/pad sequences to this length)
MAX_LEN = 256



def format_input(row):
    """
    Format a single training row into a structured text prompt
    for the language model.
    """
    # Label as "Yes" or "No" depending on correctness
    x = "Yes" if row["is_correct"] else "No"
    
    # Build the prompt
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct? {x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

# Apply formatter to each row to create the text input column
train["text"] = train.apply(format_input, axis=1)

# Quick sanity check — print one example prompt
print("Example prompt for our LLM:\n")
print(train["text"].values[0])



# ------------------------------------------------
# CHECK TOKEN LENGTH DISTRIBUTION
# ------------------------------------------------

# Compute token lengths for each formatted text (no truncation applied yet)
lengths = [len(tokenizer.encode(t, truncation=False)) for t in train["text"]]

import matplotlib.pyplot as plt  

# Plot histogram of token lengths
plt.hist(lengths, bins=50)  
plt.title("Token Length Distribution")  
plt.xlabel("Number of Tokens")  
plt.ylabel("Frequency")  
plt.grid(True)  
plt.show()



# ------------------------------------------------
# CHECK HOW MANY SAMPLES EXCEED MAX SEQUENCE LENGTH
# ------------------------------------------------

# Count samples longer than MAX_LEN
L = (np.array(lengths) > MAX_LEN).sum()
print(f"There are {L} train sample(s) with more than {MAX_LEN} tokens")

# Show sorted token lengths (to inspect distribution tail)
np.sort(lengths)



# ------------------------------------------------
# TRAIN/VALIDATION SPLIT
# ------------------------------------------------

# Split into training and validation sets (80/20 split)
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

# ------------------------------------------------
# CONVERT TO HUGGING FACE DATASETS
# ------------------------------------------------

# Keep only relevant columns for the model
COLS = ["text", "label"]

# Convert pandas DataFrames to Hugging Face Datasets
train_ds = Dataset.from_pandas(train_df[COLS])
val_ds   = Dataset.from_pandas(val_df[COLS])



# ------------------------------------------------
# TOKENIZATION FUNCTION
# ------------------------------------------------
def tokenize(batch):
    """
    Tokenize a batch of text samples.
    Applies padding, truncation, and sets max sequence length.
    """
    return tokenizer(
        batch["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=256
    )

# Apply tokenizer to train/val datasets
train_ds = train_ds.map(tokenize, batched=True)
val_ds   = val_ds.map(tokenize, batched=True)

# ------------------------------------------------
# CONVERT TO TORCH FORMAT
# ------------------------------------------------
# Keep only the relevant columns for model training
columns = ["input_ids", "attention_mask", "label"]
train_ds.set_format(type="torch", columns=columns)
val_ds.set_format(type="torch", columns=columns)



from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer  

# ------------------------------------------------
# LOAD PRETRAINED MODEL FOR SEQUENCE CLASSIFICATION
# ------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "/kaggle/input/gemma2-9b-it-bf16",  # pretrained model path
    num_labels=n_classes,               # number of output classes
    torch_dtype=torch.bfloat16,         # use bfloat16 for efficiency on modern GPUs
    device_map="auto",                  # automatically place model across available devices
)



from peft import PeftModel  

# ------------------------------------------------
# LOAD PEFT (Parameter-Efficient Fine-Tuning) MODEL
# ------------------------------------------------
# Wrap the base model with PEFT weights (LoRA/adapters/etc.)
model = PeftModel.from_pretrained(
    model,        # base model (already loaded above)
    model_name    # path to PEFT weights (local or hub)
)



training_args = TrainingArguments(
    # ------------------------------------------------
    # OUTPUT & LOGGING
    # ------------------------------------------------
    output_dir=f"./{DIR}",   # save outputs/checkpoints here
    logging_dir="./logs",    # where to store training logs
    logging_steps=50,        # log every 50 steps
    report_to="none",        # disable external reporting (e.g. wandb)

    # ------------------------------------------------
    # TRAINING / EVAL CONTROL
    # ------------------------------------------------
    do_train=True,           # enable training
    do_eval=True,            # enable evaluation
    num_train_epochs=EPOCHS, # number of full training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=16,   # batch size for evaluation
    learning_rate=2e-5,      # optimizer learning rate

    # ------------------------------------------------
    # SAVE / EVAL STRATEGY
    # ------------------------------------------------
    eval_strategy="steps",   # run evaluation every N steps
    eval_steps=200,          # evaluate every 200 steps
    save_strategy="steps",   # save checkpoint every N steps
    save_steps=200,          # checkpoint frequency
    save_total_limit=1,      # keep only 1 checkpoint (latest/best)
    load_best_model_at_end=True,     # reload best checkpoint when done
    metric_for_best_model="map@3",   # custom metric for best model
    greater_is_better=True,          # higher metric = better

    # ------------------------------------------------
    # PRECISION SETTINGS (GPU DEPENDENT)
    # ------------------------------------------------
    bf16=False,  # set True if running locally on Ampere+ GPU (supports bfloat16)
    fp16=True,   # Kaggle T4 GPUs support fp16 but NOT bf16 → use fp16 here
)



# ------------------------------------------------
# CUSTOM MAP@3 METRIC
# ------------------------------------------------
# MAP@3 = Mean Average Precision at rank 3
# Reward full score if correct answer is rank 1,
# half if rank 2, one-third if rank 3, else 0.
# ------------------------------------------------

from sklearn.metrics import average_precision_score  # (not used here, but could be for extension)

def compute_map3(eval_pred):
    """
    Compute MAP@3 (Mean Average Precision at rank 3).

    Args:
        eval_pred: tuple of (logits, labels)
            - logits: raw model outputs (batch_size x num_classes)
            - labels: true class indices

    Returns:
        dict with {"map@3": value}
    """
    logits, labels = eval_pred

    # Convert logits → probabilities
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    # Top-3 predicted class indices per sample
    top3 = np.argsort(-probs, axis=1)[:, :3]  

    # Match matrix: True if top-k prediction == ground truth
    match = (top3 == labels[:, None])

    # Compute MAP@3
    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:       # correct @ rank 1
            map3 += 1.0
        elif match[i, 1]:     # correct @ rank 2
            map3 += 1.0 / 2
        elif match[i, 2]:     # correct @ rank 3
            map3 += 1.0 / 3

    return {"map@3": map3 / len(labels)}



# ------------------------------------------------
# HUGGING FACE TRAINER SETUP
# ------------------------------------------------
trainer = Trainer(
    model=model,                 # model with classification head
    args=training_args,          # training arguments we defined earlier
    train_dataset=train_ds,      # training dataset (tokenized + torch format)
    eval_dataset=val_ds,         # validation dataset
    tokenizer=tokenizer,         # tokenizer (ensures consistency in saving/loading)
    compute_metrics=compute_map3 # custom MAP@3 metric for evaluation
)

# Uncomment below to start training:
# trainer.train()



# ------------------------------------------------
# LOAD TEST DATA
# ------------------------------------------------
test = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")

# Show basic info
print("Test set shape:", test.shape)

# Peek at first few rows
test.head()



# ------------------------------------------------
# ADD "is_correct" FLAG TO TEST SET
# ------------------------------------------------
# Merge with the "correct" answers derived earlier
test = test.merge(correct, on=["QuestionId", "MC_Answer"], how="left")

# Fill NaNs → 0 (not correct)
test["is_correct"] = test["is_correct"].fillna(0)

# ------------------------------------------------
# FORMAT TEST DATA INTO LLM-FRIENDLY PROMPTS
# ------------------------------------------------
test["text"] = test.apply(format_input, axis=1)

# Quick preview
test.head()



# ------------------------------------------------
# CONVERT TEST SET TO HUGGING FACE DATASET
# ------------------------------------------------
ds_test = Dataset.from_pandas(test[["text"]])

# Apply same tokenization as train/val
ds_test = ds_test.map(tokenize, batched=True)

# ------------------------------------------------
# RUN INFERENCE
# ------------------------------------------------
# Use Trainer's predict method on test dataset
predictions = trainer.predict(ds_test)

# Convert raw logits → probabilities
probs = torch.nn.functional.softmax(
    torch.tensor(predictions.predictions), dim=1
).numpy()



# ------------------------------------------------
# GET TOP-3 PREDICTED CLASSES
# ------------------------------------------------
# Sort probabilities in descending order and take top-3 indices
top3 = np.argsort(-probs, axis=1)[:, :3]   # shape: [num_samples, 3]

# Decode numeric indices back to original string labels
flat_top3 = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)

# Reshape back to (num_samples, 3)
top3_labels = decoded_labels.reshape(top3.shape)

# Join top-3 predictions per row (pipe-separated)
joined_preds = ["|".join(row) for row in top3_labels]

# ------------------------------------------------
# BUILD & SAVE SUBMISSION FILE
# ------------------------------------------------
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})

sub.to_csv("submission_gemma.csv", index=False)

# Preview submission
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


torch.cuda.empty_cache()
gc.collect()

torch.cuda.ipc_collect()

print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, ModernBertForSequenceClassification, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset


train               = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test                = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')


le                  = LabelEncoder()
train.Misconception     = train.Misconception.fillna('NA')
train['target']   = train.Category + ':' +train.Misconception
train['label']    = le.fit_transform(train['target'])

n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")


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

test['text'] = test.apply(format_input,axis=1)


ds_test = Dataset.from_pandas(test)



model = AutoModelForSequenceClassification.from_pretrained("/kaggle/input/deekseepmath-7b-map-competition/MAP_EXP_09_FULL", device_map="cuda:0", torch_dtype=torch.bfloat16)


tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/deekseepmath-7b-map-competition/MAP_EXP_09_FULL")
model.config.pad_token_id = tokenizer.pad_token_id

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

ds_test = ds_test.map(tokenize, batched=True)


test_args = TrainingArguments(
    output_dir="./",
    do_train=False,
    do_predict=True,
    per_device_eval_batch_size=16, # Adjust as needed
    bf16=False, # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU          
    fp16=True,
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=test_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

predictions = trainer.predict(ds_test)

predictions.predictions


top3           = np.argsort(-predictions.predictions, axis=1)[:, :]
flat_top3      = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels_cat    = decoded_labels.reshape(top3.shape)
top3_labels_cat


joined_preds = []

for preds in top3_labels_cat:
    joined_preds.append("|".join(preds))



# Save submission
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission_deepseek.csv", index=False)
sub.head()


sub.iloc[0]['Category:Misconception']



import torch
import gc

del top3_labels_cat, flat_top3, decoded_labels, top3, test, ds_test
del test_args, model, trainer, predictions
# Delete any other lingering references
for obj in list(globals().keys()):
    if isinstance(globals()[obj], torch.nn.Module) or isinstance(globals()[obj], torch.Tensor):
        del globals()[obj]

torch.cuda.empty_cache()
gc.collect()

torch.cuda.ipc_collect()

print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, ModernBertForSequenceClassification, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset


train               = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
test                = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')


le                  = LabelEncoder()
train.Misconception     = train.Misconception.fillna('NA')
train['target']   = train.Category + ':' +train.Misconception
train['label']    = le.fit_transform(train['target'])

n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")


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

test['text'] = test.apply(format_input,axis=1)


ds_test = Dataset.from_pandas(test)



model = AutoModelForSequenceClassification.from_pretrained("/kaggle/input/qwen3-8b-map-competition/MAP_EXP_16_FULL", device_map="cuda:1", torch_dtype=torch.bfloat16)


model


tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/qwen3-8b-map-competition/MAP_EXP_16_FULL")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

ds_test = ds_test.map(tokenize, batched=True)


test_args = TrainingArguments(
    output_dir="./",
    do_train=False,
    do_predict=True,
    per_device_eval_batch_size=1, # Adjust as needed
    bf16=False, # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU          
    fp16=True,
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=test_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer)
)

predictions = trainer.predict(ds_test)

predictions.predictions


top3           = np.argsort(-predictions.predictions, axis=1)[:, :]
flat_top3      = top3.flatten()
decoded_labels = le.inverse_transform(flat_top3)
top3_labels_cat    = decoded_labels.reshape(top3.shape)
top3_labels_cat


joined_preds = []

for preds in top3_labels_cat:
    joined_preds.append("|".join(preds))



# Save submission
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "Category:Misconception": joined_preds
})
sub.to_csv("submission_gemma3.csv", index=False)
sub.head()


sub.iloc[0]['Category:Misconception']



from collections import defaultdict

def get_top_k_ensemble(l1, l2, l3, k=3):
    list1, list2, list3 = l1.split('|'), l2.split('|'), l3.split('|')
    weights = [4, 4, 4]  # độ tin cậy: list1 < list2 > list3 
    lists = [list1, list2, list3]
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

print(get_top_k_ensemble(list1, list2, list3, k=3))


df1 = pd.read_csv('submission_gemma.csv').rename(columns = {'Category:Misconception':'Category:Misconception_gemma'})
df2 = pd.read_csv('submission_deepseek.csv').rename(columns = {'Category:Misconception':'Category:Misconception_deepseek'})
df3 = pd.read_csv('submission_gemma3.csv').rename(columns = {'Category:Misconception':'Category:Misconception_gemma3'})


df = pd.merge(df1, df2, on = 'row_id', how = 'inner')
df = pd.merge(df, df3, on = 'row_id', how = 'inner')


df['Category:Misconception'] = df.apply(lambda x: get_top_k_ensemble(x['Category:Misconception_gemma'], x['Category:Misconception_deepseek'], x['Category:Misconception_gemma3']), axis = 1)
df[['row_id', 'Category:Misconception']].to_csv('submission.csv', index = False)
pd.read_csv('submission.csv')

