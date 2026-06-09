%%time

import random
import os
import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *
import re
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import warnings 
warnings.filterwarnings('ignore')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

version=1
model_name_1 = "/kaggle/input/jigsaw-deberta-small-cv-0-702"
EPOCHS = 4
MAX_LEN = 488
SEED = 42
CLASSES = 1

DIR = f"Jigsaw_{version}"
os.makedirs(DIR, exist_ok=True)
set_seed(SEED)

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


%%time

train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")

tokenizer = AutoTokenizer.from_pretrained(model_name_1)

def make_prompt(row):
    return f"""[RULE]: {row['rule']}
[SUBREDDIT]: {row['subreddit']}

[COMMENT]: {row['body']}

[POSITIVE EXAMPLES]:
1. {row['positive_example_1']}
2. {row['positive_example_2']}

[NEGATIVE EXAMPLES]:
1. {row['negative_example_1']}
2. {row['negative_example_2']}

[QUESTION]: Does the comment violate the rule?
[ANSWER]:"""
    
train['text'] = train.apply(make_prompt,axis=1)

train_, val_ = train_test_split(train, test_size=0.2, random_state=42)
train_["label"] = train_["rule_violation"].astype(float)
val_["label"] = val_["rule_violation"].astype(float)

features_cols = ['text','label']
train_ds = Dataset.from_pandas(train_[features_cols])
val_ds = Dataset.from_pandas(val_[features_cols])

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)
    
train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)


%%time

def compute_column_auc(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))

    if probs.ndim == 1 or probs.shape[1] == 1:
        auc = roc_auc_score(labels, probs)
        return {"auc": auc}

    aucs = []
    for i in range(probs.shape[1]):
        try:
            auc = roc_auc_score(labels[:, i], probs[:, i])
        except ValueError:
            auc = 0.5
        aucs.append(auc)
    return {"mean_column_auc": np.mean(aucs)}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name_1,
    num_labels=CLASSES,
)

training_args = TrainingArguments(
    output_dir=f"./{DIR}",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    logging_steps=50,
    save_total_limit=1,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=7,
    learning_rate=2e-5,
    num_train_epochs=EPOCHS,
    gradient_accumulation_steps=1, 
    load_best_model_at_end=True,
    metric_for_best_model="auc",
    greater_is_better=True,
    fp16=True,   
    bf16=False,  
    report_to="none",
    logging_dir="./logs",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_column_auc,
)

# If you want to train then uncomment and run this again 
# trainer.train()


%%time

results = trainer.evaluate()
print("Mean_column_AUC:", results['eval_auc']) 


%%time

trainer.save_model(f"Jigsaw_{version}")      
tokenizer.save_pretrained(f"Jigsaw_{version}")


%%time

test['text'] = test.apply(make_prompt,axis=1)

ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True)

predictions = trainer.predict(ds_test)
probs = torch.sigmoid(torch.tensor(predictions.predictions)).numpy().flatten()



version=1
model_name_2 = "/kaggle/input/jigsaw-deberta-base-model/Jigsaw_1"
EPOCHS = 4
MAX_LEN = 488
SEED = 42
CLASSES = 1


tokenizer = AutoTokenizer.from_pretrained(model_name_1)

def make_prompt(row):
    return f"""[RULE]: {row['rule']}
    [SUBREDDIT]: {row['subreddit']}
    
    [COMMENT]: {row['body']}
    
    [POSITIVE EXAMPLES]:
    1. {row['positive_example_1']}
    2. {row['positive_example_2']}
    
    [NEGATIVE EXAMPLES]:
    1. {row['negative_example_1']}
    2. {row['negative_example_2']}
    
    [QUESTION]: Does the comment violate the rule?
    [ANSWER]:"""


def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)





model = AutoModelForSequenceClassification.from_pretrained(
    model_name_2,
    num_labels=CLASSES,
)

training_args = TrainingArguments(
    output_dir=f"./{DIR}",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    logging_steps=50,
    save_total_limit=1,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=7,
    learning_rate=2e-5,
    num_train_epochs=EPOCHS,
    gradient_accumulation_steps=1, 
    load_best_model_at_end=True,
    metric_for_best_model="auc",
    greater_is_better=True,
    fp16=True,   
    bf16=False,  
    report_to="none",
    logging_dir="./logs",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_column_auc,
)



%%time

test['text'] = test.apply(make_prompt,axis=1)

ds_test = Dataset.from_pandas(test[['text']])
ds_test = ds_test.map(tokenize, batched=True)

predictions = trainer.predict(ds_test)
probs2 = torch.sigmoid(torch.tensor(predictions.predictions)).numpy().flatten()



probs  = (0.6*probs+0.4*probs2)
sub = pd.DataFrame({
    "row_id": test.row_id.values,
    "rule_violation": probs
})
sub.to_csv("submission.csv", index=False)
sub.head()




