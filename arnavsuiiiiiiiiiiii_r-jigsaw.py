# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df=pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
dt=pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
samp=pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")


df.info()


dt.info()


df.head()


dt.head()


import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

# Load data
df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv").drop(columns=['row_id'])
dt = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
ids = dt['row_id']
dt=dt.drop(columns=['row_id'])
samp = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv")

# Dataset class
class RedditDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.body = df['body'].tolist()
        self.rule = df['rule'].tolist()
        self.labels = df['rule_violation'].tolist() if 'rule_violation' in df.columns else None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.body)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.body[idx],
            self.rule[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# Tokenizer and model
model_name = "/kaggle/input/huggingfacedebertav3variants/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Train/val split
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['rule_violation'], random_state=42)

train_dataset = RedditDataset(train_df, tokenizer)
val_dataset = RedditDataset(val_df, tokenizer)
test_dataset = RedditDataset(dt, tokenizer)

# Compute AUC
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy().squeeze()
    auc = roc_auc_score(labels, probs)
    return {"AUC": auc}

# Training arguments
DIR = "deberta-v3-results"
EPOCHS = 3

training_args = TrainingArguments(
    output_dir=f"./{DIR}",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=5e-5,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=50,
    eval_steps=50,
    save_total_limit=1,
    metric_for_best_model="AUC",     # <--- use AUC
    greater_is_better=True,          # <--- higher AUC is better
    load_best_model_at_end=True,
    report_to="none",
    bf16=False,
    fp16=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Inference
preds = trainer.predict(test_dataset).predictions
probs = torch.sigmoid(torch.tensor(preds)).numpy().squeeze()

# Submission
submission = pd.DataFrame({"row_id": ids, "rule_violation": probs})
submission.to_csv("submission.csv", index=False)



submission

