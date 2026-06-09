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


train = pd.read_csv("/kaggle/input/toxic-comments-classification-2023/train_data.csv")
test = pd.read_csv("/kaggle/input/toxic-comments-classification-2023/test_data.csv")
sample_submission = pd.read_csv("/kaggle/input/toxic-comments-classification-2023/sample_submission.csv")


train.head(), test.head()


from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict

model_name = "FacebookAI/xlm-roberta-base"
# distilbert-base-multilingual-cased
# h4g3n/distilbert-mini-multilingual-cased

tokenizer = AutoTokenizer.from_pretrained(model_name)

train_ds = train.copy()
train_ds["input"] = train_ds["comment"]
train_ds["labels"] = train_ds["toxic"]
train_ds = train_ds.drop(["comment", "toxic"], axis=1)

test_ds = test.copy()
test_ds["input"] = test_ds["comment"]
test_ds = test_ds.drop(["comment"], axis=1)

train_ds = Dataset.from_pandas(train_ds)
eval_ds = Dataset.from_pandas(test_ds)

def tknize(text): return tokenizer(text["input"])

train_ds = train_ds.map(tknize, batched=True)
eval_ds = eval_ds.map(tknize, batched=True)


from sklearn.metrics import roc_auc_score
from scipy.special import softmax

train_ds = train_ds.train_test_split(0.25, seed=42)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = np.argmax(logits, axis=-1)
    auc = roc_auc_score(labels, probs)
    
    return {
        "auc": auc
    }


# from sklearn.metrics import accuracy_score

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = np.argmax(logits, axis=1)
#     acc = accuracy_score(labels, preds)
#     return {
#         "accuracy": acc
#     }


# from sklearn.metrics import f1_score

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = np.argmax(logits, axis=1)
#     f1 = f1_score(labels, preds, average='weighted')  # or 'macro', 'micro', etc.
#     return {
#         "f1": f1
#     }


batch_size=32
epochs=3

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1) # here you put the number of labels to predict

args = TrainingArguments(
    "outputs", 
    learning_rate=8e-5, 
    warmup_ratio=0.1, 
    lr_scheduler_type="cosine", 
    fp16=True,
    eval_strategy="epoch",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size*2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to='none'
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds["train"],
    eval_dataset=train_ds["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()




