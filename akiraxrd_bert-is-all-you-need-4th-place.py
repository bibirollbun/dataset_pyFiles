import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils import data

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import Dataset

from sklearn.metrics import f1_score

# basic
import os
import warnings
import random
from pathlib import Path
from tqdm.notebook import tqdm

print('Imports successful')


SEED = 31
BATCH_SIZE = 256
EPOCHS = 3
CHECKPOINT = "google-bert/bert-base-uncased"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = Path('/kaggle/input/iaio-2026-sf-r-comments-classification')
TRAIN_PATH = BASE_DIR / 'train.csv'
TEST_PATH = BASE_DIR / 'new_test.csv'
SAMPLE_PATH = BASE_DIR / 'simple_submission.csv'

warnings.simplefilter('ignore')
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

if SAMPLE_PATH.exists():
    print('Configs successful')
else:
    print('Something\'s ain\'t right..')


DEVICE


# load main data
df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)
df_sample = pd.read_csv(SAMPLE_PATH)

df_train


# fetch labels & their amount
CLASSES = df_train.columns.tolist()[2:]
NUM_CLASSES = len(CLASSES)


# save the text columns in respective variables
TEXT = 'comment_text'
train_series = df_train[TEXT].copy()
test_series = df_test[TEXT].copy()

train_texts = train_series.values
test_texts = test_series.values


# tokenizer. nothin special..
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)


# main tokenization function
def tokenize(texts, tokenizer):
    res = []

    for text in tqdm(texts):
        out = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            #return_tensors='pt'
        )
        res.append(out)
    return res


# data dicts with the following form: list[dict[str, torch.Tensor]]
traindict = tokenize(train_texts, tokenizer)


# check whether input ids are padded correctly
np.array(traindict[0]['input_ids']).shape==np.array(traindict[1]['input_ids']).shape


# merges the tokenizer output dicts in the lists into a single dict
def mapdict(lst: list[dict[str, torch.Tensor]], labels=None) -> dict[str, torch.Tensor]:
    res = {
            'input_ids': [],
            'attention_mask': [],
        }
    if labels is not None:
        res['label'] = torch.from_numpy(labels)
        
    for d in tqdm(lst):
        res['input_ids'].append(d['input_ids'])
        res['attention_mask'].append(d['attention_mask'])

    return res

train_mapped_dict = mapdict(lst=traindict, labels=df_train[CLASSES].values)


print(train_mapped_dict['label'])


# create a labeled dataset (train+valid) and split it
labeledset = Dataset.from_dict(train_mapped_dict).train_test_split(test_size=0.2, seed=SEED)
labeledset


# save the splits in corresponding variables
trainset, validset = labeledset['train'], labeledset['test']


# load a pretrained HF-transformer from a checkpoint
model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT,
    num_labels=NUM_CLASSES,
    problem_type="multi_label_classification"
).to(DEVICE)


# define training arguments (first two are essential !!!)
train_args = TrainingArguments(
    report_to="none",
    push_to_hub=False,
    
    #weight_decay=0.01,
    logging_strategy='steps',
    #learning_rate=2e-5,
    do_train=True,
    do_eval=False,
    logging_steps=50,
    
    num_train_epochs=2,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    torch_empty_cache_steps=1,
    fp16 = torch.cuda.is_available(),
    label_names=CLASSES,
    
)


# define the trainer, obviously ..
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=trainset
)

trainer


trainer.train()


# fetch the test data 
df_test = pd.read_csv(BASE_DIR / 'new_test.csv')
test_texts = df_test[TEXT].values
testdict = tokenize(test_texts, tokenizer)
test_mapped_dict = mapdict(testdict)
testset = Dataset.from_dict(test_mapped_dict)
testset


preds = trainer.predict(testset)


final_preds = preds.predictions


# apply sigmoid-round to the multilabel logits
final_preds = torch.round(torch.sigmoid(torch.from_numpy(final_preds))).to(torch.int32).numpy()


subm = pd.DataFrame({
    'id': df_test['id']
})

subm[CLASSES] = final_preds
subm


subm.to_csv('subm_alpha.csv', index=False)

