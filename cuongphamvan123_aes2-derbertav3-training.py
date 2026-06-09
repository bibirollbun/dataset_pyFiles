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


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


MODEL_NAME = '/kaggle/input/microsoftdeberta-v3-large/transformers/default/1/'
FOLD = 2
MAX_LENGTH = 1024


import random
import json
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm


def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(seed=42)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize(sample):
    return tokenizer(sample['full_text'], max_length=MAX_LENGTH, truncation=True)


df_train = pd.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv')
df_train["fold"] = 999
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for i, (_, val_index) in enumerate(skf.split(df_train, df_train["score"])):
    df_train.loc[val_index, "fold"] = i

df_train['labels'] = df_train.score.map(lambda x: x-1)


ds_train = Dataset.from_pandas(df_train[df_train.fold!=FOLD])
ds_eval = Dataset.from_pandas(df_train[df_train.fold==FOLD])

ds_train = ds_train.map(tokenize).remove_columns(['essay_id', 'full_text', 'score', 'fold', '__index_level_0__'])
ds_eval = ds_eval.map(tokenize).remove_columns(['essay_id', 'full_text', 'score', 'fold', '__index_level_0__'])


def compute_metrics(p):
    preds, labels = p
    score = cohen_kappa_score(labels, preds.argmax(-1), weights='quadratic')
    return { 'qwk':score }


train_args = TrainingArguments(
    output_dir='/kaggle/working/', 
    fp16=True,
    learning_rate=1e-5,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    report_to="none",
    eval_strategy="steps",
    do_eval=True,
    eval_steps=200,
    save_total_limit=3,
    save_strategy="steps",
    save_steps=200,
    logging_steps=200,
    lr_scheduler_type='linear',
    metric_for_best_model="qwk",
    greater_is_better=True,
    warmup_ratio=0.1,
    weight_decay=0.01,
    save_safetensors=True,
)


model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6)


trainer = Trainer(
    model=model, 
    args=train_args, 
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    data_collator=DataCollatorWithPadding(tokenizer), 
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)



trainer.train()


trainer.save_model("/kaggle/working/my_best_model")
tokenizer.save_pretrained("/kaggle/working/my_best_model")


! zip -r my_best_model.zip /kaggle/working/my_best_model

