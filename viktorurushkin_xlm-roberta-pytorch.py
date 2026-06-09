# !pip install transformers -qq
# !pip install accelerate -qq
# !pip install evaluate -qq
# !pip install peft -qq
# !pip install datasets -qq


import copy
import gc
# from itertools import chain
import os
import pickle
import random
import time
from typing import Dict, List, Tuple, Union
import warnings


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from datasets import Dataset
from transformers import XLMRobertaConfig, XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from transformers import pipeline
import datasets
from sklearn.metrics import mean_squared_error
import torch


os.environ["WANDB_DISABLED"] = "true"


SEED: int = 42
MAX_LENGTH = 256
MODEL_NAME: str = '/kaggle/input/pretrained-transformers/xlm_roberta_base_model'
TOKENIZER_NAME: str = '/kaggle/input/pretrained-transformers/xlm_roberta_base_tokenizer'
DATA_DIR: str = '/kaggle/input/commonlitreadabilityprize'
MODEL_DIR: str  = '/kaggle/working'
MINIBATCH_SIZE: int = 32


random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}


train_df_path = os.path.join(DATA_DIR, 'train.csv')
test_df_path = os.path.join(DATA_DIR, 'test.csv')

assert os.path.isfile(train_df_path)
assert os.path.isfile(test_df_path)
print(train_df_path)


train_df = pd.read_csv(train_df_path)
test_df = pd.read_csv(test_df_path)


display(train_df.sample(5, random_state=SEED))
display(test_df.sample(5, random_state=SEED))


train_df['target'].describe()


sns.histplot(data=train_df, x='target')
plt.show()


train_df['text_sym_len'] = train_df['excerpt'].str.len()
train_df['text_token_len'] = train_df['excerpt'].str.split().map(len)


train_df[['target', 'text_sym_len', 'text_token_len']].corr().iloc[0:1]


train_df['text_token_len'].describe()


train_ds = Dataset.from_pandas(train_df[['excerpt', 'target', 'standard_error']])
test_ds = Dataset.from_pandas(test_df[['excerpt']])


print(train_ds)
print(test_ds)


tokenizer = XLMRobertaTokenizerFast.from_pretrained(TOKENIZER_NAME)


tokenizer.encode('Hello world!')


tokenizer


tokenized_train_ds = train_ds.map(
    lambda it: tokenizer(it['excerpt'], truncation=True, max_length=MAX_LENGTH),
    batched=True, batch_size=16
)

tokenized_test_ds = test_ds.map(
    lambda it: tokenizer(it['excerpt'], truncation=True, max_length=MAX_LENGTH),
    batched=True, batch_size=2
)


print(tokenized_train_ds['input_ids'][0])
print(tokenizer.convert_ids_to_tokens(tokenized_train_ds['input_ids'][0]))


tokenized_train_ds = tokenized_train_ds.remove_columns(['standard_error', 'excerpt'])


tokenized_train_ds = tokenized_train_ds.rename_columns({'target': 'label'})


tokenized_ds = tokenized_train_ds.train_test_split(test_size=0.1, seed=SEED)


tokenized_ds


model = XLMRobertaForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=1, device_map="auto", torch_dtype="auto"
)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


training_args = TrainingArguments(
    output_dir=MODEL_DIR + '/xlm_roberta',
    learning_rate=2e-5,
    per_device_train_batch_size=MINIBATCH_SIZE,
    per_device_eval_batch_size=MINIBATCH_SIZE,
    num_train_epochs=15,
    weight_decay=1e-3,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    save_total_limit=3,
    logging_steps = 5,
    # "warmup_ratio": 1/10,
    # "lr_scheduler_type": "cosine",
    fp16=True,
    data_seed=SEED,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds['train'],
    eval_dataset=tokenized_ds['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


trainer.train()


trainer.evaluate()


trainer.evaluate(tokenized_ds['train'])


sample_path = os.path.join(DATA_DIR, 'sample_submission.csv')


ss = pd.read_csv(sample_path)
ss['target'] = trainer.predict(tokenized_test_ds).predictions
ss


ss.to_csv('submission.csv',index=False)


trainer.save_model("xlm-roberta-rdbl/")




