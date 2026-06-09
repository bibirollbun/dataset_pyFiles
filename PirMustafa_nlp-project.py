from pathlib import Path

path = Path('/kaggle/input/us-patent-phrase-to-phrase-matching')


!ls {path}


import pandas as pd

df = pd.read_csv(path/'train.csv')


df


df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor


df


from datasets import Dataset, DatasetDict

ds = Dataset.from_pandas(df)


ds


# from transformers import AutoTokenizer, AutoModel

# model_nm = "microsoft/deberta-v3-small"
# tokenizer = AutoTokenizer.from_pretrained(model_nm)
# model = AutoModel.from_pretrained(model_nm)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_nm = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_nm)


# from transformers import AutoModelForSequenceClassification, AutoTokenizer
tokz = AutoTokenizer.from_pretrained(model_nm)


tokz.tokenize('Hey, its me sumair your buddy')


def tok_func(x): return tokz(x['input'])


tok_ds = ds.map(tok_func, batched=True)


row = tok_ds[0]
row['input'], row['input_ids']


tok_ds = tok_ds.rename_columns({'score' : 'labels'})


eval_df = pd.read_csv(path/'test.csv')
eval_df.describe()


dds = tok_ds.train_test_split(0.25, seed=42)
dds


eval_df['input'] = 'TEXT1: ' + eval_df.context + '; TEXT2: ' + eval_df.target + '; ANC1: ' + eval_df.anchor
eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)


from scipy.stats import pearsonr

def corr(preds, labels):
    return pearsonr(preds, labels)[0]

def corr_d(eval_pred):
    preds, labels = eval_pred
    return {'pearson': corr(preds, labels)}


from transformers import TrainingArguments,Trainer


bs = 128
epochs = 4
lr = 8e-5


args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', 
                         fp16=True, eval_strategy='epoch', per_device_train_batch_size=bs, 
                         per_device_eval_batch_size=bs*2, num_train_epochs=epochs, weight_decay=0.01, report_to='none')
                         


from transformers import AutoModelForSequenceClassification,Trainer


model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                 tokenizer=tokz, compute_metrics=corr_d
                 )


trainer.train()


preds = trainer.predict(eval_ds).predictions.astype(float)


import numpy as np
preds = np.clip(preds, 0, 1)


preds




