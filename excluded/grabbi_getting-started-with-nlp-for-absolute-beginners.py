import os
from pathlib import Path
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')


if iskaggle:
    path = Path('../input/us-patent-phrase-to-phrase-matching')
    ! pip install -q datasets


!ls {path}


import pandas as pd


df = pd.read_csv(path/'train.csv')


df


df.describe(include='object')


df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor


df.input.head()


from datasets import Dataset,DatasetDict
ds = Dataset.from_pandas(df)


ds


model_nm = 'microsoft/deberta-v3-small'


from transformers import AutoModelForSequenceClassification, AutoTokenizer
tokz = AutoTokenizer.from_pretrained(model_nm)


tokz.tokenize("Welcome to the show!")


tokz.tokenize("A platypus is an ornithorhynchus anatinus.")


def tok_func(x): return tokz(x["input"])


tok_ds = ds.map(tok_func, batched=True)


row = tok_ds[0]
row['input'], row['input_ids']


tokz.vocab['▁of']


tok_ds = tok_ds.rename_columns({'score':'labels'})


tok_ds


eval_df = pd.read_csv(path/'test.csv')
eval_df.describe()


dds = tok_ds.train_test_split(0.25, seed=42)
dds


eval_df['input'] = 'TEXT1: ' + eval_df.context + '; TEXT2: ' + eval_df.target + '; ANC1: ' + eval_df.anchor
eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)


import numpy as np
def corr(x,y): return np.corrcoef(x,y)[0][1]
def corr_d(eval_pred): return {'pearson': corr(*eval_pred)}


from transformers import TrainingArguments,Trainer


bs = 128
epochs = 4


lr = 8e-5


args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, weight_decay=0.01, report_to='none')


model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                  tokenizer=tokz, compute_metrics=corr_d)


trainer.train();


preds = trainer.predict(eval_ds).predictions.astype(float)
preds


preds = np.clip(preds, 0, 1)


preds


import datasets

submission = datasets.Dataset.from_dict({
    'id': eval_ds['id'],
    'score': preds
})

submission.to_csv('submission.csv', index=False)

