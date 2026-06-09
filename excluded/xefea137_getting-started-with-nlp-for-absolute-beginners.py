from pathlib import Path


path = Path('../input/us-patent-phrase-to-phrase-matching')
!ls {path}


import pandas as pd


df = pd.read_csv(path/'train.csv')
df


df.describe(include='object')


df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor


df['input'][0]


df.head(3)


from datasets import Dataset, DatasetDict

ds = Dataset.from_pandas(df)
ds, ds[0], df.iloc[0]


# model_nm = 'microsoft/deberta-v3-small'
# model_nm = 'microsoft/deberta-v3-large'
model_path = '/kaggle/input/huggingfacedebertav3variants/deberta-v3-small'


from transformers import AutoModelForSequenceClassification, AutoTokenizer

# tokz = AutoTokenizer.from_pretrained(model_nm)
tokz = AutoTokenizer.from_pretrained(model_path)


tokz.tokenize("G'day folks, I'm Jeremy from fast.ai!")


tokz.tokenize("A platypus is an ornithorhynchus anatinus.")


def tok_func(x):
    return tokz(x["input"])


tok_func({"input": "A platypus is an ornithorhynchus anatinus."})


tok_ds = ds.map(tok_func, batched=True)


row = tok_ds[0]
row['input'], row['input_ids']


tokz.vocab['▁of'], tokz.vocab['1']


tok_ds = tok_ds.rename_columns({'score':'labels'})


eval_df = pd.read_csv(path/'test.csv')
eval_df.describe()


dds = tok_ds.train_test_split(0.25, seed=42)
dds


eval_df['input'] = 'TEXT1: ' + eval_df.context + '; TEXT2: ' + eval_df.target + '; ANC1: ' + eval_df.anchor
eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)


import numpy as np
np.set_printoptions(precision=2, suppress=True)


def corr(x, y):
    return np.corrcoef(x, y)[0][1]


def corr_d(eval_pred):
    return {'pearson': corr(*eval_pred)}


from transformers import TrainingArguments, Trainer


bs = 128
# bs = 32
epochs = 4
lr = 8e-5


args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, weight_decay=0.01, report_to='none')


# model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                  tokenizer=tokz, compute_metrics=corr_d)


trainer.train()


preds = trainer.predict(eval_ds).predictions.astype(float)
preds


preds = np.clip(preds, 0, 1)
preds


import datasets

submission = datasets.Dataset.from_dict({
    'id': eval_ds['id'],
    'score': np.squeeze(preds)
})

submission.to_csv('submission.csv', index=False)
submission.to_pandas().head()

