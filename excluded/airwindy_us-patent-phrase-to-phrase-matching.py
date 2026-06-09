import pandas as pd



df = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv')
df.head()


df['input'] = 'Text1:' + df.context + '; Text2:' + df.target + '; Anc1:' + df.anchor



from datasets import Dataset, DatasetDict
ds = Dataset.from_pandas(df)


ds


model_nm = 'Microsoft/deberta-v3-small'


from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokz = AutoTokenizer.from_pretrained(model_nm)


def tokenize(x): return tokz(x['input'])


tok_ds = ds.map(tokenize, batched=True)


tok_ds = tok_ds.rename_columns({'score': 'labels'})


eval_df = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv')


dds = tok_ds.train_test_split(0.25, seed=42)


dds


eval_df['input'] = 'TEXT1: ' + eval_df.context + '; TEXT2: ' + eval_df.target + '; ANC1: ' + eval_df.anchor
eval_ds = Dataset.from_pandas(eval_df).map(tokenize, batched=True)


from transformers import TrainingArguments, Trainer


bs = 128
lr = 8e-5
epochs = 4


args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    eval_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, weight_decay=0.01, report_to='none')


import numpy as np
def corr(x,y): return np.corrcoef(x,y)[0][1]
def corr_d(eval_pred): return {'pearson': corr(*eval_pred)}


model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'], tokenizer=tokz, compute_metrics=corr_d)


trainer.train()


preds = trainer.predict(eval_ds).predictions.astype(float)


preds = np.clip(preds, 0, 1)


import datasets

submission = datasets.Dataset.from_dict({
    'id': eval_ds['id'],
    'score': preds
})

submission.to_csv('submission.csv', index=False)

