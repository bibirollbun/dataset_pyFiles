import pandas as pd
import datasets
from pathlib import Path
from datasets import Dataset,DatasetDict
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from numpy import corrcoef
from transformers import TrainingArguments,Trainer
from scipy.stats import pearsonr
import numpy as np
path = Path.cwd()



df = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv')
df.describe(include='object')


df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor
df.input.head()


ds = Dataset.from_pandas(df)
ds


model_nm = 'microsoft/deberta-v3-small'
tokz = AutoTokenizer.from_pretrained(model_nm)


def tok_func(x): return tokz(x["input"])
tok_ds = ds.map(tok_func, batched=True)
tok_ds = tok_ds.rename_columns({'score':'labels'})


eval_df = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv')
eval_df.describe()


dds = tok_ds.train_test_split(0.25, seed=42)
dds


eval_df['input'] = 'TEXT1: ' + eval_df.context + '; TEXT2: ' + eval_df.target + '; ANC1: ' + eval_df.anchor
eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)


def corr_d(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    pearson_corr, _ = pearsonr(predictions, labels)
    pearson_corr = float(pearson_corr)
    
    return {"pearson": pearson_corr} # Had issues with other functions so I wrote this instead.
bs = 128
epochs = 5
lr = 8e-5


args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    eval_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, weight_decay=0.01, report_to='none')


model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)
trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                  processing_class=tokz, compute_metrics=corr_d)


trainer.train();


preds = trainer.predict(eval_ds).predictions.astype(float)
preds = np.clip(preds, 0, 1)
preds


import datasets

submission = datasets.Dataset.from_dict({
    'id': eval_ds['id'],
    'score': preds
})

submission.to_csv('submission.csv', index=False)

