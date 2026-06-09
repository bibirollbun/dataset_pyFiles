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


import pandas as pd
trn_df = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv')
tst_df = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv')
trn_df


trn_df.describe(include='object')


trn = trn_df.drop(['id', 'score'], axis=1)
tst = tst_df.drop(['id'], axis=1)
# tst = tst_df.drop(['id'])
trn, tst


trn_list = trn.to_dict(orient='records')
tst_list = tst.to_dict(orient='records')
trn_list[:2]


trn_txt = [str(row) for row in trn_list]
tst_txt = [str(row) for row in tst_list]
trn_txt[:2], tst_txt[:2]


trn_df['input'] = trn_txt
tst_df['input'] = tst_txt
trn_df.input.head(), tst_df.input.head()


# trn_df.fillna('none', inplace=True)
# tst_df.fillna('none', inplace=True)

# trn_df['text'] = trn_df.text.astype(str)
# tst_df['text'] = tst_df.text.astype(str)


# save as input
# trn_df['input'] = trn_df.text
# tst_df['input'] = tst_df.text
# trn_df.input.head(), tst_df.input.head()


trn_df


from datasets import Dataset,DatasetDict

trn_ds = Dataset.from_pandas(trn_df)
tst_ds = Dataset.from_pandas(tst_df)
trn_ds, tst_ds


model_nm = 'microsoft/deberta-v3-small'  # LB 0.82592
# model_nm = 'microsoft/DeBERTa-v3-base'  # LB 0.80968
# model_nm = 'microsoft/deberta-v3-xsmall'  # LB 0.81489
from transformers import AutoModelForSequenceClassification,AutoTokenizer
tokz = AutoTokenizer.from_pretrained(model_nm)


def tok_func(x): 
    return tokz(x["input"])

trn_tok_ds = trn_ds.map(tok_func, batched=True)
tst_tok_ds = tst_ds.map(tok_func, batched=True)


trn_tok_ds = trn_tok_ds.rename_columns({'score':'labels'})


dds = trn_tok_ds.train_test_split(0.01, seed=42)
dds


from transformers import TrainingArguments,Trainer
# bs = 128  # out-of-mem
bs = 32
# epochs = 4
# epochs = 1  # LB 0.82592
# epochs = 2  # LB 0.82470
# epochs = 3  # LB 0.

epochs = 1  # LB 0.

lr = 8e-5


# args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, 
#                          lr_scheduler_type='cosine', 
# #                          fp16=True,
#     evaluation_strategy="epoch", 
#                          per_device_train_batch_size=bs, 
#                          per_device_eval_batch_size=bs*2,
#     num_train_epochs=epochs, 
#                          weight_decay=0.01, 
#                          report_to='none')

args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, 
                         lr_scheduler_type='cosine', 
                         fp16=True,
    eval_strategy="epoch", 
                         per_device_train_batch_size=bs, 
                         per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, 
                         weight_decay=0.01, 
                         report_to='none')


# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
from sklearn.metrics import f1_score
# def corr(x,y): 
#     return np.corrcoef(x,y)[0][1]

# def corr_d(eval_pred): 
#     return {'pearson': corr(*eval_pred)}

# def my_metric(eval_pred): 
#     return {'F1': f1_score(*eval_pred, zero_division=0)}

def my_metric(eval_pred): 
    return {'F1': f1_score(*eval_pred, zero_division=0)}


model = AutoModelForSequenceClassification.from_pretrained(
    model_nm, num_labels=2)

trainer = Trainer(
    model, args, 
    train_dataset=dds['train'], 
    eval_dataset=dds['test'],
                  tokenizer=tokz, 
#     compute_metrics=my_metric
)


# ?Trainer


trainer.train();


preds = trainer.predict(tst_tok_ds)
preds


# preds = trainer.predict(tst_ds).predictions.astype(float)
preds.predictions


class_preds = np.argmax(preds.predictions, axis=-1)
class_preds


sub = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/sample_submission.csv')
sub


sub['score'] = class_preds
sub


# 1 epoch = LB 0.82592  'microsoft/deberta-v3-small' 1 epochs
name = os.path.basename(model_nm)
sub.to_csv(f'submission_{name}_ep{epochs}.csv', index=False)  
sub.to_csv(f'submission.csv', index=False)  

