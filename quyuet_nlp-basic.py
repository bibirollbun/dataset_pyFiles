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


import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import pandas as pd

train_df = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv')
sample_submission_df = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/sample_submission.csv')

print(train_df.head())
print(sample_submission_df.head())


train_df.info()


print(train_df.isnull().sum())


print(train_df.shape)


print(train_df.describe())


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

mean = 0.362
std = 0.258

x = np.linspace(mean - 4*std, mean + 4*std, 100)
y = stats.norm.pdf(x, mean, std)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Phân phối chuẩn', color='blue')

plt.fill_between(x, y, where=(x >= mean-std) & (x <= mean+std), color='green', alpha=0.5, label='68% (1σ)')
plt.fill_between(x, y, where=(x >= mean-2*std) & (x <= mean+2*std), color='yellow', alpha=0.5, label='95% (2σ)')
plt.fill_between(x, y, where=(x >= mean-3*std) & (x <= mean+3*std), color='red', alpha=0.5, label='99.7% (3σ)')

plt.axvline(mean, color='black', linestyle='--', label='Mean (0.362)')
plt.title('Phân phối chuẩn và quy tắc 68-95-99.7')
plt.xlabel('Giá trị')
plt.ylabel('Mật độ xác suất')
plt.legend()
plt.show()



import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
train_df['score'].hist(bins=20, edgecolor='black')
plt.title('Phân phối của cột "score"')
plt.xlabel('Score')
plt.ylabel('Tần suất')
plt.show()

plt.figure(figsize=(8, 6))
plt.boxplot(train_df['score'])
plt.title('Boxplot của cột "score"')
plt.xlabel('Score')
plt.show()



print('Skewness:', train_df['score'].skew())
print('Kurtosis:', train_df['score'].kurtosis())



train_df.describe(include='object')


train_df['input'] = 'TEXT1: ' + train_df.context + '; TEXT2: ' + train_df.target + '; ANC1: ' + train_df.anchor


train_df.input.head()


! pip install -q datasets


from datasets import Dataset,DatasetDict

ds = Dataset.from_pandas(train_df)


ds


model_nm = 'microsoft/deberta-v3-small'


from transformers import AutoModelForSequenceClassification,AutoTokenizer
tokz = AutoTokenizer.from_pretrained(model_nm)


tokz.tokenize("G'day folks, I'm Le Minh Quy from UET!")


def tok_func(x): return tokz(x["input"])


tok_ds = ds.map(tok_func, batched=True)


row = tok_ds[0]
row['input'], row['input_ids']


tokz.vocab['▁of']


tok_ds = tok_ds.rename_columns({'score':'labels'})


eval_df = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv')


eval_df.describe()


dds = tok_ds.train_test_split(0.25, seed=42)
dds


eval_df['input'] = 'TEXT1: ' + eval_df.context + '; TEXT2: ' + eval_df.target + '; ANC1: ' + eval_df.anchor
eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)


from transformers import TrainingArguments,Trainer


bs = 128
epochs = 4


lr = 8e-5


args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    evaluation_strategy="epoch", logging_strategy='epoch', per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, weight_decay=0.01, report_to='none')


import numpy as np

def corr(x,y): return np.corrcoef(x,y)[0][1]
    
def corr_d(eval_pred): return {'pearson': corr(*eval_pred)}


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

