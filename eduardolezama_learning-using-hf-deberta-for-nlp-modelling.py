# This workbook is based on the Jeremy Howards "Getting started with NLP 
# for absolute beginners". Really great teacher! 


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pathlib import Path
import datasets
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Importing our pretrained model using automodel and local version
from transformers import AutoModelForSequenceClassification,AutoTokenizer
model_path = "/kaggle/input/deberta-v3-base/deberta_v3_base"



#Import our tokenizer depending on the selected model
#tokenizer = AutoTokenizer.from_pretrained(model_nlp, legacy=False)
#The tokenizer and the model has to be imported locally

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, legacy=False)


#Check if the tokenizer works
ids=tokenizer.tokenize("G'day my dudes!, I am Paquito Chocolatero from Wasipati")
ids


#Now, transformers models works with ID, which are numbers.
ids = tokenizer.encode("G'day my dudes!, I am Paquito Chocolatero from Wasipati")
ids


#Now let's decode to see how it works
tokenizer.decode(ids)


#Importing data
path = Path('../input/us-patent-phrase-to-phrase-matching')
df = pd.read_csv(path/"train.csv")
df


#A bit of info regarding the training set
df.describe(include="object")


#Check the distribution of context
df.context.value_counts()


#Join each row text to form a input to our model
df["input"] = "CPC_Classification: " + df.context + '; TEXT_TARGET: ' + df.target + '; TEXT_ANCHOR: ' + df.anchor
df.head()


#Transformers from HF uses a more optimized dataset object, called Dataset, lets use them
from datasets import Dataset,DatasetDict


ds = Dataset.from_pandas(df)
ds


#If you use the tokenizer variable directly and pass it a text, it will encode it automatically
tokenizer("Hello my old friend")


#Now let's create a func to tokenize our text
def tokenizer_func(x): return tokenizer(x["input"])


#Tokenize all the "input", which will create a new column
tokenized_ds = ds.map(tokenizer_func, batched=True)


#Therefor, we have 3 new columns, which results of applying tokenizer() to any phrase, as above
tokenized_ds


tokenized_ds = tokenized_ds.rename_columns({'score':'labels'})


#This is a simple random splitting, not the best option
train_valid_ds = tokenized_ds.train_test_split(0.25, seed=11)
train_valid_ds


#This is a splitting taking into consideration context and scoring proportions
from sklearn.model_selection import GroupShuffleSplit

#To use GroupShuffleSplit we have to work with a pandas dataset, not a dataset from HF
df1 = tokenized_ds.to_pandas()

#Make a division by context using GroupShuffleSplit
gss = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=11)
#Returns the indexes for training (train_idx) and validation (val_idx) rows. This is a generator.
train_idx, val_idx = next(gss.split(df1, groups=df1['context']))
#Create the df using the indexes generate above
train_df = df1.iloc[train_idx]
val_df = df1.iloc[val_idx]

#Transform again to dataset from HF
train_df = Dataset.from_pandas(train_df)
val_df = Dataset.from_pandas(val_df)


eval_df = pd.read_csv(path/'test.csv')
eval_df.describe()


# As we already know, we have to convert this 1 line text and 
# change to type dataset from HF
eval_df['input'] = 'CPC_Classification: ' + eval_df.context + '; TEXT_TARGET: ' + eval_df.target + '; TEXT_ANCHOR: ' + eval_df.anchor
eval_ds = Dataset.from_pandas(eval_df).map(tokenizer_func, batched=True)



#Function to calculate correlation of Pearson
def corr(x,y): return np.corrcoef(x,y)[0][1]

#Function to be used as metric
def corr_d(eval_pred): return {'pearson': corr(*eval_pred)}


from transformers import TrainingArguments,Trainer


bs = 128
epochs = 4


lr = 8e-5



args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    eval_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, weight_decay=0.01, report_to='none')



#Since we created 2 ways of diving our train and eval, we can change that parameter here

model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1, ignore_mismatched_sizes=True)
# using random splitter of data
#trainer = Trainer(model, args, train_dataset=train_valid_ds['train'], eval_dataset=train_valid_ds['test'],
#                  tokenizer=tokenizer, compute_metrics=corr_d)

#using splitting taking in consideration cotext
trainer = Trainer(model, args, train_dataset=train_df, eval_dataset=val_df,
                  tokenizer=tokenizer, compute_metrics=corr_d)


trainer.train();


preds = trainer.predict(eval_ds).predictions.astype(float)
preds


#Some of our predictions are above 1 and below 0, so we need to fix them:

preds = np.clip(preds, 0, 1)
preds


#Convert preds to values and not list on a list
preds = [float(p[0]) if isinstance(p, list) else float(p) for p in preds]

submission = datasets.Dataset.from_dict({
    'id': eval_ds['id'],
    'score': preds
})

submission.to_csv('submission.csv', index=False)

