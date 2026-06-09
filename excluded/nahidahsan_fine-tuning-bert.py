import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
import re, gc, os
import time
import pickle
pd.set_option('display.max_colwidth',None)
import warnings
warnings.filterwarnings('ignore')


# Process the data sets

def load_data():
    train = pd.read_csv('../input/quora/train.csv')
    train.drop_duplicates(keep='first')
    test  = pd.read_csv('../input/quora/test.csv')
    submission  = pd.read_csv('../input/quora/sample_submission.csv')
    return train, test, submission

train, test,_= load_data()


round(train['target'].value_counts(normalize=True)*100) # unbalanced data


train.sample(10)


# lowercase
train['question_text'] = train['question_text'].apply(lambda x:x.lower())
test['question_text'] = test['question_text'].apply(lambda x:x.lower())


def remove_qmark(s):
    return re.sub(r"[/?.,']+",'',s)

train['question_text'] = train['question_text'].apply(lambda x:remove_qmark(x))
test['question_text']  = test['question_text'].apply(lambda x:remove_qmark(x))


# remove whitespace
train['question_text'] = train['question_text'].apply(lambda x:' '.join(x.split()))
test['question_text'] = test['question_text'].apply(lambda x: ' '.join(x.split()))


train.rename(columns= {'qid':'idx','target':'label'},inplace =True)
test.rename(columns= {'qid':'idx'},inplace =True)


train.head(1)


train.info()


!pip install s3fs -q
!pip install fsspec==0.8.7 -qq
!pip install --no-index --find-links ../input/hf-datasets/wheels datasets -qq


import datasets
from datasets import Dataset


from sklearn.utils import shuffle
index = train[:128000].index
train = shuffle(train[:128000])
train.index = index
len(train)*.2


train.info()


df_train = train[:-25600].reset_index(drop=True) #156672 divisible by 64
df_valid = train[-25600:].reset_index(drop=True)


# choose small samples for mock training
df_train = df_train.sample(1000)
df_valid = df_valid.sample(200)



train_dataset = Dataset.from_pandas(df_train)
valid_dataset = Dataset.from_pandas(df_valid)
df_train.shape,df_valid.shape


train_dataset[0]


def change_transformers_dataset_2_right_format(dataset, label_name): 
    return dataset.map(lambda example: {'label': example[label_name]}, remove_columns=[label_name])


!pip install transformers  -q


import transformers
print(transformers.__version__)


from transformers import AutoTokenizer

model_checkpoint= "../input/bert-base-uncased"    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,use_fast=True)


task = "sst2"
batch_size = 64


def tokenizer_function(examples):
    return tokenizer(examples['question_text'],max_length =133,padding=True,truncation=True)


encoded_train = train_dataset.map(tokenizer_function, batched=True)
encoded_valid = valid_dataset.map(tokenizer_function, batched=True)


print(encoded_train[0])


gc.collect()


encoded_train.set_format('torch',columns=['input_ids','attention_mask','label'])
encoded_valid.set_format('torch',columns=['input_ids','attention_mask','label'])


gc.collect()


encoded_valid.set_format('torch',columns=['input_ids','attention_mask','label'])


from datasets import load_metric
metric = load_metric("accuracy")


metric


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

metric_name = "accuracy"
model_name = model_checkpoint.split("/")[-1]


args = TrainingArguments(
    output_dir ='/results',
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    do_predict=True
)


import sklearn
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support,accuracy_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)


import torch
# determine the device we will be using for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}".format(torch.cuda.get_device_name(0)))


torch.cuda.empty_cache()


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_train,
    eval_dataset =encoded_valid,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()


trainer.evaluate()


#del train,encoded_train,encoded_valid
gc.collect()


test.info()


test_dataset = Dataset.from_pandas(test)
encoded_test = test_dataset.map(tokenizer_function, batched=True)
encoded_test.set_format('torch',columns=['input_ids','attention_mask'])
encoded_test[0]


#encoded_test_input_ids = encoded_test['input_ids']
#encoded_test_attention_mask = encoded_test['attention_mask']


gc.collect()


output = trainer.predict(encoded_test)


preds = output[0]#predictions


labels = np.argmax(preds,axis=1)


labels

