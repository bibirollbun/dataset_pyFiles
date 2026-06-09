! pip install kaggle
! pip install datasets
! pip install pandas


from transformers import TrainingArguments,Trainer


import os
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
creds = '{"username":"ganihunglin","key":"cbbd7e7eac5701f09ced7c060c23fdaa"}'


from pathlib import Path

cred_path = Path('~/.kaggle/kaggle.json').expanduser()
if not cred_path.exists():
    cred_path.parent.mkdir(exist_ok=True)
    cred_path.write_text(creds)
    cred_path.chmod(0o600)


path = Path('us-patent-phrase-to-phrase-matching')


if iskaggle:
    path = Path('../input/us-patent-phrase-to-phrase-matching')
    ! pip install --no-index --find-links ../input/huggingface-datasets datasets -q


path


import pandas as pd
dataFrame=pd.read_csv(path/'train.csv')
dataFrame


dataFrame['input']='In context of '+dataFrame.context+', Phrase1 is '+dataFrame.anchor+', Phrase2 is '+dataFrame.target


from datasets import Dataset,DatasetDict


dataSet=Dataset.from_pandas(dataFrame)
modelName='../input/debertav3small'


from transformers import  AutoModelForSequenceClassification,AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(modelName)


tokenizer.tokenize("Test content:Nice to meet you me too")


tokenizedDataSet=dataSet.map(lambda rowHead:tokenizer(rowHead['input']),batched=True)


tokenizedDataSet=tokenizedDataSet.rename_columns({'score':'labels'})


tokenizedDataSet


evaluationDataFrame=pd.read_csv(path/'test.csv')
evaluationDataFrame['input']='In context of '+evaluationDataFrame.context+', Phrase1 is '+evaluationDataFrame.anchor+', Phrase2 is '+evaluationDataFrame.target


evaluationDataSet = Dataset.from_pandas(evaluationDataFrame).map(lambda rowHead:tokenizer(rowHead['input']), batched=True)


dictDataSet = tokenizedDataSet.train_test_split(0.20, seed=42)


import numpy as np
def corr(x,y): return np.corrcoef(x,y)[0][1]
def corr_d(eval_pred): return {'pearson': corr(*eval_pred)}





from transformers import TrainingArguments,Trainer


batchSize=128
epochs=4
learningRate=9e-5


args=TrainingArguments(
    'outputs',
    learning_rate=learningRate,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    fp16=True,
    evaluation_strategy='epoch',
    per_device_train_batch_size=batchSize,
    per_device_eval_batch_size=batchSize*2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to='none'
    )





model = AutoModelForSequenceClassification.from_pretrained(modelName,num_labels=1,ignore_mismatched_sizes=True)
trainer = Trainer(model,args,train_dataset=dictDataSet['train'],eval_dataset=dictDataSet['test'],
          tokenizer=tokenizer,compute_metrics=corr_d)




trainer.train()


preds=trainer.predict(evaluationDataSet).predictions.astype(float)
preds = np.clip(preds, 0, 1)


import datasets
submission = datasets.Dataset.from_dict({
  'id': evaluationDataSet['id'],
  'score':  preds.flatten()
})
submission.to_csv('submission.csv',index=False)




