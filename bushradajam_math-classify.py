#!pip install evaluate


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np


test_data= pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/test.csv")
test_data


sample= pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/sample_submission.csv")
sample


train_data= pd.read_csv("/kaggle/input/classification-of-math-problems-by-kasut-academy/train.csv")
train_data


train_data.info()


train_data.shape


train_data.isnull().sum()


train_data['label'].value_counts()


train_data['Question'][0]


train_df, val_df = train_test_split(train_data, test_size=0.2, stratify=train_data['label'], random_state=42)


import pandas as pd
import re
from transformers import (
    BertTokenizer,
    EncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from nltk.corpus import stopwords
from datasets import Dataset


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(batch):
    return tokenizer(batch['Question'], padding='max_length', truncation=True, max_length=128)


train_ds = Dataset.from_pandas(train_data[['Question', 'label']])
val_ds = Dataset.from_pandas(val_df[['Question', 'label']])


train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8)


training_args = TrainingArguments(
    output_dir="./results",
    #evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()


test_data['cleaned'] = test_df['Question'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)


test_dataset = Dataset.from_pandas(test_df[['cleaned']].rename(columns={"cleaned": "Question"}))
test_dataset = test_dataset.map(tokenize, batched=True)


# Predict
preds = trainer.predict(test_dataset)
labels = np.argmax(preds.predictions, axis=1)


submission = pd.DataFrame({
    'id': test_df['id'],
    'label': labels
})


submission.to_csv("submission.csv", index=False)
print("submission.csv created.")

