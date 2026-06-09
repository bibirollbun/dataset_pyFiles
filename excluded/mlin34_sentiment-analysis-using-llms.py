# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import os, re, string
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import torch

if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")


sample_submission = pd.read_csv("/kaggle/input/natural-language-processing-with-disaster-tweets/sample_submission.csv", encoding='latin-1')
X_train = pd.read_csv("/kaggle/input/natural-language-processing-with-disaster-tweets/train.csv", encoding='latin-1')
X_test = pd.read_csv("/kaggle/input/natural-language-processing-with-disaster-tweets/test.csv")
X_test = X_test.dropna(how="any", axis=1)
X_train = X_train.dropna(how="any", axis=1)
X_train['text_len'] = X_train['text'].apply(lambda x: len(x.split(' ')))

X_train.head()


X_train.describe(include='all')


X_train.info()


X_test.info()



def preprocess_data(text):
    # Clean puntuation, urls, and so on
    text = clean_text(text)
    # Remove stopwords and Stemm all the words in the sentence
    text = ' '.join(stemmer.stem(word) for word in text.split(' ') if word not in stop_words)

    return text
def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_emoji(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_html(text):
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = remove_url(text)
    text = remove_emoji(text)
    text = remove_html(text)
    return text

def preprocess_data(text):
    # Clean puntuation, urls, and so on
    text = clean_text(text)
    return text

X_train["text_clean"] = X_train["text"].apply(clean_text)
X_train


X_test["text_clean"] = X_test["text"].apply(clean_text)
X_test.head()


X_train.rename(columns={'target':'label'}, inplace=True)
train_df = X_train[['text_clean','label']]


from datasets import Dataset

# Convert pandas DataFrame with labels into a Dataset
dataset = Dataset.from_pandas(train_df[["text_clean", "label"]])

# Split into train and test (or validation)
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)



split_dataset


import random
import torch
import pandas as pd
import peft
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
!pip install evaluate
import evaluate

checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(row):
    return tokenizer(row["text_clean"], truncation=True, padding=True)


tokenized_dataset1 = split_dataset.map(tokenize_function, batched=True)
tokenized_dataset2 = dataset.map(tokenize_function, batched=True)


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


dataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels)["f1"],
    }

training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    report_to="none"
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset1['train'],
    eval_dataset=tokenized_dataset1['test'],
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)



trainer.train()


test_df = X_test[['text_clean']]
test_data = Dataset.from_pandas(X_test[["text_clean"]])
tokenized_dataset2 = test_data.map(tokenize_function, batched=True)
tokenized_dataset2


predictions = trainer.predict(tokenized_dataset2)
logits = predictions.predictions
y_pred = logits.argmax(axis=-1)




submission_df = pd.DataFrame({'id': X_test.id, 'target': y_pred})
submission_df


submission_df.to_csv('distilbert-base-uncased', index=False)





