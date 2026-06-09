# !pip install --upgrade pip


# !pip install transformers datasets torch


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


print("Version 32")


train = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
test = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")
train.shape, test.shape


import re
import string
import emoji
import nltk
from nltk.corpus import stopwords

def cleaner(df):
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        
        # remove emojis
        text = emoji.replace_emoji(text, replace=' ')
        
        # keep only alphabets, spaces, '.' and ','
        text = re.sub(r"[^a-zA-Z\s\.,]", " ", text)
        
        # lowercase
        text = text.lower()
        
        # remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    df['body'] = df['body'].apply(clean_text)
    df['positive_example_1'] = df['positive_example_1'].apply(clean_text)
    df['positive_example_2'] = df['positive_example_2'].apply(clean_text)
    df['negative_example_1'] = df['negative_example_1'].apply(clean_text)
    df['negative_example_2'] = df['negative_example_2'].apply(clean_text)
    return df

train = cleaner(train)
test = cleaner(test)

train.head()


from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model (try different models)
model_name = "answerdotai/ModernBERT-base"  # Better than BERT for classification
tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = AutoModelForSequenceClassification.from_pretrained(
#     model_name, 
#     num_labels=2
# )


from datasets import Dataset
# SIMPLE format - no roleplay, use [SEP] token
train['text'] = train['rule'] + " [SEP] " + train["subreddit"] + " [SEP] " + train['body']
train['labels'] = train['rule_violation']

# Don't augment with examples - use only original data
X_train = train[['text', 'labels']]

# Create dataset
dataset = Dataset.from_pandas(X_train)


validation1 = train.copy()
validation1['text'] = validation1['rule'] + " [SEP] " + validation1["subreddit"] + " [SEP] " + validation1['positive_example_1']
validation1['labels'] = 1
validation2 = train.copy()
validation2['text'] = validation2['rule'] + " [SEP] " + validation2["subreddit"] + " [SEP] " + validation2['positive_example_2']
validation2['labels'] = 1
validation3 = train.copy()
validation3['text'] = validation3['rule'] + " [SEP] " + validation3["subreddit"] + " [SEP] " + validation3['negative_example_1']
validation3['labels'] = 0
validation4 = train.copy()
validation4['text'] = validation4['rule'] + " [SEP] " + validation4["subreddit"] + " [SEP] " + validation4['negative_example_2']
validation4['labels'] = 0

validation = pd.concat([validation1,validation2,validation3,validation4])
# validation = pd.concat([validation1,validation3])
# Don't augment with examples - use only original data
X_val = validation[['text', 'labels']]

# Create dataset
validation = Dataset.from_pandas(X_val)


dataset, validation


lengths = []
for text in validation['text']:  # replace with your dataset
    tokens = tokenizer(text, add_special_tokens=True)  # don't truncate yet
    lengths.append(len(tokens['input_ids']))

print("Max:", max(lengths))
print("Mean:", np.mean(lengths))
print("70th percentile:", np.percentile(lengths, 70))
print("90th percentile:", np.percentile(lengths, 90))


# # Tokenize with longer max_length
# def preprocess(examples):
#     return tokenizer(examples["text"], truncation=True, max_length=250)

# tokenized = dataset.map(preprocess, batched=True)


from transformers import TrainingArguments, Trainer

# training_args = TrainingArguments(
#     output_dir="./results",
#     learning_rate=2e-5,          # Lower learning rate
#     per_device_train_batch_size=8,
#     num_train_epochs=3,           # Fewer epochs
#     weight_decay=0.01,            # Regularization
#     warmup_steps=100,
#     logging_steps=50,
#     eval_strategy="no",           # No validation split confusion
#     save_strategy="epoch",
#     load_best_model_at_end=False,
#     report_to="none"
# )

# # Unfreeze last 3 encoder layers + pooler + classifier
# for name, param in model.base_model.named_parameters():
#     # if any(layer in name for layer in ["encoder.layer.9", "encoder.layer.10", "encoder.layer.11", "pooler"]):
#     #     param.requires_grad = True
#     # else:
#     #     param.requires_grad = False
#     param.requires_grad = False

# # unfreeze base model pooling layers
# for name, param in model.base_model.named_parameters():
#     if "pooler" in name:
#         param.requires_grad = True

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized,
#     tokenizer=tokenizer
# )

# trainer.train()


tokenizer.all_special_tokens


# Format test data EXACTLY like training
test['text'] = test['rule'] + " [SEP] " + test["subreddit"] + " [SEP] " + test['body']
test_dataset = Dataset.from_pandas(test[['text']])
# tokenized_test = test_dataset.map(preprocess, batched=True)

# predictions = trainer.predict(tokenized_test)
# probs = np.exp(predictions.predictions) / np.exp(predictions.predictions).sum(-1, keepdims=True)

# submission = pd.DataFrame({
#     'row_id': test['row_id'],
#     'rule_violation': probs[:, 1]
# })


# submission.to_csv('submission.csv', index=False)
# print(submission.head())


import os, gc
from transformers import TrainingArguments, Trainer
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Train 3 different models
models_to_try = [
    # "/kaggle/input/bert-base-cased-local/ensembeled/microsoft/deberta-v3-base",
    # "/kaggle/input/bert-base-cased-local/ensembeled/kaggle/input/bert-base-cased-local/bert",
    # # "/kaggle/input/bert-base-cased-local/ensembeled/microsoft/deberta-v3-large"
    "answerdotai/ModernBERT-base"
]
save_path = 'ensembeled'

predictions_list = []
try:
    torch.cuda.empty_cache()
    del model
    del tokenizer
    gc.collect()
    print("Tried")
except:
    pass
for model_name in models_to_try:
    # Train each model (use your existing training code)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.save_pretrained(save_path+"/"+model_name)
    tokenizer.save_pretrained(save_path+"/"+model_name)

    def preprocess(examples):
        # if model_name == "/kaggle/input/bert-base-cased-local/ensembeled/kaggle/input/bert-base-cased-local/bert":
        #     return tokenizer(examples["text"], truncation=True, max_length=512)
        return tokenizer(examples["text"], truncation=True, max_length=250)
    
    tokenized = dataset.map(preprocess, batched=True)
    tokenized_val = validation.map(preprocess, batched=True)
    tokenized_test = test_dataset.map(preprocess, batched=True)
    
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=1e-6,          # Lower learning rate
        per_device_train_batch_size=8,
        num_train_epochs=3,           # Fewer epochs
        weight_decay=0.01,            # Regularization
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="epoch",           # No validation split confusion
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="none"
    )
    
    # # Unfreeze last 3 encoder layers + pooler + classifier
    # for name, param in model.base_model.named_parameters():
    #     if any(layer in name for layer in ["encoder.layer.10", "encoder.layer.11", "pooler"]):
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    #     # param.requires_grad = False
    
    # # unfreeze base model pooling layers
    # for name, param in model.base_model.named_parameters():
    #     if "pooler" in name:
    #         param.requires_grad = True
    
    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_val,
        eval_dataset=tokenized,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    # Get predictions
    preds = trainer.predict(tokenized_test)
    probs = np.exp(preds.predictions) / np.exp(preds.predictions).sum(-1, keepdims=True)
    predictions_list.append(probs[:, 1])
    torch.cuda.empty_cache()
    del model
    del tokenizer
    gc.collect()

# Average predictions
ensemble_pred = np.mean(predictions_list, axis=0)

submission = pd.DataFrame({
    'row_id': test['row_id'],
    'rule_violation': ensemble_pred
})
submission.head()


submission.to_csv("submission.csv", index=False)




