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


!pip install transformers datasets scikit-learn pandas torch


import os
os.environ["WANDB_DISABLED"] = "true"
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import zipfile


# Input path 
input_path = "/kaggle/input/word2vec-nlp-tutorial/"

# Unzip the compressed files
with zipfile.ZipFile(f"{input_path}labeledTrainData.tsv.zip", "r") as zip_ref:
    zip_ref.extractall(".")  # Extract to the current directory

with zipfile.ZipFile(f"{input_path}testData.tsv.zip", "r") as zip_ref:
    zip_ref.extractall(".")  # Extract to the current directory


# Load the dataset
train_data = pd.read_csv("labeledTrainData.tsv", sep='\t')
test_data = pd.read_csv("testData.tsv", sep='\t')

# Preprocessing
X = train_data['review']
y = train_data['sentiment']

# Use a subset for faster experimentation
X = X[:30000]
y = y[:30000]

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(data):
    return tokenizer(data["review"], padding="max_length", truncation=True, max_length=512)


# Use Transformers (Hugging Face) format
train_dataset = Dataset.from_dict({"review": X_train.tolist(), "label": y_train.tolist()})
val_dataset = Dataset.from_dict({"review": X_val.tolist(), "label": y_val.tolist()})

# Tokenize the datasets
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# Load the model (BERT  for sequence classification)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    hidden_dropout_prob=0.3,  # Increased dropout
)


# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5, 
    warmup_steps=500,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    fp16=True,
)

# Calculate metrics
def calculate_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": accuracy_score(labels, preds)}
# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=calculate_metrics,
    tokenizer=tokenizer,
)


# Training the model
print("Training Started:")
trainer.train()


# Evaluating the model on a validation dataset
results = trainer.evaluate()
print(f"Validation Accuracy: {results['eval_accuracy']*100:.2f}%")


# Tokenizing the test dataset
test_dataset = Dataset.from_dict({"review": test_data['review'].tolist()})
test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])


# Generating predictions for test dataset
print("Generating predictions for test dataset:")
predictions = trainer.predict(test_dataset)
labels = predictions.predictions.argmax(-1)

# Creating the submission file
submission_df = pd.DataFrame({"id": test_data["id"], "sentiment": labels})
submission_df.to_csv("text_classifiction.csv", index=False)
print("Submission file saved as 'text_classifiction.csv'")




