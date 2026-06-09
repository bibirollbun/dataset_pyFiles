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


import torch
import random
import optuna
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# !pip install optuna


df = pd.read_csv('/kaggle/input/aiquest-bangla-sentiment-analysis-competition/train.csv')
df.head(10)


df.columns


df.drop('id', axis=1, inplace=True)
df



df['sentiment'].value_counts() # Checking if the dataset is balanced or not


df.isnull().sum() # checking if there is any null or empty values  or not


df.shape


df.duplicated().sum() # checking if there is any duplicat values present or not


new_df = df.drop_duplicates()
print(new_df.shape)


# Converting text sentimental level into Numeric label
sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
df["sentiment"] = df["sentiment"].map(sentiment_map)
new_df["sentiment"] = new_df["sentiment"].map(sentiment_map)
new_df


# split the dataset into train and test with balance
# test size 20% and 80% in train
train_texts, val_texts, train_labels, val_labels = train_test_split(
    new_df["text"], new_df["sentiment"], test_size=0.2, random_state=42, stratify=new_df["sentiment"]
)



train_texts


MODEL_NAME = "sagorsarker/bangla-bert-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)



def preprocess_texts(texts):
    return tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt").to(device)

train_encodings = preprocess_texts(train_texts)
val_encodings = preprocess_texts(val_texts)

train_labels = torch.tensor(train_labels.values)
val_labels = torch.tensor(val_labels.values)



train_encodings


class BanglaDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]
train_dataset = BanglaDataset(train_encodings, train_labels)
val_dataset = BanglaDataset(val_encodings, val_labels)


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 5e-5,log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)

    best_val_loss = float("inf")
    patience = 3
    no_improve_epochs = 0
    train_losses = []
    val_losses = []

    for epoch in range(10):  # Train for 10 epochs
        model.train()
        total_loss = 0

        for batch in train_loader:
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs = {key: val.to(device) for key, val in inputs.items()}
                labels = labels.to(device)

                outputs = model(**inputs)
                loss = loss_fn(outputs.logits, labels)
                total_val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping triggered!")
                break

    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.show()

    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("Best Hyperparameters:", study.best_params)


best_lr = study.best_params["lr"]
best_batch_size = study.best_params["batch_size"]
best_weight_decay = study.best_params["weight_decay"]

optimizer = torch.optim.AdamW(model.parameters(), lr=best_lr, weight_decay=best_weight_decay)
loss_fn = torch.nn.CrossEntropyLoss()

train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best_batch_size, shuffle=False)

val_preds = []
val_true = []
model.eval()
with torch.no_grad():
    for batch in val_loader:
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)

        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)

        val_preds.extend(preds.cpu().numpy())
        val_true.extend(labels.cpu().numpy())

macro_f1 = f1_score(val_true, val_preds, average='macro')
print(f"Final Macro F1-score: {macro_f1:.4f}")



df_encodings = preprocess_texts(df['text'])
df_labels = torch.tensor(df['sentiment'].values)

df_dataset = BanglaDataset(df_encodings, df_labels)
df_loader = DataLoader(df_dataset, batch_size=best_batch_size, shuffle=True)

df_preds = []
df_true = []

model.eval()
with torch.no_grad():
    for batch in df_loader:
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)

        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)

        df_preds.extend(preds.cpu().numpy())
        df_true.extend(labels.cpu().numpy())

sentiment_map_reverse = {0: "negative", 1: "neutral", 2: "positive"}
df["sentiment"] = [sentiment_map_reverse[pred] for pred in df_preds]
df['id'] = df.index

df[["id", "sentiment"]].to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")


macro_f1 = f1_score(df_true, df_preds, average='macro')
print("Macro F1 score:", macro_f1)

