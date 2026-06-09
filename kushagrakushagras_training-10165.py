# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as param # data processing, CSV file I/O (e.g. pd.read_csv)
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


kfolds = 5
epochs = 1
batch_size = 16
num_classes = 3


beta_train = param.read_csv('/kaggle/input/5folds-train/5folds_trains.csv')


beta_train['text'] = 'User prompt: ' + beta_train['prompt'] +  '\n\nModel A :\n' + beta_train['response_a'] +'\n\n--------\n\nModel B:\n'  + beta_train['response_b']
print(beta_train['text'][4])


class TextDataset(Dataset):
    def __init__(self, text_list, label_list):
        self.text_list = text_list
        self.label_list = label_list

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, index):
        encoded_text = encode(self.text_list[index])
        label_tensor = torch.tensor(self.label_list[index])
        return encoded_text, label_tensor


def collate_fn(batch_items):
    encoded_batch, label_batch = zip(*batch_items)

    padded_batch = pad_sequence(encoded_batch, batch_first=True)
    label_tensor = torch.tensor(label_batch)

    return padded_batch, label_tensor


class GRUClassifier(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_dim=256,
        hidden_dim=128,
        projection_dim=64,
        num_classes=num_classes
    ):
        super().__init__()

        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_dim)
        self.gru_layer = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        self.projection_layer = nn.Linear(hidden_dim, projection_dim)
        self.output_layer = nn.Linear(projection_dim, num_classes)

    def forward(self, token_ids):
        embedded_tokens = self.embedding_layer(token_ids)

        gru_outputs, hidden_states = self.gru_layer(embedded_tokens)

        last_hidden_state = hidden_states[-1]

        projected = self.projection_layer(last_hidden_state)
        logits = self.output_layer(projected)

        return logits


for fold_index in range(kfolds):
    print(f"Training Fold: {fold_index}")

    test_texts = beta_train[beta_train["kfold"] == fold_index]["text"].values
    train_texts = beta_train[beta_train["kfold"] != fold_index]["text"].values

    test_labels = beta_train[beta_train["kfold"] == fold_index]["label"].values
    train_labels = beta_train[beta_train["kfold"] != fold_index]["label"].values

    test_tokens = [sentence.split() for sentence in test_texts]
    train_tokens = [sentence.split() for sentence in train_texts]

    print(len(test_tokens) + len(train_tokens))

    vocabulary = {"<pad>": 0, "<unk>": 1}

    for word in Counter(w for tokens in test_tokens for w in tokens):
        vocabulary[word] = len(vocabulary)

    for word in Counter(w for tokens in train_tokens for w in tokens):
        vocabulary[word] = len(vocabulary)

    def encode(sentence_tokens):
        return torch.tensor([vocabulary.get(token, 1) for token in sentence_tokens])


    training_dataset = TextDataset(train_tokens, train_labels)
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )

    test_dataset = TextDataset(test_tokens, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    model = GRUClassifier(vocabulary_size=len(vocabulary))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_inputs, batch_labels in tqdm(training_loader):
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            predictions = model(batch_inputs)

            loss = loss_fn(predictions, batch_labels)
            epoch_loss += loss.item()
            batch_count += 1

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss / batch_count:.4f}")

        model.eval()
        eval_loss = 0.0
        eval_batches = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch_inputs, batch_labels in tqdm(test_loader):
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_inputs)
                loss = loss_fn(outputs, batch_labels)

                eval_loss += loss.item()
                eval_batches += 1

                _, predicted_labels = torch.max(outputs, 1)
                correct_predictions += (predicted_labels == batch_labels).sum().item()
                total_predictions += batch_labels.size(0)

        accuracy = correct_predictions / total_predictions

        print(
            f"Epoch {epoch+1:02d} | "
            f"Test Loss: {eval_loss / eval_batches:.4f} | "
            f"Test Acc: {accuracy:.4f}"
        )

        torch.save(model.state_dict(), f"gru_classifier_fold_{fold_index}_epoch_{epoch}.pth")








