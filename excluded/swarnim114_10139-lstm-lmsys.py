import pandas as pd
import numpy as np
import re
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm



df = pd.read_csv("/kaggle/input/10139-creating-folds-lmsys/train_5folds.csv")

train_df = df[df["kfold"] != 0].reset_index(drop=True)
val_df   = df[df["kfold"] == 0].reset_index(drop=True)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))




def simple_tokenizer(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip().split()

train_tokens = train_df["prompt"].apply(simple_tokenizer).tolist()
val_tokens   = val_df["prompt"].apply(simple_tokenizer).tolist()



word_freq = defaultdict(int)
for toks in train_tokens:
    for w in toks:
        word_freq[w] += 1

vocab = {"<pad>": 0, "<unk>": 1}
for w, c in word_freq.items():
    if c >= 2:
        vocab[w] = len(vocab)

print("Vocab size:", len(vocab))



def encode(tokens):
    return torch.tensor([vocab.get(w, 1) for w in tokens], dtype=torch.long)



class LMSYSDataset(Dataset):
    def __init__(self, token_list, labels):
        self.token_list = token_list
        self.labels = labels

    def __len__(self):
        return len(self.token_list)

    def __getitem__(self, idx):
        return encode(self.token_list[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    return padded, labels



print(df.columns)



train_ds = LMSYSDataset(train_tokens, train_df["label"].values)
val_ds   = LMSYSDataset(val_tokens, val_df["label"].values)


train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)



class MyBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, (h, c) = self.lstm(x)
        h_final = torch.cat((h[-2], h[-1]), dim=1)
        h_final = self.dropout(h_final)
        logits = self.fc(h_final)
        return logits



device = "cuda" if torch.cuda.is_available() else "cpu"

model = MyBiLSTM(vocab_size=len(vocab))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



epochs = 3

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for x_batch, y_batch in tqdm(train_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    val_loss = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            val_loss += loss.item()

            _, pred_labels = torch.max(preds, 1)
            correct += (pred_labels == y_batch).sum().item()
            total += y_batch.size(0)

    print(f"Epoch {epoch+1}/{epochs}")
    print("Train Loss:", running_loss / len(train_loader))
    print("Val Loss:", val_loss / len(val_loader))
    print("Val Accuracy:", correct / total)



# Save trained model weights
model_path = "/kaggle/working/my_lstm_model.pth"
torch.save(model.state_dict(), model_path)

print("Model saved at:", model_path)



model.eval()

correct, total = 0, 0
val_loss = 0

with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        preds = model(x_batch)
        loss = criterion(preds, y_batch)

        val_loss += loss.item()

        _, predicted = torch.max(preds, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

final_accuracy = correct / total
final_loss = val_loss / len(val_loader)

print("Final Validation Loss:", final_loss)
print("Final Validation Accuracy:", final_accuracy)



train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(epochs):
    ...
    train_losses.append(running_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(correct / total)



import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(val_accuracies, label="Val Accuracy", color='green')
plt.title("Validation Accuracy")
plt.legend()

plt.show()



def predict_text(text):
    model.eval()

    tokens = simple_tokenizer(text)
    encoded = encode(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(encoded)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    labels = ["Model A", "Model B", "Tie"]
    prediction = labels[np.argmax(probs)]

    print("Probabilities:", probs)
    print("Predicted Winner:", prediction)

    return prediction

predict_text("Explain machine learning in simple terms.")



all_preds = []

model.eval()
with torch.no_grad():
    for x_batch, _ in val_loader:
        x_batch = x_batch.to(device)
        logits = model(x_batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_preds.extend(probs)

all_preds = np.array(all_preds)
pd.DataFrame(all_preds, columns=["prob_model_a","prob_model_b","prob_tie"]).head()



print("Notebook execution completed successfully!")





