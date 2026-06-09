import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import torch.optim as optim


import pandas as pd
import numpy as np
train = pd.read_csv('/kaggle/input/lmsys-kfolds/train_5folds.csv')
train.head()


train['text'] = 'User prompt: ' + train['prompt'] +  '\n\nModel A :\n' + train['response_a'] +'\n\n--------\n\nModel B:\n'  + train['response_b']
print(train['text'][4])


print(len(train))
train.head()


kfold = 0
test_texts = train[train['kfold']==kfold]['text'].values
train_texts = train[train['kfold']!=kfold]['text'].values

len(test_texts)+len(train_texts)


test_label = train[train['kfold']==kfold]['label'].values
train_label = train[train['kfold']!=kfold]['label'].values

len(test_label)+len(train_label)


batch_size = 8
num_classes = 3

# Tokenize
test_tokenized = [t.split() for t in test_texts]
train_tokenized = [t.split() for t in train_texts]
print(len(test_tokenized)+len(train_tokenized))

# Build vocabulary
vocab = {"<pad>": 0, "<unk>": 1}
for word in Counter(w for sent in test_tokenized for w in sent):
    vocab[word] = len(vocab)

for word in Counter(w for sent in train_tokenized for w in sent):
    vocab[word] = len(vocab)

def encode(sentence):
    return torch.tensor([vocab.get(w, 1) for w in sentence])



class TextDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return encode(self.tokens[idx]), torch.tensor(self.labels[idx])

def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True)
    return sequences_padded, torch.tensor(labels)

train_dataset = TextDataset(train_tokenized, train_label)
loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

test_dataset = TextDataset(test_tokenized, test_label)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)


import torch.nn as nn

class RNNClassifier(nn.Module):
    # def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, hidden_dim2=64, num_classes=num_classes):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128, hidden_dim2=64, num_classes=num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, h_n = self.rnn(x)
        output = self.fc(h_n[-1])   # use final hidden state
        logits = self.fc2(output)
        return logits

model = RNNClassifier(vocab_size=len(vocab))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    model.train()
    total_batches = 0
    train_loss = 0
    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        train_loss+=loss.item()
        total_batches+=1
        loss.backward()
        optimizer.step()
        
        

    print(f"Epoch {epoch+1}/{epochs}  Train Loss: {train_loss/total_batches:.4f}")
    
    model.eval()
    test_loss = 0.0
    total_batches = 0
    correct, total = 0, 0

    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
            total_batches += 1

            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    test_accuracy = correct / total

    print(
        f"Epoch {epoch+1:02d} | "
        f"Test Loss: {test_loss/total_batches:.4f} | "
        f"Test Acc: {test_accuracy:.4f}"
    )
    torch.save(model.state_dict(), f"rnn_classifier_{epoch}.pth")


torch.save(model.state_dict(), "rnn_classifier.pth")



model_loaded = RNNClassifier(vocab_size=len(vocab))  # same init as before
model_loaded.load_state_dict(torch.load("rnn_classifier.pth", map_location=device))
model_loaded.eval()   # set to inference mode



import torch.nn.functional as F
def predict(text):
    tokens = text.split()
    encoded = encode(tokens).unsqueeze(0)
    with torch.no_grad():
        logits = model_loaded(encoded)
        # pred = torch.argmax(logits, dim=1).item()
        probs = F.softmax(logits, dim=1)
        print(probs)
    # return "positive" if pred == 1 else "negative"

print(predict("this movie was amazing"))
print(predict("this film was awful"))

