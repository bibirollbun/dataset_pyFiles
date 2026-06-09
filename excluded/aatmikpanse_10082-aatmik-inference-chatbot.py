from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

import torch


# hyper params
kfolds = 5
epochs = 1 #5
batch_size = 16
num_classes = 3


train = train = pd.read_csv('/kaggle/input/eda-n-folds-lmsys/train_5folds.csv')
train.head()


train['text'] = 'User prompt: ' + train['prompt'] +  '\n\nModel A :\n' + train['response_a'] +'\n\n--------\n\nModel B:\n'  + train['response_b']
print(train['text'][4])


# Defining Dataset loader

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


# Using pytorch's GRU implementation

class GRUClassifier(nn.Module):
    # def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, hidden_dim2=64, num_classes=num_classes):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128, hidden_dim2=64, num_classes=num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) 
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, h_n = self.gru(x)      # GRU returns only h_n
        h_last = h_n[-1]               # final layer's last hidden state
        x = self.fc(h_last)
        logits = self.fc2(x)
        return logits



#training k_folds
for kfold in range(kfolds):
    print(f"Train Fold: {kfold}")
    test_texts = train[train['kfold']==kfold]['text'].values
    train_texts = train[train['kfold']!=kfold]['text'].values
    test_label = train[train['kfold']==kfold]['label'].values
    train_label = train[train['kfold']!=kfold]['label'].values
    
    
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
    
    
    
    train_dataset = TextDataset(train_tokenized, train_label)
    loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    
    test_dataset = TextDataset(test_tokenized, test_label)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    model = GRUClassifier(vocab_size=len(vocab))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)
    model = model.to(device)
    
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    
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
        torch.save(model.state_dict(), f"lstm_classifier_kfold_{kfold}_epoch_{epoch}.pth")




