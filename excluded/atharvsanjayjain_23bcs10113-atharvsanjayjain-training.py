import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import torch.optim as optim

import pandas as pd
import numpy as np
import torch.nn as nn

# --- Data Loading and Preprocessing (Unchanged) ---
# Assuming 'train_5folds.csv' is available at the specified path or locally.
try:
    train = pd.read_csv('/kaggle/input/creating-folds-lmsys-atharv23bcs10113/train_5folds.csv')
except FileNotFoundError:
    print("Warning: Could not find the specified CSV file. Please ensure it's available.")
    # Create dummy data for demonstration if not found, you should use your actual data.
    data = {'prompt': ['p1', 'p2', 'p3', 'p4', 'p5'],
            'response_a': ['ra1', 'ra2', 'ra3', 'ra4', 'ra5'],
            'response_b': ['rb1', 'rb2', 'rb3', 'rb4', 'rb5'],
            'label': [0, 1, 2, 1, 0],
            'kfold': [0, 1, 2, 3, 4]}
    train = pd.DataFrame(data)

train['text'] = 'User prompt: ' + train['prompt'] + ' \n\nModel A :\n' + train['response_a'] +'\n\n--------\n\nModel B:\n' + train['response_b']
print(train['text'][0]) # Changed index to 0 for consistency if using dummy data
print(len(train))


kfolds = 5
epochs = 1 #5
batch_size = 16
num_classes = 3

class TextDataset(Dataset):
    def __init__(self, tokens, labels, vocab, encode_func):
        self.tokens = tokens
        self.labels = labels
        self.vocab = vocab
        self.encode = encode_func

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        # Use the passed encode function
        return self.encode(self.tokens[idx]), torch.tensor(self.labels[idx])

def collate_fn(batch):
    sequences, labels = zip(*batch)
    # Ensure sequences are tensors of indices before padding
    sequences = [s.clone().detach() for s in sequences] 
    sequences_padded = pad_sequence(sequences, batch_first=True)
    return sequences_padded, torch.tensor(labels)

# 
# --- GRU Classifier Model (Key Change) ---
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128, hidden_dim2=64, num_classes=num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # CHANGE 1: Replaced nn.LSTM with nn.GRU
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        # CHANGE 2: GRU returns (output, h_n) - no cell state (c_n)
        output, h_n = self.gru(x)
        
        # Use the final hidden state h_n[-1] (hidden state of the last layer)
        # h_n shape is (num_layers * num_directions, batch_size, hidden_size)
        # output = self.fc(h_n[-1]) # Use this if GRU is multi-layer (which it is not by default)
        output = self.fc(h_n.squeeze(0)) # Correct for single-layer GRU with batch_first=True
        
        # OR: To stick closer to the original logic which handles multi-layer (num_layers > 1)
        # output = self.fc(h_n[-1]) 
        
        # We will use h_n[-1] to be compatible with potential future multi-layer changes
        output = self.fc(h_n[-1])

        logits = self.fc2(output)
        return logits


# --- Training Loop (Model name changed) ---
for kfold in range(kfolds):
    if kfold >= len(train['kfold'].unique()):
        print(f"Stopping fold loop as kfold={kfold} exceeds available folds in data.")
        break
        
    print(f"\n--- Train Fold: {kfold} ---")
    test_texts = train[train['kfold']==kfold]['text'].values
    train_texts = train[train['kfold']!=kfold]['text'].values
    test_label = train[train['kfold']==kfold]['label'].values
    train_label = train[train['kfold']!=kfold]['label'].values
    
    
    # Tokenize
    test_tokenized = [str(t).split() for t in test_texts]
    train_tokenized = [str(t).split() for t in train_texts]
    print(f"Total samples: {len(test_tokenized) + len(train_tokenized)}")
    
    # Build vocabulary
    vocab = {"<pad>": 0, "<unk>": 1}
    # Iterate through all tokens for vocabulary building
    all_tokens = [w for sent in test_tokenized for w in sent] + [w for sent in train_tokenized for w in sent]
    for word, count in Counter(all_tokens).items():
        if word not in vocab: # Ensure we don't overwrite <pad> and <unk>
            vocab[word] = len(vocab)
    
    def encode(sentence):
        return torch.tensor([vocab.get(w, 1) for w in sentence], dtype=torch.long)
    
    
    
    train_dataset = TextDataset(train_tokenized, train_label, vocab, encode)
    loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    
    test_dataset = TextDataset(test_tokenized, test_label, vocab, encode)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # CHANGE 3: Instantiate GRUClassifier
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
        for x, y in tqdm(loader, desc=f"Fold {kfold} Epoch {epoch+1} Train"):
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
            for X_batch, y_batch in tqdm(test_loader, desc=f"Fold {kfold} Epoch {epoch+1} Test"):
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
        # CHANGE 4: Update saved model name
        torch.save(model.state_dict(), f"gru_classifier_kfold_{kfold}_epoch_{epoch}.pth")


# New: GRUClassifier
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128, hidden_dim2=64, num_classes=num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True) # <- nn.GRU
        self.fc = nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, h_n = self.gru(x) # <- Returns (output, h_n)
        output = self.fc(h_n[-1])   # use final hidden state
        logits = self.fc2(output)
        return logits

