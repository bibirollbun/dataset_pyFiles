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


import re
import torch
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt


!unzip "/kaggle/input/quora-question-pairs/train.csv.zip" "train.csv"


!ls


df = pd.read_csv("train.csv")

print(f"Dataset shape: {df.shape}")
df.head()


# Show some duplicate question pairs
print("Examples of DUPLICATE questions:")
print("-" * 80)
for _, row in df[df['is_duplicate'] == 1].head(3).iterrows():
    print(f"Q1: {row['question1']}")
    print(f"Q2: {row['question2']}")
    print("-" * 80)

# Show some non-duplicate question pairs
print("\nExamples of NON-DUPLICATE questions:")
print("-" * 80)
for _, row in df[df['is_duplicate'] == 0].head(3).iterrows():
    print(f"Q1: {row['question1']}")
    print(f"Q2: {row['question2']}")
    print("-" * 80)

# Check class distribution
print(f"\nClass distribution:")
print(df['is_duplicate'].value_counts())

# For faster experimentation, we'll use a subset of the data
sample_size = 10000
df_sample = df.sample(sample_size, random_state=42)
print(f"\nUsing a sample of {sample_size} question pairs")


class ManualBPE:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}  # Dict of merge rules with priority
        self.vocab = set()  # Set of tokens
        
    def _get_stats(self, words):
        """Count frequency of adjacent symbol pairs"""
        pairs = defaultdict(int)
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    def _merge_vocab(self, pair, words):
        """Merge all occurrences of a pair in the vocabulary"""
        new_words = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in words.items():
            parts = word.split()
            new_word = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == pair[0] and parts[i+1] == pair[1]:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(parts[i])
                    i += 1
            new_word = ' '.join(new_word)
            new_words[new_word] = freq
        return new_words
    
    def train(self, corpus, verbose=False):
        # Pre-tokenize into words and add end token
        word_counts = Counter([word.lower() + '</w>' for sentence in corpus for word in sentence.split()])
        
        # Initialize by splitting words into characters
        words = {}
        for word, count in word_counts.items():
            words[' '.join(list(word))] = count
        
        # Build base vocabulary of individual characters
        self.vocab = set()
        for word in words:
            for char in word.split():
                self.vocab.add(char)
        
        # Determine number of merges needed
        num_merges = self.vocab_size - len(self.vocab)
        
        for i in range(num_merges):
            pairs = self._get_stats(words)
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Store this merge in our rules
            self.merges[best_pair] = i
            
            # Apply the merge
            words = self._merge_vocab(best_pair, words)
            
            # Add merged token to vocabulary
            self.vocab.add(''.join(best_pair))
            
            if verbose and (i+1) % 100 == 0:
                print(f"Merge {i+1}/{num_merges}: {best_pair} → {''.join(best_pair)}")
        
        if verbose:
            print(f"Final vocab size: {len(self.vocab)}")
                
    def encode(self, text):
        """Tokenize text using learned merges"""
        # Add end token and split into character tokens
        words = []
        for word in text.lower().split():
            word = word + '</w>'
            chars = ' '.join(list(word))
            words.append(chars)
        
        result = []
        for word in words:
            word_tokens = word.split()
            # Apply merges iteratively based on priority
            for pair, _ in sorted(self.merges.items(), key=lambda x: x[1]):
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == pair[0] and word_tokens[i+1] == pair[1]:
                        word_tokens[i:i+2] = [''.join(pair)]
                    else:
                        i += 1
            result.extend(word_tokens)
            
        return result


# Combine all questions into a corpus
corpus = df['question1'].tolist() + df['question2'].tolist()

# Initialize and train BPE
bpe = ManualBPE(vocab_size=500)
bpe.train(corpus[:1000])  # Use subset for demo

# Test encoding
sample_question = "How to invest in Bitcoin?"
encoded = bpe.encode(sample_question)
print(f"Manual BPE: {encoded}")


# Comparing with tokenizer library

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# Initialize BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

# Train
trainer = trainers.BpeTrainer(vocab_size=500)
tokenizer.train_from_iterator(corpus[:1000], trainer=trainer)

# Test encoding
encoded_lib = tokenizer.encode(sample_question).tokens
print(f"Library BPE: {encoded_lib}")


# Combine all questions into a corpus
corpus = df_sample['question1'].tolist() + df_sample['question2'].tolist()

# Initialize and train BPE
bpe = ManualBPE(vocab_size=2000)  # Adjust vocab size as needed
bpe.train(corpus, verbose=True)

# Test the tokenizer on a sample question
sample_question = "How to invest in Bitcoin?"
encoded = bpe.encode(sample_question)
print(f"\nSample tokenization:")
print(f"Original: '{sample_question}'")
print(f"Tokenized: {encoded}")

# Let's see how many tokens are in our vocabulary
print(f"\nVocabulary size: {len(bpe.vocab)}")
print("Sample vocabulary items:")
print(list(bpe.vocab)[:20])  # Show first 20 tokens


# Convert tokens to indices
word_index = {token: i+2 for i, token in enumerate(bpe.vocab)}  # +2 for [PAD]=0, [UNK]=1
word_index['[PAD]'] = 0
word_index['[UNK]'] = 1

# Function to tokenize and numericalize questions
def encode_question(question):
    tokens = bpe.encode(question)
    return [word_index.get(token, 1) for token in tokens]  # 1 = [UNK]

# Process all questions
q1_indices = [torch.tensor(encode_question(q)) for q in df_sample['question1']]
q2_indices = [torch.tensor(encode_question(q)) for q in df_sample['question2']]
labels = torch.tensor(df_sample['is_duplicate'].values, dtype=torch.float32)

# Pad sequences to same length
max_len = 64  # Truncate/pad to 64 tokens
q1_padded = pad_sequence([x[:max_len] for x in q1_indices], batch_first=True, padding_value=0)
q2_padded = pad_sequence([x[:max_len] for x in q2_indices], batch_first=True, padding_value=0)

print(f"Padded question shape: {q1_padded.shape}")
print(f"Vocabulary size: {len(word_index)}")

# Let's see an example of encoded questions
idx = 0
print(f"\nOriginal Q1: {df_sample.iloc[idx]['question1']}")
print(f"Encoded Q1: {q1_padded[idx][:20]}...")  # Show first 20 tokens


class SiameseBPE(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512):  # Larger dimensions
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2)  # Add layers
        
        lstm_output_dim = hidden_dim * 2
        combined_dim = lstm_output_dim * 3
        
        # Add intermediate layers
        self.fc1 = nn.Linear(combined_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward_one(self, x):
        emb = self.embedding(x)
        _, (hidden, _) = self.lstm(emb)
        # Take the hidden state from the last layer
        return torch.cat((hidden[-2], hidden[-1]), dim=1)
    
    def forward(self, q1, q2):
        out1 = self.forward_one(q1)
        out2 = self.forward_one(q2)
        combined = torch.cat([out1, out2, torch.abs(out1 - out2)], dim=1)
        x = self.relu(self.fc1(combined))
        x = self.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)).squeeze()


class QuoraDataset(Dataset):
    def __init__(self, q1, q2, labels):
        self.q1 = q1
        self.q2 = q2
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.q1[idx], self.q2[idx], self.labels[idx]

# Split data into train/validation sets
q1_train, q1_val, q2_train, q2_val, y_train, y_val = train_test_split(
    q1_padded, q2_padded, labels, test_size=0.2, random_state=42
)

# Create datasets
train_dataset = QuoraDataset(q1_train, q2_train, y_train)
val_dataset = QuoraDataset(q1_val, q2_val, y_val)

# Create dataloaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for q1, q2, labels in dataloader:
        q1, q2, labels = q1.to(device), q2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(q1, q2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for q1, q2, labels in dataloader:
            q1, q2, labels = q1.to(device), q2.to(device), labels.to(device)
            
            outputs = model(q1, q2)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return total_loss / len(dataloader), accuracy, f1


# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model
vocab_size = len(word_index)
model = SiameseBPE(vocab_size).to(device)

# Training hyperparameters
criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

# Training loop
num_epochs = 10
best_val_loss = float('inf')

# For tracking metrics
train_losses = []
val_losses = []
val_accuracies = []
val_f1s = []

print("Starting training...")
for epoch in range(num_epochs):
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    train_losses.append(train_loss)
    
    # Evaluate
    val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    val_f1s.append(val_f1)
    
    # Adjust learning rate
    scheduler.step(val_loss)
    
    # Print progress
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_quora_model.pt')
        print("  Saved new best model!")

# Load best model
model.load_state_dict(torch.load('best_quora_model.pt'))
final_loss, final_acc, final_f1 = evaluate(model, val_loader, criterion, device)
print("\nFinal Evaluation:")
print(f"Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}, F1: {final_f1:.4f}")


# Plot training curves
plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

# Metrics plot
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Accuracy')
plt.plot(val_f1s, label='F1 Score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.title('Validation Metrics')

plt.tight_layout()
plt.show()


# Function to predict similarity between two questions
def predict_similarity(q1, q2, model, bpe, word_index, device, threshold=0.5):
    # Tokenize and encode
    q1_tokens = encode_question(q1)
    q2_tokens = encode_question(q2)
    
    # Convert to tensors and pad
    q1_tensor = torch.tensor([q1_tokens], device=device)
    q2_tensor = torch.tensor([q2_tokens], device=device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        similarity = model(q1_tensor, q2_tensor).item()
    
    is_duplicate = similarity >= threshold
    return similarity, is_duplicate

# Test on some examples
test_pairs = [
    # Similar questions
    ("How do I lose weight fast?", "What are the best ways to lose weight quickly?"),
    ("How can I learn Python programming?", "What's the best way to start learning Python?"),
    
    # Different questions
    ("How do I cook pasta?", "What is the capital of Italy?"),
    ("What is the meaning of life?", "How do I change a flat tire?")
]

print("Testing on custom examples:")
print("-" * 80)
for q1, q2 in test_pairs:
    similarity, is_duplicate = predict_similarity(q1, q2, model, bpe, word_index, device)
    print(f"Q1: {q1}")
    print(f"Q2: {q2}")
    print(f"Similarity score: {similarity:.4f}")
    print(f"Predicted as duplicate: {is_duplicate}")
    print("-" * 80)


from transformers import AutoTokenizer

# Load pre-trained BPE tokenizer (e.g., GPT-2's tokenizer)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Tokenize using library BPE
def tokenize_lib(text):
    return tokenizer.encode(text, add_special_tokens=False)

# Compare outputs
print("Manual BPE:", encode_question("How to invest in Bitcoin?"))
print("Library BPE:", tokenize_lib("How to invest in Bitcoin?"))




