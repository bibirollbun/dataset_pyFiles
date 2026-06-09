import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import torch.optim as optim
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import copy


train = pd.read_csv('/kaggle/input/10121-yash-agarwal-kfolds-lmsys/train_5folds.csv')

train['text'] = (
    'User prompt: ' + train['prompt'] +
    '\n\nModel A :\n' + train['response_a'] +
    '\n\n--------\n\nModel B:\n' + train['response_b']
)

batch_size = 8
epochs = 5
num_classes = 3


all_texts_tokenized = [t.split() for t in train['text'].values]

vocab = {"<pad>": 0, "<unk>": 1}
for sent in all_texts_tokenized:
    for w in sent:
        if w not in vocab:
            vocab[w] = len(vocab)

vocab_size = len(vocab)

def encode_single(tokens, vocab):
    return [vocab.get(w, 1) for w in tokens]

all_encoded_texts = [
    torch.tensor(encode_single(tokens, vocab), dtype=torch.long)
    for tokens in all_texts_tokenized
]
all_labels = train['label'].values


class TextDataset(Dataset):
    def __init__(self, enc, labels):
        self.enc = enc
        self.labels = labels

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, idx):
        return self.enc[idx], torch.tensor(self.labels[idx])

def collate_fn(batch):
    seqs, labels = zip(*batch)
    seqs = pad_sequence(seqs, batch_first=True)
    return seqs, torch.tensor(labels)


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128, hidden_dim2=64):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 2-layer GRU + dropout
        self.rnn = nn.GRU(
            embed_dim,
            hidden_dim,
            batch_first=True,
            num_layers=2,
            dropout=0.3
        )

        self.fc = nn.Linear(hidden_dim, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, h_n = self.rnn(x)

        final_hidden = h_n[-1]

        # Dropout + ReLU improves generalization
        final_hidden = F.dropout(final_hidden, p=0.3, training=self.training)
        out = F.relu(self.fc(final_hidden))
        out = F.dropout(out, p=0.3, training=self.training)

        logits = self.fc2(out)
        return logits




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


for kfold in range(5):

    print("\n======================================")
    print(f"       Training Fold {kfold}")
    print("======================================")

    train_idx = train[train['kfold'] != kfold].index.tolist()
    test_idx = train[train['kfold'] == kfold].index.tolist()

    train_dataset = TextDataset([all_encoded_texts[i] for i in train_idx],
                                [all_labels[i] for i in train_idx])
    test_dataset = TextDataset([all_encoded_texts[i] for i in test_idx],
                               [all_labels[i] for i in test_idx])

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    model = GRUClassifier(vocab_size=vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()

    # weight decay reduces overfitting
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # scheduler to lower LR when accuracy stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=1, verbose=True
    )

    best_acc = 0
    model_path = f"model_fold{kfold}.pth"

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        batches = 0

        for x, y in tqdm(loader, desc=f"Fold {kfold} | Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            # gradient clipping stabilizes GRU training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            batches += 1

        avg_train_loss = train_loss / batches
        print(f"Train Loss: {avg_train_loss:.4f}")

        model.eval()
        correct = 0
        total = 0
        test_loss = 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                test_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / total
        print(f"Fold {kfold} | Epoch {epoch+1} | Test Loss: {test_loss/len(test_loader):.4f} | Test Acc: {acc:.4f}")

        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
            print(f"Saved NEW BEST MODEL for fold {kfold} → {model_path}")

    print(f"✔ Finished Fold {kfold} | Best Acc = {best_acc:.4f}")


fold_to_load = 2
load_path = f"model_fold{fold_to_load}.pth"

model_loaded = GRUClassifier(vocab_size=vocab_size).to(device)
model_loaded.load_state_dict(torch.load(load_path, map_location=device))
model_loaded.eval()

print(f"Loaded GRU model → {load_path}")


def predict(text):
    tokens = text.split()
    encoded = torch.tensor(encode_single(tokens, vocab)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model_loaded(encoded)
        probs = F.softmax(logits, dim=1)
        print("Probabilities:", probs.cpu().numpy().flatten())

print("\n--- Prediction Examples ---")
predict("this movie was amazing")
predict("this film was awful")

