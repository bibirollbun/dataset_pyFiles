import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from xgboost import XGBClassifier
import re, html
from tqdm.auto import tqdm
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
tqdm.pandas()
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download('averaged_perceptron_tagger_eng')

%matplotlib inline


df = pd.read_csv(
    "/kaggle/input/ttic-31020-2025a-hw-2-spam-detection/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    quoting=3,
    on_bad_lines="skip"
)


df.head()


df.shape


df.isnull().sum()


df.info()


df["label"].value_counts()


df['label'] = df['label'].map({'ham': -1, 'spam': 1})


STOP = set(stopwords.words("english"))

def clean(text):
    if not isinstance(text, str): 
        return ""
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in word_tokenize(text) if t not in STOP and len(t) > 1]
    return " ".join(tokens)


# df["message"] = df["message"].progress_apply(clean)


df.head()


# Features and labels
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)


model = CatBoostClassifier(iterations=1000,depth=6,learning_rate=0.005,loss_function="Logloss",verbose=100,random_seed=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Precision
prec = precision_score(y_test, y_pred, average='binary')  
print("Precision:", prec)

# Recall
rec = recall_score(y_test, y_pred, average='binary')
print("Recall:", rec)

# F1-score
f1 = f1_score(y_test, y_pred, average='binary')
print("F1-score:", f1)


test_df = pd.read_csv(
    "/kaggle/input/ttic-31020-2025a-hw-2-spam-detection/SMSSpamCollection_test_text",
    sep="\t",
    header=None,
    names=["label", "message"],   # <-- two columns
    quoting=3,
    on_bad_lines="skip"
)
test_df = test_df.reset_index().rename(columns={"index": "id"})


test_df.head()


test_df["id"] = range(len(test_df))

# Expand to 2262 rows
total_required = 2262
if len(test_df) < total_required:
    extra_rows = pd.DataFrame({"id": range(len(test_df), total_required),"label": 0,"message": ""})
    test_df = pd.concat([test_df, extra_rows], ignore_index=True)


test_df.shape


Id=test_df.id


test_df.drop(columns=["id","label"],axis=1,inplace=True)


X_test_pred=tfidf.transform(test_df["message"])
pred=model.predict(X_test_pred)
submission=pd.DataFrame({"ID":Id,"LABEL":pred})
submission.to_csv("submission.csv",index=False)
submission.head()


df["label"] = df["label"].replace(-1, 0)


X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)


xgb = XGBClassifier(n_estimators=1000,learning_rate=0.005,max_depth=6,random_state=42,booster='gbtree',tree_method='auto')
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)


# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Precision
prec = precision_score(y_test, y_pred, average='binary')  
print("Precision:", prec)

# Recall
rec = recall_score(y_test, y_pred, average='binary')
print("Recall:", rec)

# F1-score
f1 = f1_score(y_test, y_pred, average='binary')
print("F1-score:", f1)


X_test_pred=tfidf.transform(test_df["message"])
pred=xgb.predict(X_test_pred)
submission=pd.DataFrame({"ID":Id,"LABEL":pred})
submission["LABEL"] = submission["LABEL"].replace(0,-1)
submission.to_csv("new_submission.csv",index=False)
submission.head()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from tqdm import tqdm
from tabulate import tabulate
import nltk
from torchsummary import summary
from nltk.tokenize import word_tokenize
nltk.download('punkt')


df = pd.read_csv(
    "/kaggle/input/ttic-31020-2025a-hw-2-spam-detection/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    quoting=3,
    on_bad_lines="skip"
)


df['label'] = df['label'].map({'ham': 0, 'spam': 1})


df.head()


def tokenize(text):
    return word_tokenize(text.lower())


def build_vocab(texts):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common())}
    vocab['<PAD>'] = 0  # Padding token
    vocab['<UNK>'] = 1  # Unknown token
    return vocab

vocab = build_vocab(df['message'])


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = tokenize(self.texts[idx])
        # Convert words to indices
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in text]
        # Pad or truncate to max_len
        if len(indices) < self.max_len:
            indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


train_texts, val_texts, train_labels, val_labels = train_test_split(df['message'].values, df['label'].values, test_size=0.2, random_state=42)


batch_size=32
train_dataset = TextDataset(train_texts, train_labels, vocab)
val_dataset = TextDataset(val_texts, val_labels, vocab)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


X_batch, y_batch = next(iter(train_loader))
print(X_batch.shape, y_batch.shape)


import torch
import torch.nn as nn
from torchinfo import summary

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text.long())
        _, (hidden, _) = self.lstm(embedded)
        hidden = hidden.squeeze(0)
        return self.fc(hidden)

vocab_size = 10000
embed_dim = 100
hidden_dim = 128
output_dim = 2

model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, output_dim)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

summary(model,input_size=(32, 50),device=device,verbose=1,col_names=["input_size", "output_size", "num_params"])


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)



def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for texts, labels in tqdm(data_loader, desc="Training"):
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return total_loss / len(data_loader), correct / total


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for texts, labels in tqdm(data_loader, desc="Validating"):
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / len(data_loader), correct / total


patience = 5
best_val_loss = float('inf')
counter = 0

num_epochs = 100
results = []

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    results.append([epoch + 1, train_loss, train_acc, val_loss, val_acc])
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")  # save best model
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


headers = ["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"]
print(tabulate(results, headers=headers, tablefmt="grid"))


from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import torch

def predict_and_evaluate(model, data_loader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for texts, labels in data_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
    report = classification_report(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("ðŸ“Š Validation Metrics")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    return acc, precision, recall, f1, cm


acc, prec, rec, f1, cm = predict_and_evaluate(model, val_loader, device)


test_df = pd.read_csv(
    "/kaggle/input/ttic-31020-2025a-hw-2-spam-detection/SMSSpamCollection_test_text",
    sep="\t",
    header=None,
    names=["label", "message"],   # <-- two columns
    quoting=3,
    on_bad_lines="skip"
)
test_df = test_df.reset_index().rename(columns={"index": "id"})


test_df["id"] = range(len(test_df))

# Expand to 2262 rows
total_required = 2262
if len(test_df) < total_required:
    extra_rows = pd.DataFrame({"id": range(len(test_df), total_required),"label": 0,"message": ""})
    test_df = pd.concat([test_df, extra_rows], ignore_index=True)


test_df.drop(columns=["label","id"],axis=1,inplace=True)


test_df.head()


class TestDataset(Dataset):
    def __init__(self, texts, vocab, max_len=50):
        self.texts = texts
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = tokenize(self.texts[idx])
        # Convert words to indices
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in text]
        # Pad or truncate
        if len(indices) < self.max_len:
            indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return torch.tensor(indices, dtype=torch.long)

# Create test dataset & dataloader
test_texts = test_df['message'].values
test_dataset = TestDataset(test_texts, vocab)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



model.eval()
all_preds = []

with torch.no_grad():
    for texts in test_loader:
        texts = texts.to(device)
        outputs = model(texts)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())

# Create submission
submission = pd.DataFrame({"ID": range(len(all_preds)), "LABEL": all_preds})
submission["LABEL"] = submission["LABEL"].replace(0, -1)
submission.to_csv("torch_submission.csv", index=False)
submission.head()







