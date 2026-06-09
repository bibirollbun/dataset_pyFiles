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


import pandas as pd
# data=pd.read_csv('/content/sentiment-analysis-on-movie-reviews/sampleSubmission.csv')
train_df=pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip', sep='\t')
test_df=pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip', sep='\t')
data = train_df.to_dict('records')
train_df = pd.DataFrame(data)
print(f"Total reviews: {len(train_df)}")
print(f"Columns: {train_df.columns.tolist()}")

print(f"Total reviews test : {len(test_df)}")
print(f"Columns test : {test_df.columns.tolist()}")

# train_df.head()



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import re


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

df = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip', sep='\t')
df.dropna(subset=['PhraseId', 'SentenceId', 'Phrase', 'Sentiment'], inplace=True)
df['clean_review'] = df['Phrase'].apply(preprocess_text)

X = df['clean_review'].tolist()
y = df['Sentiment'].tolist()


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


tokenized = [word_tokenize(sent) for sent in X]
word_counts = Counter(word for sent in tokenized for word in sent)
vocab = {word: i+2 for i, (word, _) in enumerate(word_counts.items())}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1


def encode_sentence(sent, vocab, max_len=50):
    tokens = word_tokenize(sent)
    ids = [vocab.get(word, vocab['<UNK>']) for word in tokens]
    if len(ids) < max_len:
        ids += [vocab['<PAD>']] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

encoded_X = [encode_sentence(sent, vocab) for sent in X]



class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train, X_val, y_train, y_val = train_test_split(encoded_X, y, test_size=0.2, random_state=42, stratify=y)

train_ds = TextDataset(X_train, y_train)
val_ds = TextDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)



import torch
import torch.nn as nn
import torch.optim as optim

class SimpleRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SimpleRNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(hidden.squeeze(0))
        return logits, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.rnn.hidden_size).to(device)

def train_with_logging(model, data_loader, optimizer, criterion, seq_len, epochs, device):
    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size, device)

            optimizer.zero_grad()
            logits, hidden = model(inputs, hidden)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    return losses

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleRNNClassifier(
    vocab_size=len(vocab),
    embed_dim=256,
    hidden_dim=256,
    output_dim=5  
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


train_with_logging(model, train_loader, optimizer, criterion, seq_len=50, epochs=5, device=device)


model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        logits, _ = model(inputs)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


cm = confusion_matrix(all_labels, all_preds)
class_names = label_encoder.classes_.astype(str)


print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_.astype(str)))

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()




