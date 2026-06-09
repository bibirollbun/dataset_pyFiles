import pandas as pd
import numpy as np
import os
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk.data
from collections import Counter
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


train_df1 = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip", header=0, delimiter="\t", quoting=3)
test_df = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip", header=0, delimiter="\t", quoting=3)
train_df1 = train_df1.drop(['id'], axis=1) 

train_df2 = pd.read_csv('../input/imdb-review-dataset/imdb_master.csv', encoding="latin-1")

    
train_df2 = train_df2.drop(['Unnamed: 0','type','file'], axis=1)
train_df2.columns = ["review","sentiment"]
train_df2 = train_df2[train_df2.sentiment != 'unsup'] # 去掉无标签数据
train_df2['sentiment'] = train_df2['sentiment'].map({'pos': 1, 'neg': 0})

train_df = pd.concat([train_df1, train_df2]).reset_index(drop=True)
print(f"数据增强完成！总训练样本数: {len(train_df)}")

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()

def clean_text_lemmatize(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^\w\s]','',text, re.UNICODE) 
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words and len(word) > 1]
    return " ".join(text)


train_df = pd.concat([train_df1, train_df2]).reset_index(drop=True)

train_df = train_df[~train_df['review'].isin(test_df['review'])]

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()

def clean_text_lemmatize(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words and len(word) > 1]
    return " ".join(text)

train_df['Processed_Reviews'] = train_df['review'].apply(clean_text_lemmatize)
test_df['Processed_Reviews'] = test_df['review'].apply(clean_text_lemmatize)

VOCAB_SIZE = 25000 
MAX_LEN = 1317      
EMBEDDING_DIM = 16

word_counts = Counter(word for review in train_df['Processed_Reviews'] for word in review.split())
top_words = word_counts.most_common(VOCAB_SIZE - 2)
word_to_idx = {"<pad>": 0, "<unk>": 1}
for i, (word, count) in enumerate(top_words, start=2):
    word_to_idx[word] = i

def texts_to_sequences(reviews, word_to_idx):
    return [[word_to_idx.get(word, 1) for word in review.split()] for review in reviews]

X_all_seq = texts_to_sequences(train_df['Processed_Reviews'], word_to_idx)
X_test_seq = texts_to_sequences(test_df['Processed_Reviews'], word_to_idx)

def pad_sequences_manual(sequences, maxlen):
    padded = np.zeros((len(sequences), maxlen), dtype=int)
    for i, seq in enumerate(sequences):
        padded[i, -len(seq):] = seq[:maxlen] if len(seq) > maxlen else seq
    return padded

X_all = pad_sequences_manual(X_all_seq, maxlen=MAX_LEN)
y_all = train_df['sentiment'].values
X_test = pad_sequences_manual(X_test_seq, maxlen=MAX_LEN)

X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

X_train_tensor = torch.tensor(X_train, dtype=torch.long)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.long)

BATCH_SIZE = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor)

use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):
        attn_weights = torch.tanh(self.attn(lstm_output))
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context, attn_weights

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob=0.5):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.embed_dropout = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout_prob)
        self.attention = Attention(hidden_dim)
        self.fc_dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.embed_dropout(embedded)
        lstm_out, _ = self.lstm(embedded)
        context, _ = self.attention(lstm_out)
        dropped_out = self.fc_dropout(context)
        output = self.fc(dropped_out)
        return output

# --- 超参数 ---
HIDDEN_DIM = 8     
EPOCHS = 100          
LEARNING_RATE = 1e-3 
DROPOUT_PROB = 0.2   
L2_LAMBDA = 1e-5



model = LSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT_PROB)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_LAMBDA)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)

best_val_acc = 0.0          
best_val_loss = float('inf') 
patience_counter = 0
PATIENCE = 5     
best_acc_model_path = 'best_model_by_accuracy.pth'
best_loss_model_path = 'best_model_by_loss.pth'
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

print(f"\n开始训练 (早停基于 Accuracy)...")

for epoch in range(EPOCHS):
    model.train()
    total_train_loss, correct_train, total_train = 0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)
    avg_train_loss = total_train_loss / len(train_loader)
    train_acc = correct_train / total_train
    
    model.eval()
    total_val_loss, correct_val, total_val = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels) 
            total_val_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)
    avg_val_loss = total_val_loss / len(val_loader)
    val_acc = correct_val / total_val

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.1e} | "
          f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_loss_model_path)
        print(f"  --- New Best Loss! Saving model to {best_loss_model_path}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_acc_model_path)
        patience_counter = 0 
        print(f"  >>> New Best Accuracy! Saving model to {best_acc_model_path}")
    else:
        patience_counter += 1
        print(f"  Patience (on Acc): {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered based on Accuracy.")
            break


def predict_with_model(model_path, loader):
    model_instance = LSTMClassifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT_PROB)
    model_instance.load_state_dict(torch.load(model_path))
    model_instance.to(device)
    model_instance.eval()
    
    all_probabilities = []
    with torch.no_grad():
        for (inputs,) in loader:
            inputs = inputs.to(device)
            outputs = model_instance(inputs)
            probs = torch.sigmoid(outputs)
            all_probabilities.extend(probs.cpu().numpy())
            
    return np.array(all_probabilities)

probabilities_acc = predict_with_model(best_acc_model_path, test_loader)
sentiments_acc = (probabilities_acc > 0.5).astype(int).flatten()
output_acc = pd.DataFrame(data={"id": test_df["id"], "sentiment": sentiments_acc})
output_acc.to_csv("/kaggle/working/submission_acc.csv", index=False, quoting=3)
output_acc.to_csv("/kaggle/working/submission.csv", index=False, quoting=3)

probabilities_loss = predict_with_model(best_loss_model_path, test_loader)
sentiments_loss = (probabilities_loss > 0.5).astype(int).flatten()
output_loss = pd.DataFrame(data={"id": test_df["id"], "sentiment": sentiments_loss})
output_loss.to_csv("/kaggle/working/submission_loss.csv", index=False, quoting=3)

ensemble_probabilities = (probabilities_acc + probabilities_loss) / 2.0

final_sentiments_ensemble = (ensemble_probabilities > 0.5).astype(int).flatten()
output_ensemble = pd.DataFrame(data={"id": test_df["id"], "sentiment": final_sentiments_ensemble})
output_ensemble.to_csv("/kaggle/working/submission_mixed.csv", index=False, quoting=3)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Val')
plt.title('Accuracy')
plt.legend()
plt.show()

