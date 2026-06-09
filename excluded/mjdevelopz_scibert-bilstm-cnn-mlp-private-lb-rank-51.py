import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import random
import nltk
from nltk.corpus import stopwords
import re


def read_texts_from_dir(dir_path):
  """
  Reads the texts from a given directory and saves them in the pd.DataFrame with columns ['id', 'file_1', 'file_2'].

  Params:
    dir_path (str): path to the directory with data
  """
  dir_count = sum(os.path.isdir(os.path.join(root, d)) for root, dirs, _ in os.walk(dir_path) for d in dirs)
  data=[0 for _ in range(dir_count)]
  print(f"Number of directories: {dir_count}")

  i=0
  for folder_name in sorted(os.listdir(dir_path)):
    folder_path = os.path.join(dir_path, folder_name)
    if os.path.isdir(folder_path):
      try:
        with open(os.path.join(folder_path, 'file_1.txt'), 'r', encoding='utf-8') as f1:
          text1 = f1.read().strip()
        with open(os.path.join(folder_path, 'file_2.txt'), 'r', encoding='utf-8') as f2:
          text2 = f2.read().strip()
        index = int(folder_name[-4:])
        data[i]=(index, text1, text2)
        i+=1
      except Exception as e:
        print(f"Error reading directory {folder_name}: {e}")


  df = pd.DataFrame(data, columns=['id', 'file_1', 'file_2']).set_index('id')
  return df


train_path="/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
df_train=read_texts_from_dir(train_path)
test_path="/kaggle/input/fake-or-real-the-impostor-hunt/data/test"
df_test=read_texts_from_dir(test_path)
df_train.to_csv('df_train.csv', index=False)
df_test.to_csv('df_test.csv',index=False)


df_train = pd.read_csv('df_train.csv')
df_train = df_train.reset_index().rename(columns={"index": "id"})
df_train


df_test = pd.read_csv('df_test.csv')
df_test


train_labels = pd.read_csv('/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv')
train_labels


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

df = df_train.merge(train_labels, on="id")
df["label"] = df["real_text_id"].apply(lambda x: 0 if x == 1 else 1)

df["file_1"] = df["file_1"].astype(str).fillna("")
df["file_2"] = df["file_2"].astype(str).fillna("")


def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["file_1"] = df["file_1"].astype(str).apply(clean_text)
df["file_2"] = df["file_2"].astype(str).apply(clean_text)

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
bert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device)
bert_model.eval()

def get_bert_embeddings(texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch_texts, return_tensors="pt",
                               max_length=512, truncation=True, padding=True).to(device)
            outputs = bert_model(**inputs)
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)



emb_1 = get_bert_embeddings(df["file_1"].tolist())
emb_2 = get_bert_embeddings(df["file_2"].tolist())


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

labels = torch.tensor(df["label"].values, dtype=torch.long)
class SiameseDataset(Dataset):
    def __init__(self, emb1, emb2, labels, augment=True):
        self.emb1 = emb1
        self.emb2 = emb2
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        e1, e2, lbl = self.emb1[idx], self.emb2[idx], self.labels[idx]
        if self.augment and random.random() < 0.5:
            e1, e2 = e2, e1
            lbl = 1 - lbl
        return e1, e2, lbl



class SiameseMLP(nn.Module):
    def __init__(self, emb_dim=768, lstm_hidden=256, cnn_out_channels=128, kernel_size=3):
        super().__init__()
        self.shared_fc = nn.Linear(emb_dim, emb_dim) 

        self.bilstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.cnn = nn.Conv1d(in_channels=2*lstm_hidden, out_channels=cnn_out_channels, kernel_size=kernel_size, padding=1)

        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_channels*3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward_once(self, x):
        x = self.shared_fc(x).unsqueeze(1) 
        lstm_out, _ = self.bilstm(x)       
        lstm_out = lstm_out.transpose(1, 2) 
        cnn_out = self.cnn(lstm_out)        
        cnn_out = torch.mean(cnn_out, dim=2)  
        return cnn_out

    def forward(self, e1, e2):
        h1 = self.forward_once(e1)
        h2 = self.forward_once(e2)
        diff = h1 - h2
        diff = F.normalize(diff)
        h = torch.cat([h1, h2, diff], dim=1)
        return self.classifier(h)


num_epochs = 15
patience = 3
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies, fold_val_losses = [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(emb_1, labels)):
    X1_train, X2_train = emb_1[train_idx], emb_2[train_idx]
    X1_val, X2_val = emb_1[val_idx], emb_2[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    train_dataset = SiameseDataset(X1_train, X2_train, y_train, augment=True)
    val_dataset = SiameseDataset(X1_val, X2_val, y_val, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model_cls = SiameseMLP(emb_dim=emb_1.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model_cls.parameters(), lr=0.00005, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        
        model_cls.train()
        running_loss = 0.0
        for e1, e2, lbl in train_loader:
            e1, e2, lbl = e1.to(device), e2.to(device), lbl.to(device)
            optimizer.zero_grad()
            logits = model_cls(e1, e2)
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * e1.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        
        model_cls.eval()
        val_loss = 0.0
        all_preds = []
        with torch.no_grad():
            for e1, e2, lbl in val_loader:
                e1, e2, lbl = e1.to(device), e2.to(device), lbl.to(device)
                logits = model_cls(e1, e2)
                loss = criterion(logits, lbl)
                val_loss += loss.item() * e1.size(0)
                preds = torch.argmax(logits, dim=1).cpu()
                all_preds.append(preds)

        all_preds = torch.cat(all_preds)
        acc = accuracy_score(y_val, all_preds)
        val_loss /= len(val_loader.dataset)

        print(f"Fold {fold+1} Epoch {epoch+1}: "
              f"Train Loss = {epoch_loss:.4f}, "
              f"Val Loss = {val_loss:.4f}, "
              f"Val Acc = {acc:.4f}")

       
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model_cls.state_dict() 
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

   
    model_cls.load_state_dict(best_model_state)
    fold_accuracies.append(acc)
    fold_val_losses.append(val_loss)

print("Mean CV Accuracy:", np.mean(fold_accuracies))
print("Mean Val Loss:", np.mean(fold_val_losses))


full_dataset = SiameseDataset(emb_1, emb_2, labels, augment=True)
full_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)

final_model = SiameseMLP(emb_dim=emb_1.shape[1]).to(device)
optimizer = torch.optim.AdamW(final_model.parameters(), lr=0.00005, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(15):
    final_model.train()
    running_loss = 0.0
    for e1, e2, lbl in full_loader:
        e1, e2, lbl = e1.to(device), e2.to(device), lbl.to(device)
        optimizer.zero_grad()
        logits = final_model(e1, e2)
        loss = criterion(logits, lbl)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * e1.size(0)  

    epoch_loss = running_loss / len(full_loader.dataset)
    print(f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f}")


df_test = pd.read_csv("df_test.csv")
df_test = df_test.reset_index().rename(columns={"index": "id"})
df_test["file_1"] = df_test["file_1"].astype(str).fillna("")
df_test["file_2"] = df_test["file_2"].astype(str).fillna("")

emb_1_test = get_bert_embeddings(df_test["file_1"].tolist())
emb_2_test = get_bert_embeddings(df_test["file_2"].tolist())

test_dataset = SiameseDataset(emb_1_test, emb_2_test, torch.zeros(len(df_test)), augment=False)
test_loader = DataLoader(test_dataset, batch_size=32)

final_model.eval()
all_preds = []
with torch.no_grad():
    for e1, e2, _ in test_loader:
        e1, e2 = e1.to(device), e2.to(device)
        logits = final_model(e1, e2)
        preds = torch.argmax(logits, dim=1).cpu()
        all_preds.append(preds)
all_preds = torch.cat(all_preds)

df_test["real_text_id"] = np.where(all_preds.numpy() == 0, 1, 2)
submission = df_test[["id", "real_text_id"]]
submission.to_csv("submission.csv", index=False)
print("Submission saved!")
submission

