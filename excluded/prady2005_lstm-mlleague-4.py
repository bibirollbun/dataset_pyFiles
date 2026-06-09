import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,pad_sequence
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight


df = pd.read_csv("/kaggle/input/comments-classification/Dataset/train.csv")


df.shape


df.isnull().sum()


df['psychotic_depression'].value_counts()


df.info()


class RawCommentsData(Dataset):
    def __init__(self,path):
        self.df = pd.read_csv(path)
        

    def __getitem__(self,idx):
        comment = self.df.iloc[idx]['comment_text']
        psychotic_depression = self.df.iloc[idx]['psychotic_depression']
        return comment,psychotic_depression

    def __len__(self):
        return len(self.df)


class RawCommentsTest(Dataset):
    def __init__(self,path):
        self.df = pd.read_csv(path)
        

    def __getitem__(self,idx):
        comment = self.df.iloc[idx]['comment_text']
        return comment

    def __len__(self):
        return len(self.df)


raw_train = RawCommentsData("/kaggle/input/comments-classification/Dataset/train.csv")


raw_test = RawCommentsTest("/kaggle/input/comments-classification/Dataset/test.csv")


embedding_dict = {}

with open("/kaggle/input/glooveembeddings/other/default/1/glove.6B.50d.txt",'r',encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:],"float32")
        embedding_dict[word] = vector


embedding_dict["the"]


nltk.download("punkt")
def createCommentEmbedding(comment):
    comment_list = word_tokenize(comment.lower())
    embedding_list = []
    for token in comment_list:
        if token in embedding_dict:
            embedding_list.append(torch.tensor(embedding_dict[token]))
        else:
            embedding_list.append(torch.zeros_like(torch.tensor(embedding_dict["the"],dtype=torch.float)))

    if not embedding_list:
        embedding_list.append(torch.zeros_like(torch.tensor(embedding_dict["the"], dtype=torch.float)))

    return embedding_list


def createTensor(dataset):
    features = []
    targets = []
    for i in range(len(dataset)):
        comment,label = dataset[i]
        embedding_list = createCommentEmbedding(comment)
        if embedding_list:
            embedding_list = torch.stack(embedding_list)
            features.append(torch.tensor(embedding_list))
            targets.append(label)
    return features,targets


def createTestTensor(dataset):
    features = []
    for i in range(len(dataset)):
        comment = dataset[i]
        embedding_list = createCommentEmbedding(comment)
        if embedding_list:
            embedding_list = torch.stack(embedding_list)
            features.append(torch.tensor(embedding_list))
        
    return features


X_train, y_train = createTensor(raw_train)


X_test = createTestTensor(raw_test)


import pickle

with open("X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)


with open("y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)


# with open("/kaggle/input/train-list/X_train.pkl", "rb") as f:
#     X_train = pickle.load(f)

# with open("/kaggle/input/train-list/y_train.pkl", "rb") as f:
#     y_train = pickle.load(f)


class CommentsDataset(Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = y
        

    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


class CommentsTestDataset(Dataset):
    def __init__(self,X):
        self.X = X
        

    def __getitem__(self,idx):
        return self.X[idx]

    def __len__(self):
        return len(self.X)    
    


ssf = StratifiedShuffleSplit(n_splits=1,test_size=0.2)
split = ssf.split(X_train,y_train)


train_split, val_split = next(ssf.split(X_train, y_train))


X_tr = [X_train[ind] for ind in train_split]
y_tr = [y_train[ind] for ind in train_split]
X_vl = [X_train[ind] for ind in val_split]
y_val = [y_train[ind] for ind in val_split]


train_dataset = CommentsDataset(X_tr,y_tr)
val_dataset = CommentsDataset(X_vl,y_val)


test_dataset = CommentsTestDataset(X_test)


def collate_fn_test(batch):
    return list(batch)


def collate_fn(batch):
    X, y = zip(*batch)
    y = torch.tensor(y, dtype=torch.long)      
    return list(X), y 


train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True,collate_fn=collate_fn)
val_loader = DataLoader(val_dataset,batch_size=32,shuffle=True,collate_fn=collate_fn)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMmodel(nn.Module):
    def __init__(self,embedding_dim,hidden_size,num_layers,output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        self.classifier = nn.Linear(hidden_size,output_size)

    def forward(self,X):
        X = [seq.to(device) for seq in X]
        lengths = [len(seq) for seq in X]
        X_padded = pad_sequence(X, batch_first=True)
        packed = pack_padded_sequence(X_padded, lengths, batch_first=True, enforce_sorted=False)      
        out,(h,c) = self.lstm(packed)
        feat = h[-1] 
        out = self.classifier(feat)
        return out


model = LSTMmodel(50,100,3,2).to(device)


def train(epoch):
    pbar = tqdm(train_loader,total=len(train_loader),desc=f"Train Epoch {epoch}/{epochs}")
    epoch_loss = 0.0
    for x,y in pbar:
        y= y.to(device)
        out = model(x)
        total_loss = loss(out,y)
        epoch_loss += total_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    epoch_loss = epoch_loss/len(train_loader)
    return epoch_loss


weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)


def val():
    model.eval()
    total_val_loss = 0.0
    for x,y in val_loader:
        y = y.to(device)
        out = model(x)
        val_loss = loss(out,y)
        total_val_loss+=val_loss

    total_val_loss = total_val_loss/len(val_loader)
    return total_val_loss


loss = nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32).to(device))
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
epochs = 10
patience = 5
count = 0
best_val_loss = 1e9

for epoch in range(epochs):
    model.train()
    train_loss = train(epoch)
    validation_loss = val()
    print(f"train_loss : {train_loss:.4f} | val_loss : {validation_loss:.4f}")

    if validation_loss < best_val_loss:
        count = 0
        torch.save(model.state_dict(),"model.pth")

    else:
        count+=1

    if count == patience:
        break


model.load_state_dict(torch.load("/kaggle/working/model.pth"))


y_pred = []
y_target = []
model.eval()
for x,y in val_loader:
    out = model(x)
    out = out.detach().cpu()
    y_pred.extend(torch.argmax(out,dim=1))
    y_target.extend(y)


from sklearn.metrics import f1_score
f1_score(y_target,y_pred)


df = pd.read_csv("/kaggle/input/comments-classification/Dataset/test.csv")


test_loader = DataLoader(test_dataset,batch_size=32,shuffle=True,collate_fn=collate_fn_test)


y_pred = []
model.eval()
for x in test_loader:
    out = model(x)
    out = out.detach().cpu()
    y_pred.extend(torch.argmax(out, dim=1).tolist())


result = pd.DataFrame({
    'ID': range(1,len(y_pred)+1),
    'psychotic_depression': y_pred
})
result.to_csv('result.csv',index=False)

