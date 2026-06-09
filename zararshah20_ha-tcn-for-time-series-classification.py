import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns 
import lightgbm as lgb
from catboost import Pool, CatBoostClassifier
import itertools
import pickle, gzip
import glob
from sklearn.preprocessing import StandardScaler

# C1


import pandas as pd
df = pd.read_csv('/kaggle/input/training_set.csv')
print(df.head())



## import torch
from sklearn.preprocessing import LabelEncoder

# Load data
train = pd.read_csv('/kaggle/input/training_set.csv')
meta = pd.read_csv('/kaggle/input/training_set_metadata.csv')

# Encode target labels
labels = meta[['object_id', 'target']]
label_encoder = LabelEncoder()
labels['target'] = label_encoder.fit_transform(labels['target'])

# Merge flux data with labels
data = train.merge(labels, on='object_id')

#  --  --
features = ['flux', 'flux_err', 'passband']
sequence_data = {}
max_len = 0

# Group by object_id and get time-sorted sequences
for obj_id, group in data.groupby('object_id'):
    seq = group.sort_values('mjd')[features].values
    if len(seq) > max_len:
        max_len = len(seq)
    sequence_data[obj_id] = seq

# Pad sequences to the same length
def pad_sequence(seq, max_len):
    padded = np.zeros((max_len, len(features)))
    padded[:len(seq)] = seq
    return padded

X = []
y = []

for obj_id in sequence_data:
    padded = pad_sequence(sequence_data[obj_id], max_len)
    X.append(padded)
    y.append(labels[labels['object_id'] == obj_id]['target'].values[0])

X = np.array(X)
y = np.array(y)

print(f"Shape of X: {X.shape} | Shape of y: {y.shape}")


#C2


import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.norm(out)
        return out

class HA_TCN(nn.Module):
    def __init__(self, input_size, num_classes, num_channels=[32, 64, 128], kernel_size=3):
        super(HA_TCN, self).__init__()
        layers = []
        dilation = 1
        for i in range(len(num_channels)):
            in_ch = input_size if i == 0 else num_channels[i-1]
            layers.append(TemporalBlock(in_ch, num_channels[i], kernel_size, dilation))
            dilation *= 2
        self.network = nn.Sequential(*layers)

        # Attention: Temporal attention (HA-TCN)
        self.attn = nn.Linear(num_channels[-1], 1)
        self.classifier = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, T)
        features = self.network(x).permute(0, 2, 1)  # (B, T, C)
        attn_weights = F.softmax(self.attn(features), dim=1)  # (B, T, 1)
        attended = torch.sum(features * attn_weights, dim=1)  # (B, C)
        return self.classifier(attended)



import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HA_TCN(input_size=3, num_classes=len(label_encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
weight_decay = 1e-2  # Adjust this value as needed
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007, weight_decay=weight_decay)

# Training
for epoch in range(500):  # keep this small for now
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    if(epoch +1)%20==0:
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f}")



# Evaluate
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        _, predicted = torch.max(preds, 1)
        correct += (predicted == yb).sum().item()
        total += yb.size(0)

print(f"Validation Accuracy: {(correct / total) * 100:.2f}%")


from sklearn.metrics import log_loss

model.eval()
y_true = []
y_probs = []

with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        y_probs.extend(probs)
        y_true.extend(yb.numpy())

logloss = log_loss(y_true, y_probs)
print(f"Validation Log Loss: {logloss:.4f}")



torch.save(model.state_dict(), "ha_tcn_phase1.pth")



# # Reload libraries and model class
# import torch
# import torch.nn.functional as F

# # Load the same HA-TCN model definition
# # (Paste your model class again here)

# # Instantiate model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = HA_TCN(input_size=3, num_classes=14).to(device)
# model.load_state_dict(torch.load("/kaggle/input/your-model-path/ha_tcn_phase1.pth"))
# model.eval()



# import pandas as pd

# chunk_size = 500000  # Load in 500k row chunks
# test_reader = pd.read_csv("/kaggle/input/PLAsTiCC-2018/test_set.csv", chunksize=chunk_size)



import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict
from tqdm import tqdm  # nice progress bar

# Required setup
test_path = "/kaggle/input/test_set.csv"
chunk_size = 500_000
max_len = 352  # same as training
features = ['flux', 'flux_err', 'passband']
object_preds = {}

# Padding function
def pad_sequence(seq, max_len):
    padded = np.zeros((max_len, len(features)))
    padded[:len(seq)] = seq
    return padded

# Reader loop
reader = pd.read_csv(test_path, chunksize=chunk_size)

for i, chunk in enumerate(reader):
    print(f"\nðŸ”„ Processing chunk {i+1}")
    
    sequences = []
    obj_ids = []
    
    for obj_id, group in chunk.groupby('object_id'):
        seq = group.sort_values('mjd')[features].values
        padded = pad_sequence(seq, max_len)
        sequences.append(padded)
        obj_ids.append(obj_id)
    
    X_chunk = torch.tensor(sequences).float().to(device)
    
    with torch.no_grad():
        preds = model(X_chunk)
        probs = F.softmax(preds, dim=1).cpu().numpy()
    
    for obj_id, prob in zip(obj_ids, probs):
        object_preds[obj_id] = prob



# Load metadata to get actual target classes
meta = pd.read_csv("/kaggle/input/training_set_metadata.csv")
classes = sorted(meta["target"].unique())
columns = ['class_' + str(c) for c in classes]

submission = pd.DataFrame.from_dict(object_preds, orient='index', columns=columns)
submission.index.name = "object_id"
submission.reset_index(inplace=True)
# Add a dummy class_99 column with very low probabilities
submission['class_99'] = 1e-5

# Normalize all class columns so they sum to 1
class_cols = [col for col in submission.columns if col.startswith("class_")]
submission[class_cols] = submission[class_cols].div(submission[class_cols].sum(axis=1), axis=0)

submission.to_csv("V2Final.csv", index=False)




print("âœ… Fixed submission.csv saved with class_99 included.")


# # Convert dict to DataFrame
# columns = ['class_' + str(c) for c in sorted(meta['target'].unique())]
# submission = pd.DataFrame.from_dict(object_preds, orient='index', columns=columns)
# submission.index.name = "object_id"
# submission.reset_index(inplace=True)

# submission.to_csv("V2Final.csv", index=False)



# def plot_loss_acc(history):
#     plt.plot(history.history['loss'][1:])
#     plt.plot(history.history['val_loss'][1:])
#     plt.title('model loss')
#     plt.ylabel('val_loss')
#     plt.xlabel('epoch')
#     plt.legend(['train','Validation'], loc='upper left')
#     plt.show()
    
#     plt.plot(history.history['acc'][1:])
#     plt.plot(history.history['val_acc'][1:])
#     plt.title('model Accuracy')
#     plt.ylabel('val_acc')
#     plt.xlabel('epoch')
#     plt.legend(['train','Validation'], loc='upper left')
#     plt.show()

