import sys
sys.path.append('/kaggle/usr/lib/kaggle_metric_utilities')
sys.path.append('/kaggle/usr/lib/birdclef-roc-auc')
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as AT
from torchvision import models

from metric import score


wav_sec = 5
sample_rate = 32000
min_segment = sample_rate*wav_sec

epochs = 8


root_path = "../input/birdclef-2025/" 
input_path = '/kaggle/input/birdclef-2025-train-data/train_raw5/'

class_labels = sorted(os.listdir('../input/birdclef-2025/train_audio/'))

train_meta = pd.read_csv(root_path + 'train.csv')
display(train_meta.head())


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow((specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

def cal_score(label, pred):
    label = np.concatenate(label)
    pred = np.concatenate(pred)

    label_df = pd.DataFrame(label>0.5, columns=class_labels)
    pred_df = pd.DataFrame(pred, columns=class_labels)
    label_df['id'] = np.arange(len(label_df))
    pred_df['id'] = np.arange(len(pred_df))

    return score(label_df, pred_df, row_id_column_name='id')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


n_fft=1024
win_length=1024
hop_length=512
f_min=20
f_max=15000
n_mels=128

mel_spectrogram = AT.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    f_min=f_min,
    f_max=f_max,
    pad_mode="reflect",
    power=2.0,
    norm='slaney',
    n_mels=n_mels,
    mel_scale="htk",
    # normalized=True
)

class BirdclefDataset(Dataset):
    def __init__(self, df, mode='train'):
        self.df = df
        self.mode = mode

    def normalize_std(self, spec, eps=1e-23):
        mean = torch.mean(spec)
        std = torch.std(spec)
        return torch.where(std == 0, spec-mean, (spec - mean) / (std+eps))
                
    def __getitem__(self, index):
        sig, _ = torchaudio.load(uri=input_path+self.df.iloc[index].filename,backend="soundfile")
        sig = sig / torch.max(torch.abs(sig))
        sig = sig + 1.5849e-05*(torch.rand(1, min_segment)-0.5) 
        melspec = mel_spectrogram(sig)
        melspec = torch.log(melspec)
        melspec = self.normalize_std(melspec)

        target = self.df.iloc[index].primary_label
        y = np.array([1 if item == target else 0 for item in class_labels])
        
        return melspec, y
    
    def __len__(self):
        return len(self.df)
        
temp_dataset = BirdclefDataset(train_meta, mode='train')
temp_loader = DataLoader(temp_dataset, batch_size=24, shuffle=False, num_workers=1,drop_last=True)
for X, y in temp_loader:
    break
    
plot_spectrogram(X[0,0,:,:])    
del temp_dataset, temp_loader, X, y


class Model_resnet34(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        model = models.resnet34(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_labels))
        self.model = model

    def forward(self, x):
        x = torch.cat((x,x,x),1)
        x = self.model(x)
        return x

model = Model_resnet34(pretrained=True)
model = model.to(device)


train_df, val_df = train_test_split(train_meta, test_size=0.2, random_state=42)

train_dataset = BirdclefDataset(train_df, mode='train')
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=2,drop_last=True)

val_dataset = BirdclefDataset(val_df, mode='val')
val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=1,drop_last=True)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    model.train()
    pred_train = []
    label_train = []
    running_loss = 0.0
    for melspecs, labels in tqdm(train_loader):
        melspecs, labels = melspecs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(melspecs)
        loss = criterion(outputs, labels.to(torch.float32))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pred_train.append(torch.softmax(outputs,dim=1).detach().cpu().numpy())
        label_train.append(labels.detach().cpu().numpy())

    pred_val = []
    label_val = []
    running_loss_val = 0.0
    model.eval()
    with torch.no_grad():
        for melspecs, labels in val_loader:
            melspecs, labels = melspecs.to(device), labels.to(device)
            outputs = model(melspecs)
            loss = criterion(outputs, labels.to(torch.float32))
            running_loss_val += loss.item()
            pred_val.append(torch.softmax(outputs,dim=1).detach().cpu().numpy())
            label_val.append(labels.detach().cpu().numpy())
    
    auc_train_val = cal_score(label_train, pred_train)
    auc_val = cal_score(label_val, pred_val)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Loss_val: {running_loss_val/len(train_loader):.4f}")
    print(f"Auc: {auc_train_val:.2f}% Auc_val: {auc_val:.2f}%")

torch.save(model.state_dict(), "baseline.pth")

