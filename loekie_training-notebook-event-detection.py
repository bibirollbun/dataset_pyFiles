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

from sklearn.metrics import roc_auc_score
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as AT
from torchvision import models

from metric import score

import csv
import os


input_path_bird_clef  = "/kaggle/input/birdclef-2025-train-data/train_raw5/"

train_meta_bird_clef_original = pd.read_csv( "../input/birdclef-2025/train.csv")

train_meta_bird_clef = pd.DataFrame()
train_meta_bird_clef['id'] = pd.DataFrame(train_meta_bird_clef_original['filename'])
train_meta_bird_clef['id'] = input_path_bird_clef + train_meta_bird_clef['id'].astype(str) 
train_meta_bird_clef['label'] = 1


input_path_detection = '/kaggle/input/data-preparation-event-detection/train_raw5/'
train_meta_detection = pd.read_csv('/kaggle/input/expanded-labels/metadata_expanded.csv')
train_meta_detection['id'] = input_path_detection + train_meta_detection['id'].astype(str) + ".wav" 
train_meta_detection = train_meta_detection[train_meta_detection['label'] == 0]


train_meta = pd.concat([train_meta_detection, train_meta_bird_clef])
train_meta.reset_index().drop(columns= ['index'])


wav_sec = 5
sample_rate = 32000
min_segment = sample_rate*wav_sec

epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mel_configs = [
    {"n_fft": 1280, "hop_length": 512,  "n_mels": 64, "f_max": 16000}, ]


# Adapted cal score for a binary classification problem
def cal_score(label, pred):
    label = np.concatenate(label)       
    pred = np.concatenate(pred)          

    label = label.reshape(-1).astype(int)
    pred = pred.reshape(-1, 2)

    pred_class1_probs = pred[:, 1]

    return roc_auc_score(label, pred_class1_probs)


def log_metrics_to_csv(csv_path, name, epoch, auc, auc_val, loss, loss_val):
    # Create the file with header if it doesn't exist
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(['name', 'epoch', 'auc', 'auc_val', 'loss', 'loss_val'])
        
        writer.writerow([name, epoch, auc, auc_val, loss, loss_val])


class BirdclefDataset(Dataset):
    def __init__(self, df, mel_transform, mode='train'):
        self.df = df
        self.mode = mode
        self.mel_transform = mel_transform

    def normalize_std(self, spec, eps=1e-23):
        mean = torch.mean(spec)
        std = torch.std(spec)
        return torch.where(std == 0, spec-mean, (spec - mean) / (std+eps))
                
    def __getitem__(self, index):
        try:
            sig, _ = torchaudio.load(uri=self.df.iloc[index].id,backend="soundfile")
        except:
            sig, _ = torchaudio.load(uri=self.df.iloc[index].id,backend="ffmpeg")
        sig = sig / torch.max(torch.abs(sig))
        sig = sig + 1.5849e-05*(torch.rand(1, min_segment)-0.5) 
        melspec = self.mel_transform(sig)
        melspec = torch.log(melspec + 1e-6)
        melspec = self.normalize_std(melspec)

        y = self.df.iloc[index].label
        
        return melspec, y
    
    def __len__(self):
        return len(self.df)


class Model_resnet34(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        model = models.resnet34(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        self.model = model

    def forward(self, x):
        x = torch.cat((x,x,x),1)
        x = self.model(x)
        return x


train_df, val_df = train_test_split(train_meta, test_size=0.2, random_state=42)
print(train_df.iloc[0].id)
for i, cfg in enumerate(mel_configs):
    print(f"\nTraining model {i+1}/{len(mel_configs)} with config: {cfg}")

    mel_transform = AT.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=cfg["n_fft"],
        win_length=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        center=True,
        f_min=20,
        f_max=cfg["f_max"],
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        n_mels=cfg["n_mels"],
        mel_scale="htk",
    )

    # Datasets
    train_dataset = BirdclefDataset(train_df, mel_transform, mode='train')
    val_dataset = BirdclefDataset(val_df, mel_transform, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=1, drop_last=True)
   
    # Model and optimizer
    model = Model_resnet34(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        pred_train = []
        label_train = []
        running_loss = 0.0
        for melspecs, labels in tqdm(train_loader):
            melspecs, labels = melspecs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(melspecs)
            loss = criterion(outputs, labels)
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
                loss = criterion(outputs, labels)
                running_loss_val += loss.item()
                pred_val.append(torch.softmax(outputs,dim=1).detach().cpu().numpy())
                label_val.append(labels.detach().cpu().numpy())

        csv_path = '/kaggle/working/results.csv'
       
        auc_train_val = cal_score(label_train, pred_train)
        auc_val = cal_score(label_val, pred_val)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Loss_val: {running_loss_val/len(train_loader):.4f}")
        print(f"Auc: {auc_train_val:.2f}% Auc_val: {auc_val:.2f}%")
    
        # Log metrics
        log_metrics_to_csv(
            csv_path=csv_path,
            name=f"cfg_{i}_fft{cfg['n_fft']}_hop{cfg['hop_length']}_mels{cfg['n_mels']}",
            epoch=epoch + 1,
            auc=auc_train_val,
            auc_val=auc_val,
            loss=running_loss / len(train_loader),
            loss_val=running_loss_val / len(val_loader)
        )

    # Save model 
    torch.save(model.state_dict(), f"model_cfg_{i}.pth")





