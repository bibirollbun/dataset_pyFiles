import os
import logging
import random
import gc
import time
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import librosa

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from transformers import ASTForAudioClassification, Wav2Vec2FeatureExtractor, get_scheduler
from transformers import ASTFeatureExtractor
import math

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


class CFG:
    seed = 42
    debug = False
    print_freq = 100
    num_workers = 0
    
    OUTPUT_DIR = '/kaggle/working/'
    train_datadir = '/kaggle/input/birdclef-2025/train_audio'
    train_csv = '/kaggle/input/birdclef-2025/train.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    
    model_name = 'MIT/ast-finetuned-audioset-10-10-0.4593'
    pretrained = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 3
    batch_size = 16
    criterion = 'BCEWithLogitsLoss'
    
    optimizer = 'AdamW'
    lr = 3e-5
    weight_decay = 1e-5
    train_perc = 0.7


cfg = CFG()


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(cfg.seed)


feature_extractor = ASTFeatureExtractor.from_pretrained(cfg.model_name)
#feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.model_name)


taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
taxonomy_df.head()


taxonomy_df.info()


unique_primary_labels = taxonomy_df["primary_label"].unique()
species_ids = dict()
int_code = 0
for elt in unique_primary_labels:
    species_ids[elt] = int_code
    int_code +=1

# species_ids


train_df = pd.read_csv(cfg.train_csv)
train_df.info()


num_rows = train_df.shape[0]
num_rows_train = math.ceil(num_rows * cfg.train_perc)


train_df["primary_label"].unique()


class BirdCLEFDataset(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = data_path
        self.cfg = cfg
        self.target_length = 1024  # AST typically expects this length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = os.path.join(self.data_path, row['filename'])
        
        # Load audio and ensure it's the right length
        waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
        
        # Pad or truncate to target length
        if len(waveform) > self.target_length:
            waveform = waveform[:self.target_length]
        else:
            padding = self.target_length - len(waveform)
            waveform = np.pad(waveform, (0, padding), mode='constant')
        
        # Convert to tensor and add batch dimension
        waveform = torch.tensor(waveform).float()
        waveform = waveform.squeeze().numpy()
        
        label = species_ids[row['primary_label']]
        return {
            'input_values': waveform,
            'label': label
        }
        # return waveform, torch.tensor(label, dtype=torch.long)
    



def collate_fn(batch):
    audio_feats = [x['input_values'] for x in batch]
    labels = [x['label'] for x in batch]
    labels = torch.tensor(labels)
    out_aud_feats = feature_extractor(audio_feats,
                                      sampling_rate=16000,
                                      return_tensors='pt')['input_values']
    out_aud_feats = out_aud_feats.half()
    return out_aud_feats, labels

# def collate_fn(batch):
#     waveforms, labels = zip(*batch)
    
#     # Stack waveforms and add channel dimension (AST expects [batch_size, 1, sequence_length])
#     waveforms = torch.stack(waveforms).unsqueeze(1)
#     labels = torch.stack(labels)
    
#     return waveforms, labels


def get_model(num_classes):
    model = ASTForAudioClassification.from_pretrained(cfg.model_name, num_labels=num_classes,
                                                      ignore_mismatched_sizes=True,
                                                     torch_dtype=torch.float16)
    return model.to(cfg.device)


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader):
        # print("Input shape:", batch[0].shape)  # Devrait être (batch_size, 1, time, frequency)
        # break
        inputs, labels = batch
        inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
        
        optimizer.zero_grad()
        outputs = model(inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        break
    
    return total_loss / len(train_loader)


def validate(model, loader, criterion):
    val_loss, val_correct, val_total = 0, 0, 0
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            inputs, labels = batch
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            break
    # Log validation metrics
    val_acc = (val_correct / val_total) * 100
    print(f"Validation Loss: {val_loss / len(loader):.4f}, Validation Accuracy: {val_acc:.2f}%")


def train():
    df = pd.read_csv(cfg.train_csv)
    df = df.sample(frac=1.).reset_index(drop=True) # we shuffle the whole table
    df_train = df.head(num_rows_train)
    num_rows_val = num_rows - num_rows_train
    df_val = df.tail(num_rows_val)
    df_val = df_val.reset_index(drop=True)
    
    dataset_train = BirdCLEFDataset(df_train, cfg.train_datadir)
    dataset_val = BirdCLEFDataset(df_val, cfg.train_datadir)
    train_loader = DataLoader(dataset_train,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              #num_workers=cfg.num_workers,
                              collate_fn=collate_fn)
    val_loader = DataLoader(dataset_val,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              #num_workers=cfg.num_workers,
                              collate_fn=collate_fn)

    
    
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    species_ids = taxonomy_df['primary_label'].tolist()
    # self.num_classes = len(self.species_ids)
    
    model = get_model(num_classes=len(species_ids))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=len(train_loader) * cfg.epochs)
    
    for epoch in range(cfg.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion)
        print(f"Epoch {epoch+1}/{cfg.epochs}, Loss: {loss:.4f}")
        validate(model, val_loader, criterion)
    
    # torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "birdclef_ast.pth"))
        model.save_pretrained(f"fine_tuned_ast_epoch_{epoch}")
    print("Model saved!")





if __name__ == "__main__":
    train()





