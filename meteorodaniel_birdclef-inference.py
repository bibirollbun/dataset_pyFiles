%matplotlib inline

import os
import random
import time
import math

import cv2
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, Subset

from tqdm.auto import tqdm
import timm


class cfg:

    debug = False
    
    output_dir = '/kaggle/working/'
    root = '/kaggle/input/birdclef-2025/'
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'

    model = '/kaggle/input/model-weights/veloso.pth'  
    
    SR = 32000
    TARGET_SHAPE = (256, 256)
    TARGET_DURATION = 5.0
    N_FFT = 1024
    HOP_LENGTH = 500
    N_MELS = 128
    FMIN = 40
    FMAX = 15000
    POWER = 2
    is_normalized = False
    
                        
    model_name = 'efficientnet_b0'  
    is_pre_trained = False #fix
    input_channels = 1
    
    optimizer = 'AdamW'
    lr = 5e-4 
    weight_decay = 1e-5
    epochs = 10  
    batch_size = 32  
    criterion = 'BCEWithLogitsLoss'
    n_folds = 5


    device = 'cpu'
    gpu_on = True
    if gpu_on:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

    def load_spectrograms(self):
        if self.is_files_loaded:
            loaded_specs = '/kaggle/input/birdclef25-mel-spectrograms/birdclef2025_melspec_5sec_256_256.npy'
            return np.load(loaded_specs, allow_pickle=True).item()
        
cfg = cfg()


def audio_to_melspec(data: np.ndarray, cfg: object) -> np.ndarray:
    mel_spec = librosa.feature.melspectrogram(
            y=data,
            sr=cfg.SR,
            n_fft=cfg.N_FFT,
            hop_length=cfg.HOP_LENGTH,
            n_mels=cfg.N_MELS,
            fmin=cfg.FMIN,
            fmax=cfg.FMAX,
            power=cfg.POWER
        )
    
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    if cfg.is_normalized:
        mel_spec_norm = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
        return mel_spec_norm
        
    return mel_spec

def process_audio_slice(audio: np.ndarray, cfg: object) -> np.ndarray:
    
    target_samples = int(cfg.TARGET_DURATION * cfg.SR)
    
    if len(audio) < target_samples:
        audio = np.pad(audio, 
                       (0, target_samples - len(audio)), 
                       mode='constant')
        
    mel_spec = audio_to_melspec(audio, cfg)                # cria o mel-espectrograma
    
    if mel_spec.shape != cfg.TARGET_SHAPE:                 # resize pro input da cnn
        mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
    
    return mel_spec


class CNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = len(pd.read_csv(cfg.taxonomy_csv)) #fix 
        
        # Backbone: Camadas iniciais de um modelo. Faz a extração das características
        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.is_pre_trained,          #fix   # Se false, reseta os pesos pré treinados.
            in_chans=cfg.input_channels,                   # Número de canais de entrada.
            drop_rate=0.0,
            drop_path_rate=0.0,
        )
        self.backbone = self.backbone.to(cfg.device)
        
        # Remove a camada final do backbone e passa a diante para conectar no meu nn.Linear
        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')

        # Head: Parte da rede neural responsável por fazer a predição.
        self.pooling = nn.AdaptiveAvgPool2d(1)         # Fixa o tamanho da feature map
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, self.num_classes).to(cfg.device)    # Classificação final

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, dict):
            features = features['features']
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size((0), -1))
        logits = self.classifier(features)
        return logits


def load_model(cfg):
    checkpoint = torch.load(cfg.model, map_location=torch.device(cfg.device), weights_only=True)
    model = CNN(cfg)
    model.load_state_dict(checkpoint)
    model = model.to(cfg.device)
    model.eval()
    return model


def predict_audio_slice(audio_slice, model, cfg):
    
    mel_spec = process_audio_slice(audio_slice, cfg)
    mel_spec = torch.tensor(mel_spec).unsqueeze(0).unsqueeze(0)
    mel_spec = mel_spec.to(cfg.device)

    predictions = []
    
    with torch.no_grad():
        outputs = model(mel_spec)
        final_preds = torch.sigmoid(outputs).cpu().numpy().squeeze()
        
    predictions.append(final_preds)
    return predictions


'''
from Stefan Kahl's Notebook
https://www.kaggle.com/code/stefankahl/birdclef-2025-sample-submission/notebook
'''

import os
import librosa
import numpy as np
import pandas as pd


# Set seed
np.random.seed(42)

checkpoint = torch.load(cfg.model, map_location=torch.device(cfg.device), weights_only=True)
model = CNN(cfg)
model.load_state_dict(checkpoint)
model = model.to(cfg.device)
model.eval()

# Class labels from train audio
class_labels = sorted(os.listdir('/kaggle/input/birdclef-2025/train_audio/'))


# List of test soundscapes (only visible during submission)

test_soundscape_path = '/kaggle/input/birdclef-2025/test_soundscapes/'
test_soundscapes = [os.path.join(test_soundscape_path, afile) for afile in sorted(os.listdir(test_soundscape_path)) if afile.endswith('.ogg')]


save = True

if cfg.debug:
    test_soundscape_path = '/kaggle/input/birdclef-2025/train_soundscapes'
    test_soundscapes = [os.path.join(test_soundscape_path, afile) for afile in sorted(os.listdir(test_soundscape_path)) if afile.endswith('.ogg')]
    save = False

# Open each soundscape and make predictions for 5-second segments
# Use pandas df with 'row_id' plus class labels as columns
predictions = pd.DataFrame(columns=['row_id'] + class_labels)
for idx, soundscape in enumerate(test_soundscapes):
    if cfg.debug:
        if idx == 5:
            break
    
    # Load audio
    sig, rate = librosa.load(path=soundscape, sr=None)

    # Split into 5-second chunks
    chunks = []
    for i in range(0, len(sig), rate*5):
        chunk = sig[i:i+rate*5]
        chunks.append(chunk)
        
    # Make predictions for each chunk
    for i, chunk in enumerate(chunks):
        # Get row id  (soundscape id + end time of 5s chunk)
        row_id = os.path.basename(soundscape).split('.')[0] + f'_{i * 5 + 5}'
        
        # Make prediction (let's use random scores for now)
        # scores = model.predict...
        scores = predict_audio_slice(chunk, model, cfg)[0]
        
        # Append to predictions as new row
        new_row = pd.DataFrame([[row_id] + list(scores)], columns=['row_id'] + class_labels)
        predictions = pd.concat([predictions, new_row], axis=0, ignore_index=True)
        
# Save prediction as csv
if save:
    predictions.to_csv('submission.csv', index=False)
predictions.head()




