!pip install efficientnet_pytorch -q

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import warnings
import librosa
import csv
import os
import pandas as pd

from skimage.transform import resize
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage import exposure, util
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold


warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


num_labels = 24
learning_rate = 2e-4
epochs = 20
n_folds = 5
sr = 48000
audio_length = 10 * sr


def add_channels(img):
    return np.stack((img, img, img))

def horizontal_flip(img):
    return add_channels(img[:, ::-1])

def vertical_flip(img):
    return add_channels(img[::-1, :])

def add_noise(img):
    return add_channels(util.random_noise(img))

def contrast_stretching(img):
    return add_channels(exposure.rescale_intensity(img))

def random_gaussian(img):
    return add_channels(gaussian(img))

def random_gamma(img):
    return add_channels(exposure.adjust_gamma(img))

def gray_scale(img):
    return add_channels(rgb2gray(img))


def spec_to_image(spec):
    spec = resize(spec, (224, 400))
    
    eps = 1e-6
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    
    spec_min = spec_norm.min()
    spec_max = spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    
    return spec_scaled


class AudioDataset(Dataset):
    def __init__(self, X, y, audio_data, is_train=True):
        self.data = []
        self.labels = []
        self.is_train = is_train
        
        self.augmentations = [
            add_noise, contrast_stretching, random_gaussian, 
            random_gamma, vertical_flip, horizontal_flip
        ]
        
        for i in range(len(X)):
            recording_id = X[i]
            label = y[i]
            self.data.append(audio_data[recording_id])
            self.labels.append(label)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        
        if self.is_train:
            aug = random.choice(self.augmentations)
            img = aug(img)
        else:
            img = add_channels(img)
        
        img_tensor = torch.FloatTensor(img)
        label = self.labels[idx]
        
        return img_tensor, label


def train_model(model, train_loader, valid_loader, epochs, optimizer, scheduler):
    best_acc = 0
    best_weights = None
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = []
        
        for x, y in train_loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()
        
        model.eval()
        val_loss = []
        all_y = []
        all_yhat = []
        
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.long)
                
                y_hat = model(x)
                loss = criterion(y_hat, y)
                val_loss.append(loss.item())
                
                all_y.append(y.cpu().numpy())
                all_yhat.append(y_hat.cpu().numpy())
        
        all_y = np.concatenate(all_y)
        all_yhat = np.concatenate(all_yhat)
        accuracy = np.mean(all_yhat.argmax(axis=1) == all_y)
        
        scheduler.step(np.mean(val_loss))
        
        if accuracy > best_acc:
            best_acc = accuracy
            best_weights = copy.deepcopy(model.state_dict())
        
        print(f"Epoch {epoch}: train_loss={np.mean(train_loss):.4f}, "
              f"val_loss={np.mean(val_loss):.4f}, accuracy={accuracy:.4f}")
    
    model.load_state_dict(best_weights)
    return model


def get_model():
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_labels)
    return model.to(device)


data = pd.read_csv("../input/rfcx-species-audio-detection/train_tp.csv")


fmin = int(data['f_min'].min() * 0.9)
fmax = int(data['f_max'].max() * 1.1)
print(f"Частотный диапазон: {fmin} - {fmax}")


audio_data = {}
data_list = []
label_list = []

for i in tqdm(range(len(data))):
    recording_id = data.recording_id.values[i]
    species_id = int(data.species_id.values[i])
    
    data_list.append(recording_id)
    label_list.append(species_id)
    
    wav, _ = librosa.load(f'../input/rfcx-species-audio-detection/train/{recording_id}.flac', sr=sr)
    
    t_min = data.t_min.values[i] * sr
    t_max = data.t_max.values[i] * sr
    center = (t_min + t_max) / 2
    start = max(0, center - audio_length / 2)
    end = min(len(wav), start + audio_length)
    
    if end == len(wav):
        start = max(0, end - audio_length)
    
    audio_slice = wav[int(start):int(end)]
    
    spec = librosa.feature.melspectrogram(y=audio_slice, sr=sr, fmin=fmin, fmax=fmax)
    spec_db = librosa.power_to_db(spec, top_db=80)
    
    img = spec_to_image(spec_db)
    audio_data[recording_id] = img


skf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(data_list, label_list)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    X_train = [data_list[i] for i in train_idx]
    y_train = [label_list[i] for i in train_idx]
    X_val = [data_list[i] for i in val_idx]
    y_val = [label_list[i] for i in val_idx]
    
    train_dataset = AudioDataset(X_train, y_train, audio_data, is_train=True)
    val_dataset = AudioDataset(X_val, y_val, audio_data, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    model = get_model()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    model = train_model(model, train_loader, val_loader, epochs, optimizer, scheduler)
    
    torch.save(model.state_dict(), f"./model{fold}.pt")
    print(f"Модель сохранена model{fold}.pt")
    
    del train_dataset, val_dataset, train_loader, val_loader, model
    torch.cuda.empty_cache()


def load_test_file(filename):
    wav, _ = librosa.load(f'../input/rfcx-species-audio-detection/test/{filename}', sr=sr)
    
    segments = []
    num_segments = int(np.ceil(len(wav) / audio_length))
    
    for i in range(num_segments):
        start = i * audio_length
        end = min((i + 1) * audio_length, len(wav))
        
        if end - start < audio_length:
            start = max(0, len(wav) - audio_length)
            end = len(wav)
        
        audio_slice = wav[int(start):int(end)]
        
        spec = librosa.feature.melspectrogram(y=audio_slice, sr=sr, fmin=fmin, fmax=fmax)
        spec_db = librosa.power_to_db(spec, top_db=80)
        
        img = spec_to_image(spec_db)
        img_3ch = add_channels(img)
        
        segments.append(img_3ch)
    
    return np.array(segments)


def create_submission():
    models = []
    for i in range(n_folds):
        model = get_model()
        model.load_state_dict(torch.load(f'./model{i}.pt'))
        model.eval()
        models.append(model)
    
    test_files = os.listdir('../input/rfcx-species-audio-detection/test/')
    
    with open('submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        
        header = ['recording_id'] + [f's{i}' for i in range(num_labels)]
        writer.writerow(header)
        
        for filename in tqdm(test_files):
            file_id = filename.split('.')[0]
            
            segments = load_test_file(filename)
            
            all_predictions = []
            
            for segment in segments:
                segment_tensor = torch.FloatTensor(segment).unsqueeze(0).to(device)
                
                model_outputs = []
                for model in models:
                    with torch.no_grad():
                        output = model(segment_tensor)
                        model_outputs.append(output.cpu())
                
                avg_output = torch.mean(torch.stack(model_outputs), dim=0)
                all_predictions.append(avg_output.numpy())
            
            final_prediction = np.mean(all_predictions, axis=0)[0]
            
            row = [file_id] + list(final_prediction)
            writer.writerow(row)


create_submission()

