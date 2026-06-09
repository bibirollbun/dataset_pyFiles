# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/tensorflow-speech-recognition-challenge/train.7z'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import torch
import torchaudio
import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, TimeMasking, FrequencyMasking
from torchvision import transforms
from tqdm import tqdm
from scipy.io import wavfile
import torch.nn as nn
import torch.nn.functional as F
#from models.cnn import CNN


# torchaudio.set_audio_backend("sox_io")
torch.manual_seed(42)

DATA_DIR = "/kaggle/input/tensorflow-speech-recognition-challenge/train.7z/train"
N = 5000  # or however many you want total

commands = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']


class CNN(nn.Module):
    def __init__(self, num_classes=12):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input: (1, 128, 128)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: (32, 64, 64)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: (64, 64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: (64, 32, 32)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: (128, 32, 32)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # Output: (128, 16, 16)

        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (32, 64, 64)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # (64, 32, 32)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # (128, 16, 16)
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SpeechCommandsDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        self.label_to_index = {label: i for i, label in enumerate(commands)}
        self.mel_spec = MelSpectrogram(sample_rate=16000, n_mels=128)
        self.db_transform = AmplitudeToDB()
        self.freq_mask = FrequencyMasking(freq_mask_param=15)
        self.time_mask = TimeMasking(time_mask_param=35)

    def load_waveform(self, path):
        sample_rate, data = wavfile.read(path)
        data = data.astype('float32') / 32768.0
        waveform = torch.tensor(data).unsqueeze(0)
        return waveform, sample_rate

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        path = self.file_paths[index]
        label = self.labels[index]
        waveform, sample_rate = self.load_waveform(path)

        mel = self.mel_spec(waveform)
        mel = self.db_transform(mel)
        mel = self.freq_mask(mel)
        mel = self.time_mask(mel)
        mel = mel.squeeze(0).unsqueeze(0)

        if mel.size(-1) < 128:
            mel = torch.nn.functional.pad(mel, (0, 128 - mel.size(-1)))
        mel = mel[:, :, :128]

        label_idx = self.label_to_index[label]
        return mel, label_idx


all_files = []
all_labels = []

for label in commands:
    dir_path = os.path.join(DATA_DIR, label)
    if not os.path.exists(dir_path):
        continue
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.wav')]
    selected = random.sample(files, min(len(files), N // len(commands)))  # even dist
   
    all_files.extend(selected)
    all_labels.extend([label] * len(selected))

# Shuffle before splitting
combined = list(zip(all_files, all_labels))
random.shuffle(combined)
all_files, all_labels = zip(*combined)



# Train/val/test split
train_files, temp_files, train_labels, temp_labels = train_test_split(all_files, all_labels, test_size=0.3, random_state=42)
val_files, test_files, val_labels, test_labels = train_test_split(temp_files, temp_labels, test_size=0.5, random_state=42)




# Dataset and Dataloaders
train_dataset = SpeechCommandsDataset(train_files, train_labels)
val_dataset = SpeechCommandsDataset(val_files, val_labels)
test_dataset = SpeechCommandsDataset(test_files, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)



# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=len(commands)).to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



# Training loop
for epoch in tqdm(range(10)):
    model.train()
    total_loss, correct = 0, 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == y).sum().item()
    acc = correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Train Acc: {acc:.4f}")

