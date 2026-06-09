import os
import numpy as np
import librosa
from tqdm import tqdm
from skimage.transform import resize
from PIL import Image
import pandas as pd
import warnings
import random
import torch
import torch.utils.data as torchdata
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


fft = 2048
hop = 512
sr = 48000
length = 10 * sr
WORKING_DIR = '/kaggle/working/'
AUDIO_DATA = '/kaggle/input/rfcx-species-audio-detection/train/'
TRAIN_TP = '/kaggle/input/rfcx-species-audio-detection/train_tp.csv'
df = pd.read_csv(TRAIN_TP)


# Определяем диапазон частот
fmin = int(df['f_min'].min() * 0.9)
fmax = int(df['f_max'].max() * 1.1)

for idx, row in tqdm(df.iterrows(), total=len(df), desc='Получение спектрограмм'):
    wav, sr = librosa.load(f"{AUDIO_DATA}{row['recording_id']}.flac", sr=None)
    
    t_min = float(row['t_min']) * sr
    t_max = float(row['t_max']) * sr
    
    center = np.round((t_min + t_max) / 2)
    beginning = center - length / 2
    if beginning < 0:
        beginning = 0
    
    ending = beginning + length
    if ending > len(wav):
        ending = len(wav)
        beginning = ending - length
        
    slice = wav[int(beginning):int(ending)]
    
    mel_spec = librosa.feature.melspectrogram(
        y=slice, n_fft=fft, hop_length=hop, sr=sr, fmin=fmin, fmax=fmax, power=1.5
    )
    mel_spec = resize(mel_spec, (224, 400))
    
    mel_spec = mel_spec - np.min(mel_spec)
    mel_spec = mel_spec / np.max(mel_spec)
    mel_spec = (mel_spec*255).astype('uint8')
    
    bmp = Image.fromarray(mel_spec, 'L')
    bmp.save(f"{WORKING_DIR}{row['recording_id']}_{row['species_id']}_{int(center)}.bmp")


class RainforestDataset(Dataset):
    def __init__(self, file_list, working_dir=WORKING_DIR, num_classes=24):
        self.specs = []
        self.labels = []
        self.num_classes = num_classes

        for f in file_list:
            # Метка
            label = int(f.split('_')[1])
            label_array = np.zeros(self.num_classes, dtype=np.float32)
            label_array[label] = 1.
            self.labels.append(label_array)

            # Спектрограмма
            img = Image.open(os.path.join(working_dir, f))
            mel_spec = np.array(img).astype(np.float32) / 255.
            img.close()

            # 3 канала
            mel_spec = np.stack([mel_spec, mel_spec, mel_spec], axis=0)
            self.specs.append(mel_spec)

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        return torch.tensor(self.specs[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)


file_list = [f for f in os.listdir(WORKING_DIR) if f.endswith('.bmp')]
label_list = [int(f.split('_')[1]) for f in file_list]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
for fold_id, (train_idx, val_idx) in enumerate(skf.split(file_list, label_list)):
    if fold_id == 0:
        train_files = np.take(file_list, train_idx)
        val_files = np.take(file_list, val_idx)


batch_size = 16

train_dataset = RainforestDataset(train_files)
val_dataset = RainforestDataset(val_files)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


num_birds = 24
model = timm.create_model('resnest101e', pretrained=True)

model.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, num_birds)
)

if torch.cuda.is_available():
    model = model.cuda()


optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.4)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.ones(num_birds) * num_birds)
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

best_corrects = 0

for epoch in range(20):
    # TRAIN
    model.train()
    train_loss, train_corrects = [], []
    for data, target in train_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        preds = output.argmax(dim=1)
        targets = target.argmax(dim=1)
        train_corrects.append((preds == targets).sum().item())
        train_loss.append(loss.item())

    # VALID
    model.eval()
    val_loss, val_corrects = [], []
    with torch.no_grad():
        for data, target in val_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = loss_fn(output, target)
            val_loss.append(loss.item())

            preds = output.argmax(dim=1)
            targets = target.argmax(dim=1)
            val_corrects.append((preds == targets).sum().item())

    print(f"Epoch {epoch}: Train Loss={np.mean(train_loss):.4f}, Train Acc={sum(train_corrects)/len(train_dataset):.4f}, "
          f"Val Loss={np.mean(val_loss):.4f}, Val Acc={sum(val_corrects)/len(val_dataset):.4f}")

    if sum(val_corrects) > best_corrects:
        best_corrects = sum(val_corrects)
        WORKING_DIR = '/kaggle/working/'
        os.makedirs(WORKING_DIR, exist_ok=True)  # на всякий случай

        torch.save(model, os.path.join(WORKING_DIR, 'best_model.pt'))

    scheduler.step()


import torch
import timm
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.transform import resize
import librosa
import os

# Функция для загрузки тестового файла и превращения в спектрограммы
def load_test_file(f):
    wav, sr = librosa.load('/kaggle/input/rfcx-species-audio-detection/test/' + f, sr=None)
    segments = int(np.ceil(len(wav) / length))
    mel_array = []

    for i in range(segments):
        if (i + 1) * length > len(wav):
            slice = wav[len(wav) - length:len(wav)]
        else:
            slice = wav[i * length:(i + 1) * length]

        mel_spec = librosa.feature.melspectrogram(
            y=slice, n_fft=fft, hop_length=hop, sr=sr, fmin=fmin, fmax=fmax, power=1.5
        )
        mel_spec = resize(mel_spec, (224, 400))
        mel_spec = mel_spec - np.min(mel_spec)
        mel_spec = mel_spec / np.max(mel_spec)
        mel_spec = np.stack((mel_spec, mel_spec, mel_spec))
        mel_array.append(mel_spec)

    return mel_array

# Загружаем модель
model = timm.create_model('resnest101e', pretrained=True)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(2048, 1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(1024, 1024),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(1024, num_birds)
)

model = torch.load(WORKING_DIR + 'best_model.pt', weights_only=False)
model.eval()

if torch.cuda.is_available():
    model.cuda()

import shutil

for f in os.listdir(WORKING_DIR):
    if f == 'best_model.pt':   # пропускаем модель
        continue
    path = os.path.join(WORKING_DIR, f)
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)

# Генерация предсказаний
results = []
test_files = os.listdir('/kaggle/input/rfcx-species-audio-detection/test/')

for file_name in tqdm(test_files, desc='Processing test files'):
    data = load_test_file(file_name)
    data = torch.tensor(data).float()
    if torch.cuda.is_available():
        data = data.cuda()

    output = model(data)
    maxed_output = torch.max(output, dim=0)[0].cpu().detach().numpy()

    file_id = file_name.split('.')[0]
    row = [file_id] + maxed_output.tolist()
    results.append(row)

# Сохраняем в CSV
columns = ['recording_id'] + [f's{i}' for i in range(num_birds)]
submission_df = pd.DataFrame(results, columns=columns)
submission_df.to_csv('submission.csv', index=False)
print("submission.csv создан!")

