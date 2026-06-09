!pip install --upgrade pip


!pip install timm==0.4.12


import os
import pandas as pd
import librosa
import random
import numpy as np
from tqdm import tqdm

from skimage.transform import resize
from PIL import Image

import torch
import torch.utils.data as torchdata
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import timm


import warnings
warnings.filterwarnings('ignore')


TRAIN_TP = '/kaggle/input/rfcx-species-audio-detection/train_tp.csv'
AUDIO_DATA = '/kaggle/input/rfcx-species-audio-detection/train/'
WORKING_DIR = '/kaggle/working/'


fft = 2048
hop = 512
sr = 48000
length = 10 * sr
save_to_disk = False

df = pd.read_csv(TRAIN_TP)

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
    
    mel_spec = librosa.feature.melspectrogram(y=slice, n_fft=fft, hop_length=hop, sr=sr, fmin=fmin, fmax=fmax, power=1.5)
    mel_spec = resize(mel_spec, (224, 400))
    
    mel_spec = mel_spec - np.min(mel_spec)
    mel_spec = mel_spec / np.max(mel_spec)

    mel_spec = mel_spec * 255
    mel_spec = np.round(mel_spec)    
    mel_spec = mel_spec.astype('uint8')
    mel_spec = np.asarray(mel_spec)
    
    bmp = Image.fromarray(mel_spec, 'L')
    bmp.save(f"{WORKING_DIR}{row['recording_id']}_{row['species_id']}_{int(center)}.bmp")


num_birds = 24
batch_size = 16

rng_seed = 1234
random.seed(rng_seed)
np.random.seed(rng_seed)
os.environ['PYTHONHASHSEED'] = str(rng_seed)
torch.manual_seed(rng_seed)
torch.cuda.manual_seed(rng_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class RainforestDataset(torchdata.Dataset):
    def __init__(self, filelist):
        self.specs = []
        self.labels = []
        for f in filelist:
            label = int(str.split(f, '_')[1])
            label_array = np.zeros(num_birds, dtype=np.single)
            label_array[label] = 1.
            self.labels.append(label_array)

            img = Image.open(WORKING_DIR + f)
            mel_spec = np.array(img)
            img.close()

            mel_spec = mel_spec / 255
            mel_spec = np.stack((mel_spec, mel_spec, mel_spec))
            
            self.specs.append(mel_spec)
    
    def __len__(self):
        return len(self.specs)
    
    def __getitem__(self, item):
        return self.specs[item], self.labels[item]


file_list = []
label_list = []

for f in os.listdir(WORKING_DIR):
    if '.bmp' in f:
        file_list.append(f)
        label = str.split(f, '_')[1]
        label_list.append(label)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng_seed)

train_files = []
val_files = []

for fold_id, (train_index, val_index) in enumerate(skf.split(file_list, label_list)):
    if fold_id == 0:
        train_files = np.take(file_list, train_index)
        val_files = np.take(file_list, val_index)


train_dataset = RainforestDataset(train_files)
val_dataset = RainforestDataset(val_files)

train_loader = torchdata.DataLoader(train_dataset, batch_size=batch_size, sampler=torchdata.RandomSampler(train_dataset))
val_loader = torchdata.DataLoader(val_dataset, batch_size=batch_size, sampler=torchdata.RandomSampler(val_dataset))

model = timm.create_model('resnest101e', pretrained=True)

model.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(1024, num_birds)
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.4)

pos_weights = torch.ones(num_birds)
pos_weights = pos_weights * num_birds
loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

if torch.cuda.is_available():
    model = model.cuda()
    loss_function = loss_function.cuda()


best_corrects = 0

for e in tqdm(range(0, 20), desc='Эпоха'):
    train_loss = []
    train_corr = []

    model.train()
    for batch, (data, target) in enumerate(train_loader):
        data = data.float()
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
            
        optimizer.zero_grad()
        
        output = model(data)
        loss = loss_function(output, target)
        
        loss.backward()
        optimizer.step()

        vals, answers = torch.max(output, 1)
        vals, targets = torch.max(target, 1)
        corrects = 0
        for i in range(0, len(answers)):
            if answers[i] == targets[i]:
                corrects = corrects + 1
        train_corr.append(corrects)
        
        train_loss.append(loss.item())

    for g in optimizer.param_groups:
        lr = g['lr']

    with torch.no_grad():
        # Stats
        val_loss = []
        val_corr = []
        
        model.eval()
        for batch, (data, target) in enumerate(val_loader):
            data = data.float()
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            
            output = model(data)
            loss = loss_function(output, target)

            vals, answers = torch.max(output, 1)
            vals, targets = torch.max(target, 1)
            corrects = 0
            for i in range(0, len(answers)):
                if answers[i] == targets[i]:
                    corrects = corrects + 1
            val_corr.append(corrects)
        
            val_loss.append(loss.item())
    '''
    tqdm.set_postfix({
         'train_loss': sum(train_loss) / len(train_loss), 
         'train_correct': f'{sum(train_corr)}/{len(train_dataset)}',
         'val_loss': str(sum(val_loss) / len(val_loss)), 
         'val_correct': str(sum(val_corr)) + '/' + str(val_dataset.__len__()),
     })
    '''
    if sum(val_corr) > best_corrects:
        print('Saving new best model at epoch ' + str(e) + ' (' + str(sum(val_corr)) + '/' + str(val_dataset.__len__()) + ')')
        torch.save(model, 'best_model.pt')
        best_corrects = sum(val_corr)

    scheduler.step()

del model


def load_test_file(f):
    wav, sr = librosa.load('/kaggle/input/rfcx-species-audio-detection/test/' + f, sr=None)

    segments = len(wav) / length
    segments = int(np.ceil(segments))
    
    mel_array = []
    
    for i in range(0, segments):
        if (i + 1) * length > len(wav):
            slice = wav[len(wav) - length:len(wav)]
        else:
            slice = wav[i * length:(i + 1) * length]

        mel_spec = librosa.feature.melspectrogram(y=slice, n_fft=fft, hop_length=hop, sr=sr, fmin=fmin, fmax=fmax, power=1.5)
        mel_spec = resize(mel_spec, (224, 400))
    
        mel_spec = mel_spec - np.min(mel_spec)
        mel_spec = mel_spec / np.max(mel_spec)
        
        mel_spec = np.stack((mel_spec, mel_spec, mel_spec))

        mel_array.append(mel_spec)
    
    return mel_array


model = timm.create_model('resnest101e', pretrained=True)

model.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(1024, num_birds)
)

model = torch.load(WORKING_DIR + 'best_model.pt', weights_only=False)
model.eval()

if not save_to_disk:
    for f in os.listdir(WORKING_DIR):
        os.remove(WORKING_DIR + f)

if torch.cuda.is_available():
    model.cuda()
    
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

columns = ['recording_id'] + [f's{i}' for i in range(24)]
df = pd.DataFrame(results, columns=columns)
df.to_csv('submission.csv', index=False)

