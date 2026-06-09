!pip install efficientnet_pytorch


import torch
import torch.nn as nn
import numpy as np
import random
import copy
import warnings
import torch
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


class AudioData(Dataset):
    def __init__(self, X, y, data_type, audio_data, fmin, fmax, length):
        self.data = []
        self.labels = []
        self.augs = [addNoisy, contrast_stretching, randomGaussian, randomGamma, vertical_flip, horizontal_flip, addChannels]
        self.data_type = data_type
        self.audio_data = audio_data
        self.fmin = fmin
        self.fmax = fmax
        self.length = length

        for i in range(0, len(X)):
            recording_id = X[i]
            label = y[i]
            mel_spec = self.audio_data[recording_id]
            self.data.append(mel_spec)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.data_type == "train":
            aug = random.choice(self.augs)
            data = aug(self.data[idx])
        else:
            data = addChannels(self.data[idx])
        return data, self.labels[idx]

def horizontal_flip(img):
    horizontal_flip_img = img[:, ::-1]
    return addChannels(horizontal_flip_img)

def vertical_flip(img):
    vertical_flip_img = img[::-1, :]
    return addChannels(vertical_flip_img)

def addNoisy(img):
    noise_img = util.random_noise(img)
    return addChannels(noise_img)

def contrast_stretching(img):
    contrast_img = exposure.rescale_intensity(img)
    return addChannels(contrast_img)

def randomGaussian(img):
    gaussian_img = gaussian(img)
    return addChannels(gaussian_img)

def grayScale(img):
    gray_img = rgb2gray(img)
    return addChannels(gray_img)

def randomGamma(img):
    img_gamma = exposure.adjust_gamma(img)
    return addChannels(img_gamma)

def addChannels(img):
    return np.stack((img, img, img))

def spec_to_image(spec):
    spec = resize(spec, (224, 400))
    eps = 1e-6
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    spec_scaled = np.asarray(spec_scaled)
    return spec_scaled

def get_model(num_labels):
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_labels)
    model = model.to(device)
    return model


def generate_submission():
    members = []
    for i in range(1, nfold):
        member_model = get_model(num_labels)
        member_model.load_state_dict(torch.load('./model'+str(i)+'.pt'))
        member_model.eval()
        members.append(member_model)

    with open('submission.csv', 'w', newline='') as csvfile:
        submission_writer = csv.writer(csvfile, delimiter=',')
        submission_writer.writerow(['recording_id','s0','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11',
                                   's12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22','s23'])

        test_files = os.listdir('../input/rfcx-species-audio-detection/test/')

        for i in tqdm(list(range(0, len(test_files)))):
            data = load_test_file(test_files[i])
            data = torch.tensor(data)
            data = data.float()
            if torch.cuda.is_available():
                data = data.cuda()

            output_list = []
            for m in members:
                output = m(data)
                maxed_output = torch.max(output, dim=0)[0]
                maxed_output = maxed_output.cpu().detach()
                output_list.append(maxed_output)
            avg_maxed_output = torch.mean(torch.stack(output_list), dim=0)

            file_id = str.split(test_files[i], '.')[0]
            write_array = [file_id]

            for out in avg_maxed_output:
                write_array.append(out.item())

            submission_writer.writerow(write_array)



num_labels = 24
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
warnings.filterwarnings('ignore')
learning_rate = 2e-4
epochs = 20
loss_fn = nn.CrossEntropyLoss()
nfold = 5
sr = 48000
length = 10 * sr
fmin = 24000
fmax = 0


data = pd.read_csv("../input/rfcx-species-audio-detection/train_tp.csv")

fmin = 24000
fmax = 0
for i in range(0, len(data)):
    if fmin > float(data.iloc[i]['f_min']):
        fmin = float(data.iloc[i]['f_min'])
    if fmax < float(data.iloc[i]['f_max']):
        fmax = float(data.iloc[i]['f_max'])
        
fmin = int(fmin * 0.9)
fmax = int(fmax * 1.1)

label_list = []
data_list = []
audio_data = {}
for i in tqdm(list(range(0, len(data)))):
    recording_id = data.recording_id.values[i]
    species_id = int(data.species_id.values[i])
    data_list.append(recording_id)
    label_list.append(species_id)

    wav, sr = librosa.load('../input/rfcx-species-audio-detection/train/' + recording_id + '.flac', sr=None)
    t_min = float(data.t_min.values[i]) * sr
    t_max = float(data.t_max.values[i]) * sr
    center = np.round((t_min + t_max) / 2)
    beginning = center - length / 2
    if beginning < 0:
        beginning = 0
    ending = beginning + length
    if ending > len(wav):
        ending = len(wav)
        beginning = ending - length
    slice = wav[int(beginning):int(ending)]
    
    spec=librosa.feature.melspectrogram(y=slice, sr=sr, fmin=fmin, fmax=fmax)
    spec_db=librosa.power_to_db(spec, top_db=80)
    
    img = spec_to_image(spec_db)
    
    audio_data[recording_id] = img


def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, scheduler):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_losses = []
    valid_losses = []

    postfix_label = ""
    
    for epoch in tqdm(range(1,epochs+1), postfix=postfix_label):
        model.train()
        batch_losses=[]
        for _, data in enumerate(train_loader):
            x, y = data
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            batch_losses.append(loss.item())
            optimizer.step()
        train_losses.append(batch_losses)

        model.eval()
        batch_losses=[]
        trace_y = []
        trace_yhat = []
        
        for _, data in enumerate(valid_loader):
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy())      
            batch_losses.append(loss.item())
        valid_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)
        accuracy = np.mean(trace_yhat.argmax(axis=1)==trace_y)

        scheduler.step(np.mean(valid_losses[-1]))
        if accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

        postfix_label = "epoch = %d, train_loss = %.5f, val_loss = %.5f, val_accuracy = %.5f" % (epoch, np.mean(train_losses[-1]), np.mean(valid_losses[-1]), accuracy)

    model.load_state_dict(best_model_wts)
    return model


skf = KFold(n_splits=nfold, shuffle=True, random_state=32)

for fold_id, (train_index, val_index) in tqdm(list(enumerate(skf.split(data_list, label_list)))):
    X_train = np.take(data_list, train_index)
    y_train = np.take(label_list, train_index, axis = 0)
    X_val = np.take(data_list, val_index)
    y_val = np.take(label_list, val_index, axis = 0)

    train_data = AudioData(X_train, y_train, "train", audio_data, fmin, fmax, length)
    valid_data = AudioData(X_val, y_val, "valid", audio_data, fmin, fmax, length)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=8, shuffle=True, drop_last=True)

    model = get_model(num_labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    model = train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, scheduler)
    torch.save(model.state_dict(), "./model" + str(fold_id) + ".pt")
    
    del train_data, valid_data, train_loader, valid_loader, model, X_train, X_val, y_train, y_val


def load_test_file(f):
    wav, sr = librosa.load('../input/rfcx-species-audio-detection/test/' + f, sr=None)

    segments = len(wav) / length
    segments = int(np.ceil(segments))
    
    mel_array = []
    
    for i in range(0, segments):
        if (i + 1) * length > len(wav):
            slice = wav[len(wav) - length:len(wav)]
        else:
            slice = wav[i * length:(i + 1) * length]
        
        spec=librosa.feature.melspectrogram(y=slice, sr=sr, fmin=fmin, fmax=fmax)
        spec_db=librosa.power_to_db(spec,top_db=80)

        img = spec_to_image(spec_db)
        mel_spec = np.stack((img, img, img))
        mel_array.append(mel_spec)
    
    return mel_array


generate_submission()

