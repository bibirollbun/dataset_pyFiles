import torch.nn as nn
import numpy as np
import torch
import librosa
import os
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage import exposure, util
import pandas as pd
import copy
from tqdm import tqdm
import random
import csv
import scipy
import warnings


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
sr = 48000
length = 10 * sr

data = pd.read_csv("../input/rfcx-species-audio-detection/train_tp.csv")
fmin = sr / 2
fmax = 0
for i in range(0, len(data)):
    if fmin > float(data.iloc[i]['f_min']):
        fmin = float(data.iloc[i]['f_min'])
    if fmax < float(data.iloc[i]['f_max']):
        fmax = float(data.iloc[i]['f_max'])
        
fmin = int(fmin * 0.9)
fmax = int(fmax * 1.1)

print(device)


def spec_to_image(spec):
    spec = resize(spec, (224, 400))
    spec_norm = (spec - spec.mean()) / (spec.std() + 1e-6)
    spec_scaled = 255 * (spec_norm - spec_norm.min()) / (spec_norm.max() - spec_norm.min())
    return np.asarray(spec_scaled.astype(np.uint8))

# фильтр низких частот, чтобы убрать насекомых
def load_and_filter_audio(file_path, sr=None, remove_insect_freq=10000):
    wav, sr = librosa.load(file_path, sr=sr)
    
    sos = scipy.signal.butter(4, remove_insect_freq, btype='lowpass', fs=sr, output='sos')
    wav = scipy.signal.sosfilt(sos, wav)
    
    return wav, sr


def convert_to_rgb(img):
    return np.stack((img, img, img))

def add_noise(img):
    return convert_to_rgb(util.random_noise(img))

def enhance_contrast(img):
    return convert_to_rgb(exposure.rescale_intensity(img))

def apply_gaussian_blur(img):
    return convert_to_rgb(gaussian(img))

def adjust_gamma_correction(img):
    return convert_to_rgb(exposure.adjust_gamma(img))

class RFCXDataset(Dataset):
    def __init__(self, X, y, is_train):
        self.data = []
        self.labels = []
        self.augs = [add_noise, enhance_contrast, apply_gaussian_blur, adjust_gamma_correction, convert_to_rgb, convert_to_rgb]
        self.is_train=is_train
        for i in range(0, len(X)):
            recording_id = X[i]
            label = y[i]
            mel_spec = audio_data[recording_id]
            self.data.append(mel_spec)
            self.labels.append(label)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_train:
            data = random.choice(self.augs)(self.data[idx])
        else:
            data = convert_to_rgb(self.data[idx])
        return data, self.labels[idx]


def load_and_process_data(data, fmin, fmax, length):
    """
    Загружает аудиоданные и создает мел-спектрограммы
    """
    label_list = []
    data_list = []
    audio_data = {}
    
    for i in range(len(data)):
        recording_id = data.recording_id.values[i]
        species_id = int(data.species_id.values[i])
        data_list.append(recording_id)
        label_list.append(species_id)

        # загрузка и фильтрация аудио
        wav, sr = load_and_filter_audio('../input/rfcx-species-audio-detection/train/' + recording_id + '.flac', sr=None)
        
        # вычисление временного сегмента
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
        
        # извлечение сегмента и создание спектрограммы
        slice = wav[int(beginning):int(ending)]
        spec = librosa.feature.melspectrogram(y=slice, sr=sr, fmin=fmin, fmax=fmax)
        spec_db = librosa.power_to_db(spec, top_db=80)
        audio_data[recording_id] = spec_to_image(spec_db)
    
    return data_list, label_list, audio_data


learning_rate = 1e-4
epochs = 14
loss_fn = nn.CrossEntropyLoss()

def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, scheduler):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_losses = []
    valid_losses = []
    
    for epoch in range(1,epochs+1):
        model.train()
        batch_losses=[]
        for _, data in enumerate(tqdm(train_loader)):
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
        
        print("epoch = %d, val_accuracy = %.5f" % (epoch, accuracy))

        scheduler.step(np.mean(valid_losses[-1]))
        if accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model


def initialize_model():
    model = torchvision.models.efficientnet_b2(weights=torchvision.models.EfficientNet_B2_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(num_features, 24)
    )
    model.to(device)
    return model


def train_with_cross_validation(data_list, label_list, fold_num=5, epochs=14, learning_rate=1e-4):
    """
    Выполняет кросс-валидацию и обучение моделей
    """
    skf = KFold(n_splits=fold_num, shuffle=True, random_state=32)
    
    for fold_id, (train_index, val_index) in enumerate(skf.split(data_list, label_list)):
        print(f"Training fold {fold_id + 1}/{fold_num}")
        
        # разделение данных
        X_train = np.take(data_list, train_index)
        y_train = np.take(label_list, train_index, axis=0)
        X_val = np.take(data_list, val_index)
        y_val = np.take(label_list, val_index, axis=0)

        # создание даталоадеров
        train_data = RFCXDataset(X_train, y_train, True)
        valid_data = RFCXDataset(X_val, y_val, False)
        train_loader = DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_data, batch_size=8, shuffle=True, drop_last=True)

        # инициализация модели
        model = initialize_model()
        
        # обучение
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        model = train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, scheduler)
        
        # сохранение модели
        torch.save(model.state_dict(), f"./efficientnet_{fold_id}.pt")
        
        # очистка памяти
        del train_data, valid_data, train_loader, valid_loader, model, X_train, X_val, y_train, y_val
        torch.cuda.empty_cache()


def create_ensemble_model(fold_num=5):
    """
    Создает ансамбль моделей из сохраненных весов
    """
    ensemble_members = []
    
    for i in range(fold_num):
        model = initialize_model()
        model.load_state_dict(torch.load(f'./efficientnet_{i}.pt'))
        model.eval()
        ensemble_members.append(model)
    
    return ensemble_members

def cleanup_model_files(fold_num=5):
    for i in range(fold_num):
        os.remove(f'./efficientnet_{i}.pt')


def load_test_file(f):
    wav, sr = load_and_filter_audio('../input/rfcx-species-audio-detection/test/' + f, sr=None)

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


warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
warnings.filterwarnings('ignore', category=UserWarning, message='PySoundFile failed*')
data_list, label_list, audio_data = load_and_process_data(data, fmin, fmax, length)


# обучение
train_with_cross_validation(data_list, label_list)


# создание ансамбля моделей
fold_num = 5
members = create_ensemble_model(fold_num)
cleanup_model_files(fold_num)


# создание файла submission
with open('submission.csv', 'w', newline='') as csvfile:
    submission_writer = csv.writer(csvfile, delimiter=',')
    submission_writer.writerow(['recording_id','s0','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11',
                               's12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22','s23'])
    
    test_files = os.listdir('../input/rfcx-species-audio-detection/test/')
    print(len(test_files))
    
    for i in range(0, len(test_files)):
        data_arrays = load_test_file(test_files[i])
        data = np.array(data_arrays)
        data = torch.from_numpy(data).float()
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

