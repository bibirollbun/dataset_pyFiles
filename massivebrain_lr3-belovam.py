import torch.nn as nn
import pandas as pd
import numpy as np
import librosa
import random
import torch
import copy
import csv
import os

from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from torchvision.models import resnet50
from skimage.filters import gaussian
from skimage.transform import resize
from skimage import exposure, util
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)


LABELS = 24
SR = 48000
LENGTH = 10 * SR
F_MIN = 24000
F_MAX = 0
LEARNING_RATE = 2e-4
EPOCHS = 20
N_FOLD = 5


class AudioAugmentations:
    def __init__(self):
        self.augs = [self.add_noise, self.contrast_stretch, self.h_flip, self.v_flip]

    def h_flip(self, image):
        return np.stack([image[:, ::-1]] * 3)

    def v_flip(self, image):
        return np.stack([image[::-1, :]] * 3)

    def add_noise(self, image):
        noise_img = util.random_noise(image)
        return np.stack([noise_img] * 3)

    def contrast_stretch(self, image):
        contrast_img = exposure.rescale_intensity(image)
        return np.stack([contrast_img] * 3)

    def apply_random_augmentation(self, image):
        aug_func = random.choice(self.augs)
        return aug_func(image)


def spec_to_image(spec):
    spec = resize(spec, (224, 400))
    eps=1e-6

    mean = spec.mean()
    std = spec.std()

    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    spec_scaled = np.asarray(spec_scaled)

    return spec_scaled


def get_model():
    model = resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, LABELS)

    return model.to(device)


data = pd.read_csv("/kaggle/input/rfcx-species-audio-detection/train_tp.csv")

for i in range(0, len(data)):
    if F_MIN > float(data.iloc[i]['f_min']):
        F_MIN = float(data.iloc[i]['f_min'])
    if F_MAX < float(data.iloc[i]['f_max']):
        F_MAX = float(data.iloc[i]['f_max'])

F_MIN = int(F_MIN * 0.9)
F_MAX = int(F_MAX * 1.1)


label_list = data['species_id'].tolist()
data_list = data['recording_id'].tolist()
audio_data = {}

def process_audio(i):
    recording_id = data_list[i]
    species_id = label_list[i]

    wav, sr = librosa.load(f'/kaggle/input/rfcx-species-audio-detection/train/{recording_id}.flac', sr=None)

    t_min = int(data.at[i, 't_min'] * sr)
    t_max = int(data.at[i, 't_max'] * sr)

    center = np.round((t_min + t_max) / 2)
    beginning = max(center - LENGTH // 2, 0)
    ending = min(beginning + LENGTH, len(wav))

    beginning = ending - LENGTH if ending - beginning < LENGTH else beginning
    slice = wav[int(beginning):int(ending)]
    spec = librosa.feature.melspectrogram(y=slice, sr=sr, fmin=F_MIN, fmax=F_MAX)
    spec_db = librosa.power_to_db(spec, top_db=80)

    img = spec_to_image(spec_db)

    return recording_id, img


with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_audio, range(len(data))))

for recording_id, img in results:
    audio_data[recording_id] = img


class AudioData(Dataset):
    def __init__(self, X, y, data_type, augmentations=None):
        self.X = X
        self.y = y
        self.data_type = data_type
        self.audio_data = audio_data
        self.augmentations = augmentations

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        recording_id = self.X[idx]
        label = self.y[idx]

        img = self.audio_data[recording_id]

        if self.data_type == "train" and self.augmentations:
            img = self.augmentations.apply_random_augmentation(img)
        else:
            img = np.stack((img, img, img))

        return img, label


loss_fn = nn.CrossEntropyLoss()
audio_augmenter = AudioAugmentations()

def train(model, loss_fn, train_loader, valid_loader, optimizer, scheduler):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_losses = []
    valid_losses = []

    for epoch in tqdm(range(1, EPOCHS + 1)):
        model.train()
        batch_losses = []

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
        batch_losses = []
        trace_y = []
        trace_yhat = []

        with torch.no_grad():
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
        accuracy = np.mean(trace_yhat.argmax(axis=1) == trace_y)

        print("epoch = %d, train_loss = %.5f, val_loss = %.5f, val_accuracy = %.5f" % (
            epoch, np.mean(train_losses[-1]), np.mean(valid_losses[-1]), accuracy))

        scheduler.step(np.mean(valid_losses[-1]))
        if accuracy > best_acc:
            best_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)

    return model


skf = KFold(n_splits=N_FOLD, shuffle=True, random_state=563)

for fold_id, (train_index, val_index) in enumerate(skf.split(data_list, label_list)):
    print("Fold", fold_id)

    X_train = np.take(data_list, train_index)
    y_train = np.take(label_list, train_index, axis=0)
    X_val = np.take(data_list, val_index)
    y_val = np.take(label_list, val_index, axis=0)

    train_data = AudioData(X_train, y_train, "train", augmentations=audio_augmenter)
    valid_data = AudioData(X_val, y_val, "valid")

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=8, shuffle=True, drop_last=True)

    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    model = train(model, loss_fn, train_loader, valid_loader, optimizer, scheduler)
    torch.save(model.state_dict(), f"./model{fold_id}.pt")

    del train_data, valid_data, train_loader, valid_loader, model, X_train, X_val, y_train, y_val


def load_test_file(f):
    wav, sr = librosa.load('/kaggle/input/rfcx-species-audio-detection/test/' + f, sr=None)

    segments = len(wav) / LENGTH
    segments = int(np.ceil(segments))

    mel_array = []

    for i in range(0, segments):
        if (i + 1) * LENGTH > len(wav):
            slice = wav[len(wav) - LENGTH:len(wav)]
        else:
            slice = wav[i * LENGTH:(i + 1) * LENGTH]

        spec = librosa.feature.melspectrogram(y=slice, sr=sr, fmin=F_MIN, fmax=F_MAX)
        spec_db = librosa.power_to_db(spec, top_db=80)

        img = spec_to_image(spec_db)
        mel_spec = np.stack((img, img, img))
        mel_array.append(mel_spec)

    return mel_array


members = []

for i in range(N_FOLD):
    model = get_model()

    model.load_state_dict(torch.load('./model' + str(i) + '.pt'))
    model.eval()

    members.append(model)

    os.remove('./model' + str(i) + '.pt')


def load_and_predict(test_file, members):
    data = load_test_file(test_file)
    data = torch.tensor(data).float()

    if torch.cuda.is_available():
        data = data.cuda()

    output_list = []
    for m in members:
        output = m(data)
        maxed_output = torch.max(output, dim=0)[0]
        maxed_output = maxed_output.cpu().detach()
        output_list.append(maxed_output)

    avg_maxed_output = torch.mean(torch.stack(output_list), dim=0)
    file_id = test_file.split('.')[0]
    return [file_id] + [out.item() for out in avg_maxed_output]

def save_submission(predictions, output_file='submission.csv'):
    with open(output_file, 'w', newline='') as csvfile:
        submission_writer = csv.writer(csvfile, delimiter=',')
        submission_writer.writerow(['recording_id', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 
                                    's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23'])
        for pred in predictions:
            submission_writer.writerow(pred)

def generate_predictions(test_files, members):
    predictions = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(load_and_predict, test_file, members) for test_file in test_files]
        for future in futures:
            predictions.append(future.result())

    save_submission(predictions)


test_files = os.listdir('/kaggle/input/rfcx-species-audio-detection/test/')


if torch.cuda.is_available():
    members = [m.cuda() for m in members]

generate_predictions(test_files, members)

