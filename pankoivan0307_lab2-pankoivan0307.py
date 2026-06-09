import shutil
import os

shutil.copytree('../input/resnest50-fast-package/resnest-0.0.6b20200701/resnest', 'resnet', dirs_exist_ok=True)
os.system('pip install "./resnet" --no-deps')


import cv2
import torch

import numpy as np
import pandas as pd

import librosa as lb
import soundfile as sf

import re
import time

from pathlib import Path

from torch import nn
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from resnest.torch import resnest50


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Доступное устройство: {}'.format(device))


NUM_CLASSES = 397
SR = 32_000
DURATION = 5
THRESH = 0.25

TEST_AUDIO_ROOT = Path("../input/birdclef-2021/test_soundscapes")
SAMPLE_SUB_PATH = "../input/birdclef-2021/sample_submission.csv"
TARGET_PATH = None

if not len(list(TEST_AUDIO_ROOT.glob("*.ogg"))):
    TEST_AUDIO_ROOT = Path("../input/birdclef-2021/train_soundscapes")
    SAMPLE_SUB_PATH = None
    TARGET_PATH = Path("../input/birdclef-2021/train_soundscape_labels.csv")


class MelSpecComputer:

    def __init__(self, sr, n_mels, fmin, fmax, **kwargs):
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        kwargs["n_fft"] = kwargs.get("n_fft", self.sr//10)
        kwargs["hop_length"] = kwargs.get("hop_length", self.sr//(10*4))

        self.kwargs = kwargs

    def __call__(self, y):
        melspec = lb.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, **self.kwargs)
        melspec = lb.power_to_db(melspec).astype(np.float32)

        return melspec


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V


def crop_or_pad(y, length):
    if len(y) < length:
        y = np.concatenate([y, length - np.zeros(len(y))])
    elif len(y) > length:
        y = y[:length]
    return y


class BirdCLEFDataset(Dataset):

    def __init__(self, data, sr=SR, n_mels=128, fmin=0, fmax=None, duration=DURATION, step=None, res_type="kaiser_fast", resample=True):
        self.data = data

        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr//2

        self.duration = duration
        self.audio_length = self.duration*self.sr
        self.step = step or self.audio_length

        self.res_type = res_type
        self.resample = resample

        self.mel_spec_computer = MelSpecComputer(sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        image = np.stack([image, image, image])

        return image

    def audio_to_image(self, audio):
        melspec = self.mel_spec_computer(audio) 
        image = mono_to_color(melspec)
        image = self.normalize(image)

        return image

    def read_file(self, filepath):
        audio, orig_sr = sf.read(filepath, dtype="float32")

        if self.resample and orig_sr != self.sr:
            audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)

        audios = []
        for i in range(self.audio_length, len(audio) + self.step, self.step):
            start = max(0, i - self.audio_length)
            end = start + self.audio_length
            audios.append(audio[start:end])

        if len(audios[-1]) < self.audio_length:
            audios = audios[:-1]

        images = [self.audio_to_image(audio) for audio in audios]
        images = np.stack(images)

        return images

    def __getitem__(self, idx):
        return self.read_file(self.data.loc[idx, "filepath"])


data = pd.DataFrame(
     [(path.stem, *path.stem.split("_"), path) for path in Path(TEST_AUDIO_ROOT).glob("*.ogg")],
    columns = ["filename", "id", "site", "date", "filepath"]
)
data.head()


df_train = pd.read_csv("../input/birdclef-2021/train_metadata.csv")

LABEL_IDS = {label: label_id for label_id,label in enumerate(sorted(df_train["primary_label"].unique()))}
INV_LABEL_IDS = {val: key for key,val in LABEL_IDS.items()}

test_data = BirdCLEFDataset(data=data)
len(test_data), test_data[0].shape


def load_net(checkpoint_path, num_classes=NUM_CLASSES):
    net = resnest50(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, num_classes)

    dummy_device = torch.device("cpu")
    d = torch.load(checkpoint_path, map_location=dummy_device)

    for key in list(d.keys()):
        d[key.replace("model.", "")] = d.pop(key)

    net.load_state_dict(d)
    net = net.to(device)
    net = net.eval()

    return net


@torch.no_grad()
def get_thresh_preds(out, thresh=None):
    thresh = thresh or THRESH
    o = (-out).argsort(1)
    npreds = (out > thresh).sum(1)
    preds = []

    for oo, npred in zip(o, npreds):
        preds.append(oo[:npred].cpu().numpy().tolist())

    return preds

def get_bird_names(preds):
    bird_names = []

    for pred in preds:
        if not pred:
            bird_names.append("nocall")
        else:
            bird_names.append(" ".join([INV_LABEL_IDS[bird_id] for bird_id in pred]))

    return bird_names

def predict(nets, test_data, names=True):
    preds = []

    with torch.no_grad():
        for idx in  tqdm(list(range(len(test_data)))):
            xb = torch.from_numpy(test_data[idx]).to(device)
            pred = 0.

            for net in nets:
                o = net(xb)
                o = torch.sigmoid(o)
                pred += o

            pred /= len(nets)

            if names:
                pred = get_bird_names(get_thresh_preds(pred))

            preds.append(pred)

    return preds


checkpoint_paths = [Path("../input/kkiller-birdclef-models-public/birdclef_resnest50_fold0_epoch_10_f1_val_06471_20210417161101.pth")]

nets = [load_net(checkpoint_path.as_posix()) for checkpoint_path in checkpoint_paths]


pred_probas = predict(nets, test_data, names=False)
preds = [get_bird_names(get_thresh_preds(pred, thresh=THRESH)) for pred in pred_probas]

def preds_as_df(data, preds):
    sub = {
        "row_id": [],
        "birds": []
    }

    for row, pred in zip(data.itertuples(False), preds):
        row_id = [f"{row.id}_{row.site}_{5*i}" for i in range(1, len(pred)+1)]
        sub["birds"] += pred
        sub["row_id"] += row_id

    sub = pd.DataFrame(sub)

    if SAMPLE_SUB_PATH:
        sample_sub = pd.read_csv(SAMPLE_SUB_PATH, usecols=["row_id"])
        sub = sample_sub.merge(sub, on="row_id", how="left")
        sub["birds"] = sub["birds"].fillna("nocall")

    return sub

sub = preds_as_df(data, preds)
sub


sub.to_csv("submission.csv", index=False)

