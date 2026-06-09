import shutil
shutil.copytree('/kaggle/input/resnest50-fast-package/resnest-0.0.6b20200701/resnest', 'resnet', dirs_exist_ok=True) 
!pip install "./resnet" --no-deps


import json
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from resnest.torch import resnest50
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import soundfile as sf
import librosa as lb
from torch import nn


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Доступное устройство: {}'.format(device))


from pathlib import Path

class AudioAnalysisConfig:
    """Конфигурация для анализа аудиоданных"""
    
    def __init__(self):
        self.num_classes = 397
        self.sample_rate = 32000
        self.segment_duration = 5
        self.threshold = 0.25
        
        # Определение путей к данным
        self._setup_paths()
    
    def _setup_paths(self):
        """Настройка путей к аудиофайлам и меткам"""
        test_root = Path("../input/birdclef-2021/test_soundscapes")
        train_root = Path("../input/birdclef-2021/train_soundscapes")
        
        if self._has_audio_files(test_root):
            self.audio_directory = test_root
            self.submission_template = "../input/birdclef-2021/sample_submission.csv"
            self.ground_truth = None
        else:
            self.audio_directory = train_root
            self.submission_template = None
            self.ground_truth = Path("../input/birdclef-2021/train_soundscape_labels.csv")
    
    @staticmethod
    def _has_audio_files(directory, extension="*.ogg"):
        """Проверка наличия аудиофайлов в директории"""
        return len(list(directory.glob(extension))) > 0

# Использование
config = AudioAnalysisConfig()


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


def convert_spectrogram_to_rgb(spectrogram, epsilon=1e-6, mean_val=None, std_val=None):
    """
    Преобразует монохромный спектрограммный массив в цветное RGB-подобное представление.
    
    Параметры:
    -----------
    spectrogram : np.ndarray
        Входной монохромный спектрограммный массив
    epsilon : float, optional
        Малое значение для избежания деления на ноль (по умолчанию 1e-6)
    mean_val : float, optional
        Предварительно вычисленное среднее значение для нормализации
    std_val : float, optional
        Предварительно вычисленное стандартное отклонение для нормализации
    
    Возвращает:
    -----------
    np.ndarray
        Массив в формате uint8 (0-255), готовый для визуализации
    """
    
    # Нормализация данных
    normalization_mean = mean_val if mean_val is not None else spectrogram.mean()
    normalization_std = std_val if std_val is not None else spectrogram.std()
    
    normalized_data = (spectrogram - normalization_mean) / (normalization_std + epsilon)
    
    # Определение диапазона значений
    data_minimum = normalized_data.min()
    data_maximum = normalized_data.max()
    value_range = data_maximum - data_minimum
    
    # Преобразование в диапазон 0-255
    if value_range > epsilon:
        # Масштабирование с ограничением
        scaled_values = np.clip(normalized_data, data_minimum, data_maximum)
        rgb_values = 255 * (scaled_values - data_minimum) / value_range
        rgb_values = rgb_values.astype(np.uint8)
    else:
        # Для плоских спектрограмм возвращаем нулевой массив
        rgb_values = np.zeros_like(normalized_data, dtype=np.uint8)
    
    return rgb_values


def adjust_audio_length(audio_signal, target_length):
    """
    Обрезает или дополняет аудиосигнал до заданной длины.
    
    Параметры:
    ----------
    audio_signal : np.ndarray
        Входной аудиосигнал
    target_length : int
        Желаемая длина сигнала
    
    Возвращает:
    ----------
    np.ndarray
        Сигнал заданной длины
    """
    current_length = len(audio_signal)
    
    if current_length < target_length:
        # Дополнение нулями
        padding_length = target_length - current_length
        padding = np.zeros(padding_length, dtype=audio_signal.dtype)
        result = np.concatenate([audio_signal, padding])
    elif current_length > target_length:
        # Обрезка до нужной длины
        result = audio_signal[:target_length]
    else:
        # Длина уже правильная
        result = audio_signal
    
    return result


class BirdCLEFDataset(Dataset):

    def __init__(self, data, sr=config.sample_rate, n_mels=128, fmin=0, fmax=None, duration=config.segment_duration, step=None, res_type="kaiser_fast", resample=True):
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
        image = convert_spectrogram_to_rgb(melspec)
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


df = pd.DataFrame(
     [(path.stem, *path.stem.split("_"), path) for path in config.audio_directory.glob("*.ogg")],
    columns = ["filename", "id", "site", "date", "filepath"]
)
df.head()


df_train = pd.read_csv("../input/birdclef-2021/train_metadata.csv")

LABEL_IDS = {label: label_id for label_id,label in enumerate(sorted(df_train["primary_label"].unique()))}
INV_LABEL_IDS = {val: key for key,val in LABEL_IDS.items()}

test_data = BirdCLEFDataset(data=df)
len(test_data), test_data[0].shape


def load_net(checkpoint_path, num_classes=config.num_classes):
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
preds = [get_bird_names(get_thresh_preds(pred, thresh=config.threshold)) for pred in pred_probas]

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

    if config.submission_template:
        sample_sub = pd.read_csv(config.submission_template, usecols=["row_id"])
        sub = sample_sub.merge(sub, on="row_id", how="left")
        sub["birds"] = sub["birds"].fillna("nocall")

    return sub

sub = preds_as_df(df, preds)


sub.to_csv("submission.csv", index=False)

