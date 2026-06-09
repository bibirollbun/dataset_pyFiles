# Установка пользовательского пакета
import shutil
import os
shutil.copytree('../input/resnest50-fast-package/resnest-0.0.6b20200701/resnest', 'resnet', dirs_exist_ok=True)
os.system('pip install "./resnet" --no-deps')


# Imports
import pandas as pd
import numpy as np
import librosa as lb
import soundfile as sf
import cv2
from pathlib import Path
import re
import torch
from torch import nn
from  torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import time
from resnest.torch import resnest50
import matplotlib.pyplot as plt
import IPython.display as ipd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Переменные для вычислений
NUM_CLASSES = 397
SR = 32_000
DURATION = 5
THRESH = 0.28


# Пути к файлам
TEST_AUDIO_ROOT = Path("../input/birdclef-2021/test_soundscapes")
SAMPLE_SUB_PATH = "../input/birdclef-2021/sample_submission.csv"
TARGET_PATH = None
if not len(list(TEST_AUDIO_ROOT.glob("*.ogg"))):
    TEST_AUDIO_ROOT = Path("../input/birdclef-2021/train_soundscapes")
    SAMPLE_SUB_PATH = None
    TARGET_PATH = Path("../input/birdclef-2021/train_soundscape_labels.csv")


class MelSpecComputer:
    # Класс-обёртка для вычисления мел-спектрограмм

    def __init__(self, sr, n_mels, fmin, fmax, n_fft=None, hop_length=None):
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        # Параметры STFT — используем простые и управляемые значения
        self.n_fft = n_fft if n_fft is not None else sr // 10        # ~100ms окно
        self.hop_length = hop_length if hop_length is not None else sr // 40  # ~25ms сдвиг

    def __call__(self, audio):
        # Преобразует одномерный аудиосигнал в мел-спектрограмму
        mel = lb.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        mel_db = lb.power_to_db(mel, ref=np.max).astype(np.float32)
        return mel_db


# Преобразование монохромного массива в 8-битовое изображение
def mono_to_color(arr, eps=1e-6, mean=None, std=None):
    mu = arr.mean() if mean is None else mean
    sigma = arr.std() if std is None else std

    norm = (arr - mu) / (sigma + eps)

    vmin, vmax = norm.min(), norm.max()

    if abs(vmax - vmin) > eps:
        img = np.clip(norm, vmin, vmax)
        img = 255 * (img - vmin) / (vmax - vmin)
        img = img.astype(np.uint8)
    else:
        img = np.zeros_like(arr, dtype=np.uint8)

    return img


# Обрезка или дополнение аудиосигнала до фиксированной длины
def crop_or_pad(signal, target_len):
    cur_len = len(signal)

    if cur_len < target_len:
        pad_amt = target_len - cur_len
        signal = np.concatenate([signal, np.zeros(pad_amt)])
    elif cur_len > target_len:
        signal = signal[:target_len]

    return signal



# Функция для визуализации waveform
def plot_waveform(audio, sr, title="Waveform"):
    plt.figure(figsize=(12, 4))
    lb.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


# Функция для визуализации mel-спектрограмм
def plot_mel_spectrogram(melspec, sr, hop_length, n_mels, title="Mel Spectrogram"):
    plt.figure(figsize=(12, 6))
    lb.display.specshow(melspec, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()


# Функция для прослушивания аудио
def play_audio(audio, sr, title="Audio"):
    print(f"Playing: {title}")
    display(ipd.Audio(audio, rate=sr))


# Функция для выполнения визуализации и прослушивания
def process_and_visualize(dataset, num_samples=2):
    for i in tqdm(range(min(len(dataset), num_samples)), desc="Processing and visualizing samples"):
        filepath = dataset.data.loc[i, "filepath"]
        original_audio, sr = sf.read(filepath, dtype="float32")

        if dataset.resample_flag and sr != dataset.sr:
            original_audio = lb.resample(original_audio, sr, dataset.sr, res_type=dataset.res_type)

        # Прослушивание аудио
        play_audio(original_audio, dataset.sr, title=f"Sample {i+1}: {dataset.data.loc[i, 'filename']}")

        # Визуализация waveform
        plot_waveform(original_audio, dataset.sr, title=f"Waveform for {dataset.data.loc[i, 'filename']}")

        # Вычисление mel-спектрограммы для визуализации
        melspec_for_plot = dataset.mel_comp(original_audio)
        # Визуализация mel-спектрограммы
        plot_mel_spectrogram(melspec_for_plot, dataset.sr, dataset.mel_comp.hop_length, dataset.n_mels, title=f"Mel Spectrogram for {dataset.data.loc[i, 'filename']}")



# Dataset для обработки аудиозаписей птиц
class BirdCLEFDataset(Dataset):
    def __init__(
        self,
        data,
        sr=SR,
        n_mels=128,
        fmin=0,
        fmax=None,
        duration=DURATION,
        step=None,
        res_type="kaiser_fast",
        resample=True,
        tta=True
    ):
        self.data = data

        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sr // 2

        self.duration = duration
        self.segment_len = self.duration * self.sr
        self.step = step if step is not None else self.segment_len

        self.res_type = res_type
        self.resample_flag = resample
        self.tta = tta
        # подготовка мел-спектрографа
        self.mel_comp = MelSpecComputer(
            sr=self.sr,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )

    def __len__(self):
        return len(self.data)
    @staticmethod
    def normalize(img):
        img = img.astype("float32") / 255.0
        return np.stack([img, img, img], axis=0)

    def audio_to_image(self, audio_chunk):
        mel = self.mel_comp(audio_chunk)
        colored = mono_to_color(mel)
        return self.normalize(colored)

    def apply_tta(self, audio):
        # random gain 0.9–1.1
        gain = np.random.uniform(0.9, 1.1)
        audio = audio * gain

        return audio

    def read_file(self, filepath):
        audio, orig_sr = sf.read(filepath, dtype="float32")

        if self.resample_flag and orig_sr != self.sr:
            audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)

        if self.tta:
            audio = self.apply_tta(audio)

        # Разделение аудио на фрагменты фиксированной длины
        chunks = []
        for pos in range(self.segment_len, len(audio) + self.step, self.step):
            start_idx = max(0, pos - self.segment_len)
            end_idx = start_idx + self.segment_len
            chunks.append(audio[start_idx:end_idx])

        if len(chunks) == 0:
            chunks = [crop_or_pad(audio, self.segment_len)]

        # конвертация в изображения
        images = [self.audio_to_image(chunk) for chunk in chunks]

        return np.stack(images, axis=0)

    def __getitem__(self, idx):
        return self.read_file(self.data.loc[idx, "filepath"])



data = pd.DataFrame(
     [(path.stem, *path.stem.split("_"), path) for path in Path(TEST_AUDIO_ROOT).glob("*.ogg")],
    columns = ["filename", "id", "site", "date", "filepath"]
)
print(data.shape)
data.head()


# Обучающие данные и информация о лейблах
df_train = pd.read_csv("../input/birdclef-2021/train_metadata.csv")

LABEL_IDS = {label: label_id for label_id,label in enumerate(sorted(df_train["primary_label"].unique()))}
INV_LABEL_IDS = {val: key for key,val in LABEL_IDS.items()}

test_data = BirdCLEFDataset(data=data)
len(test_data), test_data[0].shape


# Функция для визуализации
process_and_visualize(test_data, num_samples=2)


def load_net(checkpoint_path, num_classes=NUM_CLASSES):
    # Загружает сверточную сеть ResNeSt50, 
    # модифицирует финальный слой под нужное
    # количество классов и восстанавливает 
    # веса из чекпоинта.

    # Создаём модель без предобученных параметров
    model = resnest50(pretrained=False)

    # Меняем последний классификатор на собственный
    out_features = model.fc.in_features
    model.fc = nn.Linear(out_features, num_classes)

    # Загружаем веса на CPU
    saved = torch.load(checkpoint_path, map_location="cpu")

    # Удаляем префикс "model." у ключей, если он присутствует
    cleaned_state = {}
    for k, v in saved.items():
        new_key = k.split("model.")[-1]   # удаляет только ведущий "model."
        cleaned_state[new_key] = v

    # Применяем обновлённый словарь весов
    model.load_state_dict(cleaned_state)

    # Отправляем на устройство и ставим в режим inference
    model = model.to(device)
    model.eval()

    return model



checkpoint_paths = [
    Path("../input/kkiller-birdclef-models-public/birdclef_resnest50_fold0_epoch_10_f1_val_06471_20210417161101.pth"),
]

nets = [
        load_net(checkpoint_path.as_posix()) for checkpoint_path in checkpoint_paths
]


# Постобработка предсказаний
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


# Функция для предсказания
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


pred_prob = predict(nets, test_data, names=False)
print(len(pred_prob))


preds = [get_bird_names(get_thresh_preds(pred, thresh=THRESH)) for pred in pred_prob]


def preds_as_df(data, preds):
    sub = {
        "row_id": [],
        "birds": [],
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
print(sub.shape)
sub


# Сохранение результатов
sub.to_csv("submission.csv", index=False)

