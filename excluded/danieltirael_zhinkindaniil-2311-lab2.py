import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import librosa as lb
import soundfile as sf

import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision.models as models

from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

from sklearn.metrics import f1_score

print("Torch:", torch.__version__)

# Основные константы
NUM_CLASSES = 397          # число видов птиц в BirdCLEF 2021
SR = 32000                 # частота дискретизации
DURATION = 5               # длина окна в секундах
THRESH = 0.25              # порог для выбора видов

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

DATA_ROOT = Path("../input/birdclef-2021")

TEST_AUDIO_ROOT = DATA_ROOT / "test_soundscapes"
SAMPLE_SUB_PATH = DATA_ROOT / "sample_submission.csv"
TARGET_PATH = None

# Если в тесте нет файлов, работаем в offline-режиме по train_soundscapes
if not len(list(TEST_AUDIO_ROOT.glob("*.ogg"))):
    TEST_AUDIO_ROOT = DATA_ROOT / "train_soundscapes"
    SAMPLE_SUB_PATH = None
    TARGET_PATH = DATA_ROOT / "train_soundscape_labels.csv"

print("Audio root:", TEST_AUDIO_ROOT)
print("Sample submission:", SAMPLE_SUB_PATH)
print("Train soundscape labels:", TARGET_PATH)



!ls /kaggle/input/resnest50-fast-package/resnest-0.0.6b20200701/resnest



import sys

# добавляем корень пакета в sys.path
sys.path.append("/kaggle/input/resnest50-fast-package/resnest-0.0.6b20200701")
sys.path.append("/kaggle/input/resnest50-fast-package/resnest-0.0.6b20200701/resnest")

from resnest.torch import resnest50

print("ResNeSt импортирован ок!")



# Соберём для каждого site список видов, которые реально встречались в train_soundscapes
SITE_SPECIES = {}

if TARGET_PATH is not None and TARGET_PATH.exists():
    labels_df_site = pd.read_csv(TARGET_PATH)

    for row in labels_df_site.itertuples(index=False):
        site = row.site
        birds = row.birds
        if isinstance(birds, str) and birds != "nocall":
            species = birds.split()
            if site not in SITE_SPECIES:
                SITE_SPECIES[site] = set()
            SITE_SPECIES[site].update(species)

    print("Виды по сайтам:")
    for s, sp in SITE_SPECIES.items():
        print(f"{s}: {len(sp)} видов")
else:
    print("train_soundscape_labels.csv не найден, SITE_SPECIES пустой")



# Собираем таблицу с файлами soundscapes
data = pd.DataFrame(
    [
        (path.stem, *path.stem.split("_"), path)
        for path in TEST_AUDIO_ROOT.glob("*.ogg")
    ],
    columns=["filename", "id", "site", "date", "filepath"],
)

print("Число файлов soundscapes:", data.shape[0])
data.head()



class MelSpecComputer:
    """
    Обёртка над librosa.feature.melspectrogram
    с более удобной настройкой параметров.
    """
    def __init__(self, sr, n_mels, fmin, fmax, **kwargs):
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        kwargs["n_fft"] = kwargs.get("n_fft", self.sr // 10)
        kwargs["hop_length"] = kwargs.get("hop_length", self.sr // 40)
        self.kwargs = kwargs

    def __call__(self, y):
        melspec = lb.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            **self.kwargs,
        )
        melspec = lb.power_to_db(melspec).astype(np.float32)
        return melspec


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    """
    Переводим мел-спектрограмму в "картинку":
    нормируем, растягиваем в [0,255], приводим к uint8.
    """
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



class BirdCLEFDataset(Dataset):
    """
    Датасет для длинных soundscape-записей.
    На каждый файл возвращает массив "картинок" для всех 5-секундных окон:
    (num_windows, 3, n_mels, time)
    """
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
    ):
        self.data = data

        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr // 2

        self.duration = duration
        self.audio_length = int(self.duration * self.sr)
        self.step = step or self.audio_length  # сдвигаем ровно на окно

        self.res_type = res_type
        self.resample = resample

        self.mel_spec_computer = MelSpecComputer(
            sr=self.sr,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )

    def __len__(self):
        return len(self.data)

    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        image = np.stack([image, image, image])  # [3,H,W]
        return image

    def audio_to_image(self, audio):
        melspec = self.mel_spec_computer(audio)
        image = mono_to_color(melspec)
        image = self.normalize(image)
        return image

    def read_file(self, filepath: Path):
        audio, orig_sr = sf.read(filepath, dtype="float32")

        if self.resample and orig_sr != self.sr:
            audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)

        audios = []
        # нарезаем по 5 секунд
        for i in range(self.audio_length, len(audio) + self.step, self.step):
            start = max(0, i - self.audio_length)
            end = start + self.audio_length
            audios.append(audio[start:end])

        # если последний кусок получился слишком коротким — выбрасываем
        if len(audios) > 0 and len(audios[-1]) < self.audio_length:
            audios = audios[:-1]

        images = [self.audio_to_image(a) for a in audios]
        if len(images) == 0:
            # очень короткий файл — паддим тишиной
            images = [self.audio_to_image(np.zeros(self.audio_length, dtype=np.float32))]
        images = np.stack(images)
        return images

    def __getitem__(self, idx):
        return self.read_file(self.data.loc[idx, "filepath"])



test_data = BirdCLEFDataset(data=data)

print("Всего файлов:", len(test_data))
sample_batch = test_data[0]
print("Форма для первого файла:", sample_batch.shape)  # (num_windows, 3, 128, T)

# Визуализируем одну картинку (один 5-сек отрезок)
img = sample_batch[0]  # (3, H, W)
plt.figure(figsize=(10, 4))
plt.title("Пример mel-спектрограммы (как картинка)")
plt.imshow(np.transpose(img, (1, 2, 0)))
plt.axis("off")
plt.show()



import gc, torch

gc.collect()
torch.cuda.empty_cache()
print("CUDA memory cleaned")



# Метаданные train, чтобы восстановить порядок классов
df_train = pd.read_csv(DATA_ROOT / "train_metadata.csv")

LABEL_IDS = {label: label_id for label_id, label in enumerate(sorted(df_train["primary_label"].unique()))}
INV_LABEL_IDS = {v: k for k, v in LABEL_IDS.items()}
NUM_CLASSES = len(LABEL_IDS)
print("Число классов:", NUM_CLASSES)


def load_net_resnest(checkpoint_path, num_classes=NUM_CLASSES):
    """
    Загружаем ResNeSt50, обученный на BirdCLEF
    (чекпоинт из kkiller-birdclef-models-public).
    """
    net = resnest50(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, num_classes)

    state = torch.load(checkpoint_path, map_location="cpu")

    # иногда веса сохраняются с префиксом "model."
    for k in list(state.keys()):
        if k.startswith("model."):
            state[k[6:]] = state.pop(k)

    net.load_state_dict(state, strict=True)
    net = net.to(DEVICE)
    net.eval()
    return net



checkpoint_paths = [
    Path(
        "/kaggle/input/kkiller-birdclef-models-public"
        "/birdclef_resnest50_fold0_epoch_10_f1_val_06471_20210417161101.pth"
    )
]

nets = [load_net_resnest(path) for path in checkpoint_paths]
print(f"Моделей в ансамбле: {len(nets)}")
print(nets[0].fc)





@torch.no_grad()
def get_thresh_preds(out: torch.Tensor, thresh: float = None):
    """
    out: [N, num_classes] после sigmoid
    Возвращает список списков индексов классов, прошедших порог.
    """
    thresh = thresh or THRESH
    # сортируем по вероятности по убыванию
    sorted_idx = (-out).argsort(1)  # [N, num_classes]
    num_above = (out > thresh).sum(1)  # [N]

    preds = []
    for row_idx, n in zip(sorted_idx, num_above):
        idxs = row_idx[:n].cpu().numpy().tolist()
        preds.append(idxs)
    return preds


def get_bird_names(preds, site=None):
    bird_names = []
    for p in preds:
        if not p:
            bird_names.append("nocall")
        else:
            bird_names.append(" ".join(INV_LABEL_IDS[i] for i in p))
    return bird_names



@torch.no_grad()
def predict(nets, dataset, meta_df=None, return_names=True, batch_size=16):
    """
    nets: список моделей
    dataset: BirdCLEFDataset
    meta_df: DataFrame с колонкой 'site' (наш data)
    return_names:
      - True  -> строки с названиями видов / 'nocall'
      - False -> сырые вероятности (numpy) [num_windows, num_classes]
    batch_size: сколько 5-сек окон обрабатывать за раз (важно для CUDA OOM)
    """
    all_preds = []

    for idx in tqdm(range(len(dataset)), desc="Inference over soundscapes"):
        # numpy-массив для текущего файла: [num_windows, 3, H, W]
        arr = dataset[idx]
        num_windows = arr.shape[0]

        # собираем предсказания по батчам
        probs_chunks = []

        for start in range(0, num_windows, batch_size):
            end = min(start + batch_size, num_windows)
            batch_np = arr[start:end]                          # [B, 3, H, W]
            xb = torch.from_numpy(batch_np).to(DEVICE)

            # усредняем по ансамблю
            batch_probs = torch.zeros(
                (end - start, NUM_CLASSES), device=DEVICE, dtype=torch.float32
            )
            for net in nets:
                logits = net(xb)                               # [B, C]
                batch_probs += torch.sigmoid(logits)
            batch_probs /= len(nets)

            probs_chunks.append(batch_probs.cpu())

            # чистим за собой
            del xb, logits, batch_probs
            torch.cuda.empty_cache()

        # склеиваем все батчи в [num_windows, num_classes]
        full_probs = torch.cat(probs_chunks, dim=0)

        if return_names:
            site = meta_df.iloc[idx].site if meta_df is not None else None
            idx_preds = get_thresh_preds(full_probs)           # использует THRESH
            window_preds = get_bird_names(idx_preds, site=site)
        else:
            window_preds = full_probs.numpy()

        all_preds.append(window_preds)

        # ещё немного уборки
        del probs_chunks, full_probs, window_preds
        torch.cuda.empty_cache()

    return all_preds



# Получаем предсказания: для каждого файла список строк по 5-сек окнами
preds = predict(nets, test_data, meta_df=data, return_names=True)

print(f"Сколько файлов обработано: {len(preds)}")
print("Для первого файла количество окон:", len(preds[0]))
print("Первые 10 окон и предсказанные птицы:")
for i, p in enumerate(preds[0][:10], start=1):
    print(f"{i*DURATION- DURATION:>3}-{i*DURATION:>3} сек: {p}")



def preds_as_df(data, preds, duration=DURATION):
    """
    data: DataFrame с колонками id, site
    preds: список списков строк ('nocall' или 'sp1 sp2 ...') для каждого файла
    """
    rows = {
        "row_id": [],
        "birds": [],
    }

    for row, file_preds in zip(data.itertuples(index=False), preds):
        # row.id, row.site
        for i, birds in enumerate(file_preds, start=1):
            second = i * duration
            row_id = f"{row.id}_{row.site}_{second}"
            rows["row_id"].append(row_id)
            rows["birds"].append(birds)

    sub = pd.DataFrame(rows)

    # если есть sample_submission.csv, то подстраиваемся под его row_id
    if SAMPLE_SUB_PATH is not None and SAMPLE_SUB_PATH.exists():
        sample_sub = pd.read_csv(SAMPLE_SUB_PATH, usecols=["row_id"])
        sub = sample_sub.merge(sub, on="row_id", how="left")
        sub["birds"] = sub["birds"].fillna("nocall")

    return sub


submission = preds_as_df(data, preds, duration=DURATION)
print(submission.shape)
submission.head()



submission.to_csv("submission.csv", index=False)
print("submission.csv сохранён!")



def rowwise_micro_f1(true_birds, pred_birds):
    """
    Приблизительный row-wise micro F1:
    для каждой строки сравниваем множества истинных и предсказанных видов.
    """
    tp = fp = fn = 0

    for yt, yp in zip(true_birds, pred_birds):
        # NaN -> nocall
        if isinstance(yt, float) and math.isnan(yt):
            yt = "nocall"
        if isinstance(yp, float) and math.isnan(yp):
            yp = "nocall"

        true_set = set([] if yt == "nocall" else yt.split())
        pred_set = set([] if yp == "nocall" else yp.split())

        tp += len(true_set & pred_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    if tp + fp + fn == 0:
        return 0.0

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


if TARGET_PATH is not None and TARGET_PATH.exists():
    labels_df = pd.read_csv(TARGET_PATH)
    # совмещаем по row_id
    merged = labels_df[["row_id", "birds"]].merge(
        submission, on="row_id", how="left", suffixes=("_true", "_pred")
    )

    score = rowwise_micro_f1(
        merged["birds_true"].tolist(),
        merged["birds_pred"].fillna("nocall").tolist(),
    )
    print(f"Row-wise micro F1 на train_soundscapes (offline): {score:.4f}")
else:
    print("Нет train_soundscape_labels.csv — offline-оценка недоступна.")


