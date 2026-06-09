import os
import copy
import numpy as np
import pandas as pd
import librosa
from skimage.transform import resize
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import regnet_y_16gf
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


df = pd.read_csv("/kaggle/input/rfcx-species-audio-detection/train_tp.csv")
df.head()


NUM_CLASSES = len(df["species_id"].unique())
FMIN = df["f_min"].min() * 0.9
FMAX = 24000
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
FOLD_NUM = 5
print(NUM_CLASSES, FMIN, FMAX, DEVICE)


TRAIN_DIR = "/kaggle/input/rfcx-species-audio-detection/train/"
TEST_DIR = "/kaggle/input/rfcx-species-audio-detection/test/"


cache_audio = {}
cache_splitted = {}


def get_audio(audio_dir: str, fileid: str, nocache: bool = False):
    if fileid in cache_audio:
        return cache_audio[fileid]
    filename = f"{audio_dir}/{fileid}.flac"
    if not os.path.isfile(filename):
        raise ValueError(f"Cannot find {fileid} in audio dir")
    waveform, sample_rate = librosa.load(filename, sr=None)
    if not nocache:
        cache_audio[fileid] = (waveform, sample_rate)
    return (waveform, sample_rate)


def get_audio_part(
    audio_dir: str,
    fileid: str,
    tmin: float,
    tmax: float,
    duration: int,
    nocache: bool = False,
):
    ts = round((float(tmax) + float(tmin)) / 2, 3)
    key = f"{fileid}_{ts}"
    if key in cache_splitted:
        return cache_splitted[key]
    wave, rate = get_audio(audio_dir, fileid)
    hd = duration / 2
    start = int((ts - hd) * rate)
    end = int((ts + hd) * rate)
    wave_len = len(wave)
    dur_diff = end - start - rate * duration
    if dur_diff != 0:
        end -= dur_diff
    if start < 0:
        end -= start
        start = 0
    if end > wave_len:
        start -= end - wave_len
        end = wave_len
    if start < 0 or end > wave_len:
        raise ValueError(
            f"Start or end beyond the wave length: start={start} end={end} len={wave_len}"
        )
    waveform_part = wave[start:end]
    if not nocache:
        cache_splitted[key] = (waveform_part, rate)
    return (waveform_part, rate)


def melspectrogram(w, r, fmin, fmax):
    return librosa.power_to_db(
        librosa.feature.melspectrogram(y=w, sr=r, n_mels=128, fmin=fmin, fmax=fmax)
    ).astype(np.float32)


def to_image(X):
    X = resize(X, (224, 400))
    eps = 1e-6
    mean = X.mean()
    std = X.std()
    X = (X - mean) / (std + eps)
    _min, _max = X.min(), X.max()
    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = (V - _min) / (_max - _min)
        V = V.astype(np.float32)
    else:
        V = np.zeros_like(X, dtype=np.float32)
    V = V[np.newaxis, ...]
    return np.concatenate((V, V, V))


class TrainDataset(Dataset):
    def __init__(self, data, audio_dir, fmin, fmax, duration: int = 10):
        self.data = data
        self.audio_dir = audio_dir
        self.fmin = fmin
        self.fmax = fmax
        self.duration = duration
        self.cache = {}

    def __len__(self):
        return len(self.data)

    def audio_to_image(self, audio, sr):
        return to_image(melspectrogram(audio, sr, self.fmin, self.fmax))

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        row_data = self.data.iloc[idx]
        fileid = row_data["recording_id"]
        tmin = row_data["t_min"]
        tmax = row_data["t_max"]
        s_id = row_data["species_id"]
        wave, sr = get_audio_part(self.audio_dir, fileid, tmin, tmax, self.duration)
        self.cache[idx] = (self.audio_to_image(wave, sr), s_id)
        return self.cache[idx]


train_data = TrainDataset(df, TRAIN_DIR, FMIN, FMAX)
test_item = train_data[0][0]
print(test_item.shape)

num = 3
fig, axs = plt.subplots(num, 1, figsize=(5, 10))
for i in range(num):
    axs[i].imshow(np.transpose(train_data[1 + i][0], axes=[1, 2, 0]))
plt.show()


def get_model(num_labels, device):
    model = regnet_y_16gf(weights="IMAGENET1K_SWAG_E2E_V1")
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_labels)
    model = model.to(device)
    return model


def train_model(
    model, loss_fn, train_loader, valid_loader, epochs, optimizer, scheduler
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for x, y in train_loader:
            optimizer.zero_grad()
            x = x.to(DEVICE, dtype=torch.float32)
            y = y.to(DEVICE, dtype=torch.long)
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

        for x, y in valid_loader:
            x = x.to(DEVICE, dtype=torch.float32)
            y = y.to(DEVICE, dtype=torch.long)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            trace_y.append(y.cpu().detach().numpy())
            trace_yhat.append(y_hat.cpu().detach().numpy())
            batch_losses.append(loss.item())
        valid_losses.append(batch_losses)
        trace_y = np.concatenate(trace_y)
        trace_yhat = np.concatenate(trace_yhat)
        accuracy = np.mean(trace_yhat.argmax(axis=1) == trace_y)

        print(
            f"Epoch {epoch}: train_loss = {np.mean(train_losses[-1]):.5f}, val_loss = {np.mean(valid_losses[-1]):.5f}, val_accuracy = {accuracy:.5f}"
        )

        scheduler.step(np.mean(valid_losses[-1]))
        if accuracy > best_acc:
            print(f"Best accuracy {best_acc:.5f} -> {accuracy:.5f}")
            best_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_wts)
    return model


def train_with_kfold(k: int):
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for fold_id, (train_index, val_index) in enumerate(
        kfold.split(df, df["species_id"].tolist())
    ):
        print(f"Train #{fold_id}")

        learning_rate = 2e-4
        epochs = 20
        loss_fn = torch.nn.CrossEntropyLoss()

        train_data = TrainDataset(df.loc[train_index], TRAIN_DIR, FMIN, FMAX)
        valid_data = TrainDataset(df.loc[val_index], TRAIN_DIR, FMIN, FMAX)
        train_loader = DataLoader(
            train_data, batch_size=16, shuffle=True, drop_last=True
        )
        valid_loader = DataLoader(
            valid_data, batch_size=16, shuffle=True, drop_last=True
        )
        model = get_model(NUM_CLASSES, DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=3
        )
        model = train_model(
            model, loss_fn, train_loader, valid_loader, epochs, optimizer, scheduler
        )
        print()
        torch.save(model.state_dict(), f"./model{fold_id}.pt")


train_with_kfold(FOLD_NUM)


cache_audio = {}
cache_splitted = {}


import gc

gc.collect()


models = []
for i in range(FOLD_NUM):
    model = get_model(NUM_CLASSES, DEVICE)
    model.load_state_dict(torch.load(f"./model{i}.pt", weights_only=True))
    models.append(model.eval())


class TestDataset(Dataset):
    def __init__(
        self, data, audio_dir, fmin, fmax, duration: int = 10, nocache: bool = False
    ):
        self.data = data
        self.audio_dir = audio_dir
        self.fmin = fmin
        self.fmax = fmax
        self.duration = duration
        self.cache = {}
        self.nocache = nocache

    def __len__(self):
        return len(self.data)

    def audio_to_image(self, audio, sr):
        return to_image(melspectrogram(audio, sr, self.fmin, self.fmax))

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        fileid = self.data[idx]
        fileid = fileid[: fileid.rfind(".flac")]
        wave, rate = get_audio(self.audio_dir, fileid, self.nocache)
        wave_len = len(wave) / rate
        output = []
        for i in range(int(np.ceil(wave_len / self.duration))):
            tmin = i * self.duration
            tmax = tmin + self.duration
            wave, sr = get_audio_part(
                self.audio_dir, fileid, tmin, tmax, self.duration, self.nocache
            )
            output.append(self.audio_to_image(wave, sr))
        if not self.nocache:
            self.cache[idx] = (fileid, np.array(output))
            return self.cache[idx]
        return (fileid, np.array(output))


def predict(models, device):
    test_dataset = TestDataset(
        os.listdir(TEST_DIR),
        TEST_DIR,
        FMIN,
        FMAX,
        nocache=True,
    )
    res_indexes = []
    res_probabilities = []
    one_model_rows = [[] for _ in models]
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            fileid, data = test_dataset[i]
            data = torch.tensor(data).float()
            if device == "cuda:0":
                data = data.cuda()

            pred = [torch.max(mdl(data), dim=0)[0].cpu().detach() for mdl in models]
            for pred_, model_row in zip(pred, one_model_rows):
                model_row.append([fileid] + [res.item() for res in pred_] )
            avg_pred = torch.mean(torch.stack(pred), dim=0)
            res_indexes.append(fileid)
            res_probabilities.append([avg.item() for avg in avg_pred])
    return res_indexes, res_probabilities, one_model_rows

def make_submission(preds):
    res_indexes, res_probabilities, one_model_rows = preds
    df_sub = pd.DataFrame(
        res_probabilities, columns=[f"s{i}" for i in range(NUM_CLASSES)]
    )
    df_sub["recording_id"] = res_indexes
    df_sub.to_csv("submission.csv", index=False)
    return df_sub, one_model_rows


df, one_model_rows = make_submission(predict(models, DEVICE))


ids = df.recording_id
col_names = ["recording_id"]+[f"s{i}" for i in range(NUM_CLASSES)]
for n, omrows in enumerate(one_model_rows):
    df = pd.DataFrame(omrows, columns=col_names)
    df.to_csv(f"submission_{n}.csv", index=False)
    for j, omrows2 in enumerate(one_model_rows[n + 1 :]):
        arr = np.array([[i[1:] for i in omrows], [i[1:] for i in omrows2]])
        rows_mean = np.mean(arr, axis=0)
        df = pd.DataFrame(rows_mean, columns=col_names[1:])
        df.insert(0, col_names[0], ids)
        df.to_csv(f"submission_{n}_{n+j+1}.csv", index=False)
        for k, omrows3 in enumerate(one_model_rows[n + j + 2 :]):
            arr = np.array(
                [
                    [i[1:] for i in omrows],
                    [i[1:] for i in omrows2],
                    [i[1:] for i in omrows3],
                ]
            )
            rows_mean = np.mean(arr, axis=0)
            df = pd.DataFrame(rows_mean, columns=col_names[1:])
            df.insert(0, col_names[0], ids)
            df.to_csv(f"submission_{n}_{n+j+1}_{n+j+k+2}.csv", index=False)




