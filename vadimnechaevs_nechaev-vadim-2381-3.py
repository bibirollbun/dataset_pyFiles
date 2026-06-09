
import os
import csv
import copy
import random
import warnings

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn

from tqdm import tqdm
from sklearn.model_selection import KFold
from torchvision.models import resnet101
from torch.utils.data import Dataset, DataLoader

from skimage import exposure, util
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.transform import resize

warnings.filterwarnings("ignore")

NUM_CLASSES = 24
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"



def to_3ch(img):
    return np.stack((img, img, img))


def aug_flip_h(img):
    return to_3ch(img[:, ::-1])


def aug_flip_v(img):
    return to_3ch(img[::-1, :])


def aug_noise(img):
    return to_3ch(util.random_noise(img))


def aug_gauss(img):
    return to_3ch(gaussian(img))


def aug_contrast(img):
    return to_3ch(exposure.rescale_intensity(img))


def aug_gamma(img):
    return to_3ch(exposure.adjust_gamma(img))


def aug_gray(img):
    return to_3ch(rgb2gray(img))


def normalize_mel(spec):
    spec = resize(spec, (224, 400))
    eps = 1e-6
    norm = (spec - spec.mean()) / (spec.std() + eps)
    mn, mx = norm.min(), norm.max()
    scaled = 255 * (norm - mn) / (mx - mn)
    return scaled.astype(np.uint8)




def build_model():
    net = resnet101(pretrained=True)
    inp = net.fc.in_features
    net.fc = nn.Linear(inp, NUM_CLASSES)
    net.to(DEVICE)
    return net




sr = 48000
chunk_len = sr * 10

train_df = pd.read_csv("../input/rfcx-species-audio-detection/train_tp.csv")

low = 24000
high = 0

for _, r in train_df.iterrows():
    low = min(low, float(r["f_min"]))
    high = max(high, float(r["f_max"]))

fmin = int(low * 0.9)
fmax = int(high * 1.1)

ids = []
labels = []
melbank = {}

for i in range(len(train_df)):
    rid = train_df.recording_id.iloc[i]
    sid = int(train_df.species_id.iloc[i])
    ids.append(rid)
    labels.append(sid)

    audio, sr_now = librosa.load(
        f"../input/rfcx-species-audio-detection/train/{rid}.flac", sr=None
    )

    t1 = int(train_df.t_min.iloc[i] * sr_now)
    t2 = int(train_df.t_max.iloc[i] * sr_now)

    center = int((t1 + t2) / 2)
    start = max(0, center - chunk_len // 2)
    end = start + chunk_len

    if end > len(audio):
        end = len(audio)
        start = end - chunk_len

    clip = audio[start:end]

    mel = librosa.feature.melspectrogram(y=clip, sr=sr_now, fmin=fmin, fmax=fmax)
    mel_db = librosa.power_to_db(mel, top_db=80)

    melbank[rid] = normalize_mel(mel_db)




class BirdDataset(Dataset):
    def __init__(self, X, y, mode):
        self.X = X
        self.y = y
        self.mode = mode

        self.transforms = [
            aug_noise,
            aug_contrast,
            aug_gauss,
            aug_gamma,
            aug_flip_v,
            aug_flip_h,
            to_3ch
        ]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        rid = self.X[idx]
        img = melbank[rid]

        if self.mode == "train":
            fn = random.choice(self.transforms)
            img = fn(img)
        else:
            img = to_3ch(img)

        return img, self.y[idx]




LR = 2e-4
EPOCHS = 20
criterion = nn.CrossEntropyLoss()

def fit(model, loss_fn, train_loader, valid_loader, epochs, optimizer, scheduler):
    best = 0
    best_weights = copy.deepcopy(model.state_dict())

    for ep in tqdm(range(1, epochs + 1)):
        model.train()
        train_loss = []

        for x, y in train_loader:
            optimizer.zero_grad()

            x = x.float().to(DEVICE)
            y = y.long().to(DEVICE)

            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        model.eval()
        val_loss = []
        true = []
        pred = []

        with torch.no_grad():
            for x, y in valid_loader:
                x = x.float().to(DEVICE)
                y = y.long().to(DEVICE)

                out = model(x)
                loss = loss_fn(out, y)

                val_loss.append(loss.item())
                true.append(y.cpu().numpy())
                pred.append(out.cpu().numpy())

        true = np.concatenate(true)
        pred = np.concatenate(pred)

        acc = (pred.argmax(1) == true).mean()

        print(f"epoch={ep}, train={np.mean(train_loss):.5f}, val={np.mean(val_loss):.5f}, acc={acc:.5f}")

        scheduler.step(np.mean(val_loss))

        if acc > best:
            best = acc
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model




NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=32)

for fold, (tr, vl) in enumerate(kf.split(ids, labels)):
    print("FOLD", fold)

    X_tr = np.take(ids, tr)
    y_tr = np.take(labels, tr)
    X_vl = np.take(ids, vl)
    y_vl = np.take(labels, vl)

    train_ds = BirdDataset(X_tr, y_tr, "train")
    val_ds = BirdDataset(X_vl, y_vl, "valid")

    train_ld = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=8, shuffle=True, drop_last=True)

    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=3)

    model = fit(model, criterion, train_ld, val_ld, EPOCHS, opt, sched)

    torch.save(model.state_dict(), f"./fold_model_{fold}.pt")

    del model, train_ds, val_ds, train_ld, val_ld



def prepare_test_audio(filename):
    audio, s = librosa.load(f"../input/rfcx-species-audio-detection/test/{filename}", sr=None)

    segs = int(np.ceil(len(audio) / chunk_len))
    result = []

    for i in range(segs):
        if (i + 1) * chunk_len > len(audio):
            part = audio[len(audio) - chunk_len:]
        else:
            part = audio[i * chunk_len:(i + 1) * chunk_len]

        mel = librosa.feature.melspectrogram(y=part, sr=s, fmin=fmin, fmax=fmax)
        mel_db = librosa.power_to_db(mel, top_db=80)

        img = normalize_mel(mel_db)
        img = np.stack((img, img, img))
        result.append(img)

    return result



ensemble = []

for i in range(NFOLDS):
    m = build_model()
    m.load_state_dict(torch.load(f"./fold_model_{i}.pt"))
    m.eval()
    ensemble.append(m)

for i in range(NFOLDS):
    os.remove(f"./fold_model_{i}.pt")



print("Predicting...")

with open("submission.csv", "w", newline="") as file:
    wr = csv.writer(file)
    wr.writerow(["recording_id"] + [f"s{i}" for i in range(24)])

    test_files = os.listdir("../input/rfcx-species-audio-detection/test/")
    print("Files:", len(test_files))

    for i, fname in enumerate(test_files):
        data = torch.tensor(prepare_test_audio(fname)).float()

        if torch.cuda.is_available():
            data = data.cuda()

        preds = []

        for m in ensemble:
            out = m(data)
            best = torch.max(out, dim=0)[0].cpu().detach()
            preds.append(best)

        final = torch.mean(torch.stack(preds), dim=0)

        rec_id = fname.split(".")[0]
        row = [rec_id] + [x.item() for x in final]
        wr.writerow(row)

        if i % 100 == 0 and i > 0:
            print(f"Done {i}/{len(test_files)}")

print("submission.csv готов!")


