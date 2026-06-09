import shutil
import os

shutil.copytree('../input/resnest50-fast-package/resnest-0.0.6b20200701/resnest', 'resnet', dirs_exist_ok=True)
os.system('pip install "./resnet" --no-deps')
import os
import numpy as np
import pandas as pd
import librosa
import torch
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from resnest.torch import resnest50
from sklearn.preprocessing import LabelEncoder


PATH_TEST = "../input/birdclef-2021/test_soundscapes"
PATH_TRAIN = "../input/birdclef-2021/train_soundscapes"

meta = pd.read_csv("../input/birdclef-2021/test.csv")

audio_root = PATH_TEST
if meta.shape[0] < 5:
    meta = pd.read_csv("../input/birdclef-2021/train_soundscape_labels.csv")
    audio_root = PATH_TRAIN

print(meta.shape)
display(meta.head())


audio_buf = {}
sections_buf = {}


def load_full_clip(ds, idx, target_rate=None):
    tag = f"{idx}_{ds}"
    if tag in audio_buf:
        return audio_buf[tag]

    file_found = None
    for fname in os.listdir(audio_root):
        if fname.startswith(tag):
            file_found = os.path.join(audio_root, fname)
            break

    if file_found is None:
        raise RuntimeError("Файл не найден: " + tag)

    audio, sr = librosa.load(file_found, sr=None, res_type="kaiser_fast")

    if target_rate and sr != target_rate:
        audio = librosa.resample(audio, sr, target_rate)
        sr = target_rate

    audio_buf[tag] = (audio, sr)
    return audio, sr


def load_fragment(ds, idx, sec, win=5, target_rate=None):
    key = f"{idx}_{ds}_{sec}"
    if key in sections_buf:
        return sections_buf[key]

    wav, rate = load_full_clip(ds, idx, target_rate)
    end = int(sec) * rate
    start = end - win * rate
    frag = wav[start:end]

    sections_buf[key] = (frag, rate)
    return frag, rate


def mel_transform(signal, sr):
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=128,
        fmin=0,
        fmax=sr / 2,
        n_fft=sr // 10,
        hop_length=sr // 40,
    )
    return librosa.power_to_db(mel).astype(np.float32)


def norm_to_uint(img):
    eps = 1e-6
    x = (img - img.mean()) / (img.std() + eps)
    mn, mx = x.min(), x.max()
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.uint8)
    x = np.clip(x, mn, mx)
    x = ((x - mn) / (mx - mn) * 255).astype(np.uint8)
    return x


def stack_rgb(x):
    return np.repeat(x[np.newaxis, :, :], 3, axis=0).astype(np.float32) / 255.0


class BirdClips(Dataset):
    def __init__(self, table):
        self.tbl = table
        self.target_sr = 32000
        self.win_len = 5
        self.cache = {}

    def __len__(self):
        return len(self.tbl)

    def _make_img(self, audio):
        return stack_rgb(norm_to_uint(mel_transform(audio, self.target_sr)))

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        rid = self.tbl.loc[idx, "row_id"]
        clip, ds, sec = rid.split("_")[:3]

        audio, _ = load_fragment(ds, clip, int(sec), win=self.win_len, target_rate=self.target_sr)
        img = self._make_img(audio)

        self.cache[idx] = img
        return img


dataset = BirdClips(meta)

print(dataset[0].shape)

fig, axes = plt.subplots(9, 1, figsize=(10, 20))
for i in range(9):
    axes[i].imshow(np.transpose(dataset[i], (1, 2, 0)))
plt.show()




labels_df = pd.read_csv("../input/birdclef-2021/train_metadata.csv")
encoder = LabelEncoder().fit(sorted(labels_df["primary_label"].unique()))
n_classes = len(encoder.classes_)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def init_model(weights_path):
    net = resnest50(pretrained=False)
    net.fc = torch.nn.Linear(net.fc.in_features, n_classes)

    saved = torch.load(weights_path, map_location="cpu")


    fixed = {}
    for k, v in saved.items():
        nk = k.replace("model.", "")
        fixed[nk] = v

    net.load_state_dict(fixed)
    net.to(device)
    net.eval()
    return net


net = init_model("../input/kkiller-birdclef-models-public/birdclef_resnest50_fold0_epoch_10_f1_val_06471_20210417161101.pth")


@torch.no_grad()
def pick_labels(logits):
    thr = 0.1
    idx_sorted = (-logits).argsort(1)
    cnt = (logits > thr).sum(1)
    return [i[:c].cpu().numpy().tolist() for i, c in zip(idx_sorted, cnt)]


def decode_predictions(arr):
    return [
        " ".join(encoder.inverse_transform(x)) if x else "nocall"
        for x in arr
    ]


def forward_batch(batch):
    tens = torch.from_numpy(batch).to(device)
    out = torch.sigmoid(net(tens))
    return decode_predictions(pick_labels(out))


# тестовое выполнение
sample = np.stack([dataset[200 + i] for i in range(10)])
preds = forward_batch(sample)

for i, p in enumerate(preds):
    print(f"{i*5}-{i*5+5} sec:", p)


def full_predict(ds):
    bs = 64
    L = len(ds)
    result = []
    for start in range(0, L, bs):
        items = [ds[i] for i in range(start, min(start + bs, L))]
        arr = np.stack(items)
        result.extend(forward_batch(arr))
    return result


final_preds = full_predict(dataset)


def prepare_submit(ref_df, pr):
    out = pd.DataFrame({
        "row_id": ref_df["row_id"],
        "birds": pr
    })
    return out


submission = prepare_submit(meta, final_preds)
print(submission.head())
submission.to_csv("submission.csv", index=False)


