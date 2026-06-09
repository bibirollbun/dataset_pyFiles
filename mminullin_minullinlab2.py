import cv2
import re
import torch
import time

import numpy as np
import librosa as lb
import soundfile as sf
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm

try:
    import resnest
except ModuleNotFoundError:
    import shutil
    shutil.copytree('../input/resnest50-fast-package/resnest-0.0.6b20200701/resnest', 'resnet', dirs_exist_ok=True) 
    !pip install "./resnet"


test_root = Path("../input/birdclef-2021/test_soundscapes")
train_data = Path("../input/birdclef-2021/train_soundscapes")
# for submission
if len(list(test_root.glob("*.ogg"))):
    train_data = test_root

data = pd.DataFrame(
     [(path.stem, *path.stem.split("_"), path) for path in Path(train_data).glob("*.ogg")],
    columns = ["filename", "id", "site", "date", "filepath"]
)
print(data.shape)
data.head()


def melspectrogram(y, params):
    result = lb.feature.melspectrogram(y=y, sr=params['sr'], n_mels=params['n_mels'], fmin=params['fmin'], fmax=params['fmax'])
    result = lb.power_to_db(result).astype(np.float32)
    return result

def colorize(X, eps=1e-6, mean=None, std=None):
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

def normalize(image):
    image = image.astype("float32", copy=False) / 255.0
    image = np.stack([image, image, image])
    return image

class CustomDataset(Dataset):
    def __init__(self, data, sr=32000, n_mels=128, fmin=0, fmax=None, duration=5, step=None, res_type="kaiser_fast", resample=True):
        self.data = data
        self.params = {
            'sr': sr,
            'n_mels': n_mels,
            'fmin': fmin,
            'fmax': fmax or sr // 2
        }
        self.duration = duration
        self.audio_length = duration * sr
        self.step = step or self.audio_length
        self.res_type = res_type
        self.resample = resample
    
    def __len__(self):
        return len(self.data)
    
    def audio_to_image(self, audio):
        melspec = melspectrogram(audio, self.params)
        image = normalize(colorize(melspec))
        
        return image
    
    def __getitem__(self, idx):
        audio, orig_sr = sf.read(self.data.loc[idx, "filepath"], dtype="float32")

        if self.resample and orig_sr != self.params['sr']:
            audio = lb.resample(audio, orig_sr, self.params['sr'], res_type=self.res_type)
          
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


test_data = CustomDataset(data=data)

test_item = test_data[1]
print(test_item.shape)
from matplotlib import pyplot as plt
test_images = [np.einsum('kij->ijk', img) for img in test_item]

fig, axs = plt.subplots(4, 4, figsize=(18, 9))
for i in range(16):
    axs[int(i / 4), i % 4].imshow(test_images[i + 1])
plt.show()


df_train = pd.read_csv("../input/birdclef-2021/train_metadata.csv")

id_by_label = {label: label_id for label_id,label in enumerate(sorted(df_train["primary_label"].unique()))}
label_by_id = {val: key for key,val in id_by_label.items()}
num_classes = len(label_by_id.keys())
print(id_by_label)


checkpoint_path = Path("../input/kkiller-birdclef-models-public/birdclef_resnest50_fold0_epoch_10_f1_val_06471_20210417161101.pth")
from resnest.torch import resnest50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = resnest50(pretrained=False)
net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
dummy_device = torch.device("cpu")
d = torch.load(checkpoint_path, map_location=dummy_device)
for key in list(d.keys()):
    d[key.replace("model.", "")] = d.pop(key)
net.load_state_dict(d)
net = net.to(DEVICE)
net = net.eval()


@torch.no_grad()
def get_thresh_preds(out, thresh=0.15):
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
            bird_names.append(" ".join([label_by_id[bird_id] for bird_id in pred]))
    return bird_names

xb = torch.from_numpy(test_data[3]).to(DEVICE)
o = net(xb)
pred = get_bird_names(get_thresh_preds(o))
out = ''
for idx, elem in enumerate(pred):
    if elem != 'nocall':
        out += 'found ' + elem + ' at ' + str(idx * 5) + '-' + str(idx * 5 + 5) + ' sec of audio\n'
print(out)


def predict(net, test_data):
    preds = []
    with torch.no_grad():
        for idx in  tqdm(list(range(len(test_data)))):
            xb = torch.from_numpy(test_data[idx]).to(DEVICE)
            pred = 0.
            o = net(xb)
            pred = torch.sigmoid(o)
            pred = get_bird_names(get_thresh_preds(pred))
            preds.append(pred)
    return preds

preds = predict(net, test_data)


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
    sample_sub = pd.read_csv("../input/birdclef-2021/sample_submission.csv", usecols=["row_id"])
    sub = sample_sub.merge(sub, on="row_id", how="left")
    sub["birds"] = sub["birds"].fillna("nocall")
    return sub


sub = preds_as_df(data, preds)
print(sub)
sub.to_csv("submission.csv", index=False)

