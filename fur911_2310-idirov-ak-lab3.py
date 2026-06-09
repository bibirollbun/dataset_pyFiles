# --- core
import os, csv, copy, random, warnings
import numpy as np
import pandas as pd

# --- audio / images
import librosa
from skimage.filters import gaussian
from skimage.transform import resize
from skimage import exposure, util

# --- torch / training
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50

# --- utils
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import KFold
from tqdm import tqdm

warnings.filterwarnings("ignore")

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


class SpectrogramAug:
    def __init__(self):
        self._ops = (
            self._noise,
            self._contrast,
            self._flip_lr,
            self._flip_ud,
        )

    @staticmethod
    def _flip_lr(img2d):
        return np.stack((img2d[:, ::-1],) * 3)

    @staticmethod
    def _flip_ud(img2d):
        return np.stack((img2d[::-1, :],) * 3)

    @staticmethod
    def _noise(img2d):
        noisy = util.random_noise(img2d)
        return np.stack((noisy,) * 3)

    @staticmethod
    def _contrast(img2d):
        stretched = exposure.rescale_intensity(img2d)
        return np.stack((stretched,) * 3)

    def __call__(self, img2d):
        op = random.choice(self._ops)
        return op(img2d)


def mel_to_uint8(mel_db):
    """Приводит 2D mel-спектр (dB) к uint8-картинке 0..255."""
    mel_db = resize(mel_db, (224, 400))
    eps = 1e-6

    mu = mel_db.mean()
    sigma = mel_db.std()

    z = (mel_db - mu) / (sigma + eps)
    lo, hi = z.min(), z.max()
    scaled = 255.0 * (z - lo) / (hi - lo)
    return np.asarray(scaled.astype(np.uint8))


def build_net():
    net = resnet50(pretrained=True)
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, LABELS)
    return net.to(device)


meta = pd.read_csv("/kaggle/input/rfcx-species-audio-detection/train_tp.csv")

# берём минимальный f_min и максимальный f_max, затем расширяем 0.9/1.1
F_MIN = float(meta["f_min"].min())
F_MAX = float(meta["f_max"].max())

F_MIN = int(F_MIN * 0.9)
F_MAX = int(F_MAX * 1.1)

rec_ids = np.array(meta["recording_id"].tolist())
sp_ids  = np.array(meta["species_id"].tolist())



_cache = {}

def _clip_center_window(wav, sr, t0, t1, win_len=LENGTH):
    a = int(t0 * sr)
    b = int(t1 * sr)

    mid = np.round((a + b) / 2)
    left = max(mid - win_len // 2, 0)
    right = min(left + win_len, len(wav))

    left = right - win_len if (right - left) < win_len else left
    return wav[int(left):int(right)]

def _one_train_item(i):
    rid = rec_ids[i]
    sid = sp_ids[i]

    wav, sr = librosa.load(f"/kaggle/input/rfcx-species-audio-detection/train/{rid}.flac", sr=None)
    seg = _clip_center_window(
        wav, sr,
        meta.at[i, "t_min"],
        meta.at[i, "t_max"],
        win_len=LENGTH,
    )

    mel = librosa.feature.melspectrogram(y=seg, sr=sr, fmin=F_MIN, fmax=F_MAX)
    mel_db = librosa.power_to_db(mel, top_db=80)

    img = mel_to_uint8(mel_db)
    return rid, img

# параллельная подготовка кэша изображений
with ThreadPoolExecutor() as pool:
    for rid, img in pool.map(_one_train_item, range(len(meta))):
        _cache[rid] = img


class RFCXSpectroDataset(Dataset):
    def __init__(self, ids, targets, mode="train", aug=None, cache=None):
        self.ids = np.asarray(ids)
        self.targets = np.asarray(targets)
        self.mode = mode
        self.aug = aug
        self.cache = _cache if cache is None else cache

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        rid = self.ids[idx]
        y = int(self.targets[idx])
        img2d = self.cache[rid]

        if self.mode == "train" and self.aug is not None:
            x = self.aug(img2d)
        else:
            x = np.stack((img2d, img2d, img2d))

        return x, y


criterion = nn.CrossEntropyLoss()
augment = SpectrogramAug()

def fit_one_model(net, crit, dl_train, dl_valid, optim, sched):
    best_w = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    tr_hist, va_hist = [], []

    for ep in tqdm(range(1, EPOCHS + 1)):
        # ------------------ TRAIN ------------------
        net.train()
        batch_losses = []

        for _, batch in enumerate(dl_train):
            x, y = batch
            x = x.float().to(device)
            y = y.to(device)

            optim.zero_grad()
            out = net(x)
            loss = crit(out, y)
            loss.backward()
            optim.step()

            batch_losses.append(loss.item())

        tr_hist.append(batch_losses)

        # ------------------ VALIDATION ------------------
        net.eval()
        val_losses, trace_y, trace_yhat = [], [], []

        with torch.no_grad():
            for _, batch in enumerate(dl_valid):
                x, y = batch
                x = x.float().to(device)
                y = y.to(device)

                out = net(x)
                loss = crit(out, y)
                val_losses.append(loss.item())

                # Сохраняем батчи для accuracy
                trace_y.append(y.cpu().numpy())
                trace_yhat.append(out.cpu().numpy())

        va_hist.append(val_losses)

        # ------------------ METRICS ------------------
        if len(trace_y) == 0:
            print("Warning: empty fold, skipping metric calculation")
            acc = 0
        else:
            trace_y = np.concatenate(trace_y)
            trace_yhat = np.concatenate(trace_yhat)
            acc = np.mean(trace_yhat.argmax(axis=1) == trace_y)

            print("epoch = %d, train_loss = %.5f, val_loss = %.5f, val_accuracy = %.5f" % (
                ep, np.mean(tr_hist[-1]), np.mean(va_hist[-1]), acc))

        # ------------------ SCHEDULER ------------------
        if len(va_hist[-1]) > 0:
            sched.step(np.mean(va_hist[-1]))

        # ------------------ BEST MODEL ------------------
        if acc > best_acc:
            best_acc = acc
            best_w = copy.deepcopy(net.state_dict())

    net.load_state_dict(best_w)
    return net



print("Total samples:", len(rec_ids))
print("Unique species:", len(np.unique(sp_ids)))


from sklearn.model_selection import StratifiedKFold
print(f"Cache size: {len(_cache)}")
print(f"Some keys: {list(_cache.keys())[:5]}")
missing_keys = [rid for rid in rec_ids if rid not in _cache]
print(f"Missing keys in cache: {missing_keys[:10]}")

kf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=563)
for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(rec_ids, sp_ids)):
    print("Fold", fold_idx)

    tr_ids = np.take(rec_ids, tr_idx)
    tr_y   = np.take(sp_ids, tr_idx, axis=0)
    va_ids = np.take(rec_ids, va_idx)
    va_y   = np.take(sp_ids, va_idx, axis=0)

    # Проверка размеров фолдов
    print(f"Fold {fold_idx}: train size = {len(tr_idx)}, valid size = {len(va_idx)}")
    print(f"Example train IDs: {rec_ids[tr_idx[:5]]}")
    print(f"Example valid IDs: {rec_ids[va_idx[:5]]}")
    

    
    ds_tr = RFCXSpectroDataset(tr_ids, tr_y, mode="train", aug=augment)
    ds_va = RFCXSpectroDataset(va_ids, va_y, mode="valid", aug=None)
     
    batch_size_tr = min(8, len(ds_tr))
    batch_size_va = min(8, len(ds_va))
    
    dl_tr = DataLoader(ds_tr, batch_size=batch_size_tr, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size_va, shuffle=False, drop_last=False)

    
    batch_size_tr = min(8, len(ds_tr))
    batch_size_va = min(8, len(ds_va))
    
    dl_tr = DataLoader(ds_tr, batch_size=batch_size_tr, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size_va, shuffle=False, drop_last=False)

    for x, y in dl_tr:
        print(f"Train batch: x shape = {x.shape}, y shape = {y.shape}")
        break
    for x, y in dl_va:
        print(f"Valid batch: x shape = {x.shape}, y shape = {y.shape}")
        break
    
    net = build_net()
    optim = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, "min", patience=3)

    net = fit_one_model(net, criterion, dl_tr, dl_va, optim, sched)
    torch.save(net.state_dict(), f"./model{fold_idx}.pt")

    del ds_tr, ds_va, dl_tr, dl_va, net, tr_ids, va_ids, tr_y, va_y


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

        img = mel_to_uint8(spec_db)
        mel_spec = np.stack((img, img, img))
        mel_array.append(mel_spec)

    return mel_array


members = []

for i in range(N_FOLD):
    model = build_net()

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

