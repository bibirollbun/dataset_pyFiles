# =========================
# 1) Ğ˜Ğ¼Ğ¿Ğ¾Ñ€Ñ‚Ñ‹ Ğ¸ Ğ¾ĞºÑ€ÑƒĞ¶ĞµĞ½Ğ¸Ğµ
# =========================

import os
import csv
import copy
import random
import warnings
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from torchvision.models import resnet50
try:
    # torchvision>=0.13
    from torchvision.models import ResNet50_Weights
    _HAS_NEW_TORCHVISION = True
except Exception:
    _HAS_NEW_TORCHVISION = False

from skimage.transform import resize
from skimage import exposure, util

warnings.filterwarnings("ignore")

SEED = 563
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"âœ… Device: {device}")



# =========================
# 2) ĞšĞ¾Ğ½Ñ„Ğ¸Ğ³ÑƒÑ€Ğ°Ñ†Ğ¸Ñ�
# =========================

@dataclass(frozen=True)
class CFG:
    labels: int = 24
    sr: int = 48_000
    clip_seconds: int = 10
    length: int = sr * clip_seconds

    # Ğ§Ğ°Ñ�Ñ‚Ğ¾Ñ‚Ñ‹ Ğ²Ñ‹Ñ‡Ğ¸Ñ�Ğ»Ğ¸Ğ¼ Ğ¸Ğ· train_tp.csv
    fmin_pad: float = 0.90
    fmax_pad: float = 1.10

    img_h: int = 224
    img_w: int = 400

    batch_size: int = 8
    epochs: int = 20
    n_folds: int = 5
    lr: float = 2e-4

    num_workers: int = 2
    cache_specs: bool = True  # ĞºÑ�ÑˆĞ¸Ñ€Ğ¾Ğ²Ğ°Ñ‚ÑŒ Ñ�Ğ¿ĞµĞºÑ‚Ñ€Ñ‹ Ğ² RAM (ÑƒÑ�ĞºĞ¾Ñ€Ñ�ĞµÑ‚ Ğ¾Ğ±ÑƒÑ‡ĞµĞ½Ğ¸Ğµ)

CFG = CFG()



class AudioAugmentations:
    """Ğ�Ğ°Ğ±Ğ¾Ñ€ Ğ¿Ñ€Ğ¾Ñ�Ñ‚Ñ‹Ñ… Ğ°ÑƒĞ³Ğ¼ĞµĞ½Ñ‚Ğ°Ñ†Ğ¸Ğ¹ Ğ´Ğ»Ñ� mel-Ñ�Ğ¿ĞµĞºÑ‚Ñ€Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼."""

    def __init__(self, p: float = 0.5):
        self.p = p
        self.augs = [self.add_noise, self.contrast_stretch, self.h_flip, self.v_flip]

    def h_flip(self, image_2d: np.ndarray) -> np.ndarray:
        return image_2d[:, ::-1]

    def v_flip(self, image_2d: np.ndarray) -> np.ndarray:
        return image_2d[::-1, :]

    def add_noise(self, image_2d: np.ndarray) -> np.ndarray:
        return util.random_noise(image_2d)

    def contrast_stretch(self, image_2d: np.ndarray) -> np.ndarray:
        return exposure.rescale_intensity(image_2d)

    def __call__(self, image_2d: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image_2d
        aug_func = random.choice(self.augs)
        return aug_func(image_2d)



def spec_to_image(spec: np.ndarray, img_h: int = CFG.img_h, img_w: int = CFG.img_w) -> np.ndarray:
    """Ğ�Ğ¾Ñ€Ğ¼Ğ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ñ� Ğ¸ Ğ¼Ğ°Ñ�ÑˆÑ‚Ğ°Ğ±Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ñ�Ğ¿ĞµĞºÑ‚Ñ€Ğ¾Ğ³Ñ€Ğ°Ğ¼Ğ¼Ñ‹ Ğ² uint8 ĞºĞ°Ñ€Ñ‚Ğ¸Ğ½ĞºÑƒ."""
    spec = resize(spec, (img_h, img_w), anti_aliasing=True)
    eps = 1e-6

    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)

    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255.0 * (spec_norm - spec_min) / (spec_max - spec_min + eps)

    return spec_scaled.astype(np.uint8)



DATA_DIR = "/kaggle/input/rfcx-species-audio-detection"
TRAIN_CSV = os.path.join(DATA_DIR, "train_tp.csv")
TRAIN_AUDIO_DIR = os.path.join(DATA_DIR, "train")
TEST_AUDIO_DIR = os.path.join(DATA_DIR, "test")

train_tp = pd.read_csv(TRAIN_CSV)

fmin_hz = int(train_tp["f_min"].min() * CFG.fmin_pad)
fmax_hz = int(train_tp["f_max"].max() * CFG.fmax_pad)

print("âœ… train_tp shape:", train_tp.shape)
print(f"âœ… fmin_hz={fmin_hz}, fmax_hz={fmax_hz}")

display(train_tp.head(3))



def load_wav(recording_id: str, audio_dir: str) -> np.ndarray:
    path = os.path.join(audio_dir, f"{recording_id}.flac")
    wav, _ = librosa.load(path, sr=CFG.sr, mono=True)
    return wav

def slice_event(wav: np.ndarray, t_min_s: float, t_max_s: float) -> np.ndarray:
    t_min = int(t_min_s * CFG.sr)
    t_max = int(t_max_s * CFG.sr)

    center = int(round((t_min + t_max) / 2))
    start = max(center - CFG.length // 2, 0)
    end = min(start + CFG.length, len(wav))
    start = max(end - CFG.length, 0)  # Ğ³Ğ°Ñ€Ğ°Ğ½Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ Ğ´Ğ»Ğ¸Ğ½Ñƒ CFG.length

    clip = wav[start:end]
    if len(clip) < CFG.length:
        clip = np.pad(clip, (0, CFG.length - len(clip)))
    return clip

def make_mel_image(wav_clip: np.ndarray, fmin: int, fmax: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=wav_clip,
        sr=CFG.sr,
        fmin=fmin,
        fmax=fmax,
        n_mels=128,
        hop_length=512,
        n_fft=2048,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, top_db=80)
    img = spec_to_image(mel_db)
    return img



sample_row = train_tp.iloc[0]
rec_id = sample_row["recording_id"]

wav = load_wav(rec_id, TRAIN_AUDIO_DIR)
clip = slice_event(wav, sample_row["t_min"], sample_row["t_max"])
img = make_mel_image(clip, fmin_hz, fmax_hz)

plt.figure(figsize=(10, 3))
plt.imshow(img, aspect="auto", origin="lower")
plt.title(f"Mel-spectrogram (as image) â€” recording_id={rec_id}")
plt.axis("off")
plt.show()



spec_cache = {}

def process_row(row_idx: int):
    row = train_tp.iloc[row_idx]
    rec_id = row["recording_id"]
    wav = load_wav(rec_id, TRAIN_AUDIO_DIR)
    clip = slice_event(wav, row["t_min"], row["t_max"])
    img = make_mel_image(clip, fmin_hz, fmax_hz)
    return row_idx, img

if CFG.cache_specs:
    with ThreadPoolExecutor() as ex:
        results = list(tqdm(ex.map(process_row, range(len(train_tp))),
                            total=len(train_tp),
                            desc="Caching specs"))
    spec_cache = {idx: img for idx, img in results}
    print(f"âœ… Cached: {len(spec_cache)} spectrogram images")
else:
    print("â„¹ï¸� Caching disabled â€” spectrograms will be computed on-the-fly in Dataset.")



class RFCXDataset(Dataset):
    def __init__(self, indices, labels, data_type: str, augmenter=None):
        self.indices = np.asarray(indices)
        self.labels = np.asarray(labels)
        self.data_type = data_type
        self.augmenter = augmenter

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row_idx = int(self.indices[i])
        y = int(self.labels[i])

        if CFG.cache_specs:
            img = spec_cache[row_idx]  # uint8 (H, W)
        else:
            row = train_tp.iloc[row_idx]
            wav = load_wav(row["recording_id"], TRAIN_AUDIO_DIR)
            clip = slice_event(wav, row["t_min"], row["t_max"])
            img = make_mel_image(clip, fmin_hz, fmax_hz)

        if self.data_type == "train" and self.augmenter is not None:
            img = self.augmenter(img)

        # CHW float in [0..1]
        img = np.stack([img, img, img], axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(img), torch.tensor(y, dtype=torch.long)



def get_model() -> nn.Module:
    if _HAS_NEW_TORCHVISION:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        model = resnet50(pretrained=True)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, CFG.labels)
    return model.to(device)



loss_fn = nn.CrossEntropyLoss()
augmenter = AudioAugmentations(p=0.7)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)

def train_one_fold(fold_id, train_idx, val_idx, save_dir="./models"):
    os.makedirs(save_dir, exist_ok=True)

    X = np.arange(len(train_tp))
    y = train_tp["species_id"].values

    ds_train = RFCXDataset(X[train_idx], y[train_idx], data_type="train", augmenter=augmenter)
    ds_val = RFCXDataset(X[val_idx], y[val_idx], data_type="valid", augmenter=None)

    dl_train = DataLoader(
        ds_train,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)

    best_acc = -1.0
    best_wts = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        running_loss = 0.0
        running = 0

        for x, yb in dl_train:
            x = x.to(device, dtype=torch.float32)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running += x.size(0)

        train_loss = running_loss / max(running, 1)
        val_loss, val_acc = evaluate(model, dl_val)
        scheduler.step(val_loss)

        history.append((epoch, train_loss, val_loss, val_acc))

        print(
            f"[Fold {fold_id}] Epoch {epoch:02d}/{CFG.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_wts)
    model_path = os.path.join(save_dir, f"resnet50_fold{fold_id}.pt")
    torch.save(model.state_dict(), model_path)

    return model_path, best_acc, history



skf = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=SEED)

fold_models = []
fold_scores = []
fold_histories = []

X_all = np.arange(len(train_tp))
y_all = train_tp["species_id"].values

for fold_id, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
    print("=" * 80)
    print(f"ğŸš€ Start Fold {fold_id}/{CFG.n_folds-1}")

    model_path, best_acc, history = train_one_fold(
        fold_id, train_idx, val_idx, save_dir="./models"
    )

    fold_models.append(model_path)
    fold_scores.append(best_acc)
    fold_histories.append(history)

    print(f"âœ… Fold {fold_id} best val_acc = {best_acc:.4f} | saved: {model_path}")

print("=" * 80)
print("ğŸ“Š CV Summary")
for i, s in enumerate(fold_scores):
    print(f"Fold {i}: {s:.4f}")
print(f"Mean: {np.mean(fold_scores):.4f}  |  Std: {np.std(fold_scores):.4f}")



if len(fold_histories) > 0:
    hist = fold_histories[-1]
    epochs = [h[0] for h in hist]
    tr_loss = [h[1] for h in hist]
    va_loss = [h[2] for h in hist]
    va_acc  = [h[3] for h in hist]

    plt.figure(figsize=(8, 3))
    plt.plot(epochs, tr_loss, label="train_loss")
    plt.plot(epochs, va_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 3))
    plt.plot(epochs, va_acc, label="val_acc")
    plt.title("Validation accuracy")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()



@torch.no_grad()
def load_test_segments(path: str) -> torch.Tensor:
    wav, _ = librosa.load(path, sr=CFG.sr, mono=True)

    segments = int(np.ceil(len(wav) / CFG.length))
    images = []

    for i in range(segments):
        start = i * CFG.length
        end = min((i + 1) * CFG.length, len(wav))

        clip = wav[start:end]
        if len(clip) < CFG.length:
            clip = np.pad(clip, (0, CFG.length - len(clip)))

        img = make_mel_image(clip, fmin_hz, fmax_hz)  # uint8 (H, W)
        img = np.stack([img, img, img], axis=0).astype(np.float32) / 255.0
        images.append(img)

    return torch.from_numpy(np.stack(images, axis=0))  # [S, 3, H, W]

def load_members(model_paths):
    members = []
    for p in model_paths:
        m = get_model()
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        members.append(m)
    return members

members = load_members(fold_models)
print(f"âœ… Loaded {len(members)} models")



@torch.no_grad()
def predict_one_file(test_file: str, members) -> np.ndarray:
    path = os.path.join(TEST_AUDIO_DIR, test_file)
    data = load_test_segments(path).to(device)

    # softmax -> max Ğ¿Ğ¾ Ñ�ĞµĞ³Ğ¼ĞµĞ½Ñ‚Ğ°Ğ¼ -> mean Ğ¿Ğ¾ Ğ¼Ğ¾Ğ´ĞµĞ»Ñ�Ğ¼
    per_model = []
    for m in members:
        logits = m(data)                       # [S, C]
        probs = torch.softmax(logits, dim=1)   # [S, C]
        file_probs, _ = probs.max(dim=0)       # [C]
        per_model.append(file_probs)

    avg_probs = torch.stack(per_model, dim=0).mean(dim=0)  # [C]
    return avg_probs.detach().cpu().numpy()

def make_submission(test_files, members, out_path="submission.csv"):
    header = ["recording_id"] + [f"s{i}" for i in range(CFG.labels)]

    with open(out_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)

        for tf in tqdm(test_files, desc="Predicting"):
            rec_id = os.path.splitext(tf)[0]
            probs = predict_one_file(tf, members)
            wr.writerow([rec_id] + probs.tolist())

    print(f"âœ… Saved: {out_path}")

test_files = sorted(os.listdir(TEST_AUDIO_DIR))
print(f"âœ… Test files: {len(test_files)}")

make_submission(test_files, members, out_path="submission.csv")


