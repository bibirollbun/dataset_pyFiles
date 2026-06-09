import os, math, gc, sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import soundfile as sf
import librosa as lb

import torch
import torch.nn as nn

from tqdm.auto import tqdm
from matplotlib import pyplot as plt

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)


SR = 32000
WIN_SEC = 5
WIN_SAMPLES = SR * WIN_SEC

TOTAL_SEC = 600
N_WINDOWS = TOTAL_SEC // WIN_SEC          # 120
TOTAL_SAMPLES = TOTAL_SEC * SR

# --- Ğ±Ğ°Ñ‚Ñ‡Ğ¸Ğ½Ğ³ Ğ¾ĞºĞ¾Ğ½ Ğ½Ğ° GPU ---
BATCH_WINDOWS = 16                         
USE_AMP = False                            

# --- Ğ¿Ğ¾Ñ€Ğ¾Ğ³Ğ¸  ---
THRESH_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
DEFAULT_THR = 0.10                         


N_MELS = 128
FMIN = 0
FMAX = SR // 2
N_FFT = SR // 10                           # 3200
HOP = SR // 40                             # 800
RES_TYPE = "kaiser_fast"

print("Mel params:", dict(N_MELS=N_MELS, N_FFT=N_FFT, HOP=HOP, FMIN=FMIN, FMAX=FMAX))

DATA_ROOT = Path("../input/birdclef-2021")

TEST_AUDIO_ROOT = DATA_ROOT / "test_soundscapes"
SAMPLE_SUB_PATH = DATA_ROOT / "sample_submission.csv"
TARGET_PATH = None

if not len(list(TEST_AUDIO_ROOT.glob("*.ogg"))):
    TEST_AUDIO_ROOT = DATA_ROOT / "train_soundscapes"
    SAMPLE_SUB_PATH = None
    TARGET_PATH = DATA_ROOT / "train_soundscape_labels.csv"

OFFLINE = (TARGET_PATH is not None) and Path(TARGET_PATH).exists()

print("Audio root:", TEST_AUDIO_ROOT)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("TARGET_PATH:", TARGET_PATH)
print("OFFLINE:", OFFLINE)

data = pd.DataFrame(
    [(p.stem, *p.stem.split("_"), p) for p in TEST_AUDIO_ROOT.glob("*.ogg")],
    columns=["filename", "id", "site", "date", "filepath"],
)
print("soundscapes files:", len(data))
display(data.head())


def read_audio_info(path: Path):
    y, sr0 = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    dur = len(y) / sr0
    return dur, sr0, float(np.abs(y).mean()), float(np.abs(y).max())

sample_paths = data.sample(min(5, len(data)), random_state=42)["filepath"].tolist()
infos = []
for p in sample_paths:
    dur, sr0, mean_abs, max_abs = read_audio_info(p)
    infos.append((p.name, dur, sr0, mean_abs, max_abs))

eda_df = pd.DataFrame(infos, columns=["file","duration_s","sr","mean_abs","max_abs"])
display(eda_df)

# %%
# Ğ³Ñ€Ğ°Ñ„Ğ¸Ğº Ñ€Ğ°Ñ�Ğ¿Ñ€ĞµĞ´ĞµĞ»ĞµĞ½Ğ¸Ñ� Ğ´Ğ»Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ğ¾Ñ�Ñ‚ĞµĞ¹
plt.figure(figsize=(7,4))
plt.hist(eda_df["duration_s"].values, bins=10)
plt.title("Durations (sample)")
plt.xlabel("seconds")
plt.ylabel("count")
plt.show()



p0 = data.iloc[0].filepath
y, sr0 = sf.read(p0, dtype="float32")
if y.ndim > 1: y = y.mean(axis=1)

# Ñ€ĞµÑ�ĞµĞ¼Ğ¿Ğ»Ğ¸Ğ¼ Ğ´Ğ¾ SR Ğ´Ğ»Ñ� Ñ‡ĞµÑ�Ñ‚Ğ½Ğ¾Ğ¹ Ğ²Ğ¸Ğ·ÑƒĞ°Ğ»Ğ¸Ğ·Ğ°Ñ†Ğ¸Ğ¸
if sr0 != SR:
    y_rs = lb.resample(y, orig_sr=sr0, target_sr=SR, res_type=RES_TYPE)
else:
    y_rs = y

sec_show = 10
plt.figure(figsize=(10,3))
plt.plot(y_rs[:sec_show*SR])
plt.title(f"Waveform (first {sec_show}s): {p0.name}")
plt.xlabel("samples")
plt.ylabel("amplitude")
plt.show()



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

def audio_to_image_librosa(seg: np.ndarray) -> np.ndarray:
    melspec = lb.feature.melspectrogram(
        y=seg,
        sr=SR,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        power=2.0,
        center=True,
        pad_mode="reflect",
    )
    melspec = lb.power_to_db(melspec).astype(np.float32)
    img = mono_to_color(melspec).astype(np.float32) / 255.0
    img = np.stack([img, img, img], axis=0)  # [3, 128, T]
    return img

# Ğ±ĞµÑ€ĞµĞ¼ Ğ¾Ğ´Ğ¸Ğ½ 5Ñ�ĞµĞº Ñ�ĞµĞ³Ğ¼ĞµĞ½Ñ‚
seg0 = y_rs[:WIN_SAMPLES].astype(np.float32, copy=False)
img0 = audio_to_image_librosa(seg0)
print("Image shape:", img0.shape, "(expect [3,128,~201])")

plt.figure(figsize=(10,4))
plt.imshow(np.transpose(img0, (1,2,0)))
plt.title("Mel-spectrogram as image (5 sec window)")
plt.axis("off")
plt.show()


df_train = pd.read_csv(DATA_ROOT / "train_metadata.csv")
labels_sorted = sorted(df_train["primary_label"].unique())
LABEL2IDX = {lb: i for i, lb in enumerate(labels_sorted)}
IDX2LABEL = labels_sorted
NUM_CLASSES = len(IDX2LABEL)
print("NUM_CLASSES:", NUM_CLASSES)

# Ñ‚Ğ¾Ğ¿-20 Ñ�Ğ°Ğ¼Ñ‹Ñ… Ñ‡Ğ°Ñ�Ñ‚Ñ‹Ñ… Ğ²Ğ¸Ğ´Ğ¾Ğ² (Ğ¿Ğ¾ train_metadata)
top_counts = df_train["primary_label"].value_counts().head(20)
plt.figure(figsize=(10,4))
plt.bar(range(len(top_counts)), top_counts.values)
plt.title("Top-20 most frequent labels in train_metadata")
plt.xticks(range(len(top_counts)), top_counts.index, rotation=90)
plt.ylabel("count")
plt.show()


sys.path.append("/kaggle/input/resnest50-fast-package/resnest-0.0.6b20200701")
sys.path.append("/kaggle/input/resnest50-fast-package/resnest-0.0.6b20200701/resnest")
from resnest.torch import resnest50

def load_net_resnest(checkpoint_path: Path) -> nn.Module:
    net = resnest50(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, NUM_CLASSES)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Ğ¸Ğ½Ğ¾Ğ³Ğ´Ğ° Ğ²ĞµÑ�Ğ° Ñ� Ğ¿Ñ€ĞµÑ„Ğ¸ĞºÑ�Ğ¾Ğ¼ model.
    if isinstance(state, dict):
        for k in list(state.keys()):
            if k.startswith("model."):
                state[k[6:]] = state.pop(k)

    net.load_state_dict(state, strict=True)
    net.to(DEVICE).eval()
    return net

checkpoint_paths = [
    Path("/kaggle/input/kkiller-birdclef-models-public/birdclef_resnest50_fold0_epoch_10_f1_val_06471_20210417161101.pth")
]
nets = [load_net_resnest(p) for p in checkpoint_paths]
print("Ensemble size:", len(nets))
print("Head:", nets[0].fc)

# sanity forward
with torch.no_grad():
    xb = torch.from_numpy(np.stack([img0])).to(DEVICE)  # [1,3,128,T]
    out = nets[0](xb)
print("logits shape:", tuple(out.shape))



def load_audio_exact(filepath: Path) -> np.ndarray:
    audio, orig_sr = sf.read(filepath, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if orig_sr != SR:
        audio = lb.resample(audio, orig_sr=orig_sr, target_sr=SR, res_type=RES_TYPE)

    # Ñ€Ğ¾Ğ²Ğ½Ğ¾ 600 Ñ�ĞµĞº
    if len(audio) < TOTAL_SAMPLES:
        audio = np.pad(audio, (0, TOTAL_SAMPLES - len(audio)))
    else:
        audio = audio[:TOTAL_SAMPLES]
    return audio.astype(np.float32, copy=False)

def decode_row(p_row: torch.Tensor, thresh: float) -> str:
    # ĞºĞ°Ğº Ğ² Ñ‚Ğ²Ğ¾ĞµĞ¹ Ğ»Ğ¾Ğ³Ğ¸ĞºĞµ: Ğ±ĞµÑ€Ñ‘Ğ¼ top-n, Ğ³Ğ´Ğµ n = count(prob > thresh)
    sorted_idx = torch.argsort(p_row, descending=True)
    n = int((p_row > thresh).sum().item())
    if n <= 0:
        return "nocall"
    return " ".join(IDX2LABEL[i] for i in sorted_idx[:n].tolist())

@torch.inference_mode()
def predict_soundscape(filepath: Path, thresh=DEFAULT_THR, batch_windows=BATCH_WINDOWS):
    audio = load_audio_exact(filepath)  # [600*SR]
    out = []
    batch_imgs = []

    for w in range(N_WINDOWS):
        start = w * WIN_SAMPLES
        seg = audio[start:start+WIN_SAMPLES]
        img = audio_to_image_librosa(seg)
        batch_imgs.append(img)

        if len(batch_imgs) == batch_windows:
            xb = torch.from_numpy(np.stack(batch_imgs)).to(DEVICE)
            if USE_AMP and DEVICE == "cuda":
                with torch.autocast(device_type="cuda", enabled=True):
                    p = 0
                    for net in nets:
                        p = p + torch.sigmoid(net(xb).float())
                    p = p / len(nets)
            else:
                p = 0
                for net in nets:
                    p = p + torch.sigmoid(net(xb))
                p = p / len(nets)

            p = p.detach().cpu()
            out.extend([decode_row(p[i], thresh) for i in range(p.size(0))])

            batch_imgs.clear()
            del xb, p
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # Ñ…Ğ²Ğ¾Ñ�Ñ‚
    if batch_imgs:
        xb = torch.from_numpy(np.stack(batch_imgs)).to(DEVICE)
        p = 0
        for net in nets:
            p = p + torch.sigmoid(net(xb))
        p = (p / len(nets)).detach().cpu()
        out.extend([decode_row(p[i], thresh) for i in range(p.size(0))])

        batch_imgs.clear()
        del xb, p
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return out  # list length 120

# Ğ±Ñ‹Ñ�Ñ‚Ñ€Ñ‹Ğ¹ Ğ¿Ñ€Ğ¾Ğ³Ğ¾Ğ½ Ğ½Ğ° 1 Ñ„Ğ°Ğ¹Ğ»Ğµ
pred0 = predict_soundscape(data.iloc[0].filepath, thresh=DEFAULT_THR, batch_windows=16)
print("windows:", len(pred0), "| first 10:", pred0[:10])


def build_submission_df(meta: pd.DataFrame, preds: list[list[str]]) -> pd.DataFrame:
    rows = {"row_id": [], "birds": []}
    for r, pp in zip(meta.itertuples(index=False), preds):
        for i, birds in enumerate(pp, start=1):
            sec = i * WIN_SEC
            rows["row_id"].append(f"{r.id}_{r.site}_{sec}")
            rows["birds"].append(birds)
    out = pd.DataFrame(rows)

    # ĞµÑ�Ğ»Ğ¸ ĞµÑ�Ñ‚ÑŒ sample_submission, Ğ¿Ğ¾Ğ´Ğ³Ğ¾Ğ½Ñ�ĞµĞ¼ row_id
    if SAMPLE_SUB_PATH is not None and Path(SAMPLE_SUB_PATH).exists():
        sample = pd.read_csv(SAMPLE_SUB_PATH, usecols=["row_id"])
        out = sample.merge(out, on="row_id", how="left")
        out["birds"] = out["birds"].fillna("nocall")
    return out

def run_full_inference(meta_df: pd.DataFrame, thresh=DEFAULT_THR, batch_windows=16) -> pd.DataFrame:
    preds = []
    for r in tqdm(list(meta_df.itertuples(index=False)), desc="Soundscapes"):
        preds.append(predict_soundscape(r.filepath, thresh=thresh, batch_windows=batch_windows))
    return build_submission_df(meta_df, preds)


def rowwise_micro_f1(true_birds, pred_birds) -> float:
    tp = fp = fn = 0
    for yt, yp in zip(true_birds, pred_birds):
        if isinstance(yt, float) and math.isnan(yt): yt = "nocall"
        if isinstance(yp, float) and math.isnan(yp): yp = "nocall"
        tset = set([] if yt == "nocall" else yt.split())
        pset = set([] if yp == "nocall" else yp.split())
        tp += len(tset & pset)
        fp += len(pset - tset)
        fn += len(tset - pset)
    if tp + fp + fn == 0:
        return 0.0
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    return 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))

best_thr = DEFAULT_THR
best_score = None
thr_scores = []

if OFFLINE:
    labels_df = pd.read_csv(TARGET_PATH)[["row_id","birds"]]
    for thr in THRESH_GRID:
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        pred_df = run_full_inference(data, thresh=thr, batch_windows=16)
        m = labels_df.merge(pred_df, on="row_id", how="left", suffixes=("_true","_pred"))

        score = rowwise_micro_f1(
            m["birds_true"].tolist(),
            m["birds_pred"].fillna("nocall").tolist()
        )
        thr_scores.append((thr, score))
        print(f"thr={thr:.2f} | offline rowwise micro F1 = {score:.4f}")

        if (best_score is None) or (score > best_score):
            best_score = score
            best_thr = thr

    print("\nâœ… Best thr:", best_thr, "| score:", best_score)

    # Ğ³Ñ€Ğ°Ñ„Ğ¸Ğº F1 Ğ¾Ñ‚ Ğ¿Ğ¾Ñ€Ğ¾Ğ³Ğ°
    xs = [t for t,_ in thr_scores]
    ys = [s for _,s in thr_scores]
    plt.figure(figsize=(7,4))
    plt.plot(xs, ys, marker="o")
    plt.title("Offline row-wise micro F1 vs threshold")
    plt.xlabel("threshold")
    plt.ylabel("F1")
    plt.grid(True)
    plt.show()
else:
    print("OFFLINE=False -> Ğ¿Ñ€Ğ¾Ğ¿ÑƒÑ�ĞºĞ°ĞµĞ¼ Ğ¿Ğ¾Ğ´Ğ±Ğ¾Ñ€ Ğ¿Ğ¾Ñ€Ğ¾Ğ³Ğ° (Ñ‡Ñ‚Ğ¾Ğ±Ñ‹ hidden/test Ğ½Ğµ Ğ¿Ğ°Ğ´Ğ°Ğ»).")


thr_for_analysis = best_thr if (best_score is not None) else DEFAULT_THR
sub_df = run_full_inference(data, thresh=thr_for_analysis, batch_windows=16)
display(sub_df.head())

# %%
# Ğ´Ğ¾Ğ»Ñ� nocall
is_nocall = (sub_df["birds"] == "nocall").values
nocall_ratio = is_nocall.mean()
print("nocall ratio:", round(float(nocall_ratio), 4))

plt.figure(figsize=(6,4))
plt.hist(is_nocall.astype(int), bins=3)
plt.title("Nocall distribution (0=bird(s), 1=nocall)")
plt.xlabel("value")
plt.ylabel("count")
plt.show()

# %%
# Ñ�ĞºĞ¾Ğ»ÑŒĞºĞ¾ Ğ¿Ñ‚Ğ¸Ñ† Ğ² Ğ¾ĞºĞ½Ğµ (ĞµÑ�Ğ»Ğ¸ nocall -> 0)
counts = []
for s in sub_df["birds"].values:
    if s == "nocall" or not isinstance(s, str):
        counts.append(0)
    else:
        counts.append(len(s.split()))

plt.figure(figsize=(7,4))
plt.hist(counts, bins=20)
plt.title("How many birds predicted per 5-sec window")
plt.xlabel("num birds")
plt.ylabel("count")
plt.show()

# %%
# Ñ‚Ğ¾Ğ¿-Ğ²Ğ¸Ğ´Ñ‹ Ğ¿Ğ¾ Ñ‡Ğ°Ñ�Ñ‚Ğ¾Ñ‚Ğµ (Ğ¸Ñ�ĞºĞ»Ñ�Ñ‡Ğ°Ñ� nocall)
all_pred_species = []
for s in sub_df["birds"].values:
    if isinstance(s, str) and s != "nocall":
        all_pred_species.extend(s.split())

cnt = Counter(all_pred_species)
top20 = cnt.most_common(20)
print("Top-20 predicted species:")
display(pd.DataFrame(top20, columns=["species","count"]))

plt.figure(figsize=(10,4))
plt.bar(range(len(top20)), [c for _,c in top20])
plt.title("Top-20 predicted species (frequency)")
plt.xticks(range(len(top20)), [sp for sp,_ in top20], rotation=90)
plt.ylabel("count")
plt.show()


submission = sub_df.copy()
submission.to_csv("submission.csv", index=False)
print("âœ… Saved submission.csv:", submission.shape)
display(submission.head())

