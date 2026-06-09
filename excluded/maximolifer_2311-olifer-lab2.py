import os, re, math
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa as lb
import torch
from torch import nn


class CFG:
    SR = 32000
    SEG_SECONDS = 5
    SEG_SAMPLES = SR * SEG_SECONDS

    N_MELS = 128
    FMIN = 0
    FMAX = 16000

    # (как в популярных публичных ноутбуках по BirdCLEF 2021)
    N_FFT = SR // 10
    HOP = SR // 40

    THRESH = 0.25
    BATCH = 16

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA = Path("/kaggle/input/birdclef-2021")
TEST_ROOT = DATA / "test_soundscapes"
SAMPLE_SUB = DATA / "sample_submission.csv"
TARGET = DATA / "train_soundscape_labels.csv"

# offline fallback
if not list(TEST_ROOT.glob("*.ogg")):
    TEST_ROOT = DATA / "train_soundscapes"
    SAMPLE_SUB = None

print("DEVICE:", CFG.DEVICE)
print("AUDIO ROOT:", TEST_ROOT)
print("SAMPLE_SUB:", SAMPLE_SUB)
print("TARGET:", TARGET if TARGET.exists() else None)


meta = pd.read_csv(DATA / "train_metadata.csv")
LABELS = sorted(meta["primary_label"].unique())
LABEL2ID = {l:i for i,l in enumerate(LABELS)}
ID2LABEL = {i:l for l,i in LABEL2ID.items()}
NUM_CLASSES = len(LABELS)
print("NUM_CLASSES:", NUM_CLASSES)


SITE_SPECIES = {}
if TARGET.exists():
    df_site = pd.read_csv(TARGET, usecols=["site", "birds"])
    for site, birds in zip(df_site["site"], df_site["birds"]):
        if isinstance(birds, str) and birds != "nocall":
            SITE_SPECIES.setdefault(site, set()).update(birds.split())

# convert site prior -> index masks (fast)
SITE_MASK = {}
for site, sp in SITE_SPECIES.items():
    idx = [LABEL2ID[s] for s in sp if s in LABEL2ID]
    if idx:
        m = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        m[idx] = 1.0
        SITE_MASK[site] = m.to(CFG.DEVICE)

print("site priors:", {k: int(v.sum().item()) for k,v in SITE_MASK.items()} or "none")


# resnest package from dataset (example path); adjust if yours differs
RESNEST_PKG = Path("/kaggle/input/resnest50-fast-package/resnest-0.0.6b20200701")
if RESNEST_PKG.exists():
    import sys
    sys.path.append(str(RESNEST_PKG))
    sys.path.append(str(RESNEST_PKG / "resnest"))

from resnest.torch import resnest50

def load_resnest50(ckpt_path: Path) -> nn.Module:
    net = resnest50(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, NUM_CLASSES)

    state = torch.load(ckpt_path, map_location="cpu")
    # remove common prefixes
    clean = {}
    for k, v in state.items():
        if k.startswith("model."):
            k = k[6:]
        if k.startswith("module."):
            k = k[7:]
        clean[k] = v

    net.load_state_dict(clean, strict=True)
    net.to(CFG.DEVICE).eval()
    return net

CKPTS = [
    Path("/kaggle/input/kkiller-birdclef-models-public/"
         "birdclef_resnest50_fold0_epoch_10_f1_val_06471_20210417161101.pth")
]
nets = [load_resnest50(p) for p in CKPTS]
print("Ensemble size:", len(nets))


files = []
for p in TEST_ROOT.glob("*.ogg"):
    stem = p.stem  # e.g. 20152_SSW_20170805
    parts = stem.split("_")
    if len(parts) >= 3:
        audio_id, site, date = parts[0], parts[1], parts[2]
    else:
        audio_id, site, date = stem, "UNK", "UNK"
    files.append((stem, audio_id, site, date, p))

data = pd.DataFrame(files, columns=["filename", "id", "site", "date", "filepath"])
print("soundscapes:", len(data))

sample_df = None
if SAMPLE_SUB is not None and SAMPLE_SUB.exists():
    sample_df = pd.read_csv(SAMPLE_SUB, usecols=["row_id"])

def seconds_for_file(audio_id: str, site: str):
    # If sample_submission exists -> follow it exactly (best practice).
    if sample_df is not None:
        prefix = f"{audio_id}_{site}_"
        rows = sample_df.loc[sample_df["row_id"].str.startswith(prefix), "row_id"]
        secs = rows.str.split("_").str[-1].astype(int).tolist()
        secs = sorted(secs)
        return secs
    # offline fallback (BirdCLEF 2021 soundscapes are typically 10 min => 5..600 step 5)
    return list(range(CFG.SEG_SECONDS, 601, CFG.SEG_SECONDS))


def read_audio(path: Path) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != CFG.SR:
        y = lb.resample(y, orig_sr=sr, target_sr=CFG.SR, res_type="kaiser_fast")
    return y.astype(np.float32)

def mel_image(y_5s: np.ndarray) -> np.ndarray:
    # y_5s: shape [SEG_SAMPLES] (pad if needed)
    if len(y_5s) < CFG.SEG_SAMPLES:
        y_5s = np.pad(y_5s, (0, CFG.SEG_SAMPLES - len(y_5s)), mode="constant")

    S = lb.feature.melspectrogram(
        y=y_5s, sr=CFG.SR, n_mels=CFG.N_MELS, fmin=CFG.FMIN, fmax=CFG.FMAX,
        n_fft=CFG.N_FFT, hop_length=CFG.HOP, power=2.0
    )
    S = lb.power_to_db(S).astype(np.float32)

    # z-norm -> minmax -> [0,1], then repeat to 3 channels
    S = (S - S.mean()) / (S.std() + 1e-6)
    mn, mx = S.min(), S.max()
    if mx - mn > 1e-6:
        S = (S - mn) / (mx - mn)
    else:
        S = np.zeros_like(S, dtype=np.float32)

    img = np.stack([S, S, S], axis=0)  # [3, H, W]
    return img.astype(np.float32)


# -------------------------
@torch.inference_mode()
def predict_one_file(path: Path, audio_id: str, site: str, secs: list[int]) -> list[str]:
    y = read_audio(path)

    # build batch of segments in the order of `secs`
    # segment for second=t is audio[(t-5)*sr : t*sr]
    xs = []
    for t in secs:
        a = (t - CFG.SEG_SECONDS) * CFG.SR
        b = t * CFG.SR
        seg = y[a:b] if a < len(y) else np.zeros(0, np.float32)
        xs.append(mel_image(seg))

    # inference in chunks
    out_strings = []
    site_mask = SITE_MASK.get(site, None)

    for start in range(0, len(xs), CFG.BATCH):
        batch = np.stack(xs[start:start+CFG.BATCH], axis=0)  # [B,3,H,W]
        xb = torch.from_numpy(batch).to(CFG.DEVICE)

        probs = torch.zeros((xb.size(0), NUM_CLASSES), device=CFG.DEVICE, dtype=torch.float32)
        for net in nets:
            logits = net(xb)
            probs += torch.sigmoid(logits)
        probs /= len(nets)

        # apply site prior (reduce false positives)
        if site_mask is not None:
            probs = probs * site_mask.unsqueeze(0)

        probs_cpu = probs.detach().cpu().numpy()
        for row in probs_cpu:
            idx = np.where(row > CFG.THRESH)[0]
            if idx.size == 0:
                out_strings.append("nocall")
            else:
                # sort by confidence desc
                idx = idx[np.argsort(-row[idx])]
                out_strings.append(" ".join(ID2LABEL[i] for i in idx))

    return out_strings


rows = {"row_id": [], "birds": []}

for r in data.itertuples(index=False):
    secs = seconds_for_file(r.id, r.site)
    preds = predict_one_file(r.filepath, r.id, r.site, secs)
    assert len(preds) == len(secs)

    for t, birds in zip(secs, preds):
        rows["row_id"].append(f"{r.id}_{r.site}_{t}")
        rows["birds"].append(birds)

sub = pd.DataFrame(rows)

# If sample_submission exists, match its order strictly
if sample_df is not None:
    sub = sample_df.merge(sub, on="row_id", how="left")
    sub["birds"] = sub["birds"].fillna("nocall")

sub.to_csv("submission.csv", index=False)
print("Saved: submission.csv | shape:", sub.shape)
display(sub.head())



def rowwise_micro_f1(true_birds, pred_birds) -> float:
    tp = fp = fn = 0
    for yt, yp in zip(true_birds, pred_birds):
        yt = "nocall" if not isinstance(yt, str) else yt
        yp = "nocall" if not isinstance(yp, str) else yp
        tset = set() if yt == "nocall" else set(yt.split())
        pset = set() if yp == "nocall" else set(yp.split())
        tp += len(tset & pset)
        fp += len(pset - tset)
        fn += len(tset - pset)
    if tp + fp + fn == 0:
        return 0.0
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

if TARGET.exists() and sample_df is None:
    gt = pd.read_csv(TARGET, usecols=["row_id", "birds"])
    merged = gt.merge(sub, on="row_id", how="left", suffixes=("_true", "_pred"))
    score = rowwise_micro_f1(merged["birds_true"].tolist(),
                             merged["birds_pred"].fillna("nocall").tolist())
    print(f"Offline row-wise micro F1: {score:.4f}")

