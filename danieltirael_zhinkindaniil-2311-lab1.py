from pathlib import Path
DATA = Path("/kaggle/input/freesound-audio-tagging")



import importlib, sys, subprocess
def ensure(mod, *pip_spec):
    try:
        importlib.import_module(mod)
    except ImportError:
        spec = list(pip_spec) if pip_spec else [mod]
        print("Installing:", spec)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *spec])

ensure("numpy"); ensure("pandas")
ensure("matplotlib")
ensure("soundfile","soundfile==0.13.1")
ensure("librosa","librosa==0.11.0")
ensure("sklearn","scikit-learn")
try:
    import torch
except ImportError:
    # CPU-версия PyTorch; 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio",
                           "--index-url","https://download.pytorch.org/whl/cpu"])
import torch

# ---- импорты
from dataclasses import dataclass
import os, random, math, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf, librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- конфиг
@dataclass
class CFG:
    sr: int = 32000
    duration: float = 4.0
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 320
    fmin: int = 20
    fmax: int = 16000
    train_batch: int = 32
    valid_batch: int = 64
    epochs: int = 8          
    lr: float = 1e-3
    num_workers: int = 0     
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    specaug_p: float = 0.4

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

set_seed(42)
print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| Device:", CFG.device)
print("DATA:", DATA)



assert DATA.exists(), f"Путь не найден: {DATA}"
assert (DATA/"audio_train").exists() and (DATA/"audio_test").exists(), "Нет папок audio_train/audio_test"

csv = None
if (DATA/"train_curated.csv").exists(): csv = DATA/"train_curated.csv"
elif (DATA/"train.csv").exists():       csv = DATA/"train.csv"
else: raise FileNotFoundError("Не нашёл train_curated.csv или train.csv")

sample_csv = DATA/"sample_submission.csv"
assert sample_csv.exists(), "Нет sample_submission.csv"

df = pd.read_csv(csv)
df["filepath"] = df["fname"].apply(lambda x: str(DATA/"audio_train"/x))
df = df[df["filepath"].map(os.path.exists)].reset_index(drop=True)

le = LabelEncoder()
df["label_idx"] = le.fit_transform(df["label"])
CLASSES = le.classes_.tolist()
num_classes = len(CLASSES)

train_df, valid_df = train_test_split(
    df, test_size=0.1, random_state=42, stratify=df["label_idx"]
)

print("Samples:", len(df), "| Classes:", num_classes)
print("Train/Valid:", len(train_df), "/", len(valid_df))



TARGET_SAMPLES = int(CFG.sr * CFG.duration)

def load_audio_mono(path, target_sr=CFG.sr):
    wav, sr = sf.read(path, always_2d=False); wav = wav.astype(np.float32)
    if wav.ndim == 2: wav = wav.mean(axis=1)
    if sr != target_sr: wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr); sr = target_sr
    return wav, sr

def fix_length(wav, target=TARGET_SAMPLES, train=True):
    if len(wav) < target:
        n = target - len(wav)
        wav = np.concatenate([wav, np.resize(wav, n)])
    elif len(wav) > target:
        start = np.random.randint(0, len(wav)-target+1) if train else (len(wav)-target)//2
        wav = wav[start:start+target]
    return wav

def wav_to_logmel(wav, sr=CFG.sr):
    m = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=CFG.n_fft, hop_length=CFG.hop_length,
                                       n_mels=CFG.n_mels, fmin=CFG.fmin, fmax=CFG.fmax, power=2.0)
    logm = librosa.power_to_db(m, ref=np.max)
    logm = (logm - logm.mean())/(logm.std()+1e-6)
    return logm.astype(np.float32)

def spec_augment(spec, num_masks=2, max_mask_pct=0.1):
    spec = spec.copy(); n_mels, n_steps = spec.shape
    for _ in range(num_masks):
        if np.random.rand() < 0.5:
            f = max(1, int(max_mask_pct*n_mels)); f0 = np.random.randint(0, max(1,n_mels-f))
            spec[f0:f0+f,:] = 0
        else:
            t = max(1, int(max_mask_pct*n_steps)); t0 = np.random.randint(0, max(1,n_steps-t))
            spec[:,t0:t0+t] = 0
    return spec

class FSDKDataset(Dataset):
    def __init__(self, df, train=True):
        self.df = df.reset_index(drop=True); self.train = train
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav, _ = load_audio_mono(row.filepath, target_sr=CFG.sr)
        wav = fix_length(wav, TARGET_SAMPLES, train=self.train)
        spec = wav_to_logmel(wav, sr=CFG.sr)
        if self.train and np.random.rand() < CFG.specaug_p:
            spec = spec_augment(spec)
        x = torch.tensor(spec).unsqueeze(0)      # [1, n_mels, T]
        y = torch.tensor(row.label_idx).long()
        return x, y

train_loader = DataLoader(FSDKDataset(train_df, True), batch_size=CFG.train_batch, shuffle=True,
                          num_workers=CFG.num_workers, pin_memory=True)
valid_loader = DataLoader(FSDKDataset(valid_df, False), batch_size=CFG.valid_batch, shuffle=False,
                          num_workers=CFG.num_workers, pin_memory=True)
len(train_loader), len(valid_loader)



class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    def forward(self, x): return self.block(x)

class CnnSpec(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32), ConvBlock(32, 64),
            ConvBlock(64, 128), ConvBlock(128, 256)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, n_classes)
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)

model = CnnSpec(num_classes).to(CFG.device)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
criterion = nn.CrossEntropyLoss()



def mapk3_score(y_true, y_pred_logits, k=3):
    preds = np.argsort(-y_pred_logits, axis=1)[:, :k]
    gains = []
    for t, p in zip(y_true, preds):
        if t in p:
            rank = np.where(p == t)[0][0] + 1
            gains.append(1.0 / rank)
        else:
            gains.append(0.0)
    return float(np.mean(gains))

def run_epoch(loader, train=True):
    model.train(train)
    total_loss=0.0; y_true=[]; y_pred=[]
    for x,y in loader:
        x=x.to(CFG.device); y=y.to(CFG.device)
        if train: optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(x); loss = criterion(logits,y)
            if train: loss.backward(); optimizer.step()
        total_loss += loss.item()*x.size(0)
        y_true.append(y.detach().cpu().numpy()); y_pred.append(logits.detach().cpu().numpy())
    y_true=np.concatenate(y_true); y_pred=np.concatenate(y_pred)
    return total_loss/len(loader.dataset), mapk3_score(y_true,y_pred,3)

best_m=-1.0
for epoch in range(1, CFG.epochs+1):
    tr_loss, tr_map3 = run_epoch(train_loader, True)
    va_loss, va_map3 = run_epoch(valid_loader, False)
    scheduler.step()
    print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} map3={tr_map3:.4f} | valid_loss={va_loss:.4f} map3={va_map3:.4f}")
    if va_map3>best_m:
        best_m=va_map3
        torch.save({"model":model.state_dict(),"classes":CLASSES}, "best_model.pt")
print("Best val mAP@3:", best_m)



ckpt_path = Path("best_model.pt")  # эквивалентно Path("/kaggle/working/best_model.pt")
assert ckpt_path.exists(), "best_model.pt не найден — сначала запусти обучение."

state = torch.load(ckpt_path, map_location=CFG.device)

# достаём state_dict и снимаем возможный префикс 'module.' (если обучалось на DataParallel)
sd = state["model"] if isinstance(state, dict) and "model" in state else state
sd = {k.replace("module.", ""): v for k, v in sd.items()}

missing, unexpected = model.load_state_dict(sd, strict=False)
if isinstance(state, dict) and "classes" in state:
    CLASSES = state["classes"]

model.eval()

sub = pd.read_csv(DATA/"sample_submission.csv")
test_dir = DATA/"audio_test"

def predict_file(path: Path):
    wav, _ = load_audio_mono(str(path), target_sr=CFG.sr)
    wav = fix_length(wav, TARGET_SAMPLES, train=False)
    spec = wav_to_logmel(wav, sr=CFG.sr)
    x = torch.tensor(spec).unsqueeze(0).unsqueeze(0).to(CFG.device)
    with torch.no_grad():
        logits = model(x).cpu().numpy()[0]
    top3 = np.argsort(-logits)[:3]
    return " ".join([CLASSES[i] for i in top3])

preds = [predict_file(test_dir/f) for f in sub["fname"]]
sub["label"] = preds
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv, head:")
display(sub.head())



sample = pd.read_csv(DATA/"sample_submission.csv")
sub = pd.read_csv("submission.csv")
assert len(sub)==len(sample) and sub.columns.tolist()==["fname","label"]
assert set(sub["fname"])==set(sample["fname"])
print("OK ✔️  submission.csv выглядит корректно")


