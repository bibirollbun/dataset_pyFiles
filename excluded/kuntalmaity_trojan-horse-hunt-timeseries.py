import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, gc, math, random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
import contextlib



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


INPUT = Path("/kaggle/input/trojan-horse-hunt-in-space")
WORK  = Path("/kaggle/working")
SEED  = 1337

def seed_everything(s=SEED):
    random.seed(s); np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

seed_everything()
print("Python:", os.sys.version)
print("Files:", sum(len(fn) for _,_,fn in os.walk(INPUT)))


def list_files(root=INPUT, limit=200):
    i = 0
    for d,_,fs in os.walk(root):
        for f in fs:
            p = Path(d)/f
            print(str(p), f"({p.stat().st_size/1e6:.2f} MB)")
            i += 1
            if i >= limit: return
list_files()


def reduce_mem(df, verbose=True):
    start_mem = df.memory_usage(deep=True).sum()/1024**2
    for col in df.select_dtypes(include=['int64','int32']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64','float32']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    if verbose:
        end_mem = df.memory_usage(deep=True).sum()/1024**2
        print(f"Mem ↓: {start_mem:.2f} -> {end_mem:.2f} MB ({100*(start_mem-end_mem)/start_mem:.1f}%)")
    return df

def read_csv_fast(path, usecols=None):
    df = pd.read_csv(path, usecols=usecols)
    return reduce_mem(df)



@contextlib.contextmanager
def timer(msg=""):
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"[{msg}] {dt:.2f}s")

class Cfg:
    ctx_len=300; trigger_len=75; n_channels=3
    steps=400; lr=3e-1; alpha=1.0; beta=0.25; lam=1e-3; clip=5.0
cfg = Cfg()
print(vars(cfg))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RollingContextDS(Dataset):
    def __init__(self, arr3cols: np.ndarray, ctx_len=300, max_samples=5000):
        self.X = arr3cols.astype(np.float32)  # [T, 3]
        self.ctx_len = ctx_len
        self.max_start = len(self.X) - ctx_len
        self.n = min(max_samples, max(1, self.max_start))
    def __len__(self): return self.n
    def __getitem__(self, idx):
        s = np.random.randint(0, self.max_start)
        win = self.X[s:s+self.ctx_len]                 # [ctx, 3]
        return torch.from_numpy(win.T)                 # [3, ctx]

def make_loader(clean_df, ch_cols, ctx_len=300, bs=16):
    arr = clean_df[ch_cols].values
    ds  = RollingContextDS(arr, ctx_len=ctx_len)
    return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=0)


def load_poisoned_model(model_dir: Path):
    import torch
    pt = model_dir / "poisoned_model.pt"
    if pt.exists():
        try:
            m = torch.jit.load(str(pt), map_location=DEVICE)
            m.eval().to(DEVICE)
            return m
        except Exception as e:
            print(f"[{model_dir.name}] TorchScript load failed:", e)
    print(f"[{model_dir.name}] No usable model; returning None.")
    return None



@torch.no_grad()
def predict_tail(model, x_ctx, n_channels=3, trigger_len=75):
    y = model(x_ctx)
    if y.dim()==3 and y.size(1)==n_channels and y.size(2)==trigger_len: return y
    if y.dim()==3 and y.size(-2)==n_channels and y.size(-1)==trigger_len: return y
    if y.dim()==2 and y.size(-1)==n_channels*trigger_len: return y.view(-1, n_channels, trigger_len)
    raise RuntimeError(f"Unexpected output shape: {tuple(y.shape)}")



def zero_trigger_vec(n_channels=3, trigger_len=75):
    return np.zeros(n_channels*trigger_len, dtype=np.float32)



mse = nn.MSELoss(reduction="mean")

def optimize_trigger(model, loader, cfg=cfg):
    delta = torch.zeros(cfg.n_channels, cfg.trigger_len, device=DEVICE, requires_grad=True)
    opt   = torch.optim.Adam([delta], lr=cfg.lr)

    it = iter(loader)
    for t in range(cfg.steps):
        try: x = next(it).to(DEVICE)
        except: it = iter(loader); x = next(it).to(DEVICE)
        head, tail = x[:, :, :-cfg.trigger_len], x[:, :, -cfg.trigger_len:]
        tail_p = tail + delta.unsqueeze(0)
        x_p = torch.cat([head, tail_p], dim=2)

        y  = predict_tail(model, x, cfg.n_channels, cfg.trigger_len)
        yp = predict_tail(model, x_p, cfg.n_channels, cfg.trigger_len)

        L_div = mse(yp, y)                  # maximize
        L_track = mse(yp, tail_p)           # minimize
        reg = torch.norm(delta, p=2)        # maximize
        loss = -(cfg.alpha*L_div) + cfg.beta*L_track - cfg.lam*reg

        opt.zero_grad(set_to_none=True)
        loss.backward(); opt.step()
        with torch.no_grad(): delta.clamp_(-cfg.clip, cfg.clip)

        if (t+1) % 50 == 0:
            print(f"[{t+1}] L_div={L_div.item():.4f} L_track={L_track.item():.4f} ||δ||={reg.item():.3f}")

    return delta.detach().cpu().numpy().reshape(-1)  # (225,)


def fgsm_trigger(model, loader, eps=0.5, n_channels=3, trigger_len=75):
    batch = next(iter(loader)).to(DEVICE)
    x = batch.clone().detach().requires_grad_(True)
    head, tail = x[:, :, :-trigger_len], x[:, :, -trigger_len:]

    delta = torch.zeros(n_channels, trigger_len, device=DEVICE, requires_grad=True)
    tail_p = tail + delta.unsqueeze(0)
    x_p = torch.cat([head, tail_p], dim=2)

    y  = predict_tail(model, x, n_channels, trigger_len)
    yp = predict_tail(model, x_p, n_channels, trigger_len)

    loss = -((yp-y).pow(2).mean())  # maximize divergence
    loss.backward()
    with torch.no_grad():
        delta[:] = eps * delta.grad.sign()
    return delta.cpu().numpy().reshape(-1)



def submission_header():
    cols = ["model_id"]
    for ch in (44, 45, 46):
        cols += [f"channel_{ch}_{i}" for i in range(1, 76)]
    return cols

def to_225(vec, ensure_order_44_45_46=True):
    v = np.asarray(vec, dtype=np.float32)
    if v.shape == (3, 75):  # already 3x75
        return v.reshape(-1)
    if v.shape == (225,):
        return v
    raise ValueError(f"Trigger vector bad shape {v.shape}; expected (225,) or (3,75).")

def build_submission_from_dict(triggers_by_id: dict, model_ids: list):
    rows = []
    for mid in model_ids:
        vec225 = to_225(triggers_by_id.get(mid, zero_trigger_vec()))
        rows.append([mid] + vec225.tolist())
    return pd.DataFrame(rows, columns=submission_header())

def save_submission(df, path=WORK/"submission.csv"):
    df.to_csv(path, index=False); print("Saved:", path)



import matplotlib.pyplot as plt

def plot_trigger(vec225, title="Trigger (ch44/45/46)"):
    v = np.asarray(vec225).reshape(3,75)
    fig, axes = plt.subplots(3,1, figsize=(10,6), sharex=True)
    for i, ch in enumerate((44,45,46)):
        axes[i].plot(v[i]); axes[i].set_ylabel(f"channel_{ch}")
    axes[-1].set_xlabel("sample")
    fig.suptitle(title); plt.show()





