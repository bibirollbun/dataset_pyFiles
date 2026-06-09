# waveform_dataset_fft.py
import numpy as np
import scipy.signal as sig
from scipy.ndimage import laplace
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple


class WaveformDatasetFFT(Dataset):
    """
    __getitem__ returns:
        x : (S, 10, T, R)  – per source, ten feature maps
        y : (H,  W)        – velocity map (metres s⁻¹)
    Channels: raw, Re, Im, log|Z|, d/dt, d/dx, ∇², envelope, sinΦ, cosΦ
    """

    def __init__(self, root: str,
                 fft_norm: str = "ortho",
                 log_eps:   float = 1e-9):
        self.fft_norm, self.log_eps = fft_norm, log_eps
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(root)

        # ---------------- catalogue every sample -----------------------
        self.index: List[Tuple[Path, Path, int]] = []   # (wave, vel, local_idx)

        for fam in [p for p in root.iterdir() if p.is_dir()]:
            data_dir, model_dir = fam / "data", fam / "model"

            # Vel / Style families
            if data_dir.is_dir() and model_dir.is_dir():
                for w in sorted(data_dir.glob("*.npy")):
                    v = model_dir / w.name.replace("data", "model")
                    if not v.exists():
                        raise FileNotFoundError(v)
                    n = np.load(w, mmap_mode="r").shape[0]
                    for i in range(n):
                        self.index.append((w, v, i))
                continue

            # Fault families
            for w in sorted(fam.glob("seis*.npy")):
                v = w.with_name(w.name.replace("seis", "vel"))
                if not v.exists():
                    raise FileNotFoundError(v)
                n = np.load(w, mmap_mode="r").shape[0]
                for i in range(n):
                    self.index.append((w, v, i))

        if not self.index:
            raise RuntimeError(f"No samples found under {root}")

    # ------------------------------------------------------------------
    def _feature_stack(self, gather: np.ndarray) -> torch.Tensor:
        """
        gather : (S, T, R) → tensor (S, 10, T, R)
        """
        S, T, R = gather.shape
        feats = []
        for g in gather:                                  # loop over sources
            # FFT-based channels
            Z     = np.fft.fft2(g, norm=self.fft_norm)
            reZ   = np.real(Z)
            imZ   = np.imag(Z)
            lmag  = np.log(np.abs(Z) + self.log_eps)

            # Derivatives / local operators
            d_t   = sig.convolve2d(g,  [[-1],[0],[1]], mode="same")   # ∂/∂t
            d_x   = sig.convolve2d(g,  [[-1,0,1]],      mode="same")  # ∂/∂x
            lap   = laplace(g, mode="nearest")                        # ∇²

            # Analytic-signal envelope & phase
            hilb  = sig.hilbert(g, axis=0)
            env   = np.abs(hilb)
            phase = np.angle(hilb)
            sinp, cosp = np.sin(phase), np.cos(phase)

            feats.append(
                np.stack([g, reZ, imZ, lmag,
                          d_t, d_x, lap,
                          env, sinp, cosp], axis=0)
            )                                             # (10,T,R)

        return torch.from_numpy(np.stack(feats)).float()  # (S,10,T,R)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        w_path, v_path, i = self.index[idx]

        gather = np.load(w_path, mmap_mode="r")[i]          # (S,T,R)
        vel    = np.load(v_path,  mmap_mode="r")[i].copy()  # (H,W) or (1,H,W)
        vel    = np.squeeze(vel)                            # ensure (H,W)

        x = self._feature_stack(gather)                     # (S,10,T,R)
        y = torch.from_numpy(vel).float()

        return x, y



# model_fft_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────── encodes one source's 10-channel image ──────────────
class SmallSourceEncoder(nn.Module):
    def __init__(self, in_ch: int = 10, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base,      3, padding=1), nn.ReLU(),
            nn.Conv2d(base,  base * 2,  3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(base*2, base * 4, 3, stride=2, padding=1), nn.ReLU(),
        )

    def forward(self, x):            # (B,10,T,R)
        return self.net(x)           # (B,base*4,T/4,R/4)


# ───────────────── simple up-conv decoder ─────────────────────────────
class FusionDecoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 4, base: int = 64):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_ch, base*4, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.conv_final = nn.Conv2d(base*2, out_ch, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        return self.conv_final(x)    # (B,out_ch,H′,W′)


# ───────────────── full model ─────────────────────────────────────────
class FFTNetLateFusion(nn.Module):
    """
    Input  : (B,S,10,T,R)
    Output : (B,70,70)
    """
    def __init__(self, base: int = 32):
        super().__init__()
        self.enc = SmallSourceEncoder(in_ch=10, base=base)
        self.dec = FusionDecoder(in_ch=base*4, out_ch=4, base=base)

    def forward(self, x):
        B, S, C, T, R = x.shape

        # Encode each source separately (shared weights)
        x = x.view(-1, C, T, R)           # (B*S,10,T,R)
        feat = self.enc(x)                # (B*S,C′,T/4,R/4)

        # Fuse across sources by mean (replace with attention later)
        _, C2, H2, W2 = feat.shape
        feat = feat.view(B, S, C2, H2, W2).mean(dim=1)  # (B,C′,H2,W2)

        # Decode & project to 70×70 velocity plane
        out = self.dec(feat)              # (B,4,H′,W′)
        out = F.adaptive_avg_pool2d(out, (70, 70))  # (B,4,70,70)
        out = out.mean(dim=1)             # (B,70,70)

        return out



# train_fft_mae.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────
ROOT    = "/kaggle/input/waveform-inversion/train_samples"   # edit as needed
BATCH   = 4
EPOCHS  = 30
LR      = 2e-4            # slightly lower for richer input
WORKERS = 4

# ── DATA ──────────────────────────────────────────────────────────────
ds = WaveformDatasetFFT(ROOT)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True,
                num_workers=WORKERS, pin_memory=True)

# ── MODEL ─────────────────────────────────────────────────────────────
net  = FFTNetLateFusion(base=32).cuda()
opt  = optim.AdamW(net.parameters(), lr=LR)
loss_fn = nn.L1Loss()

# ── TRAIN ─────────────────────────────────────────────────────────────
for epoch in tqdm(range(1, EPOCHS + 1)):
    net.train()
    running = 0.0
    for x, y in tqdm(dl, desc=f"Epoch: {epoch}"):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)           # (B,70,70)

        pred = net(x)                           # (B,70,70)

        # competition slice – keep odd columns
        pred_sub = pred[:, :, 1::2]
        y_sub    = y[:,  :, 1::2]

        loss = loss_fn(pred_sub, y_sub)

        opt.zero_grad()
        loss.backward()
        opt.step()

        running += loss.item() * x.size(0)

    mae = running / len(ds)
    print(f"Epoch {epoch:02d} · MAE {mae:8.3f} m s⁻¹")



print("pred", pred.shape)  # torch.Size([B, 70, 70])
print("y   ", y.shape)     # should print torch.Size([B, 1, 70, 70]) or [B,70,70]

pred_sub = pred[:, :, 1::2]
y        = y.squeeze(1) if y.dim() == 4 else y
y_sub    = y[:, :, 1::2]

print("sub shapes", pred_sub.shape, y_sub.shape)  # both (B,70,35)

