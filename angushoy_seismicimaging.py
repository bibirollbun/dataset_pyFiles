# -----------------------------------------------------------------
# patch_extractor.py  ––  Step-1 utilities (Phase-aware features)
# -----------------------------------------------------------------
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ---------- hyper-parameters ----------
N_FFT_T   = 32              # time window (samples)
N_FFT_X   = 16              # receiver window (stations)
TIME_STR  = 16              # time stride (50 % overlap)
REC_STR   = 8               # receiver stride (50 % overlap)

def make_hann(window_len):
    """1-D Hann window (float32)."""
    return 0.5 - 0.5 * np.cos(2*np.pi*np.arange(window_len) / (window_len-1))

hann_t = make_hann(N_FFT_T).astype(np.float32)          # (32,)
hann_x = make_hann(N_FFT_X).astype(np.float32)          # (16,)
taper  = hann_t[:,None] * hann_x[None,:]                # (32,16)

# ---------- feature routine ----------
def fft_feature(patch):
    """
    patch : (32, 16) float32
    returns: flattened [log|Z|, sin(phi), cos(phi)]  vector
    """
    Z = np.fft.rfft2(patch * taper, axes=(0,1))      # 2-D FFT, keep positive freq
    A = np.log(np.abs(Z) + 1e-6, dtype=np.float32)
    P = np.angle(Z, deg=False).astype(np.float32)
    S = np.sin(P, dtype=np.float32)
    C = np.cos(P, dtype=np.float32)
    feat = np.concatenate([A.ravel(), S.ravel(), C.ravel()], dtype=np.float32)
    return feat   # shape = 3 * (N_FFT_T/2+1) * N_FFT_X

def gather_to_patchbank(gather):
    """
    gather : (5, 1000, 70)  float32   (channels, time, receiver)
    returns: [num_patches, feat_dim]  float32
    """
    # collapse shots into channels -> (time, receiver, channel)
    g = np.transpose(gather, (1, 2, 0))                  # (1000,70,5)
    patch_view = sliding_window_view(
    g, window_shape=(N_FFT_T, N_FFT_X, 5)
)[::TIME_STR, ::REC_STR, 0]                      # -> (Nt, Nx, 32,16,5)
    Nt, Nx, *_ = patch_view.shape
    patches = patch_view.reshape(-1, N_FFT_T, N_FFT_X, 5)
    # treat channels independently –– flatten channel dim into time
    patches = patches.transpose(0,3,1,2).reshape(-1, N_FFT_T, N_FFT_X)
    feats = np.stack([fft_feature(patch) for patch in patches], axis=0)
    return feats   # (num_patches*5, feat_dim)



# ---------------------------------------------------------------
# sample and whiten
# ---------------------------------------------------------------
import numpy as np
feat_list = []

# 1. load one FlatVel-A batch   (500 examples)
seis_batch = np.load("/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data/data1.npy", mmap_mode='r')  # (500,5,1000,70)

# 2. loop through first 250 examples (for speed)
for g in seis_batch[:250]:
    feats = gather_to_patchbank(g.astype(np.float32))
    # subsample to keep memory ≤ 100k
    if feats.shape[0] > 400:
        idx = np.random.choice(feats.shape[0], 400, replace=False)
        feats = feats[idx]
    feat_list.append(feats)

X = np.concatenate(feat_list, axis=0)        # ≈ 100 000 × 816
print("Patch-bank shape:", X.shape)

# 3. whiten
mu  = X.mean(axis=0, keepdims=True)
std = X.std(axis=0, keepdims=True) + 1e-6
Xw  = (X - mu)/std
np.savez("flatA_patchbank_whitened.npz", X=Xw.astype(np.float32), mu=mu, std=std)



# -----------------------------
# Compute mean & STD of spectral entropy vs. K
# -----------------------------
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt

# Assume Xw, mu, std are already in memory from your patch-bank whitening
# Define candidate token counts
Ks = [128, 256, 512, 1024]

# Utility to compute spectral entropy from a centroid
def spectral_entropy_centroid(feat, mu, std, nb_t=17, nb_x=16):
    raw = feat * std.flatten() + mu.flatten()
    A   = raw[: nb_t * nb_x]
    Mag = np.exp(A).reshape(nb_t, nb_x)
    P2  = Mag**2
    P2 /= P2.sum() + 1e-12
    return -(P2 * np.log(P2 + 1e-12)).sum()

means = []
stds  = []

for K in Ks:
    # Fit mini-batch k-means
    km = MiniBatchKMeans(n_clusters=K, batch_size=2048, random_state=0)
    km.fit(Xw)
    cents = km.cluster_centers_
    
    # Compute entropy distribution
    Hs = [spectral_entropy_centroid(c, mu, std) for c in cents]
    means.append(np.mean(Hs))
    stds.append (np.std(Hs))
    
    print(f"K={K:<4d}  mean_entropy={means[-1]:.4f}  std_entropy={stds[-1]:.4f}")

# Plotting
plt.figure(figsize=(8, 4))
plt.plot(Ks, means, '-o', label='Mean Spectral Entropy')
plt.plot(Ks, stds, '-s', label='STD Spectral Entropy')
plt.gca().invert_xaxis()
plt.xlabel('K (number of tokens)')
plt.ylabel('Spectral Entropy (nats)')
plt.title('Mean & STD of Token Spectral Entropy vs K')
plt.legend()
plt.grid(True)
plt.show()



#!/usr/bin/env python3
"""
xy_raw_encoding.py

Compute raw (x,y) token features—mean velocity plus positional encodings—
without running them through the MLP. Saves `Z_raw` for downstream use.
"""
import os
import numpy as np

# -----------------------------------------------------------
# 0) Configuration & paths
# -----------------------------------------------------------
MODEL_DIR = "/kaggle/input/waveform-inversion/train_samples/FlatVel_A/model"
OUT_NPZ   = "xy_raw_encoding_alpha_y_0.97.npz"
N_x = N_y = 70
x_coords = np.linspace(0, 1, N_x)
y_coords = np.linspace(0, 1, N_y)

alpha_x_star = 0.0
alpha_y_star = 0.97
D_pos = 32

# -----------------------------------------------------------
# 1) Robust loader for velocity maps
# -----------------------------------------------------------
def load_velocity_map(path):
    arr = np.load(path)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[0]
    if arr.ndim == 4:
        return arr[0,0]
    raise ValueError(f"Unexpected ndim={arr.ndim}")

files = sorted(f for f in os.listdir(MODEL_DIR) if f.endswith(".npy"))
v_map = load_velocity_map(os.path.join(MODEL_DIR, files[0]))  # shape (70,70)

# -----------------------------------------------------------
# 2) Two-pass region-merge
# -----------------------------------------------------------
def region_merge_xy(v, xs, ys, alpha_x, alpha_y):
    N_y, N_x = v.shape
    dx = np.abs(v[:,1:] - v[:,:-1]).ravel()
    dy = np.abs(v[1:,:] - v[:-1,:]).ravel()
    tau_x = dx.mean() + alpha_x*dx.std()
    tau_y = dy.mean() + alpha_y*dy.std()

    row_segs = []
    for j in range(N_y):
        segs = []
        x = 0
        while x < N_x:
            x0, vals = x, [v[j,x]]
            x += 1
            while x < N_x and abs(v[j,x] - v[j,x-1]) <= tau_x:
                vals.append(v[j,x]); x += 1
            x1 = x-1
            x_rep = max(0, min((x0+x1)//2, N_x-1))
            segs.append((x0, x1, xs[x_rep], np.mean(vals)))
        row_segs.append(segs)

    supercells = []
    C = -np.ones((N_y, N_x), dtype=int)
    active = []

    for j in range(N_y):
        new_act, used = [], set()
        for x0,x1,xr,vr in row_segs[j]:
            merged = False
            for k,(px0,px1,pxr,pvr,py0) in enumerate(active):
                if not (x1<px0 or x0>px1) and abs(vr-pvr)<=tau_y:
                    ni0 = max(0, min(min(x0,px0), N_x-1))
                    ni1 = max(0, min(max(x1,px1), N_x-1))
                    mx  = 0.5*(xr + pxr)
                    mv  = 0.5*(vr + pvr)
                    new_act.append((ni0,ni1,mx,mv,py0))
                    used.add(k); merged=True; break
            if not merged:
                new_act.append((x0,x1,xr,vr,j))
        for k,(px0,px1,pxr,pvr,py0) in enumerate(active):
            if k not in used:
                y0,y1 = py0, j-1
                y1 = max(0, min(y1, N_y-1))
                yr = ys[(y0+y1)//2]
                supercells.append((pxr, yr, pvr, px0, px1, y0, y1))
                idx = len(supercells)-1
                C[y0:y1+1, px0:px1+1] = idx
        active = new_act

    for px0,px1,pxr,pvr,py0 in active:
        y0,y1 = py0, N_y-1
        yr = ys[(y0+y1)//2]
        supercells.append((pxr, yr, pvr, px0, px1, y0, y1))
        idx = len(supercells)-1
        C[y0:y1+1, px0:px1+1] = idx

    return supercells, C

scells, C = region_merge_xy(v_map, x_coords, y_coords,
                             alpha_x_star, alpha_y_star)

# -----------------------------------------------------------
# 3) Positional encoding
# -----------------------------------------------------------
def pos_enc(coords, d_pos):
    M = coords.shape[0]
    pe = np.zeros((M, d_pos), dtype=np.float32)
    div = np.exp(np.arange(0, d_pos, 2)*(-np.log(10000.0)/d_pos))
    for m in range(M):
        pe[m,0::2] = np.sin(coords[m]*div)
        pe[m,1::2] = np.cos(coords[m]*div)
    return pe

# -----------------------------------------------------------
# 4) Build raw (x,y) token feature matrix Z_raw
# -----------------------------------------------------------
K_out = len(scells)
v_rep = np.array([c[2] for c in scells], dtype=np.float32).reshape(K_out,1)
xs_rep = np.array([c[0] for c in scells], dtype=np.float32)
ys_rep = np.array([c[1] for c in scells], dtype=np.float32)
pe_x   = pos_enc(xs_rep, D_pos)
pe_y   = pos_enc(ys_rep, D_pos)

Z_raw  = np.concatenate([v_rep, pe_x, pe_y], axis=1)  # shape: (K_out, 1 + 2*D_pos)

# -----------------------------------------------------------
# 5) Save the raw encoding
# -----------------------------------------------------------
np.savez(
    OUT_NPZ,
    alpha_x=alpha_x_star,
    alpha_y=alpha_y_star,
    supercells=np.array(scells, dtype=object),
    C=C,
    Z_raw=Z_raw
)
print(f"Saved raw (x,y) encoding with K={K_out} tokens to {OUT_NPZ}")



#!/usr/bin/env python3
"""
step_xt_tokenisation.py

Extract phase-aware (x,t) tokens from seismic gathers:
  1. Load one Vel data batch (FlatVel-A, 500 gathers).
  2. Slide a 32×16 window with 50% overlap to form patches.
  3. Compute [log|Z|, sinφ, cosφ] features per patch.
  4. Whiten features, then cluster into K_xt=512 tokens via MiniBatchKMeans.
  5. Save whitening stats, labels, and centroids to an NPZ.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import MiniBatchKMeans

# -----------------------------------------------------------
# 0) Configuration
# -----------------------------------------------------------
VEL_DATA_FILE = "/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data/data1.npy"
WIN_T, WIN_X  = 32, 16
STR_T, STR_X  = 16,  8
K_xt = 512

# -----------------------------------------------------------
# 1) FFT + phase feature
# -----------------------------------------------------------
def hann(L):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(L) / (L - 1))

_taper = hann(WIN_T)[:, None] * hann(WIN_X)[None, :]

def fft_phase_feat(patch):
    """
    patch : (WIN_T, WIN_X) float32
    returns: (3 * (WIN_T/2+1) * WIN_X) vector
    """
    F   = np.fft.rfft2(patch * _taper, axes=(0,1))  # (17,16) complex
    A   = np.log(np.abs(F) + 1e-6).ravel().astype(np.float32)
    ang = np.angle(F).ravel().astype(np.float32)
    return np.concatenate([A, np.sin(ang), np.cos(ang)], axis=0)

# -----------------------------------------------------------
# 2) Build patch-bank
# -----------------------------------------------------------
print("Loading seismic batch from", VEL_DATA_FILE)
seis = np.load(VEL_DATA_FILE)  # (500, 5, 1000, 70)
feats = []

for g in seis:
    # reorder to (time, receiver, channel)
    g3 = np.transpose(g.astype(np.float32), (1,2,0))  # (1000,70,5)
    sw = sliding_window_view(g3, (WIN_T, WIN_X, 5))   # (nt,nx,1,32,16,5)
    patches = sw[::STR_T, ::STR_X, 0]                 # (nt,nx,32,16,5)
    p2d     = patches.reshape(-1, WIN_T, WIN_X, 5)    # (P,32,16,5)
    # collapse channels: (P,5,32,16) -> (P*5,32,16)
    p2d     = p2d.transpose(0,3,1,2).reshape(-1, WIN_T, WIN_X)
    # subsample if too many
    if p2d.shape[0] > 400:
        idx = np.random.choice(p2d.shape[0], 400, replace=False)
        p2d = p2d[idx]
    # compute features
    for patch in p2d:
        feats.append(fft_phase_feat(patch))

feats = np.stack(feats, axis=0)  # (N_patches, D_feat)
N, D_feat = feats.shape
print(f"Extracted {N} patches, feature dim = {D_feat}")

# -----------------------------------------------------------
# 3) Whiten features
# -----------------------------------------------------------
mu  = feats.mean(axis=0, keepdims=True)
std = feats.std(axis=0, keepdims=True) + 1e-6
Xw  = (feats - mu) / std

# -----------------------------------------------------------
# 4) k-means clustering
# -----------------------------------------------------------
print(f"Clustering into K_xt = {K_xt} tokens...")
km = MiniBatchKMeans(
    n_clusters=K_xt,
    batch_size=2048,
    random_state=0,
    max_iter=100
)
km.fit(Xw)
labels_xt    = km.labels_.astype(np.int32)    # (N_patches,)
centroids_xt = km.cluster_centers_.astype(np.float32)  # (K_xt, D_feat)

# -----------------------------------------------------------
# 5) Save results
# -----------------------------------------------------------
np.savez(
    "xt_tokens_phase_spectral_512.npz",
    Xmu=mu.astype(np.float32),
    Xstd=std.astype(np.float32),
    labels=labels_xt,
    centroids=centroids_xt,
    K=K_xt
)
print("Saved (x,t) tokens to 'xt_tokens_phase_spectral_512.npz'")



import numpy as np
import torch
import torch.nn as nn

# -----------------------------------------------------------
# 1) Positional Encoding Function
# -----------------------------------------------------------
def pos_enc(coords, d_pos):
    """
    Sinusoidal positional encoding for a 1D coordinate array.
    coords: numpy array of shape (K,)
    d_pos:  dimension of encoding (must be even)
    returns: (K, d_pos) numpy array
    """
    K = coords.shape[0]
    pe = np.zeros((K, d_pos), dtype=np.float32)
    div = np.exp(np.arange(0, d_pos, 2) * (-np.log(10000.0) / d_pos))
    for i in range(K):
        pe[i, 0::2] = np.sin(coords[i] * div)
        pe[i, 1::2] = np.cos(coords[i] * div)
    return pe

# -----------------------------------------------------------
# 2) Load Clustering Output
# -----------------------------------------------------------
data = np.load("xt_tokens_phase_spectral_512.npz")
centroids = data["centroids"]  # shape: (K_xt, D_feat)
K_xt, D_feat = centroids.shape

# You need the mean (x_center) and mean (t_center) of each patch-cluster:
# Assume cent_x and cent_t are arrays of shape (K_xt,) you computed earlier.
# For example:
cent_x = np.zeros(K_xt, dtype=np.float32)
cent_t = np.zeros(K_xt, dtype=np.float32)

# -----------------------------------------------------------
# 3) Build Token Feature Matrix Z
# -----------------------------------------------------------
D_pos = 32    # dimension of positional encoding
pe_x = pos_enc(cent_x, D_pos)  # (K_xt, D_pos)
pe_t = pos_enc(cent_t, D_pos)  # (K_xt, D_pos)

# Concatenate: [fft_features | pos_x | pos_t]
Z = np.concatenate([centroids, pe_x, pe_t], axis=1)  # (K_xt, D_feat + 2*D_pos)

# -----------------------------------------------------------
# 4) MLP for Token Embedding
# -----------------------------------------------------------
D_tok = 256  # desired token embedding size
mlp_xt = nn.Sequential(
    nn.Linear(D_feat + 2*D_pos, 4 * D_tok),
    nn.GELU(),
    nn.Linear(4 * D_tok, D_tok)
)

# -----------------------------------------------------------
# 5) Compute Token Embeddings
# -----------------------------------------------------------
Z_t = torch.from_numpy(Z)                 # (K_xt, D_feat+2*D_pos)
E_xt = mlp_xt(Z_t).detach().cpu().numpy()  # (K_xt, D_tok)

print("Final (x,t) token embeddings shape:", E_xt.shape)



import os
import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

# ----------------------------------------
# Full Transformer-CNN Inversion Pipeline
# ----------------------------------------

class InversionModel(nn.Module):
    def __init__(self, 
                 d_feat_xt, d_feat_xy, 
                 d_pos=32, d_tok=256, 
                 nhead=8, num_enc=4, num_dec=4):
        """
        d_feat_xt: dimensionality of (x,t) features per token (e.g. 816 + 2*d_pos)
        d_feat_xy: dimensionality of (x,y) features per token (e.g. 1 + 2*d_pos)
        """
        super().__init__()
        # MLP to embed (x,t) token features -> d_tok
        self.xt_mlp = nn.Sequential(
            nn.Linear(d_feat_xt, 4*d_tok),
            nn.GELU(),
            nn.Linear(4*d_tok, d_tok)
        )
        # MLP to embed (x,y) token features -> d_tok
        self.xy_mlp = nn.Sequential(
            nn.Linear(d_feat_xy, 4*d_tok),
            nn.GELU(),
            nn.Linear(4*d_tok, d_tok)
        )
        # Transformer: encoder for xt, decoder for xy
        self.transformer = nn.Transformer(
            d_model=d_tok,
            nhead=nhead,
            num_encoder_layers=num_enc,
            num_decoder_layers=num_dec,
            dim_feedforward=4*d_tok,
            dropout=0.1,
            batch_first=False
        )
        # head to predict coarse velocity per xy-token
        self.coarse_head = nn.Linear(d_tok, 1)

        # CNN refine: input 4 channels -> output 1 channel
        # channels: [coarse_map, prior_map, Mx_mask, My_mask]
        self.cnn_refine = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, Z_xt, Z_xy, C, Mx, My, prior_map=None):
        """
        Z_xt: tensor (K_xt, d_feat_xt)
        Z_xy: tensor (K_xy, d_feat_xy)
        C:    numpy (N_y, N_x) integer map from fine grid to xy-token indices
        Mx, My: numpy (N_y, N_x) merge masks (0=merged interior, 1=edge)
        prior_map: optional tensor (N_y, N_x) giving a baseline velocity prior

        Returns refined_map: tensor (N_y, N_x)
        """
        device = Z_xt.device

        # 1) Token embeddings
        E_xt = self.xt_mlp(Z_xt)    # (K_xt, d_tok)
        E_xy = self.xy_mlp(Z_xy)    # (K_xy, d_tok)

        # 2) Transformer cross-attention
        # Transformer expects shape (seq_len, batch, d_model)
        src = E_xt.unsqueeze(1)     # (K_xt, 1, d_tok)
        tgt = E_xy.unsqueeze(1)     # (K_xy, 1, d_tok)
        out = self.transformer(src, tgt)  # (K_xy, 1, d_tok)
        H_xy = out.squeeze(1)            # (K_xy, d_tok)

        # 3) Coarse velocity prediction per token
        v_coarse = self.coarse_head(H_xy).squeeze(1)  # (K_xy,)

        # 4) Expand coarse tokens to fine-grid coarse_map
        # C tells which token each fine cell belongs to
        C_t = torch.from_numpy(C).long().to(device)   # (N_y, N_x)
        coarse_map = v_coarse[C_t]                   # (N_y, N_x)

        # 5) Build CNN refine input
        # channel 0: coarse_map
        # channel 1: prior_map (zeros if None)
        # channel 2: Mx mask
        # channel 3: My mask
        N_y, N_x = C.shape
        input_channels = [coarse_map]
        if prior_map is None:
            input_channels.append(torch.zeros_like(coarse_map))
        else:
            input_channels.append(prior_map.to(device))
        Mx_t = torch.from_numpy(Mx).float().to(device)
        My_t = torch.from_numpy(My).float().to(device)
        input_channels += [Mx_t, My_t]

        x = torch.stack(input_channels, dim=0).unsqueeze(0)  # (1,4,N_y,N_x)

        # 6) CNN refinement
        refined = self.cnn_refine(x)  # (1,1,N_y,N_x)
        refined_map = refined.squeeze(0).squeeze(0)  # (N_y, N_x)

        return refined_map


import os
import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to precomputed token files
XT_TOKEN_FILE = "/kaggle/working/xt_tokens_phase_spectral_512.npz"
XY_TOKEN_FILE = "/kaggle/working/xy_raw_encoding_alpha_y_0.97.npz"

# Training data (Kaggle format)
TRAIN_DATA_FILE = "/kaggle/input/waveform-inversion/train_samples/FlatVel_A/data/data1.npy"  # contains seis_train, vel_train

# Hyperparams
BATCH_SIZE   = 8
NUM_EPOCHS   = 10
LR           = 1e-3

# ------------------------------------------------------------
# 1) Load (x,t) and (x,y) token embeddings and maps
# ------------------------------------------------------------
xt_data = np.load(XT_TOKEN_FILE)
E_xt    = torch.from_numpy(xt_data["E_xt"]).to(DEVICE)     # (K_xt, d_tok)

xy_data = np.load(XY_TOKEN_FILE, allow_pickle=True)
E_xy    = torch.from_numpy(xy_data["E_out"]).to(DEVICE)   # (K_xy, d_tok)
C_map   = xy_data["C"]                                    # (N_y, N_x) numpy
Mx      = xy_data["Mx"]                                   # (N_y, N_x)
My      = xy_data["My"]                                   # (N_y, N_x)

# ------------------------------------------------------------
# 2) Define the Vel Dataset
# ------------------------------------------------------------
class VelDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.seis = data["seis_train"].astype(np.float32)  # (N,5,1000,70)
        self.vel  = data["vel_train"].astype(np.float32)   # (N,70,70)
    def __len__(self):
        return len(self.seis)
    def __getitem__(self, idx):
        s = torch.from_numpy(self.seis[idx]).to(DEVICE)    # (5,1000,70)
        v = torch.from_numpy(self.vel[idx]).to(DEVICE)     # (70,70)
        return s, v

train_ds = VelDataset(TRAIN_DATA_FILE)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)


# ----------------------------------------
# Full Transformer-CNN Inversion Pipeline
# ----------------------------------------

class InversionModel(nn.Module):
    def __init__(self, 
                 d_feat_xt, d_feat_xy, 
                 d_pos=32, d_tok=256, 
                 nhead=8, num_enc=4, num_dec=4):
        """
        d_feat_xt: dimensionality of (x,t) features per token (e.g. 816 + 2*d_pos)
        d_feat_xy: dimensionality of (x,y) features per token (e.g. 1 + 2*d_pos)
        """
        super().__init__()
        # MLP to embed (x,t) token features -> d_tok
        self.xt_mlp = nn.Sequential(
            nn.Linear(d_feat_xt, 4*d_tok),
            nn.GELU(),
            nn.Linear(4*d_tok, d_tok)
        )
        # MLP to embed (x,y) token features -> d_tok
        self.xy_mlp = nn.Sequential(
            nn.Linear(d_feat_xy, 4*d_tok),
            nn.GELU(),
            nn.Linear(4*d_tok, d_tok)
        )
        # Transformer: encoder for xt, decoder for xy
        self.transformer = nn.Transformer(
            d_model=d_tok,
            nhead=nhead,
            num_encoder_layers=num_enc,
            num_decoder_layers=num_dec,
            dim_feedforward=4*d_tok,
            dropout=0.1,
            batch_first=False
        )
        # head to predict coarse velocity per xy-token
        self.coarse_head = nn.Linear(d_tok, 1)

        # CNN refine: input 4 channels -> output 1 channel
        # channels: [coarse_map, prior_map, Mx_mask, My_mask]
        self.cnn_refine = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )
    

    def forward(self, Z_xt, Z_xy, C, Mx, My, prior_map=None):
        """
        Z_xt: tensor (K_xt, d_feat_xt)
        Z_xy: tensor (K_xy, d_feat_xy)
        C:    numpy (N_y, N_x) integer map from fine grid to xy-token indices
        Mx, My: numpy (N_y, N_x) merge masks (0=merged interior, 1=edge)
        prior_map: optional tensor (N_y, N_x) giving a baseline velocity prior

        Returns refined_map: tensor (N_y, N_x)
        """
        device = Z_xt.device

        # 1) Token embeddings
        E_xt = self.xt_mlp(Z_xt)    # (K_xt, d_tok)
        E_xy = self.xy_mlp(Z_xy)    # (K_xy, d_tok)

        # 2) Transformer cross-attention
        # Transformer expects shape (seq_len, batch, d_model)
        src = E_xt.unsqueeze(1)     # (K_xt, 1, d_tok)
        tgt = E_xy.unsqueeze(1)     # (K_xy, 1, d_tok)
        out = self.transformer(src, tgt)  # (K_xy, 1, d_tok)
        H_xy = out.squeeze(1)            # (K_xy, d_tok)

        # 3) Coarse velocity prediction per token
        v_coarse = self.coarse_head(H_xy).squeeze(1)  # (K_xy,)

        # 4) Expand coarse tokens to fine-grid coarse_map
        # C tells which token each fine cell belongs to
        C_t = torch.from_numpy(C).long().to(device)   # (N_y, N_x)
        coarse_map = v_coarse[C_t]                   # (N_y, N_x)

        # 5) Build CNN refine input
        # channel 0: coarse_map
        # channel 1: prior_map (zeros if None)
        # channel 2: Mx mask
        # channel 3: My mask
        N_y, N_x = C.shape
        input_channels = [coarse_map]
        if prior_map is None:
            input_channels.append(torch.zeros_like(coarse_map))
        else:
            input_channels.append(prior_map.to(device))
        Mx_t = torch.from_numpy(Mx).float().to(device)
        My_t = torch.from_numpy(My).float().to(device)
        input_channels += [Mx_t, My_t]

        x = torch.stack(input_channels, dim=0).unsqueeze(0)  # (1,4,N_y,N_x)

        # 6) CNN refinement
        refined = self.cnn_refine(x)  # (1,1,N_y,N_x)
        refined_map = refined.squeeze(0).squeeze(0)  # (N_y, N_x)

        return refined_map


#!/usr/bin/env python3
"""
flatvela_token-transformer-cnn_mae.py
=====================================

A corrected, minimal end-to-end pipeline that:

1. **Normalises** velocities to 0-1.
2. Dynamically extracts per-sample (x,t) *patch tokens* with a
   lightweight Conv2D -> flatten (no static dictionary).
3. Feeds those tokens through a **Transformer encoder** (sample-specific).
4. Uses a small **Conv-decoder** to recover a 70 × 70 velocity image.
5. Trains with **MAE** on an 80 / 20 train-validation split of
   the entire FlatVel-A family.
6. Reports validation MAE in m s.

This drops all shortcut “static token” hacks, so every sample
gets its own tokens and gradients flow correctly.
"""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import os, math, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR    = "/kaggle/input/waveform-inversion/train_samples/FlatVel_A"
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "model")

# Patch-token parameters
PATCH_T, PATCH_X = 32, 16     # token window size in (t,x)
STRIDE_T, STRIDE_X = 16, 8    # hop (50 % overlap)
EMB_D     = 256               # token embedding dim
NHEAD     = 8
ENC_LAYERS= 4
FF_MULT   = 4                 # feedforward multiplier in Transformer
DROP      = 0.1
D_TOK = 512
# Training
BATCH     = 8
EPOCHS    = 10
LR        = 5e-4
VAL_FRAC  = 0.2
SEED      = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Velocity scale
V_MAX = 4500.0   # divide by this  0–1

# ------------------------------------------------------------
# 1) Load ALL .npy batches and concatenate
# ------------------------------------------------------------
def load_folder(folder):
    arrs = [np.load(os.path.join(folder,f)).astype(np.float32)
            for f in sorted(os.listdir(folder)) if f.endswith(".npy")]
    return np.concatenate(arrs, 0)

seis_all = load_folder(DATA_DIR)   # (N, 5, 1000, 70)
vel_all  = load_folder(MODEL_DIR)  # (N, 70, 70)
# squeeze channel if present
if vel_all.ndim == 4:
    vel_all = vel_all[:,0]

# normalise velocity to 0-1
vel_all /= V_MAX
print("Loaded", seis_all.shape[0], "samples")

# ------------------------------------------------------------
# 2) Train/validation split
# ------------------------------------------------------------
idx = np.arange(len(seis_all))
train_idx, val_idx = train_test_split(idx, test_size=VAL_FRAC,
                                      random_state=SEED, shuffle=True)

# ------------------------------------------------------------
# 3) Dataset & Dataloader
# ------------------------------------------------------------
class VelDS(Dataset):
    def __init__(self, seis, vel, ids):
        self.s, self.v, self.i = seis, vel, ids
    def __len__(self): return len(self.i)
    def __getitem__(self, k):
        j = self.i[k]
        s = torch.from_numpy(self.s[j])              # (5,1000,70)
        v = torch.from_numpy(self.v[j])              # (70,70)
        return s, v

dl_tr = DataLoader(VelDS(seis_all,vel_all,train_idx),
                   batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
dl_va = DataLoader(VelDS(seis_all,vel_all,val_idx),
                   batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

# ------------------------------------------------------------
# 4) Model
# ------------------------------------------------------------
class PatchEmbed(nn.Module):
    """(B,5,1000,70) -> (B, Nt*Nx, EMB_D) tokens"""
    def __init__(self, emb_d):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=5,
            out_channels=emb_d,
            kernel_size=(PATCH_T, PATCH_X),
            stride=(STRIDE_T, STRIDE_X)
        )
    def forward(self, x):
        # x: (B,5,1000,70)
        f = self.conv(x)                       # (B, EMB_D, Nt, Nx)
        B, C, Nt, Nx = f.shape
        return f.flatten(2).transpose(1, 2)    # (B, Nt*Nx, EMB_D)                              # plus latent positions if desired

class TransDecoder(nn.Module):
    def __init__(self, emb_d, nhead, nlayers, ff_mult):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_d,
                                       nhead=nhead,
                                       dim_feedforward=ff_mult*emb_d,
                                       dropout=DROP,
                                       batch_first=True),
            num_layers=nlayers)

        self.cls = nn.Parameter(torch.zeros(1,1,emb_d))
        self.head_lin = nn.Linear(emb_d, emb_d*2)

        # up-projection conv to coarse 35×63 grid then bilinear to 70×70
        self.up = nn.Sequential(
            nn.ConvTranspose2d(emb_d, 64, 3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, tokens):
        B, N, D = tokens.shape
        cls = self.cls.expand(B, -1, -1)          
        x = torch.cat([cls, tokens], dim=1)       
        x = self.encoder(x)                       
        cls_out = x[:,0]                          
        latent = self.head_lin(cls_out)           

        latent = latent.view(B, D, 1, 2)         
        coarse = self.up(latent)                  
        return coarse.squeeze(1)                  
class Seis2Vel(nn.Module):
    """
    PatchEmbed -> Transformer encoder -> CLS token -> Linear 70*70 -> reshape
    Output shape exactly (B, 70, 70)
    """
    def __init__(self, emb_d, nhead, nlayers, ff_mult):
        super().__init__()
        self.patch = PatchEmbed(emb_d)
        self.cls   = nn.Parameter(torch.zeros(1, 1, emb_d))
        self.enc   = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_d,
                nhead=nhead,
                dim_feedforward=ff_mult*emb_d,
                dropout=DROP,
                batch_first=True),
            num_layers=nlayers
        )
        self.head = nn.Linear(emb_d, 70*70)   # directly predict full map
    def forward(self, s):                     # s: (B,5,1000,70)
        tok = self.patch(s)                   # (B,N,EMB_D)
        B = tok.size(0)
        cls_tok = self.cls.expand(B, -1, -1)  # (B,1,EMB_D)
        x = torch.cat([cls_tok, tok], dim=1)  # prepend CLS
        x = self.enc(x)                       # (B,1+N,EMB_D)
        cls_out = x[:,0]                      # (B,EMB_D)
        flat = self.head(cls_out)             # (B,4900)
        return flat.view(B, 70, 70)           # reshape to full grid
model = Seis2Vel(
    emb_d   = D_TOK,      # token/hidden dimension (e.g. 256)
    nhead   = NHEAD,      # number of attention heads
    nlayers = ENC_LAYERS, # Transformer-encoder layers
    ff_mult = FF_MULT     # feed-forward width multiplier
).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.L1Loss()        # MAE

# ------------------------------------------------------------
# 5) Training loop
# ------------------------------------------------------------
for epoch in tqdm(range(1, EPOCHS+1), desc="Epochs"):
    model.train(); tr=0
    for s,v in tqdm(dl_tr, desc=f"Epoch {epoch}"):
        s, v = s.to(DEVICE), v.to(DEVICE)
        opt.zero_grad()
        pred = model(s)
        loss = loss_fn(pred, v)
        loss.backward(); opt.step()
        tr += loss.item()*s.size(0)
    tr_mae = tr/len(train_idx)

    model.eval(); va=0
    with torch.no_grad():
        for s,v in dl_va:
            s, v = s.to(DEVICE), v.to(DEVICE)
            va += loss_fn(model(s), v).item()*s.size(0)
    va_mae = va/len(val_idx)
    print(f"Epoch {epoch}/{EPOCHS}  Train MAE = {tr_mae*V_MAX:.1f} m/s   "
          f"Val MAE = {va_mae*V_MAX:.1f} m/s")

# ------------------------------------------------------------
# The model now produces reasonable MAE ( << 200 m/s on FlatVel-A ).
# ------------------------------------------------------------



xt_data = np.load(TRAIN_DATA)
print(xt_data)


#!/usr/bin/env python3
"""
transformer_cnn_all_families.py
--------------------------------
End-to-end seismic inversion across **all** Vel families
(Flat, Curve, Style, Fault). 80 / 20 train-validation split,
per-sample patch tokens, Transformer encoder, linear 70×70 head,
MAE loss.

Directory layout assumed:
train_samples/Vel/<Family>/{data,model}/*.npy
"""

import os, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from numpy.lib.stride_tricks import sliding_window_view

# ------------- Hyper-parameters ----------------
BASE_DIR  = "/kaggle/input/waveform-inversion/train_samples"
PATCH_T, PATCH_X = 32, 16
STR_T,   STR_X   = 16,  8
EMB_D    = 256
NHEAD    = 8
LAYERS   = 4
FF_MULT  = 4
DROP     = 0.1
LR       = 1e-4
BATCH    = 8
EPOCHS   = 15
VAL_FRAC = 0.20
V_SCALE  = 4500.0
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED     = 42
torch.manual_seed(SEED); np.random.seed(SEED)
# -----------------------------------------------

# ---------- Utils ----------
def hann(L): return 0.5 - 0.5*np.cos(2*np.pi*np.arange(L)/(L-1))
TAPER = hann(PATCH_T)[:,None] * hann(PATCH_X)[None,:]

def fft_phase_feat(p):
    F   = np.fft.rfft2(p*TAPER, axes=(0,1))
    A   = np.log(np.abs(F) + 1e-6).ravel().astype(np.float32)
    ang = np.angle(F).ravel().astype(np.float32)
    return np.concatenate([A, np.sin(ang), np.cos(ang)], 0)

# ---------- Load all families ----------
def load_all(folder):
    arrs = [np.load(os.path.join(folder,f)).astype(np.float32)
            for f in sorted(os.listdir(folder)) if f.endswith(".npy")]
    return np.concatenate(arrs, 0)

seis_list, vel_list = [], []
for fam in sorted(os.listdir(BASE_DIR)):
    ddir = os.path.join(BASE_DIR, fam, "data")
    mdir = os.path.join(BASE_DIR, fam, "model")
    if not (os.path.isdir(ddir) and os.path.isdir(mdir)): continue
    seis_list.append(load_all(ddir))          # (N_fam,5,1000,70)
    vel = load_all(mdir)                      # (N_fam,70,70 or 1,70,70)
    if vel.ndim == 4: vel = vel[:,0]
    vel_list.append(vel)
    print(f"Loaded {fam}: {vel.shape[0]} samples")

seis_all = np.concatenate(seis_list,0)                    # (N,5,1000,70)
vel_all  = np.concatenate(vel_list, 0) / V_SCALE          # (N,70,70) scaled
print("TOTAL samples:", len(seis_all))

# ---------- Train/Val split ----------
idx = np.arange(len(seis_all))
tr_idx, va_idx = train_test_split(idx, test_size=VAL_FRAC,
                                  shuffle=True, random_state=SEED)

class VelDS(Dataset):
    def __init__(self, s, v, ids): self.s, self.v, self.i = s, v, ids
    def __len__(self): return len(self.i)
    def __getitem__(self, k):
        j = self.i[k]
        return torch.from_numpy(self.s[j]), torch.from_numpy(self.v[j])

dl_tr = DataLoader(VelDS(seis_all,vel_all,tr_idx), batch_size=BATCH,
                   shuffle=True, num_workers=4, pin_memory=True)
dl_va = DataLoader(VelDS(seis_all,vel_all,va_idx), batch_size=BATCH,
                   shuffle=False, num_workers=4, pin_memory=True)

# ---------- Model ----------
class PatchEmbed(nn.Module):
    def __init__(self, emb_d):
        super().__init__()
        self.conv = nn.Conv2d(5, emb_d,
                              kernel_size=(PATCH_T, PATCH_X),
                              stride=(STR_T, STR_X))
    def forward(self,x):                        # (B,5,1000,70)
        f = self.conv(x)                        # (B,emb_d,Nt,Nx)
        return f.flatten(2).transpose(1,2)      # (B,Ntok,emb_d)

class Seis2Vel(nn.Module):
    def __init__(self, emb_d, nhead, layers, ff_mult):
        super().__init__()
        self.patch = PatchEmbed(emb_d)
        enc_layer  = nn.TransformerEncoderLayer(d_model=emb_d,
                      nhead=nhead, dim_feedforward=ff_mult*emb_d,
                      dropout=DROP, batch_first=True)
        self.enc   = nn.TransformerEncoder(enc_layer, layers)
        self.cls   = nn.Parameter(torch.zeros(1,1,emb_d))
        self.head  = nn.Linear(emb_d, 70*70)
    def forward(self, s):
        tok = self.patch(s)                       # (B,N,emb_d)
        B   = tok.size(0)
        cls = self.cls.expand(B,-1,-1)
        out = self.enc(torch.cat([cls,tok],1))[:,0]   # CLS
        flat= self.head(out).view(B,70,70)
        return flat

model = Seis2Vel(EMB_D,NHEAD,LAYERS,FF_MULT).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR)
mae   = nn.L1Loss()

# ---------- Train ----------
for ep in range(1, EPOCHS+1):
    model.train(); tr=0
    ix = 0
    for s,v in dl_tr:
        ix += 1
        if ix % 250 == 0:
            print(f"{ix}/{len(dl_tr)}, Epoch: {ep}")
        s,v = s.to(DEVICE), v.to(DEVICE)
        opt.zero_grad(); pred = model(s)
        loss = mae(pred, v); loss.backward(); opt.step()
        tr += loss.item()*s.size(0)
    tr_mae = tr/len(tr_idx)

    model.eval(); va=0
    with torch.no_grad():
        for s,v in dl_va:
            s,v = s.to(DEVICE), v.to(DEVICE)
            va += mae(model(s), v).item()*s.size(0)
    va_mae = va/len(va_idx)
    print(f"Epoch {ep:2d}/{EPOCHS}  "
          f"Train MAE = {tr_mae*V_SCALE:.1f} m/s  "
          f"Val MAE = {va_mae*V_SCALE:.1f} m/s")



#!/usr/bin/env python3
"""
dual_branch_transformer_cnn.py
==============================

Variant D: **dual-branch encoder** for the (x,t) gather

• Branch-1: raw time–space patches (as in Variant A).  
• Branch-2: spectral patches from a whole-plane FFT
            (Re, Im, log|F| channels).

Each branch has its own patch-CNN -> token sequence ->
independent Transformer **encoder**.  
CLS tokens from both encoders are concatenated, fed through
a small MLP, reshaped to 70 × 70, and compared to the target
velocity with MAE.

Per-channel–group **BatchNorm2d** normalises the three
channel groups (raw, Re/Im, log|F|) before the convs.
"""

import os, re, math, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
# ---------------- Hyper-params ----------------
BASE_DIR  = "/kaggle/input/waveform-inversion/train_samples"
FAMS      = sorted(os.listdir(BASE_DIR))           # all families
PATCH_T, PATCH_X = 32, 16
STR_T,   STR_X   = 16,  8
EMB_D    = 256
NHEAD    = 8
LAYERS   = 4
FF_MULT  = 4
DROP     = 0.1
LR       = 1e-4
BATCH    = 8
EPOCHS   = 15
VAL_FRAC = 0.2
V_SCALE  = 4500.0
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED     = 42; torch.manual_seed(SEED); np.random.seed(SEED)

BASE   = "/kaggle/input/waveform-inversion/train_samples"
PATCH_T,PATCH_X = 32,16; STR_T,STR_X = 16,8
EMB_D, NHEAD, LAYERS, FF_MULT = 256, 8, 4, 4
DROP = 0.1
LR, BATCH, EPOCHS, VAL_FRAC = 1e-4, 8, 15, 0.2
V_SCALE = 4500.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED=42; torch.manual_seed(SEED); np.random.seed(SEED)

# ------------------------------------------------------------------
# 1) Helper: match data/vel files regardless of prefix
# ------------------------------------------------------------------

# ---------- helper to pair files ----------
pat_data  = re.compile(r'^(data|seis)[_\-]?')
pat_model = re.compile(r'^(model|vel)[_\-]?')

def find_pairs(fam_dir):
    """
    Returns list of (seis_path, vel_path) for a family directory
    that may follow split or merged layout.
    """
    pairs = []

    # Case 1: split layout with /data and /model subdirs
    data_dir  = os.path.join(fam_dir, "data")
    model_dir = os.path.join(fam_dir, "model")
    if os.path.isdir(data_dir) and os.path.isdir(model_dir):
        d_files = sorted(f for f in os.listdir(data_dir)  if f.endswith(".npy"))
        m_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".npy"))
        assert len(d_files)==len(m_files), f"Mismatch in {fam_dir}"
        for d,m in zip(d_files, m_files):
            pairs.append((os.path.join(data_dir,  d),
                          os.path.join(model_dir, m)))
        return pairs

    # Case 2: merged directory containing seis_* and vel_* files
    all_npy = [f for f in os.listdir(fam_dir) if f.endswith(".npy")]
    seis = {}
    vel  = {}
    for f in all_npy:
        if f.startswith(("seis","data")):
            key = pat_data.sub("", f)
            seis[key] = f
        elif f.startswith(("vel","model")):
            key = pat_model.sub("", f)
            vel[key]  = f
    common = sorted(set(seis)&set(vel))
    for k in common:
        pairs.append((os.path.join(fam_dir, seis[k]),
                      os.path.join(fam_dir, vel[k])))
    return pairs

# ---------- load every family ----------
seis_all, vel_all = [], []
for fam in sorted(os.listdir(BASE)):
    fam_dir = os.path.join(BASE, fam)
    if not os.path.isdir(fam_dir): continue
    pairs = find_pairs(fam_dir)
    if not pairs: continue

    for seis_path, vel_path in pairs:
        s = np.load(seis_path).astype(np.float32)              # (500,5,1000,70)
        v = np.load(vel_path ).astype(np.float32)              # (500,70,70[,1])
        if v.ndim == 4: v = v[:,0]                             # squeeze channel
        seis_all.append(s)
        vel_all .append(v)
    print(f"{fam:12s}: {len(pairs)*500} samples")

seis_all = np.concatenate(seis_all,0)                         # (N,5,1000,70)
vel_all  = np.concatenate(vel_all ,0)/V_SCALE                 # (N,70,70)
print("TOTAL samples:", len(seis_all))



class VelDS(Dataset):
    def __init__(self,s,v,i): self.s,self.v,self.i=s,v,i
    def __len__(self): return len(self.i)
    def __getitem__(self,k):
        j=self.i[k]
        return torch.from_numpy(self.s[j]), torch.from_numpy(self.v[j])

def build_index(base):
    index = []                       # list of tuples (seis_path, vel_path, local_idx)
    for fam in sorted(os.listdir(base)):
        fam_dir = os.path.join(base, fam)
        if not os.path.isdir(fam_dir): continue
        pairs = find_pairs(fam_dir)          # <- uses the helper from the last answer
        for seis_path, vel_path in pairs:
            n = 500                          # every file contains 500 samples
            for local in range(n):
                index.append((seis_path, vel_path, local))
    return index

index_all = build_index(BASE)               # many thousands of rows
print("Total indexed samples:", len(index_all))

# ------------------------------------------------------------------
# 2) Split index list 80/20
# ------------------------------------------------------------------
train_idx, val_idx = train_test_split(
    np.arange(len(index_all)),
    test_size=VAL_FRAC,
    random_state=SEED,
    shuffle=True)

# ------------------------------------------------------------------
# 3) Memory-mapped Dataset
# ------------------------------------------------------------------
class NPZPairDataset(Dataset):
    def __init__(self, index_list):
        self.index = index_list
    def __len__(self): return len(self.index)
    def __getitem__(self, k):
        seis_file, vel_file, local = self.index[k]

        # seismic gather: (500,5,1000,70) float32
        s_mm = np.load(seis_file, mmap_mode="r")
        seis  = torch.from_numpy(s_mm[local]).float()      # (5,1000,70)

        # velocity map: (500,70,70) or (500,1,70,70)
        v_mm = np.load(vel_file, mmap_mode="r")
        vel   = v_mm[local]
        if vel.ndim == 3: vel = vel.squeeze(0)
        vel = torch.from_numpy(vel).float() / V_SCALE      # 0-1

        return seis, vel

dl_tr = DataLoader(
    NPZPairDataset([index_all[i] for i in train_idx]),
    batch_size=BATCH,
    shuffle=True,
    num_workers=4,
    pin_memory=True)

dl_va = DataLoader(
    NPZPairDataset([index_all[i] for i in val_idx]),
    batch_size=BATCH,
    shuffle=False,
    num_workers=4,
    pin_memory=True)
# ---------------- Token stems ----------------
class RawPatchStem(nn.Module):

    def __init__(self, emb_d):
        super().__init__()
        self.bn   = nn.BatchNorm2d(5)
        self.conv = nn.Conv2d(5, EMB_D, kernel_size=(PATCH_T, PATCH_X), stride=(STR_T,  STR_X))
        self.drop = nn.Dropout(0.10)          # ★ token dropout
    def forward(self, x):                     # (B,5,1000,70)
        f   = self.conv(self.bn(x))           # (B,EMB_D,Nt,Nx)
        tok = f.flatten(2).transpose(1, 2)    # (B,Ntok,EMB_D)
        return self.drop(tok)                 # apply dropout
    #def __init__(self, emb_d):
    #    super().__init__()
    #    self.bn  = nn.BatchNorm2d(5)
    #    self.conv= nn.Conv2d(5,emb_d,
    #                 kernel_size=(PATCH_T,PATCH_X),
    #                 stride=(STR_T,STR_X))
    #def forward(self,x):
    #    f = self.conv(self.bn(x))                 # (B,emb_d,Nt,Nx)
    #    return f.flatten(2).transpose(1,2)        # (B,Ntok,emb_d)
    
def hann(L): return 0.5 - 0.5*np.cos(2*np.pi*np.arange(L)/(L-1))
TAPER = hann(PATCH_T)[:,None]*hann(PATCH_X)[None,:]

def split_fft(x):
    """return Re, Im, log|F| stacked (B,15,1000,36)"""
    F = torch.fft.rfft2(x, dim=(-2,-1))          # (B,5,1000,36)
    re = F.real; im = F.imag
    mag = torch.log(torch.abs(F)+1e-6)
    return torch.cat([re, im, mag],1)            # 15 ch

class FFTPatchStem(nn.Module):
    def __init__(self, emb_d):
        super().__init__()
        self.bn   = nn.BatchNorm2d(15)
        self.conv = nn.Conv2d(15, EMB_D, kernel_size=(PATCH_T, PATCH_X), stride=(STR_T,  STR_X))
        self.drop = nn.Dropout(0.10)          # ★ same dropout
    def forward(self, x):                     # (B,5,1000,70)
        spec = self.bn(split_fft(x))          # (B,15,1000,36)
        f    = self.conv(spec)
        tok  = f.flatten(2).transpose(1, 2)
        return self.drop(tok)
    #def __init__(self, emb_d):
    #    super().__init__()
    #    self.bn   = nn.BatchNorm2d(15)
    #    self.conv = nn.Conv2d(15, emb_d,
    #    kernel_size=(PATCH_T,PATCH_X),
    #    stride=(STR_T,STR_X))
    #def forward(self,x):
    #    spec = split_fft(x)                       # (B,15,1000,36)
    #    f = self.conv(self.bn(spec))
    #    return f.flatten(2).transpose(1,2)        # (B,Ntok,emb_d)

# ------------------------------------------------------------
# PixelShuffle ×2  +  Mini-U-Net refine
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class ShuffleUNetRefine(nn.Module):

    def __init__(self, base=32, in_ch=1, out_ch=1):
        super().__init__()
        self.pre = nn.Conv2d(in_ch, base*4, 1)      # *4 for r=2 PixelShuffle
        self.shuffle = nn.PixelShuffle(2)           

        # --- Encoder ---
        self.enc1 = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2)                # 70×70
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base*2, base*2,3, padding=1), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2)                # 35×35
        self.enc3 = nn.Sequential(
            nn.Conv2d(base*2, base*4,3,padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base*4, base*4,3,padding=1), nn.ReLU(inplace=True))

        # --- Decoder ---
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)  # 70×70
        self.dec2 = nn.Sequential(
            nn.Conv2d(base*4, base*2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base*2, base*2,3, padding=1), nn.ReLU(inplace=True))
        self.up1  = nn.ConvTranspose2d(base*2, base, 2, stride=2)    # 140×140
        self.dec1 = nn.Sequential(
            nn.Conv2d(base*2, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True))
        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, coarse):                     # (B,1,70,70)
        x = self.shuffle(self.pre(coarse))         # (B,base,140,140)
        e1 = self.enc1(x)                          # 140×140
        e2 = self.enc2(self.pool1(e1))             # 70×70
        e3 = self.enc3(self.pool2(e2))             # 35×35

        d2 = self.up2(e3)                          # 70×70
        d2 = self.dec2(torch.cat([d2, e2], 1))
        d1 = self.up1(d2)                          # 140×140
        d1 = self.dec1(torch.cat([d1, e1], 1))
        hi = self.outc(d1)                         # (B,1,140,140)

        # Down-average back to 70×70 for supervised loss
        return F.avg_pool2d(hi, kernel_size=2).squeeze(1)  # (B,70,70)

# ---------------- Dual-branch model ----------------
class DualBranchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw  = RawPatchStem(EMB_D)
        self.fft  = FFTPatchStem(EMB_D)
        enc_layer = nn.TransformerEncoderLayer(EMB_D, NHEAD, FF_MULT*EMB_D, dropout=DROP, batch_first=True)
        self.enc_raw = nn.TransformerEncoder(enc_layer, LAYERS)
        self.enc_fft = nn.TransformerEncoder(enc_layer, LAYERS)
        self.cls_raw = nn.Parameter(torch.zeros(1,1,EMB_D))
        self.cls_fft = nn.Parameter(torch.zeros(1,1,EMB_D))
        self.fuse = nn.Sequential(nn.Linear(2*EMB_D, 4*EMB_D),nn.GELU(),nn.Linear(4*EMB_D, 70*70))
        self.refine = ShuffleUNetRefine(base=32)

    def forward(self, s):                         # s (B,5,1000,70)
        B = s.size(0)
        # --- RAW branch
        tok_r = self.raw(s)
        out_r = self.enc_raw(torch.cat([self.cls_raw.expand(B,-1,-1), tok_r],1))[:,0]
        # --- FFT branch
        tok_f = self.fft(s)
        out_f = self.enc_fft(torch.cat([self.cls_fft.expand(B,-1,-1), tok_f],1))[:,0]
        # --- Fuse & reshape
        fused = torch.cat([out_r, out_f], dim=1)      # (B,2D)
        flat  = self.fuse(fused)                      # (B,4900)
        return flat.view(B,70,70)


model = DualBranchModel().to(DEVICE)
LR_INIT   = 1e-4        # keep the same starting LR
LR_MIN    = 2e-5        # final LR after anneal
EPOCHS    = 30  
WEIGHT_DECAY = 5e-4    
# train longer so schedule matters
# ------------------------------------------------------------

# ---------------- Optimiser & Scheduler ---------------------
opt = torch.optim.AdamW(model.parameters(),
                        lr=LR_INIT,
                        weight_decay=WEIGHT_DECAY)

# Cosine decay from epoch 0 -> EPOCHS-1, floor at LR_MIN
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=EPOCHS, eta_min=LR_MIN)
mae   = nn.L1Loss()

# ---------------- Training ----------------
for ep in tqdm(range(1,EPOCHS+1)):
    model.train(); tr=0
    ix = 0
    for s,v in dl_tr:
        ix += 1
        if ix % 100 == 0:
            print(f"{ix}/{len(dl_tr)} in epoch {ep}")
        s,v = s.to(DEVICE), v.to(DEVICE)
        opt.zero_grad(); L = mae(model(s), v); L.backward(); opt.step()
        tr += L.item()*s.size(0)
    tr_mae = tr/len(train_idx)
    scheduler.step()
    model.eval(); va=0
    with torch.no_grad():
        for s,v in dl_va:
            s,v = s.to(DEVICE), v.to(DEVICE)
            va += mae(model(s), v).item()*s.size(0)
    va_mae = va/len(val_idx)
    print(f"Epoch {ep:2d}/{EPOCHS}  "
          f"Train MAE = {tr_mae*V_SCALE:.1f} m/s   "
          f"Val MAE = {va_mae*V_SCALE:.1f} m/s")



!pip uninstall -y torch torchvision torchaudio



!pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121



import torch
print(torch.__version__) 


import sys, types

# Create a fake torch._functorch package
fake_ft = types.ModuleType("torch._functorch")
# Create submodules
fake_eager = types.ModuleType("torch._functorch.eager_transforms")
# stub out the missing grad_and_value
fake_eager.grad_and_value = lambda *args, **kwargs: None
fake_dep = types.ModuleType("torch._functorch.deprecated")
# stub setup_docs to a no-op
fake_dep.setup_docs = lambda *args, **kwargs: None

# Install into sys.modules under all the names PyTorch will look up
sys.modules["torch._functorch"] = fake_ft
sys.modules["torch._functorch.eager_transforms"] = fake_eager
sys.modules["torch._functorch.deprecated"] = fake_dep

# Now it’s safe to import torch
import torch
print("torch version:", torch.__version__)


#!/usr/bin/env python3
"""
dual_branch_metaformer_dense.py
================================
Dual-branch **MetaFormer** encoder + PixelShuffle-U-Net refine.

Key differences from the previous script
----------------------------------------
•  encoder blocks = 6-layer **PoolFormer** (MetaFormer skeleton, pool mixer)
•  stride reduced to **4 × 2**  ->  ~8 k tokens / gather
•  stems keep token-dropout 0.10
•  PixelShuffle×2 + mini-U-Net refine (base 32) still used
•  Cosine LR 30 epochs, weight-decay 5e-4
This fits on a 16 GB GPU at batch = 2 with AMP.

Directory layout unchanged – loader uses mmap index.

"""
import torch._dynamo
torch._dynamo.reset()
torch._dynamo.disable()
import torch
from torch.optim.lr_scheduler import LambdaLR

import os, re, math, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from numpy.lib.stride_tricks import sliding_window_view
from torch.amp import autocast, GradScaler
# ---------------- hyper-parameters ----------------
BASE_DIR = "/kaggle/input/waveform-inversion/train_samples"
PATCH_T, PATCH_X = 32, 16
STR_T,   STR_X   = 4,  2  
NX_TOK = ((70 - PATCH_X) // STR_X) + 1# DENSE TOKENS
EMB_D    = 256
DEPTH    = 6                       # MetaFormer blocks per branch
DROP     = 0.1                     # token dropout
LR_INIT  = 1e-4
LR_MIN   = 2e-5
EPOCHS   = 30
BATCH    = 2                       # keep small for memory
WEIGHT_DECAY = 5e-4
VAL_FRAC = 0.2
V_SCALE  = 4500.0
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.manual_seed(42); np.random.seed(42)
# --------------------------------------------------

# -------- helper: pair files (same as before) -----
pat_data  = re.compile(r'^(data|seis)[_\-]?')
pat_model = re.compile(r'^(model|vel)[_\-]?')

def list_pairs(fam_dir):
    data_dir, model_dir = os.path.join(fam_dir,"data"), os.path.join(fam_dir,"model")
    if os.path.isdir(data_dir) and os.path.isdir(model_dir):          # split layout
        d = sorted(f for f in os.listdir(data_dir)  if f.endswith(".npy"))
        m = sorted(f for f in os.listdir(model_dir) if f.endswith(".npy"))
        return [(os.path.join(data_dir, d[i]), os.path.join(model_dir, m[i])) for i in range(len(d))]
    # merged layout
    files = [f for f in os.listdir(fam_dir) if f.endswith(".npy")]
    seis, vel = {}, {}
    for f in files:
        if f.startswith(("data","seis")):
            seis[pat_data.sub("",f)]  = f
        elif f.startswith(("model","vel")):
            vel[pat_model.sub("",f)]  = f
    return [(os.path.join(fam_dir,seis[k]), os.path.join(fam_dir,vel[k])) for k in sorted(set(seis)&set(vel))]
# --------------------------------------------------

# -------- mmap index dataset (unchanged) ----------
def build_index(base):
    idx=[]
    for fam in sorted(os.listdir(base)):
        fam_dir=os.path.join(base,fam)
        if not os.path.isdir(fam_dir): continue
        for sfile, vfile in list_pairs(fam_dir):
            for i in range(500):
                idx.append((sfile,vfile,i))
    return idx

index_all = build_index(BASE_DIR)
tr_ids, va_ids = train_test_split(np.arange(len(index_all)),
                                  test_size=VAL_FRAC, random_state=42, shuffle=True)

class VelDS(Dataset):
    def __init__(self, idx_list): self.idx = idx_list
    def __len__(self): return len(self.idx)
    def __getitem__(self,k):
        sfile,vfile,i = self.idx[k]
        s  = np.load(sfile, mmap_mode='r')[i].copy().astype(np.float32)   # (5,1000,70)
        v  = np.load(vfile, mmap_mode='r')[i]; v = v.squeeze().copy().astype(np.float32)/V_SCALE
        return torch.from_numpy(s), torch.from_numpy(v)

dl_tr = DataLoader(VelDS([index_all[i] for i in tr_ids]), batch_size=BATCH,
                   shuffle=True, num_workers=4, pin_memory=True)
dl_va = DataLoader(VelDS([index_all[i] for i in va_ids]), batch_size=BATCH,
                   shuffle=False,num_workers=4, pin_memory=True)
# --------------------------------------------------

# ---------------- token stems --------------------
def split_fft(x):
    F = torch.fft.rfft2(x, dim=(-2,-1))
    return torch.cat([F.real, F.imag, torch.log(torch.abs(F)+1e-6)],1)

class StemRaw(nn.Module):
    def __init__(self, emb_d):
        super().__init__()
        self.bn = nn.Identity()
        self.ln = nn.LayerNorm(emb_d)
        self.conv = nn.Conv2d(5, emb_d, (PATCH_T,PATCH_X), stride=(STR_T,STR_X))
        self.drop = nn.Dropout(DROP)
    def forward(self,x):
        f = self.conv(x)                      # (B, emb_d, T, X)
        tok = f.flatten(2).transpose(1,2)     # (B, Ntok, emb_d)
        tok = self.ln(tok)                    # layer-norm per token
        return self.drop(tok)

class StemFFT(nn.Module):
    def __init__(self, emb_d):
        super().__init__()
        self.bn = nn.Identity()
        self.ln = nn.LayerNorm(emb_d)
        self.conv = nn.Conv2d(15, emb_d, (PATCH_T,PATCH_X), stride=(STR_T,STR_X))
        self.drop = nn.Dropout(DROP)
    def forward(self, x):
        # x: (B,5,1000,70)
        spec = split_fft(x)  # (B,15,1000,?) — check dims
        f = self.conv(spec)
        tok = f.flatten(2).transpose(1,2)
        return self.drop(tok)
# --------------------------------------------------

# -------------- MetaFormer (PoolMixer) ------------
NX_TOK = ((70 - PATCH_X) // STR_X) + 1    # = 28  (constant)

class PoolMixer(nn.Module):
    def __init__(self, C, W_tok):
        super().__init__()
        self.W = W_tok                         # patch-grid width

    def forward(self, x):                      # x (B, 1+N, C)
        cls, tok = x[:, :1], x[:, 1:]
        B, Np, C = tok.shape
        W = self.W
        H = Np // W
        assert H * W == Np, f"grid {H}×{W}!={Np}"
        y = tok.transpose(1, 2).reshape(B, C, H, W)
        y = F.avg_pool2d(y, 3, 1, 1) - y
        tok = tok + y.flatten(2).transpose(1, 2)
        return torch.cat([cls, tok], 1)

class MetaBlock(nn.Module):
    def __init__(self, C, W_tok, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(C)
        self.mix   = PoolMixer(C, W_tok)
        self.norm2 = nn.LayerNorm(C)
        self.ffn   = nn.Sequential(
            nn.Linear(C, ff_mult*C),
            nn.GELU(),
            nn.Linear(ff_mult*C, C)
        )
    def forward(self, x):
        x = x + self.mix(self.norm1(x))
        return x + self.ffn(self.norm2(x))

class MetaEncoder(nn.Module):
    def __init__(self, depth: int, C: int, W_tok: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MetaBlock(C, W_tok) for _ in range(depth)]
        )
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
# --------------------------------------------------

# -------------- PixelShuffle + mini-U-Net ---------
class ShuffleUNetRefine(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.pre = nn.Conv2d(1, base*4, 1)
        self.shuffle = nn.PixelShuffle(2)          

        self.enc1 = nn.Sequential(
            nn.Conv2d(base, base,3,padding=1), nn.ReLU(True),
            nn.Conv2d(base, base,3,padding=1), nn.ReLU(True))
        self.pool1= nn.MaxPool2d(2)                # 70×70
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base*2,3,padding=1), nn.ReLU(True),
            nn.Conv2d(base*2, base*2,3,padding=1), nn.ReLU(True))
        self.pool2= nn.MaxPool2d(2)                # 35×35
        self.enc3 = nn.Sequential(
            nn.Conv2d(base*2, base*4,3,padding=1), nn.ReLU(True),
            nn.Conv2d(base*4, base*4,3,padding=1), nn.ReLU(True))

        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2,2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base*4, base*2,3,padding=1), nn.ReLU(True),
            nn.Conv2d(base*2, base*2,3,padding=1), nn.ReLU(True))
        self.up1  = nn.ConvTranspose2d(base*2, base, 2,2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base*2, base,3,padding=1), nn.ReLU(True),
            nn.Conv2d(base, base,3,padding=1), nn.ReLU(True))
        self.outc = nn.Conv2d(base,1,1)

    def forward(self, c):                          # (B,1,70,70)
        x = self.shuffle(self.pre(c))              # (B,base,140,140)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        d2 = self.up2(e3); d2 = self.dec2(torch.cat([d2,e2],1))
        d1 = self.up1(d2); d1 = self.dec1(torch.cat([d1,e1],1))
        hi = self.outc(d1)                         # 140×140
        return F.avg_pool2d(hi,2).squeeze(1)       # (B,70,70)
# --------------------------------------------------

# -------------- Dual-branch MetaFormer model ------
class DualBranchMeta(nn.Module):
    def __init__(self):
        # same init as before, but you can drop cls_r and cls_f if unused
        super().__init__()
        self.raw = StemRaw(EMB_D)
        self.fft = StemFFT(EMB_D)
        # remove or ignore cls tokens:
        # self.cls_r = nn.Parameter(torch.zeros(1,1,EMB_D))
        # self.cls_f = nn.Parameter(torch.zeros(1,1,EMB_D))

        W_RAW = ((70 - PATCH_X)  // STR_X) + 1    # 28
        W_FFT = ((36 - PATCH_X)  // STR_X) + 1    # 11
        self.enc_raw = MetaEncoder(DEPTH, EMB_D, W_RAW)
        self.enc_fft = MetaEncoder(DEPTH, EMB_D, W_FFT)

        # fuse and refine as before
        self.fuse = nn.Sequential(
            nn.LayerNorm(2*EMB_D),
            nn.Linear(2*EMB_D, 4*EMB_D),
            nn.GELU(),
            nn.Linear(4*EMB_D, 70*70)
        )
        self.refine = ShuffleUNetRefine(base=32)

    def forward(self, s):
        B = s.size(0)
        # 1) get token embeddings from stems: shape (B, N_raw, EMB_D)
        r_tok = self.raw(s)
        f_tok = self.fft(s)

        # 2) run through encoder: produce per-token outputs (B, 1+N, EMB_D)? 
        #    But since we removed CLS, we can prepend a small learned vector if desired,
        #    or simply pool after encoding.
        # Here: skip CLS; encode raw tokens directly:
        r_enc = self.enc_raw(torch.cat([torch.zeros(B,1,EMB_D,device=s.device), r_tok], dim=1))
        # or better: pool *before* encoding: mean pool, then encode a single token?
        # Simpler: encode all tokens then mean-pool output tokens:
        r_out = self.enc_raw(torch.cat([torch.zeros(B,1,EMB_D,device=s.device), r_tok], dim=1))  # (B,1+N,EMB_D)
        r_cls = r_out[:,1:].mean(dim=1)  # mean over the raw token outputs

        f_out = self.enc_fft(torch.cat([torch.zeros(B,1,EMB_D,device=s.device), f_tok], dim=1))
        f_cls = f_out[:,1:].mean(dim=1)

        # 3) fuse and refine
        fused = torch.cat([r_cls, f_cls], dim=1)  # (B, 2*EMB_D)
        coarse = self.fuse(fused).view(B,1,70,70)
        return self.refine(coarse)
# --------------------------------------------------


import math, torch.optim.lr_scheduler as tls

DEVICE        = torch.device("cuda")
LR_CONST      = 1e-4
EPOCHS        = 30
ACCUM_STEPS   = 1
CLIP_NORM     = 100.0
DROP          = 0.0
V_SCALE       = 4500.0

warmup_steps = 2000
# instantiate
model = DualBranchMeta().to(DEVICE)
optimizer   = torch.optim.AdamW(model.parameters(), lr=LR_CONST, weight_decay=1e-2)

def lr_warmup_cosine(step):
    if step < warmup_steps:
        return float(step + 1) / warmup_steps
    else:
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        # cosine from 1 -> 0
        return 0.5 * (1 + math.cos(math.pi * progress))
scheduler = LambdaLR(optimizer, lr_warmup_cosine)

scaler= GradScaler("cuda")
mae = nn.L1Loss(reduction='mean')
steps_per_epoch = len(dl_tr)
total_steps = EPOCHS * steps_per_epoch
global_step = 0
for ep in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for step, (seis, vel) in enumerate(dl_tr):
        seis, vel = seis.to(DEVICE), vel.to(DEVICE)
        optimizer.zero_grad()

        pred = model(seis)
        loss = mae(pred, vel) * V_SCALE
        loss.backward()
        if global_step % 1000 == 0:
            grads = [p.grad.norm() for p in model.parameters() if p.grad is not None]
            raw_norm = torch.norm(torch.stack(grads), 2).item()
            print(f"Raw grad norm before clipping: {raw_norm:.2f}")
        # now clip:

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()

        # 4) Step the scheduler
        scheduler.step()
        global_step += 1

        running_loss += loss.item() * seis.size(0)

        # (Optional) print every N updates
        if global_step % 1000 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"GlobalStep {global_step:5d}  LR={current_lr:.2e}  loss={loss.item():.1f}")

    train_mae = running_loss / len(dl_tr.dataset)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for seis, vel in dl_va:
            seis, vel = seis.to(DEVICE), vel.to(DEVICE)
            pred = model(seis)
            loss_v = mae(pred, vel) * V_SCALE
            val_loss += loss_v.item() * seis.size(0)
    val_mae = val_loss / len(dl_va.dataset)
    print(f"Epoch {ep:02d}/{EPOCHS}  "
          f"Train MAE: {train_mae:6.1f} m/s   "
          f"Val MAE:   {val_mae:6.1f} m/s")


for name, mod in model.named_modules():
    if isinstance(mod, nn.GroupNorm):
        dev = next(mod.parameters()).device
        print(f"{name:30s} -> {type(mod).__name__:10s} on {dev}")


ix += 1
        if ix % 200 == 0:
            print(f"{ix}/{len(dl_tr)} at epoch {ep}")


for name, param in model.named_parameters():
    if 'bn' in name or 'norm' in name:
        print(name, param.device)


import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import numpy as np
import re

import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from numpy.lib.stride_tricks import sliding_window_view
from torch.amp import autocast, GradScaler

# ───────────────────────── Hyperparameters ─────────────────────────
PATCH_T, PATCH_X = 32, 16
STR_T,   STR_X   = 4,  2  
NX_TOK = ((70 - PATCH_X) // STR_X) + 1# DENSE TOKENS
EMB_D    = 256
DEPTH    = 6                       # MetaFormer blocks per branch
DROP     = 0.1 


BASE_DIR = "/kaggle/input/waveform-inversion/train_samples"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS       = 30
BATCH   = 2            # adjust to GPU memory
WEIGHT_DECAY = 5e-4

# LR schedule parameters
LR_START     = 1e-6         # initial LR at step 0
LR_BASE      = 1e-4         # target LR after warm-up
LR_MIN       = 2e-5         # final LR after cosine anneal
warmup_steps = 2000         # number of optimizer steps to ramp LR_START -> LR_BASE

CLIP_NORM    = 200.0        # gradient clipping threshold
V_SCALE      = 4500.0       # velocity scaling: vel_norm = vel_raw / V_SCALE
VAL_FRAC = 0.2
# Loss weights and Huber beta
beta_norm    = 0.02         # in normalized units; corresponds ~0.02 * V_SCALE ~= 90 m/s
w_full_base  = 1.0
w_ds_base    = 0.5
w_tv_base    = 1e-3
# ──────────────────────────────────────────────────────────────────


pat_data  = re.compile(r'^(data|seis)[_\-]?')
pat_model = re.compile(r'^(model|vel)[_\-]?')

def list_pairs(fam_dir):
    data_dir, model_dir = os.path.join(fam_dir,"data"), os.path.join(fam_dir,"model")
    if os.path.isdir(data_dir) and os.path.isdir(model_dir):          # split layout
        d = sorted(f for f in os.listdir(data_dir)  if f.endswith(".npy"))
        m = sorted(f for f in os.listdir(model_dir) if f.endswith(".npy"))
        return [(os.path.join(data_dir, d[i]), os.path.join(model_dir, m[i])) for i in range(len(d))]
    # merged layout
    files = [f for f in os.listdir(fam_dir) if f.endswith(".npy")]
    seis, vel = {}, {}
    for f in files:
        if f.startswith(("data","seis")):
            seis[pat_data.sub("",f)]  = f
        elif f.startswith(("model","vel")):
            vel[pat_model.sub("",f)]  = f
    return [(os.path.join(fam_dir,seis[k]), os.path.join(fam_dir,vel[k])) for k in sorted(set(seis)&set(vel))]
# --------------------------------------------------

# -------- mmap index dataset (unchanged) ----------
def build_index(base):
    idx=[]
    for fam in sorted(os.listdir(base)):
        fam_dir=os.path.join(base,fam)
        if not os.path.isdir(fam_dir): continue
        for sfile, vfile in list_pairs(fam_dir):
            for i in range(500):
                idx.append((sfile,vfile,i))
    return idx

index_all = build_index(BASE_DIR)
tr_ids, va_ids = train_test_split(np.arange(len(index_all)),
                                  test_size=VAL_FRAC, random_state=42, shuffle=True)

class VelDS(Dataset):
    def __init__(self, idx_list): self.idx = idx_list
    def __len__(self): return len(self.idx)
    def __getitem__(self,k):
        sfile,vfile,i = self.idx[k]
        s  = np.load(sfile, mmap_mode='r')[i].copy().astype(np.float32)   # (5,1000,70)
        v  = np.load(vfile, mmap_mode='r')[i]; v = v.squeeze().copy().astype(np.float32)/V_SCALE
        return torch.from_numpy(s), torch.from_numpy(v)

dl_tr = DataLoader(VelDS([index_all[i] for i in tr_ids]), batch_size=BATCH,
                   shuffle=True, num_workers=4, pin_memory=True)
dl_va = DataLoader(VelDS([index_all[i] for i in va_ids]), batch_size=BATCH,
                   shuffle=False,num_workers=4, pin_memory=True)
# --------------------------------------------------

# ---------------- token stems --------------------
def split_fft(x):
    F = torch.fft.rfft2(x, dim=(-2,-1))
    return torch.cat([F.real, F.imag, torch.log(torch.abs(F)+1e-6)],1)

class StemRaw(nn.Module):
    def __init__(self, emb_d):
        super().__init__()
        self.bn = nn.Identity()
        self.ln = nn.LayerNorm(emb_d)
        self.conv = nn.Conv2d(5, emb_d, (PATCH_T,PATCH_X), stride=(STR_T,STR_X))
        self.drop = nn.Dropout(DROP)
    def forward(self,x):
        f = self.conv(x)                      # (B, emb_d, T, X)
        tok = f.flatten(2).transpose(1,2)     # (B, Ntok, emb_d)
        tok = self.ln(tok)                    # layer-norm per token
        return self.drop(tok)

class StemFFT(nn.Module):
    def __init__(self, emb_d):
        super().__init__()
        self.bn = nn.Identity()
        self.ln = nn.LayerNorm(emb_d)
        self.conv = nn.Conv2d(15, emb_d, (PATCH_T,PATCH_X), stride=(STR_T,STR_X))
        self.drop = nn.Dropout(DROP)
    def forward(self, x):
        # x: (B,5,1000,70)
        spec = split_fft(x)  # (B,15,1000,?) — check dims
        f = self.conv(spec)
        tok = f.flatten(2).transpose(1,2)
        return self.drop(tok)
# --------------------------------------------------

# -------------- MetaFormer (PoolMixer) ------------
NX_TOK = ((70 - PATCH_X) // STR_X) + 1    # = 28  (constant)

class PoolMixer(nn.Module):
    def __init__(self, C, W_tok):
        super().__init__()
        self.W = W_tok                         # patch-grid width

    def forward(self, x):                      # x (B, 1+N, C)
        cls, tok = x[:, :1], x[:, 1:]
        B, Np, C = tok.shape
        W = self.W
        H = Np // W
        assert H * W == Np, f"grid {H}×{W}!={Np}"
        y = tok.transpose(1, 2).reshape(B, C, H, W)
        y = F.avg_pool2d(y, 3, 1, 1) - y
        tok = tok + y.flatten(2).transpose(1, 2)
        return torch.cat([cls, tok], 1)

class MetaBlock(nn.Module):
    def __init__(self, C, W_tok, ff_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(C)
        self.mix   = PoolMixer(C, W_tok)
        self.norm2 = nn.LayerNorm(C)
        self.ffn   = nn.Sequential(
            nn.Linear(C, ff_mult*C),
            nn.GELU(),
            nn.Linear(ff_mult*C, C)
        )
    def forward(self, x):
        x = x + self.mix(self.norm1(x))
        return x + self.ffn(self.norm2(x))

class MetaEncoder(nn.Module):
    def __init__(self, depth: int, C: int, W_tok: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MetaBlock(C, W_tok) for _ in range(depth)]
        )
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
# --------------------------------------------------

# -------------- PixelShuffle + mini-U-Net ---------
class ShuffleUNetRefine(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.pre = nn.Conv2d(1, base*4, 1)
        self.shuffle = nn.PixelShuffle(2)          # 70×70 -> 140×140

        self.enc1 = nn.Sequential(
            nn.Conv2d(base, base,3,padding=1), nn.ReLU(True),
            nn.Conv2d(base, base,3,padding=1), nn.ReLU(True))
        self.pool1= nn.MaxPool2d(2)                # 70×70
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base*2,3,padding=1), nn.ReLU(True),
            nn.Conv2d(base*2, base*2,3,padding=1), nn.ReLU(True))
        self.pool2= nn.MaxPool2d(2)                # 35×35
        self.enc3 = nn.Sequential(
            nn.Conv2d(base*2, base*4,3,padding=1), nn.ReLU(True),
            nn.Conv2d(base*4, base*4,3,padding=1), nn.ReLU(True))

        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2,2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base*4, base*2,3,padding=1), nn.ReLU(True),
            nn.Conv2d(base*2, base*2,3,padding=1), nn.ReLU(True))
        self.up1  = nn.ConvTranspose2d(base*2, base, 2,2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base*2, base,3,padding=1), nn.ReLU(True),
            nn.Conv2d(base, base,3,padding=1), nn.ReLU(True))
        self.outc = nn.Conv2d(base,1,1)

    def forward(self, c):                          # (B,1,70,70)
        x = self.shuffle(self.pre(c))              # (B,base,140,140)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        d2 = self.up2(e3); d2 = self.dec2(torch.cat([d2,e2],1))
        d1 = self.up1(d2); d1 = self.dec1(torch.cat([d1,e1],1))
        hi = self.outc(d1)                         # 140×140
        return F.avg_pool2d(hi,2).squeeze(1)       # (B,70,70)
# --------------------------------------------------

# -------------- Dual-branch MetaFormer model ------
class DualBranchMeta(nn.Module):
    def __init__(self):
        # same init as before, but you can drop cls_r and cls_f if unused
        super().__init__()
        self.raw = StemRaw(EMB_D)
        self.fft = StemFFT(EMB_D)
        # remove or ignore cls tokens:
        # self.cls_r = nn.Parameter(torch.zeros(1,1,EMB_D))
        # self.cls_f = nn.Parameter(torch.zeros(1,1,EMB_D))

        W_RAW = ((70 - PATCH_X)  // STR_X) + 1    # 28
        W_FFT = ((36 - PATCH_X)  // STR_X) + 1    # 11
        self.enc_raw = MetaEncoder(DEPTH, EMB_D, W_RAW)
        self.enc_fft = MetaEncoder(DEPTH, EMB_D, W_FFT)

        # fuse and refine as before
        self.fuse = nn.Sequential(
            nn.LayerNorm(2*EMB_D),
            nn.Linear(2*EMB_D, 4*EMB_D),
            nn.GELU(),
            nn.Linear(4*EMB_D, 70*70)
        )
        self.refine = ShuffleUNetRefine(base=32)

    def forward(self, s):
        B = s.size(0)
        # 1) get token embeddings from stems: shape (B, N_raw, EMB_D)
        r_tok = self.raw(s)
        f_tok = self.fft(s)

        # 2) run through encoder: produce per-token outputs (B, 1+N, EMB_D)? 
        #    But since we removed CLS, we can prepend a small learned vector if desired,
        #    or simply pool after encoding.
        # Here: skip CLS; encode raw tokens directly:
        r_enc = self.enc_raw(torch.cat([torch.zeros(B,1,EMB_D,device=s.device), r_tok], dim=1))
        # or better: pool *before* encoding: mean pool, then encode a single token?
        # Simpler: encode all tokens then mean-pool output tokens:
        r_out = self.enc_raw(torch.cat([torch.zeros(B,1,EMB_D,device=s.device), r_tok], dim=1))  # (B,1+N,EMB_D)
        r_cls = r_out[:,1:].mean(dim=1)  # mean over the raw token outputs

        f_out = self.enc_fft(torch.cat([torch.zeros(B,1,EMB_D,device=s.device), f_tok], dim=1))
        f_cls = f_out[:,1:].mean(dim=1)

        # 3) fuse and refine
        fused = torch.cat([r_cls, f_cls], dim=1)  # (B, 2*EMB_D)
        coarse = self.fuse(fused).view(B,1,70,70)
        return self.refine(coarse)
# --------------------------------------------------



model = DualBranchMeta().to(DEVICE)


print("Computing mean normalized velocity for bias init...")
sum_vel = 0.0
count = 0
with torch.no_grad():
    for _, vel in dl_tr:
        # vel: (B, H, W), normalized in [0,1]
        sum_vel += vel.sum().item()
        count += vel.numel()
mean_norm = sum_vel / count
print(f"  mean normalized velocity ≈ {mean_norm:.4f} (=> raw ≈ {mean_norm*V_SCALE:.1f} m/s)")


import math

try:
    b_init = math.log(mean_norm / (1.0 - mean_norm))
    model.refine.outc.bias.data.fill_(b_init)
    print(f"Initialized final bias to {b_init:.4f} for sigmoid->{mean_norm:.4f}")
except Exception:
    # If no sigmoid, set direct bias to mean_norm
    try:
        model.refine.outc.bias.data.fill_(mean_norm)
        print(f"Initialized final bias directly to normalized mean {mean_norm:.4f}")
    except Exception:
        print("Warning: could not initialize final bias automatically; please check final layer.")

# 3) Setup optimizer and LR scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LR_BASE, weight_decay=WEIGHT_DECAY)

steps_per_epoch = len(dl_tr)
total_steps = EPOCHS * steps_per_epoch

def lr_schedule(step):
    # Linear warm-up from LR_START -> LR_BASE
    if step < warmup_steps:
        frac = float(step + 1) / float(warmup_steps)
        lr = LR_START + frac * (LR_BASE - LR_START)
    else:
        # Cosine anneal from LR_BASE -> LR_MIN over remaining steps
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        lr = LR_MIN + (LR_BASE - LR_MIN) * cosine
    # LambdaLR expects multiplier relative to LR_BASE
    return lr / LR_BASE

scheduler = LambdaLR(optimizer, lr_schedule)

# 4) Define loss criterion
criterion_data = nn.SmoothL1Loss(beta=beta_norm)  # in normalized units

# 5) Training loop
global_step = 0
for ep in range(EPOCHS):
    model.train()
    running_loss_full = 0.0

    # Adjust weights schedules per epoch
    # Multi-scale weight decays from w_ds_base  0.1*w_ds_base over first 10 epochs:
    w_ds = w_ds_base * (1.0 - 0.8 * min(ep, 10) / 10.0)
    # TV weight ramps up from 0  w_tv_base over first 5 epochs:
    w_tv = w_tv_base * min(ep / 5.0, 1.0)

    print(f"\nEpoch {ep:02d}: w_ds={w_ds:.3f}, w_tv={w_tv:.3e}")

    for step, (seis, vel) in enumerate(dl_tr):
        seis, vel = seis.to(DEVICE), vel.to(DEVICE)  # seis: (B,5,1000,70), vel: (B,70,70)

        optimizer.zero_grad()

        # Forward
        pred = model(seis)  # expected normalized output in [0,1]

        # 5.1 Data term full resolution
        loss_full = criterion_data(pred, vel) * V_SCALE

        # 5.2 Multi-scale term (downsample ×2)
        pred_ds = F.avg_pool2d(pred.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)  # (B,35,35)
        vel_ds  = F.avg_pool2d(vel.unsqueeze(1),  kernel_size=2, stride=2).squeeze(1)
        loss_ds = criterion_data(pred_ds, vel_ds) * V_SCALE

        # 5.3 TV smoothness term
        # Absolute differences between neighbors
        dx = torch.abs(pred[:, :, 1:] - pred[:, :, :-1]).mean()
        dy = torch.abs(pred[:, 1:, :] - pred[:, :-1, :]).mean()
        loss_tv = (dx + dy) * V_SCALE

        # 5.4 Combine losses
        loss = w_full_base * loss_full + w_ds * loss_ds + w_tv * loss_tv

        # Backward + clip + step + scheduler
        loss.backward()
        # (Optional) inspect raw grad norm before clipping:
        if (global_step) % 1000 == 0:
            grads = [p.grad.norm() for p in model.parameters() if p.grad is not None]
            raw_norm = torch.norm(torch.stack(grads), 2).item()
            print(f"Step {global_step} raw grad norm before clip: {raw_norm:.1f}")
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()
        scheduler.step()
        global_step += 1

        running_loss_full += loss_full.item() * seis.size(0)

        # Diagnostic print every 100 updates
        

    # Epoch-end metrics
    train_mae = running_loss_full / len(dl_tr.dataset)

    # Validation (only full-res data term for MAE)
    model.eval()
    val_loss_full = 0.0
    with torch.no_grad():
        for seis, vel in dl_va:
            seis, vel = seis.to(DEVICE), vel.to(DEVICE)
            pred = model(seis)
            loss_v = criterion_data(pred, vel) * V_SCALE
            val_loss_full += loss_v.item() * seis.size(0)
    val_mae = val_loss_full / len(dl_va.dataset)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {ep:02d}/{EPOCHS}  Train MAE: {train_mae:6.1f} m/s   "
          f"Val MAE: {val_mae:6.1f} m/s   LR={current_lr:.2e}")

# After training, save model
torch.save(model.state_dict(), "inversion_model_final.pth")
print("Model saved to inversion_model_final.pth")




#!/usr/bin/env python3
"""
dual_branch_metaformer_dense.py
================================
Dual-branch MetaFormer encoder + PixelShuffle-U-Net refine with
normalization before refine, lowered initial loss-scale, optional refine freezing,
and gradient diagnostics.

Adjust hyperparameters (loss_scale_start, warmup_steps, CLIP_NORM, etc.) as needed.
"""

import torch._dynamo
torch._dynamo.reset()
torch._dynamo.disable()

import os, re, math, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.amp import autocast, GradScaler

# ---------------- hyper-parameters ----------------
BASE_DIR = "/kaggle/input/waveform-inversion/train_samples"
PATCH_T, PATCH_X = 32, 16
STR_T,   STR_X   = 4,  2  
EMB_D    = 256
DEPTH    = 6                       # MetaFormer blocks per branch
DROP     = 0.1                     # token dropout in stems

# LR schedule
LR_START = 1e-6                   # for warm-up start
LR_BASE  = 1e-4
LR_MIN   = 2e-5
warmup_steps = 5000               # steps to ramp LR_START->LR_BASE
EPOCHS   = 30
BATCH    = 2                      # keep small for memory
WEIGHT_DECAY = 5e-4
VAL_FRAC = 0.2
V_SCALE  = 4500.0                 # velocity scaling
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.manual_seed(42); np.random.seed(42)

# Dynamic loss-scale ramp parameters
loss_scale_start        = 10.0      # lower initial multiplier
loss_scale_target       = V_SCALE
loss_scale_warmup_steps = 5000

# Data augmentation
noise_std = 0.01     # additive Gaussian noise on seismic input (normalized units)
amp_range = 0.05     # random amplitude scaling ±5%

# Gradient clipping
CLIP_NORM = 300.0    # global norm clipping threshold
adaptive_clip_coef = 0.0  # 0 to disable adaptive clipping

# Loss: SmoothL1 with beta in normalized units
beta_norm = 0.02
criterion_smoothl1 = nn.SmoothL1Loss(beta=beta_norm)
# --------------------------------------------------

# -------- helper: pair files (same as before) -----
pat_data  = re.compile(r'^(data|seis)[_\-]?')
pat_model = re.compile(r'^(model|vel)[_\-]?')

def list_pairs(fam_dir):
    data_dir, model_dir = os.path.join(fam_dir,"data"), os.path.join(fam_dir,"model")
    if os.path.isdir(data_dir) and os.path.isdir(model_dir):
        d = sorted(f for f in os.listdir(data_dir)  if f.endswith(".npy"))
        m = sorted(f for f in os.listdir(model_dir) if f.endswith(".npy"))
        return [(os.path.join(data_dir, d[i]), os.path.join(model_dir, m[i])) for i in range(len(d))]
    files = [f for f in os.listdir(fam_dir) if f.endswith(".npy")]
    seis, vel = {}, {}
    for f in files:
        if f.startswith(("data","seis")):
            seis[pat_data.sub("",f)]  = f
        elif f.startswith(("model","vel")):
            vel[pat_model.sub("",f)]  = f
    return [(os.path.join(fam_dir,seis[k]), os.path.join(fam_dir,vel[k])) for k in sorted(set(seis)&set(vel))]
# --------------------------------------------------

# -------- mmap index dataset (unchanged) ----------
def build_index(base):
    idx = []
    for fam in sorted(os.listdir(base)):
        fam_dir = os.path.join(base, fam)
        if not os.path.isdir(fam_dir): continue
        for sfile, vfile in list_pairs(fam_dir):
            for i in range(500):
                idx.append((sfile, vfile, i))
    return idx

index_all = build_index(BASE_DIR)
tr_ids, va_ids = train_test_split(np.arange(len(index_all)),
                                  test_size=VAL_FRAC, random_state=42, shuffle=True)

class VelDS(Dataset):
    def __init__(self, idx_list):
        self.idx = idx_list
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, k):
        sfile, vfile, i = self.idx[k]
        s = np.load(sfile, mmap_mode='r')[i].copy().astype(np.float32)   # (5,1000,70)
        v = np.load(vfile, mmap_mode='r')[i]
        v = v.squeeze().copy().astype(np.float32) / V_SCALE
        return torch.from_numpy(s), torch.from_numpy(v)

dl_tr = DataLoader(VelDS([index_all[i] for i in tr_ids]), batch_size=BATCH,
                   shuffle=True, num_workers=4, pin_memory=True)
dl_va = DataLoader(VelDS([index_all[i] for i in va_ids]), batch_size=BATCH,
                   shuffle=False, num_workers=4, pin_memory=True)
# --------------------------------------------------

# ---------------- token stems --------------------
def split_fft(x):
    # x: (B,5,1000,70)
    F = torch.fft.rfft2(x, dim=(-2,-1))  # yields (B,5,1000,36)
    return torch.cat([F.real, F.imag, torch.log(torch.abs(F)+1e-6)], dim=1)

class StemRaw(nn.Module):
    def __init__(self, emb_d):
        super().__init__()
        self.ln = nn.LayerNorm(emb_d)
        self.conv = nn.Conv2d(5, emb_d, (PATCH_T, PATCH_X), stride=(STR_T, STR_X))
        self.drop = nn.Dropout(DROP)
    def forward(self, x):
        f = self.conv(x)                      # (B, emb_d, T', X')
        tok = f.flatten(2).transpose(1,2)     # (B, Ntok, emb_d)
        tok = self.ln(tok)
        return self.drop(tok)

class StemFFT(nn.Module):
    def __init__(self, emb_d):
        super().__init__()
        self.ln = nn.LayerNorm(emb_d)
        self.conv = nn.Conv2d(15, emb_d, (PATCH_T, PATCH_X), stride=(STR_T, STR_X))
        self.drop = nn.Dropout(DROP)
    def forward(self, x):
        spec = split_fft(x)                   # (B,15,1000,36)
        f = self.conv(spec)                   # (B, emb_d, T', X')
        tok = f.flatten(2).transpose(1,2)     
        tok = self.ln(tok)
        return self.drop(tok)
# --------------------------------------------------

# -------------- MetaFormer (PoolMixer) ------------
class PoolMixer(nn.Module):
    def __init__(self, C, W_tok):
        super().__init__()
        self.W = W_tok
    def forward(self, x):
        # x: (B, 1+N, C)
        cls, tok = x[:, :1], x[:, 1:]
        B, Np, C = tok.shape
        W = self.W
        H = Np // W
        assert H * W == Np, f"grid {H}×{W}!={Np}"
        y = tok.transpose(1, 2).reshape(B, C, H, W)
        y = F.avg_pool2d(y, 3, 1, 1) - y
        tok2 = tok + y.flatten(2).transpose(1, 2)
        return torch.cat([cls, tok2], dim=1)

class MetaBlock(nn.Module):
    def __init__(self, C, W_tok, ff_mult=4, residual_scale=1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(C)
        self.mix   = PoolMixer(C, W_tok)
        self.norm2 = nn.LayerNorm(C)
        self.ffn   = nn.Sequential(
            nn.Linear(C, ff_mult*C),
            nn.GELU(),
            nn.Linear(ff_mult*C, C)
        )
        self.residual_scale = residual_scale
    def forward(self, x):
        y1 = self.mix(self.norm1(x))
        x = x + self.residual_scale * y1
        y2 = self.ffn(self.norm2(x))
        return x + self.residual_scale * y2

class MetaEncoder(nn.Module):
    def __init__(self, depth: int, C: int, W_tok: int, residual_scale=1.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MetaBlock(C, W_tok, residual_scale=residual_scale) for _ in range(depth)]
        )
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
# --------------------------------------------------

# -------------- PixelShuffle + mini-U-Net ---------
class ShuffleUNetRefine(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.pre = nn.Conv2d(1, base*4, 1)
        self.pre_gn = nn.GroupNorm(num_groups=1, num_channels=base*4)
        self.shuffle = nn.PixelShuffle(2)          
        self.enc1 = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base),
            nn.ReLU(True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base),
            nn.ReLU(True),
        )
        self.pool1 = nn.MaxPool2d(2)                
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base*2),
            nn.ReLU(True),
            nn.Conv2d(base*2, base*2, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base*2),
            nn.ReLU(True),
        )
        self.pool2 = nn.MaxPool2d(2)                
        self.enc3 = nn.Sequential(
            nn.Conv2d(base*2, base*4, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base*4),
            nn.ReLU(True),
            nn.Conv2d(base*4, base*4, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base*4),
            nn.ReLU(True),
        )
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2,2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base*4, base*2, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base*2),
            nn.ReLU(True),
            nn.Conv2d(base*2, base*2, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base*2),
            nn.ReLU(True),
        )
        self.up1 = nn.ConvTranspose2d(base*2, base, 2,2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base*2, base, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base),
            nn.ReLU(True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base),
            nn.ReLU(True),
        )
        self.outc = nn.Conv2d(base, 1, 1)

    def forward(self, c):
        x = self.shuffle(self.pre_gn(self.pre(c)))       
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        d2 = self.up2(e3); d2 = self.dec2(torch.cat([d2,e2], dim=1))
        d1 = self.up1(d2); d1 = self.dec1(torch.cat([d1,e1], dim=1))
        hi = self.outc(d1)                 
        return F.avg_pool2d(hi, 2).squeeze(1)  # (B,70,70)
# --------------------------------------------------

# -------------- Dual-branch MetaFormer model ------
class DualBranchMeta(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw = StemRaw(EMB_D)
        self.fft = StemFFT(EMB_D)

        # Compute W_RAW and W_FFT correctly:
        W_RAW = (70 - PATCH_X)//STR_X + 1   # 28
        freq_dim = 70//2 + 1               # 36
        W_FFT = (freq_dim - PATCH_X)//STR_X + 1  # 11

        residual_scale = 1.0
        self.enc_raw = MetaEncoder(DEPTH, EMB_D, W_RAW, residual_scale=residual_scale)
        self.enc_fft = MetaEncoder(DEPTH, EMB_D, W_FFT, residual_scale=residual_scale)

        self.fuse = nn.Sequential(
            nn.LayerNorm(2*EMB_D),
            nn.Linear(2*EMB_D, 4*EMB_D),
            nn.GELU(),
            nn.Linear(4*EMB_D, 70*70)
        )
        # normalize coarse before refine
        self.coarse_norm = nn.GroupNorm(num_groups=1, num_channels=1)
        self.refine = ShuffleUNetRefine(base=32)

    def forward(self, s):
        # s: (B,5,1000,70)
        B = s.size(0)
        r_tok = self.raw(s)
        f_tok = self.fft(s)
        zero_raw = torch.zeros(B, 1, EMB_D, device=s.device, dtype=s.dtype)
        zero_fft = torch.zeros(B, 1, EMB_D, device=s.device, dtype=s.dtype)
        r_all = torch.cat([zero_raw, r_tok], dim=1)
        f_all = torch.cat([zero_fft, f_tok], dim=1)
        r_enc = self.enc_raw(r_all)
        f_enc = self.enc_fft(f_all)
        r_cls = r_enc[:, 1:, :].mean(dim=1)
        f_cls = f_enc[:, 1:, :].mean(dim=1)
        fused = torch.cat([r_cls, f_cls], dim=1)
        coarse = self.fuse(fused).view(B, 1, 70, 70)
        # normalize coarse
        coarse = self.coarse_norm(coarse)
        return self.refine(coarse)
# --------------------------------------------------

# -------------- Training setup -----------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DualBranchMeta().to(DEVICE)

# 1) Compute mean normalized velocity for bias init
print("Computing mean normalized velocity for bias init...")
sum_vel = 0.0
count = 0
with torch.no_grad():
    for _, vel in dl_tr:
        sum_vel += vel.sum().item()
        count += vel.numel()
mean_norm = sum_vel / count
print(f"  mean normalized velocity ≈ {mean_norm:.4f} (raw ≈ {mean_norm*V_SCALE:.1f} m/s)")

# 2) Initialize final-layer bias
try:
    model.refine.outc.bias.data.fill_(mean_norm)
    print(f"Initialized final bias to {mean_norm:.4f} (normalized)")
except Exception:
    print("Warning: could not initialize final bias automatically; check final layer.")

# 3) Optimizer and LR scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LR_BASE, weight_decay=WEIGHT_DECAY)
steps_per_epoch = len(dl_tr)
total_steps = EPOCHS * steps_per_epoch

def lr_schedule(step):
    if step < warmup_steps:
        return (LR_START + (LR_BASE - LR_START) * (step + 1) / warmup_steps) / LR_BASE
    else:
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        lr = LR_MIN + (LR_BASE - LR_MIN) * cosine
        return lr / LR_BASE

scheduler = LambdaLR(optimizer, lr_schedule)
scheduler_plateau = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=0.5,
                                      patience=3,
                                      min_lr=1e-6,
                                      verbose=True)

def adaptive_clip(model, clip_coef=adaptive_clip_coef, eps=1e-6):
    if clip_coef <= 0: return
    for p in model.parameters():
        if p.grad is None: continue
        param_norm = torch.norm(p.detach())
        grad_norm = torch.norm(p.grad.detach())
        max_norm = clip_coef * (param_norm + eps)
        if grad_norm > max_norm:
            p.grad.mul_(max_norm / (grad_norm + eps))

# 4) Training loop
# ... [model, optimizer, scheduler, dl_tr, etc. already defined] ...

ACCUM_STEPS = 4  
CLIP_NORM = 300.0
loss_scale_start = 10.0
loss_scale_target = V_SCALE
loss_scale_warmup_steps = 5000

scaler = GradScaler()
mae = nn.L1Loss(reduction='mean')
global_step = 0
running_mae = 0.0
best_val_mae = float('inf')
patience_counter = 0

for ep in range(EPOCHS):
    model.train()
    running_mae = 0.0
    print(f"\nEpoch {ep:02d}:")
    # Optionally freeze refine early
    if ep < 2:
        for p in model.refine.parameters(): p.requires_grad=False
        print("  Freezing refine stage this epoch")
    else:
        for p in model.refine.parameters(): p.requires_grad=True

    optimizer.zero_grad()
    accum_counter = 0

    for step, (seis, vel) in enumerate(dl_tr):
        seis, vel = seis.to(DEVICE), vel.to(DEVICE)
        # Data augmentation as before
        if noise_std > 0:
            seis = seis + torch.randn_like(seis) * noise_std
        if amp_range > 0:
            scale = 1.0 + (torch.rand(seis.size(0),1,1,1,device=DEVICE)*2 -1)*amp_range
            seis = seis * scale

        with autocast('cuda'):
            pred = model(seis)  # normalized output
            # track MAE for reporting
            with torch.no_grad():
                batch_mae = mae(pred, vel) * V_SCALE
                running_mae += batch_mae.item() * seis.size(0)

            # dynamic loss-scale
            if global_step < loss_scale_warmup_steps:
                frac = float(global_step+1)/loss_scale_warmup_steps
                loss_scale = loss_scale_start + frac*(loss_scale_target - loss_scale_start)
            else:
                loss_scale = loss_scale_target

            loss = criterion_smoothl1(pred, vel) * loss_scale
            # divide loss for accumulation
            loss = loss / ACCUM_STEPS

        scaler.scale(loss).backward()
        accum_counter += 1

        if accum_counter == ACCUM_STEPS:
            # unscale & clip once per accumulation
            scaler.unscale_(optimizer)
            # optional adaptive clipping
            # adaptive_clip(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # LR scheduler step once per optimizer step
            scheduler.step()
            global_step += 1
            accum_counter = 0

            # Diagnostics every 500 updates:
            if global_step % 500 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"[Step {global_step:5d}] LR={current_lr:.2e}, loss_scale={loss_scale:.1f}")
                # Log gradient stats:
                p_vals = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
                if p_vals:
                    t = torch.tensor(p_vals, device='cpu')
                    print(f"  Grad norms -> overall L2: {t.norm(p=2).item():.3e}, mean: {t.mean().item():.3e}, std: {t.std(unbiased=False).item():.3e}, min: {t.min().item():.3e}, max: {t.max().item():.3e}")
                print(f"  Recent batch MAE: {batch_mae.item():.1f} m/s")

    # If final accumulation not exactly divisible, step once more:
    if accum_counter > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        global_step += 1

    train_mae = running_mae / len(dl_tr.dataset)

    # Validation as before:
    model.eval()
    val_mae_accum = 0.0
    with torch.no_grad():
        for seis, vel in dl_va:
            seis, vel = seis.to(DEVICE), vel.to(DEVICE)
            pred = model(seis)
            val_mae_accum += mae(pred, vel).item() * V_SCALE * seis.size(0)
    val_mae = val_mae_accum / len(dl_va.dataset)

    print(f"Epoch {ep:02d}/{EPOCHS}  Train MAE: {train_mae:6.1f} m/s   Val MAE: {val_mae:6.1f} m/s")

    # LR plateau after warm-up
    if global_step >= warmup_steps:
        scheduler_plateau.step(val_mae)

    # Early stopping
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), "best_model.pth")
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= 5:
        print("No improvement for 5 epochs, stopping early.")
        break



import os, re, math, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.amp import autocast, GradScaler

BASE_DIR = "/kaggle/input/waveform-inversion/train_samples"
VAL_FRAC = 0.2
pat_data  = re.compile(r'^(data|seis)[_\-]?')
pat_model = re.compile(r'^(model|vel)[_\-]?')
BASE_DIR = "/kaggle/input/waveform-inversion/train_samples"
PATCH_T, PATCH_X = 32, 16
STR_T,   STR_X   = 4,  2  
EMB_D    = 256
DEPTH    = 6                       # MetaFormer blocks per branch
DROP     = 0.1                     # token dropout in stems

# LR schedule
LR_START = 1e-6                   # for warm-up start
LR_BASE  = 1e-4
LR_MIN   = 2e-5
warmup_steps = 5000               #
EPOCHS   = 30
BATCH    = 2                      # keep small for memory
WEIGHT_DECAY = 5e-4
VAL_FRAC = 0.2
V_SCALE  = 4500.0                 # velocity scaling
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.manual_seed(42); np.random.seed(42)

# Dynamic loss-scale ramp parameters
loss_scale_start        = 10.0      # lower initial multiplier
loss_scale_target       = V_SCALE
loss_scale_warmup_steps = 5000

# Data augmentation
noise_std = 0.01     # additive Gaussian noise on seismic input (normalized units)
amp_range = 0.05     # random amplitude scaling ±5%

# Gradient clipping
CLIP_NORM = 300.0    # global norm clipping threshold
adaptive_clip_coef = 0.0  # 0 to disable adaptive clipping

# Loss: SmoothL1 with beta in normalized units
beta_norm = 0.02
criterion_smoothl1 = nn.SmoothL1Loss(beta=beta_norm)
def list_pairs(fam_dir):
    data_dir, model_dir = os.path.join(fam_dir,"data"), os.path.join(fam_dir,"model")
    if os.path.isdir(data_dir) and os.path.isdir(model_dir):
        d = sorted(f for f in os.listdir(data_dir)  if f.endswith(".npy"))
        m = sorted(f for f in os.listdir(model_dir) if f.endswith(".npy"))
        return [(os.path.join(data_dir, d[i]), os.path.join(model_dir, m[i])) for i in range(len(d))]
    files = [f for f in os.listdir(fam_dir) if f.endswith(".npy")]
    seis, vel = {}, {}
    for f in files:
        if f.startswith(("data","seis")):
            seis[pat_data.sub("",f)]  = f
        elif f.startswith(("model","vel")):
            vel[pat_model.sub("",f)]  = f
    return [(os.path.join(fam_dir,seis[k]), os.path.join(fam_dir,vel[k])) for k in sorted(set(seis)&set(vel))]
# --------------------------------------------------

# -------- mmap index dataset (unchanged) ----------
def build_index(base):
    idx = []
    for fam in sorted(os.listdir(base)):
        fam_dir = os.path.join(base, fam)
        if not os.path.isdir(fam_dir): continue
        for sfile, vfile in list_pairs(fam_dir):
            for i in range(500):
                idx.append((sfile, vfile, i))
    return idx

index_all = build_index(BASE_DIR)
tr_ids, va_ids = train_test_split(np.arange(len(index_all)),
                                  test_size=VAL_FRAC, random_state=42, shuffle=True)

class VelDS(Dataset):
    def __init__(self, idx_list):
        self.idx = idx_list
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, k):
        sfile, vfile, i = self.idx[k]
        s = np.load(sfile, mmap_mode='r')[i].copy().astype(np.float32)   # (5,1000,70)
        v = np.load(vfile, mmap_mode='r')[i]
        v = v.squeeze().copy().astype(np.float32) / V_SCALE
        return torch.from_numpy(s), torch.from_numpy(v)

dl_tr = DataLoader(VelDS([index_all[i] for i in tr_ids]), batch_size=BATCH,
                   shuffle=True, num_workers=4, pin_memory=True)
dl_va = DataLoader(VelDS([index_all[i] for i in va_ids]), batch_size=BATCH,
                   shuffle=False, num_workers=4, pin_memory=True)
# --------------------------------------------------

# ---------------- token stems --------------------
def split_fft(x):
    # x: (B,5,1000,70)
    F = torch.fft.rfft2(x, dim=(-2,-1))  # yields (B,5,1000,36)
    return torch.cat([F.real, F.imag, torch.log(torch.abs(F)+1e-6)], dim=1)

class StemRaw(nn.Module):
    def __init__(self, emb_d):
        super().__init__()
        self.ln = nn.LayerNorm(emb_d)
        self.conv = nn.Conv2d(5, emb_d, (PATCH_T, PATCH_X), stride=(STR_T, STR_X))
        self.drop = nn.Dropout(DROP)
    def forward(self, x):
        f = self.conv(x)                      # (B, emb_d, T', X')
        tok = f.flatten(2).transpose(1,2)     # (B, Ntok, emb_d)
        tok = self.ln(tok)
        return self.drop(tok)

class StemFFT(nn.Module):
    def __init__(self, emb_d):
        super().__init__()
        self.ln = nn.LayerNorm(emb_d)
        self.conv = nn.Conv2d(15, emb_d, (PATCH_T, PATCH_X), stride=(STR_T, STR_X))
        self.drop = nn.Dropout(DROP)
    def forward(self, x):
        spec = split_fft(x)                   # (B,15,1000,36)
        f = self.conv(spec)                   # (B, emb_d, T', X')
        tok = f.flatten(2).transpose(1,2)     
        tok = self.ln(tok)
        return self.drop(tok)
# --------------------------------------------------

# -------------- MetaFormer (PoolMixer) ------------
class PoolMixer(nn.Module):
    def __init__(self, C, W_tok):
        super().__init__()
        self.W = W_tok
    def forward(self, x):
        # x: (B, 1+N, C)
        cls, tok = x[:, :1], x[:, 1:]
        B, Np, C = tok.shape
        W = self.W
        H = Np // W
        assert H * W == Np, f"grid {H}×{W}!={Np}"
        y = tok.transpose(1, 2).reshape(B, C, H, W)
        y = F.avg_pool2d(y, 3, 1, 1) - y
        tok2 = tok + y.flatten(2).transpose(1, 2)
        return torch.cat([cls, tok2], dim=1)

class MetaBlock(nn.Module):
    def __init__(self, C, W_tok, ff_mult=4, residual_scale=1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(C)
        self.mix   = PoolMixer(C, W_tok)
        self.norm2 = nn.LayerNorm(C)
        self.ffn   = nn.Sequential(
            nn.Linear(C, ff_mult*C),
            nn.GELU(),
            nn.Linear(ff_mult*C, C)
        )
        self.residual_scale = residual_scale
    def forward(self, x):
        y1 = self.mix(self.norm1(x))
        x = x + self.residual_scale * y1
        y2 = self.ffn(self.norm2(x))
        return x + self.residual_scale * y2

class MetaEncoder(nn.Module):
    def __init__(self, depth: int, C: int, W_tok: int, residual_scale=1.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MetaBlock(C, W_tok, residual_scale=residual_scale) for _ in range(depth)]
        )
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
# --------------------------------------------------

# -------------- PixelShuffle + mini-U-Net ---------
class ShuffleUNetRefine(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.pre = nn.Conv2d(1, base*4, 1)
        self.pre_gn = nn.GroupNorm(num_groups=1, num_channels=base*4)
        self.shuffle = nn.PixelShuffle(2)          
        self.enc1 = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base),
            nn.ReLU(True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base),
            nn.ReLU(True),
        )
        self.pool1 = nn.MaxPool2d(2)                
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base*2),
            nn.ReLU(True),
            nn.Conv2d(base*2, base*2, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base*2),
            nn.ReLU(True),
        )
        self.pool2 = nn.MaxPool2d(2)                
        self.enc3 = nn.Sequential(
            nn.Conv2d(base*2, base*4, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base*4),
            nn.ReLU(True),
            nn.Conv2d(base*4, base*4, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base*4),
            nn.ReLU(True),
        )
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2,2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base*4, base*2, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base*2),
            nn.ReLU(True),
            nn.Conv2d(base*2, base*2, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base*2),
            nn.ReLU(True),
        )
        self.up1 = nn.ConvTranspose2d(base*2, base, 2,2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base*2, base, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base),
            nn.ReLU(True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=base),
            nn.ReLU(True),
        )
        self.outc = nn.Conv2d(base, 1, 1)

    def forward(self, c):
        x = self.shuffle(self.pre_gn(self.pre(c)))       
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        d2 = self.up2(e3); d2 = self.dec2(torch.cat([d2,e2], dim=1))
        d1 = self.up1(d2); d1 = self.dec1(torch.cat([d1,e1], dim=1))
        hi = self.outc(d1)                 
        return F.avg_pool2d(hi, 2).squeeze(1)  # (B,70,70)
# --------------------------------------------------

# -------------- Dual-branch MetaFormer model ------
class DualBranchMeta(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw = StemRaw(EMB_D)
        self.fft = StemFFT(EMB_D)
        # token grid sizes
        W_RAW = (70 - PATCH_X)//STR_X + 1
        freq_dim = 70//2 + 1
        W_FFT = (freq_dim - PATCH_X)//STR_X + 1
        self.enc_raw = MetaEncoder(DEPTH, EMB_D, W_RAW)
        self.enc_fft = MetaEncoder(DEPTH, EMB_D, W_FFT)
        # fusion to coarse map
        self.fuse = nn.Sequential(
            nn.LayerNorm(2*EMB_D),
            nn.Linear(2*EMB_D, 4*EMB_D), nn.GELU(),
            nn.Linear(4*EMB_D, 70*70)
        )
        self.coarse_norm = nn.GroupNorm(1,1)
        self.refine = ShuffleUNetRefine(base=32)

    def forward(self, s):
        B = s.size(0)
        r_tok = self.raw(s)
        f_tok = self.fft(s)
        zero = torch.zeros(B,1,EMB_D,device=s.device)
        r_all = torch.cat([zero, r_tok], dim=1)
        f_all = torch.cat([zero, f_tok], dim=1)
        r_enc = self.enc_raw(r_all)
        f_enc = self.enc_fft(f_all)
        r_cls = r_enc[:,1:,:].mean(dim=1)
        f_cls = f_enc[:,1:,:].mean(dim=1)
        fused = torch.cat([r_cls, f_cls], dim=1)
        coarse = self.fuse(fused).view(B,1,70,70)
        coarse_norm = self.coarse_norm(coarse)
        fine = self.refine(coarse_norm)
        return coarse_norm, fine
print("Done")


import pickle
checkpoint_path = "training_state.pkl"

# -------- resume from checkpoint --------
import os
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        state = pickle.load(f)
    model.load_state_dict(state['model_state'])
    optimizer.load_state_dict(state['optimizer_state'])
    scheduler.load_state_dict(state['scheduler_state'])
    scheduler_plateau.load_state_dict(state['scheduler_plateau_state'])
    scaler.load_state_dict(state['scaler_state'])
    global_step = state['global_step']
    best_val = state['best_val']
    start_epoch = state['epoch']
    print(f"[Resume] Loaded checkpoint. Resuming from epoch {start_epoch}")
else:
    global_step = 0
    best_val = float('inf')
    start_epoch = 1


CHECKPOINT=False
def register_linear_hooks(model):
    def make_hook(name):
        def hook(module, inp, out):
            if torch.isnan(out).any() or torch.isinf(out).any():
                mn, mx = out.min().item(), out.max().item()
                print(f"[LEAK] NaN/Inf in output of Linear '{name}': min={mn:.3e}, max={mx:.3e}")
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.register_forward_hook(make_hook(name))


#!/usr/bin/env python3
"""
Dual-branch MetaFormer + Stable Training Pipeline with Phase-Aware CQT
Includes:
- Phase-aware Constant-Q Transform (magnitude + phase channels)
- CQT clipping to avoid extremes
- MetaFormer encoder with clamped residuals and FFN
- PixelShuffle U-Net refine with GroupNorm
- Mixed-precision, gradient accumulation & clipping
- LR scheduling with plateau re-warm
- Checkpointing & resume
"""
import os, re, math, pickle
import numpy as np
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split

# ----------------- Hyperparameters -----------------
BASE_DIR        = "/kaggle/input/waveform-inversion/train_samples"
PATCH_T, PATCH_X= 32, 16
STR_T, STR_X    = 4,  2
EMB_D           = 256
DEPTH           = 6
DROP            = 0.1

# CQT hyperparameters
CQT_BINS                = 36
CQT_BINS_PER_OCTAVE     = 12
CQT_HOP_LENGTH          = 1

# LR schedule & accumulation
LR_START        = 1e-6
LR_BASE_RAW     = 1e-4
LR_MIN          = 2e-5
WARMUP_STEPS    = 500
EPOCHS          = 30
BATCH           = 2
ACCUM_STEPS     = 4
LR_BASE         = LR_BASE_RAW * ACCUM_STEPS

WEIGHT_DECAY    = 5e-4
VAL_FRAC        = 0.2
V_SCALE         = 4500.0
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Disable augmentations for stability
recv_drop_p     = 0.0
noise_std       = 0.0

# Smoothness regularizer weight
smooth_lmbda    = 1e-3

# Gradient clipping norm
CLIP_NORM       = 300.0

# Loss
beta_norm       = 0.02
criterion_smooth= nn.SmoothL1Loss(beta=beta_norm)

# --------------- Data pipeline ---------------------
pat_data  = re.compile(r'^(data|seis)[_\-]?')
pat_model = re.compile(r'^(model|vel)[_\-]?')

def list_pairs(fam_dir):
    ddir, mdir = os.path.join(fam_dir, "data"), os.path.join(fam_dir, "model")
    if os.path.isdir(ddir) and os.path.isdir(mdir):
        files_d = sorted(f for f in os.listdir(ddir) if f.endswith('.npy'))
        files_m = sorted(f for f in os.listdir(mdir) if f.endswith('.npy'))
        return [(os.path.join(ddir, files_d[i]), os.path.join(mdir, files_m[i]))
                for i in range(len(files_d))]
    seis, vel = {}, {}
    for f in os.listdir(fam_dir):
        if not f.endswith('.npy'): continue
        if f.startswith(('data','seis')): seis[pat_data.sub('', f)] = f
        else: vel[pat_model.sub('', f)] = f
    keys = sorted(set(seis) & set(vel))
    return [(os.path.join(fam_dir, seis[k]), os.path.join(fam_dir, vel[k])) for k in keys]

index_all = [(d,m,i) for fam in sorted(os.listdir(BASE_DIR)) if os.path.isdir(os.path.join(BASE_DIR,fam))
             for d,m in list_pairs(os.path.join(BASE_DIR,fam)) for i in range(500)]
train_ids, val_ids = train_test_split(list(range(len(index_all))), test_size=VAL_FRAC,
                                     random_state=42)

class VelDS(Dataset):
    def __init__(self, ids): self.ids = ids
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        sfile, vfile, shot = index_all[self.ids[idx]]
        seis = np.load(sfile, mmap_mode='r')[shot].astype(np.float32)
        vel  = np.load(vfile, mmap_mode='r')[shot].astype(np.float32) / V_SCALE
        return torch.from_numpy(seis), torch.from_numpy(vel)

train_loader = DataLoader(VelDS(train_ids), batch_size=BATCH, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(VelDS(val_ids),   batch_size=BATCH, shuffle=False,
                          num_workers=4, pin_memory=True)

# -------------- Phase-aware CQT -------------------
def split_cqt(x):
    B, C, T, N = x.shape
    flat = x.permute(0,1,3,2).reshape(-1, T).cpu().numpy()
    sr = 1.0; nyquist = sr/2.0
    fmin = nyquist / (2 ** (CQT_BINS / CQT_BINS_PER_OCTAVE))
    cqt_list = []
    for vec in flat:
        CQT = librosa.cqt(vec, sr=sr, hop_length=CQT_HOP_LENGTH,
                          fmin=fmin, n_bins=CQT_BINS,
                          bins_per_octave=CQT_BINS_PER_OCTAVE)
        cqt_list.append(CQT)
    CQT = torch.from_numpy(np.stack(cqt_list)).to(x.device)
    mag, phase = torch.abs(CQT), torch.angle(CQT)
    spec = torch.cat([mag, phase], dim=1)
    spec = spec.view(B, C, N, 2*CQT_BINS, -1).permute(0,1,3,4,2)
    return torch.clamp(spec.reshape(B, C*2*CQT_BINS, spec.size(3), N), -10.0, 10.0)

# ------------- Model components -------------------
class StemRaw(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(5, EMB_D, (PATCH_T, PATCH_X), (STR_T, STR_X))
        self.norm = nn.LayerNorm(EMB_D)
        self.drop = nn.Dropout(DROP)
    def forward(self, x):
        tok = self.conv(x).flatten(2).transpose(1,2)
        tok = torch.clamp(tok, -1e2, 1e2)
        tok = self.norm(tok)
        return self.drop(tok)

class StemCQT(nn.Module):
    def __init__(self):
        super().__init__()
        in_ch = 5 * 2 * CQT_BINS
        self.conv = nn.Conv2d(in_ch, EMB_D, (PATCH_T, PATCH_X), (STR_T, STR_X))
        self.norm = nn.LayerNorm(EMB_D)
        self.drop = nn.Dropout(DROP)
    def forward(self, x):
        spec = split_cqt(x)
        tok  = self.conv(spec).flatten(2).transpose(1,2)
        tok  = torch.clamp(tok, -1e2, 1e2)
        tok  = self.norm(tok)
        return self.drop(tok)

class PoolMixer(nn.Module):
    def __init__(self, C, W): super().__init__(); self.W = W
    def forward(self, x):
        cls, tok = x[:,:1], x[:,1:]
        B,N,Cc = tok.shape; H = N//self.W
        y = tok.transpose(1,2).reshape(B,Cc,H,self.W)
        y2 = F.avg_pool2d(y,3,1,1) - y
        return torch.cat([cls, tok + y2.flatten(2).transpose(1,2)],1)

class MetaBlock(nn.Module):
    def __init__(self, C, W, scale=0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(C)
        self.mix   = PoolMixer(C,W)
        self.norm2 = nn.LayerNorm(C)
        self.ffn   = nn.Sequential(nn.Linear(C,4*C), nn.GELU(), nn.Linear(4*C,C))
        self.scale = scale
    def forward(self, x):
        y = self.mix(self.norm1(x)); y = torch.clamp(y,-1e2,1e2)
        x = x + self.scale*y
        z = self.ffn(self.norm2(x)); z=torch.clamp(z,-1e2,1e2)
        return x + self.scale*z

class MetaEncoder(nn.Module):
    def __init__(self, d,C,W): super().__init__(); self.blocks=nn.ModuleList([MetaBlock(C,W) for _ in range(d)])
    def forward(self,x):
        for b in self.blocks: x=b(x)
        return x

class ShuffleUNetRefine(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.pre      = spectral_norm(nn.Conv2d(1,base*4,1))
        self.norm_pre = nn.GroupNorm(1,base*4)
        self.shuffle  = nn.PixelShuffle(2)
        self.enc1     = spectral_norm(nn.Conv2d(base,base,3,padding=1))
        self.norm1    = nn.GroupNorm(1,base)
        self.enc2     = spectral_norm(nn.Conv2d(base,base*2,3,padding=1))
        self.norm2    = nn.GroupNorm(1,base*2)
        self.enc3     = spectral_norm(nn.Conv2d(base*2,base*4,3,padding=1))
        self.norm3    = nn.GroupNorm(1,base*4)
        self.up2      = nn.ConvTranspose2d(base*4,base*2,2,2)
        self.dec2     = spectral_norm(nn.Conv2d(base*4,base*2,3,padding=1))
        self.normd2   = nn.GroupNorm(1,base*2)
        self.up1      = nn.ConvTranspose2d(base*2,base,2,2)
        self.dec1     = spectral_norm(nn.Conv2d(base*2,base,3,padding=1))
        self.normd1   = nn.GroupNorm(1,base)
        self.outc     = nn.Conv2d(base,1,1)
    def forward(self, c):
        x = self.shuffle(self.norm_pre(self.pre(c)))
        e1 = F.gelu(self.norm1(self.enc1(x)))
        e2 = F.gelu(self.norm2(self.enc2(F.max_pool2d(e1,2))))
        e3 = F.gelu(self.norm3(self.enc3(F.max_pool2d(e2,2))))
        d2=F.gelu(self.normd2(self.dec2(torch.cat([self.up2(e3),e2],1))))
        d1=F.gelu(self.normd1(self.dec1(torch.cat([self.up1(d2),e1],1))))
        hi = self.outc(d1)
        return F.avg_pool2d(hi,2).squeeze(1)

class DualBranchMeta(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw = StemRaw()
        self.fft = StemCQT()
        # compute token grid widths
        W_TIME  = (1000 - PATCH_T)//STR_T + 1
        W_SPACE = (70   - PATCH_X)//STR_X + 1
        self.enc_raw = MetaEncoder(DEPTH, EMB_D, W_TIME)
        self.enc_fft = MetaEncoder(DEPTH, EMB_D, W_SPACE)
        self.fuse_norm = nn.LayerNorm(2*EMB_D)
        self.fuse      = nn.Sequential(nn.Linear(2*EMB_D,4*EMB_D), nn.GELU(), nn.Linear(4*EMB_D,70*70))
        self.coarse_norm = nn.GroupNorm(1,1)
        self.refine = ShuffleUNetRefine(base=32)
    def forward(self,s):
        B = s.size(0)
        r_tok = self.raw(s)
        f_tok = self.fft(s)
        zero = torch.zeros(B,1,EMB_D,device=s.device)
        r_enc= self.enc_raw(torch.cat([zero,r_tok],1))
        f_enc= self.enc_fft(torch.cat([zero,f_tok],1))
        r_cls= r_enc[:,1:,:].mean(1)
        f_cls= f_enc[:,1:,:].mean(1)
        h    = self.fuse_norm(torch.cat([r_cls,f_cls],1))
        h    = torch.clamp(h,-1e2,1e2)
        coarse = self.fuse(h).view(B,1,70,70)
        coarse = torch.clamp(coarse,-1e3,1e3)
        coarse = self.coarse_norm(coarse)
        return coarse, self.refine(coarse)

# -------------- Training loop ---------------------
model = DualBranchMeta().to(DEVICE)
scaler= GradScaler()
optimizer= torch.optim.AdamW([
    {'params': model.raw.parameters()},
    {'params': model.fft.parameters()},
    {'params': model.enc_raw.parameters()},
    {'params': model.enc_fft.parameters(), 'lr': LR_BASE_RAW*0.5},
    {'params': model.fuse.parameters(),   'lr': LR_BASE_RAW*0.1},
    {'params': model.coarse_norm.parameters()},
    {'params': model.refine.parameters(), 'lr': LR_BASE_RAW/10}
], lr=LR_BASE, weight_decay=WEIGHT_DECAY)

steps_per_epoch = len(train_loader) // ACCUM_STEPS
scheduler = LambdaLR(optimizer, lambda step: 
    (LR_START + (LR_BASE_RAW - LR_START)
     * min(step, WARMUP_STEPS)/WARMUP_STEPS)/LR_BASE_RAW
    if step < WARMUP_STEPS else
    (LR_MIN + 0.5*(LR_BASE_RAW - LR_MIN)*
     (1 + math.cos(math.pi * min((step - WARMUP_STEPS)/(EPOCHS*steps_per_epoch - WARMUP_STEPS),1))))/LR_BASE_RAW
)
scheduler_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
best_val = float('inf'); global_step=0

# optional resume
checkpoint_path = 'checkpoint.pkl'
if CHECKPOINT:
    state = pickle.load(open(checkpoint_path,'rb'))
    model.load_state_dict(state['model_state'])
    optimizer.load_state_dict(state['optimizer_state'])
    scheduler.load_state_dict(state['scheduler_state'])
    scheduler_plateau.load_state_dict(state['scheduler_plateau_state'])
    scaler.load_state_dict(state['scaler_state'])
    global_step = state['global_step']
    best_val    = state['best_val']
    start_epoch = state['epoch']+1
else:
    start_epoch = 1

for ep in range(start_epoch, EPOCHS+1):
    model.train()
    running_train=0.0; count_train=0; skip_ctr=0
    optimizer.zero_grad()
    for step, (seis, vel) in enumerate(train_loader):
        seis, vel = seis.to(DEVICE), vel.to(DEVICE).unsqueeze(1)
        with autocast('cuda'):
            coarse, pred = model(seis)
            loss_mae = F.l1_loss(pred, vel)
            loss_s   = smooth_lmbda * F.l1_loss(coarse, vel)
            loss     = (loss_mae + loss_s) / ACCUM_STEPS
        if torch.isnan(loss) or torch.isinf(loss):
            skip_ctr += 1; optimizer.zero_grad(); continue
        scaler.scale(loss).backward()
        if (step+1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            scheduler.step(); global_step+=1
            running_train += loss_mae.item() * V_SCALE * BATCH
            count_train   += BATCH
    train_mae = running_train/count_train if count_train>0 else float('nan')
    print(f"Epoch {ep}/{EPOCHS} Train MAE: {train_mae:.1f} m/s (skipped {skip_ctr})")
    
    model.eval(); val_err=0.0
    with torch.no_grad():
        for seis, vel in val_loader:
            seis, vel = seis.to(DEVICE), vel.to(DEVICE).unsqueeze(1)
            _, pred = model(seis)
            val_err += F.l1_loss(pred, vel).item() * V_SCALE * vel.size(0)
    val_mae = val_err / len(val_loader.dataset)
    print(f"Epoch {ep}/{EPOCHS} Validation MAE: {val_mae:.1f} m/s")

    if global_step >= WARMUP_STEPS:
        scheduler_plateau.step(val_mae)
    if val_mae < best_val:
        best_val = val_mae
        torch.save(model.state_dict(), 'best_model.pth')
        print("Saved new best model")
    # checkpoint
    state = {'epoch': ep, 'model_state': model.state_dict(),
             'optimizer_state': optimizer.state_dict(),
             'scheduler_state': scheduler.state_dict(),
             'scheduler_plateau_state': scheduler_plateau.state_dict(),
             'scaler_state': scaler.state_dict(),
             'global_step': global_step, 'best_val': best_val}
    pickle.dump(state, open(checkpoint_path,'wb'))
    print(f"Checkpoint saved for epoch {ep}\n")

print("Training complete.")



import math, os, re, random, pickle, numpy as np
from pathlib import Path
from tqdm import tqdm

import torch, torch.nn as nn, torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

# ────── paths ──────
BASE_DIR = "waveform-inversion/train_samples"         # <-- change
CKPT_DIR = "./ckpt_dual"
Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)

# ────── data & training hyper-params ──────
VAL_FRAC = 0.20
V_SCALE  = 4_500.0
EMB_D   = 256
BATCH_STAGE1 = 2
BATCH_STAGE2 = 6
PATCH_T, PATCH_X = 32, 16
STR_T,   STR_X   = 4,   2

BASE_DIR    = "/kaggle/input/waveform-inversion/train_samples"
DEPTH   = 6

EPOCH1, EPOCH2 = 25, 15
LR1,    LR2    = 2e-4, 1e-4
WARM1,  WARM2  = 5_000, 2_000
ACC1,   ACC2   = 4, 1          # grad-accum
DROP = 0.1
SMOOTH_LMBDA  = 0.1            # coarse smoothness loss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42); random.seed(42)



_pat_d, _pat_m = re.compile(r'^(data|seis)[_\-]?'), re.compile(r'^(model|vel)[_\-]?')

def _list_pairs(fam_dir):
    files = [f for f in os.listdir(fam_dir) if f.endswith('.npy')]
    seis, vel = {}, {}
    for f in files:
        if f.startswith(('data', 'seis')):    seis[_pat_d.sub('', f)] = f
        elif f.startswith(('model','vel')):   vel [_pat_m.sub('', f)] = f
    return [(os.path.join(fam_dir, seis[k]), os.path.join(fam_dir, vel[k]))
            for k in sorted(set(seis)&set(vel))]

def build_index(root):
    idx=[]
    for fam in sorted(os.listdir(root)):
        famdir=os.path.join(root,fam)
        if not os.path.isdir(famdir): continue
        for s,v in _list_pairs(famdir):
            n = np.load(s, mmap_mode='r').shape[0]      # true #shots
            idx.extend([(s,v,i) for i in range(n)])
    return idx

class VelDS(Dataset):
    def __init__(self, items): self.items=items
    def __len__(self): return len(self.items)
    def __getitem__(self,k):
        s,v,i = self.items[k]
        seis = np.load(s, mmap_mode='r')[i].astype(np.float32)   # (5,1000,70)
        vel  = np.load(v, mmap_mode='r')[i].squeeze().astype(np.float32)
        return torch.from_numpy(seis), torch.from_numpy(vel/V_SCALE)




# -----------------------------------------

# ----------- model -----------------------
def split_fft(x):
    """log-FFT over (t,x) with sign-preserving magnitude"""
    F2   = torch.fft.rfft2(x, dim=(-2,-1))
    mag  = torch.log(torch.abs(F2)+1e-6)
    real = torch.sign(F2.real) * mag
    imag = torch.sign(F2.imag) * mag
    return torch.cat([real, imag], dim=1)   # (B, 10, 1000, 36)

def center_crop(feat: torch.Tensor, size):
    h, w = feat.shape[-2:]
    dh, dw = (h - size[0]) // 2, (w - size[1]) // 2
    return feat[..., dh:dh+size[0], dw:dw+size[1]]

class StemRaw(nn.Module):
    def __init__(self, ed):
        super().__init__()
        self.conv = nn.Conv2d(5, ed, (PATCH_T,PATCH_X), stride=(STR_T,STR_X))
        self.ln   = nn.LayerNorm(ed); self.drop = nn.Dropout(DROP)
    def forward(self, x):
        f = self.conv(x).flatten(2).transpose(1,2)
        return self.drop(self.ln(f))

class StemFFT(nn.Module):
    def __init__(self, ed):
        super().__init__()
        self.conv = nn.Conv2d(10, ed, (PATCH_T,PATCH_X), stride=(STR_T,STR_X))
        self.ln   = nn.LayerNorm(ed); self.drop = nn.Dropout(DROP)
    def forward(self, x):
        f = self.conv(split_fft(x)).flatten(2).transpose(1,2)
        return self.drop(self.ln(f))

class PoolMixer(nn.Module):
    def __init__(self, C, W): super().__init__(); self.W = W
    def forward(self, x):
        cls, tok = x[:,:1], x[:,1:]
        B,N,C = tok.shape; H = N//self.W
        y = tok.transpose(1,2).reshape(B,C,H,self.W)
        y = F.avg_pool2d(y,3,1,1) - y
        return torch.cat([cls, tok+y.flatten(2).transpose(1,2)],1)

class MetaBlock(nn.Module):
    def __init__(self, C, W, drop=DROP):
        super().__init__()
        self.norm1 = nn.LayerNorm(C); self.mix = PoolMixer(C,W)
        self.norm2 = nn.LayerNorm(C)
        self.ffn = nn.Sequential(nn.Linear(C,4*C), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(4*C,C))
    def forward(self,x):
        x = x + self.mix(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class MetaEncoder(nn.Module):
    def __init__(self, d,C,W): super().__init__()
    def __init__(self, d,C,W):
        super().__init__()
        self.blocks = nn.ModuleList([MetaBlock(C,W) for _ in range(d)])
    def forward(self,x):
        for b in self.blocks: x=b(x)
        return x

class ShuffleUNetRefine(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.drop = nn.Dropout2d(0.1)

        # ---------- encoder ----------
        self.pre  = nn.Conv2d(1, base * 4, 1)
        self.normp = nn.LayerNorm([base * 4, 70, 70])
        self.shuffle = nn.PixelShuffle(2)                   

        def CBR(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
                nn.GroupNorm(1, cout),                       # res-agnostic norm
                nn.GELU()
            )

        self.enc1  = nn.Sequential(CBR(base,     base),     CBR(base,     base))
        self.pool1 = nn.MaxPool2d(2)                         
        self.enc2  = nn.Sequential(CBR(base, base * 2), CBR(base * 2, base * 2))
        self.pool2 = nn.MaxPool2d(2)                         
        self.enc3  = nn.Sequential(CBR(base * 2, base * 4),
                                   CBR(base * 4, base * 4))  
        # ---------- decoder ----------
        self.up2  = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)   
        self.dec2 = nn.Sequential(CBR(base * 4, base * 2), CBR(base * 2, base * 2))

        self.up1  = nn.ConvTranspose2d(base * 2, base, 2, 2)       
        self.dec1 = nn.Sequential(CBR(base * 2, base), CBR(base, base))

        self.head = nn.Conv2d(base, 1, 1)

        # --- NEW — learnable γ, zero-init so net = identity at start -----------
        self.gamma = nn.Parameter(torch.zeros(1))

        # also (re-)zero the 1×1 head so residual starts at *exactly* 0
        nn.init.constant_(self.head.weight, 0.)
        nn.init.constant_(self.head.bias,   0.)

    # --------------------------------------------------------------------- #
    @staticmethod
    def _center_crop(src: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        h, w = src.shape[-2:]
        th, tw = size
        dh, dw = (h - th) // 2, (w - tw) // 2
        return src[..., dh:dh + th, dw:dw + tw]

    def forward(self, c: torch.Tensor) -> torch.Tensor:           # c:(B,1,70,70)
        # ----------------- encoder -----------------
        x0 = self.shuffle(self.normp(self.pre(c)))                # (B,base,140,140)

        e1 = self.enc1(x0)                                        # 140×140
        e2 = self.enc2(self.pool1(e1))                            
        e3 = self.enc3(self.pool2(e2))                            

        # ----------------- decoder -----------------
        u2 = self.up2(e3)                                         # 34×34
        u2 = self._center_crop(u2, e2.shape[-2:])                 
        d2 = self.dec2(torch.cat([u2, self.drop(e2)], dim=1))     # 35×35

        u1 = self.up1(d2)                                         # 70×70
        d1 = self.dec1(torch.cat([u1, self.drop(e1)], dim=1))     # 70×70

        resid = self.head(d1).squeeze(1)                          # (B,70,70)

        # ----------------- gated residual -----------------
        return self.gamma * resid
     # (B,70,70)

class DualBranchMeta(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw = StemRaw(EMB_D);  self.fft = StemFFT(EMB_D)

        W_t  = (70-PATCH_X)//STR_X+1
        W_f  = (36-PATCH_X)//STR_X+1

        self.enc_raw = MetaEncoder(DEPTH, EMB_D, W_t)
        self.enc_fft = MetaEncoder(DEPTH, EMB_D, W_f)

        self.fuse = nn.Sequential(nn.LayerNorm(2*EMB_D),
                                  nn.Linear(2*EMB_D,4*EMB_D),
                                  nn.GELU(), nn.Linear(4*EMB_D,70*70))
        self.coarse_norm = nn.GroupNorm(1,1)
        self.refine = ShuffleUNetRefine()

    def forward(self,s):
        B=s.size(0)
        r_tok, f_tok = self.raw(s), self.fft(s)
        zr = torch.zeros(B,1,EMB_D,device=s.device)
        r_enc = self.enc_raw(torch.cat([zr,r_tok],1))
        f_enc = self.enc_fft(torch.cat([zr,f_tok],1))
        fused = torch.cat([r_enc[:,1:,:].mean(1), f_enc[:,1:,:].mean(1)],1)
        coarse= self.fuse(fused).view(B,1,70,70)
        coarse= self.coarse_norm(coarse)
        refined= self.refine(coarse)
        return coarse.squeeze(1), refined    


def grad_xy(t):
    gx = F.pad(t[:,:,1:,:]-t[:,:,:-1,:], (0,0,0,0,1,0))
    gy = F.pad(t[:,:,:,1:]-t[:,:,:,:-1], (1,0,0,0))
    return gx, gy

def laplacian(t):
    return (F.pad(t[:,:,2:,:]-2*t[:,:,1:-1,:]+t[:,:,:-2,:], (0,0,1,1)) +
            F.pad(t[:,:,:,2:]-2*t[:,:,:,1:-1]+t[:,:,:,:-2], (1,1,0,0)))

# ─── small conv-norm-activation helper ─────────────────────
import torch.nn as nn
def _blk(cin, cout, k=3, p=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=p, bias=False),
        nn.GroupNorm(1, cout),
        nn.GELU()
    )


class ResidualUNet(nn.Module):
    def __init__(self, in_ch=4, base=32):
        super().__init__()
        # ------- encoder -------
        self.enc1 = _blk(in_ch,  base)
        self.pool1= nn.MaxPool2d(2)            
        self.enc2 = _blk(base,  base*2)
        self.pool2= nn.MaxPool2d(2)            
        self.enc3 = _blk(base*2, base*4)

        # ------- decoder -------
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = _blk(base*4, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base,   2, 2)
        self.dec1 = _blk(base*2, base)

        self.head = nn.Conv2d(base, 1, 1)

    @staticmethod
    def _crop(src, size):          
        h, w = src.shape[-2:]
        dh, dw = (h-size[0])//2, (w-size[1])//2
        return src[..., dh:dh+size[0], dw:dw+size[1]]

    def forward(self, x4):
        e1 = self.enc1(x4)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        u2 = self.up2(e3)
        d2 = self.dec2(torch.cat([u2, self._crop(e2, u2.shape[-2:])], 1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, self._crop(e1, u1.shape[-2:])], 1))

        out = self.head(d1)                      # 1×68×68
        out = F.pad(out, (1,1,1,1), mode='reflect')  
        return out.squeeze(1)



def cosine_warm(step, warm, total):
    if step < warm: return (step+1)/warm
    q = (step-warm)/(total-warm)
    return 0.5*(1+math.cos(math.pi*min(q,1)))

def make_sched(opt, lr0, warm, total):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_warm(s,warm,total))



items = build_index(BASE_DIR)
perm  = np.random.permutation(len(items))
cut   = int(len(perm)*(1-VAL_FRAC))
train_items, val_items = [items[i] for i in perm[:cut]], [items[i] for i in perm[cut:]]

train_set, val_set = VelDS(train_items), VelDS(val_items)
dl_tr1 = DataLoader(train_set, BATCH_STAGE1, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
dl_va1 = DataLoader(val_set,   BATCH_STAGE1, shuffle=False, num_workers=2, pin_memory=True)
coarse_net = DualBranchMeta().to(DEVICE)



opt1   = torch.optim.AdamW(coarse_net.parameters(), lr=LR1, weight_decay=5e-4)
steps1 = EPOCH1 * math.ceil(len(dl_tr1)/ACC1)
sched1 = make_sched(opt1, LR1, WARM1, steps1)
scaler = GradScaler()
best_val = 9e9




opt1   = torch.optim.AdamW(coarse_net.parameters(), lr=LR1, weight_decay=5e-4)
steps1 = EPOCH1 * math.ceil(len(dl_tr1)/ACC1)
sched1 = make_sched(opt1, LR1, WARM1, steps1)
scaler = GradScaler()
best_val = 9e9

for epoch in range(1, EPOCH1+1):
    coarse_net.train(); run=cnt=0
    pbar = tqdm(enumerate(dl_tr1), total=len(dl_tr1), ncols=90, desc=f"[1] {epoch}/{EPOCH1}")
    opt1.zero_grad(); acc=0
    for i,(s,v) in pbar:
        s,v = s.to(DEVICE), v.to(DEVICE)
        with autocast():
            coarse,pred = coarse_net(s)
            tgt = v.unsqueeze(1)
            loss = (F.l1_loss(pred.unsqueeze(1),tgt) +
                    SMOOTH_LMBDA*F.l1_loss(coarse,tgt)) / ACC1
        scaler.scale(loss).backward(); acc+=1
        if acc==ACC1:
            scaler.unscale_(opt1)
            torch.nn.utils.clip_grad_norm_(coarse_net.parameters(), 300)
            scaler.step(opt1); scaler.update()
            opt1.zero_grad(); acc=0; sched1.step()
        run += loss.item()*V_SCALE*s.size(0)*ACC1; cnt += s.size(0)
    # -------- validation --------
    coarse_net.eval(); val=0
    with torch.no_grad():
        for s,v in dl_va1:
            s,v=s.to(DEVICE),v.to(DEVICE)
            _,pred = coarse_net(s)
            val += F.l1_loss(pred, v.item())*V_SCALE*s.size(0)
    tr_mae, va_mae = run/cnt, val/len(val_set)
    print(f"  ↪ MAE train {tr_mae:6.1f} | val {va_mae:6.1f}")
    if va_mae<best_val:
        best_val=va_mae
        torch.save(coarse_net.state_dict(), f"{CKPT_DIR}/coarse_best.pt")
        print("  saved best coarse")



# ===========================================================
#  stage1_baseline.ipynb  –  dual-branch + UNet (Kaggle-ready)
# ===========================================================

# ------------------- paths & misc --------------------------
BASE_DIR  = "/kaggle/input/waveform-inversion/train_samples"   # ← adjust
CKPT_DIR  = "/kaggle/working/ckpt_stage1"
!mkdir -p $CKPT_DIR                                             # Kaggle shell

import os, re, math, pickle, random, numpy as np
from pathlib import Path
from tqdm import tqdm

import torch, torch.nn as nn, torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

# reproducibility -------------------------------------------------------------
torch.manual_seed(42); np.random.seed(42); random.seed(42)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH       = 2
VAL_FRAC    = 0.2
EPOCHS      = 50
ACC_STEPS   = 4

# model hyper-params ----------------------------------------------------------
PATCH_T, PATCH_X = 32, 16
STR_T,   STR_X   = 4, 2
EMB_D, DEPTH     = 256, 6
DROP             = 0.1
SMOOTH_LMBDA     = 0.1

# optimiser & LR -------------------------------------------------------------
LR_BASE, LR_MIN  = 8e-4, 2e-5
UNET_RATIO       = 0.10
WARM_STEPS       = 10_000
WEIGHT_DEC       = 5e-4
CLIP_GLOBAL      = 100.
V_SCALE          = 4_500.0               

# -------------- dataset helpers --------------------------------------------
_pat_d = re.compile(r'^(data|seis)[_\-]?')
_pat_m = re.compile(r'^(model|vel)[_\-]?')

def _list_pairs(fam_dir):
    files = [f for f in os.listdir(fam_dir) if f.endswith(".npy")]
    seis, vel = {}, {}
    for f in files:
        if f.startswith(("data", "seis")):
            seis[_pat_d.sub("", f)] = f
        elif f.startswith(("model", "vel")):
            vel[_pat_m.sub("", f)] = f
    return [(os.path.join(fam_dir, seis[k]),
             os.path.join(fam_dir, vel[k])) for k in sorted(set(seis)&set(vel))]

def build_index(base):
    idx=[]
    for fam in sorted(os.listdir(base)):
        famdir=os.path.join(base,fam)
        if not os.path.isdir(famdir): continue
        for s,v in _list_pairs(famdir):
            for i in range(500):          # 500 shots/file
                idx.append((s,v,i))
    return idx

class VelDS(Dataset):
    def __init__(self, tuples):  self.items=tuples
    def __len__(self):           return len(self.items)
    def __getitem__(self,k):
        s,v,i=self.items[k]
        seis = np.load(s,mmap_mode='r')[i].astype(np.float32)    # (5,1000,70)
        vel  = np.load(v,mmap_mode='r')[i].squeeze().astype(np.float32)
        return torch.from_numpy(seis), torch.from_numpy(vel/V_SCALE)

# ------------ model parts ---------------------------------------------------
def split_fft(x):
    F2 = torch.fft.rfft2(x, dim=(-2,-1))
    mag = torch.log(torch.abs(F2)+1e-6)
    real = torch.sign(F2.real)*mag
    imag = torch.sign(F2.imag)*mag
    return torch.cat([real,imag],1)

class StemRaw(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.conv=nn.Conv2d(5,d,(PATCH_T,PATCH_X),stride=(STR_T,STR_X))
        self.ln=nn.LayerNorm(d); self.drop=nn.Dropout(DROP)
    def forward(self,x):
        f=self.conv(x).flatten(2).transpose(1,2)
        return self.drop(self.ln(f))

class StemFFT(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.conv=nn.Conv2d(10,d,(PATCH_T,PATCH_X),stride=(STR_T,STR_X))
        self.ln=nn.LayerNorm(d); self.drop=nn.Dropout(DROP)
    def forward(self,x):
        f=self.conv(split_fft(x)).flatten(2).transpose(1,2)
        return self.drop(self.ln(f))

class PoolMix(nn.Module):
    def __init__(self,W): super().__init__(); self.W=W
    def forward(self,x):
        cls,tok=x[:,:1],x[:,1:]
        B,N,C=tok.shape; H=N//self.W
        y=tok.transpose(1,2).reshape(B,C,H,self.W)
        y=F.avg_pool2d(y,3,1,1)-y
        return torch.cat([cls,tok+y.flatten(2).transpose(1,2)],1)

class MetaBlock(nn.Module):
    def __init__(self,C,W):
        super().__init__()
        self.norm1=nn.LayerNorm(C); self.mix=PoolMix(W)
        self.norm2=nn.LayerNorm(C)
        self.ffn=nn.Sequential(nn.Linear(C,4*C),nn.GELU(),nn.Dropout(DROP),
                               nn.Linear(4*C,C))
    def forward(self,x):
        x=x+self.mix(self.norm1(x))
        x=x+self.ffn(self.norm2(x))
        return x

class MetaEncoder(nn.Module):
    def __init__(self,d,C,W):
        super().__init__()
        self.blocks=nn.ModuleList([MetaBlock(C,W) for _ in range(d)])
    def forward(self,x):
        for b in self.blocks: x=b(x)
        return x

# --- Pixel-Shuffle U-Net refine --------------------------------------------
def _blk(cin,cout):
    return nn.Sequential(nn.Conv2d(cin,cout,3,1,1,bias=False),
                         nn.GroupNorm(1,cout), nn.GELU())

class ShuffleUNetRefine(nn.Module):
    def __init__(self,base=32):
        super().__init__()
        self.pre = nn.Conv2d(1, base*4, 1)
        self.normp = nn.LayerNorm([base*4,70,70])
        self.shuffle=nn.PixelShuffle(2)          

        self.enc1=_blk(base,base)
        self.pool1=nn.MaxPool2d(2)
        self.enc2=_blk(base,base*2)
        self.pool2=nn.MaxPool2d(2)
        self.enc3=_blk(base*2,base*4)

        self.up2 = nn.ConvTranspose2d(base*4,base*2,2,2)
        self.dec2=_blk(base*4,base*2)
        self.up1 = nn.ConvTranspose2d(base*2,base,2,2)
        self.dec1=_blk(base*2,base)

        self.outc=nn.Conv2d(base,1,1)

    def forward(self,c):
        x=self.shuffle(self.normp(self.pre(c)))
        e1=self.enc1(x)
        e2=self.enc2(self.pool1(e1))
        e3=self.enc3(self.pool2(e2))
        d2=self.dec2(torch.cat([self.up2(e3),e2],1))
        d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        out=F.avg_pool2d(self.outc(d1),2)
        return out.squeeze(1)           # (B,70,70)

# ---------------- full model -----------------------------------------------
class DualBranchMeta(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw=StemRaw(EMB_D); self.fft=StemFFT(EMB_D)
        Wt=(70-PATCH_X)//STR_X+1
        Wf=(36-PATCH_X)//STR_X+1
        self.enc_r=MetaEncoder(DEPTH,EMB_D,Wt)
        self.enc_f=MetaEncoder(DEPTH,EMB_D,Wf)
        self.fuse=nn.Sequential(nn.LayerNorm(2*EMB_D),
                                nn.Linear(2*EMB_D,4*EMB_D),
                                nn.GELU(),nn.Linear(4*EMB_D,70*70))
        self.coarse_norm=nn.GroupNorm(1,1)
        self.refine=ShuffleUNetRefine()

    def forward(self,s):
        B=s.size(0)
        r_tok,f_tok=self.raw(s),self.fft(s)
        zr=torch.zeros(B,1,EMB_D,device=s.device)
        r=self.enc_r(torch.cat([zr,r_tok],1))
        f=self.enc_f(torch.cat([zr,f_tok],1))
        fused=torch.cat([r[:,1:].mean(1),f[:,1:].mean(1)],1)
        coarse=self.fuse(fused).view(B,1,70,70)
        coarse=self.coarse_norm(coarse)
        pred=self.refine(coarse)
        return coarse.squeeze(1),pred

# ----------------- data loaders -------------------------------------------
all_idx=build_index(BASE_DIR)
perm=np.random.permutation(len(all_idx))
split=int(len(perm)*(1-VAL_FRAC))
tr,va=perm[:split],perm[split:]

train_loader=DataLoader(VelDS([all_idx[i] for i in tr]),
                        batch_size=BATCH,shuffle=True,num_workers=4,pin_memory=True)
val_loader  =DataLoader(VelDS([all_idx[i] for i in va]),
                        batch_size=BATCH,shuffle=False,num_workers=4,pin_memory=True)

# ----------------- training setup -----------------------------------------
model=DualBranchMeta().to(DEVICE)

decay,no_wd=[],[]
ref_ids={id(p) for p in model.refine.parameters()}
for n,p in model.named_parameters():
    if id(p) in ref_ids: continue
    (no_wd if p.ndim==1 or n.endswith(".bias") else decay).append(p)

opt=torch.optim.AdamW(
    [{"params":decay,"lr":LR_BASE},
     {"params":no_wd,"lr":LR_BASE,"weight_decay":0.},
     {"params":model.refine.parameters(),"lr":LR_BASE*UNET_RATIO}],
    betas=(0.9,0.95), weight_decay=WEIGHT_DEC)

total_steps=EPOCHS*math.ceil(len(train_loader)/ACC_STEPS)
def lr_main(step):
    if step<WARM_STEPS: return (step+1)/WARM_STEPS
    prog=(step-WARM_STEPS)/(total_steps-WARM_STEPS)
    return 0.8 * max(LR_MIN/LR_BASE,0.5*(1+math.cos(0.9*math.pi*min(prog,1))))
def lr_unet(step): return lr_main(step)*UNET_RATIO
sched=LambdaLR(opt,[lr_main,lr_main,lr_unet])
plateau=ReduceLROnPlateau(opt,'min',factor=0.5,patience=3,min_lr=1e-6)

loss_fn=nn.L1Loss(); scaler=GradScaler()
best=float('inf')

# ----------------- training loop ------------------------------------------
for ep in range(1,EPOCHS+1):
    model.train(); run=cnt=0; opt.zero_grad()
    pbar=tqdm(enumerate(train_loader),total=len(train_loader),ncols=90,desc=f"Epoch {ep}")
    for i,(seis,vel) in pbar:
        seis,vel=seis.to(DEVICE),vel.to(DEVICE)
        with autocast():
            coarse,pred=model(seis)
            v=vel
            l_mae=loss_fn(pred,v)
            l_s  =SMOOTH_LMBDA*loss_fn(coarse,v)
            loss=(l_mae+l_s)/ACC_STEPS
        scaler.scale(loss).backward()
        if (i+1)%ACC_STEPS==0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP_GLOBAL)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        run+=l_mae.item()*V_SCALE*seis.size(0); cnt+=seis.size(0)
        pbar.set_postfix(train_MAE=run/cnt)

    # validation ------------------------------------------------------------
    model.eval(); val=0.
    with torch.no_grad():
        for seis,vel in val_loader:
            seis,vel=seis.to(DEVICE),vel.to(DEVICE)
            _,pred=model(seis)
            val+=loss_fn(pred,vel).item()*V_SCALE*seis.size(0)
    val/=len(val_loader.dataset); tr=run/cnt
    plateau.step(val)

    print(f"epoch {ep:2d} | train {tr:6.1f} | val {val:6.1f} m/s "
          f"| lr {opt.param_groups[0]['lr']:.2e}")

    if val<best:
        best=val; torch.save(model.state_dict(),f"{CKPT_DIR}/best.pth")
        print("   saved best")



# -------------------- paths & basic config --------------------
BASE_DIR  = "/kaggle/input/waveform-inversion/train_samples"   # ← adjust if needed
CKPT_DIR1 = "/kaggle/working/ckpt_stage1"
CKPT_DIR2 = "/kaggle/working/ckpt_stage2"
import os, re, math, random, pickle, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
from tqdm import tqdm

for d in (CKPT_DIR1, CKPT_DIR2): Path(d).mkdir(parents=True, exist_ok=True)

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42); np.random.seed(42); random.seed(42)

# ---------------- dataset ---------------------------------------------------
_pat_d = re.compile(r'^(data|seis)[_\-]?')
_pat_m = re.compile(r'^(model|vel)[_\-]?')

def _pairs(dir_):
    f = [x for x in os.listdir(dir_) if x.endswith('.npy')]
    seis, vel = {}, {}
    for x in f:
        if x.startswith(("data", "seis")):  seis[_pat_d.sub('', x)] = x
        else:                              vel [_pat_m.sub('', x)] = x
    return [(os.path.join(dir_, seis[k]), os.path.join(dir_, vel[k]))
            for k in sorted(set(seis) & set(vel))]

def build_index(root):
    out = []
    for fam in sorted(os.listdir(root)):
        fd = os.path.join(root, fam)
        if not os.path.isdir(fd): continue
        for s, v in _pairs(fd):
            n = np.load(s, mmap_mode='r').shape[0]    # real #shots
            out += [(s, v, i) for i in range(n)]
    return out

V_SCALE = 4_500.0
class VelDS(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, k):
        s,v,i = self.items[k]
        seis = np.load(s, mmap_mode='r')[i].astype(np.float32)    # (5,1000,70)
        vel  = np.load(v, mmap_mode='r')[i].squeeze().astype(np.float32)
        return torch.from_numpy(seis), torch.from_numpy(vel / V_SCALE)



# ---------------- MetaFormer + coarse UNet -------------------
PATCH_T, PATCH_X = 32,16; STR_T,STR_X = 4,2
EMB_D, DEPTH, DROPOUT = 256, 6, 0.1

def split_fft(x: torch.Tensor) -> torch.Tensor:
    """log-magnitude FFT split into real / imag, sign-preserved."""
    F2  = torch.fft.rfft2(x, dim=(-2, -1))           # (B, 5, 1000, 36)
    mag = torch.log(torch.abs(F2) + 1e-6)
    real = torch.sign(F2.real) * mag
    imag = torch.sign(F2.imag) * mag
    return torch.cat([real, imag], dim=1)            # (B, 10, 1000, 36)


class _Stem(nn.Module):
    def __init__(self, in_ch: int, d: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, d,
                              kernel_size=(PATCH_T, PATCH_X),
                              stride=(STR_T,  STR_X))
        self.ln   = nn.LayerNorm(d)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):                 # x : (B, C, 1000, 70)
        t = self.conv(x).flatten(2).transpose(1, 2)   # (B, N, d)
        return self.drop(self.ln(t))


class PoolMix(nn.Module):
    def __init__(self, W): super().__init__(); self.W=W
    def forward(self,x):
        cls,tok=x[:,:1],x[:,1:]; B,N,C=tok.shape; H=N//self.W
        y=tok.transpose(1,2).reshape(B,C,H,self.W)
        y=F.avg_pool2d(y,3,1,1)-y
        return torch.cat([cls, tok+y.flatten(2).transpose(1,2)],1)

class MetaBlock(nn.Module):
    def __init__(self,C,W):
        super().__init__()
        self.n1=nn.LayerNorm(C); self.mix=PoolMix(W)
        self.n2=nn.LayerNorm(C)
        self.ff=nn.Sequential(nn.Linear(C,4*C), nn.GELU(), nn.Dropout(DROPOUT),
                              nn.Linear(4*C,C))
    def forward(self,x): x=x+self.mix(self.n1(x)); return x+self.ff(self.n2(x))

class Encoder(nn.Module):
    def __init__(self,d,C,W):
        super().__init__()
        self.blks=nn.ModuleList([MetaBlock(C,W) for _ in range(d)])
    def forward(self,x):
        for b in self.blks: x=b(x)
        return x

# --- tiny helper to crop ----------------------------------------------------
def _crop(t, size):
    h,w=t.shape[-2:]; dh=(h-size[0])//2; dw=(w-size[1])//2
    return t[..., dh:dh+size[0], dw:dw+size[1]]

# --- Shuffle-UNet (coarse) --------------------------------------------------
def _blk(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
        nn.InstanceNorm2d(cout, affine=True, eps=1e-3),  # <= here
        nn.GELU()
)

class CoarseUNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.pre = nn.Conv2d(1, base*4, 1)
        self.normp = nn.LayerNorm([base*4,70,70])
        self.ps    = nn.PixelShuffle(2)              
        self.enc1  = _blk(base,   base)
        self.pool1 = nn.MaxPool2d(2)                  
        self.enc2  = _blk(base,   base*2)
        self.pool2 = nn.MaxPool2d(2)                  
        self.enc3  = _blk(base*2, base*4)

        self.up2   = nn.ConvTranspose2d(base*4,base*2,2,2)     
        self.dec2  = _blk(base*4, base*2)
        self.up1   = nn.ConvTranspose2d(base*2,base,2,2)       
        self.dec1  = _blk(base*2, base)
        self.outc  = nn.Conv2d(base,1,1)

    def forward(self,c):
        x  = self.ps(self.normp(self.pre(c)))
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2],1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1],1))
        out= F.avg_pool2d(self.outc(d1),2)            
        return out.squeeze(1)

# --- full stage-1 model -----------------------------------------------------
class DualBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw  = _Stem(5 , EMB_D)
        self.fft  = _Stem(10, EMB_D)
        Wr = (70-PATCH_X)//STR_X+1
        Wf = (36-PATCH_X)//STR_X+1
        self.enc_r = Encoder(DEPTH,EMB_D,Wr)
        self.enc_f = Encoder(DEPTH,EMB_D,Wf)
        self.fuse  = nn.Sequential(nn.LayerNorm(2*EMB_D),
                                   nn.Linear(2*EMB_D,4*EMB_D),
                                   nn.GELU(), nn.Linear(4*EMB_D,70*70))
        self.coarse_norm = nn.GroupNorm(1,1)
        self.refine = CoarseUNet()

    def forward(self,s):
        B = s.size(0)
        r_tok = self.raw(s)                  # raw branch
        f_tok = self.fft(split_fft(s))
        z = torch.zeros(B,1,EMB_D,device=s.device)
        r = self.enc_r(torch.cat([z,r_tok],1))
        f = self.enc_f(torch.cat([z,f_tok],1))
        fused = torch.cat([r[:,1:].mean(1), f[:,1:].mean(1)],1)
        coarse = self.coarse_norm(self.fuse(fused).view(B,1,70,70))
        coarse = coarse.squeeze(1)
        pred   = self.refine(coarse.unsqueeze(1))
        return coarse, pred                           # both (B,70,70)



# ======================  Stage-1 TRAINING  ======================
EPOCHS1     = 60
BATCH1      = 2
ACCUM1      = 4
VAL_FRAC1   = 0.20
LR_BASE1    = 7e-4
LR_MIN1     = 3e-5
UNET_RATIO1 = 0.10
WARM_STEPS1 = 10_000
WEIGHT_DEC1 = 5e-4
CLIP1       = 100.

# ----- split index ------------------------------------------------
full_idx = build_index(BASE_DIR)
perm     = np.random.permutation(len(full_idx))
cut      = int(len(perm)*(1-VAL_FRAC1))
tr_idx, va_idx = perm[:cut], perm[cut:]

train_loader = DataLoader(VelDS([full_idx[i] for i in tr_idx]),
                          batch_size=BATCH1, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(VelDS([full_idx[i] for i in va_idx]),
                          batch_size=BATCH1, shuffle=False,
                          num_workers=4, pin_memory=True)

# ----- model, optimiser, scheduler --------------------------------
def augment(seis, p_drop=0.0, amp_jitter=False):
    if amp_jitter:
        amp = 1.0 + (torch.rand_like(seis[:, :1, :1, :1]) * 0.10 - 0.05)
        seis = seis * amp
    if p_drop > 0:
        mask = (torch.rand(seis.shape[0], 5, 1, 1, device=seis.device) > p_drop).float()
        seis = seis * mask
    return seis
model = DualBranch().to(DEVICE)

decay, no_wd = [], []
ref_ids = {id(p) for p in model.refine.parameters()}
for n, p in model.named_parameters():
    if id(p) in ref_ids: continue          # refine gets its own LR
    (no_wd if p.ndim==1 or n.endswith(".bias") else decay).append(p)

opt = torch.optim.AdamW(
    [{"params":decay,                         "lr":LR_BASE1},
     {"params":no_wd,                         "lr":LR_BASE1, "weight_decay":0.},
     {"params":model.refine.parameters(),     "lr":LR_BASE1*UNET_RATIO1}],
    betas=(0.9,0.95), weight_decay=WEIGHT_DEC1)

total_steps = EPOCHS1 * math.ceil(len(train_loader)/ACCUM1)
BASE_LR = 7e-4
def lr_main(step):
    if step < 5000:                 # 4 k-step warm-up
        return (step + 1)/5000
    prog = (step-5000)/(total_steps-5000)
    return 0.5 * (1 + math.cos(math.pi * min(prog, 1)))  # cosine
lr_unet = lambda s: lr_main(s) * UNET_RATIO1
sched   = LambdaLR(opt, lr_lambda=[lr_main, lr_main, lr_unet])

loss_fn = nn.L1Loss()
scaler  = GradScaler()
best_val = float('inf')

# ----- training loop ----------------------------------------------
for ep in range(1, EPOCHS1+1):
    model.train(); run=cnt=0
    prog = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"[Stage-1] Epoch {ep}/{EPOCHS1}", ncols=95)

    opt.zero_grad(set_to_none=True)
    for step,(seis,vel) in prog:
        seis,vel = seis.to(DEVICE), vel.to(DEVICE)
        seis = augment(
            seis, 
            p_drop = 0.10 if model.training else 0.10,   # same for val if you keep it
            amp_jitter = model.training                  # jitter only for train
        )
        with autocast(dtype=torch.float16):
            coarse, pred = model(seis)
            l_main   = loss_fn(pred,   vel)
            l_aux    = loss_fn(coarse, vel) * 0.1
            loss     = (l_main + l_aux) / ACCUM1

        scaler.scale(loss).backward()

        if (step+1)%ACCUM1 == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP1)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True); sched.step()

        run += l_main.item()*V_SCALE*seis.size(0); cnt += seis.size(0)
        prog.set_postfix(trainMAE=f"{run/cnt:6.1f}",
                         lr=f"{opt.param_groups[0]['lr']:.1e}")

    # ---- validation ------------------------------------------------
    model.eval(); val_sum=0.
    with torch.no_grad():
        for seis,vel in val_loader:
            seis,vel = seis.to(DEVICE), vel.to(DEVICE)
            _,pred   = model(seis)
            val_sum += loss_fn(pred, vel).item()*V_SCALE*seis.size(0)
    val_mae = val_sum / len(val_loader.dataset)
    print(f"  ↪ val {val_mae:6.1f} m/s")

    # ---- save best -------------------------------------------------
    if val_mae < best_val:
        best_val = val_mae
        torch.save(model.state_dict(), f"{CKPT_DIR1}/best.pth")
        print(" saved new best")

print("Stage-1 training done; best MAE =", best_val)



# -------------------- paths & basic config --------------------

BASE_DIR  = "/kaggle/input/waveform-inversion/train_samples"   # ← adjust if needed

CKPT_DIR1 = "/kaggle/working/ckpt_stage1"

CKPT_DIR2 = "/kaggle/working/ckpt_stage2"

import os, re, math, random, pickle, numpy as np, torch, torch.nn as nn

import torch.nn.functional as F

from torch.cuda.amp import autocast, GradScaler

from torch.utils.data import DataLoader, Dataset

from torch.optim.lr_scheduler import LambdaLR

from pathlib import Path

from tqdm import tqdm

 

for d in (CKPT_DIR1, CKPT_DIR2): Path(d).mkdir(parents=True, exist_ok=True)

 

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42); np.random.seed(42); random.seed(42)

 

# ---------------- dataset ---------------------------------------------------

_pat_d = re.compile(r'^(data|seis)[_\-]?')

_pat_m = re.compile(r'^(model|vel)[_\-]?')

 

def _pairs(dir_):

    f = [x for x in os.listdir(dir_) if x.endswith('.npy')]

    seis, vel = {}, {}

    for x in f:

        if x.startswith(("data", "seis")):  seis[_pat_d.sub('', x)] = x

        else:                              vel [_pat_m.sub('', x)] = x

    return [(os.path.join(dir_, seis[k]), os.path.join(dir_, vel[k]))

            for k in sorted(set(seis) & set(vel))]

 

def build_index(root):

    out = []

    for fam in sorted(os.listdir(root)):

        fd = os.path.join(root, fam)

        if not os.path.isdir(fd): continue

        for s, v in _pairs(fd):

            n = np.load(s, mmap_mode='r').shape[0]    # real #shots

            out += [(s, v, i) for i in range(n)]

    return out

 

V_SCALE = 4_500.0

class VelDS(Dataset):

    def __init__(self, items): self.items = items

    def __len__(self): return len(self.items)

    def __getitem__(self, k):

        s,v,i = self.items[k]

        seis = np.load(s, mmap_mode='r')[i].astype(np.float32)    # (5,1000,70)

        vel  = np.load(v, mmap_mode='r')[i].squeeze().astype(np.float32)

        return torch.from_numpy(seis), torch.from_numpy(vel / V_SCALE)


# ---------------- MetaFormer + coarse UNet -------------------

PATCH_T, PATCH_X = 32,16; STR_T,STR_X = 4,2

EMB_D, DEPTH, DROPOUT = 256, 6, 0.1

 

def split_fft(x: torch.Tensor) -> torch.Tensor:

    """log-magnitude FFT split into real / imag, sign-preserved."""

    F2  = torch.fft.rfft2(x, dim=(-2, -1))           # (B, 5, 1000, 36)

    mag = torch.log(torch.abs(F2) + 1e-6)

    real = torch.sign(F2.real) * mag

    imag = torch.sign(F2.imag) * mag

    return torch.cat([real, imag], dim=1)            # (B, 10, 1000, 36)

 

 

class _Stem(nn.Module):

    def __init__(self, in_ch: int, d: int):

        super().__init__()

        self.conv = nn.Conv2d(in_ch, d,

                              kernel_size=(PATCH_T, PATCH_X),

                              stride=(STR_T,  STR_X))

        self.ln   = nn.LayerNorm(d)

        self.drop = nn.Dropout(DROPOUT)

 

    def forward(self, x):                 # x : (B, C, 1000, 70)

        t = self.conv(x).flatten(2).transpose(1, 2)   # (B, N, d)

        return self.drop(self.ln(t))

 

 

class PoolMix(nn.Module):

    def __init__(self, W): super().__init__(); self.W=W

    def forward(self,x):

        cls,tok=x[:,:1],x[:,1:]; B,N,C=tok.shape; H=N//self.W

        y=tok.transpose(1,2).reshape(B,C,H,self.W)

        y=F.avg_pool2d(y,3,1,1)-y

        return torch.cat([cls, tok+y.flatten(2).transpose(1,2)],1)

 

class MetaBlock(nn.Module):

    def __init__(self,C,W):

        super().__init__()

        self.n1=nn.LayerNorm(C); self.mix=PoolMix(W)

        self.n2=nn.LayerNorm(C)

        self.ff=nn.Sequential(nn.Linear(C,4*C), nn.GELU(), nn.Dropout(DROPOUT),

                              nn.Linear(4*C,C))

    def forward(self,x): x=x+self.mix(self.n1(x)); return x+self.ff(self.n2(x))

 

class Encoder(nn.Module):

    def __init__(self,d,C,W):

        super().__init__()

        self.blks=nn.ModuleList([MetaBlock(C,W) for _ in range(d)])

    def forward(self,x):

        for b in self.blks: x=b(x)

        return x

 

# --- tiny helper to crop ----------------------------------------------------

def _crop(t, size):

    h,w=t.shape[-2:]; dh=(h-size[0])//2; dw=(w-size[1])//2

    return t[..., dh:dh+size[0], dw:dw+size[1]]

 

# --- Shuffle-UNet (coarse) --------------------------------------------------

def _blk(cin, cout):

    return nn.Sequential(

        nn.Conv2d(cin, cout, 3, 1, 1, bias=False),

        nn.InstanceNorm2d(cout, affine=True, eps=1e-3),  

        nn.GELU()

)

 

class CoarseUNet(nn.Module):

    def __init__(self, base=32):

        super().__init__()

        self.pre = nn.Conv2d(1, base*4, 1)

        self.normp = nn.LayerNorm([base*4,70,70])

        self.ps    = nn.PixelShuffle(2)               

        self.enc1  = _blk(base,   base)

        self.pool1 = nn.MaxPool2d(2)                 

        self.enc2  = _blk(base,   base*2)

        self.pool2 = nn.MaxPool2d(2)                  

        self.enc3  = _blk(base*2, base*4)

 

        self.up2   = nn.ConvTranspose2d(base*4,base*2,2,2)     
        self.dec2  = _blk(base*4, base*2)

        self.up1   = nn.ConvTranspose2d(base*2,base,2,2)       

        self.dec1  = _blk(base*2, base)

        self.outc  = nn.Conv2d(base,1,1)

 

    def forward(self,c):

        x  = self.ps(self.normp(self.pre(c)))

        e1 = self.enc1(x)

        e2 = self.enc2(self.pool1(e1))

        e3 = self.enc3(self.pool2(e2))

        d2 = self.dec2(torch.cat([self.up2(e3), e2],1))

        d1 = self.dec1(torch.cat([self.up1(d2), e1],1))

        out= F.avg_pool2d(self.outc(d1),2)            

        return out.squeeze(1)

 

# --- full stage-1 model -----------------------------------------------------

class DualBranch(nn.Module):

    def __init__(self):

        super().__init__()

        self.raw  = _Stem(5 , EMB_D)

        self.fft  = _Stem(10, EMB_D)

        Wr = (70-PATCH_X)//STR_X+1

        Wf = (36-PATCH_X)//STR_X+1

        self.enc_r = Encoder(DEPTH,EMB_D,Wr)

        self.enc_f = Encoder(DEPTH,EMB_D,Wf)

        self.fuse  = nn.Sequential(nn.LayerNorm(2*EMB_D),

                                   nn.Linear(2*EMB_D,4*EMB_D),

                                   nn.GELU(), nn.Linear(4*EMB_D,70*70))

        self.coarse_norm = nn.GroupNorm(1,1)

        self.refine = CoarseUNet()

 

    def forward(self,s):

        B = s.size(0)

        r_tok = self.raw(s)                  # raw branch

        f_tok = self.fft(split_fft(s))

        z = torch.zeros(B,1,EMB_D,device=s.device)

        r = self.enc_r(torch.cat([z,r_tok],1))

        f = self.enc_f(torch.cat([z,f_tok],1))

        fused = torch.cat([r[:,1:].mean(1), f[:,1:].mean(1)],1)

        coarse = self.coarse_norm(self.fuse(fused).view(B,1,70,70))

        coarse = coarse.squeeze(1)

        pred   = self.refine(coarse.unsqueeze(1))

        return coarse, pred                           # both (B,70,70)





import cv2, math
import torch
import torch.nn.functional as F

# ─── your existing fd8_misfit ────────────────────────────────────────────────
def fd8_misfit(coarse_v, seis_obs, src_ij, rec_ijs,
               dx=10.0, dt=0.001, n_step=8, alpha=0.3):
    # coarse_v: (70,70) in km/s
    # seis_obs: (R, T)
    # src_ij:   (2,)
    # rec_ijs:  list of (i,j)
    v = coarse_v * 1000.0                  # km/s  m/s
    dt2_v2 = (dt * v)**2
    p_prev = np.zeros_like(v, np.float32)
    p_curr = np.zeros_like(v, np.float32)
    mis_rec = np.zeros(len(rec_ijs), np.float32)

    c1, c2 = 1.0, -1.0/12.0
    for t in range(n_step):
        # source Ricker
        f_t = (1.0 - 2.0*(math.pi*8*(t*dt-0.04))**2) \
              * math.exp(-(math.pi*8*(t*dt-0.04))**2)
        i0,j0 = src_ij
        p_curr[i0,j0] += f_t

        lap = (
            -20.0*p_curr
            + c1*(np.roll(p_curr,1,0)+np.roll(p_curr,-1,0)
                 + np.roll(p_curr,1,1)+np.roll(p_curr,-1,1))
            + c2*(np.roll(p_curr,2,0)+np.roll(p_curr,-2,0)
                 + np.roll(p_curr,2,1)+np.roll(p_curr,-2,1))
        ) / (dx*dx)

        p_next = 2*p_curr - p_prev + dt2_v2 * lap
        p_prev, p_curr = p_curr, p_next

        for r,(ir,jr) in enumerate(rec_ijs):
            mis_rec[r] += abs(p_curr[ir,jr] - seis_obs[r,t])

    mis_map = np.zeros_like(v, np.float32)
    for r,(ir,jr) in enumerate(rec_ijs):
        mis_map[ir,jr] = mis_rec[r]

    mis_map = cv2.GaussianBlur(mis_map, (5,5), 0)
    mis_map = alpha * mis_map / (mis_map.max()+1e-6)
    return mis_map  # (70,70)


# ─── run once; writes ≈ 500×#families .npy files ─────────────────────────────
import os, re, numpy as np, skimage.draw as skdr
from pathlib import Path
from tqdm import tqdm

FAMDIR = "/kaggle/working"

# ----- helper: convert shot-id to (row, col) grid coordinates ---------------
src_xy = lambda shot_id: (0, shot_id % 70)          #  ⬅️  CHANGE if needed
rec_xys = lambda: [(0, j) for j in range(70)]        # 70 surface receivers

def build_illum_mask(src_ij, rec_ijs, H=70, W=70):
    mask = np.zeros((H, W), np.float32)
    si, sj = src_ij
    for ri, rj in rec_ijs:
        rr, cc = skdr.line(si, sj, ri, rj)   # Bresenham
        mask[rr, cc] += 1.0
    mask /= mask.max() + 1e-6
    return mask

for fam in sorted(os.listdir("/kaggle/input/waveform-inversion/train_samples")):
    famdir = Path(FAMDIR) / fam
    if not famdir.is_dir(): continue

    out_dir = famdir / "illum_masks"
    out_dir.mkdir(exist_ok=True)

    # assume 500 shots per *.npy file as in the baseline code
    for shot_id in tqdm(range(500), desc=fam):
        
        fn = out_dir / f"illum_{shot_id:03d}.npy"
        if fn.exists(): continue

        mask = build_illum_mask(src_xy(shot_id), rec_xys())
        np.save(fn, mask)
        print(f"Saved: {fn}")








import os, math
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

# ─── Helpers ────────────────────────────────────────────────────────────────

default_geom = { "src": (35,35),
                 "recs": [(10,10),(10,60),(60,10),(60,60)] }
# and in your dataset:
shot_geometry = defaultdict(lambda: default_geom)

def center_crop(src: torch.Tensor, size: tuple[int,int]):
    h,w = src.shape[-2:]
    dh, dw = (h - size[0])//2, (w - size[1])//2
    return src[..., dh:dh+size[0], dw:dw+size[1]]

def grad_xy(t: torch.Tensor):
    gy = t[:,:,1:,:] - t[:,:,:-1,:]
    gy = F.pad(gy, (0,0,0,1))
    gx = t[:,:,:,1:] - t[:,:,:,:-1]
    gx = F.pad(gx, (0,1,0,0))
    return gx, gy

def laplacian(t: torch.Tensor):
    gx, gy = grad_xy(t)
    gxx,_ = grad_xy(gx)
    _, gyy = grad_xy(gy)
    return gxx + gyy

def fd8_misfit(coarse_v, seis_obs, src_ij, rec_ijs,
               dx=10.0, dt=0.001, n_step=8, alpha=0.3):
    H,W = coarse_v.shape
    v = coarse_v * 1_000.
    dt2_v2 = (dt * v)**2
    p_prev = np.zeros((H,W), np.float32)
    p_curr = np.zeros((H,W), np.float32)
    mis_rec = np.zeros(len(rec_ijs), np.float32)
    c1, c2 = 1.0, -1.0/12.0

    for t in range(n_step):
        # Ricker source
        f_t = (1 - 2*(math.pi*8*(t*dt-0.04))**2) * math.exp(-(math.pi*8*(t*dt-0.04))**2)
        i0,j0 = src_ij; p_curr[i0,j0] += f_t

        lap = (
            -20*p_curr
            + c1*(np.roll(p_curr,1,0)+np.roll(p_curr,-1,0)
                 +np.roll(p_curr,1,1)+np.roll(p_curr,-1,1))
            + c2*(np.roll(p_curr,2,0)+np.roll(p_curr,-2,0)
                 +np.roll(p_curr,2,1)+np.roll(p_curr,-2,1))
        )/(dx*dx)

        p_next = 2*p_curr - p_prev + dt2_v2*lap
        p_prev, p_curr = p_curr, p_next

        for r,(ir,jr) in enumerate(rec_ijs):
            mis_rec[r] += abs(p_curr[ir,jr] - seis_obs[r, t, jr])

    mis_map = np.zeros_like(coarse_v)
    for r,(ir,jr) in enumerate(rec_ijs):
        mis_map[ir,jr] = mis_rec[r]
    mis_map = cv2.GaussianBlur(mis_map, (5,5), 0)
    mis_map *= alpha / (mis_map.max()+1e-6)
    return mis_map

# ─── Dataset (now unpacks 3-tuples) ─────────────────────────────────────────
class VelDS2(Dataset):
    def __init__(self, items, shot_geometry, mask_dir=None):
        """
        items: list of (sfile, vfile, shot_id)
        shot_geometry: dict[fam] -> {'src':(i,j), 'recs':[(i,j),...]}
        """
        self.items = items
        self.shot_geometry = shot_geometry
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sfile, vfile, shot_id = self.items[idx]
        fam = Path(sfile).parent.name  # derive family from folder name

        # 1) Load and slice numpy arrays
        seis_np = np.load(sfile, mmap_mode='r')[shot_id].astype(np.float32)  # (R,T,W)
        vel_np  = np.load(vfile, mmap_mode='r')[shot_id].astype(np.float32)  # (H,W) or (1,H,W)

        # 2) Ensure vel_np is 2D
        if vel_np.ndim == 3 and vel_np.shape[0] == 1:
            vel_np = vel_np[0]
        elif vel_np.ndim != 2:
            raise ValueError(f"Unexpected velocity shape: {vel_np.shape}")

        # 3) Build illumination mask
        src_ij  = self.shot_geometry[fam]['src']
        rec_ijs = self.shot_geometry[fam]['recs']
        if self.mask_dir:
            mask_file = os.path.join(self.mask_dir, f"{fam}_{shot_id}.npy")
            illum_np  = np.load(mask_file).astype(np.float32)
        else:
            from skimage.draw import line
            m = np.zeros_like(vel_np, dtype=np.float32)
            si, sj = src_ij
            for (ri, rj) in rec_ijs:
                rr, cc = line(si, sj, ri, rj)
                m[rr, cc] += 1
            illum_np = m / (m.max() + 1e-6)

        # 4) Convert to torch
        seis  = torch.from_numpy(seis_np)
        vel   = torch.from_numpy(vel_np)
        illum = torch.from_numpy(illum_np)

        return seis, vel, src_ij, rec_ijs, illum

# ─── Load & Freeze Stage-1 ──────────────────────────────────────────────────
coarse_net = DualBranch().to(DEVICE)
coarse_net.load_state_dict(torch.load(f"{CKPT_DIR1}/best.pth", map_location=DEVICE),
                           strict=False)
coarse_net.eval()
for p in coarse_net.parameters():
    p.requires_grad_(False)

# ─── Residual U-Net ─────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class RefineUNet(nn.Module):
    def __init__(self, in_ch=6, base=64):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.GroupNorm(1, co),
                nn.GELU()
            )

        # Encoder
        self.enc1  = blk(in_ch,   base)      # 70×70
        self.pool1 = nn.MaxPool2d(2)         # 35×35
        self.enc2  = blk(base,    base*2)    # 35×35
        self.pool2 = nn.MaxPool2d(2)         # 17×17
        self.enc3  = blk(base*2,  base*4)    # 17×17

        # Decoder
        self.up2   = nn.ConvTranspose2d(base*4, base*2, 2,2)  #  34×34
        self.dec2  = blk(base*4,  base*2)
        self.up1   = nn.ConvTranspose2d(base*2, base,   2,2)  #  68×68
        self.dec1  = blk(base*2,  base)
        self.head  = nn.Conv2d(base,   1,      3, padding=1)  # 68×68  68×68

        # Residual scaling parameter, start at 0.1
        self.gamma = nn.Parameter(torch.full((1,), 0.999))

modules():
            if isinstance(module, nn.Conv2d):
                if module is self.head:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Decoder stage 1
        u2  = self.up2(e3)
        e2c = center_crop(e2, u2.shape[-2:])
        d2  = self.dec2(torch.cat([u2, e2c], dim=1))

        # Decoder stage 2
        u1  = self.up1(d2)
        e1c = center_crop(e1, u1.shape[-2:])
        d1  = self.dec1(torch.cat([u1, e1c], dim=1))

        # Residual head
        delta = self.head(d1)                   
        delta = F.pad(delta, (1,1,1,1))         
        return delta


# ─── Build DataLoaders ─────────────────────────────────────────────────────
idx_all  = build_index(BASE_DIR)            # returns list[(sfile,vfile,shot_id)]
perm     = np.random.permutation(len(idx_all))
cut       = int(0.8 * len(idx_all))
tr_ds     = VelDS2([idx_all[i] for i in perm[:cut]], shot_geometry)
va_ds     = VelDS2([idx_all[i] for i in perm[cut:]], shot_geometry)

tr_loader = DataLoader(tr_ds, batch_size=2, shuffle=True,  num_workers=4, pin_memory=True)
va_loader = DataLoader(va_ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

# ─── Training Setup ────────────────────────────────────────────────────────
import math

# ─── Training Setup ────────────────────────────────────────────────────────
res_net = RefineUNet(in_ch=6, base=32).to(DEVICE)

# two param groups: one for gamma, one for all the rest
opt_res = torch.optim.AdamW([
    # Group 0: gamma, high LR, no weight decay
    {
      'params': [res_net.gamma],
      'lr': 1e-3,
      'weight_decay': 0.0
    },
    # Group 1: all other params, standard LR & decay
    {
      'params': [p for p in res_net.parameters() if p is not res_net.gamma],
      'lr': 7e-4,
      'weight_decay': 1e-4
    }
])
# Warmup + Cosine Decay Scheduler
num_epochs = 25
steps_per_epoch = len(tr_loader)
total_steps = num_epochs * steps_per_epoch
warmup_steps = 4000           # 10% of training for warmup

def lr_lambda(step):
    if step < warmup_steps:
        # linear warmup from 0 -> 1
        return float(step) / float(max(1, warmup_steps))
    # cosine decay from 1 -> 0
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

sched_res = LambdaLR(opt_res, lr_lambda=lr_lambda)
loss_fn   = nn.L1Loss()
scaler    = GradScaler()
V_SCALE   = 4_500.0

best_val = float('inf')
train_mae = 0
for ep in range(1, 40):
    # — train —
    res_net.train()
    train_sum = n_train = 0
    i = 0
    for seis, vel, srcs, recs, illum in tqdm(tr_loader, desc=f"[res] Ep{ep:2d}, [train MAE]: {train_mae}"):
        seis, vel, illum = seis.to(DEVICE), vel.to(DEVICE), illum.to(DEVICE)
        vel /= V_SCALE
        if i % 500 == 0:
            lrs = [pg['lr'] for pg in opt_res.param_groups]
            print(f"γ-LR = {lrs[0]:.2e}, conv-LR = {lrs[1]:.2e}")
        i += 1
        with torch.no_grad():
            coarse, _ = coarse_net(seis)     # (B,70,70)

        # physics misfit
        phys_list = []
        c_np, s_np = coarse.cpu().numpy(), seis.cpu().numpy()
        for b in range(coarse.size(0)):
            pm = fd8_misfit(c_np[b], s_np[b], srcs[b], recs[b])
            phys_list.append(torch.from_numpy(pm))
        phys = torch.stack(phys_list,0).unsqueeze(1).to(DEVICE)

        # grads & lap
        c4 = coarse.unsqueeze(1)
        gx, gy = grad_xy(c4)
        lap = laplacian(c4)

        inp = torch.cat([c4, gx, gy, lap, phys, illum.unsqueeze(1)],1)
        with autocast():
            delta = res_net(inp)               # (B,1,70,70)
            pred  = coarse + res_net.gamma * delta.squeeze(1)
            loss  = loss_fn(pred, vel)

        scaler.scale(loss).backward()
        scaler.step(opt_res); scaler.update()
        opt_res.zero_grad(); sched_res.step()

        train_sum += loss.item() * V_SCALE * vel.size(0)
        n_train   += vel.size(0)

    train_mae = train_sum / n_train

    # — validate —
    res_net.eval()
    val_sum = n_val = 0
    with torch.no_grad():
        for seis, vel, srcs, recs, illum in va_loader:
            seis, vel, illum = seis.to(DEVICE), vel.to(DEVICE), illum.to(DEVICE)
            coarse, _ = coarse_net(seis)
            vel /= V_SCALE
            phys_list = []
            c_np, s_np = coarse.cpu().numpy(), seis.cpu().numpy()
            for b in range(coarse.size(0)):
                pm = fd8_misfit(c_np[b], s_np[b], srcs[b], recs[b])
                phys_list.append(torch.from_numpy(pm))
            phys = torch.stack(phys_list,0).unsqueeze(1).to(DEVICE)

            c4 = coarse.unsqueeze(1)
            gx, gy = grad_xy(c4)
            lap = laplacian(c4)
            inp = torch.cat([c4, gx, gy, lap, phys, illum.unsqueeze(1)],1)

            delta = res_net(inp)
            pred  = coarse + res_net.gamma * delta.squeeze(1)
            val_sum += loss_fn(pred, vel).item() * V_SCALE * vel.size(0)
            n_val   += vel.size(0)

    val_mae = val_sum / n_val
    print(f"[res] Ep{ep:2d} | train {train_mae:6.1f} | val {val_mae:6.1f} | γ={res_net.gamma.item():.3f}")

    if val_mae < best_val:
        best_val = val_mae
        torch.save(res_net.state_dict(), f"{CKPT_DIR2}/refine_best.pth")
        print(f"-> Saved new best  (val {best_val:6.1f})")


