# H690 — Fast pipeline (45-minute target)
# Author: ChatGPT — production-ready Kaggle notebook
# Strategy: Precompute two augmented ResNet features per image + radial profile -> train small head quickly

# -------------------------
# 0) Imports & settings
# -------------------------
import os, math, random, time
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.neighbors import NearestNeighbors

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------------
# 1) Config (changeable)
# -------------------------
CACHE_ROOT = Path('/kaggle/working/h690_cache')   # cache for precomputed data
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

PRECOMP_IMG_SIZE = 224        # smaller for fast feature extraction
BACKBONE = 'resnet50'
BACKBONE_FEATURE_DIM = 2048   # ResNet50 feature dim after global pool
PROJ_DIM = 128                # projection dimension for contrastive loss
PROFILE_ANGLES = 256
BATCH_SIZE_PRECOMP = 256      # batch size when extracting backbone features (GPU heavy)
BATCH_SIZE_TRAIN = 512        # batch size when training head (operates on precomputed arrays → small GPU mem)
EPOCHS_HEAD = 3               # keep small; adjust if you have more time
LR_HEAD = 1e-3
TEMPERATURE = 0.5
TOP_K = 8
SIMILARITY_THRESHOLD = 0.88

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = max(0, (os.cpu_count() or 2) - 2)

print("Device:", DEVICE, "Num workers:", NUM_WORKERS)

# -------------------------
# 2) Auto-detect dataset root
# -------------------------
def find_data_root():
    base = Path('/kaggle/input')
    if base.exists():
        # try to find h690-like folder
        for child in base.iterdir():
            if 'h690' in child.name.lower() or 'daxin' in child.name.lower() or 'gdppc' in child.name.lower():
                return child
        # fallback: if only one dataset mounted, return it
        entries = list(base.iterdir())
        if len(entries) == 1:
            return entries[0]
    # other fallbacks
    for p in [Path('/input'), Path('/mnt/data'), Path.cwd()]:
        if p.exists():
            for child in p.iterdir():
                if 'h690' in child.name.lower() or 'daxin' in child.name.lower():
                    return child
    return None

DATA_ROOT = find_data_root()
if DATA_ROOT is None:
    raise RuntimeError("Could not auto-detect dataset under /kaggle/input. Attach the dataset and rerun.")
print("Data root:", DATA_ROOT)

# -------------------------
# 3) Build dataframe of images (robust)
# -------------------------
def list_image_files(root):
    exts = ('.png','.jpg','.jpeg','.tif','.tiff')
    files = []
    for ext in exts:
        files.extend(sorted([str(p) for p in Path(root).rglob(f'*{ext}')]))
    return files

# try to use metadata CSV if available
csvs = list(DATA_ROOT.glob('*.csv')) + list(DATA_ROOT.rglob('*.csv'))
meta_csv = None
for c in csvs:
    ln = c.name.lower()
    if 'meta' in ln or 'metadata' in ln or 'info' in ln or 'gdppc' in ln:
        meta_csv = c; break
if meta_csv is None and csvs:
    meta_csv = csvs[0]

if meta_csv:
    df = pd.read_csv(meta_csv)
    # try different heuristics to find filenames
    if 'image_path' in df.columns:
        df['image_path'] = df['image_path'].astype(str)
    elif 'filename' in df.columns:
        df['image_path'] = df['filename'].apply(lambda x: str(DATA_ROOT / x))
    elif 'id' in df.columns:
        def guess(x):
            for ext in ('.png','.jpg','.jpeg'):
                p = DATA_ROOT / f"{x}{ext}"
                if p.exists(): return str(p)
            return str(DATA_ROOT / f"{x}.png")
        df['image_path'] = df['id'].apply(guess)
    else:
        imgs = list_image_files(DATA_ROOT)
        df = pd.DataFrame({'id':[Path(p).stem for p in imgs], 'image_path':imgs})
else:
    imgs = list_image_files(DATA_ROOT)
    df = pd.DataFrame({'id':[Path(p).stem for p in imgs], 'image_path':imgs})

# ensure these files exist
df['exists'] = df['image_path'].apply(lambda x: Path(x).exists())
df = df[df['exists']].reset_index(drop=True)
print("Found images:", len(df))
if len(df) == 0:
    raise RuntimeError("No images found. Check DATA_ROOT or how files are organized.")

# -------------------------
# 4) Precompute: two augmented backbone features + radial profiles
#    - saves: feat_a.npy, feat_b.npy, profile.npy, ids.csv
# -------------------------
FEAT_A_FILE = CACHE_ROOT / 'feat_a.npy'
FEAT_B_FILE = CACHE_ROOT / 'feat_b.npy'
PROFILE_FILE = CACHE_ROOT / 'profiles.npy'
IDS_FILE = CACHE_ROOT / 'ids.npy'

def do_precompute(df):
    if FEAT_A_FILE.exists() and FEAT_B_FILE.exists() and PROFILE_FILE.exists() and IDS_FILE.exists():
        print("Precomputed files found. Loading.")
        feats_a = np.load(FEAT_A_FILE, mmap_mode='r')
        feats_b = np.load(FEAT_B_FILE, mmap_mode='r')
        profiles = np.load(PROFILE_FILE, mmap_mode='r')
        ids = np.load(IDS_FILE)
        return ids, feats_a, feats_b, profiles

    print("Starting precompute of features and profiles ...")
    # setup backbone (frozen)
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # modern API
    modules = list(backbone.children())[:-1]  # remove fc
    backbone = nn.Sequential(*modules).to(DEVICE)
    backbone.eval()
    for p in backbone.parameters(): p.requires_grad = False

    # image augment pipeline for precompute: two different strong-ish augmentations
    aug1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(PRECOMP_IMG_SIZE, scale=(0.7,1.0), ratio=(0.9,1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    aug2 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(PRECOMP_IMG_SIZE, scale=(0.6,0.98), ratio=(0.85,1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.06),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    def img_loader(path):
        img = Image.open(path).convert('RGB')
        return np.array(img)

    N = len(df)
    feats_a = np.zeros((N, BACKBONE_FEATURE_DIM), dtype=np.float32)
    feats_b = np.zeros((N, BACKBONE_FEATURE_DIM), dtype=np.float32)
    profiles = np.zeros((N, PROFILE_ANGLES), dtype=np.float32)
    ids = []

    # create indices list for batching
    idxs = list(range(N))
    batch_size = BATCH_SIZE_PRECOMP if torch.cuda.is_available() else 64

    # helper: compute radial profile from resized image
    def compute_profile_from_arr(arr_rgb, angles=PROFILE_ANGLES):
        gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), 0)
        th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(angles, dtype=np.float32)
        c = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(th)
        cv2.drawContours(mask, [c], -1, 255, -1)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return np.zeros(angles, dtype=np.float32)
        cx, cy = xs.mean(), ys.mean()
        h, w = mask.shape
        prof = np.zeros(angles, dtype=np.float32)
        angles_arr = np.linspace(0, 2*np.pi, angles, endpoint=False)
        for i, a in enumerate(angles_arr):
            dx, dy = math.cos(a), math.sin(a)
            r = 0
            while True:
                x = int(round(cx + r*dx)); y = int(round(cy + r*dy))
                if x < 0 or x >= w or y < 0 or y >= h: break
                if mask[y,x] == 0: break
                r += 1
                if r > max(w,h): break
            prof[i] = r
        mx = prof.max()
        if mx > 0:
            prof = prof / (mx + 1e-9)
        return prof.astype(np.float32)

    # iterate in batches to extract features (two augmented views)
    with torch.no_grad():
        for start in tqdm(range(0, N, batch_size), desc='Precompute batches'):
            end = min(N, start + batch_size)
            batch_indices = idxs[start:end]
            # load images to list for augmentation
            imgs = []
            for i in batch_indices:
                p = df.loc[i, 'image_path']
                arr = img_loader(p)
                imgs.append(arr)
                ids.append(df.loc[i, 'id'])
            # create two augmented batches
            tensor_a = torch.stack([aug1(img) for img in imgs], dim=0).to(DEVICE)
            tensor_b = torch.stack([aug2(img) for img in imgs], dim=0).to(DEVICE)
            # backbone forward (B x C x H x W) -> global pool (B x feat)
            featsA = backbone(tensor_a).view(len(batch_indices), -1).cpu().numpy()
            featsB = backbone(tensor_b).view(len(batch_indices), -1).cpu().numpy()
            # store
            for j, idx in enumerate(batch_indices):
                feats_a[idx, :] = featsA[j]
                feats_b[idx, :] = featsB[j]
            # compute profiles from resized (use PRECOMP_IMG_SIZE)
            for j, arr in enumerate(imgs):
                try:
                    pil = Image.fromarray(arr).resize((PRECOMP_IMG_SIZE, PRECOMP_IMG_SIZE), Image.BICUBIC)
                    arr_small = np.array(pil)
                except Exception:
                    arr_small = cv2.resize(arr, (PRECOMP_IMG_SIZE, PRECOMP_IMG_SIZE))
                profiles[start + j] = compute_profile_from_arr(arr_small)
    # save arrays to disk
    np.save(FEAT_A_FILE, feats_a)
    np.save(FEAT_B_FILE, feats_b)
    np.save(PROFILE_FILE, profiles)
    np.save(IDS_FILE, np.array(ids))
    print("Precompute done. Files saved to:", CACHE_ROOT)
    return np.array(ids), feats_a, feats_b, profiles

ids, feats_a, feats_b, profiles = do_precompute(df)

# -------------------------
# 5) Create Dataset that loads precomputed feats and profiles
# -------------------------
class PrecompContrastiveDataset(Dataset):
    def __init__(self, feats_a, feats_b, profiles, ids):
        assert len(feats_a) == len(feats_b) == len(profiles) == len(ids)
        self.feats_a = feats_a.astype(np.float32)
        self.feats_b = feats_b.astype(np.float32)
        self.profiles = profiles.astype(np.float32)
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        a = self.feats_a[idx]
        b = self.feats_b[idx]
        prof = self.profiles[idx]
        return {
            'id': self.ids[idx],
            'a': torch.from_numpy(a),
            'b': torch.from_numpy(b),
            'profile': torch.from_numpy(prof)
        }

# -------------------------
# 6) Head model that fuses feat + profile -> projection
# -------------------------
class HeadNet(nn.Module):
    def __init__(self, feat_dim=BACKBONE_FEATURE_DIM, profile_dim=PROFILE_ANGLES, proj_dim=PROJ_DIM):
        super().__init__()
        # small MLP for feat
        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, proj_dim)
        )
        # small MLP for profile
        self.prof_mlp = nn.Sequential(
            nn.Linear(profile_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )
        # final fusion projection
        self.proj = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
    def forward(self, feat, prof):
        # feat: B x feat_dim (not normalized); prof: B x profile_dim
        f = self.feat_mlp(feat)
        p = self.prof_mlp(prof)
        f = F.normalize(f, dim=1)
        p = F.normalize(p, dim=1)
        fused = F.normalize(f + p, dim=1)  # B x proj_dim
        z = self.proj(fused)
        z = F.normalize(z, dim=1)
        return z

# -------------------------
# 7) NT-Xent (efficient on precomputed embeddings)
# -------------------------
def nt_xent(z1, z2, temperature=TEMPERATURE):
    z = torch.cat([z1, z2], dim=0)  # 2N x D
    sim = torch.matmul(z, z.t()) / temperature  # cosine since z normalized
    N = z1.size(0)
    mask = (~torch.eye(2*N, device=z.device).bool()).float()
    exp_sim = torch.exp(sim) * mask
    # positive sims: i <-> i+N
    pos = torch.exp(torch.sum(z1 * z2, dim=1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    denom = exp_sim.sum(dim=1) + 1e-9
    loss = -torch.log(pos / denom)
    return loss.mean()

# -------------------------
# 8) Train head quickly
# -------------------------
# split dataset into train/val (90/10)
N = len(ids)
perm = np.random.RandomState(SEED).permutation(N)
n_train = int(0.90 * N)
train_idx = perm[:n_train]
val_idx = perm[n_train:]

train_dataset = PrecompContrastiveDataset(feats_a[train_idx], feats_b[train_idx], profiles[train_idx], ids[train_idx])
val_dataset = PrecompContrastiveDataset(feats_a[val_idx], feats_b[val_idx], profiles[val_idx], ids[val_idx])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

print("Train/Val sizes:", len(train_dataset), len(val_dataset))

head = HeadNet(feat_dim=BACKBONE_FEATURE_DIM, profile_dim=PROFILE_ANGLES, proj_dim=PROJ_DIM).to(DEVICE)
optimizer = torch.optim.AdamW(head.parameters(), lr=LR_HEAD, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

start_time = time.time()
for epoch in range(1, EPOCHS_HEAD+1):
    head.train()
    total_loss = 0.0
    count = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS_HEAD}")
    for batch in pbar:
        a = batch['a'].to(DEVICE).float()
        b = batch['b'].to(DEVICE).float()
        prof = batch['profile'].to(DEVICE).float()
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                z1 = head(a, prof)
                z2 = head(b, prof)
                loss = nt_xent(z1, z2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            z1 = head(a, prof)
            z2 = head(b, prof)
            loss = nt_xent(z1, z2)
            loss.backward()
            optimizer.step()
        batch_sz = a.size(0)
        total_loss += float(loss.item()) * batch_sz
        count += batch_sz
        pbar.set_postfix(loss=total_loss / count)
    avg_loss = total_loss / max(1, count)
    print(f"Epoch {epoch} train loss: {avg_loss:.5f}")
    # quick val step (compute average loss)
    head.eval()
    val_loss = 0.0; vcount = 0
    with torch.no_grad():
        for batch in val_loader:
            a = batch['a'].to(DEVICE).float()
            b = batch['b'].to(DEVICE).float()
            prof = batch['profile'].to(DEVICE).float()
            z1 = head(a, prof); z2 = head(b, prof)
            loss = nt_xent(z1, z2)
            val_loss += float(loss.item()) * a.size(0)
            vcount += a.size(0)
    if vcount:
        print(f"Val loss: {val_loss/vcount:.5f}")
    # save head checkpoint
    ckpt = CACHE_ROOT / f'head_epoch{epoch}.pth'
    torch.save(head.state_dict(), ckpt)
    print("Saved checkpoint:", ckpt)

print("Head training done in {:.1f} minutes".format((time.time() - start_time)/60))

# -------------------------
# 9) Produce final embeddings for all images (fusion via head on backbone feats)
# -------------------------
print("Producing final fused embeddings for all images ...")
all_dataset = PrecompContrastiveDataset(feats_a, feats_b, profiles, ids)  # feats_a used as canonical view
all_loader = DataLoader(all_dataset, batch_size=256, shuffle=False, num_workers=0)
head.eval()
embs = []
with torch.no_grad():
    for batch in tqdm(all_loader):
        f = batch['a'].to(DEVICE).float()
        prof = batch['profile'].to(DEVICE).float()
        z = head(f, prof)  # normalized
        embs.append(z.cpu().numpy())
emb_all = np.vstack(embs)
# ensure normalized
norms = np.linalg.norm(emb_all, axis=1, keepdims=True)
emb_all = emb_all / (norms + 1e-9)
print("Embeddings shape:", emb_all.shape)

# save embeddings
np.save(CACHE_ROOT / 'emb_all.npy', emb_all)
np.save(CACHE_ROOT / 'ids_all.npy', np.array(ids))
print("Saved final embeddings to cache")

# -------------------------
# 10) Build top-k neighbors & assemble (fast)
# -------------------------
print("Building top-k neighbors ...")
nn = NearestNeighbors(n_neighbors=min(TOP_K+1, len(emb_all)), metric='cosine', n_jobs=-1)
nn.fit(emb_all)
dists, idxs = nn.kneighbors(emb_all)
neighbors = {}
for i, id_ in enumerate(ids):
    neighs = []
    for j_idx, d in zip(idxs[i], dists[i]):
        if j_idx == i: continue
        sim = 1.0 - float(d)
        neighs.append((ids[j_idx], sim))
        if len(neighs) >= TOP_K: break
    neighbors[id_] = neighs

# union-find assembly
class UF:
    def __init__(self):
        self.p = {}
    def find(self, x):
        if x not in self.p: self.p[x] = x; return x
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra
    def comps(self):
        groups = defaultdict(list)
        for k in self.p.keys():
            groups[self.find(k)].append(k)
        return list(groups.values())

uf = UF()
for n, neighs in neighbors.items():
    uf.find(n)
    for nid, sim in neighs:
        if sim >= SIMILARITY_THRESHOLD:
            uf.union(n, nid)
components = uf.comps()
# fallback: if no unions (rare), make singletons
if not components:
    components = [[k] for k in neighbors.keys()]

print("Assembled components:", len(components))
# save components
rows = []
for cid, comp in enumerate(components):
    for s in comp:
        rows.append({'component_id': cid, 'shard_id': s})
pd.DataFrame(rows).to_csv('/kaggle/working/assembled_components.csv', index=False)
print("Saved assembled components to /kaggle/working/assembled_components.csv")

print("Pipeline complete. You can now iterate (augmentations, threshold, top_k, proj_dim) to improve score.")


