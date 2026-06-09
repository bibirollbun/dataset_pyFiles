# H690 â€” Robust submission generator (auto-match sample_submission headers)
# Paste into a Kaggle Notebook. Enable GPU for speed.
# NOTE: This script aims to produce a submission whose columns exactly match the
# sample_submission file included in the competition dataset (so "image_id" errors are fixed).

import os, math, random, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

# -------------------------
# Settings (tweak if needed)
# -------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = max(0, (os.cpu_count() or 2) - 2)

CACHE_ROOT = Path('/kaggle/working/h690_cache'); CACHE_ROOT.mkdir(parents=True, exist_ok=True)
PRECOMP_IMG_SIZE = 224
BATCH_SIZE_PRECOMP = 256 if torch.cuda.is_available() else 64
BATCH_SIZE_HEAD = 512
BACKBONE_FEATURE_DIM = 2048
PROJ_DIM = 128
PROFILE_ANGLES = 256
TOP_K = 8
SIMILARITY_THRESHOLD = 0.88
EPOCHS_HEAD = 3
LR_HEAD = 1e-3
TEMPERATURE = 0.5

print("Device:", DEVICE, "Num workers:", NUM_WORKERS)

# -------------------------
# Auto-detect dataset root
# -------------------------
def find_data_root(preferred_names=('h690','daxin','gdppc')):
    base = Path('/kaggle/input')
    if base.exists():
        for child in base.iterdir():
            low = child.name.lower()
            if any(p in low for p in preferred_names):
                return child
        entries = list(base.iterdir())
        if len(entries) == 1:
            return entries[0]
    for p in [Path('/input'), Path('/mnt/data'), Path.cwd()]:
        if p.exists():
            for child in p.iterdir():
                low = child.name.lower()
                if any(pn in low for pn in preferred_names):
                    return child
    return None

DATA_ROOT = find_data_root()
if DATA_ROOT is None:
    raise RuntimeError("Dataset root not found under /kaggle/input; attach competition dataset and rerun.")
print("Data root:", DATA_ROOT)

# -------------------------
# Build dataframe of all images (robust)
# -------------------------
def list_image_files(root: Path, exts=('.png','.jpg','.jpeg','.tif','.tiff')):
    res = []
    for ext in exts:
        res.extend(sorted([str(p) for p in root.rglob(f'*{ext}')]))
    return res

# Find a metadata CSV if present (we will still scan images if not)
csv_candidates = list(DATA_ROOT.glob('*.csv')) + list(DATA_ROOT.rglob('*.csv'))
meta_csv = None
for c in csv_candidates:
    ln = c.name.lower()
    # prefer obvious metadata or sample submission
    if 'sample' in ln or 'meta' in ln or 'metadata' in ln or 'images' in ln or 'shard' in ln or 'info' in ln:
        meta_csv = c
        # don't break on sample_submission; we still want sample_submission later
# prefer a CSV that isn't sample_submission for image metadata
meta_for_images = None
for c in csv_candidates:
    if c.name.lower().startswith('sample'):
        continue
    meta_for_images = c
    break
if meta_for_images is not None:
    meta_csv = meta_for_images

# Build df_all robustly
if meta_csv is not None:
    try:
        print("Reading CSV metadata candidate:", meta_csv.name)
        df_meta = pd.read_csv(meta_csv)
        # heuristics to find image paths
        if 'image_path' in df_meta.columns:
            df_meta['image_path'] = df_meta['image_path'].astype(str)
        elif 'filename' in df_meta.columns:
            df_meta['image_path'] = df_meta['filename'].apply(lambda x: str(DATA_ROOT / x))
        elif 'id' in df_meta.columns:
            # best-effort guess
            def guess_path(rid):
                for ext in ('.png','.jpg','.jpeg'):
                    p = DATA_ROOT / f"{rid}{ext}"
                    if p.exists(): return str(p)
                return str(DATA_ROOT / f"{rid}.png")
            df_meta['image_path'] = df_meta['id'].apply(guess_path)
        else:
            # fallback: scan files
            imgs = list_image_files(DATA_ROOT)
            df_meta = pd.DataFrame({'id':[Path(p).stem for p in imgs], 'image_path':imgs})
        # ensure id column
        if 'id' not in df_meta.columns:
            df_meta['id'] = df_meta['image_path'].apply(lambda x: Path(x).stem)
        df_all = df_meta[['id','image_path']].drop_duplicates().reset_index(drop=True)
    except Exception as e:
        print("Failed to read meta CSV robustly:", e)
        imgs = list_image_files(DATA_ROOT)
        df_all = pd.DataFrame({'id':[Path(p).stem for p in imgs], 'image_path':imgs})
else:
    imgs = list_image_files(DATA_ROOT)
    df_all = pd.DataFrame({'id':[Path(p).stem for p in imgs], 'image_path':imgs})

# Validate existence
df_all['exists'] = df_all['image_path'].apply(lambda x: Path(x).exists())
df_all = df_all[df_all['exists']].reset_index(drop=True)
print("Usable images:", len(df_all))
if len(df_all) == 0:
    raise RuntimeError("No usable images found under dataset root.")

# -------------------------
# Train/Val split (for head)
# -------------------------
df_all = df_all.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_train = int(0.90 * len(df_all))
df_train = df_all.iloc[:n_train].reset_index(drop=True)
df_val = df_all.iloc[n_train:].reset_index(drop=True)
print("Train/Val sizes:", len(df_train), len(df_val))

# -------------------------
# Precompute cache filenames
# -------------------------
FEAT_A_FILE = CACHE_ROOT / 'feat_a.npy'
FEAT_B_FILE = CACHE_ROOT / 'feat_b.npy'
PROFILE_FILE = CACHE_ROOT / 'profiles.npy'
IDS_FILE = CACHE_ROOT / 'ids.npy'

# -------------------------
# Helper: radial profile
# -------------------------
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

# -------------------------
# Precompute / load features
# -------------------------
def precompute_or_load(df_images):
    # returns ids (list), feats_a (N x 2048), feats_b (N x 2048), profiles (N x PROFILE_ANGLES)
    if FEAT_A_FILE.exists() and FEAT_B_FILE.exists() and PROFILE_FILE.exists() and IDS_FILE.exists():
        print("Loading cached features from", CACHE_ROOT)
        feats_a = np.load(FEAT_A_FILE, mmap_mode='r')
        feats_b = np.load(FEAT_B_FILE, mmap_mode='r')
        profiles = np.load(PROFILE_FILE, mmap_mode='r')
        ids = np.load(IDS_FILE)
        return list(ids), feats_a, feats_b, profiles

    print("Precomputing features & profiles (this runs once) ...")
    # backbone: try to load pretrained safely; fallback to weights=None if network blocked
    try:
        backbone_full = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    except Exception as e:
        print("Warning: could not load pretrained weights automatically:", e)
        print("Falling back to resnet50(weights=None). This is slower to converge but will run.")
        backbone_full = models.resnet50(weights=None)
    modules = list(backbone_full.children())[:-1]
    backbone = nn.Sequential(*modules).to(DEVICE)
    backbone.eval()
    for p in backbone.parameters(): p.requires_grad = False

    # simple transforms for precompute
    import torchvision.transforms as T
    aug1 = T.Compose([T.ToPILImage(), T.Resize((PRECOMP_IMG_SIZE, PRECOMP_IMG_SIZE)), T.RandomHorizontalFlip(p=0.5),
                      T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    aug2 = T.Compose([T.ToPILImage(), T.Resize((PRECOMP_IMG_SIZE, PRECOMP_IMG_SIZE)), T.RandomRotation(15),
                      T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    N = len(df_images)
    feats_a = np.zeros((N, BACKBONE_FEATURE_DIM), dtype=np.float32)
    feats_b = np.zeros((N, BACKBONE_FEATURE_DIM), dtype=np.float32)
    profiles = np.zeros((N, PROFILE_ANGLES), dtype=np.float32)
    ids = []

    batch_size = BATCH_SIZE_PRECOMP

    with torch.no_grad():
        for start in tqdm(range(0, N, batch_size), desc='Precompute batches'):
            end = min(N, start + batch_size)
            batch_idx = list(range(start, end))
            imgs = []
            for i in batch_idx:
                p = df_images.loc[i, 'image_path']
                try:
                    arr = np.array(Image.open(p).convert('RGB'))
                except Exception:
                    arr = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                imgs.append(arr)
                ids.append(df_images.loc[i, 'id'])
            # create tensors
            tensor_a = torch.stack([aug1(img) for img in imgs], dim=0).to(DEVICE)
            tensor_b = torch.stack([aug2(img) for img in imgs], dim=0).to(DEVICE)
            # forward
            featsA = backbone(tensor_a).view(len(batch_idx), -1).cpu().numpy()
            featsB = backbone(tensor_b).view(len(batch_idx), -1).cpu().numpy()
            for j, idx in enumerate(batch_idx):
                feats_a[idx, :] = featsA[j]
                feats_b[idx, :] = featsB[j]
            # profiles
            for j, arr in enumerate(imgs):
                try:
                    pil = Image.fromarray(arr).resize((PRECOMP_IMG_SIZE, PRECOMP_IMG_SIZE), Image.BICUBIC)
                    arr_small = np.array(pil)
                except Exception:
                    arr_small = cv2.resize(arr, (PRECOMP_IMG_SIZE, PRECOMP_IMG_SIZE))
                profiles[start + j] = compute_profile_from_arr(arr_small)
    # save
    np.save(FEAT_A_FILE, feats_a)
    np.save(FEAT_B_FILE, feats_b)
    np.save(PROFILE_FILE, profiles)
    np.save(IDS_FILE, np.array(ids))
    print("Saved cached features to", CACHE_ROOT)
    return list(ids), feats_a, feats_b, profiles

ids, feats_a, feats_b, profiles = precompute_or_load(df_all)

# -------------------------
# Create train/val splits for head training (use precomputed)
# -------------------------
N = len(ids)
perm = np.random.RandomState(SEED).permutation(N)
n_train = int(0.90 * N)
train_idx = perm[:n_train]
val_idx = perm[n_train:]

# datasets for head training
class PrecompDataset(Dataset):
    def __init__(self, feats_a, feats_b, profiles, ids):
        self.feats_a = feats_a
        self.feats_b = feats_b
        self.profiles = profiles
        self.ids = ids
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        return {
            'id': self.ids[idx],
            'a': torch.from_numpy(self.feats_a[idx]).float(),
            'b': torch.from_numpy(self.feats_b[idx]).float(),
            'profile': torch.from_numpy(self.profiles[idx]).float()
        }

train_ds = PrecompDataset(feats_a[train_idx], feats_b[train_idx], profiles[train_idx], [ids[i] for i in train_idx])
val_ds   = PrecompDataset(feats_a[val_idx],   feats_b[val_idx],   profiles[val_idx],   [ids[i] for i in val_idx])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_HEAD, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE_HEAD, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

# -------------------------
# Head model
# -------------------------
class HeadNet(nn.Module):
    def __init__(self, feat_dim=BACKBONE_FEATURE_DIM, prof_dim=PROFILE_ANGLES, proj_dim=PROJ_DIM):
        super().__init__()
        self.feat_mlp = nn.Sequential(nn.Linear(feat_dim, 1024), nn.ReLU(), nn.Linear(1024, proj_dim))
        self.prof_mlp = nn.Sequential(nn.Linear(prof_dim, 256), nn.ReLU(), nn.Linear(256, proj_dim))
        self.proj = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
    def forward(self, feat, prof):
        f = self.feat_mlp(feat); p = self.prof_mlp(prof)
        f = F.normalize(f, dim=1); p = F.normalize(p, dim=1)
        fused = F.normalize(f + p, dim=1)
        z = self.proj(fused); z = F.normalize(z, dim=1)
        return z

head = HeadNet().to(DEVICE)
opt = torch.optim.AdamW(head.parameters(), lr=LR_HEAD, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

def nt_xent(z1, z2, temp=TEMPERATURE):
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / temp
    N = z1.size(0)
    mask = (~torch.eye(2*N, device=sim.device).bool()).float()
    exp_sim = torch.exp(sim) * mask
    pos = torch.exp(torch.sum(z1 * z2, dim=1) / temp)
    pos = torch.cat([pos, pos], dim=0)
    denom = exp_sim.sum(dim=1) + 1e-9
    loss = -torch.log(pos / denom)
    return loss.mean()

# Train head quickly
print("Training head (frozen backbone, small projection head)...")
for epoch in range(1, EPOCHS_HEAD+1):
    head.train()
    tot_loss = 0.0; n=0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS_HEAD}"):
        a = batch['a'].to(DEVICE); b = batch['b'].to(DEVICE); prof = batch['profile'].to(DEVICE)
        opt.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                z1 = head(a, prof); z2 = head(b, prof); loss = nt_xent(z1, z2)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            z1 = head(a, prof); z2 = head(b, prof); loss = nt_xent(z1, z2)
            loss.backward(); opt.step()
        bs = a.size(0); tot_loss += float(loss.item()) * bs; n += bs
    print(f"Epoch {epoch} train loss: {tot_loss / max(1,n):.5f}")
    # quick val
    head.eval(); vloss=0.0; vn=0
    with torch.no_grad():
        for batch in val_loader:
            a = batch['a'].to(DEVICE); b = batch['b'].to(DEVICE); prof = batch['profile'].to(DEVICE)
            z1 = head(a, prof); z2 = head(b, prof); loss = nt_xent(z1, z2)
            vloss += float(loss.item()) * a.size(0); vn += a.size(0)
    if vn: print(f"Val loss: {vloss/vn:.5f}")

# -------------------------
# Compute final embeddings for all images and build graph
# -------------------------
print("Computing final embeddings for all images ...")
all_ds = PrecompDataset(feats_a, feats_b, profiles, ids)
all_loader = DataLoader(all_ds, batch_size=256, shuffle=False, num_workers=0)
head.eval()
embs = []
with torch.no_grad():
    for batch in tqdm(all_loader):
        f = batch['a'].to(DEVICE); prof = batch['profile'].to(DEVICE)
        z = head(f, prof)
        embs.append(z.cpu().numpy())
emb_all = np.vstack(embs)
# normalize
emb_all = emb_all / (np.linalg.norm(emb_all, axis=1, keepdims=True) + 1e-9)
print("Embeddings shape:", emb_all.shape)

# nearest neighbors
print("Building nearest neighbors ...")
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

# union-find assembly with threshold
class UF:
    def __init__(self):
        self.parent = {}
    def find(self, x):
        if x not in self.parent: self.parent[x] = x; return x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.parent[rb] = ra
    def comps(self):
        groups = defaultdict(list)
        for x in self.parent.keys():
            groups[self.find(x)].append(x)
        return list(groups.values())

uf = UF()
for node, neighs in neighbors.items():
    uf.find(node)
    for nid, sim in neighs:
        if sim >= SIMILARITY_THRESHOLD:
            uf.union(node, nid)
components = uf.comps()
if not components:
    components = [[k] for k in neighbors.keys()]
print("Assembled components:", len(components))

# create map shard_id -> component_id
comp_map = {}
for cid, comp in enumerate(components):
    for s in comp:
        comp_map[s] = int(cid)

# Save a generic components CSV (for debugging / reuse)
rows = []
for cid, comp in enumerate(components):
    for s in comp:
        rows.append({'component_id': cid, 'shard_id': s})
pd.DataFrame(rows).to_csv('/kaggle/working/assembled_components.csv', index=False)
print("Saved /kaggle/working/assembled_components.csv")

# -------------------------
# Build submission using sample_submission.csv header (robust)
# -------------------------
sample_submission_path = None
for c in csv_candidates:
    if 'sample' in c.name.lower():
        sample_submission_path = c
        break
if sample_submission_path is None:
    # try common name
    if (DATA_ROOT / 'sample_submission.csv').exists():
        sample_submission_path = DATA_ROOT / 'sample_submission.csv'

if sample_submission_path is None:
    print("Warning: sample_submission.csv not found in dataset. Generating a best-effort submission with columns image_id,component_id.")
    id_col = 'image_id'
    pred_col = 'component_id'
    sample_df = pd.DataFrame({id_col: [x for x in ids]})
else:
    sample_df = pd.read_csv(sample_submission_path)
    id_col = sample_df.columns[0]
    if len(sample_df.columns) < 2:
        raise RuntimeError("Sample submission seems to have only one column; cannot infer prediction column.")
    pred_col = sample_df.columns[1]
    print("Using sample submission:", sample_submission_path.name, "id_col:", id_col, "pred_col:", pred_col)

# Attempt to map sample IDs to your shard ids:
# sample IDs may be filename stems or filenames with extension. We'll try several strategies.
def map_id_to_comp(value):
    # try direct match
    if value in comp_map:
        return comp_map[value]
    # try as filename (with extension) -> check stems
    val = str(value)
    stem = Path(val).stem
    if stem in comp_map:
        return comp_map[stem]
    # try adding common extensions
    for ext in ['.png','.jpg','.jpeg','.tif','.tiff']:
        if (stem + ext) in comp_map:
            return comp_map[stem + ext]
    # try removing possible prefix/suffix
    if val.startswith('img_') and val[4:] in comp_map:
        return comp_map[val[4:]]
    # not found
    return -1

# Build submission frame
out_sub = sample_df[[id_col]].copy()
out_sub[pred_col] = out_sub[id_col].apply(map_id_to_comp).astype(int)

# Diagnostics: count missing
n_missing = int((out_sub[pred_col] == -1).sum())
print(f"Mapped predictions for {len(out_sub)} rows; missing/unmapped: {n_missing}")

# If many missing, try fallback: if sample ids are not the same universe, try using our ids in order
if n_missing > 0.5 * len(out_sub):
    print("Many sample IDs could not be matched to internal shard IDs. As a fallback we'll attempt to align by filename order.")
    # If sample contains same number of rows as our ids, map by order
    if len(out_sub) == len(ids):
        fallback_map = dict(zip(ids, [comp_map.get(i, -1) for i in ids]))
        out_sub[pred_col] = [fallback_map.get(Path(v).stem, -1) for v in out_sub[id_col]]
        n_missing = int((out_sub[pred_col] == -1).sum())
        print("After fallback mapping, missing:", n_missing)
    else:
        print("Fallback by order not possible: sample size != number of images.")

# Final save
submission_path = Path('/kaggle/working/submission.csv')
out_sub.to_csv(submission_path, index=False)
print("Saved submission to", submission_path)
print("Submission columns:", list(out_sub.columns))
print("You can now upload /kaggle/working/submission.csv to the competition.")

# End


