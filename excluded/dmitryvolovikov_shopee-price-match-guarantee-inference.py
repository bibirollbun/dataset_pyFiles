# =========================
# Shopee Inference Notebook (OOM-safe, FAISS-optional)
# =========================

import os, gc, math, sys
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------
# Config & paths
# -----------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = '/kaggle/input/shopee-product-matching'
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test_images')

# === Weights ===
# Option A: lightweight embedding package (recommended)
EMBED_PTH = '/kaggle/input/shopee-second-attempt/embedding_extractor (1).pth'  # <-- проверь путь
USE_EMBED_PKG = True

# Option B: full training checkpoint (если нет EMBED_PTH)
CKPT_PTH = None   # например: '/kaggle/input/your-upload/arcface_resnet50_shopee_ckpt.pth'

# Retrieval params
IMSIZE = 224
BATCH_SIZE = 32                # инференс эмбеддингов
NUM_WORKERS = 2
K_CAP = 50                     # лимит Shopee
K_SEARCH = 100                 # ищем шире, потом обрезаем до 50
MUTUAL = False                 # включи True, если хватает времени/памяти
FALLBACK_TAU = 0.50            # если best_tau не сохранён
QBS = 64                      # запросов на чанк в torch-fallback (регулируй при 70k)

# -----------------
# Data & transforms
# -----------------
test_df = pd.read_csv(TEST_CSV)

eval_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMSIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

class ShopeeTestDS(Dataset):
    def __init__(self, df, img_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row['image'])
        if not img_name.lower().endswith('.jpg'):
            img_name = f"{img_name}.jpg"
        img_path = os.path.join(self.img_root, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform: img = self.transform(img)
        return {'image': img, 'posting_id': row['posting_id']}

test_ds = ShopeeTestDS(test_df, TEST_IMG_DIR, eval_tfms)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

# -----------------
# Rebuild model (must match training)
# -----------------
def build_embedding_head(in_dim, emb_dim=512):
    return nn.Sequential(
        nn.Linear(in_dim, emb_dim, bias=False),
        nn.BatchNorm1d(emb_dim)
    )

if USE_EMBED_PKG:
    pkg = torch.load(EMBED_PTH, map_location=device)
    backbone_name = pkg['backbone_name']
    emb_dim = int(pkg['emb_dim'])
    best_tau = float(pkg.get('best_tau', FALLBACK_TAU))

    backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0).to(device).eval()
    embedding_head = build_embedding_head(backbone.num_features, emb_dim).to(device).eval()

    backbone.load_state_dict(pkg['state_dict']['backbone'])
    embedding_head.load_state_dict(pkg['state_dict']['embedding_head'])
    print(f"[LOAD] EMBED pkg | backbone={backbone_name}, emb_dim={emb_dim}, tau={best_tau:.2f}")
else:
    if not CKPT_PTH:
        raise ValueError("Set CKPT_PTH or USE_EMBED_PKG=True")
    ckpt = torch.load(CKPT_PTH, map_location=device)
    backbone_name = ckpt['backbone_name']
    emb_dim = int(ckpt['emb_dim'])
    best_tau = float(ckpt.get('best_tau', FALLBACK_TAU))

    backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0).to(device).eval()
    embedding_head = build_embedding_head(backbone.num_features, emb_dim).to(device).eval()

    backbone.load_state_dict(ckpt['state_dict']['backbone'])
    embedding_head.load_state_dict(ckpt['state_dict']['embedding_head'])
    print(f"[LOAD] FULL ckpt | backbone={backbone_name}, emb_dim={emb_dim}, tau={best_tau:.2f}")

@torch.no_grad()
def embed_loader(loader):
    embs, ids = [], []
    for b in tqdm(loader, desc="Embed/test"):
        x = b['image'].to(device, non_blocking=True)
        f = backbone(x)
        e = embedding_head(f)
        e = F.normalize(e, dim=1)
        embs.append(e.cpu())
        ids.extend(b['posting_id'])
    embs = torch.cat(embs, dim=0).numpy().astype('float32')  # (N, D)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    embs = (embs / norms).astype('float16')                   # экономим RAM
    return embs, ids

# -----------------
# Build embeddings
# -----------------
test_embs, test_ids = embed_loader(test_loader)
N, D = test_embs.shape
print(f"[INFO] Test embeddings: {test_embs.shape} (stored float16)")

# -----------------
# KNN search: FAISS if available, else torch chunked top-K
# -----------------
KQ = min(max(K_CAP, K_SEARCH), N)  # сколько соседей ищем до cap

try:
    import faiss
    use_faiss = True
except Exception:
    use_faiss = False
    print("[INFO] faiss не найден — используем torch chunked top-K")

if use_faiss:
    def build_ivf_index(vecs_f16, nlist=None, nprobe=16):
        D = vecs_f16.shape[1]
        if nlist is None:
            nlist = max(4096, int(len(vecs_f16)//16))
        xb = vecs_f16.astype('float32', copy=False)
        quantizer = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
        ntrain = min(20000, len(vecs_f16))
        sel = np.random.choice(len(vecs_f16), size=ntrain, replace=False)
        index.train(vecs_f16[sel].astype('float32', copy=False))
        index.add(xb)
        index.nprobe = nprobe
        return index

    def build_hnsw_index(vecs_f16, M=32, efSearch=64):
        D = vecs_f16.shape[1]
        index = faiss.IndexHNSWFlat(D, M)
        index.hnsw.efSearch = efSearch
        index.add(vecs_f16.astype('float32', copy=False))
        return index

    try:
        index = build_ivf_index(test_embs, nlist=None, nprobe=16)
        method = "IVF"
    except Exception as e:
        print("[WARN] IVF failed, fallback to HNSW:", e)
        try:
            index = build_hnsw_index(test_embs, M=32, efSearch=64)
            method = "HNSW"
        except Exception as e2:
            print("[WARN] HNSW failed, fallback to Flat:", e2)
            index = faiss.IndexFlatIP(D)
            index.add(test_embs.astype('float32', copy=False))
            method = "FlatIP"
    print(f"[FAISS] Built index: {method}")

    def faiss_search_batched(index, vecs_f16, K, bs=5000):
        Nq = vecs_f16.shape[0]
        I_all = np.empty((Nq, K), dtype='int32')
        D_all = np.empty((Nq, K), dtype='float32')
        ptr = 0
        while ptr < Nq:
            end = min(ptr + bs, Nq)
            q = vecs_f16[ptr:end].astype('float32', copy=False)
            Dq, Iq = index.search(q, K)
            D_all[ptr:end] = Dq
            I_all[ptr:end] = Iq
            ptr = end
        return D_all, I_all

    sims, idxs = faiss_search_batched(index, test_embs, K=KQ, bs=5000)

else:
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    db = torch.from_numpy(test_embs.astype('float32', copy=False)).to(torch_device, non_blocking=True)
    qbs = QBS
    idxs_list, sims_list = [], []
    for start in tqdm(range(0, N, qbs), desc="TopK (torch-chunk)"):
        q = db[start:start+qbs]                  # (qbs, D)
        sim_chunk = torch.matmul(q, db.T)        # (qbs, N), косинус (L2-норм)
        topk_vals, topk_idx = torch.topk(sim_chunk, k=KQ, dim=1, largest=True, sorted=True)
        idxs_list.append(topk_idx.cpu().numpy().astype('int32'))
        sims_list.append(topk_vals.cpu().numpy().astype('float32'))
        del sim_chunk, topk_vals, topk_idx
        if torch_device.type == 'cuda':
            torch.cuda.empty_cache()
    idxs = np.vstack(idxs_list)   # (N, KQ)
    sims = np.vstack(sims_list)   # (N, KQ)
    del db; gc.collect()

# -----------------
# Build matches with global tau & optional mutual check
# -----------------
tau = float(best_tau)
print(f"[INFO] Using global tau={tau:.2f}, KQ={KQ}, MUTUAL={MUTUAL}")

preds = []
for i in range(N):
    neigh = idxs[i]
    simv  = sims[i]
    cand = []
    for j, s in zip(neigh, simv):
        if s < tau:
            continue
        if MUTUAL:
            # взаимность: i должен быть в топ-KQ j
            if np.any(idxs[j] == i):
                cand.append(test_ids[j])
        else:
            cand.append(test_ids[j])
    if test_ids[i] not in cand:
        cand = [test_ids[i]] + cand
    preds.append(" ".join(cand[:K_CAP]))

sub = pd.DataFrame({'posting_id': test_ids, 'matches': preds})
out_path = '/kaggle/working/submission.csv'
sub.to_csv(out_path, index=False)
print(f"[SAVE] submission -> {out_path}")
display(sub.head())




