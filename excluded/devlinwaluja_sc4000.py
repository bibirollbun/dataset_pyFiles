import cv2
import os
import numpy as np

def resize(folder_path, output_folder_name):
    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in input folder
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)

        # Load image (support PNG with alpha)
        image = cv2.imread(img_path, -1)
        if image is None:
            print(f"⚠️ Skipping {filename} (not a valid image)")
            continue

        # Resize to 1280 (width) x 1600 (height)
        resized = cv2.resize(image, (1280, 1600))

        # Save to output folder (same filename)
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, resized)
        print(f"✅ Saved resized: {out_path}")
    
# overwrites
def resize(folder_path):
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)

        # Only process images
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        # Load image (preserve alpha for PNGs)
        image = cv2.imread(img_path, -1)
        if image is None:
            print(f"⚠️ Skipping {filename} (not a valid image)")
            continue

        # Resize to 1280 (width) x 1600 (height)
        resized = cv2.resize(image, (1280, 1600))

        # Overwrite original file
        cv2.imwrite(img_path, resized)



import cv2
import os
import subprocess

def resize_with_subfolders(input_root, output_root):
    # Create main output folder
    os.makedirs(output_root, exist_ok=True)

    # Loop through subfolders in input_root
    for subfolder in os.listdir(input_root):
        subfolder_path = os.path.join(input_root, subfolder)
        if not os.path.isdir(subfolder_path):
            continue  # skip files at root level

        # Name the output subfolder
        out_subfolder_name = f"{subfolder}_resized"
        out_subfolder_path = os.path.join(output_root, out_subfolder_name)
        os.makedirs(out_subfolder_path, exist_ok=True)

        # Loop through images in subfolder
        for filename in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, filename)

            # Load image (support PNG with alpha)
            image = cv2.imread(img_path, -1)
            if image is None:
                print(f"⚠️ Skipping {filename} (not a valid image)")
                continue

            # Resize to 1280 (width) x 1600 (height)
            resized = cv2.resize(image, (1280, 1600))

            # Save to output subfolder
            out_path = os.path.join(out_subfolder_path, filename)
            cv2.imwrite(out_path, resized)



import torch
import numpy as np
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel
from PIL import Image, ImageFile
import torchvision.transforms as T

# allow PIL to load slightly truncated/partial PNGs instead of throwing
ImageFile.LOAD_TRUNCATED_IMAGES = True

DINOV2_DIR = Path("/kaggle/input/dinov2/pytorch/base/1")

def load_dinov2_from_dir(local_dir: Path):
    """
    local_dir must contain:
      - config.json
      - preprocessor_config.json
      - pytorch_model.bin

    Returns (model, preprocess, device).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # You instantiate processor here mostly for correctness, e.g. size info.
    # You don't strictly need to return it since we build our own torchvision pipeline.
    _ = AutoImageProcessor.from_pretrained(str(local_dir))

    model = AutoModel.from_pretrained(str(local_dir))
    model = model.eval().to(device)

    preprocess = T.Compose([
        T.Resize(518, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(518),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])
    
    return model, preprocess, device


model, preprocess, device = load_dinov2_from_dir(DINOV2_DIR)
print("Model loaded on:", device)


@torch.no_grad()
def embed_image(
    img_path: str,
    model,
    preprocess,
    device: str = "cpu",
    l2_normalize: bool = True
) -> np.ndarray:
    """
    Load image from img_path, force RGB (so RGBA / grayscale won't break),
    preprocess -> model -> 1D descriptor, L2-normalize, return np.float32.

    This is robust to:
    - alpha channel PNGs
    - grayscale images
    - slightly truncated PNGs
    - HF model outputs that aren't just raw tensors
    """
    # 1. Load image and force exactly 3 channels
    img = Image.open(img_path).convert("RGB")

    # 2. Preprocess into model input tensor
    x = preprocess(img).unsqueeze(0).to(device)  # [1, 3, H, W]

    # 3. Forward pass
    feats = model(x)

    # 4. Handle different return types:
    #    - Some HF vision models return BaseModelOutput with .last_hidden_state
    #    - Some return just a tensor
    if isinstance(feats, dict):
        # e.g. {'last_hidden_state': ..., 'pooler_output': ...}
        if "pooler_output" in feats and feats["pooler_output"] is not None:
            tensor_out = feats["pooler_output"]          # [1, D]
        elif "last_hidden_state" in feats:
            # take CLS token [0,0,:] or mean pool
            lh = feats["last_hidden_state"]              # [1, seq, dim]
            tensor_out = lh.mean(dim=1)                  # simple global pool -> [1, D]
        else:
            # fallback: grab first value in dict
            tensor_out = list(feats.values())[0]
    else:
        # Could be a tuple or a plain tensor
        if isinstance(feats, (list, tuple)):
            tensor_out = feats[0]
        else:
            tensor_out = feats

    # 5. Flatten to shape [1, D] if needed
    if tensor_out.ndim > 2:
        # e.g. [1, C, h, w] -> global average pool then flatten
        tensor_out = tensor_out.mean(dim=list(range(2, tensor_out.ndim)))
        # now [1, C]

    # 6. Squeeze batch to get [D]
    vec = tensor_out.squeeze(0).detach().cpu().numpy().astype(np.float32)

    # 7. Optional L2 norm
    if l2_normalize:
        denom = np.linalg.norm(vec) + 1e-12
        vec = vec / denom

    return vec



from pathlib import Path
import csv
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def list_images_under(roots):
    imgs = []
    for root in roots:
        r = Path(root)
        if not r.exists():
            print(f"[WARN] Missing: {r}")
            continue
        for p in r.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                imgs.append(p)
    return imgs

def build_vectors_csv(
    roots=("test_resized",),
    l2_normalize=True,
    out_csv="test_vectors.csv",
):
    all_imgs = list_images_under(roots)
    print(f"Found {len(all_imgs)} images under {roots}.")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "file", "vector"])  # header

        for p in all_imgs:
            try:
                vec = embed_image(str(p), model, preprocess, device, l2_normalize=l2_normalize)
                folder = p.parent.name
                file = p.stem
                # store vector as string of floats
                writer.writerow([folder, file, vec.tolist()])
            except Exception as e:
                print(f"[ERROR] {p}: {e}")
                continue

    print(f"[OK] Saved CSV: {out_csv} with {len(all_imgs)} rows")


from tqdm.auto import tqdm    # nice in-notebook progress bars
tqdm.pandas()                 # adds .progress_apply if you need it


# === IMC2025 — MASt3R-style clustering for Task 1 (scene grouping) ===
# With tqdm progress bars
import os, math, cv2, numpy as np, pandas as pd, networkx as nx
from typing import Optional, List, Tuple, Dict
from tqdm.auto import tqdm
tqdm.pandas()  # optional: enables .progress_apply on pandas

# ---------- file resolver ----------
_EXTS = [".png",".jpg",".jpeg",".JPG",".PNG",".JPEG",".webp",".bmp",".tif",".tiff"]
def resolve_path(root: str, folder: str, fname: str) -> Optional[str]:
    base = os.path.join(root, folder, fname)
    if os.path.exists(base): return base
    for e in _EXTS:
        p = base + e
        if os.path.exists(p): return p
    return None

# ---------- global retrieval (cosine on your embeddings) ----------
def shortlist_indices(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    # X assumed L2-normalized
    sims = X @ X.T                              # cosine similarity
    np.fill_diagonal(sims, -1.0)
    idx = np.argpartition(-sims, kth=np.minimum(k, X.shape[0]-1), axis=1)[:, :k]
    # keep strongest neighbors sorted desc
    row = np.arange(X.shape[0])[:, None]
    s = sims[row, idx]
    order = np.argsort(-s, axis=1)
    return idx[row, order], sims

# ---------- geometric verification backends ----------
def _try_mast3r():
    try:
        # If you have MASt3R installed, adapt the import/API below to your env.
        from mast3r.matcher import FastMatch  # placeholder; adjust to your local import
        return FastMatch()
    except Exception:
        return None

_MAST3R = _try_mast3r()

# ORB fallback (fast, CPU)
_ORB = cv2.ORB_create(nfeatures=2000)
def _orb_desc(gray: np.ndarray):
    return _ORB.detectAndCompute(gray, None)

def _load_gray(path: str, max_side: int = 960) -> Optional[np.ndarray]:
    if path is None: return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    h, w = img.shape[:2]; m = max(h, w)
    if m > max_side:
        s = max_side / m
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img

def verify_pair_mast3r(pA: str, pB: str, mast3r, max_side=960) -> Tuple[int, float]:
    # returns (n_inliers_like, score) — adapt to your mast3r binding
    try:
        n_inliers, score = mast3r.match(pA, pB, max_side=max_side)  # pseudo-API
        return int(n_inliers), float(score)
    except Exception:
        return 0, 0.0

def _robust_fundamental(ptsA, ptsB, ransac_px=2.5):
    """Try USAC_MAGSAC, then FM_RANSAC. Returns (n_inliers, mask) or (0, None)."""
    # OpenCV likes float64 here
    A = np.asarray(ptsA, np.float64)
    B = np.asarray(ptsB, np.float64)
    if len(A) < 8:
        return 0, None
    # basic sanity: remove exact duplicates
    AB = np.hstack([A, B])
    _, uniq_idx = np.unique(AB, axis=0, return_index=True)
    A = A[uniq_idx]; B = B[uniq_idx]
    if len(A) < 8:
        return 0, None
    # spread check (avoid all points on a pixel or line)
    if (A.std(axis=0).max() < 1e-3) or (B.std(axis=0).max() < 1e-3):
        return 0, None
    # 1) USAC_MAGSAC
    try:
        F, m = cv2.findFundamentalMat(A, B, cv2.USAC_MAGSAC, float(max(0.5, ransac_px)), 0.999, 2000)
        if F is not None and m is not None:
            n_in = int(m.sum())
            if n_in >= 8:
                return n_in, m
    except cv2.error:
        pass
    # 2) FM_RANSAC (classic)
    try:
        F, m = cv2.findFundamentalMat(A, B, cv2.FM_RANSAC, float(max(0.5, ransac_px)), 0.999)
        if F is not None and m is not None:
            n_in = int(m.sum())
            if n_in >= 8:
                return n_in, m
    except cv2.error:
        pass
    return 0, None

def _robust_homography(ptsA, ptsB, ransac_px=3.0):
    """Homography fallback for near-planar pairs."""
    A = np.asarray(ptsA, np.float64)
    B = np.asarray(ptsB, np.float64)
    if len(A) < 4:
        return 0, None
    try:
        H, m = cv2.findHomography(A, B, cv2.USAC_MAGSAC, float(max(0.5, ransac_px)), 0.999, 2000)
        if H is not None and m is not None:
            return int(m.sum()), m
    except cv2.error:
        pass
    try:
        H, m = cv2.findHomography(A, B, cv2.RANSAC, float(max(0.5, ransac_px)))
        if H is not None and m is not None:
            return int(m.sum()), m
    except cv2.error:
        pass
    return 0, None

def verify_pair_orb(pA: str, pB: str, ratio=0.8, ransac_px=2.5, max_side=960) -> Tuple[int, float]:
    A = _load_gray(pA, max_side=max_side); B = _load_gray(pB, max_side=max_side)
    if A is None or B is None: 
        return 0, 0.0
    kpA, desA = _ORB.detectAndCompute(A, None)
    kpB, desB = _ORB.detectAndCompute(B, None)
    if desA is None or desB is None or len(kpA) < 8 or len(kpB) < 8:
        return 0, 0.0

    # KNN + ratio
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(desA, desB, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2: 
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < 8:
        return 0, 0.0

    # dedup by query AND train
    best_by_q = {}
    for m in good:
        q = m.queryIdx
        if (q not in best_by_q) or (m.distance < best_by_q[q].distance):
            best_by_q[q] = m
    # also dedup by train to avoid many-to-one
    best_by_t = {}
    for m in best_by_q.values():
        t = m.trainIdx
        if (t not in best_by_t) or (m.distance < best_by_t[t].distance):
            best_by_t[t] = m
    good = list(best_by_t.values())
    if len(good) < 8:
        return 0, 0.0

    ptsA = np.float32([kpA[m.queryIdx].pt for m in good])
    ptsB = np.float32([kpB[m.trainIdx].pt for m in good])

    # fundamental with robust fallbacks
    try:
        nF, _ = _robust_fundamental(ptsA, ptsB, ransac_px=ransac_px)
    except Exception:
        nF = 0
    if nF >= 8:
        # geometric score in [0,1], cap at 80 inliers
        return nF, min(1.0, nF / 80.0)

    # homography fallback (planar scenes, walls, posters, etc.)
    try:
        nH, _ = _robust_homography(ptsA, ptsB, ransac_px=max(3.0, ransac_px))
    except Exception:
        nH = 0
    if nH >= 10:
        return nH, min(1.0, nH / 80.0)

    return 0, 0.0

def verify_pair(pA: str, pB: str, use_mast3r=True, **kw) -> Tuple[int, float]:
    if use_mast3r and (_MAST3R is not None):
        n_in, score = verify_pair_mast3r(pA, pB, _MAST3R, max_side=kw.get("max_side", 960))
        if n_in > 0:
            return n_in, min(1.0, score)
    return verify_pair_orb(pA, pB,
                           ratio=kw.get("ratio", 0.8),
                           ransac_px=kw.get("ransac_px", 2.5),
                           max_side=kw.get("max_side", 960))


def cluster_mast3r_style(
    df_vectors: pd.DataFrame,
    folder_id: str,
    image_root: str,
    # retrieval
    k_retrieve: int = 80,
    t_accept: float = 0.85,
    t_verify: float = 0.25,
    t_reject: float = 0.10,
    # geometry
    use_mast3r: bool = True,
    inlier_min: int = 12,
    max_side: int = 960,
    ratio: float = 0.80,
    ransac_px: float = 2.5,
    # graph + clustering
    alpha: float = 0.8,
    min_degree: int = 1,
    use_mutual: bool = True,
    per_node_cap: int = 8,
    # NEW: post rules
    rel_drop_ratio: float = 0.60,     # drop clusters ≤ 0.60 * largest
    pop_drop_ratio: float = 0.10,     # drop clusters ≤ 0.10 * population
    # NEW: outlier rescue
    rescue_outliers: bool = True,
    rescue_top_clusters: int = 3,     # search refs only in top-K clusters
    topk_refs: int = 10,              # per candidate outlier, per cluster
    rescue_inlier_avg: int = 20,      # avg inliers threshold
    rescue_success_frac: float = 0.30,# fraction of refs with >0 inliers
    # progress
    show_progress: bool = True
) -> pd.DataFrame:

    sub = df_vectors[df_vectors["folder"] == folder_id].reset_index(drop=True)
    n = len(sub)
    if n == 0:
        return pd.DataFrame(columns=["folder","file","cluster_label"])
    if n == 1:
        out = sub[["folder","file"]].copy(); out["cluster_label"] = 0; return out

#    if n <= 5 : 
#        return out

    # --- vectors (L2) ---
    X = np.stack(sub["vector"].to_list()).astype("float32")
    X /= np.maximum(1e-8, np.linalg.norm(X, axis=1, keepdims=True))
    k = max(1, min(k_retrieve, n-1))

    nbrs, sims_full = shortlist_indices(X, k)
    sims_full = np.clip(sims_full, -1.0, 1.0)
    cos01 = 0.5 * (sims_full + 1.0)  # map to [0,1]

    # paths
    paths = [resolve_path(image_root, folder_id, f) for f in sub["file"]]

    # ------- candidate pairs (only j>i; optional mutual) -------
    pairs = []
    if use_mutual:
        # build a quick set view for mutual test
        nbr_sets = [set(map(int, nbrs[i].tolist())) for i in range(n)]
    for i in range(n):
        for j_raw in nbrs[i]:
            j = int(j_raw)
            if j <= i: 
                continue
            if use_mutual and (i not in nbr_sets[j]): 
                continue
            pairs.append((i, j, float(cos01[i, j])))

    pbar = tqdm(total=len(pairs), desc="build+verify pairs", unit="pair",
                disable=not show_progress)

    # collect edges per node to cap degree later
    edges_by_node: Dict[int, List[Tuple[int,float]]] = {u: [] for u in range(n)}

    for i, j, c in pairs:
        if c >= t_accept:
            w = alpha*c + (1-alpha)*1.0
            edges_by_node[i].append((j, w)); edges_by_node[j].append((i, w))
            pbar.update(1); continue
        if c < t_reject:
            pbar.update(1); continue

        n_in, g = verify_pair(paths[i], paths[j], use_mast3r=use_mast3r,
                              max_side=max_side, ratio=ratio, ransac_px=ransac_px)
        if n_in >= inlier_min:
            w = alpha*c + (1-alpha)*g
            edges_by_node[i].append((j, w)); edges_by_node[j].append((i, w))
        pbar.update(1)

    pbar.close()

    # ------- per-node cap (keep strongest M) -------
    G = nx.Graph(); G.add_nodes_from(range(n))
    if per_node_cap and per_node_cap > 0:
        for u in range(n):
            if not edges_by_node[u]: continue
            edges_by_node[u].sort(key=lambda t: -t[1])
            edges_by_node[u] = edges_by_node[u][:per_node_cap]

    for u in range(n):
        for v, w in edges_by_node[u]:
            a, b = (u, v) if u < v else (v, u)
            if G.has_edge(a, b):
                if w > G[a][b]['weight']: G[a][b]['weight'] = float(w)
            else:
                G.add_edge(a, b, weight=float(max(1e-6, w)))

    # ------- connected components → labels -------
    if G.number_of_edges() == 0:
        labels = -1 * np.ones(n, np.int32)
    else:
        comps = list(nx.connected_components(G))
        labels = -1 * np.ones(n, np.int32)
        for cid, nodes in enumerate(comps):
            for u in nodes: labels[u] = cid

    # degree-based outliers
    deg = np.array([d for _, d in G.degree()])
    labels[deg < max(0, min_degree)] = -1

    # ------- POST-RULE: drop clusters by size (absolute or relative) -------
    valid = labels != -1
    if valid.any():
        # (a) absolute 10% of population
        abs_thresh = int(np.floor( pop_drop_ratio * n))  # <= 10% of total
        vals, counts = np.unique(labels[valid], return_counts=True)
        # largest cluster size
        Nmax = counts.max() if counts.size > 0 else 0
        # (b) relative ≤ 60% of the largest
        rel_thresh = int(np.floor(rel_drop_ratio * Nmax))

        to_drop = set()
        for v, c in zip(vals, counts):
            if c <= abs_thresh or c <= rel_thresh:
                to_drop.add(v)

        if to_drop:
            labels[np.isin(labels, list(to_drop))] = -1

    # ------- outlier rescue (fast, top clusters only) -------
    if rescue_outliers:
        labels = _rescue_outliers_topk(
            labels, X, cos01, paths,
        )

    # -------- Safe recursive version --------
    outliers_file = sub.loc[labels == -1, "file"]
    params_fixed = dict(
                    k_retrieve=50,
                    t_accept=0.85, t_verify=0.50, t_reject=0.10,
                    use_mast3r=True, inlier_min=20, max_side=960, ratio=0.80, ransac_px=2.5,
                    alpha=0.7, min_degree=2,
                    use_mutual=True, per_node_cap=8,
                    # post-rule: drop clusters ≤ 60% of largest
                    rel_drop_ratio=0.6,
                    pop_drop_ratio=0.1,
                    # one-pass outlier rescue against top-K largest clusters
                    rescue_outliers=True, rescue_top_clusters=5,
                    topk_refs=20, rescue_inlier_avg=20, rescue_success_frac=0.30,
                    show_progress=True
                )
    print("Size of outlier cluster: " , len(outliers_file) )
    print("Size of parent dataset: " , n )

    if len(outliers_file) == n:
        out = sub[["folder", "file"]].copy()
        out["cluster_label"] = -1
        print(f"[warn] All {n} images are outliers for folder '{folder_id}' — no clusters formed.")
        return out
    
    if len(outliers_file) >= 0.5 * n:
        print("Outlier group is larger than 50% of population --> Clustering this seperated cluster !!!")
        outliers_group = df_vectors[df_vectors["file"].isin(outliers_file)].reset_index(drop=True)
        
        df_cluster_out = cluster_mast3r_style(outliers_group, folder_id, image_root, **params_fixed)
        outliers_group = df_vectors[df_vectors["file"].isin(
            df_cluster_out.loc[df_cluster_out["cluster_label"] == -1, "file"]
        )].reset_index(drop=True)
    
        # Merge new clusters back to main df
        if (df_cluster_out["cluster_label"] != -1).any():
            offset = labels[labels != -1].max() + 1 if (labels != -1).any() else 0
            for _, row in df_cluster_out.iterrows():
                if row["cluster_label"] != -1:
                    labels[sub["file"] == row["file"]] = offset + row["cluster_label"]

        if rescue_outliers:
            remaining_outliers = (labels == -1).sum()
            if remaining_outliers > 0:
                labels = _rescue_outliers_topk(labels, X, cos01, paths,
                                               sim_high_thr = 0.8, sim_mid_low = 0.7, 
                                               inlier_cond = 20, success_cond = 0.50 )
            
    elif len(outliers_file) >= 0.3 * n:
        print("Outlier group is between 30% and 50% of population  --> Clustering this seperated cluster !!!")
        outliers_group = df_vectors[df_vectors["file"].isin(outliers_file)].reset_index(drop=True)
        
        df_cluster_out = cluster_mast3r_style(outliers_group, folder_id, image_root, **params_fixed)
        outliers_group = df_vectors[df_vectors["file"].isin(
            df_cluster_out.loc[df_cluster_out["cluster_label"] == -1, "file"]
        )].reset_index(drop=True)
    
        # Merge new clusters back to main df
        if (df_cluster_out["cluster_label"] != -1).any():
            offset = labels[labels != -1].max() + 1 if (labels != -1).any() else 0
            for _, row in df_cluster_out.iterrows():
                if row["cluster_label"] != -1:
                    labels[sub["file"] == row["file"]] = offset + row["cluster_label"]

        if rescue_outliers:
            remaining_outliers = (labels == -1).sum()
            if remaining_outliers > 0:
                labels = _rescue_outliers_topk(labels, X, cos01, paths,
                                               sim_high_thr = 0.8, sim_mid_low = 0.65, 
                                               inlier_cond = 9, success_cond = 0.50 )

    out = sub[["folder","file"]].copy()
    out["cluster_label"] = labels
    return out


def _rescue_outliers_topk(
    labels: np.ndarray,
    X: np.ndarray,
    cos01: np.ndarray,
    paths: List[Optional[str]],
    rescue_top_clusters: int = 3,
    topk_refs: int = 10,
    # legacy knobs (not used by the new rule but kept for signature compatibility)
    inlier_avg_thr: int = 20,
    success_frac_thr: float = 0.30,
    # geometry params
    ratio: float = 0.80,
    ransac_px: float = 2.5,
    max_side: int = 960,
    # --- NEW rule thresholds (tunable) ---
    sim_high_thr: float = 0.80,
    sim_mid_low: float = 0.70,
    inlier_cond: float = 9.8,
    success_cond: float = 0.50,
    verbose: bool = False,
) -> np.ndarray:
    """
    Relabel selected outliers into best cluster using your two-stage rule:
      1) avg_sim >= sim_high_thr -> choose highest avg_sim (tie by avg_inliers)
      2) else among mid band [sim_mid_low, sim_high_thr):
           require avg_inliers >= inlier_cond and success_frac >= success_cond
           choose highest avg_sim (tie by avg_inliers)
      3) else keep as -1
    """
    
    n = len(labels)
    out = labels.copy()

    # which nodes are in some cluster
    valid = out != -1
    if not valid.any():
        return out

    # top-K (largest) clusters
    vals, counts = np.unique(out[valid], return_counts=True)
    order = np.argsort(-counts)
    top_clusters = vals[order[:rescue_top_clusters]]
    if len(top_clusters) == 0:
        return out

    # members dict for quick retrieval (updated when we reassign)
    members = {c: np.where(out == c)[0] for c in top_clusters}

    # precompute once
    # X is L2-normalized; cos01 already provided
    for i in np.where(out == -1)[0]:
        # collect per-candidate stats
        cand_stats = []  # list of (cluster_id, avg_sim, avg_inliers, success_frac)

        for c in top_clusters:
            idxs = members.get(c, None)
            if idxs is None or idxs.size == 0:
                continue

            # top-k most similar refs from this cluster
            sims = cos01[i, idxs]
            take = min(topk_refs, idxs.size)
            order_refs = np.argsort(-sims)[:take]
            ref_idx = idxs[order_refs]
            ref_sims = sims[order_refs]

            if take == 0:
                continue

            # run geometry to compute inliers
            inliers = []
            p_i = paths[i]
            if p_i is None:
                continue
            for j in ref_idx:
                p_j = paths[j]
                if p_j is None:
                    inliers.append(0)
                    continue
                n_in, _ = verify_pair(
                    p_i, p_j,
                    use_mast3r=False,         # keep ORB+MAGSAC here for speed & stability
                    max_side=max_side,
                    ratio=ratio,
                    ransac_px=ransac_px
                )
                inliers.append(int(max(0, n_in)))

            if len(inliers) == 0:
                continue

            avg_sim     = float(np.mean(ref_sims))
            avg_inliers = float(np.mean(inliers))
            success_frac = float(np.mean([x > 0 for x in inliers]))

            cand_stats.append((c, avg_sim, avg_inliers, success_frac))

            if verbose:
                print(f"[rescue] outlier idx={i} vs cluster {c}: "
                      f"avg_sim={avg_sim:.3f}, avg_in={avg_inliers:.2f}, succ={success_frac:.2f}, "
                      f"size={len(idxs)}")

        if not cand_stats:
            continue

        # --- Apply your selection rule ---
        # 1) High similarity bucket
        high = [t for t in cand_stats if t[1] >= sim_high_thr]
        choice = None
        if high:
            # pick by highest avg_sim, tie by avg_inliers
            high.sort(key=lambda x: (x[1], x[2]), reverse=True)
            choice = high[0]
        else:
            # 2) Mid similarity with geometric evidence
            mid = [
                t for t in cand_stats
                if (sim_mid_low <= t[1] < sim_high_thr) and (t[2] >= inlier_cond) and (t[3] >= success_cond)
            ]
            if mid:
                mid.sort(key=lambda x: (x[1], x[2]), reverse=True)
                choice = mid[0]

        # 3) If a choice exists, reassign this outlier
        if choice is not None:
            chosen_cluster = int(choice[0])
            out[i] = chosen_cluster
            # update members so subsequent outliers can also consider this
            if chosen_cluster in members:
                members[chosen_cluster] = np.append(members[chosen_cluster], i)
            else:
                members[chosen_cluster] = np.array([i], dtype=int)

            if verbose:
                _, asim, ain, sfrac = choice
                print(f"[rescue] ✅ reassigned idx={i} to cluster {chosen_cluster} "
                      f"(avg_sim={asim:.3f}, avg_in={ain:.1f}, succ={sfrac:.2f})")

    return out


import os
import pandas as pd
import numpy as np
from typing import Dict, List

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff",
               ".PNG", ".JPG", ".JPEG"}

def build_base_from_test_root(test_root: str) -> pd.DataFrame:
    """Scan test_root/<dataset>/* and return ['dataset','image']."""
    rows = []
    for dataset in sorted(os.listdir(test_root)):
        dpath = os.path.join(test_root, dataset)
        if not os.path.isdir(dpath):
            continue
        for fn in sorted(os.listdir(dpath)):
            _, ext = os.path.splitext(fn)
            if ext in _IMAGE_EXTS:
                rows.append((dataset, fn))
    base = pd.DataFrame(rows, columns=["dataset","image"])
    if base.empty:
        raise RuntimeError(f"No images found under: {test_root}")
    return base


def build_submission(
    test_root: str,
    # Map: dataset(folder) -> dataframe with columns ['file','cluster_label']
    predictions_by_dataset: Dict[str, pd.DataFrame],
    output_csv: str
) -> pd.DataFrame:
    """
    - Scans test_root.
    - Left-joins per-dataset predictions by filename (handles missing extensions).
    - Preserves your labels exactly, including -1.
    - scene = f"cluster_{label}"
    - image_id = image + "_private"
    """
    base = build_base_from_test_root(test_root)  # ['dataset','image']

    all_rows: List[pd.DataFrame] = []
    for ds, sub_df in base.groupby("dataset"):
        sub = sub_df.copy()

        pred = predictions_by_dataset.get(ds, None)
        if pred is None or pred.empty:
            sub["cluster_label"] = -1
        else:
            p = pred.copy()
            if "file" not in p.columns or "cluster_label" not in p.columns:
                raise ValueError(f"Predictions for '{ds}' must have ['file','cluster_label']")

            # normalize names for merge
            p["file_noext"] = p["file"].apply(lambda s: os.path.splitext(str(s))[0])
            sub["image_noext"] = sub["image"].apply(lambda s: os.path.splitext(str(s))[0])

            sub = sub.merge(
                p[["file_noext", "cluster_label"]],
                left_on="image_noext", right_on="file_noext",
                how="left"
            )

            sub["cluster_label"] = sub["cluster_label"].fillna(-1).astype(int)
            sub.drop(columns=["image_noext","file_noext"], inplace=True)

        all_rows.append(sub)

    merged = pd.concat(all_rows, ignore_index=True)

    # Preserve labels exactly
    merged["scene"] = merged["cluster_label"].apply(lambda v: f"cluster_{int(v)}")

    # image_id per rule
    merged["image_id"] = merged["image"].astype(str) + "_private"

    # Final submission columns
    subm = merged[["image_id","dataset","scene","image"]].copy()

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    subm.to_csv(output_csv, index=False)
    return subm


resize_with_subfolders("/kaggle/input/image-matching-challenge-2025/test", "/kaggle/working/test_resized")
print("Resize Done")

# ---- Run it
build_vectors_csv( roots=("/kaggle/working/test_resized",),l2_normalize=True, out_csv="/kaggle/working/test_vectors.csv")
print("Converting Vector Done")

df_vectors = pd.read_csv("/kaggle/working/test_vectors.csv")   # folder, file, vector

def parse_vec(x):
    if isinstance(x, (list, np.ndarray)):
        return np.asarray(x, dtype=np.float32)
    try:
        v = np.asarray(ast.literal_eval(x), dtype=np.float32)
    except Exception:
        # fallback: remove brackets and split by space or comma
        x = x.strip("[] ")
        parts = [float(p) for p in x.replace(",", " ").split()]
        v = np.asarray(parts, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

df_vectors["vector"] = df_vectors["vector"].apply(parse_vec)

list_of_dataset = list( df_vectors['folder'].unique() )

for idx in range( len(list_of_dataset) ):
    if list_of_dataset[idx][-7:] == "resized" :
        list_of_dataset[idx] = list_of_dataset[idx][:-8]
    else: 
        list_of_dataset.pop(idx)
        idx -= 1

pred = {}

image_root = "/kaggle/working/test_resized"
params = dict(
    k_retrieve=50,
    t_accept=0.85, t_verify=0.50, t_reject=0.10,
    use_mast3r=True, inlier_min=20, max_side=960, ratio=0.80, ransac_px=2.5,
    alpha=0.7, min_degree=2,
    use_mutual=True, per_node_cap=8,
    # post-rule: drop clusters ≤ 60% of largest
    rel_drop_ratio=0.6,
    pop_drop_ratio=0.1,
    # one-pass outlier rescue against top-K largest clusters
    rescue_outliers=True, rescue_top_clusters=5,
    topk_refs=20, rescue_inlier_avg=20, rescue_success_frac=0.30,
    show_progress=True
)

for dataset in list_of_dataset:
    folder = dataset + "_resized"
    df_cluster_one = cluster_mast3r_style(df_vectors, folder, image_root, **params)
    pred[dataset] = df_cluster_one

print("Clustering Done")


# 2) Build submission
test_root = "/kaggle/input/image-matching-challenge-2025/test"   # top-level directory with subfolders per dataset
submission = build_submission(test_root, pred, "submission.csv")
print("Output Done")


pred


df = pd.read_csv("/kaggle/working/test_vectors.csv")
df.head(5)


test_root = "/kaggle/input/image-matching-challenge-2025/test"   # top-level directory with subfolders per dataset
submission = build_submission(test_root, pred, "submission.csv")
print("Output Done")


submission.head(25)


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    data = submission.sort_values("scene", ascending=True).reset_index(drop=True)
    print(data)


##!pip -q install --upgrade pip
##!pip -q install pycolmap


import pycolmap


df_result=find_poses_for_scene(submission)


import pandas as pd
import numpy as np
import os
import shutil
import types
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import subprocess
import sys

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
ROOT_IMAGE_DIR = "/kaggle/input/image-matching-challenge-2025/test"
WORKSPACE_DIR_BASE = "colmap_workspace"

# We'll CALL this string as a binary. We assume the runtime has 'colmap' on PATH.
COLMAP_EXE = "colmap"


# -------------------------------------------------
# pycolmap import (with safe fallback)
# -------------------------------------------------
try:
    import pycolmap
except ImportError:
    print("CRITICAL: pycolmap library not found. Using mock. Pose extraction will be mostly empty.")

    class MockReconstruction:
        def __init__(self, model_dir):
            pass
        def num_reg_images(self):
            return 0
        def reg_image_ids(self):
            return []
        @property
        def images(self):
            return {}

    pycolmap = types.SimpleNamespace(Reconstruction=MockReconstruction)


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def average_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """
    Average quaternions while handling antipodal symmetry.
    Returns unit quaternion [w, x, y, z].
    """
    if len(quaternions) == 0:
        # identity rotation
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # Flip signs to make them all "point the same way" before averaging
    first_q = quaternions[0]
    aligned = []
    for q in quaternions:
        if np.dot(first_q, q) < 0:
            aligned.append(-q)
        else:
            aligned.append(q)
    aligned = np.stack(aligned, axis=0)

    mean_q = np.mean(aligned, axis=0)
    norm = np.linalg.norm(mean_q) + 1e-12
    return mean_q / norm


def _colmap_available() -> bool:
    """
    Quick sanity check: can we even run the colmap binary in this environment?
    If not, we bail early and let caller fall back to random pose.
    """
    try:
        test = subprocess.run(
            [COLMAP_EXE, "--help"],
            capture_output=True,
            text=True
        )
        # returncode 0 or 1 is fine (colmap tends to exit 1 after printing help)
        return test.returncode in (0, 1)
    except FileNotFoundError:
        return False
    except PermissionError:
        return False


def run_colmap_pipeline(scene_image_dir: Path, workspace_path: Path):
    """
    Run COLMAP (feature_extractor -> exhaustive_matcher -> mapper)
    on scene_image_dir.
    Returns Path to sparse model dir (workspace_path/sparse/0) or None.
    """

    if not _colmap_available():
        print("   [COLMAP ERROR] 'colmap' binary not available in this environment. Skipping SfM.")
        return None

    db_path = workspace_path / "database.db"
    sparse_dir = workspace_path / "sparse"
    os.makedirs(sparse_dir, exist_ok=True)

    # 1. feature_extractor
    print(f"   [COLMAP] Running feature_extractor on {scene_image_dir}")
    cmd = [
        COLMAP_EXE, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(scene_image_dir),
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--ImageReader.single_camera_per_folder", "1",
        "--SiftExtraction.use_gpu", "true",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"   [COLMAP ERROR] feature_extractor failed:\n{e.stderr}")
        return None

    # 2. exhaustive_matcher
    print("   [COLMAP] Running exhaustive_matcher...")
    cmd = [
        COLMAP_EXE, "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", "true",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"   [COLMAP ERROR] exhaustive_matcher failed:\n{e.stderr}")
        return None

    # 3. mapper
    print("   [COLMAP] Running mapper...")
    cmd = [
        COLMAP_EXE, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(scene_image_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.init_min_num_inliers", "5",
        "--Mapper.abs_pose_min_num_inliers", "5",
        "--Mapper.filter_max_reproj_error", "10.0",
        "--Mapper.min_num_matches", "5",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("   [COLMAP ERROR] mapper failed. Scene probably has no consistent matches.")
        print(f"   [COLMAP STDERR]\n{e.stderr}")
        return None

    # COLMAP writes sparse models to sparse/0, sparse/1, ...
    model_dir = sparse_dir / "0"
    if not model_dir.exists():
        print(f"   [COLMAP] Mapper ran but produced no model at {model_dir}")
        return None

    print(f"   [COLMAP] Success! Model created at {model_dir}")
    return model_dir


# -------------------------------------------------
# MAIN FUNCTION (pose estimation per scene)
# -------------------------------------------------
def find_poses_for_scene(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (dataset, scene):
    - Copy that scene's images to a temp workspace
    - Run COLMAP to estimate poses
    - Extract rotation matrix (3x3) and translation (3,)
    - Write them back into df as strings "r11;r12;...;r33" and "tx;ty;tz"

    cluster_-1 scenes get NaN pose.

    If COLMAP fails totally, we generate a random fallback pose so that downstream
    code still has something to write.
    """

    df_filled = df.copy()

    # ensure pose columns exist and are dtype=object so we can assign strings later
    for col in ['rotation_matrix', 'translation_vector']:
        if col not in df_filled.columns:
            df_filled[col] = None
        df_filled[col] = df_filled[col].astype(object)

    # group by dataset + scene
    for (dataset, scene), scene_group in df_filled.groupby(['dataset', 'scene']):
        print("\n=======================================================")
        print(f"| PROCESSING SCENE: {scene.upper()} (Dataset: {dataset})")
        print("=======================================================")

        # special case: cluster_-1 is "junk / outlier"
        if scene == "cluster_-1":
            print("   [INFO] Scene is 'cluster_-1'. Skipping COLMAP and writing NaN values.")
            dummy_rot_str = ";".join(["NaN"] * 9)
            dummy_trans_str = ";".join(["NaN"] * 3)
            for index, _row in scene_group.iterrows():
                df_filled.loc[index, 'rotation_matrix'] = dummy_rot_str
                df_filled.loc[index, 'translation_vector'] = dummy_trans_str
            continue

        # source folder with the real test images
        dataset_image_dir = Path(ROOT_IMAGE_DIR) / dataset
        if not dataset_image_dir.exists():
            print(f"⚠️ Error: Dataset directory not found, skipping: {dataset_image_dir}")
            continue

        # make a fresh workspace for this dataset+scene
        workspace_path = Path(f"{WORKSPACE_DIR_BASE}_{dataset}_{scene}")
        temp_scene_image_dir = workspace_path / "temp_scene_images"

        if workspace_path.exists():
            shutil.rmtree(workspace_path)
        os.makedirs(temp_scene_image_dir, exist_ok=True)

        # copy only the images that belong to this scene
        print(f"   [INFO] Copying {len(scene_group)} images for scene '{scene}' to temporary folder...")
        images_copied = 0
        for index, row in scene_group.iterrows():
            image_name = row['image']  # should include extension like .png or .jpg
            src_path = dataset_image_dir / image_name
            dst_path = temp_scene_image_dir / image_name
            if src_path.exists():
                shutil.copy(str(src_path), str(dst_path))
                images_copied += 1
            else:
                print(f"   [WARN] Image not found: {src_path}")

        if images_copied == 0:
            print(f"   [ERROR] No images copied for scene '{scene}'. Skipping COLMAP.")
            shutil.rmtree(workspace_path, ignore_errors=True)
            continue

        # run COLMAP pipeline
        pose_results = {}      # image_name -> (Rij[3x3], t[3])
        successful_quats = []  # list of [w,x,y,z]
        successful_trans = []  # list of [tx,ty,tz]

        mean_R = None
        mean_t = None

        try:
            model_dir = run_colmap_pipeline(temp_scene_image_dir, workspace_path)

            if model_dir:
                # try reading poses with pycolmap
                try:
                    reconstruction = pycolmap.Reconstruction(str(model_dir))
                    if reconstruction.num_reg_images() > 0:
                        for image_id in reconstruction.reg_image_ids():
                            image = reconstruction.images[image_id]

                            # camera-from-world pose
                            pose = image.cam_from_world()

                            # pycolmap returns rotation as quaternion (w,x,y,z)
                            q = pose.rotation.quat
                            t = pose.translation  # np.array([tx,ty,tz])

                            successful_quats.append(q)
                            successful_trans.append(t)

                            # convert quaternion -> 3x3 rotation matrix (scipy expects [x,y,z,w])
                            q_scipy = [q[1], q[2], q[3], q[0]]
                            rot_matrix = R.from_quat(q_scipy).as_matrix()

                            pose_results[image.name] = (rot_matrix, t)

                        # compute mean pose across registered images
                        mean_q = average_quaternions(np.array(successful_quats))
                        mean_t = np.mean(successful_trans, axis=0)

                        mean_q_scipy = [mean_q[1], mean_q[2], mean_q[3], mean_q[0]]
                        mean_R = R.from_quat(mean_q_scipy).as_matrix()

                        print(f"   [PyColmap] Successfully read {len(pose_results)} poses.")
                        print(f"   [PyColmap] Mean t: {mean_t}")
                        print(f"   [PyColmap] Mean R:\n{mean_R}")
                    else:
                        print("   [PyColmap] COLMAP ran but registered 0 images.")
                except Exception as e:
                    print(f"❌ CRITICAL ERROR loading reconstruction with pycolmap: {e}")
            else:
                print("   [PyColmap] COLMAP pipeline produced no model. Using fallback.")

            # still nothing? -> fallback random pose to avoid NaNs
            if mean_R is None:
                print("   [INFO] No valid COLMAP pose. Generating random fallback pose.")
                mean_R = R.random().as_matrix()          # (3,3)
                mean_t = np.random.rand(3)               # (3,)
                print(f"   [INFO] Fallback t: {mean_t}")
                print(f"   [INFO] Fallback R:\n{mean_R}")

            # stringify mean pose for default fill
            mean_R_str = ";".join(map(str, mean_R.flatten()))
            mean_t_str = ";".join(map(str, mean_t.flatten()))

            # write per-row results into df_filled
            for index, row in scene_group.iterrows():
                image_name = row['image']
                if image_name in pose_results:
                    rot_matrix, t_vec = pose_results[image_name]
                    rot_str = ";".join(map(str, rot_matrix.flatten()))
                    trans_str = ";".join(map(str, t_vec.flatten()))
                    df_filled.loc[index, 'rotation_matrix'] = rot_str
                    df_filled.loc[index, 'translation_vector'] = trans_str
                else:
                    # use scene mean (which might be fallback random)
                    df_filled.loc[index, 'rotation_matrix'] = mean_R_str
                    df_filled.loc[index, 'translation_vector'] = mean_t_str

        except Exception as e:
            print(f"❌ UNHANDLED CRITICAL ERROR during processing scene '{scene}': {e}")

        finally:
            # clean workspace
            shutil.rmtree(workspace_path, ignore_errors=True)

    return df_filled



df_result=find_poses_for_scene(submission)


df_result.head(23)


sub = pd.read_csv("/kaggle/working/submission.csv")
print(sub.shape)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(sub)

