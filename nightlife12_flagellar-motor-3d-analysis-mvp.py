!pip install streamlit


!pip install pyngrok --quiet
!pip install reportlab --quiet



# ================================================================
# ğŸ§¬ Phase 1 â€“ Flagellar Motor 3D Analysis (Research-Ready Baseline)
# (Updated: B3-strict processed cache policy â€” rebuild if older than 24h)
# ================================================================

import os, gc, warnings, random, time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from PIL import Image

import torch
import torch.nn as nn

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from typing import Tuple

# ---------------------------------------------------------------
# âœ… 1. Deterministic Reproducibility
# ---------------------------------------------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # enables mixed-data speeds
    torch.backends.cudnn.benchmark = True

set_seed(42)

# ---------------------------------------------------------------
# âœ… 2. Optional Import: skimage (surface reconstruction)
# ---------------------------------------------------------------
try:
    from skimage import measure
except Exception:
    measure = None
    warnings.warn("skimage.measure unavailable â€” 3D marching_cubes disabled.", UserWarning)

# ---------------------------------------------------------------
# âœ… 3. Global Configuration
# ---------------------------------------------------------------
DATA_ROOT = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"
TRAIN_FOLDER = "train"
GT_CSV = "train_labels.csv"
CACHE_DIR = Path("/kaggle/working/cache_phase1")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Compute / Visualization Control ---
MAX_POINTS, MAX_VIS_POINTS = 15000, 100000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Training Defaults ---
PATCH_SIZE = 64
PATCH_STRIDE = 32
TRAIN_PATCHES_PER_EPOCH = 64
BATCH_SIZE = 1
EPOCHS = 5
LEARNING_RATE = 1e-3
SIGMA_HEATMAP = 1.2
THRESH_STD = 0.15
DBSCAN_EPS = 2.0

# processed cache freshness policy (B3-strict): rebuild if older than 24 hours
PROCESSED_CACHE_MAX_AGE_HOURS = 24

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# ---------------------------------------------------------------
# âœ… 4. Streamlit Environment Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Flagellar Motor 3D Analysis", layout="wide")
st.title("ğŸ§¬ Flagellar Motor 3D Analysis (Baseline)")

# ---------------------------------------------------------------
# âœ… 5. Core I/O Utilities (robust + supports .npy tomograms)
# ---------------------------------------------------------------
def find_tomograms(root: str | Path):
    """
    Return tomogram folders or volume files under dataset path.
    Supports both subfolders (with .jpg/.png) and direct .npy volumes.
    """
    root = Path(root)
    p = root / TRAIN_FOLDER if (root / TRAIN_FOLDER).exists() else root
    if not p.exists():
        st.warning(f"âš ï¸� Dataset path not found: {p}")
        return []
    tomos = []
    for d in p.iterdir():
        if d.is_dir():
            imgs = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            if len(imgs) > 0:
                tomos.append(d)
        elif d.suffix == ".npy":
            tomos.append(d)
    if not tomos:
        st.warning(f"âš ï¸� No tomograms found under {p}")
    return tomos


def save_memmap_and_return(path: Path, arr: np.ndarray) -> np.memmap:
    """Save NumPy array and reopen as memmap for large-volume streaming."""
    np.save(path, arr)
    return np.load(path, mmap_mode="r")


@st.cache_data(show_spinner=False)
def load_volume_from_jpegs_cached(tomo_path: str | Path):
    """Load stack of 2D slices or .npy volume into a raw memmap and cache raw file.
       Returns: np.memmap of raw volume (unprocessed).
    """
    tomo_p = Path(tomo_path)
    cache_file = CACHE_DIR / f"{tomo_p.stem}_raw.npy"

    # âœ… Case 1: cached raw memmap
    if cache_file.exists():
        try:
            return np.load(cache_file, mmap_mode="r")
        except Exception:
            # corrupted raw cache â€” remove and rebuild
            try:
                cache_file.unlink(missing_ok=True)
            except Exception:
                pass

    # âœ… Case 2: directory of slices
    if tomo_p.is_dir():
        slices = sorted(tomo_p.glob("*.jpg")) + sorted(tomo_p.glob("*.png"))
        if not slices:
            st.error(f"â�Œ No image slices found in {tomo_p}")
            return None
        vol = np.stack([np.array(Image.open(s)) for s in slices], axis=0).astype(np.float32)
        return save_memmap_and_return(cache_file, vol)

    # âœ… Case 3: direct .npy file
    if tomo_p.is_file() and tomo_p.suffix == ".npy":
        try:
            return np.load(tomo_p, mmap_mode="r")
        except Exception:
            st.warning(f"Corrupted .npy at {tomo_p} â€” cannot load as raw memmap.")
            return None

    st.error(f"â�Œ Unsupported tomogram path: {tomo_p}")
    return None


def _is_file_older_than(path: Path, hours: float) -> bool:
    """Return True if file is older than given hours."""
    try:
        mtime = path.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600.0
        return age_hours > float(hours)
    except Exception:
        return True


# ---------------------------------------------------------------
# âœ… 8. Volume Preprocessing + Caching (used in later stages)
# (modified: will not overwrite valid cache unless forced; returns memmap)
# ---------------------------------------------------------------
def preprocess_volume_cached_memmap(
    vol_arr_or_memmap,
    cache_dir: Path = CACHE_DIR,
    name_prefix: str = None,
    normalize: bool = True,
    clip_sigma: float = 3.0,
    force_rebuild: bool = False
):
    """
    Normalize + denoise a 3-D tomogram and cache as memmap (.npy).
    Returns: np.memmap

    - If name_prefix not provided, attempt to infer from input memmap/file.
    - If the processed cache exists and is younger than PROCESSED_CACHE_MAX_AGE_HOURS
      and force_rebuild is False, the existing cache will be returned (fast).
    - If cache is corrupted (EOFError / invalid shape) it will be rebuilt.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- Infer name_prefix from path-like input if not given
    if name_prefix is None:
        try:
            if isinstance(vol_arr_or_memmap, (str, Path)):
                name_prefix = Path(vol_arr_or_memmap).stem
            else:
                # If memmap-like object returned by np.load has .filename attribute
                fname = getattr(vol_arr_or_memmap, "filename", None)
                if fname:
                    name_prefix = Path(fname).stem.replace("_raw", "").replace("_proc", "")
                else:
                    name_prefix = "tomo"
        except Exception:
            name_prefix = "tomo"

    out_path = Path(cache_dir) / f"{name_prefix}_proc.npy"

    # --- If cache exists and is recent and not forced, try to return it (B3-strict: 24h)
    if out_path.exists() and not force_rebuild:
        try:
            if not _is_file_older_than(out_path, PROCESSED_CACHE_MAX_AGE_HOURS):
                mm = np.load(out_path, mmap_mode="r")
                # sanity check shape & size
                if getattr(mm, "shape", None) and np.prod(mm.shape) > 0:
                    return mm
                else:
                    # corrupted, remove and rebuild
                    out_path.unlink(missing_ok=True)
            else:
                # older than allowed -> rebuild
                out_path.unlink(missing_ok=True)
        except EOFError:
            # corrupted file
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
        except Exception:
            # in any other error, attempt to remove and rebuild
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass

    # --- Load from path if needed
    if isinstance(vol_arr_or_memmap, (str, Path)):
        try:
            vol = np.load(vol_arr_or_memmap, mmap_mode="r")
            vol = np.array(vol, dtype=np.float32)
        except Exception:
            raise
    else:
        vol = np.array(vol_arr_or_memmap, dtype=np.float32)

    # --- Sigma clipping for Cryo-ET stabilization ---
    if clip_sigma and clip_sigma > 0:
        mu, sig = np.mean(vol), np.std(vol)
        vol = np.clip(vol, mu - clip_sigma * sig, mu + clip_sigma * sig)

    # --- Light Gaussian smoothing ---
    vol = ndimage.gaussian_filter(vol, sigma=0.8)

    # --- Normalization to [0,1] ---
    if normalize:
        vmin, vmax = np.percentile(vol, (1, 99))
        vol = np.clip((vol - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # --- Save to cache using inferred name_prefix ---
    np.save(out_path, vol.astype(np.float32))
    return np.load(out_path, mmap_mode="r")


def load_processed_memmap(raw_or_path: str | Path, name_prefix: str = None, max_age_hours: float = PROCESSED_CACHE_MAX_AGE_HOURS):
    """
    Convenience loader that enforces B3-strict policy for processed cache:
      - If processed cache exists and is younger than max_age_hours -> return it.
      - If processed cache is older, corrupted, or missing -> (re)build via preprocess_volume_cached_memmap().
    Accepts either:
      - raw_or_path = raw memmap path (e.g., '/kaggle/.../tomo_raw.npy') OR
      - raw_or_path = directory of slices OR
      - raw_or_path = in-memory array (np.ndarray)
    Returns: np.memmap (processed)
    """
    # infer name prefix
    if name_prefix is None:
        if isinstance(raw_or_path, (str, Path)):
            name_prefix = Path(raw_or_path).stem
        else:
            # try nanme from memmap .filename
            name_prefix = getattr(raw_or_path, "filename", None)
            if name_prefix:
                name_prefix = Path(name_prefix).stem.replace("_raw", "").replace("_proc", "")
            else:
                name_prefix = "tomo"

    proc_path = CACHE_DIR / f"{name_prefix}_proc.npy"

    # quick try to load if fresh
    if proc_path.exists():
        try:
            if not _is_file_older_than(proc_path, max_age_hours):
                mm = np.load(proc_path, mmap_mode="r")
                if getattr(mm, "shape", None) and np.prod(mm.shape) > 0:
                    return mm
                else:
                    proc_path.unlink(missing_ok=True)
            else:
                # older than allowed: remove and rebuild
                proc_path.unlink(missing_ok=True)
        except EOFError:
            proc_path.unlink(missing_ok=True)
        except Exception:
            proc_path.unlink(missing_ok=True)

    # if we reach here, build processed cache
    return preprocess_volume_cached_memmap(raw_or_path, cache_dir=CACHE_DIR, name_prefix=name_prefix, normalize=True, clip_sigma=3.0, force_rebuild=True)


# ---------------------------------------------------------------
# âœ… 6. Lightweight Geometry Utilities
# ---------------------------------------------------------------
def fallback_pointcloud(volume_memmap, downsample=4, threshold=None) -> np.ndarray:
    """Quick voxel-threshold to approximate 3D coordinates (fallback)."""
    if volume_memmap is None:
        return np.zeros((0, 3))
    mm = np.load(volume_memmap, mmap_mode="r") if isinstance(volume_memmap, (str, Path)) else volume_memmap
    if threshold is None:
        sample = []
        step = max(1, mm.shape[0] // 10)
        for zi in range(0, mm.shape[0], step):
            sample.append(mm[zi])
        sample = np.concatenate([s.ravel() for s in sample])
        threshold = sample.mean() + 0.5 * sample.std()
        del sample
    coords_list = []
    for zi in range(mm.shape[0]):
        sl = mm[zi]
        ys, xs = np.nonzero(sl > threshold)
        if ys.size > 0:
            zcol = np.full_like(ys, zi, dtype=float)
            coords_list.append(np.stack([zcol, ys.astype(float), xs.astype(float)], axis=1))
    if not coords_list:
        return np.zeros((0, 3))
    coords = np.vstack(coords_list)
    if coords.shape[0] > MAX_POINTS:
        idx = np.random.choice(coords.shape[0], size=MAX_POINTS, replace=False)
        coords = coords[idx]
    return coords[:, [2, 1, 0]].astype(float)

@st.cache_data(show_spinner=False)
def marching_cubes_mesh_adaptive(volume_memmap, level: float = None, target_max_vertices: int = MAX_POINTS):
    """Adaptive marching_cubes mesh with safe downsampling fallback."""
    if measure is None:
        return None, None
    mm = np.load(volume_memmap, mmap_mode="r") if isinstance(volume_memmap, (str, Path)) else volume_memmap
    if mm is None or mm.size == 0:
        return None, None
    vol = np.array(mm)
    if level is None:
        level = vol.mean() + 0.5 * vol.std()
    try:
        verts, faces, _, _ = measure.marching_cubes(vol, level=level)
    except Exception as e:
        warnings.warn(f"marching_cubes failed: {e}")
        return None, None
    if verts.shape[0] > target_max_vertices:
        factor = int(np.ceil((verts.shape[0] / target_max_vertices) ** (1/3)))
        if factor > 1:
            vol_ds = vol[::factor, ::factor, ::factor]
            try:
                verts, faces, _, _ = measure.marching_cubes(vol_ds, level=level)
                verts *= factor
            except Exception as e:
                warnings.warn(f"Downsampled marching_cubes failed: {e}")
                idx = np.random.choice(verts.shape[0], size=target_max_vertices, replace=False)
                verts = verts[idx]
    return verts[:, [2, 1, 0]], faces

def cluster_points(coords: np.ndarray, eps=DBSCAN_EPS, min_samples: int = 1):
    """DBSCAN clustering (GPU-safe). Returns cluster centroids (x,y,z)."""
    if coords is None or coords.shape[0] == 0:
        return np.zeros((0, 3))
    try:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = clustering.labels_
        keep_idx = labels >= 0
        if not np.any(keep_idx):
            return np.zeros((0, 3))
        coords_valid = coords[keep_idx]
        labels = labels[keep_idx]
        # compute centroids per cluster
        centroids = []
        for lab in np.unique(labels):
            members = coords_valid[labels == lab]
            centroids.append(members.mean(axis=0))
        return np.array(centroids, dtype=float)
    except Exception:
        # fallback: try returning unique coords
        try:
            uniq = np.unique(coords, axis=0)
            return uniq
        except Exception:
            return coords


def match_predictions_to_gt(pred_pts, gt_pts, radius=5.0):
    """Simple nearest-neighbor matching (used by Part 2 metrics)."""
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 0, len(gt_pts), np.array([])
    nbrs = NearestNeighbors(n_neighbors=1).fit(pred_pts)
    dists, _ = nbrs.kneighbors(gt_pts)
    matched = dists[:, 0] <= radius
    return int(matched.sum()), int(len(gt_pts)), dists[:, 0]

# ---------------------------------------------------------------
# âœ… 6b. Physical-Unit Coordinate Normalization (nm-scale)
# ---------------------------------------------------------------
from typing import Tuple

def get_default_voxel_size(tomo_name: str | Path) -> Tuple[float, float, float]:
    """
    Returns voxel size (z_nm, y_nm, x_nm) for a tomogram.
    Replace with metadata lookup if available.
    Example defaults below assume anisotropy typical in cryo-ET (zâ‰ˆ4â€“5Ã— coarser).
    """
    name = str(tomo_name).lower()
    # âš™ï¸� Customize per dataset / MotorBench calibration
    if "byu" in name or "motor" in name:
        return (5.0, 1.0, 1.0)
    return (1.0, 1.0, 1.0)

def vox_to_nm(coords: np.ndarray, voxel_size_nm: Tuple[float, float, float]):
    """
    Convert voxel indices (x,y,z) â†’ nanometers.
    coords: Nx3 (x,y,z)
    voxel_size_nm: (z_nm, y_nm, x_nm)
    """
    if coords is None or coords.size == 0:
        return coords
    z_nm, y_nm, x_nm = voxel_size_nm
    scale = np.array([x_nm, y_nm, z_nm], dtype=float)
    return coords * scale

def nm_to_vox(coords_nm: np.ndarray, voxel_size_nm: Tuple[float, float, float]):
    """Convert nanometer coordinates back to voxel units."""
    if coords_nm is None or coords_nm.size == 0:
        return coords_nm
    z_nm, y_nm, x_nm = voxel_size_nm
    inv = np.array([1/x_nm, 1/y_nm, 1/z_nm], dtype=float)
    return coords_nm * inv

def compute_metrics_nm(pred_pts: np.ndarray, gt_pts: np.ndarray,
                       voxel_size_nm: Tuple[float, float, float], radius_nm: float = 10.0):
    """
    Compute metrics (precision, recall, F2, mean_dist_nm) in physical nanometer units.
    """
    if len(gt_pts) == 0:
        return dict(precision=0, recall=0, f2=0, mean_dist_nm=0, matched=0)
    if len(pred_pts) == 0:
        return dict(precision=0, recall=0, f2=0, mean_dist_nm=np.nan, matched=0)

    from sklearn.neighbors import NearestNeighbors
    pred_nm = vox_to_nm(pred_pts, voxel_size_nm)
    gt_nm   = vox_to_nm(gt_pts, voxel_size_nm)
    nbrs = NearestNeighbors(n_neighbors=1).fit(pred_nm)
    dists, _ = nbrs.kneighbors(gt_nm)
    matched = dists[:, 0] <= radius_nm

    tp = matched.sum()
    fp = len(pred_nm) - tp
    fn = len(gt_nm) - tp
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f2        = (1 + 2**2) * (precision * recall) / (4 * precision + recall + 1e-8)

    return dict(
        precision=float(precision),
        recall=float(recall),
        f2=float(f2),
        mean_dist_nm=float(dists.mean()),
        matched=int(tp)
    )

# ---------------------------------------------------------------
# âœ… 7. Visualization (shared by later parts)
# ---------------------------------------------------------------
def safe_sample_points(arr: np.ndarray, max_points: int = MAX_VIS_POINTS):
    if arr is None or arr.size == 0:
        return np.zeros((0, 3))
    if arr.shape[0] > max_points:
        idx = np.random.choice(arr.shape[0], size=max_points, replace=False)
        return arr[idx]
    return arr

def plot_volume_mesh_and_points(mesh_verts, mesh_faces,
                                points=None, gt_pts=None, pred_pts=None,
                                title="3D Mesh + Points"):
    """3D plot combining mesh and predicted points."""
    points_vis = safe_sample_points(points)
    gt_vis = safe_sample_points(gt_pts)
    mesh_vis = safe_sample_points(mesh_verts, max_points=MAX_POINTS) if mesh_verts is not None else None
    fig = go.Figure()
    if mesh_vis is not None and mesh_faces is not None and len(mesh_faces) > 0:
        i, j, k = mesh_faces.T
        fig.add_trace(go.Mesh3d(x=mesh_vis[:, 0], y=mesh_vis[:, 1], z=mesh_vis[:, 2],
                                i=i, j=j, k=k, opacity=0.2, color="lightblue", name="volume mesh"))
    for pts, name, color, size in [(points_vis, "volume points", "blue", 1),
                                   (gt_vis, "ground-truth", "green", 4),
                                   (pred_pts, "predicted", "red", 4)]:
        if pts is not None and getattr(pts, "shape", (0,))[0] > 0:
            fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                                       mode="markers", marker=dict(size=size, color=color), name=name))
    fig.update_layout(scene=dict(aspectmode="data"), title=title, width=900, height=700)
    return fig

def plot_slice_with_points_memmap(memmap_path: Path, z_index: int,
                                  points=None, gt_pts=None, pred_pts=None,
                                  title="Slice View"):
    """Interactive 2D slice viewer with overlayed points."""
    mm = np.load(memmap_path, mmap_mode="r")
    z_index = int(max(0, min(z_index, mm.shape[0] - 1)))
    slice_img = np.array(mm[z_index, :, :])

    def pts_on_slice(pts):
        if pts is None or pts.size == 0:
            return np.zeros((0, 3))
        z_inds = np.round(pts[:, 2]).astype(int)
        return pts[z_inds == z_index]

    pts = pts_on_slice(points)
    gt = pts_on_slice(gt_pts)
    pred = pts_on_slice(pred_pts)

    fig = px.imshow(slice_img, color_continuous_scale="gray", origin="lower")
    if pts.shape[0] > 0:
        fig.add_scatter(x=pts[:, 0], y=pts[:, 1], mode="markers",
                        marker=dict(size=3, color="blue"), name="volume")
    if gt.shape[0] > 0:
        fig.add_scatter(x=gt[:, 0], y=gt[:, 1], mode="markers",
                        marker=dict(size=5, color="green"), name="GT")
    if pred.shape[0] > 0:
        fig.add_scatter(x=pred[:, 0], y=pred[:, 1], mode="markers",
                        marker=dict(size=5, color="red"), name="Pred")
    fig.update_layout(title=f"{title} â€“ Z={z_index}", width=700, height=700)
    return fig


# ================================================================
# Part 2 â€“ Tiny3DCNN + Confidence Filtering & Extended Training (UPDATED with S3-B)
# ================================================================

import os, gc, time, random, math, json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from scipy import ndimage

# ---------------- Constants (kept small/sane) ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 64
PATCH_STRIDE = 32
TRAIN_PATCHES_PER_EPOCH = 64
EPOCHS = 10   # extended training for smoother heatmaps
LEARNING_RATE = 1e-3
SIGMA_HEATMAP = 1.2
THRESH_STD = 0.20   # improved recall
DBSCAN_EPS = 2.0
MAX_POINTS = 15000
CACHE_DIR = Path("/kaggle/working/cache_phase1")
LOG_CSV = CACHE_DIR / "benchmark_runs.csv"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Preprocess metadata version (bump when preprocessing code changes)
PREPROCESS_VERSION = "v2"

# Stale threshold for processed cache (B2: strict) in seconds (6 hours)
PROC_CACHE_STALE_S = 6 * 3600

# ---------------- Seeding ----------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# ---------------- Model definitions ----------------
class Tiny3DCNN(nn.Module):
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, 3, padding=1), nn.ReLU(),
            nn.Conv3d(base_ch, base_ch*2, 3, padding=1), nn.ReLU(),
            nn.Conv3d(base_ch*2, base_ch, 3, padding=1), nn.ReLU(),
            nn.Conv3d(base_ch, 1, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# ---------------- Utility: safe memmap loader ----------------
def _safe_load_memmap(path: Path):
    """Try to np.load memmap and run quick sanity checks; return None if bad."""
    try:
        mm = np.load(path, mmap_mode="r")
        shp = getattr(mm, "shape", None)
        if shp is None or np.prod(shp) == 0:
            return None
        return mm
    except Exception:
        return None

# ---------------- Preprocess-meta helpers ----------------
def _write_proc_meta(proc_path: Path, vol_shape):
    meta = {
        "preprocess_version": PREPROCESS_VERSION,
        "shape": list(map(int, vol_shape)),
        "timestamp": int(time.time())
    }
    try:
        with open(str(proc_path) + ".meta.json", "w") as f:
            json.dump(meta, f)
    except Exception:
        pass

def _read_proc_meta(proc_path: Path):
    meta_path = Path(str(proc_path) + ".meta.json")
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except Exception:
        return None

# ---------------- Fallback cluster_points if not defined ----------------
if "cluster_points" not in globals():
    def cluster_points(coords: np.ndarray, eps=DBSCAN_EPS, min_samples: int = 1):
        """Lightweight DBSCAN fallback returning cluster centroids."""
        if coords is None or coords.shape[0] == 0:
            return np.zeros((0, 3))
        try:
            from sklearn.cluster import DBSCAN as _DB
            clustering = _DB(eps=eps, min_samples=min_samples).fit(coords)
            labels = clustering.labels_
            keep_idx = labels >= 0
            if not np.any(keep_idx):
                return np.zeros((0, 3))
            coords_valid = coords[keep_idx]
            labels = labels[keep_idx]
            centroids = []
            for lab in np.unique(labels):
                members = coords_valid[labels == lab]
                centroids.append(members.mean(axis=0))
            return np.array(centroids, dtype=float)
        except Exception:
            try:
                return np.unique(coords, axis=0)
            except Exception:
                return coords

# ---------------- Metric helpers ----------------
def compute_metrics(pred_pts, gt_pts, radius=5.0):
    """Legacy voxel-space metrics (kept for compatibility)."""
    if len(gt_pts) == 0:
        return dict(precision=0.0, recall=0.0, f2=0.0, mean_dist=0.0, matched=0)
    if len(pred_pts) == 0:
        return dict(precision=0.0, recall=0.0, f2=0.0, mean_dist=np.nan, matched=0)
    nbrs = NearestNeighbors(n_neighbors=1).fit(pred_pts)
    dists, _ = nbrs.kneighbors(gt_pts)
    matched = dists[:, 0] <= radius
    tp = int(matched.sum())
    fp = len(pred_pts) - tp
    fn = len(gt_pts) - tp
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f2        = (1 + 2**2) * (precision * recall) / (4 * precision + recall + 1e-8)
    return dict(precision=float(precision), recall=float(recall),
                f2=float(f2), mean_dist=float(dists.mean()), matched=int(tp))

# ---------------- Patches extraction ----------------
def extract_random_patches(volume_arr, target, n_patches, patch_size):
    Z, Y, X = volume_arr.shape
    patches_v, patches_t = [], []
    for _ in range(n_patches):
        zi = np.random.randint(0, max(1, Z - patch_size + 1))
        yi = np.random.randint(0, max(1, Y - patch_size + 1))
        xi = np.random.randint(0, max(1, X - patch_size + 1))
        patches_v.append(volume_arr[zi:zi+patch_size, yi:yi+patch_size, xi:xi+patch_size])
        patches_t.append(target[zi:zi+patch_size, yi:yi+patch_size, xi:xi+patch_size])
    return np.stack(patches_v), np.stack(patches_t)

# ---------------- Sliding-window inference (returns recon, conf_map) ----------------
def sliding_window_inference(volume_arr, model, patch_size=PATCH_SIZE, stride=PATCH_STRIDE, min_confidence=0.0):
    model.eval()
    Z, Y, X = volume_arr.shape
    accum = np.zeros((Z, Y, X), np.float32)
    norm = np.zeros((Z, Y, X), np.float32)
    conf_map = np.zeros((Z, Y, X), np.float32)

    use_amp = (DEVICE == "cuda")
    from torch.cuda.amp import autocast
    with torch.no_grad():
        for zi in range(0, max(1, Z - patch_size + 1), stride):
            for yi in range(0, max(1, Y - patch_size + 1), stride):
                for xi in range(0, max(1, X - patch_size + 1), stride):
                    patch = volume_arr[zi:zi+patch_size, yi:yi+patch_size, xi:xi+patch_size].astype(np.float32)
                    if patch.shape != (patch_size, patch_size, patch_size):
                        # pad if near edges
                        pad_z = patch_size - patch.shape[0]
                        pad_y = patch_size - patch.shape[1]
                        pad_x = patch_size - patch.shape[2]
                        pad = ((0,pad_z),(0,pad_y),(0,pad_x))
                        patch = np.pad(patch, pad, mode="constant", constant_values=0)
                    inp = torch.from_numpy(patch[None, None, ...]).to(DEVICE)
                    with autocast(enabled=use_amp):
                        out = model(inp)
                    out_np = out[0,0].detach().cpu().numpy()
                    conf = float(out_np.mean())
                    if conf < min_confidence:
                        del inp, out, out_np
                        torch.cuda.empty_cache()
                        continue
                    accum[zi:zi+patch_size, yi:yi+patch_size, xi:xi+patch_size] += out_np[:patch.shape[0], :patch.shape[1], :patch.shape[2]]
                    norm[zi:zi+patch_size, yi:yi+patch_size, xi:xi+patch_size] += 1.0
                    conf_map[zi:zi+patch_size, yi:yi+patch_size, xi:xi+patch_size] += conf
                    del inp, out, out_np
                    torch.cuda.empty_cache()
    norm[norm == 0] = 1.0
    conf_map = conf_map / norm
    gc.collect(); torch.cuda.empty_cache()
    return accum / norm, conf_map

# ---------------- GT / tomo mapping debug helper ----------------
def _debug_check_gt_for_tomo(tomo_name, gt_df):
    """Return whether tomo id present and sample rows for quick debug."""
    if gt_df is None or 'tomo_id' not in gt_df.columns:
        return False, []
    present = (gt_df['tomo_id'].astype(str) == str(tomo_name)).any()
    sample = gt_df[gt_df['tomo_id'].astype(str) == str(tomo_name)].head().to_dict(orient='records')
    return present, sample

# ---------------- Streamlit: Data & Cache UI (S3-B) ----------------
def _cache_staleness_ui_for_path(raw_path: Path):
    """
    If a processed cache exists for this raw path, evaluate staleness and present
    a non-blocking sidebar selectbox to choose behavior.
    Returns: cache_behavior in {"use_existing","rebuild_now","inference_only"}
    """
    proc_path = Path(CACHE_DIR) / f"{raw_path.stem}_proc.npy"
    if not proc_path.exists():
        return "rebuild_now"

    # read meta if available
    meta = _read_proc_meta(proc_path)
    try:
        mtime = proc_path.stat().st_mtime
        age_s = time.time() - mtime
    except Exception:
        age_s = float("inf")

    stale_reasons = []
    if age_s > PROC_CACHE_STALE_S:
        stale_reasons.append(f"age({age_s/3600:.1f}h)>6h")
    if meta is None:
        stale_reasons.append("no_meta")
    else:
        if meta.get("preprocess_version") != PREPROCESS_VERSION:
            stale_reasons.append(f"proc_ver({meta.get('preprocess_version')})!={PREPROCESS_VERSION}")
        # we can't easily compare shape here to raw without loading raw; that's handled in runtime
    if stale_reasons:
        st.sidebar.warning(f"âš ï¸� Processed cache seems stale: {', '.join(stale_reasons)}")
        choice = st.sidebar.selectbox(
            "Processed cache action (stale detected)",
            options=["Use existing processed cache", "Rebuild processed cache now", "Inference only (skip training)"],
            index=1
        )
        mapping = {
            "Use existing processed cache": "use_existing",
            "Rebuild processed cache now": "rebuild_now",
            "Inference only (skip training)": "inference_only"
        }
        return mapping.get(choice, "use_existing")
    else:
        # fresh cache, give lightweight option
        choice = st.sidebar.selectbox(
            "Processed cache available",
            options=["Use existing processed cache", "Rebuild processed cache now", "Inference only (skip training)"],
            index=0
        )
        mapping = {
            "Use existing processed cache": "use_existing",
            "Rebuild processed cache now": "rebuild_now",
            "Inference only (skip training)": "inference_only"
        }
        return mapping.get(choice, "use_existing")

# ---------------- Main train+infer function (run_tiny_cnn with S3-B) ----------------
def run_tiny_cnn(volume_memmap_or_arr,
                 gt_points,
                 epochs=EPOCHS,
                 preset="balanced",
                 stride_factor=1.0,
                 model_choice="tinycnn",
                 base_ch=16,
                 cache_behavior: str = None):
    """
    Train a selected 3D CNN (Tiny / UNet3D_small / ResNet3D_small if available), perform inference,
    apply confidence filtering, compute nm-aware metrics, save artifacts and log results.
    cache_behavior: "use_existing", "rebuild_now", or "inference_only" (auto-detected via sidebar if None)
    Returns: clustered_points (Nx3 float), metrics (dict)
    """

    t0_total = time.time()
    gpu_start_mem = torch.cuda.memory_allocated() if DEVICE == "cuda" else 0

    # ---- Determine whether input is path (so we can inspect processed cache) ----
    input_path = None
    if isinstance(volume_memmap_or_arr, (str, Path)):
        input_path = Path(volume_memmap_or_arr)

    # ---- If a path, decide cache behavior (sidebar prompt when stale) ----
    if input_path is not None and cache_behavior is None:
        cache_behavior = _cache_staleness_ui_for_path(input_path)

    # ---- Load raw/cached volume ----
    vol = None
    proc_cache_path = None
    if input_path is not None:
        raw_memmap = None
        try:
            # attempt to load raw memmap (unprocessed) first if exists
            raw_mempath_candidate = input_path
            if raw_mempath_candidate.exists():
                raw_memmap = _safe_load_memmap(raw_mempath_candidate) or None
        except Exception:
            raw_memmap = None

        proc_cache_path = Path(CACHE_DIR) / f"{input_path.stem}_proc.npy"
        # Behavior routing
        if cache_behavior == "use_existing" and proc_cache_path.exists():
            mm_try = _safe_load_memmap(proc_cache_path)
            # additional shape check vs raw (if we have raw loaded)
            meta = _read_proc_meta(proc_cache_path)
            shape_ok = True
            if mm_try is None:
                shape_ok = False
            else:
                if raw_memmap is not None and getattr(raw_memmap, "shape", None) != getattr(mm_try, "shape", None):
                    shape_ok = False
            if not shape_ok:
                st.warning("Existing processed cache appears invalid or mismatched â€” rebuilding instead.")
                cache_behavior = "rebuild_now"
            else:
                vol = np.array(mm_try)
        if cache_behavior == "rebuild_now":
            # call the official preprocessing function (assumes available in runtime)
            st.info("Rebuilding processed cache now (this will overwrite existing processed cache).")
            try:
                # call preprocess_volume_cached_memmap from Part 1/8; if absent, fallback to simple normalization
                if "preprocess_volume_cached_memmap" in globals():
                    # this function saves the processed memmap and returns a memmap
                    mm_proc = preprocess_volume_cached_memmap(raw_memmap if raw_memmap is not None else str(input_path),
                                                              name_prefix=input_path.stem, normalize=True)
                    # write meta sidecar
                    try:
                        _write_proc_meta(Path(mm_proc.filename), getattr(mm_proc, "shape", ()))
                    except Exception:
                        pass
                    vol = np.array(mm_proc)
                    proc_cache_path = Path(mm_proc.filename)
                else:
                    # fallback: simple load raw, normalize and save
                    if raw_memmap is None:
                        raw_memmap = _safe_load_memmap(input_path)
                        if raw_memmap is None:
                            st.error("Unable to load input tomogram to rebuild processed cache.")
                            return np.zeros((0,3)), {}
                    vol_tmp = np.array(raw_memmap, dtype=np.float32)
                    mu, sig = vol_tmp.mean(), vol_tmp.std()
                    vol_tmp = np.clip((vol_tmp - (mu - 3*sig)) / (2*3*sig + 1e-8), 0, 1)
                    out_path = Path(CACHE_DIR) / f"{input_path.stem}_proc.npy"
                    np.save(out_path, vol_tmp.astype(np.float32))
                    _write_proc_meta(out_path, vol_tmp.shape)
                    vol = vol_tmp
                    proc_cache_path = out_path
            except Exception as e:
                st.error(f"Rebuild processed cache failed: {e}")
                return np.zeros((0,3)), {}
        if cache_behavior == "inference_only":
            # try to use processed cache if exists, else rebuild lightly then infer
            if proc_cache_path.exists():
                mm_try = _safe_load_memmap(proc_cache_path)
                if mm_try is None:
                    st.warning("Processed cache corrupted â€” attempting a light rebuild for inference.")
                    cache_behavior = "rebuild_now"
                else:
                    vol = np.array(mm_try)
            else:
                st.info("No processed cache â€” doing a light rebuild for inference.")
                cache_behavior = "rebuild_now"

    # If input was array-like or we've set vol via cache handling above, ensure vol is set
    if vol is None:
        if isinstance(volume_memmap_or_arr, (str, Path)):
            # final attempt to load raw (if not handled above)
            mm = _safe_load_memmap(Path(volume_memmap_or_arr))
            if mm is None:
                st.error("Could not load tomogram (raw or processed). Aborting.")
                return np.zeros((0,3)), {}
            vol = np.array(mm)
        else:
            vol = np.array(volume_memmap_or_arr)

    if vol.size == 0:
        st.warning("Empty volume â€” cannot run CNN.")
        return np.zeros((0,3)), {}

    # ---- Build target heatmap (voxel-space) ----
    Z, Y, X = vol.shape
    target = np.zeros_like(vol, dtype=np.float32)
    if gt_points is not None and len(gt_points) > 0:
        for z, y, x in np.asarray(gt_points).astype(int):
            if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
                target[z, y, x] = 1.0
    target = ndimage.gaussian_filter(target, sigma=SIGMA_HEATMAP)

    # ---- Preset adjustments ----
    preset = (preset or "balanced").lower()
    min_confidence = 0.25
    if preset == "precision-boost":
        min_confidence = 0.40
        epochs = max(1, int(epochs * 0.8))
    elif preset == "recall-boost":
        min_confidence = 0.10
        epochs = max(1, int(epochs * 1.2))
    else:  # balanced
        min_confidence = 0.25

    # ---- Instantiate selected model (graceful fallback) ----
    model_choice = (model_choice or "tinycnn").lower()
    model = None
    try:
        if model_choice in ("tiny", "tinycnn"):
            model = Tiny3DCNN(in_ch=1, base_ch=base_ch).to(DEVICE)
        elif model_choice in ("unet", "unet3d") and "UNet3D_small" in globals():
            model = globals()["UNet3D_small"](in_ch=1, base_ch=base_ch).to(DEVICE)
        elif model_choice in ("resnet", "resnet3d") and "ResNet3D_small" in globals():
            model = globals()["ResNet3D_small"](in_ch=1, base_ch=base_ch).to(DEVICE)
        else:
            model = Tiny3DCNN(in_ch=1, base_ch=base_ch).to(DEVICE)
    except Exception as e:
        st.warning(f"Model instantiation failed ({e}), falling back to Tiny3DCNN.")
        model = Tiny3DCNN(in_ch=1, base_ch=base_ch).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    total_steps = max(1, epochs * TRAIN_PATCHES_PER_EPOCH)
    pb = st.progress(0)
    losses = []

    # ---- Training loop (skip if inference_only) ----
    inference_only_mode = (cache_behavior == "inference_only")
    if inference_only_mode:
        st.info("Inference-only mode: skipping training and using cached model-initialized weights.")
    else:
        st.info(f"Training {model_choice.upper()} ({preset}) for {epochs} epoch(s) â€” patches/epoch={TRAIN_PATCHES_PER_EPOCH}")
        for ep in range(epochs):
            patches_v, patches_t = extract_random_patches(vol, target, TRAIN_PATCHES_PER_EPOCH, PATCH_SIZE)
            model.train()
            for i in range(patches_v.shape[0]):
                inp = torch.from_numpy(patches_v[i:i+1]).unsqueeze(1).to(DEVICE)
                tgt = torch.from_numpy(patches_t[i:i+1]).unsqueeze(1).to(DEVICE)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                    out = model(inp)
                    loss = loss_fn(out, tgt)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu().numpy()))
                step_idx = ep * TRAIN_PATCHES_PER_EPOCH + i + 1
                pb.progress(min(100, int(100 * (step_idx / (total_steps + 1)))))
                del inp, tgt, out, loss
                torch.cuda.empty_cache()
            gc.collect(); torch.cuda.empty_cache()

    # ---- Inference with sliding window and confidence filtering ----
    st.info("Running inference with patch confidence filtering...")
    recon, conf_map = sliding_window_inference(
        vol, model, patch_size=PATCH_SIZE, stride=max(1,int(PATCH_STRIDE * stride_factor)), min_confidence=min_confidence
    )

    # ---- Debug logging: recon stats + threshold counts ----
    try:
        recon_np = np.array(recon, dtype=np.float32)
        st.write(f"DEBUG: recon.shape={recon_np.shape}, recon.min={recon_np.min():.5f}, recon.max={recon_np.max():.5f}, recon.mean={recon_np.mean():.5f}, recon.std={recon_np.std():.5f}")
        thresh = recon_np.mean() + THRESH_STD * recon_np.std()
        above_n = int((recon_np > thresh).sum())
        st.write(f"DEBUG: threshold={thresh:.5f}, voxels_above_threshold={above_n}, conf_map.mean={float(conf_map.mean()):.5f}")
    except Exception:
        st.write("DEBUG: failed to compute recon debug stats")

    # ---- adaptive thresholding using confidence_map ----
    dynamic_thresh = (float(np.mean(recon)) + THRESH_STD * float(np.std(recon))) * (1 - 0.15 * float(conf_map.mean()))
    coords = np.array(np.nonzero(recon > dynamic_thresh)).T
    if coords.shape[0] == 0:
        st.warning("âš ï¸� No voxels exceeded the threshold. Possible causes:\n"
                   "- recon values are extremely small (check recon.min/max)\n"
                   "- THRESH_STD too high for current output distribution\n"
                   "- patch-level confidence filter removed too many patches\n                   Suggestions: lower THRESH_STD, reduce min_confidence, or inspect saved recon NPZ.")
    if coords.shape[0] > MAX_POINTS:
        coords = coords[np.random.choice(coords.shape[0], MAX_POINTS, replace=False)]
    coords_xyz = coords[:, [2, 1, 0]].astype(float)
    clustered = cluster_points(coords_xyz, eps=DBSCAN_EPS)

    # ---- Save heatmap + conf_map for debugging/visualization ----
    try:
        artifact_path = CACHE_DIR / f"recon_conf_{int(time.time())}.npz"
        np.savez_compressed(artifact_path, recon=recon, conf_map=conf_map)
        st.write(f"Saved recon/conf_map to {artifact_path}")
    except Exception as e:
        st.warning(f"Could not save recon/conf_map: {e}")

    # ---- Compute nm-aware metrics if helper exists ----
    metrics = {}
    try:
        if "get_default_voxel_size" in globals() and "compute_metrics_nm" in globals():
            voxel_size = get_default_voxel_size("byu_motor")
            nm_metrics = compute_metrics_nm(clustered, gt_points, voxel_size, radius_nm=12.0)  # use 12 nm as per your choice
            metrics.update(nm_metrics)  # includes precision, recall, f2, mean_dist_nm, matched
        else:
            vox_metrics = compute_metrics(clustered, gt_points, radius=5.0)
            metrics.update({
                "precision": vox_metrics["precision"],
                "recall": vox_metrics["recall"],
                "f2": vox_metrics["f2"],
                "mean_dist_nm": float(vox_metrics.get("mean_dist", np.nan)),
                "matched": vox_metrics.get("matched", 0)
            })
    except Exception as e:
        metrics.update({"precision": 0.0, "recall": 0.0, "f2": 0.0, "mean_dist_nm": np.nan, "matched": 0, "metrics_error": str(e)})

    # ---- Aggregate additional bookkeeping fields ----
    metrics.update({
        "train_time_s": round(time.time() - t0_total, 2),
        "gpu_mem_MB": round(((torch.cuda.max_memory_allocated() if DEVICE == "cuda" else 0) - gpu_start_mem) / (1024**2), 2) if DEVICE == "cuda" else 0.0,
        "mean_train_loss": float(np.mean(losses)) if losses else None,
        "pred_count": int(len(clustered)),
        "gt_count": int(len(gt_points)) if gt_points is not None else 0,
        "confidence_mean": float(conf_map.mean()),
        "voxel_size_nm_est_z": (get_default_voxel_size("byu_motor")[0] if "get_default_voxel_size" in globals() else np.nan),
        "voxel_size_nm_est_y": (get_default_voxel_size("byu_motor")[1] if "get_default_voxel_size" in globals() else np.nan),
        "voxel_size_nm_est_x": (get_default_voxel_size("byu_motor")[2] if "get_default_voxel_size" in globals() else np.nan),
        "mean_dist_nm": float(metrics.get("mean_dist_nm", np.nan)),
        "cache_behavior_used": cache_behavior or "auto"
    })

    # ---- Save predictions and metrics to disk/log CSV ----
    try:
        out_path = CACHE_DIR / f"pred_{model_choice}_{int(time.time())}.npz"
        np.savez_compressed(out_path, points=clustered)
        metrics["saved_path"] = str(out_path)
    except Exception:
        metrics["saved_path"] = None

    try:
        df_log = pd.DataFrame([metrics])
        if LOG_CSV.exists():
            prev = pd.read_csv(LOG_CSV)
            df_log = pd.concat([prev, df_log], ignore_index=True)
        df_log.to_csv(LOG_CSV, index=False)
    except Exception as e:
        st.warning(f"Could not append to {LOG_CSV}: {e}")

    # ---- Human-readable summary & return ----
    st.success(
        f"âœ… Done â€” Fâ‚‚={metrics.get('f2', np.nan):.3f}, "
        f"Precision={metrics.get('precision', np.nan):.3f}, Recall={metrics.get('recall', np.nan):.3f}, "
        f"Mean Dist â‰ˆ {metrics.get('mean_dist_nm', np.nan):.2f} nm"
    )
    st.write("Mean Patch Confidence:", f"{metrics.get('confidence_mean', np.nan):.3f}")
    st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0:"value"}))

    gc.collect(); torch.cuda.empty_cache()
    return clustered, metrics


# ===============================================================
# Part 3 â€” Multi-Architecture Integration (T4-Optimized, Unified)
# Upgraded: checkpoint .ckpt (weights+opt+epoch), medium Streamlit logging (L2)
# ===============================================================
import os
import time
import traceback
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.express as px

# --- Assumes helper functions exist elsewhere in app:
# get_default_voxel_size(), compute_metrics_nm(), extract_random_patches(), sliding_window_inference(),
# cluster_points(), preprocess_volume_cached_memmap(), marching_cubes_mesh_adaptive()
# If those are in other parts, this will call them directly.

# ---------------- Checkpoint helpers (full .ckpt) ----------------
def save_full_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer],
                         epoch: int, path: Path, extra: Optional[dict] = None) -> str:
    try:
        payload = {
            "epoch": int(epoch),
            "model_state": model.state_dict(),
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        if extra:
            payload["extra"] = extra
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(path))
        return str(path)
    except Exception as e:
        st.warning(f"[checkpoint] save failed: {e}")
        return ""

def load_full_checkpoint(path: Path, model: Optional[nn.Module] = None,
                         optimizer: Optional[torch.optim.Optimizer] = None,
                         map_location: Optional[str] = None) -> Dict[str, Any]:
    try:
        chk = torch.load(str(path), map_location=(map_location or ("cpu" if not torch.cuda.is_available() else None)))
    except Exception as e:
        st.warning(f"[checkpoint] load failed: {e}")
        return {}
    if model is not None and "model_state" in chk:
        try:
            model.load_state_dict(chk["model_state"])
        except Exception as e:
            st.warning(f"[checkpoint] load -> model.load_state_dict failed: {e}")
    if optimizer is not None and "optimizer_state" in chk:
        try:
            optimizer.load_state_dict(chk["optimizer_state"])
        except Exception as e:
            st.warning(f"[checkpoint] load -> optimizer.load_state_dict failed: {e}")
    return chk

# ---------------- Cache helpers ----------------
@st.cache_resource
def cached_instantiate_model(model_name: str = "tinycnn", base_ch: int = 16, device: str = "cpu"):
    """Return new model instance (cached) â€” keying by args lets Streamlit keep resource alive."""
    # Reuse existing get_model if available, otherwise fall back to Tiny3DCNN or minimal fallback
    try:
        if "get_model" in globals():
            m = get_model(model_name, in_ch=1, base_ch=base_ch)
        else:
            # fallback simple tiny model
            class TinyFallback(nn.Module):
                def __init__(self, in_ch=1, base_ch=16):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Conv3d(in_ch, base_ch, 3, padding=1), nn.ReLU(),
                        nn.Conv3d(base_ch, base_ch, 3, padding=1), nn.ReLU(),
                        nn.Conv3d(base_ch, 1, 1), nn.Sigmoid()
                    )
                def forward(self, x): return self.net(x)
            m = TinyFallback(in_ch=1, base_ch=base_ch)
        return m.to(device)
    except Exception as e:
        st.warning(f"[model cache] instantiate failed: {e}")
        # final fallback
        return cached_instantiate_model.__wrapped__(model_name, base_ch, device)

# ---------------- Utilities ----------------
def _safe_np_savez(path: Path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(str(path), **kwargs)
        return True
    except Exception as e:
        st.warning(f"[io] savez failed {path}: {e}")
        return False

def _safe_np_load(path: Path):
    try:
        return np.load(str(path), mmap_mode="r")
    except Exception as e:
        st.warning(f"[io] load failed {path}: {e}")
        return None

# ---------------- Training loop with EarlyStopping (val F2) ----------------
class EarlyStopper:
    def __init__(self, patience: int = 3, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = None
        self.wait = 0
        self.stopped_epoch = None

    def step(self, current, epoch) -> bool:
        """Return True if should stop."""
        if self.best is None:
            self.best = current
            self.wait = 0
            return False
        improve = (current - self.best) if self.mode == "max" else (self.best - current)
        if improve > self.min_delta:
            self.best = current
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                return True
            return False

# ---------------- The unified upgraded runner ----------------
def run_model_pipeline_upgraded(
    volume_memmap_or_arr,
    gt_points,
    model_name: str = "tinycnn",
    base_ch: int = 16,
    epochs: int = 6,
    preset: str = "balanced",
    patch_size: int = 64,
    stride_factor: float = 1.0,
    learning_rate: float = 1e-3,
    batch_patches_per_epoch: int = 64,
    checkpoint_dir: str = None,
    cache_recon: bool = True,
    val_frac: float = 0.15,
    earlystop_patience: int = 3,
    device: Optional[str] = None,
    resume_ckpt: Optional[str] = None
) -> Tuple[np.ndarray, dict]:
    """
    Upgraded unified runner:
      - T2 training (early stop on val F2), L2 logging (loss curve + val F2 curve)
      - C3 caching of recon + artifacts
      - Checkpoint (.ckpt) saves best model (weights+optimizer+epoch)
    Returns: clustered_points (nx3), metrics dict (nm-aware)
    """
    t0_global = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    PATCH = int(patch_size)
    STRIDE = max(1, int(PATCH // 2))
    EPOCHS = int(epochs)
    PATCHES = int(batch_patches_per_epoch)
    lr = float(learning_rate)

    # Prepare checkpoint/cache paths
    ckpt_dir = Path(checkpoint_dir or (Path(CACHE_DIR) / "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(CACHE_DIR) / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # model (cached instantiation keyed by model_name+base_ch+device)
    model_cache_key = f"{model_name}_{base_ch}_{device}"
    model = cached_instantiate_model(model_name, base_ch, device)
    # ensure optim & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # optionally resume checkpoint
    start_epoch = 0
    best_val_f2 = -np.inf
    best_ckpt_path = None
    if resume_ckpt is not None:
        try:
            chk = load_full_checkpoint(Path(resume_ckpt), model=model, optimizer=optimizer, map_location=device)
            if chk:
                start_epoch = int(chk.get("epoch", 0)) + 1
                st.info(f"[resume] loaded {resume_ckpt} -> starting epoch {start_epoch}")
        except Exception as e:
            st.warning(f"[resume] failed to load resume checkpoint: {e}")

    # data load: support memmap path or array
    if isinstance(volume_memmap_or_arr, (str, Path)):
        raw = _safe_np_load(Path(volume_memmap_or_arr))
        if raw is None:
            st.error("Failed to load input volume.")
            return np.zeros((0,3)), {}
        vol = np.array(raw)
    else:
        vol = np.array(volume_memmap_or_arr)

    if vol.size == 0:
        st.warning("Empty volume â€” aborting.")
        return np.zeros((0,3)), {}

    # build target heatmap if GT provided
    Z, Y, X = vol.shape
    target = np.zeros_like(vol, dtype=np.float32)
    if gt_points is not None and len(gt_points) > 0:
        for z, y, x in np.asarray(gt_points).astype(int):
            if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
                target[z, y, x] = 1.0
        # smoothing
        if "ndimage" in globals():
            target = ndimage.gaussian_filter(target, sigma=1.2)

    # Create patch dataset (we sample on the fly)
    def sample_patches(n):
        return extract_random_patches(vol, target, n, PATCH)

    # validation split is patch-based: sample a moderate pool and split
    pool_v, pool_t = sample_patches(min(512, max(128, PATCHES * 4)))
    try:
        pv_train, pv_val, pt_train, pt_val = train_test_split(pool_v, pool_t, test_size=val_frac, random_state=42)
    except Exception:
        # fallback naive split
        split = int(len(pool_v) * (1 - val_frac))
        pv_train, pv_val = pool_v[:split], pool_v[split:]
        pt_train, pt_val = pool_t[:split], pool_t[split:]

    # early stopper monitors nm-aware val F2
    early = EarlyStopper(patience=earlystop_patience, min_delta=1e-4, mode="max")

    # training bookkeeping
    train_losses = []
    val_f2_list = []
    epochs_done = 0

    # Streamlit placeholders for medium logging (L2)
    loss_plot_placeholder = st.empty()
    val_plot_placeholder = st.empty()
    progress_bar = st.progress(0)

    # main training loop
    for ep in range(start_epoch, EPOCHS):
        model.train()
        ep_losses = []
        # sample random patches for this epoch
        train_patches_v, train_patches_t = extract_random_patches(vol, target, PATCHES, PATCH)
        for i in range(train_patches_v.shape[0]):
            inp = torch.from_numpy(train_patches_v[i:i+1]).unsqueeze(1).to(device)
            tgt = torch.from_numpy(train_patches_t[i:i+1]).unsqueeze(1).to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(inp)
                loss = loss_fn(out, tgt)
            loss.backward()
            optimizer.step()
            val_loss = float(loss.detach().cpu().numpy())
            ep_losses.append(val_loss)
            train_losses.append(val_loss)
            # progress
            progress_bar.progress(int(100 * ((ep * PATCHES + i + 1) / (EPOCHS * PATCHES + 1))))
            # free
            del inp, tgt, out, loss
            torch.cuda.empty_cache()
        epochs_done += 1

        # validation: run inference on pv_val set (patch-level reconstruction -> assemble small recon on-the-fly)
        model.eval()
        # quick per-patch inference and produce centroids for predicted vs GT
        pred_centroids = []
        gt_centroids = []
        try:
            for j in range(len(pv_val)):
                with torch.no_grad():
                    inp = torch.from_numpy(pv_val[j:j+1]).unsqueeze(1).to(device)
                    out = model(inp)[0,0].detach().cpu().numpy()
                # threshold locally on patch
                t_local = out.mean() + 0.2 * out.std()
                coords_patch = np.array(np.nonzero(out > t_local)).T
                if coords_patch.shape[0] == 0:
                    continue
                # convert patch coords -> global approx by sampling a random anchor from pt_val mapping
                # (patch sampling loses global positions; we use counts as proxy for F2)
                centroid = coords_patch.mean(axis=0)
                pred_centroids.append(centroid)
                # ground truth centroid in patch
                gt_coords_patch = np.array(np.nonzero(pt_val[j] > 0.5)).T
                if gt_coords_patch.shape[0] > 0:
                    gt_centroids.append(gt_coords_patch.mean(axis=0))
        except Exception:
            # if patch-level val fails, skip to full-vol validation later
            pred_centroids = []
            gt_centroids = []

        # compute validation F2 via nm-aware full-volume inference (if gt_points available) OR via patch proxy
        val_f2 = -np.inf
        if gt_points is not None and len(gt_points) > 0:
            # run a lighter-weight full inference for validation using smaller stride to estimate recon
            recon_cache_key = f"recon_{model_name}_{base_ch}_{hash(volume_memmap_or_arr.tobytes() if isinstance(volume_memmap_or_arr, np.ndarray) else str(volume_memmap_or_arr))}_{ep}"
            # run sliding-window inference (fast but approximate)
            recon_full, conf_map = sliding_window_inference(vol, model, patch_size=PATCH, stride=int(STRIDE * stride_factor))
            coords_full = np.array(np.nonzero(recon_full > (recon_full.mean() + 0.25 * recon_full.std()))).T
            coords_xyz = coords_full[:, [2,1,0]].astype(float) if coords_full.shape[0] > 0 else np.zeros((0,3))
            clustered_val = cluster_points(coords_xyz, eps=2.0)
            try:
                vsz = get_default_voxel_size("byu_motor") if "get_default_voxel_size" in globals() else (1.0,1.0,1.0)
                nm_m = compute_metrics_nm(clustered_val, gt_points, vsz, radius_nm=12.0 if True else 10.0)
                val_f2 = nm_m.get("f2", 0.0)
            except Exception:
                val_f2 = 0.0
        else:
            # patch proxy
            try:
                if len(pred_centroids) and len(gt_centroids):
                    # simple euclidean proxy
                    pred_arr = np.vstack(pred_centroids)
                    gt_arr = np.vstack(gt_centroids)
                    from sklearn.neighbors import NearestNeighbors
                    nbrs = NearestNeighbors(n_neighbors=1).fit(pred_arr)
                    d, _ = nbrs.kneighbors(gt_arr)
                    matched = (d[:,0] <= 3.0).sum()
                    tp = matched; fp = len(pred_arr) - tp; fn = len(gt_arr) - tp
                    precision = tp / (tp + fp + 1e-8); recall = tp / (tp + fn + 1e-8)
                    val_f2 = (1 + 2**2) * (precision * recall) / (4 * precision + recall + 1e-8)
                else:
                    val_f2 = 0.0
            except Exception:
                val_f2 = 0.0

        val_f2_list.append(val_f2)

        # Plot medium logs (L2): training loss and validation F2
        try:
            df_plot = pd.DataFrame({
                "train_loss": pd.Series(train_losses),
                "epoch": [i // PATCHES for i in range(len(train_losses))]
            })
            fig1 = px.line(df_plot, y="train_loss", x=df_plot.index, title="Training Loss (per step)")
            loss_plot_placeholder.plotly_chart(fig1, use_container_width=True)
        except Exception:
            pass

        try:
            fig2 = px.line(pd.DataFrame({"val_f2": val_f2_list}), y="val_f2", x=pd.Series(range(len(val_f2_list))),
                           title="Validation Fâ‚‚ (nm-aware)")
            val_plot_placeholder.plotly_chart(fig2, use_container_width=True)
        except Exception:
            pass

        st.write(f"[epoch {ep+1}/{EPOCHS}] train_loss={np.mean(ep_losses):.4f} | val_f2={val_f2:.4f}")

        # on improvement, save checkpoint
        if val_f2 > best_val_f2:
            best_val_f2 = float(val_f2)
            best_ckpt_path = ckpt_dir / f"best_{model_name}_ep{ep+1}.ckpt"
            save_full_checkpoint(model, optimizer, epoch=ep, path=best_ckpt_path,
                                 extra={"val_f2": best_val_f2, "model_name": model_name, "base_ch": base_ch})
            st.info(f"[checkpoint] new best val_f2={best_val_f2:.4f} saved -> {best_ckpt_path.name}")

        # early stop decision
        if early.step(val_f2, ep):
            st.info(f"[earlystop] stopping at epoch {ep+1} (patience={early.patience})")
            break

    # ---- After training: final inference (full-volume) ----
    st.info("Final full-volume inference (this may take a bit)â€¦")
    recon_final, conf_map_final = sliding_window_inference(vol, model, patch_size=PATCH, stride=int(STRIDE * stride_factor))
    # debug stats
    try:
        rmin = float(np.nanmin(recon_final)); rmax = float(np.nanmax(recon_final))
        rmean = float(np.nanmean(recon_final)); rstd = float(np.nanstd(recon_final))
        st.write(f"[recon final] shape={recon_final.shape}, min={rmin:.6f}, max={rmax:.6f}, mean={rmean:.6f}, std={rstd:.6f}")
    except Exception:
        pass

    # compute threshold used (saved)
    thresh_used = float(np.nanmean(recon_final) + 0.20 * float(np.nanstd(recon_final)))
    coords = np.array(np.nonzero(recon_final > thresh_used)).T
    coords_xyz = coords[:, [2,1,0]].astype(float) if coords.shape[0] > 0 else np.zeros((0,3))
    clustered = cluster_points(coords_xyz, eps=2.0)

    # nm-aware metrics
    metrics = {}
    try:
        vsz = get_default_voxel_size("byu_motor") if "get_default_voxel_size" in globals() else (1.0,1.0,1.0)
        metrics = compute_metrics_nm(clustered, gt_points, vsz, radius_nm=12.0)
    except Exception as e:
        metrics = {"precision": 0.0, "recall": 0.0, "f2": 0.0, "mean_dist_nm": np.nan, "matched": 0, "metrics_err": str(e)}

    # aggregate housekeeping fields
    metrics.update({
        "model": model_name,
        "base_ch": base_ch,
        "patch_size": PATCH,
        "stride": STRIDE,
        "thresh_used": thresh_used,
        "train_time_s": round(time.time() - t0_global, 2),
        "epochs_done": epochs_done,
        "best_val_f2": float(best_val_f2) if best_val_f2 is not None else np.nan,
    })

    # save artifacts (C3): recon npz, coords_precluster, clustered points, metrics json, checkpoint path
    timestamp = int(time.time())
    recon_path = artifacts_dir / f"recon_{model_name}_{timestamp}.npz"
    _safe_np_savez(recon_path, recon=recon_final.astype(np.float32), conf_map=conf_map_final.astype(np.float32))
    coords_pre_path = artifacts_dir / f"coords_pre_{model_name}_{timestamp}.npy"
    np.save(coords_pre_path, coords_xyz.astype(np.float32))
    clustered_path = artifacts_dir / f"clustered_{model_name}_{timestamp}.npy"
    np.save(clustered_path, clustered.astype(np.float32))
    metrics_path = artifacts_dir / f"metrics_{model_name}_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # save final-best checkpoint if not already
    if best_ckpt_path is None:
        best_ckpt_path = ckpt_dir / f"final_{model_name}_{timestamp}.ckpt"
        save_full_checkpoint(model, optimizer, epoch=epochs_done, path=best_ckpt_path,
                             extra={"val_f2": best_val_f2})

    # return results and info
    metrics.update({
        "recon_path": str(recon_path),
        "coords_pre_path": str(coords_pre_path),
        "clustered_path": str(clustered_path),
        "metrics_path": str(metrics_path),
        "ckpt_path": str(best_ckpt_path),
    })

    st.success(f"Run complete â€” F2={metrics.get('f2', np.nan):.3f}, mean_dist_nm={metrics.get('mean_dist_nm', np.nan):.2f}")
    # show compact metrics table
    st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "value"}))
    return clustered, metrics


# ================================================================
# Part 4 â€“ Configurable Ablation Controller (Streamlit Integrated, nm-aware, Stable)
# (Fixed TinyCNN workflow, 12nm weighted-F2, GT debug + failure signatures)
# ================================================================

import itertools, json, csv, time, gc, torch
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# ---------------- Sidebar UI for Ablation Sweep ----------------
st.sidebar.header("âš™ï¸� Ablation Controller (Fixed TinyCNN)")
with st.sidebar.expander("Sweep Settings", expanded=True):
    patch_sizes = st.multiselect("Patch sizes (voxels)",
                                 options=[32, 48, 64, 80], default=[48, 64])
    stride_factors = st.multiselect("Stride factors (speed tradeoff)",
                                    options=[0.75, 1.0, 1.25, 1.5], default=[1.0, 1.25])
    dbscan_eps_list = st.multiselect("DBSCAN eps (voxels)",
                                     options=[1.0, 1.5, 2.0, 3.0], default=[1.5, 2.0])
    presets = st.multiselect("Training preset",
                             options=["fast", "balanced", "thorough"], default=["balanced"])
    max_runs = st.number_input("Max total runs (safety cap)", min_value=1, max_value=200, value=8, step=1)
    max_runtime_per_run = st.slider("Max runtime per run (minutes)", min_value=5, max_value=120, value=30, step=5)
    max_total_runtime = st.number_input("Max total runtime (minutes)", min_value=10, max_value=480, value=120)

st.sidebar.markdown("---")
run_button = st.sidebar.button("ğŸš€ Run Ablation Sweep")
save_button = st.sidebar.button("ğŸ’¾ Save Current Results (.csv/.json)")

# ---------------- Session State ----------------
if "ablation_results" not in st.session_state:
    st.session_state["ablation_results"] = []

# ---------------- Helpers ----------------
def _make_grid(patches, strides, eps_list, presets_list, max_runs_allowed):
    combos = list(itertools.product(patches, strides, eps_list, presets_list))
    if len(combos) > max_runs_allowed:
        combos = combos[:max_runs_allowed]
    return combos

def _save_results_file(results_list, prefix="ablation_results"):
    """Save results list to CSV + JSON."""
    if not results_list:
        return None
    df = pd.DataFrame(results_list)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(CACHE_DIR) / f"{prefix}_{ts}.csv"
    json_path = Path(CACHE_DIR) / f"{prefix}_{ts}.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(results_list, f, indent=2)
    return str(csv_path), str(json_path)

def _debug_check_gt_for_tomo(tomo_name, gt_df):
    """Return whether tomo_id present in GT CSV and up to 5 sample rows for quick debug."""
    try:
        present = (gt_df['tomo_id'] == tomo_name).any()
        sample = gt_df[gt_df['tomo_id'] == tomo_name].head().to_dict(orient='records')
        return bool(present), sample
    except Exception:
        return False, []

def _sanitize_metrics(metrics):
    """Ensure required metric keys exist and are numeric where expected."""
    m = dict(metrics or {})
    # canonical numeric keys
    for k in ("precision", "recall", "f2", "mean_dist_nm", "matched", "pred_count", "gt_count"):
        if k not in m or m.get(k) is None:
            m[k] = 0.0 if k not in ("matched", "pred_count", "gt_count") else int(m.get(k, 0))
    # coerce types safely
    try:
        m["mean_dist_nm"] = float(np.nan) if str(m["mean_dist_nm"]) in ("nan","None","") else float(m["mean_dist_nm"])
    except Exception:
        m["mean_dist_nm"] = float(np.nan)
    try:
        m["f2"] = float(m.get("f2", 0.0))
    except Exception:
        m["f2"] = 0.0
    return m

def _compute_12nm_weighted_f2(metrics_dict):
    """
    Weighted F2 using 12 nm reference:
      weighted_f2 = f2 / (1 + mean_dist_nm / 12.0)
    Handles NaN mean_dist_nm by falling back to f2.
    """
    m = _sanitize_metrics(metrics_dict)
    f2 = float(m.get("f2", 0.0))
    md = m.get("mean_dist_nm", None)
    if md is None or np.isnan(md) or md <= 0:
        return float(f2)
    return float(f2) / (1.0 + float(md) / 12.0)

# ---------------- Single combo runner (wraps unified pipeline) ----------------
def _run_single_combo(volume_path_or_arr, gt_points,
                      patch_size, stride_factor, dbscan_eps, preset, max_runtime_s):
    """
    Run one ablation configuration with time limits.
    Calls run_model_pipeline (assumed available in runtime).
    Emits diagnostic logs and produces standardized metrics.
    """
    import time, traceback
    # Safely capture previous globals (may be undefined on first run)
    prev_eps = globals().get("DBSCAN_EPS", None)
    prev_patch = globals().get("PATCH_SIZE", None)
    prev_stride = globals().get("PATCH_STRIDE", None)

    # set new globals for run
    globals()["DBSCAN_EPS"] = float(dbscan_eps)
    globals()["PATCH_SIZE"] = int(patch_size)
    globals()["PATCH_STRIDE"] = max(1, int(patch_size // 2))

    try:
        t0 = time.time()
        st.write(f"â†’ Starting combo: patch={patch_size}, stride_factor={stride_factor}, eps={dbscan_eps}, preset={preset}")
        # call the unified runner (TinyCNN fixed)
        clustered, metrics = run_model_pipeline(
            volume_memmap_or_arr=volume_path_or_arr,
            gt_points=gt_points,
            model_name="tinycnn",
            model_kwargs={"base_ch": 16},
            epochs=EPOCHS,
            preset=preset,
            stride_factor=stride_factor,
            patch_size=patch_size
        )

        # sanitize
        metrics = metrics if isinstance(metrics, dict) else {"error": "metrics_not_returned"}
        metrics = _sanitize_metrics(metrics)

        # attach run metadata
        metrics.update({
            "patch_size": int(patch_size),
            "stride_factor": float(stride_factor),
            "dbscan_eps": float(dbscan_eps),
            "preset": preset,
            "runtime_s": round(time.time() - t0, 2),
            "timestamp": datetime.now().isoformat(),
        })

        # compute 12nm-weighted f2
        metrics["weighted_f2_12nm"] = _compute_12nm_weighted_f2(metrics)

        # add failure flags
        metrics["no_detections"] = bool(int(metrics.get("pred_count", 0)) == 0 or float(metrics.get("f2", 0.0)) == 0.0)
        metrics["suspicious_nan"] = any([np.isnan(metrics.get("mean_dist_nm", np.nan)),
                                        np.isnan(metrics.get("precision", np.nan)),
                                        np.isnan(metrics.get("recall", np.nan))])

        return clustered, metrics

    except Exception as e:
        tb = traceback.format_exc()
        st.error(f"_run_single_combo exception: {e}")
        return np.zeros((0,3)), {
            "error": str(e),
            "traceback": tb,
            "patch_size": patch_size,
            "stride_factor": stride_factor,
            "dbscan_eps": dbscan_eps,
            "preset": preset,
            "timestamp": datetime.now().isoformat()
        }

    finally:
        # restore previous globals only if they were present
        if prev_eps is not None:
            globals()["DBSCAN_EPS"] = prev_eps
        else:
            globals().pop("DBSCAN_EPS", None)
        if prev_patch is not None:
            globals()["PATCH_SIZE"] = prev_patch
        else:
            globals().pop("PATCH_SIZE", None)
        if prev_stride is not None:
            globals()["PATCH_STRIDE"] = prev_stride
        else:
            globals().pop("PATCH_STRIDE", None)

        gc.collect(); torch.cuda.empty_cache()

# ---------------- Main Ablation Runner ----------------
if run_button:
    tomos = find_tomograms(DATA_ROOT)
    if len(tomos) == 0:
        st.error("â�Œ No tomograms found. Check DATA_ROOT.")
    else:
        colA, colB = st.columns([1, 3])
        with colA:
            tomo_names = [t.name for t in tomos]
            selected_tomo_name = st.selectbox("Select Tomogram for Ablation", tomo_names, index=0)
        selected_tomo = [t for t in tomos if t.name == selected_tomo_name][0]

        # ---------------- Reset on tomogram change (MEDIUM level wipe) ----------------
        # clears previous results, clears cached Streamlit resources, and removes the processed memmap
        proc_cache_path = Path(CACHE_DIR) / f"{selected_tomo.name}_proc.npy"
        if ("last_tomo" not in st.session_state) or (st.session_state.get("last_tomo") != selected_tomo_name):
            st.session_state["last_tomo"] = selected_tomo_name
            st.session_state["ablation_results"] = []
            try:
                # clear Streamlit cached resources (functions annotated with st.cache_resource / st.cache_data)
                st.cache_resource.clear()
                st.cache_data.clear()
            except Exception:
                # older streamlit versions may not have clear(); ignore
                pass
            # force rebuild processed memmap for this tomo to avoid stale memmaps
            if proc_cache_path.exists():
                try:
                    proc_cache_path.unlink(missing_ok=True)
                    st.info("ğŸ”„ Tomogram changed â€” removed processed cache to force rebuild.")
                except Exception:
                    st.warning("Could not remove existing processed cache; proceeding (may use stale cache).")

        # Load memmap + GT (hybrid cache: reuse if valid, rebuild if corrupt/shape mismatch)
        try:
            raw_memmap = load_volume_from_jpegs_cached(str(selected_tomo))
            proc_cache_path = Path(CACHE_DIR) / f"{selected_tomo.name}_proc.npy"

            # Hybrid caching: try to load existing processed cache; if it fails or is invalid, rebuild
            def _safe_load_proc(path):
                try:
                    mm = np.load(path, mmap_mode="r")
                    if getattr(mm, "shape", None) is None:
                        return None
                    # a basic sanity check: nonzero size
                    if np.prod(mm.shape) == 0:
                        return None
                    return mm
                except Exception:
                    return None

            if proc_cache_path.exists():
                mm_try = _safe_load_proc(proc_cache_path)
                if mm_try is None:
                    st.warning("Existing proc cache is corrupted â€” rebuilding processed memmap.")
                    proc_cache_path.unlink(missing_ok=True)
                    volume_memmap = preprocess_volume_cached_memmap(raw_memmap, name_prefix=selected_tomo.name)
                    proc_cache_path = Path(volume_memmap.filename)
                else:
                    volume_memmap = mm_try
            else:
                volume_memmap = preprocess_volume_cached_memmap(raw_memmap, name_prefix=selected_tomo.name)
                proc_cache_path = Path(volume_memmap.filename)

            st.write(f"Loaded volume: shape={getattr(volume_memmap,'shape',None)}, cache={proc_cache_path}")
        except Exception as e:
            st.error(f"â�Œ Failed to load tomogram: {e}")
            volume_memmap = None

        # Load GT and debug mapping
        gt_points = np.zeros((0, 3))
        gt_csv = Path(DATA_ROOT) / GT_CSV
        if gt_csv.exists():
            try:
                df = pd.read_csv(gt_csv)
                present, sample_rows = _debug_check_gt_for_tomo(selected_tomo_name, df)
                st.write(f"GT mapping debug â€” tomo_id present? {present}; sample_rows (up to 5): {sample_rows}")
                sel = df[df["tomo_id"] == selected_tomo_name]
                if {"Motor axis 2", "Motor axis 1", "Motor axis 0"}.issubset(sel.columns) and len(sel) > 0:
                    gt_points = sel[["Motor axis 2", "Motor axis 1", "Motor axis 0"]].values.astype(float)
                else:
                    st.warning("GT columns missing or no rows matched â€” GT will be empty for this tomo.")
            except Exception as e:
                st.warning(f"GT CSV parse failed: {e}")
                gt_points = np.zeros((0, 3))

        if volume_memmap is None:
            st.error("âš ï¸� No volume loaded â€” aborting ablation run.")
        else:
            combos = _make_grid(patch_sizes, stride_factors, dbscan_eps_list, presets, max_runs)
            st.info(f"ğŸ§ª Starting ablation sweep: {len(combos)} configurations on {selected_tomo_name}")

            progress_bar = st.progress(0)
            results_local = []
            sweep_start = time.time()

            for idx, (ps, sf, eps, pr) in enumerate(combos):
                elapsed_min = (time.time() - sweep_start) / 60.0
                if elapsed_min > float(max_total_runtime):
                    st.warning("â�±ï¸� Global sweep time limit reached â€” stopping early.")
                    break

                st.write(f"Run {idx+1}/{len(combos)} â†’ Patch={ps}, Stride={sf}, EPS={eps}, Preset={pr}")
                try:
                    clustered, metrics = _run_single_combo(
                        volume_memmap, gt_points, patch_size=ps,
                        stride_factor=sf, dbscan_eps=eps,
                        preset=pr, max_runtime_s=max_runtime_per_run * 60.0
                    )

                    # Add tomo/run metadata
                    metrics.update({"tomo": selected_tomo_name, "run_index": idx + 1})

                    # Diagnostic messages for common failure modes
                    if metrics.get("no_detections"):
                        st.error("âš ï¸� No detections produced in this run (pred_count=0 or f2=0).")
                    if metrics.get("suspicious_nan"):
                        st.warning("âš ï¸� NaN detected in key metrics (mean_dist_nm/precision/recall).")

                    results_local.append(metrics)
                    st.write("âœ… Run metrics:", metrics)
                except Exception as e:
                    st.error(f"Run failed for {ps, sf, eps, pr}: {e}")
                    results_local.append({
                        "tomo": selected_tomo_name, "run_index": idx + 1,
                        "patch_size": ps, "stride_factor": sf, "dbscan_eps": eps,
                        "preset": pr, "error": str(e), "timestamp": datetime.now().isoformat()
                    })

                progress_bar.progress(int(100 * (idx + 1) / len(combos)))
                # persist incrementally
                st.session_state["ablation_results"].extend(results_local)
                _save_results_file(st.session_state["ablation_results"], prefix="ablation_intermediate")
                results_local = []
                gc.collect(); torch.cuda.empty_cache()

            saved = _save_results_file(st.session_state["ablation_results"], prefix="ablation_final")
            if saved:
                st.success(f"âœ… Sweep complete. Results saved to {saved[0]} and {saved[1]}")
            else:
                st.warning("âš ï¸� No results saved.")

# ---------------- Result Display ----------------
st.header("ğŸ“Š Ablation Results Summary (nm-normalized, 12nm-weighted F2)")
if st.session_state.get("ablation_results"):
    df_results = pd.DataFrame(st.session_state["ablation_results"])

    # Ensure mean_dist_nm exists
    if "mean_dist_nm" not in df_results.columns:
        # attempt to fall back to mean_dist if present
        if "mean_dist" in df_results.columns:
            df_results["mean_dist_nm"] = df_results["mean_dist"]
        else:
            df_results["mean_dist_nm"] = np.nan

    # Ensure f2 exists
    if "f2" not in df_results.columns:
        df_results["f2"] = 0.0

    # Compute 12nm weighted f2 column if not present
    if "weighted_f2_12nm" not in df_results.columns:
        df_results["weighted_f2_12nm"] = df_results.apply(lambda r: _compute_12nm_weighted_f2(r.to_dict()), axis=1)

    # Sort by weighted metric
    df_results = df_results.sort_values(by=["weighted_f2_12nm"], ascending=False).reset_index(drop=True)
    st.dataframe(df_results)

    try:
        topk = df_results.head(10)
        fig_rank = px.bar(
            topk,
            x=topk.index.astype(str),
            y="weighted_f2_12nm",
            hover_data=[c for c in ["precision", "recall", "mean_dist_nm",
                        "patch_size", "preset", "stride_factor"] if c in df_results.columns],
            title="Top 10 Runs by Weighted Fâ‚‚ (12 nm normalization)",
            labels={"x": "Run #", "weighted_f2_12nm": "Weighted Fâ‚‚ (12nm)"}
        )
        st.plotly_chart(fig_rank, width='stretch')
    except Exception:
        st.warning("âš ï¸� Could not render Fâ‚‚ chart â€” check logged fields.")
else:
    st.info("No ablation results yet â€” run the sweep from the sidebar.")

# ---------------- Manual Save ----------------
if save_button:
    if st.session_state.get("ablation_results"):
        csvp, jsonp = _save_results_file(st.session_state["ablation_results"], prefix="ablation_manual_save")
        st.success(f"ğŸ’¾ Saved current ablation results to: {csvp} and {jsonp}")
    else:
        st.warning("No results to save.")


# ============================================================
# Part 5 â€” Automated Reports & Visualization Dashboard (T4-safe, nm-aware, PDF Snapshot + Runtime Chart)
# ============================================================

import io, time, psutil, platform
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import plotly.express as px
import plotly.io as pio

# ğŸ”’ Ensure CACHE_DIR is always a Path
CACHE_DIR = Path("/kaggle/working/cache_phase1")

st.title("ğŸ”¬ Cryo-ET 3D Localization Benchmark Dashboard")

tomos = find_tomograms(DATA_ROOT)
if len(tomos) == 0:
    st.error("No tomogram folders found under dataset path.")
    st.stop()

# Sidebar configuration
with st.sidebar:
    st.header("âš™ï¸� Configuration")
    selected_name = st.selectbox("Select Tomogram", [t.name for t in tomos])
    selected_file = [t for t in tomos if t.name == selected_name][0]
    model_name = st.selectbox("Select Architecture", ["tinycnn", "unet3d", "resnet3d"])
    preset = st.radio("Preset", ["fast", "balanced", "accurate"], index=1)
    base_ch = st.slider("Base Channels", 8, 32, 16, 4)
    n_epochs = st.slider("Epochs", 1, 10, 3)
    run_trigger = st.button("ğŸš€ Run Model")

# ---- Load data ----
raw_memmap = load_volume_from_jpegs_cached(str(selected_file))
proc_cache_path = CACHE_DIR / f"{selected_file.name}_proc.npy"
if proc_cache_path.exists():
    vol_memmap = np.load(proc_cache_path, mmap_mode="r")
else:
    vol_memmap = preprocess_volume_cached_memmap(raw_memmap, name_prefix=selected_file.name)
    proc_cache_path = Path(vol_memmap.filename)
st.write(f"Volume shape: {getattr(vol_memmap,'shape',None)}, memmap: {proc_cache_path}")

# ---- Load GT ----
gt_points = np.zeros((0, 3))
gt_csv = Path(DATA_ROOT) / GT_CSV
if gt_csv.exists():
    try:
        df = pd.read_csv(gt_csv)
        sel = df[df["tomo_id"] == selected_name]
        if {"Motor axis 2", "Motor axis 1", "Motor axis 0"}.issubset(sel.columns) and len(sel) > 0:
            gt_points = sel[["Motor axis 2", "Motor axis 1", "Motor axis 0"]].values.astype(float)
    except Exception:
        pass

# ---- Tabs ----
tab_run, tab_metrics, tab_viz, tab_report = st.tabs(["ğŸ�ƒ Run", "ğŸ“ˆ Metrics", "ğŸ§­ Visualization", "ğŸ“„ Report Export"])

# ------------------------------------------------
# TAB 1 â€” Run
# ------------------------------------------------
with tab_run:
    if run_trigger:
        st.info(f"Running {model_name.upper()} ({preset})...")
        start_time = time.time()
        mem_before = psutil.virtual_memory().used / (1024**3)

        recon_points, metrics = run_model_pipeline(
            vol_memmap, gt_points,
            model_name=model_name,
            model_kwargs={"base_ch": base_ch},
            epochs=n_epochs,
            preset=preset
        )

        runtime = time.time() - start_time
        gpu_mem = torch.cuda.memory_allocated() / (1024**3) if DEVICE == "cuda" else 0.0

        matched, gt_count, dists = match_predictions_to_gt(recon_points, gt_points)
        results_row = {
            "tomo": selected_name,
            "model": model_name,
            "preset": preset,
            "base_ch": base_ch,
            "runtime_s": round(runtime, 2),
            "gpu_mem_gb": round(gpu_mem, 2),
            "matched": int(matched),
            "gt_count": int(gt_count),
            "pred_count": int(len(recon_points)),
            "mean_dist": float(np.mean(dists)) if len(dists) else np.nan,
            # prefer nm-aware metric from model pipeline; fallback to mean_dist
            "mean_dist_nm": metrics.get("mean_dist_nm", np.nan),
            "f2": metrics.get("f2", np.nan),
            "precision": metrics.get("precision", np.nan),
            "recall": metrics.get("recall", np.nan),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        metrics_path = CACHE_DIR / "benchmark_metrics.csv"
        pd.DataFrame([results_row]).to_csv(metrics_path, mode="a", index=False, header=not metrics_path.exists())

        st.success(f"âœ… {model_name.upper()} done in {runtime:.1f}s, GPU {gpu_mem:.2f} GB used")
        st.write(f"Matched {matched}/{gt_count}, mean distance = {results_row['mean_dist']:.3f}")
        st.write(f"Fâ‚‚ = {metrics.get('f2',0):.3f}â€ƒPrecision = {metrics.get('precision',0):.3f}â€ƒRecall = {metrics.get('recall',0):.3f}")
        np.save(CACHE_DIR / f"{selected_name}_{model_name}_pts.npy", recon_points)
    else:
        st.info("Click ğŸš€ Run Model in sidebar to start training & inference.")

# ------------------------------------------------
# TAB 2 â€” Metrics
# ------------------------------------------------
with tab_metrics:
    metrics_path = CACHE_DIR / "benchmark_metrics.csv"
    if metrics_path.exists():
        dfm = pd.read_csv(metrics_path)

        # coerce numeric columns to numeric (prevents string formatting errors)
        for col in ["f2", "mean_dist_nm", "precision", "recall", "runtime_s", "matched", "pred_count", "gt_count", "gpu_mem_gb"]:
            if col in dfm.columns:
                dfm[col] = pd.to_numeric(dfm[col], errors="coerce")

        st.dataframe(dfm.sort_values("timestamp", ascending=False), width='stretch')

        if len(dfm) > 1:
            # ğŸ§  Ensure 'mean_dist_nm' column exists (fallback to mean_dist if present)
            if "mean_dist_nm" not in dfm.columns:
                if "mean_dist" in dfm.columns:
                    dfm["mean_dist_nm"] = pd.to_numeric(dfm["mean_dist"], errors="coerce")
                else:
                    dfm["mean_dist_nm"] = np.nan

            # âœ… Dynamically choose hover columns that exist
            hover_cols = [c for c in ["tomo", "pred_count", "f2", "precision", "recall"] if c in dfm.columns]

            fig = px.scatter_3d(
                dfm,
                x="runtime_s",
                y="mean_dist_nm",
                z="gpu_mem_gb",
                color="model",
                symbol="preset",
                size="matched" if "matched" in dfm.columns else None,
                hover_data=hover_cols,
                title="Runtime vs Accuracy (nm) vs GPU Usage",
            )
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("No benchmark metrics yet â€” run at least one model.")


# ------------------------------------------------
# TAB 3 â€” Visualization
# ------------------------------------------------
with tab_viz:
    st.subheader("3D Mesh + Points (adaptive)")
    mesh_verts, mesh_faces = marching_cubes_mesh_adaptive(proc_cache_path)

    pred_path = CACHE_DIR / f"{selected_name}_{model_name}_pts.npy"
    recon_points = np.load(pred_path) if pred_path.exists() else np.zeros((0, 3))

    fig3d = plot_volume_mesh_and_points(
        mesh_verts, mesh_faces,
        points=recon_points, gt_pts=gt_points, pred_pts=recon_points,
        title=selected_name
    )
    st.plotly_chart(fig3d, width='stretch')

    # ğŸ“¸ Save snapshot for report
    snapshot_path = CACHE_DIR / "last_viz_snapshot.png"
    try:
        pio.write_image(fig3d, str(snapshot_path), format="png", width=900, height=700, scale=2)
        st.success(f"ğŸ“¸ Snapshot saved for report: {snapshot_path.name}")
    except Exception as e:
        st.warning(f"Snapshot export skipped: {e}")

    st.subheader("Slice Viewer")
    z_index = st.slider("Select Z slice", 0, max(0, vol_memmap.shape[0]-1), 0)
    fig_slice = plot_slice_with_points_memmap(
        proc_cache_path, z_index,
        points=recon_points, gt_pts=gt_points, pred_pts=recon_points,
        title=selected_name
    )
    st.plotly_chart(fig_slice, width='stretch')

# ------------------------------------------------
# TAB 4 â€” Report Export (PDF with Snapshot + Chart)
# ------------------------------------------------
with tab_report:
    st.subheader("ğŸ“„ Generate Summary PDF (with Visualization & Fâ‚‚ Chart)")
    metrics_path = CACHE_DIR / "benchmark_metrics.csv"
    if metrics_path.exists():
        dfm = pd.read_csv(metrics_path)

        # --- Ensure numeric columns are floats (coerce silently) ---
        for col in ["f2", "mean_dist_nm", "precision", "recall", "runtime_s", "matched"]:
            if col in dfm.columns:
                dfm[col] = pd.to_numeric(dfm[col], errors="coerce")

        # ğŸ“Š Generate Fâ‚‚ vs mean distance chart
        chart_path = CACHE_DIR / "runtime_vs_f2_chart.png"
        try:
            # ensure columns exist for plotting
            xcol = "mean_dist_nm" if "mean_dist_nm" in dfm.columns else ("mean_dist" if "mean_dist" in dfm.columns else None)
            if xcol is not None and "f2" in dfm.columns:
                fig_chart = px.scatter(
                    dfm, x=xcol, y="f2",
                    color="model" if "model" in dfm.columns else None,
                    symbol="preset" if "preset" in dfm.columns else None,
                    size="matched" if "matched" in dfm.columns else None,
                    title="Fâ‚‚ vs Mean Distance (nm)",
                    labels={xcol: "Mean Distance (nm)", "f2": "Fâ‚‚ Score"}
                )
                pio.write_image(fig_chart, str(chart_path), format="png", width=800, height=600, scale=2)
                st.image(str(chart_path), caption="Fâ‚‚ vs Mean Distance Chart (preview)", width='stretch')
            else:
                st.info("Not enough numeric columns to render Fâ‚‚ chart.")
        except Exception as e:
            st.warning(f"Chart export skipped: {e}")

        # ğŸ§¾ Assemble PDF
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter)
        styles = getSampleStyleSheet()

        # --- Safe metric extraction: coerce to numeric and handle NaN ---
        best_f2 = pd.to_numeric(dfm.get("f2"), errors="coerce").max() if "f2" in dfm.columns else np.nan
        min_dist = pd.to_numeric(dfm.get("mean_dist_nm"), errors="coerce").min() if "mean_dist_nm" in dfm.columns else np.nan

        story = [
            Paragraph("<b>3D Cryo-ET Localization Benchmark Summary</b>", styles["Title"]),
            Spacer(1, 12),
            Paragraph(f"System: {platform.node()} | {platform.processor()}", styles["Normal"]),
            Paragraph(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
            Spacer(1, 12),
            Paragraph(f"Total Runs: {len(dfm)}", styles["Normal"]),
            Paragraph(f"Best Fâ‚‚: {best_f2:.3f}" if pd.notna(best_f2) else "Best Fâ‚‚: N/A", styles["Normal"]),
            Paragraph(f"Min Mean Distance (nm): {min_dist:.2f}" if pd.notna(min_dist) else "Min Mean Distance (nm): N/A", styles["Normal"]),
            Spacer(1, 12),
        ]

        # ğŸ–¼ Add snapshot and chart if available
        snap_img = CACHE_DIR / "last_viz_snapshot.png"
        if snap_img.exists():
            story += [
                Paragraph("<b>3D Visualization Snapshot</b>", styles["Heading2"]),
                Spacer(1, 6),
                Image(str(snap_img), width=450, height=350),
                Spacer(1, 12),
            ]
        if chart_path.exists():
            story += [
                Paragraph("<b>Fâ‚‚ vs Mean Distance Chart</b>", styles["Heading2"]),
                Spacer(1, 6),
                Image(str(chart_path), width=400, height=300),
                Spacer(1, 12),
            ]

        # ğŸ“‹ Add metrics table (safely convert non-serializable types)
        table_data = [list(dfm.columns)] + dfm.fillna("").values.tolist()
        tbl = Table(table_data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
        ]))
        story.append(tbl)

        doc.build(story)
        pdf_bytes = buf.getvalue()
        st.download_button("â¬‡ï¸� Download PDF Report", pdf_bytes, file_name="CryoET_Benchmark_Report.pdf")
    else:
        st.info("No metrics logged yet â€” run a model to generate report.")



!sed -i 's/global PATCH_SIZE as _PATCH_SIZE_GLOBAL, PATCH_STRIDE as _PATCH_STRIDE_GLOBAL/global PATCH_SIZE, PATCH_STRIDE/' /kaggle/working/app.py



!grep -n "global PATCH_SIZE" /kaggle/working/app.py



# ============================================================
# Part 6 v2 â€” Data Augmentation & Semi-Synthetic Expansion (Upgraded)
# (CPU-safe, nm-aware, with GT-transform, logging & visualization)
# ============================================================

import os, random, gc, math
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter, zoom
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

# --- ensure cache & aug directories exist ---
CACHE_DIR = Path("/kaggle/working/cache_phase1")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
AUG_DIR = CACHE_DIR / "augmented"
AUG_DIR.mkdir(parents=True, exist_ok=True)
AUG_LOG = CACHE_DIR / "augmentation_log.csv"

# --------------------------------------------------------------------------------
# Helper: transform GT points to match an augmented variant that used zoom + padding
# --------------------------------------------------------------------------------
def transform_gt_points_for_variant(
    gt_points: np.ndarray,
    original_shape: Tuple[int, int, int],
    scale: float,
    pad_before: Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int]]
) -> np.ndarray:
    """
    Map GT points from original volume coordinates -> augmented volume coordinates.
    - gt_points: Nx3 array in (x, y, z) voxel coordinates (consistent with your code).
    - original_shape: (Z, Y, X) of the base volume used for augmentation.
    - scale: zoom factor applied to base -> vol_scaled (float)
    - pad_before: ((pad_z0,pad_z1),(pad_y0,pad_y1),(pad_x0,pad_x1)) used before final cropping.
    Returns transformed Nx3 float coordinates.
    """
    if gt_points is None or len(gt_points) == 0:
        return gt_points.copy() if gt_points is not None else np.zeros((0,3))

    # The augmentation pipeline does: scaled = zoom(base, scale)  -> pad -> crop to base.shape
    # So mapping original -> scaled coords: scaled_coord = coord * scale
    # Then padded: shifted_coord = scaled_coord + pad_before
    gt = np.array(gt_points, dtype=float).copy()  # Nx3 (x,y,z)
    # scale x,y,z â€” zoom used same factor for all dims in this pipeline
    gt_scaled = gt * float(scale)
    # pad_before is ((pz0,pz1),(py0,py1),(px0,px1)) but we stored it as list-of-tuples in meta
    # Our coords order is (x,y,z) while pad tuple is (z,y,x)
    pz0, pz1 = pad_before[0]
    py0, py1 = pad_before[1]
    px0, px1 = pad_before[2]
    gt_shifted = gt_scaled + np.array([px0, py0, pz0], dtype=float)
    # Finally, cropping in augmentation ensured shape==original_shape; keep points inside bounds
    Z, Y, X = original_shape
    keep_mask = (
        (gt_shifted[:, 2] >= 0) & (gt_shifted[:, 2] < Z) &
        (gt_shifted[:, 1] >= 0) & (gt_shifted[:, 1] < Y) &
        (gt_shifted[:, 0] >= 0) & (gt_shifted[:, 0] < X)
    )
    return gt_shifted[keep_mask]

# --------------------------------------------------------------------------------
# Augmentation generator (returns paths + per-variant metadata including scale & pad)
# --------------------------------------------------------------------------------
def generate_augmented_variants(
    volume_memmap_or_arr,
    name_prefix: str,
    n_variants: int = 3,
    seed: int = None
) -> Tuple[List[Path], pd.DataFrame]:
    """
    Generate semi-synthetic noisy / blurred / rescaled variants.
    Returns list of .npy paths and a DataFrame with per-variant metadata:
      - variant, path, gaussian_sigma, blur_sigma, scale, pad_before, snr_db, mean, std
    The pad_before entry is stored as a stringified tuple for CSV compatibility.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # load base
    if isinstance(volume_memmap_or_arr, (str, Path)):
        mm = np.load(volume_memmap_or_arr, mmap_mode='r')
        base = np.array(mm, dtype=np.float32)
    else:
        base = np.array(volume_memmap_or_arr, dtype=np.float32)

    variants = []
    meta = []
    base_std = float(base.std()) + 1e-12

    for i in range(n_variants):
        vol = base.copy()
        scale = 1.0
        pad_before = ((0, 0), (0, 0), (0, 0))

        # 1) Gaussian noise magnitude relative to base std
        sigma_frac = np.random.uniform(0.01, 0.12)  # broader range
        sigma_n = sigma_frac * base_std
        gauss = np.random.normal(0, sigma_n, vol.shape).astype(np.float32)
        vol = vol + gauss

        # Optional Poisson-like shot noise
        if np.random.rand() < 0.45:
            lam = np.random.uniform(0.6, 2.0)
            safe = np.clip(vol - vol.min(), 0, None)
            vol = (np.random.poisson(safe * lam).astype(np.float32) / max(lam, 1e-6)) + vol.min()

        # 2) Defocus blur
        blur_sigma = np.random.uniform(0.3, 1.6)
        vol = gaussian_filter(vol, sigma=blur_sigma)

        # 3) Random voxel scaling / downsampling (preserve original shape by pad/crop)
        if np.random.rand() < 0.66:
            scale = float(np.random.uniform(0.6, 1.0))
            vol_scaled = zoom(vol, scale, order=1)  # linear interpolation
            vz, vy, vx = vol_scaled.shape
            bz, by, bx = base.shape
            # compute pad amounts (before cropping back to base shape)
            pad_z0 = max(0, (bz - vz) // 2)
            pad_y0 = max(0, (by - vy) // 2)
            pad_x0 = max(0, (bx - vx) // 2)
            pad_z1 = max(0, bz - vz - pad_z0)
            pad_y1 = max(0, by - vy - pad_y0)
            pad_x1 = max(0, bx - vx - pad_x0)
            pad_before = ((pad_z0, pad_z1), (pad_y0, pad_y1), (pad_x0, pad_x1))
            # pad then crop
            vol = np.pad(vol_scaled, pad_before, mode='reflect')
            vol = vol[:bz, :by, :bx]
        else:
            scale = 1.0
            pad_before = ((0, 0), (0, 0), (0, 0))

        # 4) Contrast normalization to stable range
        vol = (vol - vol.mean()) / (vol.std() + 1e-8)
        vol = np.clip(vol, -3, 3)

        # compute SNR estimate
        snr_db = float(20 * np.log10((base_std + 1e-8) / (sigma_n + 1e-8)))

        out_path = AUG_DIR / f"{name_prefix}_aug_{i+1}.npy"
        np.save(out_path, vol.astype(np.float32))
        variants.append(out_path)

        meta.append({
            "variant": out_path.stem,
            "path": str(out_path),
            "gaussian_sigma": float(round(sigma_n, 6)),
            "blur_sigma": float(round(blur_sigma, 4)),
            "scale": float(round(scale, 4)),
            # store pad as serializable tuple-of-tuples
            "pad_before": str(((int(pad_before[0][0]), int(pad_before[0][1])),
                              (int(pad_before[1][0]), int(pad_before[1][1])),
                              (int(pad_before[2][0]), int(pad_before[2][1])))),
            "snr_db": float(round(snr_db, 2)),
            "mean": float(round(float(vol.mean()), 6)),
            "std": float(round(float(vol.std()), 6)),
        })

        gc.collect()

    # Save metadata log (append if exists)
    df_meta = pd.DataFrame(meta)
    if AUG_LOG.exists():
        prev = pd.read_csv(AUG_LOG)
        # avoid duplicate columns mismatch; align columns
        try:
            df_meta = pd.concat([prev, df_meta], ignore_index=True, sort=False)
        except Exception:
            df_meta = df_meta
    df_meta.to_csv(AUG_LOG, index=False)
    return variants, df_meta

# --------------------------------------------------------------------------------
# Streamlit UI + Integration for Augmentation + Auto-Train
# --------------------------------------------------------------------------------
with st.expander("ğŸ§¬ Semi-Synthetic Data Augmentation (v2)", expanded=False):
    st.markdown("""
    Create semi-synthetic tomograms (noise, blur, voxel scaling).
    Variants record the exact transform (scale + pad) so ground-truth points
    are transformed consistently when re-training / evaluating.
    """)
    n_aug = st.slider("Number of augmented variants", 1, 6, 3)
    gen_seed = st.number_input("Augmentation seed (optional)", min_value=0, value=42, step=1)
    if st.button("Generate Augmented Tomograms"):
        with st.spinner("Generating semi-synthetic tomograms..."):
            aug_paths, meta_df = generate_augmented_variants(proc_cache_path, selected_name, n_variants=n_aug, seed=int(gen_seed))
        st.success(f"âœ… {len(aug_paths)} augmented tomograms saved to {AUG_DIR}")
        st.dataframe(meta_df)

        # Preview one variant
        rand_path = np.random.choice(aug_paths)
        rand_aug = np.load(rand_path, mmap_mode='r')
        mid_z = rand_aug.shape[0] // 2

        col1, col2 = st.columns(2)
        with col1:
            st.image(rand_aug[mid_z, :, :], caption=f"Preview of {Path(rand_path).name}", use_container_width=True)
        with col2:
            fig, ax = plt.subplots()
            ax.hist(rand_aug.flatten(), bins=60)
            ax.set_title("Voxel Intensity Distribution")
            st.pyplot(fig)

        st.caption("Tip: Use 'Augment & Train Automatically' to train on these variants with GT transformed accordingly.")

# --------------------------------------------------------------------------------
# Auto Augment -> Train loop (applies GT transform + 12 nm normalization)
# --------------------------------------------------------------------------------
if st.button("ğŸ§  Augment & Train Automatically (TinyCNN baseline, nm-radius=12)"):
    st.info("Running augmentation + model pipeline...")
    # load current GT (original)
    gt_csv = Path(DATA_ROOT) / GT_CSV
    orig_gt_df = None
    if gt_csv.exists():
        try:
            orig_gt_df = pd.read_csv(gt_csv)
        except Exception as e:
            st.warning(f"Could not load GT CSV: {e}")
            orig_gt_df = None

    aug_paths, meta_df = generate_augmented_variants(proc_cache_path, selected_name, n_variants=n_aug, seed=int(gen_seed))
    st.write(f"Generated {len(aug_paths)} variants; starting training loop...")

    results = []
    for path in aug_paths:
        st.write(f"â†’ Training on {Path(path).name} ...")
        # fetch meta for this variant
        meta_row = meta_df[meta_df["path"] == str(path)]
        if meta_row.empty:
            st.warning("Missing metadata for variant â€” skipping GT transform (using original GT).")
            variant_meta = None
        else:
            variant_meta = meta_row.iloc[0].to_dict()

        # parse pad_before and scale
        try:
            scale = float(variant_meta.get("scale", 1.0)) if variant_meta is not None else 1.0
            pad_before_str = variant_meta.get("pad_before", "((0, 0),(0, 0),(0, 0))") if variant_meta is not None else "((0, 0),(0, 0),(0, 0))"
            # safe eval of small tuple string
            pad_before = eval(pad_before_str)
            # pad_before is ((pz0,pz1),(py0,py1),(px0,px1))
        except Exception:
            scale = 1.0
            pad_before = ((0, 0), (0, 0), (0, 0))

        # transform GT for this variant (if GT available)
        if orig_gt_df is not None and not orig_gt_df.empty:
            sel = orig_gt_df[orig_gt_df["tomo_id"] == selected_name]
            if {"Motor axis 2", "Motor axis 1", "Motor axis 0"}.issubset(sel.columns):
                # original GT is in columns (Motor axis 2 -> z, axis1 -> y, axis0 -> x)
                gt_base = sel[["Motor axis 0", "Motor axis 1", "Motor axis 2"]].values.astype(float)  # x,y,z
                # original_shape from proc_cache_path
                base_mm = np.load(proc_cache_path, mmap_mode='r')
                original_shape = base_mm.shape  # (Z,Y,X)
                # transform
                gt_transformed = transform_gt_points_for_variant(gt_base, original_shape, scale, pad_before)
            else:
                gt_transformed = np.zeros((0, 3))
        else:
            gt_transformed = np.zeros((0, 3))

        # Run the pipeline on augmented volume with transformed GT
        try:
            recon_pts, metrics = run_model_pipeline(
                volume_memmap_or_arr=str(path),
                gt_points=gt_transformed,
                model_name="tinycnn",
                model_kwargs={"base_ch": 16},
                epochs=3,
                preset="balanced"  # more stable than fast for augmented noisy data
            )
        except Exception as e:
            st.error(f"Pipeline error for {Path(path).name}: {e}")
            recon_pts, metrics = np.zeros((0,3)), {"error": str(e)}

        # If pipeline produced internal recon stats (from Part 4 debug), display them
        if isinstance(metrics, dict):
            recon_mean = metrics.get("recon_mean", None)
            recon_std = metrics.get("recon_std", None)
            vox_above = metrics.get("voxels_above_threshold", None)
            if recon_mean is not None:
                st.write(f"Recon stats â€” mean: {recon_mean:.6f}, std: {recon_std:.6f}, vox_above: {vox_above}")
            else:
                # best-effort compute from saved pred if available
                st.write("Recon stats not available from pipeline; showing returned metrics summary.")
        else:
            st.write("No metrics returned from pipeline.")

        # Re-compute nm-metrics with explicit radius_nm=12 and record augmentation metadata
        try:
            voxel_size = get_default_voxel_size(selected_name)  # existing helper from Part 1
            nm_metrics_override = compute_metrics_nm(recon_pts, gt_transformed, voxel_size_nm=voxel_size, radius_nm=12.0)
        except Exception as e:
            nm_metrics_override = {"precision": np.nan, "recall": np.nan, "f2": np.nan, "mean_dist_nm": np.nan, "matched": 0}

        # Merge/overwrite important metric keys (ensures 12 nm normalization)
        if not isinstance(metrics, dict):
            metrics = {}
        metrics.update({
            "precision": float(nm_metrics_override.get("precision", np.nan)),
            "recall": float(nm_metrics_override.get("recall", np.nan)),
            "f2": float(nm_metrics_override.get("f2", np.nan)),
            "mean_dist_nm": float(nm_metrics_override.get("mean_dist_nm", np.nan)),
            "matched": int(nm_metrics_override.get("matched", 0)),
            "aug_variant": Path(path).stem,
            "aug_scale": float(scale),
            "aug_pad_before": str(pad_before),
            "aug_gaussian_sigma": float(variant_meta.get("gaussian_sigma", np.nan)) if variant_meta is not None else np.nan,
            "aug_blur_sigma": float(variant_meta.get("blur_sigma", np.nan)) if variant_meta is not None else np.nan,
            "aug_snr_db": float(variant_meta.get("snr_db", np.nan)) if variant_meta is not None else np.nan,
            "pred_count": int(len(recon_pts))
        })

        # Safe numeric cleaning
        for k in ["precision","recall","f2","mean_dist_nm"]:
            try:
                metrics[k] = float(metrics.get(k, np.nan))
            except Exception:
                metrics[k] = np.nan

        # Append to benchmark_metrics.csv (same file used by Part 5)
        metrics_path = CACHE_DIR / "benchmark_metrics.csv"
        try:
            pd.DataFrame([metrics]).to_csv(metrics_path, mode="a", index=False, header=not metrics_path.exists())
        except Exception as e:
            st.warning(f"Could not write metrics row for {Path(path).name}: {e}")

        # Show summary to user
        st.success(
            f"Trained on {Path(path).name} â€” Fâ‚‚={metrics.get('f2', np.nan):.3f}, "
            f"MeanDist_nm={metrics.get('mean_dist_nm', np.nan):.2f}, Preds={metrics.get('pred_count',0)}"
        )
        results.append(metrics)

    st.write("Augment & Train loop complete. Summary table:")
    if len(results):
        st.dataframe(pd.DataFrame(results))

    st.balloons()



# ================================================================
# Part 7 â€” Reproducibility Tracker, Seed Control & Auto-Logging (v2)
# ================================================================
import os, json, random, socket, subprocess, warnings, hashlib, textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import platform
import streamlit as st
import plotly.express as px

# ---------------- Directories ----------------
CACHE_DIR = Path("/kaggle/working/cache_phase1")
CACHE_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR = CACHE_DIR / "experiment_logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_CSV = CACHE_DIR / "experiment_summary.csv"
AUG_LOG = CACHE_DIR / "augmentation_log.csv"
_CHECKSUM_FILE = LOG_DIR / "logged_checksums.txt"   # used to avoid double-logging
BENCHMARK_CSV = Path(CACHE_DIR) / "benchmark_metrics.csv"  # canonical metrics source

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Seed control ----------------
def set_seed(seed: int = 42, deterministic: bool = False):
    """Set global seeds for reproducibility and return RNG fingerprint."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    state = (
        str(seed)
        + str(np.random.get_state())
        + str(random.getstate()[1][:10])
        + str(torch.get_rng_state()[:10].tolist())
    )
    fingerprint = hashlib.sha1(state.encode()).hexdigest()[:12]
    return seed, fingerprint

# default seed + fingerprint
GLOBAL_SEED, GLOBAL_FP = set_seed(42)

# ---------------- Environment capture ----------------
def _get_env_info():
    info = {
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "seed": GLOBAL_SEED,
        "fingerprint": GLOBAL_FP,
        "timestamp": datetime.now().isoformat()
    }
    pkgs = {}
    for pkg in ("torch", "numpy", "pandas", "scipy", "sklearn", "plotly"):
        try:
            mod = __import__(pkg)
            pkgs[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            pkgs[pkg] = "not_installed"
    info["packages"] = pkgs

    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = {
                "name": props.name,
                "total_mem_GB": round(props.total_memory / 1e9, 2),
                "count": torch.cuda.device_count(),
            }
        except Exception:
            info["gpu"] = None
    else:
        info["gpu"] = None

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        info["git_commit"] = commit
    except Exception:
        info["git_commit"] = None
    return info

# ---------------- Helpers ----------------
def _row_checksum(row: dict) -> str:
    """Compute a stable checksum for a metrics/config row to avoid double-logging."""
    h = hashlib.sha1()
    # ensure deterministic ordering
    txt = json.dumps(row, sort_keys=True, default=str)
    h.update(txt.encode())
    return h.hexdigest()

def _mark_logged(checksum: str):
    _CHECKSUM_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _CHECKSUM_FILE.exists():
        with open(_CHECKSUM_FILE, "a") as f:
            f.write(checksum + "\n")
    else:
        with open(_CHECKSUM_FILE, "w") as f:
            f.write(checksum + "\n")

def _is_logged(checksum: str) -> bool:
    if not _CHECKSUM_FILE.exists():
        return False
    with open(_CHECKSUM_FILE, "r") as f:
        s = set(line.strip() for line in f)
    return checksum in s

def _serializable_model_summary(model: nn.Module):
    """Return a lightweight serializable summary of the model architecture."""
    if model is None:
        return None
    try:
        summary = {"class": model.__class__.__name__, "layers": []}
        for k, v in model.state_dict().items():
            summary["layers"].append({"name": k, "shape": list(v.shape), "numel": int(v.numel())})
        summary["total_params"] = int(sum(p.numel() for p in model.parameters()))
        return summary
    except Exception:
        # fallback: return class name only
        return {"class": model.__class__.__name__ if hasattr(model, "__class__") else str(type(model))}

# ---------------- Checkpoint helpers ----------------
def save_checkpoint(model: nn.Module, optimizer=None, path: Path = None, extra: dict = None):
    """Save model weights (+ optimizer if provided)."""
    if path is None:
        path = LOG_DIR / f"ckpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    try:
        payload = {"model_state": model.state_dict()}
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        if extra:
            payload["extra"] = extra
        torch.save(payload, str(path))
        return str(path)
    except Exception as e:
        warnings.warn(f"Checkpoint save failed: {e}")
        return None

def load_checkpoint(path: str, model: nn.Module = None, optimizer=None, map_location=None):
    chk = torch.load(path, map_location=(map_location or DEVICE))
    if model is not None and "model_state" in chk:
        model.load_state_dict(chk["model_state"])
    if optimizer is not None and "optimizer_state" in chk:
        optimizer.load_state_dict(chk["optimizer_state"])
    return chk

# ---------------- Experiment logging ----------------
def _make_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{np.random.randint(0,1e6):06d}"

def log_experiment(config: dict, metrics: dict, model: nn.Module = None, optimizer=None,
                   save_model: bool = False, extra: dict = None):
    """
    Atomically log an experiment run + optional checkpoint.
    - config: dict describing run configuration (tomo, model_name, preset, etc.)
    - metrics: dict of computed metrics (should include mean_dist_nm, f2, precision, recall, etc.)
    - model: optional torch model to snapshot architecture summary (not full weights unless save_model=True)
    - extra: optional dict to store additional items (loss_curve, recon_stats, voxel_size_nm, gt_counts, etc.)
    """
    # sanitize inputs
    config = config or {}
    metrics = metrics or {}
    extra = extra or {}

    run_id = _make_run_id()
    env = _get_env_info()

    # model summary (lightweight)
    model_summary = _serializable_model_summary(model) if model is not None else None

    # coerce numeric fields to basic types
    def _safe_num(x, default=None):
        try:
            if x is None:
                return default
            if isinstance(x, (np.floating, np.integer)):
                return float(x)
            return float(x)
        except Exception:
            return default

    # canonical mean_dist_nm fallback
    mean_dist_nm = metrics.get("mean_dist_nm", metrics.get("mean_dist", None))

    log = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
        "extra": extra,
        "env": env,
        "model_summary": model_summary
    }

    # write json
    json_path = LOG_DIR / f"exp_{run_id}.json"
    try:
        with open(json_path, "w") as f:
            json.dump(log, f, indent=2, default=str)
    except Exception as e:
        warnings.warn(f"Failed to write experiment JSON: {e}")
        json_path = None

    model_path = None
    if save_model and model is not None:
        model_path = save_checkpoint(
            model, optimizer=optimizer,
            path=LOG_DIR / f"model_{run_id}.pth",
            extra={"run_id": run_id, "fingerprint": GLOBAL_FP},
        )

    # flatten row for SUMMARY_CSV (keep consistent column names; mean_dist stored in nm)
    row = {
        "run_id": run_id,
        "timestamp": log["timestamp"],
        "tomo": config.get("tomo"),
        "model": config.get("model_name", config.get("model")),
        "preset": config.get("preset"),
        "mean_dist": mean_dist_nm,
        "precision": _safe_num(metrics.get("precision")),
        "recall": _safe_num(metrics.get("recall")),
        "f2": _safe_num(metrics.get("f2")),
        "runtime_s": _safe_num(metrics.get("runtime_s")),
        "gpu_mem_gb": _safe_num(metrics.get("gpu_mem_gb")),
        "gt_count": int(extra.get("gt_count", metrics.get("gt_count", 0) or 0)),
        "pred_count": int(extra.get("pred_count", metrics.get("pred_count", 0) or 0)),
        "matched": int(extra.get("matched", metrics.get("matched", 0) or 0)),
        "voxel_size_nm": extra.get("voxel_size_nm"),
        "recon_stats": extra.get("recon_stats"),             # e.g. {"threshold":..., "voxels_above":...}
        "loss_curve": extra.get("loss_curve"),               # store as list if provided
        "model_summary": model_summary,
        "fingerprint": env.get("fingerprint"),
        "model_path": model_path
    }

    # append to SUMMARY_CSV (idempotent check)
    checksum = _row_checksum(row)
    if _is_logged(checksum):
        # already logged -> return info
        return {"json": str(json_path) if json_path else None, "model": model_path, "summary": row, "skipped": True}

    df_row = pd.DataFrame([row])
    try:
        df_row.to_csv(SUMMARY_CSV, mode="a", index=False, header=not SUMMARY_CSV.exists())
    except Exception as e:
        warnings.warn(f"Failed to write summary CSV row: {e}")

    # link augmentation metadata if present
    if AUG_LOG.exists():
        try:
            df_aug = pd.read_csv(AUG_LOG)
            df_aug["linked_run_id"] = run_id
            df_aug.to_csv(AUG_LOG, index=False)
        except Exception:
            pass

    # mark checksum to avoid double logs
    _mark_logged(checksum)

    return {"json": str(json_path) if json_path else None, "model": model_path, "summary": row, "skipped": False}

# ---------------- Auto-log helper (ingest benchmark_metrics.csv) ----------------
def _coerce_row_types(sr: pd.Series) -> dict:
    """Return plain dict with cleaned values for logging."""
    d = sr.to_dict()
    # convert numpy types -> python
    for k, v in list(d.items()):
        if isinstance(v, (np.floating, np.integer)):
            d[k] = float(v)
        elif pd.isna(v):
            d[k] = None
        else:
            # leave strings as-is
            d[k] = v
    return d

def auto_log_from_benchmark_csv(benchmark_csv_path: Path = BENCHMARK_CSV, dry_run: bool = False):
    """
    Read benchmark_metrics.csv and auto-create JSON experiment logs for rows that
    haven't yet been logged to LOG_DIR. This implements your 'auto-log all runs' policy.
    """
    if not benchmark_csv_path.exists():
        return {"logged": 0, "skipped": 0, "reason": "benchmarks not found"}

    df = pd.read_csv(benchmark_csv_path)
    logged = 0
    skipped = 0
    for _, r in df.iterrows():
        row = _coerce_row_types(r)
        # construct minimal config & metrics & extra payload (safe)
        config = {
            "tomo": row.get("tomo"),
            "model": row.get("model"),
            "model_name": row.get("model"),
            "preset": row.get("preset"),
            "base_ch": row.get("base_ch"),
        }
        metrics = {
            "f2": row.get("f2"),
            "precision": row.get("precision"),
            "recall": row.get("recall"),
            # ensure canonical mean_dist_nm
            "mean_dist_nm": row.get("mean_dist_nm", row.get("mean_dist"))
        }

        extra = {
            "runtime_s": row.get("runtime_s"),
            "gpu_mem_gb": row.get("gpu_mem_gb"),
            "gt_count": int(row.get("gt_count") or 0),
            "pred_count": int(row.get("pred_count") or 0),
            "matched": int(row.get("matched") or 0),
            "voxel_size_nm": row.get("voxel_size_nm"),
            # recon_stats & loss_curve may exist as JSON-like strings; try to parse
        }

        # try parse recon_stats if present
        if "recon_stats" in row and pd.notna(row.get("recon_stats")):
            try:
                extra["recon_stats"] = json.loads(row.get("recon_stats"))
            except Exception:
                extra["recon_stats"] = str(row.get("recon_stats"))

        # loss_curve parsing
        if "loss_curve" in row and pd.notna(row.get("loss_curve")):
            try:
                extra["loss_curve"] = json.loads(row.get("loss_curve"))
            except Exception:
                extra["loss_curve"] = str(row.get("loss_curve"))

        # log it (no model object available here)
        res = log_experiment(config=config, metrics=metrics, model=None, optimizer=None, save_model=False, extra=extra)
        if res.get("skipped"):
            skipped += 1
        else:
            logged += 1

    return {"logged": logged, "skipped": skipped}

# ---- run an initial auto-sync at import time (idempotent) ----
try:
    auto_sync_res = auto_log_from_benchmark_csv()
except Exception:
    auto_sync_res = {"logged": 0, "skipped": 0, "reason": "error"}

# ---------------- Load all logs ----------------
def load_all_experiments(as_df: bool = True):
    rows = []
    for p in sorted(LOG_DIR.glob("exp_*.json")):
        try:
            with open(p, "r") as f:
                rows.append(json.load(f))
        except Exception:
            continue
    if as_df:
        flat = []
        for r in rows:
            flat.append({
                "run_id": r.get("run_id"),
                "timestamp": r.get("timestamp"),
                "tomo": r.get("config", {}).get("tomo"),
                "model": r.get("config", {}).get("model_name", r.get("config", {}).get("model")),
                "preset": r.get("config", {}).get("preset"),
                "mean_dist": r.get("metrics", {}).get("mean_dist_nm", r.get("metrics", {}).get("mean_dist")),
                "precision": r.get("metrics", {}).get("precision"),
                "recall": r.get("metrics", {}).get("recall"),
                "f2": r.get("metrics", {}).get("f2"),
                "gt_count": r.get("extra", {}).get("gt_count"),
                "pred_count": r.get("extra", {}).get("pred_count"),
                "matched": r.get("extra", {}).get("matched"),
                "voxel_size_nm": r.get("extra", {}).get("voxel_size_nm"),
            })
        return pd.DataFrame(flat)
    else:
        return rows

# ---------------- Streamlit UI ----------------
with st.expander("ğŸ—‚ï¸� Experiment Logs & Reproducibility (Part 7)", expanded=False):
    st.write("All experiment JSON logs are saved in:", str(LOG_DIR))
    col1, col2 = st.columns([2, 1])
    with col1:
        if SUMMARY_CSV.exists():
            try:
                df_summary = pd.read_csv(SUMMARY_CSV)
                st.dataframe(df_summary.sort_values("timestamp", ascending=False), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load summary CSV: {e}")
        else:
            st.info("No summary CSV yet â€” run experiments to populate logs.")

    with col2:
        st.markdown("### Sync & Controls")
        st.write(f"Auto-sync result (this session): logged={auto_sync_res.get('logged')}, skipped={auto_sync_res.get('skipped')}")
        if st.button("â†» Re-run auto-sync from benchmark_metrics.csv"):
            res = auto_log_from_benchmark_csv()
            st.success(f"Auto-sync completed: logged={res['logged']}, skipped={res['skipped']}")

        if st.button("ğŸ“� Open experiment folder (print)"):
            files = sorted([p.name for p in LOG_DIR.iterdir()])
            st.write(files[:200])

    # quick viz if summary csv present
    if SUMMARY_CSV.exists():
        try:
            df_summary = pd.read_csv(SUMMARY_CSV)
            if len(df_summary) > 1:
                fig = px.scatter(
                    df_summary,
                    x="runtime_s", y="f2", color="model",
                    size="precision", hover_data=["tomo", "recall", "mean_dist"],
                    title="Experiment Fâ‚‚ vs Runtime (per model)"
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    if st.button("Reload JSON Logs (preview)"):
        df_logs = load_all_experiments(as_df=True)
        if df_logs is None or df_logs.empty:
            st.info("No JSON logs found.")
        else:
            st.dataframe(df_logs.sort_values("timestamp", ascending=False), use_container_width=True)






# ============================================================
# âœ… Part 8 â€” Secure Streamlit + Ngrok Launcher (Updated)
# ============================================================
# Run this cell in a Kaggle Notebook AFTER saving /kaggle/working/app.py

!pip install pyngrok --quiet

import os, time, threading, torch, platform, subprocess
from pathlib import Path
from pyngrok import ngrok, conf

# ---------------- Paths ----------------
APP_PATH   = "/kaggle/working/app.py"
CACHE_DIR  = Path("/kaggle/working/cache_phase1")
LOG_DIR    = CACHE_DIR / "experiment_logs"
AUG_DIR    = CACHE_DIR / "augmented"

for d in [CACHE_DIR, LOG_DIR, AUG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------- Ngrok Authentication ----------------
# ğŸ”¥ Recommended: set this once per session before running this cell:
os.environ["NGROK_AUTH_TOKEN"] = "34vhUIr2n7HrpZoRZt0v16M7io5_4LoHcdPB4XQB9y5o5TY4L"

NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "").strip()

if not NGROK_AUTH_TOKEN:
    print("âš ï¸�  No ngrok token detected.")
    print('    Set it before running: os.environ["NGROK_AUTH_TOKEN"] = "your_token_here"')
else:
    conf.get_default().auth_token = NGROK_AUTH_TOKEN
    print("ğŸ”‘  Ngrok token loaded successfully.")

# ---------------- Kill Any Previous Runs ----------------
print("\nğŸ§¹ Cleaning previous Streamlit/ngrok sessions...")
os.system("pkill -f streamlit || true")
try:
    ngrok.kill()
except:
    pass

# ---------------- Environment Summary ----------------
print("\nğŸ§  Environment Summary:")
print(f"Python: {platform.python_version()}")
print(f"Torch:  {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} â€” {round(props.total_memory / 1e9, 2)} GB VRAM Ã— {torch.cuda.device_count()}")

# ---------------- Launch Streamlit via ngrok ----------------
print("\nğŸš€ Launching Streamlit...")

public_url = None
try:
    public_url = ngrok.connect(addr=8501)
    print(f"ğŸŒ� Public URL:\nğŸ‘‰ {public_url}")
except Exception as e:
    print("â�Œ Ngrok failed to connect:", e)
    print("Running Streamlit in local-only mode...")

def run_streamlit():
    cmd = [
        "streamlit", "run", APP_PATH,
        "--server.headless", "true",
        "--server.port", "8501",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "light"
    ]
    subprocess.run(cmd)

thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()

time.sleep(7)
print("\nâš™ï¸� Streamlit startingâ€¦ (wait 10â€“20 seconds)")
print("ğŸ“� Cache:", CACHE_DIR)
print("ğŸ“� Logs :", LOG_DIR)
print("ğŸ“� Aug  :", AUG_DIR)
print("âš ï¸� Keep this cell running â€” closing it stops the app.")





