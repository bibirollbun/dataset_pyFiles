!pip install streamlit


!pip install pyngrok --quiet
!pip install reportlab --quiet



%%writefile /kaggle/working/app.py
# ================================================================
# ğŸ§¬ Phase 1 â€“ Flagellar Motor 3D Analysis (Research-Ready Baseline)
# ================================================================

import os, gc, warnings, random, time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from PIL import Image as PILImage

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
CACHE_DIR = "/kaggle/working/cache_phase1"
os.makedirs(CACHE_DIR, exist_ok=True)

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
THRESH_STD = 0.4
DBSCAN_EPS = 2.0

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# ---------------------------------------------------------------
# âœ… 4. Streamlit Environment Setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Flagellar Motor 3D Analysis", layout="wide")
st.title("ğŸ§¬ Phase 1 â€“ Flagellar Motor 3D Analysis (Research-Ready Baseline)")

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
    """Load stack of 2D slices or .npy volume into a memmap."""
    tomo_p = Path(tomo_path)
    cache_file = Path(CACHE_DIR) / f"{tomo_p.stem}_raw.npy"

    # âœ… Case 1: cached memmap
    if cache_file.exists():
        try:
            return np.load(cache_file, mmap_mode="r")
        except Exception:
            cache_file.unlink(missing_ok=True)

    # âœ… Case 2: directory of slices
    if tomo_p.is_dir():
        slices = sorted(tomo_p.glob("*.jpg")) + sorted(tomo_p.glob("*.png"))
        if not slices:
            st.error(f"â�Œ No image slices found in {tomo_p}")
            return None
        vol = np.stack(
            [np.array(PILImage.open(s)) for s in slices],
            axis=0
        ).astype(np.float32)

        return save_memmap_and_return(cache_file, vol)

    # âœ… Case 3: direct .npy file
    if tomo_p.is_file() and tomo_p.suffix == ".npy":
        return np.load(tomo_p, mmap_mode="r")

    st.error(f"â�Œ Unsupported tomogram path: {tomo_p}")
    return None

# ---------------------------------------------------------------
# âœ… 5b. Cached Preprocessing (ALWAYS disk-backed)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def preprocess_volume_cached_memmap(volume_memmap):
    """
    Ensure volume is disk-backed memmap with valid filename.
    This guarantees downstream compatibility.
    """
    if volume_memmap is None:
        return None

    cache_path = Path(CACHE_DIR) / "preprocessed_volume.npy"

    # Case 1: already a memmap with filename
    if isinstance(volume_memmap, np.memmap) and volume_memmap.filename is not None:
        return volume_memmap

    # Case 2: memmap without filename OR ndarray â†’ force save
    arr = np.array(volume_memmap, dtype=np.float32)
    np.save(cache_path, arr)

    return np.load(cache_path, mmap_mode="r")

def ensure_disk_memmap(mm):
    if isinstance(mm, np.memmap) and mm.filename is not None:
        return mm
    path = Path(CACHE_DIR) / "forced_disk_memmap.npy"
    np.save(path, np.array(mm, dtype=np.float32))
    return np.load(path, mmap_mode="r")



# ---------------------------------------------------------------
# âœ… 6a. Tomogram Selection + Volume Preparation (CRITICAL)
# ---------------------------------------------------------------
tomos = find_tomograms(DATA_ROOT)
if not tomos:
    st.stop()

selected_tomo = st.selectbox(
    "Select Tomogram",
    tomos,
    format_func=lambda p: p.name if isinstance(p, Path) else str(p)
)

# ---- Load selected tomogram ----
raw_memmap = load_volume_from_jpegs_cached(selected_tomo)
if raw_memmap is None:
    st.error("â�Œ Failed to load tomogram.")
    st.stop()

# ---- Preprocess (cached, disk-backed) ----
vol_memmap = preprocess_volume_cached_memmap(raw_memmap)
if vol_memmap is None:
    st.error("â�Œ Preprocessing failed.")
    st.stop()

# ---- Final safety: enforce disk-backed memmap ----
vol_memmap = ensure_disk_memmap(vol_memmap)

# ---- HARD GUARD: filename must exist ----
if not isinstance(vol_memmap, np.memmap) or vol_memmap.filename is None:
    st.error("â�Œ Phase 1 volume is not disk-backed.")
    st.stop()

# ---- Safe to use filename now ----
proc_cache_path = Path(vol_memmap.filename)
st.success(f"âœ… Volume ready: {vol_memmap.shape}")



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
    """Adaptive marching_cubes mesh with pre-emptive downsampling to prevent OOM."""
    if measure is None:
        return None, None
        
    mm = np.load(volume_memmap, mmap_mode="r") if isinstance(volume_memmap, (str, Path)) else volume_memmap
    if mm is None or mm.size == 0:
        return None, None
        
    # âœ… FIX: Pre-emptive downsampling to prevent RAM blowout!
    # Cap processing at ~5 million voxels for safety on Kaggle
    target_voxels = 5_000_000 
    factor = int(np.ceil((mm.size / target_voxels) ** (1/3)))
    factor = max(1, factor)
    
    # Efficient strided read from disk (memmap) into RAM
    vol = np.array(mm[::factor, ::factor, ::factor], dtype=np.float32)
    
    if level is None:
        level = float(vol.mean() + 0.5 * vol.std())
        
    try:
        verts, faces, _, _ = measure.marching_cubes(vol, level=level)
        verts *= factor  # Scale coordinates back up to original volume space
    except Exception as e:
        warnings.warn(f"marching_cubes failed even after downsampling: {e}")
        return None, None
        
    # Safety cap for Plotly rendering speed in browser
    if verts.shape[0] > target_max_vertices:
        # If we subsample vertices, faces become invalid, so we drop faces and just return a point cloud
        faces = np.zeros((0, 3), dtype=int)
        idx = np.random.choice(verts.shape[0], size=target_max_vertices, replace=False)
        verts = verts[idx]
        
    return verts[:, [2, 1, 0]], faces

def cluster_points(coords: np.ndarray, eps=DBSCAN_EPS):
    """DBSCAN clustering (GPU-safe)."""
    if coords is None or coords.shape[0] == 0:
        return np.zeros((0, 3))
    try:
        clustering = DBSCAN(eps=eps, min_samples=2).fit(coords)
        return coords[clustering.labels_ >= 0]
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
        return (5.0, 1.0, 1.0)  # 5 nm along Z, 1 nm along Y/X
    return (1.0, 1.0, 1.0)  # fallback isotropic

def vox_to_nm(coords: np.ndarray, voxel_size_nm: Tuple[float, float, float]):
    """Convert voxel indices (z,y,x) â†’ nanometers."""
    if coords is None or coords.size == 0:
        return coords
    z_nm, y_nm, x_nm = voxel_size_nm
    # âœ… FIX: Match the scale array to the Z, Y, X coordinate format
    scale = np.array([z_nm, y_nm, x_nm], dtype=float)
    return coords * scale

def nm_to_vox(coords_nm: np.ndarray, voxel_size_nm: Tuple[float, float, float]):
    """Convert nanometer coordinates back to voxel units."""
    if coords_nm is None or coords_nm.size == 0:
        return coords_nm
    z_nm, y_nm, x_nm = voxel_size_nm
    # âœ… FIX: Match the scale array to the Z, Y, X coordinate format
    inv = np.array([1/z_nm, 1/y_nm, 1/x_nm], dtype=float)
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

# ---------------------------------------------------------------
# ğŸ”’ GLOBAL VOLUME GUARD â€” prevents Streamlit rerun crashes
# ---------------------------------------------------------------
if (
    "vol_memmap" not in globals()
    or vol_memmap is None
    or not isinstance(vol_memmap, np.memmap)
    or vol_memmap.filename is None
):
    st.warning("âš ï¸� No valid Phase 1 volume â€” downstream stages disabled.")
    st.stop()


# ================================================================
# Part 2 â€“ Tiny3DCNN + Confidence Filtering & Extended Training
# ================================================================

import os, gc, time, random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from scipy import ndimage

# ---------------- Constants ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_SIZE = 64
PATCH_STRIDE = 32
TRAIN_PATCHES_PER_EPOCH = 64
EPOCHS = 10   # âš—ï¸� extended training for smoother heatmaps
LEARNING_RATE = 1e-3
SIGMA_HEATMAP = 1.2
THRESH_STD = 0.4
DBSCAN_EPS = 2.0
MAX_POINTS = 15000
CACHE_DIR = "/kaggle/working/cache_phase1"
LOG_CSV = Path(CACHE_DIR) / "benchmark_runs.csv"

# ---------------- Seeding ----------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# ---------------- Model ----------------
class Tiny3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv3d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv3d(16, 1, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# ---------------- Metrics ----------------
def compute_metrics(pred_pts, gt_pts, radius=5.0):
    if len(gt_pts) == 0:
        return dict(precision=0, recall=0, f2=0, mean_dist=0, matched=0)
    if len(pred_pts) == 0:
        return dict(precision=0, recall=0, f2=0, mean_dist=np.nan, matched=0)
    nbrs = NearestNeighbors(n_neighbors=1).fit(pred_pts)
    dists, _ = nbrs.kneighbors(gt_pts)
    matched = dists[:, 0] <= radius
    tp = matched.sum(); fp = len(pred_pts) - tp; fn = len(gt_pts) - tp
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f2        = (1 + 2**2) * (precision * recall) / (4 * precision + recall + 1e-8)
    return dict(precision=float(precision), recall=float(recall),
                f2=float(f2), mean_dist=float(dists.mean()), matched=int(tp))

# ---------------- Patches ----------------
def extract_random_patches(volume_arr, target, n_patches, patch_size):
    """Extract patches, ensuring 50% of them contain ground truth targets if available."""
    Z, Y, X = volume_arr.shape
    patches_v, patches_t = [], []
    
    # ğŸ§  FIX: Use a relative threshold based on the actual Gaussian peak
    t_max = target.max()
    has_positives = False
    if t_max > 1e-5:
        # Grab voxels that are at least 10% of the maximum peak
        pos_z, pos_y, pos_x = np.nonzero(target > (t_max * 0.1))
        has_positives = len(pos_z) > 0

    for i in range(n_patches):
        # 50% chance to aggressively sample around a known target
        if has_positives and (i % 2 == 0):
            idx = np.random.randint(0, len(pos_z))
            zi = max(0, min(pos_z[idx] - patch_size // 2, Z - patch_size))
            yi = max(0, min(pos_y[idx] - patch_size // 2, Y - patch_size))
            xi = max(0, min(pos_x[idx] - patch_size // 2, X - patch_size))
        else:
            # Random background sampling
            zi = np.random.randint(0, max(1, Z - patch_size + 1))
            yi = np.random.randint(0, max(1, Y - patch_size + 1))
            xi = np.random.randint(0, max(1, X - patch_size + 1))
            
        patches_v.append(volume_arr[zi:zi+patch_size, yi:yi+patch_size, xi:xi+patch_size])
        patches_t.append(target[zi:zi+patch_size, yi:yi+patch_size, xi:xi+patch_size])
        
    return np.stack(patches_v), np.stack(patches_t)

# ---------------- Inference (with patch confidence) ----------------
def sliding_window_inference(volume_arr, model, patch_size=PATCH_SIZE, stride=PATCH_STRIDE):
    model.eval()
    Z, Y, X = volume_arr.shape
    accum, norm, conf_map = (np.zeros((Z, Y, X), np.float32) for _ in range(3))
    
    with torch.no_grad():
        for zi in range(0, Z - patch_size + 1, stride):
            for yi in range(0, Y - patch_size + 1, stride):
                for xi in range(0, X - patch_size + 1, stride):
                    patch = volume_arr[zi:zi+patch_size, yi:yi+patch_size, xi:xi+patch_size].astype(np.float32)
                    inp = torch.from_numpy(patch[None, None, ...]).to(DEVICE)
                    
                    with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                        out = model(inp)
                        
                    out_np = out[0, 0].cpu().numpy()
                    
                    # ğŸ§  FIX: Use .max() instead of .mean() for sparse 3D keypoints!
                    # We only care if there is a strong peak *somewhere* in the patch.
                    conf = float(out_np.max())
                    if conf < 0.1:  # skip patches that are completely empty
                        continue
                        
                    accum[zi:zi+patch_size, yi:yi+patch_size, xi:xi+patch_size] += out_np
                    norm [zi:zi+patch_size, yi:yi+patch_size, xi:xi+patch_size] += 1.0
                    conf_map[zi:zi+patch_size, yi:yi+patch_size, xi:xi+patch_size] += conf
                    
                    del inp, out, out_np
                    torch.cuda.empty_cache()
                    
    norm[norm == 0] = 1.0
    conf_map /= norm
    import gc
    gc.collect(); torch.cuda.empty_cache()
    return accum / norm, conf_map

# ---------------- Train + Infer + Log ----------------

def run_tiny_cnn(volume_memmap_or_arr, gt_points, epochs=EPOCHS, preset="balanced",
                 stride_factor=1.0):
    """
    Train Tiny3DCNN, infer on tomogram, and compute nm-scale benchmark metrics.
    Includes patch-wise confidence filtering for better recall balance.
    """
    t0_total = time.time()
    gpu_start_mem = torch.cuda.memory_allocated(DEVICE) if DEVICE == "cuda" else 0

    # ---- Load volume ----
    if isinstance(volume_memmap_or_arr, (str, Path)):
        mm = np.load(volume_memmap_or_arr, mmap_mode='r')
        vol = np.array(mm)
    else:
        vol = np.array(volume_memmap_or_arr)
    if vol.size == 0:
        st.warning("Empty volume â€” cannot run CNN.")
        return np.zeros((0,3)), {}

    # ---- Build target heatmap ----
    Z, Y, X = vol.shape
    target = np.zeros_like(vol, dtype=np.float32)
    for z, y, x in np.asarray(gt_points).astype(int):
        if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
            target[z, y, x] = 1.0
    target = ndimage.gaussian_filter(target, sigma=SIGMA_HEATMAP)

    # ---- Model setup ----
    model = Tiny3DCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    total_steps = epochs * TRAIN_PATCHES_PER_EPOCH
    pb = st.progress(0)
    losses = []

    # ---- Training ----
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
            losses.append(loss.item())
            pb.progress(int(100 * ((ep * TRAIN_PATCHES_PER_EPOCH + i + 1) / total_steps)))
            del inp, tgt, out, loss
        gc.collect(); torch.cuda.empty_cache()

    # ---- Inference + confidence ----
    st.info("Running inference with patch confidence filtering...")
    recon, conf_map = sliding_window_inference(
        vol, model, patch_size=PATCH_SIZE, stride=int(PATCH_STRIDE * stride_factor)
    )

    # ---- Adaptive thresholding (confidence-weighted) ----
    dynamic_thresh = (recon.mean() + THRESH_STD * recon.std()) * (1 - 0.15 * conf_map.mean())
    coords = np.array(np.nonzero(recon > dynamic_thresh)).T
    if coords.shape[0] > MAX_POINTS:
        coords = coords[np.random.choice(coords.shape[0], MAX_POINTS, replace=False)]
    coords_xyz = coords[:, [2, 1, 0]].astype(float)
    clustered = cluster_points(coords_xyz, eps=DBSCAN_EPS)

    # ---- Metrics (nm-normalized) ----
    voxel_size_nm = get_default_voxel_size("byu_motor")  # can adapt per dataset
    metrics = compute_metrics_nm(clustered, gt_points, voxel_size_nm, radius_nm=10.0)

    metrics.update({
        "train_time_s": round(time.time() - t0_total, 2),
        "gpu_mem_MB": round((torch.cuda.max_memory_allocated(DEVICE) - gpu_start_mem) / (1024**2), 2)
                      if DEVICE == "cuda" else 0,
        "mean_train_loss": float(np.mean(losses)),
        "pred_count": len(clustered),
        "gt_count": len(gt_points),
        "confidence_mean": float(conf_map.mean()),
        "voxel_size_nm_z": voxel_size_nm[0],
        "voxel_size_nm_y": voxel_size_nm[1],
        "voxel_size_nm_x": voxel_size_nm[2],
        "mean_dist_nm": metrics.get("mean_dist_nm", np.nan)
    })

    # ---- Save ----
    out_path = Path(CACHE_DIR) / f"pred_{int(np.random.rand()*1e9)}.npz"
    np.savez_compressed(out_path, points=clustered)
    metrics["saved_path"] = str(out_path)

    # ---- Log ----
    df_log = pd.DataFrame([metrics])
    if LOG_CSV.exists():
        prev = pd.read_csv(LOG_CSV)
        df_log = pd.concat([prev, df_log], ignore_index=True)
    df_log.to_csv(LOG_CSV, index=False)

    # ---- Report ----
    st.success(
        f"âœ… Done â€” Fâ‚‚={metrics['f2']:.3f}, "
        f"Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, "
        f"Mean Dist â‰ˆ {metrics['mean_dist_nm']:.2f} nm"
    )
    st.write("Mean Patch Confidence:", f"{metrics['confidence_mean']:.3f}")
    st.dataframe(pd.DataFrame([metrics]))
    return clustered, metrics

# ================================================================
# ğŸ“Š Î”Fâ‚‚ / Precision / Recall Visual Analytics (Preset + Model-aware)
# ================================================================

import plotly.express as px

st.subheader("ğŸ“ˆ Î”Fâ‚‚ / Precision / Recall Analysis Across Runs")

log_path = Path(CACHE_DIR) / "benchmark_runs.csv"
if log_path.exists():
    df = pd.read_csv(log_path)

    if len(df) >= 2:
        df = df.sort_values("train_time_s").reset_index(drop=True)

        # Compute percentage deltas
        df["Î”F2_%"] = df["f2"].pct_change() * 100
        df["Î”Recall_%"] = df["recall"].pct_change() * 100
        df["Î”Precision_%"] = df["precision"].pct_change() * 100

        # Fill in missing columns gracefully
        if "preset" not in df.columns:
            df["preset"] = "balanced"
        if "model" not in df.columns:
            df["model"] = "Tiny3DCNN"

        # ğŸ“Š Fâ‚‚ Trend by Preset
        fig_f2 = px.line(
            df, x=df.index, y="f2",
            color="preset", symbol="model", markers=True,
            title="Fâ‚‚ Evolution Across Runs (Grouped by Preset)",
            labels={"index": "Run #", "f2": "Fâ‚‚ Score"}
        )
        st.plotly_chart(fig_f2, use_container_width=True)

        # ğŸ“Š Precision vs Recall Trend
        fig_pr = px.line(
            df, x=df.index, y=["precision", "recall"],
            color_discrete_sequence=["#007bff", "#ff5733"],
            title="Precision vs Recall Progression",
            labels={"index": "Run #", "value": "Score"},
            markers=True
        )
        st.plotly_chart(fig_pr, use_container_width=True)

        # ğŸ“Š Fâ‚‚ Improvement Heatmap
        fig_delta = px.bar(
            df, x=df.index, y="Î”F2_%", color="preset",
            title="Î”Fâ‚‚ % Change Between Consecutive Runs",
            labels={"index": "Run #", "Î”F2_%": "Î”Fâ‚‚ (%)"},
            text=df["Î”F2_%"].round(2)
        )
        fig_delta.update_traces(textposition="outside")
        st.plotly_chart(fig_delta, use_container_width=True)

        # Summary Table
        st.write("ğŸ“‹ Summary of Improvement Metrics:")
        delta_cols = ["model", "preset", "f2", "precision", "recall", "Î”F2_%", "Î”Recall_%", "Î”Precision_%", "train_time_s", "pred_count"]
        st.dataframe(df[delta_cols].round(3))

        # Best Run Highlight
        best_run = df.loc[df["f2"].idxmax()]
        st.success(
            f"ğŸ�† Best Fâ‚‚ = {best_run['f2']:.3f} "
            f"| Recall = {best_run['recall']:.3f} "
            f"| Precision = {best_run['precision']:.3f} "
            f"| Preset: {best_run.get('preset','balanced')} "
            f"| Model: {best_run.get('model','Tiny3DCNN')}"
        )

    else:
        st.info("Need at least 2 logged runs to visualize Î”Fâ‚‚ trends.")
else:
    st.info("No benchmark_runs.csv found â€” run the model at least twice to generate Fâ‚‚ deltas.")
#************************
## Part 3 â€“ Ablations
#************************
import itertools, json, time, gc, torch
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# ---------------- Sidebar UI ----------------
st.sidebar.header("âš™ï¸� Ablation Controller")

with st.sidebar.expander("Sweep Settings", expanded=True):
    patch_sizes = st.multiselect("Patch sizes (voxels)", [32, 48, 64, 80], default=[64])
    stride_factors = st.multiselect("Stride factors (speed tradeoff)", [0.75, 1.0, 1.25, 1.5], default=[1.0])
    dbscan_eps_list = st.multiselect("DBSCAN eps (voxels)", [1.0, 1.5, 2.0, 3.0], default=[1.5])
    presets = st.multiselect("Training preset", ["fast", "balanced", "thorough"], default=["balanced"])
    max_runs = st.number_input("Max total runs (safety cap)", min_value=1, max_value=200, value=10)
    max_runtime_per_run = st.slider("Max runtime per run (minutes)", 5, 120, 30)
    max_total_runtime = st.number_input("Max total runtime (minutes)", 10, 480, 120)

model_choice = st.sidebar.selectbox("Select Architecture", options=["tinycnn"], index=0)

run_button = st.sidebar.button("ğŸš€ Run Ablation Sweep")
save_button = st.sidebar.button("ğŸ’¾ Save Current Results")

# ---------------- Session State ----------------
if "ablation_results" not in st.session_state:
    st.session_state["ablation_results"] = []

# ---------------- Helpers ----------------
def _make_grid(patches, strides, eps_list, presets_list, cap):
    combos = list(itertools.product(patches, strides, eps_list, presets_list))
    return combos[:cap]

def _save_results(results, prefix):
    if not results:
        return None
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csvp = Path(CACHE_DIR) / f"{prefix}_{ts}.csv"
    jsonp = Path(CACHE_DIR) / f"{prefix}_{ts}.json"
    pd.DataFrame(results).to_csv(csvp, index=False)
    with open(jsonp, "w") as f:
        json.dump(results, f, indent=2)
    return str(csvp), str(jsonp)

def _run_single_combo(volume_memmap, gt_points, patch_size, stride_factor, dbscan_eps, preset, model_name, max_runtime_s):
    t0 = time.time()
    global DBSCAN_EPS
    prev_eps = DBSCAN_EPS
    DBSCAN_EPS = float(dbscan_eps)

    try:
        _, metrics = run_model_pipeline(
            volume_memmap_or_arr=volume_memmap,
            gt_points=gt_points,
            model_name=model_name,
            epochs=EPOCHS,
            preset=preset,
            patch_size=int(patch_size),
            stride_factor=float(stride_factor)
        )
    finally:
        DBSCAN_EPS = prev_eps

    if time.time() - t0 > max_runtime_s:
        raise TimeoutError("Run exceeded time budget")

    if gt_points.shape[0] == 0:
        metrics = {"precision": np.nan, "recall": np.nan, "f2": np.nan, "mean_dist_nm": np.nan, "note": "No GT available"}

    metrics.update({
        "model": model_name,
        "patch_size": int(patch_size),
        "stride_factor": float(stride_factor),
        "dbscan_eps": float(dbscan_eps),
        "preset": preset,
        "run_time_s": round(time.time() - t0, 2),
        "timestamp": datetime.now().isoformat(),
    })

    gc.collect()
    torch.cuda.empty_cache()
    return metrics

# ---------------- Results ----------------
st.header("ğŸ“Š Ablation Results Summary (nm-normalized)")
if st.session_state["ablation_results"]:
    df = pd.DataFrame(st.session_state["ablation_results"])
    if "f2" not in df.columns:
        st.info("No successful runs yet â€” metrics not available.")
        st.dataframe(df)
    else:
        if "mean_dist_nm" in df.columns:
            df["weighted_f2"] = df["f2"] / (1 + df["mean_dist_nm"].clip(lower=1))
        else:
            df["weighted_f2"] = df["f2"]
        df = df.sort_values("weighted_f2", ascending=False)
        st.dataframe(df)
        fig = px.bar(df.head(10), y="weighted_f2", hover_data=["model", "patch_size", "stride_factor", "dbscan_eps", "preset", "precision", "recall"], title="Top Ablation Runs by Weighted Fâ‚‚")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No ablation results yet.")

# ---------------- Manual Save ----------------
if save_button and st.session_state["ablation_results"]:
    paths = _save_results(st.session_state["ablation_results"], "ablation_manual")
    if paths:
        st.success(f"Saved to {paths[0]} and {paths[1]}")


# ===============================================================
# Part 4 â€” Multi-Architecture Integration (T4-Optimized, Unified)
# ===============================================================
import math
import torch.nn.functional as F

# ---- Lightweight Squeeze-Excite (cheap attention) ----
class ConvSEBlock3D(nn.Module):
    """Small conv block with squeeze-excite channel attention (3D)."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        # SE
        self.se_fc1 = nn.Conv3d(out_ch, max(1, out_ch // 8), kernel_size=1)
        self.se_fc2 = nn.Conv3d(max(1, out_ch // 8), out_ch, kernel_size=1)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        se = out.mean(dim=(2, 3, 4), keepdim=True)
        se = F.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        return out * se


# ---- Tiny 3D CNN (baseline, fastest) ----
class Tiny3DCNN(nn.Module):
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, 3, padding=1), nn.ReLU(),
            nn.Conv3d(base_ch, base_ch * 2, 3, padding=1), nn.ReLU(),
            nn.Conv3d(base_ch * 2, base_ch, 3, padding=1), nn.ReLU(),
            nn.Conv3d(base_ch, 1, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)


# ---- Small 3D U-Net (memory-conscious) ----
class UNet3D_small(nn.Module):
    """Depth = 3; base_ch controls width (8â€“32 recommended)."""
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        self.enc1 = nn.Sequential(ConvSEBlock3D(in_ch, base_ch),
                                  ConvSEBlock3D(base_ch, base_ch))
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = nn.Sequential(ConvSEBlock3D(base_ch, base_ch * 2),
                                  ConvSEBlock3D(base_ch * 2, base_ch * 2))
        self.pool2 = nn.MaxPool3d(2)
        self.bottleneck = nn.Sequential(ConvSEBlock3D(base_ch * 2, base_ch * 4),
                                        ConvSEBlock3D(base_ch * 4, base_ch * 4))
        # decoder
        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = nn.Sequential(ConvSEBlock3D(base_ch * 4, base_ch * 2),
                                  ConvSEBlock3D(base_ch * 2, base_ch * 2))
        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = nn.Sequential(ConvSEBlock3D(base_ch * 2, base_ch),
                                  ConvSEBlock3D(base_ch, base_ch))
        self.final = nn.Conv3d(base_ch, 1, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out_act(self.final(d1))


# ---- Small 3D ResNet ----
class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = self.relu(out + identity)
        return out


class ResNet3D_small(nn.Module):
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        self.inp = nn.Conv3d(in_ch, base_ch, 3, padding=1)
        self.l1 = ResBlock3D(base_ch, base_ch)
        self.pool1 = nn.MaxPool3d(2)
        self.l2 = ResBlock3D(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.l3 = ResBlock3D(base_ch * 2, base_ch * 4)
        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, 2, stride=2)
        self.outc = nn.Conv3d(base_ch, 1, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x0 = F.relu(self.inp(x))
        x1 = self.l1(x0)
        x2 = self.l2(self.pool1(x1))
        b = self.l3(self.pool2(x2))
        out = self.up1(self.up2(b) + x2) + x1
        return self.act(self.outc(out))


# ---- Model factory ----
def get_model(name: str = "tinycnn", in_ch: int = 1, base_ch: int = 16, **kwargs):
    """Instantiate architecture by name."""
    n = name.lower()
    if n in ("tiny", "tinycnn"):
        return Tiny3DCNN(in_ch, base_ch)
    if n in ("unet", "unet3d"):
        return UNet3D_small(in_ch, base_ch)
    if n in ("resnet", "resnet3d"):
        return ResNet3D_small(in_ch, base_ch)
    raise ValueError(f"Unknown model name: {name}")


# ---- Unified runner (ready for ablation + Streamlit) ----
def run_model_pipeline(
    volume_memmap_or_arr,
    gt_points,
    model: nn.Module = None,
    model_name: str = "tinycnn",
    model_kwargs: dict = None,
    epochs: int = EPOCHS,
    preset: str = "balanced",
    patch_size: int = None,
    stride_factor: float = 1.0,
    base_ch: int = None,
):
    """
    Unified training/inference for all architectures.
    Returns: clustered_points (Nx3 float), metrics (dict)
    - Accepts either `model` (nn.Module) or `model_name` + `model_kwargs`.
    - patch_size overrides global PATCH_SIZE if provided.
    - stride_factor scales PATCH_STRIDE during inference.
    - base_ch overrides model width if provided.
    """
    import time, traceback, gc
    t0 = time.time()
    model_kwargs = dict(model_kwargs or {})
    saved_path = None

    # apply local patch/stride if provided (restore at end)
    global PATCH_SIZE as _PATCH_SIZE_GLOBAL, PATCH_STRIDE as _PATCH_STRIDE_GLOBAL
    prev_patch, prev_stride = PATCH_SIZE, PATCH_STRIDE
    if patch_size is not None:
        PATCH_SIZE = int(patch_size)
        PATCH_STRIDE = max(1, int(PATCH_SIZE // 2))
    else:
        patch_size = PATCH_SIZE

    try:
        # Preset tuning
        p = (preset or "balanced").lower()
        if p == "fast":
            epochs = max(1, int(epochs * 0.4))
            patches_per_epoch = max(8, TRAIN_PATCHES_PER_EPOCH // 2)
        elif p in ("accurate", "thorough"):
            epochs = min(20, int(epochs * 1.5))
            patches_per_epoch = max(TRAIN_PATCHES_PER_EPOCH, TRAIN_PATCHES_PER_EPOCH * 2)
        else:
            patches_per_epoch = TRAIN_PATCHES_PER_EPOCH

        # Load volume
        if isinstance(volume_memmap_or_arr, (str, Path)):
            mm = np.load(volume_memmap_or_arr, mmap_mode="r")
            vol = np.array(mm)
        else:
            vol = np.array(volume_memmap_or_arr)
        if vol.size == 0:
            st.warning("Empty volume â€” cannot run model.")
            return np.zeros((0, 3)), {}

        # Build target heatmap
        Z, Y, X = vol.shape
        target = np.zeros_like(vol, dtype=np.float32)
        if gt_points is not None and len(gt_points) > 0:
            for z, y, x in np.asarray(gt_points).astype(int):
                if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
                    target[z, y, x] = 1.0
            from scipy import ndimage
            target = ndimage.gaussian_filter(target, sigma=SIGMA_HEATMAP)
        else:
            target = np.zeros_like(vol, dtype=np.float32)

        # Instantiate model (if not provided)
        if model is None:
            if base_ch is not None:
                model_kwargs.setdefault("base_ch", int(base_ch))
            model = get_model(model_name, in_ch=1, **model_kwargs).to(DEVICE)
        else:
            model = model.to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.MSELoss()
        
        # Determine scaler usage depending on the PyTorch version / environment
        try:
            scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

        total_steps = max(1, epochs * patches_per_epoch)
        step = 0
        pb = st.progress(0)
        losses = []

        st.info(f"Training {model_name.upper()} ({p}) for {epochs} epoch(s) â€” patches/epoch={patches_per_epoch}")

        # Training loop (memory-friendly, patch-wise)
        for ep in range(epochs):
            patches_v, patches_t = extract_random_patches(vol, target, patches_per_epoch, PATCH_SIZE)
            model.train()
            for i in range(patches_v.shape[0]):
                inp = torch.from_numpy(patches_v[i:i+1]).unsqueeze(1).to(DEVICE)
                tgt = torch.from_numpy(patches_t[i:i+1]).unsqueeze(1).to(DEVICE)
                optimizer.zero_grad()
                
                try:
                    with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                        out = model(inp)
                        loss = loss_fn(out, tgt)
                except Exception:
                    with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                        out = model(inp)
                        loss = loss_fn(out, tgt)
                        
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu().numpy()))
                step += 1
                if step % 5 == 0:
                    pb.progress(int(100 * step / total_steps))
                # free ASAP
                del inp, tgt, out, loss
                torch.cuda.empty_cache()
            gc.collect(); torch.cuda.empty_cache()

        # Inference (sliding window)
        st.info("Running sliding-window inference...")
        recon, conf_map = sliding_window_inference(vol, model, patch_size=PATCH_SIZE, stride=int(PATCH_STRIDE * stride_factor))

        # Threshold + clustering (FIXED: Coordinate Swap & Sigmoid Threshold)
        # 1. Lowered hard minimum to 0.15 so we don't accidentally erase soft peaks
        dynamic_thresh = (recon.mean() + 3.0 * recon.std()) * (1 - 0.15 * conf_map.mean())
        thresh = max(0.15, dynamic_thresh) 
        
        coords = np.array(np.nonzero(recon > thresh)).T
        
        if coords.shape[0] > MAX_POINTS:
            idx = np.random.choice(coords.shape[0], size=MAX_POINTS, replace=False)
            coords = coords[idx]
            
        coords_zyx = coords.astype(float)
        
        # ğŸ§  THE MASSIVE FIX: Extract CENTROIDS, not thousands of raw voxels!
        if coords_zyx.shape[0] > 0:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=2).fit(coords_zyx)
            labels = clustering.labels_
            centroids = []
            # Calculate the mean (center) of each distinct cluster
            for k in set(labels):
                if k != -1:  # Ignore unclustered noise (-1)
                    centroids.append(coords_zyx[labels == k].mean(axis=0))
            clustered = np.vstack(centroids) if centroids else np.zeros((0, 3))
        else:
            clustered = np.zeros((0, 3))

        # ğŸ§  METRICS FIX: Safely use compute_metrics_nm to avoid the tuple crash
        try:
            if "get_default_voxel_size" in globals() and "compute_metrics_nm" in globals():
                vx = get_default_voxel_size("byu_motor")
                # âœ… FIX: Relax the strictness radius to 40nm to allow for realistic motor volume sizes
                metrics = compute_metrics_nm(clustered, gt_points, voxel_size_nm=vx, radius_nm=40.0)
            else:
                metrics = compute_metrics(clustered, gt_points, radius=5.0)
        except Exception as e:
            st.warning(f"Metric computation warning: {e}")
            metrics = compute_metrics(clustered, gt_points, radius=5.0)
            metrics["mean_dist_nm"] = None

        metrics.update({
            "model_name": model_name if model_name else model.__class__.__name__,
            "patch_size": int(PATCH_SIZE),
            "stride_factor": float(stride_factor),
            "patches_per_epoch": int(patches_per_epoch),
            "epochs": int(epochs),
            "train_steps": int(total_steps),
            "train_loss_mean": float(np.mean(losses)) if losses else None,
            "train_time_s": round(time.time() - t0, 2),
            "gpu_mem_MB": round(torch.cuda.max_memory_allocated() / (1024**2), 2) if DEVICE == "cuda" else 0.0
        })

        # Save predictions (compressed)
        try:
            out_path = Path(CACHE_DIR) / f"pred_{model_name}_{int(np.random.rand()*1e9)}.npz"
            np.savez_compressed(out_path, points=clustered)
            saved_path = str(out_path)
            metrics["saved_path"] = saved_path
        except Exception:
            metrics["saved_path"] = None

        st.success(f"âœ… {model_name.upper()} done in {time.time() - t0:.1f}s â€” {len(clustered)} pts")
        gc.collect(); torch.cuda.empty_cache()
        return clustered, metrics

    except Exception as e:
        # return empty results + error info
        tb = traceback.format_exc()
        st.error(f"run_model_pipeline failed: {e}")
        return np.zeros((0, 3)), {"error": str(e), "traceback": tb}

    finally:
        # restore globals
        PATCH_SIZE = prev_patch
        PATCH_STRIDE = prev_stride
        gc.collect(); torch.cuda.empty_cache()

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
    # If we already saved it specifically for this tomogram, load it safely
    vol_memmap = np.load(proc_cache_path, mmap_mode="r")
else:
    # 1. Process it (Streamlit cache might strip the memmap properties here)
    processed_arr = preprocess_volume_cached_memmap(raw_memmap)
    
    # 2. Explicitly save the array to our known path to guarantee it is on disk
    np.save(proc_cache_path, np.array(processed_arr, dtype=np.float32))
    
    # 3. Reload it directly with NumPy so it is a 100% valid memmap
    vol_memmap = np.load(proc_cache_path, mmap_mode="r")

st.write(f"Volume shape: {vol_memmap.shape}, memmap: {proc_cache_path}")

# ---- Load GT ----
gt_points = np.zeros((0, 3))
gt_csv = Path(DATA_ROOT) / GT_CSV
if gt_csv.exists():
    try:
        df = pd.read_csv(gt_csv)
        sel = df[df["tomo_id"] == selected_name]
        # âœ… FIX: Load in 0, 1, 2 order to perfectly match numpy's Z, Y, X shape
        if {"Motor axis 0", "Motor axis 1", "Motor axis 2"}.issubset(sel.columns):
            gt_points = sel[["Motor axis 0", "Motor axis 1", "Motor axis 2"]].values.astype(float)
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
            "matched": matched,
            "gt_count": int(gt_count),
            "pred_count": len(recon_points),
            "mean_dist": float(np.mean(dists)) if len(dists) else 0.0,
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
        st.dataframe(dfm.sort_values("timestamp", ascending=False), use_container_width=True)

        if len(dfm) > 1:
            # ğŸ§  Ensure 'mean_dist_nm' column exists
            if "mean_dist_nm" not in dfm.columns:
                if "mean_dist" in dfm.columns:
                    dfm["mean_dist_nm"] = dfm["mean_dist"]
                else:
                    dfm["mean_dist_nm"] = 0.0

            # âœ… Dynamically choose hover columns that exist
            hover_cols = [c for c in ["tomo", "pred_count", "f2", "precision", "recall"]
                          if c in dfm.columns]

            fig = px.scatter_3d(
                dfm,
                x="runtime_s",
                y="mean_dist_nm",
                z="gpu_mem_gb",
                color="model",
                symbol="preset",
                size="matched",
                hover_data=hover_cols,
                title="Runtime vs Accuracy (nm) vs GPU Usage",
            )
            st.plotly_chart(fig, use_container_width=True)
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
    st.plotly_chart(fig3d, use_container_width=True)

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
    st.plotly_chart(fig_slice, use_container_width=True)

# ------------------------------------------------
# TAB 4 â€” Report Export (PDF with Visualization & Fâ‚‚ Chart, Safe)
# ------------------------------------------------
with tab_report:
    st.subheader("ğŸ“„ Generate Summary PDF (with Visualization & Fâ‚‚ Chart)")
    metrics_path = CACHE_DIR / "benchmark_metrics.csv"
    if metrics_path.exists():
        dfm = pd.read_csv(metrics_path)

        # ğŸ§® Auto-compute Fâ‚‚ if missing and precision/recall available
        if "f2" not in dfm.columns and {"precision", "recall"}.issubset(dfm.columns):
            dfm["f2"] = np.where(
                (dfm["precision"] + dfm["recall"]) > 0,
                5 * (dfm["precision"] * dfm["recall"]) /
                (4 * dfm["precision"] + dfm["recall"]),
                0
            )

        # ğŸ”¬ Add placeholder mean_dist_nm if absent
        if "mean_dist_nm" not in dfm.columns:
            dfm["mean_dist_nm"] = dfm.get("mean_dist", np.zeros(len(dfm)))

        # ğŸ“Š Generate Fâ‚‚ vs mean distance chart
        chart_path = CACHE_DIR / "runtime_vs_f2_chart.png"
        try:
            if "f2" in dfm.columns and dfm["f2"].notna().any():
                fig_chart = px.scatter(
                    dfm, x="mean_dist_nm", y="f2",
                    color="model", symbol="preset", size="matched",
                    title="Fâ‚‚ vs Mean Distance (nm)",
                    labels={"mean_dist_nm": "Mean Distance (nm)", "f2": "Fâ‚‚ Score"}
                )
                import plotly.io as pio
                try:
                    pio.write_image(fig_chart, str(chart_path),
                                    format="png", width=800, height=600, scale=2)
                    st.image(str(chart_path),
                             caption="Fâ‚‚ vs Mean Distance Chart (preview)",
                             use_container_width=True)
                except Exception as e:
                    st.warning(f"Chart image export skipped (kaleido missing?): {e}")
            else:
                st.info("No valid Fâ‚‚ data available for chart rendering.")
        except Exception as e:
            st.warning(f"Chart export skipped: {e}")

        # ğŸ§¾ Assemble PDF
        from reportlab.platypus import Image  # lazy import for safety
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter)
        styles = getSampleStyleSheet()

        best_f2 = dfm["f2"].max() if "f2" in dfm.columns else 0.0
        best_md = dfm["mean_dist_nm"].min() if "mean_dist_nm" in dfm.columns else 0.0

        story = [
            Paragraph("<b>3D Cryo-ET Localization Benchmark Summary</b>", styles["Title"]),
            Spacer(1, 12),
            Paragraph(f"System: {platform.node()} | {platform.processor()}", styles["Normal"]),
            Paragraph(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
            Spacer(1, 12),
            Paragraph(f"Total Runs: {len(dfm)}", styles["Normal"]),
            Paragraph(f"Best Fâ‚‚: {best_f2:.3f}", styles["Normal"]),
            Paragraph(f"Lowest Mean Distance (nm): {best_md:.2f}", styles["Normal"]),
            Spacer(1, 12)
        ]

        # ğŸ–¼ Visualization snapshot & chart
        snap_img = CACHE_DIR / "last_viz_snapshot.png"
        if snap_img.exists():
            story += [
                Paragraph("<b>3D Visualization Snapshot</b>", styles["Heading2"]),
                Spacer(1, 6),
                Image(str(snap_img), width=450, height=350),
                Spacer(1, 12)
            ]
        if chart_path.exists():
            story += [
                Paragraph("<b>Fâ‚‚ vs Mean Distance Chart</b>", styles["Heading2"]),
                Spacer(1, 6),
                Image(str(chart_path), width=400, height=300),
                Spacer(1, 12)
            ]

        # ğŸ“‹ Metrics table
        table_data = [list(dfm.columns)] + dfm.astype(str).values.tolist()
        from reportlab.platypus import Table, TableStyle
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
        st.download_button("â¬‡ï¸� Download PDF Report",
                           pdf_bytes,
                           file_name="CryoET_Benchmark_Report.pdf")
    else:
        st.info("No metrics logged yet â€” run a model to generate report.")


!sed -i 's/global PATCH_SIZE as _PATCH_SIZE_GLOBAL, PATCH_STRIDE as _PATCH_STRIDE_GLOBAL/global PATCH_SIZE, PATCH_STRIDE/' /kaggle/working/app.py



!grep -n "global PATCH_SIZE" /kaggle/working/app.py



# ============================================================
# Part 6 â€” Data Augmentation & Semi-Synthetic Expansion
# (CPU-safe, nm-aware, with logging & visualization)
# ============================================================

import os, random, gc
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter, zoom
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- ensure cache & aug directories exist ---
CACHE_DIR = Path("/kaggle/working/cache_phase1")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
AUG_DIR = CACHE_DIR / "augmented"
AUG_DIR.mkdir(parents=True, exist_ok=True)
AUG_LOG = CACHE_DIR / "augmentation_log.csv"

# ------------------------------------------------------------
# ğŸ”¬ Semi-Synthetic Augmentation Function
# ------------------------------------------------------------
def generate_augmented_variants(volume_memmap_or_arr, name_prefix: str, n_variants: int = 3):
    """
    Generate semi-synthetic noisy / blurred / rescaled variants
    to simulate realistic Cryo-ET tomogram variability.
    Returns list of .npy paths + summary metadata.
    """
    if isinstance(volume_memmap_or_arr, (str, Path)):
        mm = np.load(volume_memmap_or_arr, mmap_mode='r')
        base = np.array(mm, dtype=np.float32)
    else:
        base = np.array(volume_memmap_or_arr, dtype=np.float32)

    variants, meta = [], []
    base_mean, base_std = float(base.mean()), float(base.std())

    for i in range(n_variants):
        vol = base.copy()

        # --- 1ï¸�âƒ£ Add Gaussian + optional Poisson noise ---
        sigma_n = np.random.uniform(0.01, 0.1) * vol.std()
        gauss = np.random.normal(0, sigma_n, vol.shape).astype(np.float32)
        vol += gauss
        if np.random.rand() < 0.5:
            lam = np.random.uniform(0.5, 2.0)
            vol = np.random.poisson(np.clip(vol, 0, None) * lam).astype(np.float32) / max(lam, 1e-6)

        # --- 2ï¸�âƒ£ Defocus blur ---
        blur_sigma = np.random.uniform(0.4, 1.5)
        vol = gaussian_filter(vol, sigma=blur_sigma)

        # --- 3ï¸�âƒ£ Random voxel scaling / downsampling ---
        if np.random.rand() < 0.6:
            scale = np.random.uniform(0.6, 1.0)
            vol = zoom(vol, scale, order=1)
            pad = [(max(0, b - v) // 2, max(0, b - v) - max(0, b - v) // 2)
                   for b, v in zip(base.shape, vol.shape)]
            vol = np.pad(vol, pad, mode='reflect')
            vol = vol[:base.shape[0], :base.shape[1], :base.shape[2]]

        # --- 4ï¸�âƒ£ Contrast normalization ---
        vol = (vol - vol.mean()) / (vol.std() + 1e-8)
        vol = np.clip(vol, -3, 3)

        # --- ğŸ§® Compute SNR & save variant ---
        snr = float(20 * np.log10((base_std + 1e-8) / (sigma_n + 1e-8)))
        out_path = AUG_DIR / f"{name_prefix}_aug{i+1}.npy"
        np.save(out_path, vol.astype(np.float32))
        variants.append(out_path)

        meta.append({
            "variant": f"{name_prefix}_aug{i+1}",
            "gaussian_sigma": round(sigma_n, 4),
            "blur_sigma": round(blur_sigma, 3),
            "scale_factor": round(scale if 'scale' in locals() else 1.0, 3),
            "snr_db": round(snr, 2),
            "mean": round(float(vol.mean()), 4),
            "std": round(float(vol.std()), 4)
        })

        gc.collect()

    # --- Save metadata log ---
    df_meta = pd.DataFrame(meta)
    if AUG_LOG.exists():
        prev = pd.read_csv(AUG_LOG)
        df_meta = pd.concat([prev, df_meta], ignore_index=True)
    df_meta.to_csv(AUG_LOG, index=False)

    return variants, df_meta


# ------------------------------------------------------------
# ğŸ§­ Streamlit Integration
# ------------------------------------------------------------
with st.expander("ğŸ§¬ Semi-Synthetic Data Augmentation", expanded=False):
    st.markdown("""
    Create semi-synthetic tomograms to mimic realistic Cryo-ET variability.
    Each variant introduces random blur, noise, voxel scaling, and normalization
    to emulate electron dose, focus drift, and reconstruction artifacts.
    """)

    n_aug = st.slider("Number of augmented variants", 1, 6, 3)
    if st.button("Generate Augmented Tomograms"):
        with st.spinner("Generating semi-synthetic tomograms..."):
            aug_paths, meta_df = generate_augmented_variants(proc_cache_path, selected_name, n_variants=n_aug)
        st.success(f"âœ… {len(aug_paths)} augmented tomograms saved to {AUG_DIR}")
        st.dataframe(meta_df)

        # ğŸ–¼ Preview middle slice & histogram
        rand_path = random.choice(aug_paths)
        rand_aug = np.load(rand_path, mmap_mode='r')
        mid_z = rand_aug.shape[0] // 2

        col1, col2 = st.columns(2)
        with col1:
            st.image(rand_aug[mid_z, :, :], caption=f"Preview of {rand_path.name}", use_container_width=True)
        with col2:
            fig, ax = plt.subplots()
            ax.hist(rand_aug.flatten(), bins=50, color='gray')
            ax.set_title("Voxel Intensity Distribution")
            st.pyplot(fig)

        st.caption("Tip: You can now re-run your TinyCNN or U-Net using these augmented volumes for data diversity.")

# ------------------------------------------------------------
# Optional Hook: Direct Augment â†’ Train Loop (for automation)
# ------------------------------------------------------------
if st.button("ğŸ§  Augment & Train Automatically (TinyCNN baseline)"):
    st.info("Running augmentation + model pipeline...")
    aug_paths, _ = generate_augmented_variants(proc_cache_path, selected_name, n_variants=n_aug)
    for path in aug_paths:
        st.write(f"Training on {path.name} ...")
        recon, metrics = run_model_pipeline(
            path, gt_points,
            model_name="tinycnn",
            epochs=3,
            preset="fast"
        )
        st.success(f"Done: Fâ‚‚={metrics.get('f2',0):.3f} | MeanDist_nm={metrics.get('mean_dist_nm',0):.2f}")



# ================================================================
# Part 7 â€” Reproducibility Tracker, Seed Control & Auto-Logging
# ================================================================

import os, json, random, socket, subprocess, warnings, hashlib
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
LOG_DIR = CACHE_DIR / "experiment_logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_CSV = CACHE_DIR / "experiment_summary.csv"
AUG_LOG = CACHE_DIR / "augmentation_log.csv"

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

    # create reproducibility fingerprint
    state = (
        str(seed)
        + str(np.random.get_state())
        + str(random.getstate()[1][:10])
        + str(torch.get_rng_state()[:10].tolist())
    )
    fingerprint = hashlib.sha1(state.encode()).hexdigest()[:12]
    return seed, fingerprint


# âœ… default seed + fingerprint
GLOBAL_SEED, GLOBAL_FP = set_seed(42)

# ---------------- Environment capture ----------------
def _get_env_info():
    """Collect environment + package version info."""
    info = {
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "seed": GLOBAL_SEED,
        "fingerprint": GLOBAL_FP,
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
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = {
            "name": props.name,
            "total_mem_GB": round(props.total_memory / 1e9, 2),
            "count": torch.cuda.device_count(),
        }
    else:
        info["gpu"] = None

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        info["git_commit"] = commit
    except Exception:
        info["git_commit"] = None
    return info


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
    """Load checkpoint into model/optimizer."""
    chk = torch.load(path, map_location=(map_location or DEVICE))
    if model is not None and "model_state" in chk:
        model.load_state_dict(chk["model_state"])
    if optimizer is not None and "optimizer_state" in chk:
        optimizer.load_state_dict(chk["optimizer_state"])
    return chk


# ---------------- Experiment logging ----------------
def _make_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{np.random.randint(0,1e6):06d}"


def log_experiment(config: dict, metrics: dict, model: nn.Module = None, optimizer=None, save_model: bool = False):
    """Atomically log an experiment run + optional checkpoint."""
    run_id = _make_run_id()
    env = _get_env_info()
    log = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
        "env": env,
    }

    json_path = LOG_DIR / f"exp_{run_id}.json"
    with open(json_path, "w") as f:
        json.dump(log, f, indent=2)

    model_path = None
    if save_model and model is not None:
        model_path = save_checkpoint(
            model, optimizer=optimizer,
            path=LOG_DIR / f"model_{run_id}.pth",
            extra={"run_id": run_id, "fingerprint": GLOBAL_FP},
        )

    # flat summary CSV
    row = {
        "run_id": run_id,
        "timestamp": log["timestamp"],
        "tomo": config.get("tomo"),
        "model": config.get("model_name", config.get("model")),
        "preset": config.get("preset"),
        "mean_dist": metrics.get("mean_dist"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f2": metrics.get("f2"),
        "runtime_s": metrics.get("runtime_s"),
        "gpu_mem_gb": metrics.get("gpu_mem_gb"),
        "seed": env.get("seed"),
        "fingerprint": env.get("fingerprint"),
        "model_path": model_path,
    }

    df_row = pd.DataFrame([row])
    df_row.to_csv(SUMMARY_CSV, mode="a", index=False, header=not SUMMARY_CSV.exists())

    # link augmentation metadata if present
    if AUG_LOG.exists():
        df_aug = pd.read_csv(AUG_LOG)
        df_aug["linked_run_id"] = run_id
        df_aug.to_csv(AUG_LOG, index=False)

    return {"json": str(json_path), "model": model_path, "summary": row}


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
                "mean_dist": r.get("metrics", {}).get("mean_dist"),
                "precision": r.get("metrics", {}).get("precision"),
                "recall": r.get("metrics", {}).get("recall"),
                "f2": r.get("metrics", {}).get("f2"),
                "seed": r.get("env", {}).get("seed"),
                "fingerprint": r.get("env", {}).get("fingerprint"),
            })
        return pd.DataFrame(flat)
    else:
        return rows


# ---------------- Streamlit UI ----------------
with st.expander("ğŸ—‚ï¸� Experiment Logs & Reproducibility", expanded=False):
    st.write("All experiment JSON logs are saved in:", str(LOG_DIR))
    if SUMMARY_CSV.exists():
        df_summary = pd.read_csv(SUMMARY_CSV)
        st.dataframe(df_summary.sort_values("timestamp", ascending=False))

        # ğŸ“Š Quick visual summary
        if len(df_summary) > 1:
            fig = px.scatter(
                df_summary,
                x="runtime_s", y="f2", color="model",
                size="precision", hover_data=["tomo", "recall", "mean_dist"],
                title="Experiment Fâ‚‚ vs Runtime (per model)"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "â¬‡ï¸� Download CSV", df_summary.to_csv(index=False).encode(), "experiment_summary.csv"
        )
    else:
        st.info("No summary CSV yet â€” run experiments and call log_experiment(...) to create entries.")

    if st.button("Reload JSON Logs"):
        df_logs = load_all_experiments(as_df=True)
        if df_logs.empty:
            st.info("No JSON logs found.")
        else:
            st.dataframe(df_logs.sort_values("timestamp", ascending=False))
# ------------------------------------------------------------
# ğŸ”¬ Experiment Comparison Utility
# ------------------------------------------------------------
st.markdown("---")
st.subheader("ğŸ”� Compare Two Experiments")

if SUMMARY_CSV.exists():
    df_summary = pd.read_csv(SUMMARY_CSV)
    run_ids = df_summary["run_id"].tolist()
    if len(run_ids) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            run_a = st.selectbox("Select First Run", run_ids, key="cmp_a")
        with col2:
            run_b = st.selectbox("Select Second Run", run_ids, index=1, key="cmp_b")

        if st.button("Compare Selected Runs"):
            row_a = df_summary[df_summary["run_id"] == run_a].iloc[0].to_dict()
            row_b = df_summary[df_summary["run_id"] == run_b].iloc[0].to_dict()

            # --- Compute differences ---
            metrics = ["mean_dist", "precision", "recall", "f2", "runtime_s"]
            diff = {m: round(row_b[m] - row_a[m], 4) for m in metrics if m in row_a}

            # --- Display tables ---
            st.markdown("#### ğŸ§¾ Configuration Comparison")
            cfg_df = pd.DataFrame([
                {"Parameter": k, "Run A": row_a.get(k), "Run B": row_b.get(k)}
                for k in ["tomo", "model", "preset", "seed", "fingerprint", "gpu_mem_gb"]
            ])
            st.dataframe(cfg_df, use_container_width=True)

            st.markdown("#### ğŸ“Š Metric Deltas (B âˆ’ A)")
            delta_df = pd.DataFrame(
                [{"Metric": k, "Î” Value": v, "Direction": "â¬†ï¸�" if v > 0 else "â¬‡ï¸�"} for k, v in diff.items()]
            )
            st.dataframe(delta_df, use_container_width=True)

            # --- Optional similarity score ---
            match_keys = ["tomo", "model", "preset", "seed"]
            match_score = sum(row_a[k] == row_b[k] for k in match_keys) / len(match_keys)
            st.info(f"ğŸ”— Configuration similarity score: **{match_score:.2f}**")

            # --- Visualization ---
            import plotly.graph_objects as go
            fig = go.Figure()
            for k in metrics:
                if k in row_a and k in row_b:
                    fig.add_trace(go.Bar(
                        x=[k],
                        y=[row_a[k]],
                        name=f"{run_a} (A)",
                        marker_color="royalblue"
                    ))
                    fig.add_trace(go.Bar(
                        x=[k],
                        y=[row_b[k]],
                        name=f"{run_b} (B)",
                        marker_color="darkorange"
                    ))
            fig.update_layout(barmode="group", title="Metric Comparison (A vs B)", height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least two experiment logs to compare.")
else:
    st.warning("No experiments logged yet â€” run a model to enable comparison.")



%%writefile -a /kaggle/working/app.py

# ================================================================
# ğŸš€ MAIN RUNNER FOR ABLATION (Must be at the very bottom of file)
# ================================================================
if run_button:
    tomos = find_tomograms(DATA_ROOT)
    if not tomos:
        st.error("â�Œ No tomograms found.")
        st.stop()

    tomo_names = [t.name for t in tomos]
    selected_tomo_name = st.selectbox("Select Tomogram for Ablation", tomo_names)
    selected_tomo = next((t for t in tomos if t.name == selected_tomo_name), None)
    
    if selected_tomo is None:
        st.error("Tomogram not found.")
        st.stop()

    try:
        raw_memmap = load_volume_from_jpegs_cached(selected_tomo)
        if raw_memmap is None:
            raise RuntimeError("Raw volume load failed")
        volume_memmap = preprocess_volume_cached_memmap(raw_memmap)
        volume_memmap = ensure_disk_memmap(volume_memmap)
        if not isinstance(volume_memmap, np.memmap) or volume_memmap.filename is None:
            raise RuntimeError("volume_memmap is not disk-backed")
    except Exception as e:
        st.error(f"â�Œ Failed to load tomogram: {e}")
        st.stop()

    gt_points = np.zeros((0, 3))
    gt_csv = Path(DATA_ROOT) / GT_CSV
    if gt_csv.exists():
        df = pd.read_csv(gt_csv)
        sel = df[df["tomo_id"] == selected_tomo_name]
        # âœ… FIX: Load in 0, 1, 2 order to perfectly match numpy's Z, Y, X shape
        if {"Motor axis 0", "Motor axis 1", "Motor axis 2"}.issubset(sel.columns):
            gt_points = sel[["Motor axis 0", "Motor axis 1", "Motor axis 2"]].values.astype(float)

    if gt_points.shape[0] == 0:
        st.warning("âš ï¸� No GT points found â€” metrics will be NaN.")

    combos = _make_grid(patch_sizes, stride_factors, dbscan_eps_list, presets, max_runs)
    st.info(f"ğŸ§ª Running {len(combos)} ablations on {selected_tomo_name}")
    prog = st.progress(0)
    start = time.time()

    for i, (ps, sf, eps, pr) in enumerate(combos):
        if (time.time() - start) / 60 > max_total_runtime:
            st.warning("â�±ï¸� Global time limit reached.")
            break
        try:
            metrics = _run_single_combo(
                volume_memmap, gt_points, ps, sf, eps, pr, model_choice, max_runtime_per_run * 60
            )
            metrics.update({"tomo": selected_tomo_name, "run_index": i + 1})
            st.session_state["ablation_results"].append(metrics)
            st.write("âœ…", metrics)
        except Exception as e:
            st.error(f"Run failed: {e}")
            st.session_state["ablation_results"].append({
                "tomo": selected_tomo_name, "run_index": i + 1, "patch_size": ps,
                "stride_factor": sf, "dbscan_eps": eps, "preset": pr, "model": model_choice,
                "error": str(e), "timestamp": datetime.now().isoformat(),
            })
        prog.progress(int(100 * (i + 1) / len(combos)))

    _save_results(st.session_state["ablation_results"], "ablation_final")


import os
os.environ["NGROK_AUTH_TOKEN"] = "3A1jjzQstZSEnCZrTCNVTRDMts7_7rq2hNBqtWmmvS3nwGb9v"


# ============================================================
# âœ… Secure Streamlit + Ngrok Launcher (Kaggle Stable Version)
# ============================================================
# Run AFTER saving /kaggle/working/app.py

# ---------------- Install Dependency ----------------
!pip install -q pyngrok

# ---------------- Imports ----------------
import os
import time
import torch
import platform
import subprocess
import threading
from pathlib import Path
from pyngrok import ngrok, conf

# ---------------- Paths ----------------
APP_PATH = "/kaggle/working/app.py"
LOG_FILE = "/kaggle/working/streamlit_log.txt"

CACHE_DIR = Path("/kaggle/working/cache_phase1")
LOG_DIR   = CACHE_DIR / "experiment_logs"
AUG_DIR   = CACHE_DIR / "augmented"

for d in [CACHE_DIR, LOG_DIR, AUG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------- Validate App Exists ----------------
if not Path(APP_PATH).exists():
    raise FileNotFoundError(f"â�Œ {APP_PATH} not found. Save app.py first.")

# ---------------- Kill Old Sessions ----------------
print("ğŸ§¹ Cleaning previous Streamlit/Ngrok processes...")
os.system("pkill -9 -f streamlit || true")
os.system("pkill -9 -f ngrok || true")
time.sleep(2)

# ---------------- Environment Summary ----------------
print("\nğŸ§  Environment Summary")
print(f"Python: {platform.python_version()}")
print(f"Torch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"VRAM: {round(props.total_memory / 1e9, 2)} GB")

# ---------------- Ngrok Authentication ----------------
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN")

if not NGROK_AUTH_TOKEN:
    raise ValueError(
        "â�Œ NGROK_AUTH_TOKEN not found.\n"
        "Run this first:\n"
        'os.environ["NGROK_AUTH_TOKEN"] = "your_token_here"'
    )

conf.get_default().auth_token = NGROK_AUTH_TOKEN
print("ğŸ”‘ Ngrok authentication successful.")

# ---------------- Launch Streamlit ----------------
def run_streamlit():
    with open(LOG_FILE, "w") as log:
        subprocess.run(
            [
                "streamlit", "run", APP_PATH,
                "--server.headless", "true",
                "--server.port", "8501",
                "--server.address", "0.0.0.0",
                "--browser.gatherUsageStats", "false"
            ],
            stdout=log,
            stderr=subprocess.STDOUT
        )

print("\nğŸš€ Starting Streamlit server...")
threading.Thread(target=run_streamlit, daemon=True).start()

# Give Streamlit time to initialize
time.sleep(8)

# ---------------- Start Ngrok Tunnel ----------------
try:
    tunnel = ngrok.connect(8501)
    print("\nğŸŒ� Public URL:")
    print("ğŸ‘‰", tunnel.public_url)
except Exception as e:
    print(f"â�Œ Ngrok failed: {e}")

print("\nâœ… Launcher completed. Wait a few seconds if app is still loading.")



!python -m py_compile /kaggle/working/app.py


!cat /kaggle/working/streamlit_log.txt

