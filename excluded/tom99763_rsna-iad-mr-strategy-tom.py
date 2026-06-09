import os
import sys
from pathlib import Path
from typing import List, Tuple

import ast
import cv2
from matplotlib import image
import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import multiprocessing
import glob
import matplotlib.pyplot as plt
import ast


data_path = './data'
windows = {
    'CT': (40, 80),
    'CTA': (50, 350),
    'MRA': (600, 1200),
    'MR': (600, 1200),
    'MRI': (40, 80),
}

LABELS_TO_IDX = {
            'Anterior Communicating Artery': 0,
            'Basilar Tip': 1,
            'Left Anterior Cerebral Artery': 2,
            'Left Infraclinoid Internal Carotid Artery': 3,
            'Left Middle Cerebral Artery': 4,
            'Left Posterior Communicating Artery': 5,
            'Left Supraclinoid Internal Carotid Artery': 6,
            'Other Posterior Circulation': 7,
            'Right Anterior Cerebral Artery': 8,
            'Right Infraclinoid Internal Carotid Artery': 9,
            'Right Middle Cerebral Artery': 10,
            'Right Posterior Communicating Artery': 11,
            'Right Supraclinoid Internal Carotid Artery': 12
}

IMG_SIZE = 512
FACTOR = 1
SEED = 42
N_FOLDS = 5
CORES = 4


from skimage.filters import sobel, frangi
from skimage.morphology import white_tophat, disk
from scipy.ndimage import gaussian_gradient_magnitude
from __future__ import annotations
import os
import ast
import math
import multiprocessing as mp
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import pydicom
import cv2
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from scipy.ndimage import zoom as nd_zoom
from skimage import filters
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects


#current_dir = Path(__file__).parent
#parent_dir = current_dir.parent
#sys.path.insert(0, str(parent_dir))

TARGET_DEPTH = 32
TARGET_SIZE = 384
HU_MIN = -1200.0
HU_MAX = 4000.0
STORE_NORMALIZED = False  # Set True to revert to [0,1] scaling

# Globals for worker processes

data_path = '/kaggle/input/rsna-intracranial-aneurysm-detection'


def _load_series_dicom_paths(series_uid: str, root: Path) -> List[Path]:
    series_dir = root / 'series' / series_uid
    paths = []
    for r, _, files in os.walk(series_dir):
        for f in files:
            if f.endswith('.dcm'):
                paths.append(Path(r) / f)
    return paths


def _read_dicom(path: Path):
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    return ds, arr


def _extract_slice_position(ds) -> float:
    # Prefer ImagePositionPatient z, fallback to InstanceNumber
    if hasattr(ds, 'ImagePositionPatient') and len(ds.ImagePositionPatient) == 3:
        try:
            return float(ds.ImagePositionPatient[2])
        except Exception:
            pass
    if hasattr(ds, 'InstanceNumber'):
        try:
            return float(ds.InstanceNumber)
        except Exception:
            pass
    return 0.0


def _resample_depth(volume: np.ndarray, target_depth: int) -> np.ndarray:
    if volume.shape[0] == target_depth:
        return volume
    depth_zoom = target_depth / volume.shape[0]
    # zoom along depth only, order=1 linear
    return nd_zoom(volume, (depth_zoom, 1.0, 1.0), order=1)


def _resize_inplane(volume: np.ndarray, target_hw: int) -> np.ndarray:
    d, h, w = volume.shape
    if h == target_hw and w == target_hw:
        return volume
    resized = np.empty((d, target_hw, target_hw), dtype=volume.dtype)
    for i in range(d):
        resized[i] = cv2.resize(volume[i], (target_hw, target_hw), interpolation=cv2.INTER_LINEAR)
    return resized


def _clip_or_normalize(volume: np.ndarray) -> np.ndarray:
    """Either clip-only (raw HU retained in range) or clip+normalize to [0,1]."""
    vol = np.clip(volume, HU_MIN, HU_MAX).astype(np.float32)
    if STORE_NORMALIZED:
        vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)
    return vol


def extract_details(vol):
    mip = vol.max(axis=0, keepdims=True)
    std_proj = vol.std(axis=0, keepdims=True)
    edges = np.stack([sobel(slice_) for slice_ in vol], axis=0)
    edge_proj = edges.max(axis=0, keepdims=True)
    vesselness = np.stack([frangi(slice_) for slice_ in vol], axis=0)
    vessel_proj = vesselness.max(axis=0, keepdims=True)
    gradmag = np.stack([gaussian_gradient_magnitude(slice_, sigma=1) for slice_ in vol], axis=0)
    grad_proj = gradmag.max(axis=0, keepdims=True)
    extracted_vol = np.concatenate([vol, mip, std_proj, edge_proj, vessel_proj, grad_proj], axis=0) #(32 + 5, 384, 384)
    return extracted_vol


def _process_single_series(uid: str, root: Path) -> Dict[str, Any]:
    try:
        dcm_paths = _load_series_dicom_paths(uid, root)
        if not dcm_paths:
            return {"series_uid": uid, "volume_filename": None, "num_slices_raw": 0}
        slices: List[Tuple[float, np.ndarray]] = []
        for p in dcm_paths:
            try:
                ds, arr = _read_dicom(p)
                # If multi-frame (arr.ndim==3) stack frames individually
                if arr.ndim == 3 and arr.shape[-1] != 3:
                    for fi in range(arr.shape[0]):
                        slices.append((_extract_slice_position(ds) + fi * 0.001, arr[fi].astype(np.float32)))
                else:
                    if arr.ndim == 3 and arr.shape[-1] == 3:
                        # Convert RGB to grayscale
                        arr = cv2.cvtColor(arr.astype(np.float32), cv2.COLOR_BGR2GRAY)
                    slices.append((_extract_slice_position(ds), arr.astype(np.float32)))
            except Exception:
                continue
        if not slices:
            return {"series_uid": uid, "volume_filename": None, "num_slices_raw": 0}
        # Sort by z
        slices.sort(key=lambda x: x[0])
        vol = np.stack([s[1] for s in slices], axis=0)  # (D, H, W)
        num_raw = vol.shape[0]
        # Clip HU range (optionally normalize based on STORE_NORMALIZED)
        vol = _clip_or_normalize(vol)
        # Depth resample
        vol = _resample_depth(vol, TARGET_DEPTH)
        # In-plane resize
        vol = _resize_inplane(vol, TARGET_SIZE)
        # Save
        vol_filename = f"{uid}_d{TARGET_DEPTH}_sz{TARGET_SIZE}.npz"
        # Save meta: [HU_MIN, HU_MAX, normalized_flag]
        meta = np.array([HU_MIN, HU_MAX, 1.0 if STORE_NORMALIZED else 0.0], dtype=np.float32)
        return {"series_uid": uid, "volume_filename": vol_filename, "num_slices_raw": num_raw, 'volume': vol, 'meta': meta}
    except Exception as e:
        return {"series_uid": uid, "volume_filename": None, "error": str(e), "num_slices_raw": 0}



root = Path(data_path)
processed = root / 'processed'
vol_dir = processed / 'volumes_3d'
train_df = pd.read_csv(root / 'train.csv')
label_df = pd.read_csv(root / 'train_localizers.csv')
mf_dicom_uids = pd.read_csv(root / 'multiframe_dicoms.csv') if (root / 'multiframe_dicoms.csv').exists() else pd.DataFrame(columns=['SeriesInstanceUID'])

ignore_uids = set([
    '1.2.826.0.1.3680043.8.498.11145695452143851764832708867797988068',
    '1.2.826.0.1.3680043.8.498.35204126697881966597435252550544407444',
    '1.2.826.0.1.3680043.8.498.87480891990277582946346790136781912242',
]) | set(mf_dicom_uids['SeriesInstanceUID'].tolist())

train_df = train_df[~train_df['SeriesInstanceUID'].isin(ignore_uids)].reset_index(drop=True)
train_df['fold_id'] = 0

skf = StratifiedKFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)
for fold, (_, val_idx) in enumerate(skf.split(train_df['SeriesInstanceUID'], train_df['Aneurysm Present'])):
    train_df.loc[val_idx, 'fold_id'] = fold

uids = train_df[train_df["Modality"].isin(['MRA'])]['SeriesInstanceUID'].unique().tolist()
print(f"Preparing 3D volumes for {len(uids)} series -> target shape ({TARGET_DEPTH}, {TARGET_SIZE}, {TARGET_SIZE})")


len(uids)


def hyst_thres(vol, LOW_THRESHOLD=50, HIGH_THRESHOLD=60, min_size=100):
    '''
    shape of vol: (depth, height, width)
    '''
    x = vol.transpose(1, 2, 0)
    #smooth_arr = gaussian_filter(x, sigma=sigma) #GAUSSIAN FILTER (HESSIAN FILTER ALREADY HAS GAUSSIAN FILTER IF USING- IF NOT, REPLACE "MRA_FILE" WITH "smooth_array"
    percentile_element = np.percentile(x, 99.9)
    max_element = np.amax(x)  # maximum intensity value in array
    L_thresh = (percentile_element / 100) * (HIGH_THRESHOLD)
    H_thresh = (percentile_element / 100) * (LOW_THRESHOLD)
    hyst = filters.apply_hysteresis_threshold(x, L_thresh, H_thresh).astype(int)
    hyst = hyst.transpose(2, 0, 1)
    mask = hyst.max(axis=0, keepdims=True)
    mask = remove_small_objects(mask.astype(bool), min_size=min_size)
    return mask


def refine_masking(x):
    h, w = x.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Define ellipse parameters
    center = (w // 2, h // 2)       # center of ellipse
    axes = (100, 150)               # (width radius, height radius) adjust as needed
    angle = 0                       # rotation angle
    startAngle, endAngle = 0, 360   # full ellipse
    
    # Draw filled ellipse on mask
    cv2.ellipse(mask, center, axes, angle, startAngle, endAngle, 255, -1)
    
    # Apply mask
    x = cv2.bitwise_and(x, x, mask=mask)
    return x


train_df.Modality.unique()


#golden sample
idx = 3
uid1 = uids[idx]
mod = train_df[train_df['SeriesInstanceUID'] == uid1].Modality
label = train_df[train_df['SeriesInstanceUID'] == uid1]['Aneurysm Present']
print(mod.values)
print(label.values)
series_dict1  = _process_single_series(uid1, root)


vol1 = series_dict1['volume']


mip1 = vol1.max(axis=0)


mask = hyst_thres(vol1, 60, 70, min_size=10000)
mask = hyst_thres(refine_masking(mask[0]*mip1)[None], 40, min_size = 500)
golden_sample = mask[0]*mip1


plt.imshow(mip1)


plt.imshow(golden_sample)


#query
uid2 = uids[10]


series_dict2  = _process_single_series(uid2, root)


vol2 = series_dict2['volume']


mip2 = vol2.max(axis=0)


fig, ax  = plt.subplots(ncols = 2, figsize = (12, 12))
ax[0].imshow(golden_sample)
ax[0].set_title('golden sample')
ax[1].imshow(mip2)
ax[1].set_title('query sample')
plt.show()


mip1_gray = cv2.normalize(golden_sample, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')


orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(mip1_gray, None)

mip2_gray = cv2.normalize(mip2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

kp2, des2 = orb.detectAndCompute(mip2_gray, None)


# Draw simple keypoints (just dots)
img_kp1 = cv2.drawKeypoints(
    mip1_gray, kp1, None,
    color=(0, 255, 0),   # green dots
    flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT  # no circles, just points
)

img_kp2 = cv2.drawKeypoints(
    mip2_gray, kp2, None,
    color=(255, 0, 0),   # blue dots
    flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
)

# Show side by side
plt.figure(figsize=(12, 12))
plt.subplot(1, 2, 1)
plt.imshow(img_kp1, cmap="gray")
plt.title("Golden Sample")

plt.subplot(1, 2, 2)
plt.imshow(img_kp2, cmap="gray")
plt.title("Query Image")

plt.show()


# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)

print('total matches:', len(matches))
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv2.drawMatches(mip1_gray,kp1,mip2_gray,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize = (12, 12))
plt.imshow(img3)
plt.show()


def create_circle_mask(mip, kps, radius = 10):
    # create empty mask
    mask = np.zeros_like(mip, dtype=np.uint8)
    
    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(mask, (x, y), radius, 255, -1)
    return mask


def keypoint_mip_masking(mip, nfeatures = 1000, radius=10):
    mip_gray = cv2.normalize(mip, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kps, des = orb.detectAndCompute(mip_gray, None)
    mask = np.zeros_like(mip, dtype=np.uint8)
    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(mask, (x, y), radius, 255, -1)
    return mask * mip


circle_mask = create_circle_mask(mip2, kp2, 10)


(circle_mask).astype('float32').max()


plt.imshow((circle_mask * mip2).astype('float32')/255.)


import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
from networkx.algorithms import bipartite


# --- Build bipartite graph ---
G = nx.Graph()
n1, n2 = len(des1), len(des2)

# Node sets
G.add_nodes_from(range(n1), bipartite=0)                     # left = kp1
G.add_nodes_from(range(n1, n1+n2), bipartite=1)              # right = kp2

# Edge weights = Hamming distance
dist_matrix = cdist(des1, des2, metric="hamming")
for i in range(n1):
    for j in range(n2):
        G.add_edge(i, n1+j, weight=dist_matrix[i, j])

# --- Compute bipartite minimum weight matching ---
matching = bipartite.minimum_weight_full_matching(G, weight="weight")


# --- Draw bipartite graph ---
pos = {}
# Layout: left nodes on x=0, right nodes on x=1
pos.update((i, (0, -i)) for i in range(n1))
pos.update((n1+j, (1, -j)) for j in range(n2))

plt.figure(figsize=(10, 8))

# Draw nodes
nx.draw_networkx_nodes(G, pos, nodelist=range(n1), node_color="lightblue", label="img1 keypoints")
nx.draw_networkx_nodes(G, pos, nodelist=range(n1, n1+n2), node_color="lightgreen", label="img2 keypoints")

# Draw only matching edges
matched_edges = [(u, v) for u, v in matching.items() if u < v]
nx.draw_networkx_edges(G, pos, edgelist=matched_edges, edge_color="red", width=1.5)

# No labels (too many keypoints)
plt.title("Bipartite Keypoint Matching")
plt.axis("off")
plt.legend()
plt.show()


# --- Convert bipartite matching to cv2.DMatch objects ---
matches = []
for u, v in matching.items():
    if u < v:  # ensure each pair only once
        if u < n1:
            i, j = u, v - n1
        else:
            i, j = v, u - n1
        d = dist_matrix[i, j]
        matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=d))

# Sort by distance for visualization
matches = sorted(matches, key=lambda x: x.distance)

# --- Draw matches on stitched image ---
img_matches = cv2.drawMatches(
    mip1_gray, kp1, mip2_gray, kp2,
    matches, None,  # show first 20 matches
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(12, 6))
plt.imshow(img_matches, cmap="gray")
plt.title("ORB Keypoints with Bipartite Matching")
plt.axis("off")
plt.show()


from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi


_, binary = cv2.threshold(mip1_gray, 30, 255, cv2.THRESH_BINARY)

# Compute distance transform
distance = ndi.distance_transform_edt(binary)

# Find local maxima
coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=binary)

# Marker labeling
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)

# Apply watershed
labels = watershed(-distance, markers, mask=binary)


labels_display = labels.astype(float)
labels_display[labels_display == 0] = np.nan

# Use Spectral colormap and set NaN color to black
cmap = plt.cm.Spectral.copy()
cmap.set_bad(color='black')

plt.imshow(labels_display, cmap=cmap)
plt.imshow(mip1, alpha=0.5)
plt.colorbar()
plt.show()







